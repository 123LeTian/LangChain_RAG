"""
RAG Service — Owner: C
High-level service that orchestrates the full RAG pipeline: retrieval → augmentation → generation.

Responsibilities:
  - Strategy selection (explicit or auto-routed)
  - Pipeline execution with tracing
  - Graceful fallback on failure
  - Streaming and non-streaming paths
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from src.models.rag import (
    RAGContext,
    RAGPipelineConfig,
    RAGQuery,
    RAGResponse,
    StrategyType,
)
from src.rag.base import (
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategyBase,
    RetrieverProtocol,
)
from src.rag.registry import (
    StrategyNotFoundError,
    StrategyRegistry,
    get_registry,
)
from src.rag.trace import (
    SpanStatus,
    Trace,
    TraceContext,
    TraceEventType,
    TraceStore,
    get_trace_store,
)

logger = logging.getLogger(__name__)


# ── RAG Status (if not in models) ─────────────────────────────────────────

class RAGStatus:
    """Pipeline status constants."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


# ── Simplified streaming event types ──────────────────────────────────────

class StreamEventType:
    """Stream event type constants for the RAG pipeline."""
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_END = "retrieval_end"
    RERANK = "rerank"
    GENERATION_START = "generation_start"
    TOKEN = "token"
    SOURCE = "source"
    DONE = "done"
    ERROR = "error"
    AGENT_STEP = "agent_step"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"


class StreamEvent:
    """A single streaming event from the RAG pipeline."""

    def __init__(self, event_type: str, data: Any = None, trace_id: Optional[str] = None) -> None:
        self.type = event_type
        self.data = data
        self.trace_id = trace_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "trace_id": self.trace_id,
        }


# ── Service ───────────────────────────────────────────────────────────────


class RAGService:
    """Unified RAG orchestration service.

    This is the primary entry point that the API layer (Owner: A) calls.
    It selects the appropriate strategy, executes the pipeline, records traces,
    and handles fallback on failure.

    Usage:
        service = RAGService(
            retrieval=my_retriever,
            llm=my_generator,
            graph=my_graph_retriever,   # optional
        )
        response = await service.run(RAGQuery(text="What is RAG?", strategy=StrategyType.ADVANCED))
    """

    def __init__(
        self,
        retrieval: RetrieverProtocol,
        llm: GeneratorProtocol,
        graph: Optional[GraphRetrieverProtocol] = None,
        *,
        registry: Optional[StrategyRegistry] = None,
        trace_store: Optional[TraceStore] = None,
        default_strategy: StrategyType = StrategyType.NAIVE,
        config: Optional[RAGPipelineConfig] = None,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._graph = graph
        self._registry = registry or get_registry()
        self._trace_store = trace_store or get_trace_store()
        self._default_strategy = default_strategy
        self._config = config or RAGPipelineConfig()

    # ── Public API ─────────────────────────────────────────────────────

    async def run(self, query: RAGQuery) -> RAGResponse:
        """Execute the full RAG pipeline and return a response.

        Args:
            query: The user query with optional strategy selection.

        Returns:
            RAGResponse with answer, sources, trace_id, etc.
        """
        trace = Trace(query_id=query.query_id)
        strategy_type = self._resolve_strategy(query)

        with TraceContext(trace, "pipeline") as pipeline_span:
            pipeline_span.add_event(
                TraceEventType.PIPELINE_START,
                query=query.text,
                strategy=strategy_type.value,
            )

            try:
                strategy = self._get_strategy(strategy_type)
                response = await strategy.run(query)
                response.trace_id = trace.trace_id
                pipeline_span.add_event(
                    TraceEventType.PIPELINE_END,
                    answer_len=len(response.answer),
                    latency_ms=response.latency_ms,
                )
                self._trace_store.save(trace)
                return response

            except Exception as exc:
                logger.warning(
                    "Strategy %s failed: %s. Attempting fallback.", strategy_type, exc
                )
                pipeline_span.fail(exc)
                fallback_response = await self._execute_fallback(query, trace, exc)
                self._trace_store.save(trace)
                return fallback_response

    async def stream(self, query: RAGQuery) -> AsyncIterator[StreamEvent]:
        """Streaming variant of run().

        Yields StreamEvent objects as the pipeline progresses:
        retrieval_start → SOURCE* → retrieval_end → GENERATION_START → TOKEN* → DONE.
        """
        trace = Trace(query_id=query.query_id)
        strategy_type = self._resolve_strategy(query)

        try:
            strategy = self._get_strategy(strategy_type)
        except StrategyNotFoundError:
            yield StreamEvent(StreamEventType.ERROR, data="No strategy available", trace_id=trace.trace_id)
            return

        # Try streaming if the strategy supports it
        stream_method = getattr(strategy, "stream", None)
        if callable(stream_method):
            try:
                async for event in stream_method(query):
                    event.trace_id = trace.trace_id
                    yield event
                    if event.type == StreamEventType.DONE:
                        self._trace_store.save(trace)
                        return
            except Exception as exc:
                logger.warning("Stream failed, falling back: %s", exc)
                yield StreamEvent(StreamEventType.ERROR, data=str(exc), trace_id=trace.trace_id)

        # Non-streaming fallback
        response = await self.run(query)
        response.trace_id = trace.trace_id
        if response.context:
            for chunk in response.context.chunks:
                yield StreamEvent(
                    StreamEventType.SOURCE,
                    data={"chunk_id": chunk.chunk_id, "content": chunk.content[:200]},
                    trace_id=trace.trace_id,
                )
        yield StreamEvent(
            StreamEventType.TOKEN, data=response.answer, trace_id=trace.trace_id
        )
        yield StreamEvent(StreamEventType.DONE, data={}, trace_id=trace.trace_id)
        self._trace_store.save(trace)

    # ── Strategy management ────────────────────────────────────────────

    def register_strategy(
        self, strategy_type: StrategyType, strategy_cls: type, **kwargs: Any
    ) -> None:
        """Register a strategy class at runtime."""
        self._registry.register(strategy_type, strategy_cls, **kwargs)

    def list_strategies(self) -> List[str]:
        """Return names of all registered strategies."""
        return self._registry.list_names()

    def set_default_strategy(self, strategy_type: StrategyType) -> None:
        """Change the default strategy."""
        self._default_strategy = strategy_type

    # ── Trace access ───────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trace by ID (for API)."""
        trace = self._trace_store.get(trace_id)
        return trace.to_dict() if trace else None

    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent traces (for API)."""
        traces = self._trace_store.list_recent(limit)
        return [t.to_dict() for t in traces]

    # ── Internal ───────────────────────────────────────────────────────

    def _resolve_strategy(self, query: RAGQuery) -> StrategyType:
        """Determine which strategy to use.

        Priority: query.strategy > pipeline config > default.
        """
        if query.strategy:
            return query.strategy
        if self._config.strategy:
            return self._config.strategy
        return self._default_strategy

    def _get_strategy(self, strategy_type: StrategyType) -> RAGStrategyBase:
        """Get the strategy instance for the given type.

        Injects the protocol dependencies that the strategy needs.
        """
        strategy = self._registry.get(strategy_type)

        # Inject dependencies if the strategy supports it
        if hasattr(strategy, "set_retriever"):
            strategy.set_retriever(self._retrieval)
        if hasattr(strategy, "set_generator"):
            strategy.set_generator(self._llm)
        if hasattr(strategy, "set_graph_retriever") and self._graph is not None:
            strategy.set_graph_retriever(self._graph)

        return strategy

    async def _execute_fallback(
        self,
        query: RAGQuery,
        trace: Trace,
        original_error: Exception,
    ) -> RAGResponse:
        """Execute fallback chain: try alternative strategies in order."""
        fallback_order = self._config.fallback_strategies or [
            StrategyType.ADVANCED,
            StrategyType.NAIVE,
        ]

        errors: List[str] = [f"{type(original_error).__name__}: {original_error}"]

        for fb_type in fallback_order:
            if fb_type == query.strategy:
                continue  # Skip the one that already failed
            if not self._registry.is_registered(fb_type):
                continue

            try:
                with TraceContext(trace, f"fallback_{fb_type.value}") as fb_span:
                    fb_span.add_event(
                        TraceEventType.AGENT_FALLBACK,
                        from_strategy=query.strategy.value,
                        to_strategy=fb_type.value,
                    )
                    strategy = self._get_strategy(fb_type)
                    response = await strategy.run(query)
                    response.trace_id = trace.trace_id
                    response.metadata["status"] = RAGStatus.PARTIAL
                    response.metadata["fallback_from"] = query.strategy.value
                    response.metadata["fallback_errors"] = errors
                    response.error = (
                        f"Primary strategy ({query.strategy.value}) failed. "
                        f"Fell back to {fb_type.value}. Errors: {'; '.join(errors)}"
                    )
                    return response
            except Exception as fb_exc:
                errors.append(f"{fb_type.value}: {type(fb_exc).__name__}: {fb_exc}")
                continue

        # All fallbacks exhausted
        return RAGResponse(
            query_id=query.query_id,
            answer="",
            strategy=query.strategy,
            error=f"All strategies exhausted. Errors: {'; '.join(errors)}",
            trace_id=trace.trace_id,
        )


# ── Factory ───────────────────────────────────────────────────────────────


def create_rag_service(
    retrieval: RetrieverProtocol,
    llm: GeneratorProtocol,
    graph: Optional[GraphRetrieverProtocol] = None,
    **kwargs: Any,
) -> RAGService:
    """Factory function: create a fully initialized RAGService.

    Auto-discovers strategies from the strategies package.

    Args:
        retrieval: Object conforming to RetrieverProtocol (from Owner: B).
        llm: Object conforming to GeneratorProtocol (from Owner: A or B).
        graph: Optional object conforming to GraphRetrieverProtocol (from Owner: D).
        **kwargs: Passed to RAGService constructor.

    Returns:
        Configured RAGService ready to handle queries.
    """
    registry = kwargs.pop("registry", get_registry())

    # Auto-discover strategies
    if len(registry) == 0:
        registry.discover_all("src.rag.strategies")

    return RAGService(
        retrieval=retrieval,
        llm=llm,
        graph=graph,
        registry=registry,
        **kwargs,
    )
