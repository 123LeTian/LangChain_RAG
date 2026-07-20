"""
Agentic RAG Strategy — Owner: C
Agent-driven RAG with tool use, multi-step reasoning, and dynamic retrieval decisions.

This strategy wraps the LangGraph agent workflow and presents it through the
standard RAGStrategyBase interface. The agent can:
  - Plan multi-step retrieval strategies
  - Call tools (vector search, graph search, calculator)
  - Reason about retrieved information
  - Reflect on answer quality and loop back if needed
  - Gracefully degrade on errors

Dependencies:
  - src/agent/workflow.py: the LangGraph state machine
  - src/agent/state.py: AgentState definition
  - src/agent/tools.py: ToolRegistry and ToolExecutor
  - src/agent/router.py: QueryRouter for auto-routing (optional)
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.agent.router import QueryRouter
from src.agent.state import AgentState, AgentStatus, initial_state, state_snapshot
from src.agent.tools import (
    CalculatorTool,
    DirectAnswerTool,
    GraphSearchTool,
    ToolExecutor,
    ToolRegistry,
    VectorSearchTool,
)
from src.agent.workflow import get_agent_graph
from src.models.rag import (
    RAGCitation,
    RAGContext,
    RAGChunk,
    RAGMode,
    RAGQuery,
    RAGResponse,
    RAGSource,
    StrategyType,
)
from src.rag.base import (
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategyBase,
    RetrieverProtocol,
)
from src.rag.registry import get_registry

logger = logging.getLogger(__name__)


class AgenticRAGStrategy(RAGStrategyBase):
    """Agent-driven RAG using LangGraph for multi-step reasoning.

    The agent:
      1. Plans: Decomposes the query into steps.
      2. Retrieves: Uses vector + graph search tools.
      3. Reasons: Analyzes retrieved information.
      4. Generates: Produces a grounded answer.
      5. Reflects: Checks quality; may loop back for more retrieval.

    This strategy auto-registers with the StrategyRegistry.
    """

    strategy_type: StrategyType = StrategyType.AGENTIC

    def __init__(
        self,
        retrieval: Optional[RetrieverProtocol] = None,
        llm: Optional[GeneratorProtocol] = None,
        graph: Optional[GraphRetrieverProtocol] = None,
        *,
        max_iterations: int = 10,
        tool_timeout: float = 30.0,
        use_router: bool = True,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config)
        self._retrieval = retrieval
        self._llm = llm
        self._graph = graph
        self._max_iterations = max_iterations
        self._tool_timeout = tool_timeout
        self._router = QueryRouter() if use_router else None

        # Lazily built
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None

    # ── Dependency injection ───────────────────────────────────────────

    def set_retriever(self, retrieval: RetrieverProtocol) -> None:
        self._retrieval = retrieval
        self._tool_registry = None  # Rebuild tools on next use

    def set_generator(self, llm: GeneratorProtocol) -> None:
        self._llm = llm

    def set_graph_retriever(self, graph: GraphRetrieverProtocol) -> None:
        self._graph = graph
        self._tool_registry = None  # Rebuild tools on next use

    # ── Tool setup ─────────────────────────────────────────────────────

    def _get_tool_executor(self) -> ToolExecutor:
        """Return the tool executor, building it lazily."""
        if self._tool_executor is None:
            registry = ToolRegistry()

            if self._retrieval:
                registry.register(VectorSearchTool(self._retrieval))
            if self._graph:
                registry.register(GraphSearchTool(self._graph))
            registry.register(CalculatorTool())
            registry.register(DirectAnswerTool())

            self._tool_registry = registry
            self._tool_executor = ToolExecutor(
                registry,
                default_timeout=self._tool_timeout,
                max_retries=1,
            )
        return self._tool_executor

    # ── State preparation ──────────────────────────────────────────────

    def _prepare_state(self, query: RAGQuery) -> AgentState:
        """Build the initial AgentState with all dependencies bound in.

        The LangGraph nodes read dependencies (retrieval, llm, graph, tools)
        from the state dict, so we inject them here.
        """
        state = initial_state(
            query=query.text,
            max_iterations=self._max_iterations,
        )

        # Inject dependencies into state for nodes to use
        state["retrieval"] = self._retrieval  # type: ignore[typeddict-unknown-key]
        state["llm"] = self._llm  # type: ignore[typeddict-unknown-key]
        state["graph"] = self._graph  # type: ignore[typeddict-unknown-key]
        state["tool_executor"] = self._get_tool_executor()  # type: ignore[typeddict-unknown-key]

        # If router is available, pre-compute routing
        if self._router:
            decision = self._router.route(query.text)
            state["plan"] = [  # type: ignore[typeddict-unknown-key]
                f"[Router: {decision.strategy.value}, confidence={decision.confidence:.0%}] "
                f"{decision.reasoning}"
            ]

        return state

    # ── RAGStrategyBase implementation ─────────────────────────────────

    async def retrieve(self, query: RAGQuery) -> RAGContext:
        """Run only the retrieval phase using agent tools."""
        t0 = time.perf_counter()
        all_chunks: List[RAGChunk] = []

        if self._retrieval:
            try:
                context = self._retrieval.retrieve(query=query.text, top_k=query.top_k)
                all_chunks.extend(context.chunks)
            except Exception as exc:
                logger.error("Agentic retrieve (vector) failed: %s", exc)

        if self._graph:
            try:
                graph_context = self._graph.graph_search(
                    query=query.text, top_k=query.top_k
                )
                all_chunks.extend(graph_context.chunks)
            except Exception as exc:
                logger.error("Agentic retrieve (graph) failed: %s", exc)

        return RAGContext(
            query=query.text,
            chunks=all_chunks,
            retrieval_method="agent",
            retrieval_latency_ms=(time.perf_counter() - t0) * 1000,
            total_candidates=len(all_chunks),
        )

    async def generate(self, context: RAGContext, query: RAGQuery) -> RAGResponse:
        """Generate an answer from the given context."""
        if not self._llm:
            return RAGResponse(
                query_id=query.query_id,
                answer="[No LLM available for generation]",
                strategy=StrategyType.AGENTIC,
                context=context,
            )

        t0 = time.perf_counter()
        try:
            answer = await self._llm.generate(
                prompt="Answer the question based on the provided context. Be thorough and cite sources.",
                context=context.combined_text,
            )
        except Exception as exc:
            answer = f"Error during generation: {exc}"

        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            strategy=StrategyType.AGENTIC,
            context=context,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def run(self, query: RAGQuery) -> RAGResponse:
        """Execute the full agentic RAG workflow.

        This is the main entry point. It:
          1. Builds initial state with bound dependencies
          2. Runs the LangGraph agent workflow
          3. Extracts the answer, citations, and metadata from final state
          4. Falls back to direct retrieve→generate if the graph is unavailable

        Returns:
            RAGResponse with answer, citations, trace metadata, etc.
        """
        t0 = time.perf_counter()

        # ── Try LangGraph workflow ────────────────────────────────────
        try:
            state = self._prepare_state(query)
            graph = get_agent_graph()
            final_state = await graph.ainvoke(state)
            return self._build_response(query, final_state, t0)
        except ImportError:
            logger.warning("LangGraph not available — falling back to simple retrieve→generate")
        except Exception as exc:
            logger.error("Agent workflow failed: %s. Falling back.", exc)

        # ── Fallback: simple retrieve → generate ──────────────────────
        return await self._fallback_run(query, t0)

    async def stream(self, query: RAGQuery) -> AsyncIterator[Any]:
        """Stream the agent workflow execution.

        Yields state snapshots as the agent progresses through nodes.
        """
        try:
            state = self._prepare_state(query)
            graph = get_agent_graph()

            async for event in graph.astream(state):
                for node_name, node_state in event.items():
                    yield {
                        "type": "agent_step",
                        "node": node_name,
                        "snapshot": state_snapshot(node_state),
                    }
            yield {"type": "done", "answer": node_state.get("answer", "")}

        except ImportError:
            yield {"type": "error", "message": "LangGraph not available"}
        except Exception as exc:
            logger.error("Agent stream failed: %s", exc)
            yield {"type": "error", "message": str(exc)}

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_response(
        self, query: RAGQuery, final_state: AgentState, t0: float
    ) -> RAGResponse:
        """Build a RAGResponse from the final AgentState."""
        answer = final_state.get("answer") or ""
        status = final_state.get("status", AgentStatus.COMPLETED.value)
        citations_raw = final_state.get("citations", [])

        # Convert citations
        citations = [
            RAGCitation(
                chunk_id=c.get("chunk_id", ""),
                document_id=c.get("document_id", ""),
                text_snippet=c.get("snippet", ""),
            )
            for c in citations_raw
        ]

        # Build context from retrieved chunks
        chunks = [
            RAGChunk(
                content=c.get("content", ""),
                source=RAGSource(
                    document_id=c.get("source", "unknown"),
                    score=c.get("score"),
                ),
            )
            for c in final_state.get("retrieved_chunks", [])
        ]

        # Extract token usage if available
        token_usage: Dict[str, int] = {}
        if "token_usage" in final_state:
            token_usage = dict(final_state.get("token_usage", {}))  # type: ignore[arg-type]

        latency = (time.perf_counter() - t0) * 1000

        error_msg: Optional[str] = None
        if status == AgentStatus.FAILED.value:
            error_msg = final_state.get("fatal_error") or "Agent workflow failed"
        elif status == AgentStatus.DEGRADED.value:
            error_msg = "Agent completed with degradation"
        elif final_state.get("fallback_used"):
            status = AgentStatus.DEGRADED.value
            error_msg = "Fallback was used during agent execution"

        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            strategy=StrategyType.AGENTIC,
            context=RAGContext(
                query=query.text,
                chunks=chunks,
                retrieval_method="agent",
                retrieval_latency_ms=None,
            ),
            citations=citations,
            latency_ms=latency,
            token_usage=token_usage,
            error=error_msg,
            metadata={
                "agent_status": status,
                "iterations": final_state.get("iteration_count", 0),
                "plan": final_state.get("plan", []),
                "reasoning_steps": len(final_state.get("reasoning_chain", [])),
                "errors": final_state.get("errors", []),
                "tools_used": list(final_state.get("tool_results", {}).keys()),
                "steps": [
                    {
                        "action": s.get("action"),
                        "tool": s.get("tool_name"),
                        "error": s.get("error"),
                    }
                    for s in final_state.get("steps", [])
                ],
            },
        )

    async def _fallback_run(self, query: RAGQuery, t0: float) -> RAGResponse:
        """Simple retrieve → generate fallback when the agent workflow is unavailable."""
        try:
            context = await self.retrieve(query)
            response = await self.generate(context, query)
            response.latency_ms = (time.perf_counter() - t0) * 1000
            response.metadata["fallback"] = True
            response.metadata["fallback_reason"] = "Agent workflow unavailable"
            return response
        except Exception as exc:
            return RAGResponse(
                query_id=query.query_id,
                answer="",
                strategy=StrategyType.AGENTIC,
                error=f"Agentic RAG failed completely: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )


# ── Auto-register ─────────────────────────────────────────────────────────

try:
    _reg = get_registry()
    if not _reg.is_registered(RAGMode.AGENTIC):
        _reg.register(RAGMode.AGENTIC, AgenticRAGStrategy())
except Exception:
    pass  # Registration will be handled by explicit setup
