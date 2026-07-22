"""
RAG Service — Owner: C
Unified RAG orchestration entry point.

Responsibilities:
  - Accept a RAGRequest, return a RAGResult
  - Create trace_id for observability
  - Resolve strategy from registry by RAGMode
  - Build execution context (inject retriever, LLM, tools, tracer, config)
  - Invoke strategy.run(request, context)
  - Catch and wrap exceptions as structured errors
  - Support timeout and cancellation

The service contains NO RAG algorithms — it delegates 100% to strategies.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from src.models.rag import (
    RAGContext,
    RAGMode,
    RAGRequest,
    RAGResult,
)
from src.rag.base import (
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategy,
    RetrieverProtocol,
)
from src.rag.registry import (
    RAGStrategyRegistry,
    StrategyNotRegisteredError,
    get_registry,
)
from src.rag.trace import TraceStore, get_trace_store

logger = logging.getLogger(__name__)


# ── Structured Error ──────────────────────────────────────────────────────


class RAGServiceError(Exception):
    """Base exception for RAGService-level errors."""


class StrategyNotAvailableError(RAGServiceError):
    """Raised when no strategy is registered for the requested mode."""

    def __init__(self, mode: RAGMode) -> None:
        self.mode = mode
        super().__init__(f"No strategy available for mode '{mode.value}'")


class ExecutionTimeoutError(RAGServiceError):
    """Raised when strategy execution exceeds the configured timeout."""

    def __init__(self, mode: RAGMode, timeout_s: float) -> None:
        self.mode = mode
        self.timeout_s = timeout_s
        super().__init__(
            f"Strategy '{mode.value}' timed out after {timeout_s:.1f}s"
        )


class ExecutionCancelledError(RAGServiceError):
    """Raised when strategy execution is explicitly cancelled."""

    def __init__(self, mode: RAGMode) -> None:
        self.mode = mode
        super().__init__(f"Strategy '{mode.value}' execution was cancelled")


# ── Service ───────────────────────────────────────────────────────────────


class RAGService:
    """Unified RAG orchestration service.

    This is THE entry point called by the API layer (Owner: A).
    It owns the strategy registry, trace store, and dependency injection.

    Usage::

        service = RAGService(
            retriever=my_retriever,
            llm=my_llm,
            graph=my_graph,          # optional
        )
        result = await service.run(RAGRequest(query="What is RAG?", mode=RAGMode.NAIVE))
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: GeneratorProtocol,
        graph: Optional[GraphRetrieverProtocol] = None,
        *,
        registry: Optional[RAGStrategyRegistry] = None,
        trace_store: Optional[TraceStore] = None,
        default_timeout: float = 60.0,
        tools: Optional[List[Any]] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._graph = graph
        self._registry = registry if registry is not None else get_registry()
        self._trace_store = trace_store or get_trace_store()
        self._default_timeout = default_timeout
        self._tools = tools or []

    # ── Public API ─────────────────────────────────────────────────────

    async def run(
        self,
        request: RAGRequest,
        *,
        timeout: Optional[float] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> RAGResult:
        """Execute the full RAG pipeline and return a unified result.

        Execution flow:
            1. Create trace_id for observability
            2. Resolve strategy from registry by request.mode
            3. Build RAGContext (inject retriever, LLM, tools, tracer, config)
            4. Call strategy.run(request, context)
            5. Catch exceptions → structured error in RAGResult.warnings
            6. Return RAGResult

        Args:
            request:      The user's query with mode selection and options.
            timeout:      Maximum execution time in seconds.  If exceeded,
                          raises :class:`ExecutionTimeoutError`.
            cancel_event: An optional asyncio.Event.  Set this event to
                          cancel execution mid-flight.

        Returns:
            RAGResult with answer, citations, hits, trace, usage, warnings.

        Raises:
            StrategyNotAvailableError: If no strategy is registered for
                                       *request.mode*.
            ExecutionTimeoutError:     If *timeout* is exceeded.
            ExecutionCancelledError:   If *cancel_event* is set.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"

        # ── 1. Validate mode ─────────────────────────────────────────
        mode = request.mode
        if not self._registry.is_registered(mode):
            raise StrategyNotAvailableError(mode)

        # ── 2. Build execution context ───────────────────────────────
        context = self._build_context(request, trace_id)

        # ── 3. Get strategy ──────────────────────────────────────────
        strategy = self._registry.get(mode)

        # ── 4. Execute with timeout + cancel support ─────────────────
        try:
            result = await self._execute_with_guard(
                strategy, request, context,
                timeout_s=effective_timeout,
                cancel_event=cancel_event,
                trace_id=trace_id,
            )
            return result
        except asyncio.TimeoutError:
            raise ExecutionTimeoutError(mode, effective_timeout)
        except asyncio.CancelledError:
            raise ExecutionCancelledError(mode)

    async def run_safe(
        self,
        request: RAGRequest,
        *,
        timeout: Optional[float] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> RAGResult:
        """Like :meth:`run`, but NEVER raises.

        All errors are captured and returned as a RAGResult with the error
        in ``warnings`` and an empty answer.  Suitable for API handlers
        that must always return a 200 response.
        """
        try:
            return await self.run(
                request, timeout=timeout, cancel_event=cancel_event
            )
        except StrategyNotAvailableError as exc:
            return RAGResult(
                answer="",
                warnings=[f"StrategyNotAvailable: {exc}"],
            )
        except ExecutionTimeoutError as exc:
            return RAGResult(
                answer="",
                warnings=[f"Timeout: {exc}"],
            )
        except ExecutionCancelledError as exc:
            return RAGResult(
                answer="",
                warnings=[f"Cancelled: {exc}"],
            )
        except Exception as exc:
            logger.exception("Unexpected error in RAG pipeline")
            return RAGResult(
                answer="",
                warnings=[f"UnexpectedError: {type(exc).__name__}: {exc}"],
            )

    # ── Strategy management ───────────────────────────────────────────

    def register(self, mode: RAGMode, strategy: Any) -> None:
        """Register a strategy instance for *mode*."""
        self._registry.register(mode, strategy)

    def list_modes(self) -> List[RAGMode]:
        """Return all registered RAGMode values."""
        return self._registry.list_modes()

    def is_available(self, mode: RAGMode) -> bool:
        """Check if a strategy is available for *mode*."""
        return self._registry.is_registered(mode)

    # ── Internal ───────────────────────────────────────────────────────

    def _build_context(
        self, request: RAGRequest, trace_id: str
    ) -> RAGContext:
        """Build a RAGContext with all dependencies injected.

        Strategy reads dependencies from the context — never creates its own.
        """
        from src.rag.trace import TraceRecorder

        recorder = TraceRecorder()
        recorder.start_trace(trace_id)

        context = RAGContext(
            query=request.query,
            chunks=[],
            retrieval_method="pending",
            llm=self._llm,
            retriever=self._retriever,
            tools=list(self._tools),
            trace_recorder=recorder,
            config=request.options,
            metadata={
                "trace_id": trace_id,
                "mode": request.mode.value,
                "session_id": request.session_id,
                "kb_id": request.kb_id,
            },
        )

        # Inject graph retriever for GraphRAG and Agentic strategies
        if self._graph is not None:
            context.graph = self._graph
            context.config["graph_retriever"] = self._graph

        return context

    async def _execute_with_guard(
        self,
        strategy: RAGStrategy,
        request: RAGRequest,
        context: RAGContext,
        timeout_s: float,
        cancel_event: Optional[asyncio.Event],
        trace_id: str,
    ) -> RAGResult:
        """Execute the strategy with timeout and cancellation guards."""
        async def _run() -> RAGResult:
            return await strategy.run(request, context)

        task = asyncio.create_task(_run())

        try:
            if cancel_event is not None:
                # Create a task for the cancel event waiter
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout_s,
                )

                if cancel_task.done() and cancel_event.is_set():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise asyncio.CancelledError(
                        f"Execution cancelled for trace {trace_id}"
                    )

                if task not in done:
                    # Timeout occurred
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise asyncio.TimeoutError(
                        f"Execution timed out after {timeout_s}s for trace {trace_id}"
                    )

                # Clean up cancel task
                if not cancel_task.done():
                    cancel_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await cancel_task

            else:
                result = await asyncio.wait_for(task, timeout=timeout_s)

            return task.result()

        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            raise
        except asyncio.TimeoutError:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            raise
