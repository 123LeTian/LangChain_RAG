"""
Tests for src/rag/service.py — RAGService (C3).
"""

import asyncio
import pytest

from src.models.rag import (
    RAGChunk,
    RAGContext,
    RAGMode,
    RAGRequest,
    RAGResult,
    RAGSource,
    StrategyType,
    TraceStage,
)
from src.rag.base import (
    GeneratorProtocol,
    RAGStrategy,
    RetrieverProtocol,
)
from src.rag.registry import RAGStrategyRegistry, get_registry, set_registry
from src.rag.service import (
    ExecutionCancelledError,
    ExecutionTimeoutError,
    RAGService,
    RAGServiceError,
    StrategyNotAvailableError,
)


# ── Global cleanup ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_registry():
    """Reset the global registry before each test to avoid cross-test contamination."""
    set_registry(RAGStrategyRegistry())
    yield
    set_registry(RAGStrategyRegistry())


# ── Mock Implementations ───────────────────────────────────────────────────


class MockRetriever:
    """Fake retriever that returns predictable chunks."""

    def __init__(self, chunks=None):
        self._chunks = chunks or [
            RAGChunk(content="Mock content.", source=RAGSource(document_id="doc-1"))
        ]
        self.call_count = 0

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RAGContext:
        self.call_count += 1
        return RAGContext(query=query, chunks=list(self._chunks))

    @property
    def retriever_name(self) -> str:
        return "mock_retriever"


class MockGenerator:
    """Fake LLM generator."""

    def __init__(self, response: str = "Mock answer."):
        self._response = response
        self.call_count = 0

    async def generate(self, prompt: str, context: str, **kwargs) -> str:
        self.call_count += 1
        return self._response

    async def generate_with_tokens(self, prompt: str, context: str, **kwargs):
        self.call_count += 1
        return (self._response, {"prompt": 10, "completion": 5, "total": 15})


class _SuccessStrategy(RAGStrategy):
    """Happy-path: completes successfully with answer."""

    mode: RAGMode = RAGMode.NAIVE

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        assert context.retriever is not None, "retriever not injected"
        assert context.llm is not None, "llm not injected"

        ctx = context.retriever.retrieve(request.query)
        answer = await context.llm.generate(prompt="test", context=ctx.combined_text)

        self._record(context, TraceStage.GENERATE,
                     request.query, answer, 10.0)

        return RAGResult(answer=answer, hits=ctx.chunks)


class _SlowStrategy(RAGStrategy):
    """Slow strategy used to test timeout."""

    mode: RAGMode = RAGMode.NAIVE

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        await asyncio.sleep(10.0)
        return RAGResult(answer="too late")


class _CancellableStrategy(RAGStrategy):
    """Strategy that completes quickly for cancellation tests."""

    mode: RAGMode = RAGMode.NAIVE

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        await asyncio.sleep(0.5)
        return RAGResult(answer="done")


class _FailingStrategy(RAGStrategy):
    """Strategy that always raises."""

    mode: RAGMode = RAGMode.AGENTIC

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        raise RuntimeError("deliberate strategy failure")


class _GraphStrategy(RAGStrategy):
    """Strategy for GRAPH mode."""

    mode: RAGMode = RAGMode.GRAPH

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        return RAGResult(answer="graph result")


class _InspectorStrategy(RAGStrategy):
    """Strategy that captures its context for inspection."""

    mode: RAGMode = RAGMode.NAIVE

    captured_retriever = None
    captured_llm = None

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        _InspectorStrategy.captured_retriever = context.retriever
        _InspectorStrategy.captured_llm = context.llm
        return RAGResult(answer="ok")


class _PureStrategy(RAGStrategy):
    """Strategy using ONLY context-provided deps."""

    mode: RAGMode = RAGMode.NAIVE

    captured_context = None

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        _PureStrategy.captured_context = context
        result = context.retriever.retrieve(request.query)
        answer = await context.llm.generate("prompt", result.combined_text)
        return RAGResult(answer=answer, hits=result.chunks)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def retriever():
    return MockRetriever()


@pytest.fixture
def generator():
    return MockGenerator(response="Mock answer from LLM.")


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    return RAGStrategyRegistry()


@pytest.fixture
def service(retriever, generator, registry):
    return RAGService(
        retriever=retriever,
        llm=generator,
        registry=registry,
        default_timeout=5.0,
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestRAGServiceRun:
    """Tests for RAGService.run() — the unified orchestration entry point."""

    def test_service_creation(self, service):
        assert service._retriever is not None
        assert service._llm is not None
        assert service._registry is not None

    @pytest.mark.asyncio
    async def test_run_happy_path(self, service):
        """Full flow: register → run → get result."""
        service.register(RAGMode.NAIVE, _SuccessStrategy())

        request = RAGRequest(query="What is RAG?", mode=RAGMode.NAIVE)
        result = await service.run(request)

        assert isinstance(result, RAGResult)
        assert result.answer == "Mock answer from LLM."
        assert len(result.hits) == 1
        assert result.hits[0].content == "Mock content."

    @pytest.mark.asyncio
    async def test_run_unregistered_mode_raises(self, service):
        """Requesting an unregistered mode raises StrategyNotAvailableError."""
        request = RAGRequest(query="test", mode=RAGMode.GRAPH)
        with pytest.raises(StrategyNotAvailableError) as exc:
            await service.run(request)
        assert "graph" in str(exc.value)

    @pytest.mark.asyncio
    async def test_run_strategy_error_propagates(self, service):
        """Strategy exceptions propagate to the caller."""
        service.register(RAGMode.AGENTIC, _FailingStrategy())

        request = RAGRequest(query="test", mode=RAGMode.AGENTIC)
        with pytest.raises(RuntimeError, match="deliberate strategy failure"):
            await service.run(request)

    @pytest.mark.asyncio
    async def test_run_timeout(self, retriever, generator):
        """Execution exceeds timeout → ExecutionTimeoutError."""
        reg = RAGStrategyRegistry()
        svc = RAGService(
            retriever=retriever, llm=generator,
            registry=reg, default_timeout=0.1,
        )
        reg.register(RAGMode.NAIVE, _SlowStrategy())

        request = RAGRequest(query="test", mode=RAGMode.NAIVE)
        with pytest.raises(ExecutionTimeoutError):
            await svc.run(request, timeout=0.05)

    @pytest.mark.asyncio
    async def test_run_cancel_via_event(self, retriever, generator):
        """Cancel event set mid-flight → ExecutionCancelledError."""
        reg = RAGStrategyRegistry()
        svc = RAGService(retriever=retriever, llm=generator, registry=reg)
        reg.register(RAGMode.NAIVE, _CancellableStrategy())

        cancel_evt = asyncio.Event()

        async def cancel_soon():
            await asyncio.sleep(0.01)
            cancel_evt.set()

        request = RAGRequest(query="test", mode=RAGMode.NAIVE)
        with pytest.raises(ExecutionCancelledError):
            await asyncio.gather(
                svc.run(request, cancel_event=cancel_evt),
                cancel_soon(),
            )

    @pytest.mark.asyncio
    async def test_run_injects_dependencies_into_context(self, service):
        """Service injects retriever + LLM + tools into context."""
        service.register(RAGMode.NAIVE, _InspectorStrategy())
        await service.run(RAGRequest(query="test", mode=RAGMode.NAIVE))

        assert _InspectorStrategy.captured_retriever is service._retriever
        assert _InspectorStrategy.captured_llm is service._llm

    @pytest.mark.asyncio
    async def test_run_strategy_never_creates_own_deps(self, service):
        """Verify strategy receives deps from context, not from globals."""
        service.register(RAGMode.NAIVE, _PureStrategy())
        result = await service.run(RAGRequest(query="test", mode=RAGMode.NAIVE))

        assert result.answer == "Mock answer from LLM."
        assert _PureStrategy.captured_context is not None


class TestRAGServiceRunSafe:
    """Tests for RAGService.run_safe() — never-raises variant."""

    @pytest.fixture
    def svc(self, retriever, generator):
        reg = RAGStrategyRegistry()
        return RAGService(retriever=retriever, llm=generator, registry=reg)

    @pytest.mark.asyncio
    async def test_run_safe_success(self, svc):
        svc.register(RAGMode.NAIVE, _SuccessStrategy())
        result = await svc.run_safe(RAGRequest(query="test", mode=RAGMode.NAIVE))
        assert result.answer == "Mock answer from LLM."

    @pytest.mark.asyncio
    async def test_run_safe_unregistered_returns_warning(self, svc):
        result = await svc.run_safe(RAGRequest(query="test", mode=RAGMode.GRAPH))
        assert result.answer == ""
        assert any("StrategyNotAvailable" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_run_safe_timeout_returns_warning(self, retriever, generator):
        reg = RAGStrategyRegistry()
        svc = RAGService(retriever=retriever, llm=generator, registry=reg, default_timeout=0.1)
        reg.register(RAGMode.NAIVE, _SlowStrategy())
        result = await svc.run_safe(RAGRequest(query="test", mode=RAGMode.NAIVE), timeout=0.05)
        assert result.answer == ""
        assert any("Timeout" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_run_safe_strategy_error_returns_warning(self, svc):
        svc.register(RAGMode.AGENTIC, _FailingStrategy())
        result = await svc.run_safe(RAGRequest(query="test", mode=RAGMode.AGENTIC))
        assert result.answer == ""
        assert any("UnexpectedError" in w for w in result.warnings)
        assert any("RuntimeError" in w for w in result.warnings)


class TestRAGServiceManagement:
    """Tests for strategy registration and lookup via service."""

    @pytest.fixture
    def svc(self, retriever, generator):
        return RAGService(retriever=retriever, llm=generator, registry=RAGStrategyRegistry())

    def test_register_and_list_modes(self, svc):
        svc.register(RAGMode.NAIVE, _SuccessStrategy())
        svc.register(RAGMode.GRAPH, _GraphStrategy())
        modes = svc.list_modes()
        assert RAGMode.NAIVE in modes
        assert RAGMode.GRAPH in modes

    def test_list_modes_multi(self, svc):
        svc.register(RAGMode.NAIVE, _SuccessStrategy())
        svc.register(RAGMode.AGENTIC, _FailingStrategy())
        modes = svc.list_modes()
        assert RAGMode.NAIVE in modes
        assert RAGMode.AGENTIC in modes
        assert len(modes) == 2

    def test_is_available(self, svc):
        assert not svc.is_available(RAGMode.GRAPH)
        svc.register(RAGMode.GRAPH, _GraphStrategy())
        assert svc.is_available(RAGMode.GRAPH)


class TestServiceErrorHierarchy:
    """Verify structured error class hierarchy."""

    def test_errors_extend_base(self):
        assert issubclass(StrategyNotAvailableError, RAGServiceError)
        assert issubclass(ExecutionTimeoutError, RAGServiceError)
        assert issubclass(ExecutionCancelledError, RAGServiceError)

    def test_errors_are_exceptions(self):
        assert issubclass(RAGServiceError, Exception)
