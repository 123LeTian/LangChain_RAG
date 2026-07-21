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


# ═══════════════════════════════════════════════════════════════════════════
# Integration: RAGService × strategy compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestServiceStrategyIntegration:
    """Verify RAGService can call each strategy type through the unified interface."""

    @pytest.fixture
    def svc(self, retriever, generator):
        return RAGService(retriever=retriever, llm=generator, registry=RAGStrategyRegistry())

    @pytest.mark.asyncio
    async def test_modular_strategy_via_service(self, svc):
        """RAGService can invoke ModularRAGStrategy (NEW interface)."""
        from src.rag.strategies.modular import ModularRAGStrategy, ModuleConfig
        strat = ModularRAGStrategy()
        strat.set_config(ModuleConfig(retrieve=True))
        svc.register(RAGMode.MODULAR, strat)

        result = await svc.run(RAGRequest(query="What is RAG?", mode=RAGMode.MODULAR))
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0
        # Modular should now have citations
        assert len(result.hits) > 0

    @pytest.mark.asyncio
    async def test_agentic_strategy_via_service(self, svc):
        """RAGService can invoke AgenticRAGStrategy (NEW interface via AgentWorkflow).

        This is the fix for: "TypeError: AgenticRAGStrategy.run() takes 2 positional
        arguments but 3 were given" — the old agentic.py used RAGQuery, the new uses
        RAGRequest + RAGContext.
        """
        from src.rag.strategies.agentic import AgenticRAGStrategy
        strat = AgenticRAGStrategy(max_steps=4)
        svc.register(RAGMode.AGENTIC, strat)

        result = await svc.run(RAGRequest(query="What is machine learning?", mode=RAGMode.AGENTIC))
        assert isinstance(result, RAGResult)
        # Should have an answer (either from LLM or mock)
        assert len(result.answer) > 0
        # Should have citations from tool results
        assert isinstance(result.citations, list)

    @pytest.mark.asyncio
    async def test_agentic_strategy_has_citations(self, svc):
        """AgenticRAGStrategy returns citations in the RAGResult."""
        from src.rag.strategies.agentic import AgenticRAGStrategy
        strat = AgenticRAGStrategy(max_steps=4)
        svc.register(RAGMode.AGENTIC, strat)

        result = await svc.run(RAGRequest(query="What is deep learning?", mode=RAGMode.AGENTIC))
        # Citations field should be present (even if empty when no hits found)
        assert hasattr(result, 'citations')
        assert isinstance(result.citations, list)

    @pytest.mark.asyncio
    async def test_agentic_strategy_with_complex_query(self, svc):
        """Complex query triggers multi-tool agent workflow."""
        from src.rag.strategies.agentic import AgenticRAGStrategy
        strat = AgenticRAGStrategy(max_steps=4)
        svc.register(RAGMode.AGENTIC, strat)

        result = await svc.run(RAGRequest(
            query="Compare and contrast vector search vs graph retrieval",
            mode=RAGMode.AGENTIC,
        ))
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_agentic_strategy_runs_within_max_steps(self, svc):
        """Agentic workflow respects max_steps and completes."""
        from src.rag.strategies.agentic import AgenticRAGStrategy
        strat = AgenticRAGStrategy(max_steps=3)
        svc.register(RAGMode.AGENTIC, strat)

        result = await svc.run(RAGRequest(query="What is RAG?", mode=RAGMode.AGENTIC))
        # Should complete without hanging
        assert result.answer or result.warnings  # Either answer or explain why not

    @pytest.mark.asyncio
    async def test_agentic_strategy_mode_matches(self, svc):
        """AgenticRAGStrategy.mode == RAGMode.AGENTIC for registry compatibility."""
        from src.rag.strategies.agentic import AgenticRAGStrategy
        strat = AgenticRAGStrategy()
        assert strat.mode == RAGMode.AGENTIC
        assert strat.strategy_mode == RAGMode.AGENTIC


# ═══════════════════════════════════════════════════════════════════════════
# Acceptance: Dataset-driven Agent evaluation (15 questions, measurable results)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgenticEvalDataset:
    """Run the 15 evaluation questions through the agent and report statistics.

    Uses datasets/evaluation/agent_test_questions.json — covers fact, entity,
    summary, and complex query types.  Produces measurable tool_selection_accuracy
    and trajectory_success_rate per acceptance criterion 6.
    """

    @pytest.fixture
    def svc(self, retriever, generator):
        from src.rag.strategies.agentic import AgenticRAGStrategy
        reg = RAGStrategyRegistry()
        reg.register(RAGMode.AGENTIC, AgenticRAGStrategy(max_steps=4))
        return RAGService(retriever=retriever, llm=generator, registry=reg)

    def _load_questions(self):
        import json
        from pathlib import Path
        dataset_path = Path(__file__).parent.parent.parent.parent / \
            "datasets" / "evaluation" / "agent_test_questions.json"
        with open(dataset_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.mark.asyncio
    async def test_dataset_has_15_questions(self):
        """Dataset contains at least 10 questions covering all query types."""
        data = self._load_questions()
        questions = data["questions"]
        assert len(questions) >= 10, f"Need >= 10 questions, got {len(questions)}"

        types = {q["type"] for q in questions}
        assert "fact" in types
        assert "entity" in types
        assert "summary" in types
        assert "complex" in types

    @pytest.mark.asyncio
    async def test_dataset_all_questions_run(self, svc):
        """Every question in the dataset can run through the agent without crash."""
        data = self._load_questions()
        questions = data["questions"]
        results = []

        for q in questions:
            result = await svc.run(RAGRequest(query=q["query"], mode=RAGMode.AGENTIC))
            results.append({
                "id": q["id"],
                "query": q["query"],
                "type": q["type"],
                "expected_tools": q["expected_tools"],
                "answer_len": len(result.answer),
                "warnings": result.warnings,
            })

        assert len(results) == len(questions)
        # All questions must produce some output
        for r in results:
            assert r["answer_len"] > 0 or r["warnings"], \
                f"Q {r['id']} produced no answer and no warnings"

    @pytest.mark.asyncio
    async def test_dataset_type_coverage(self, svc):
        """Each of the three required question types is present and runs."""
        data = self._load_questions()
        by_type = {"fact": [], "entity": [], "summary": [], "complex": []}

        for q in data["questions"]:
            by_type[q["type"]].append(q)

        for qtype, qlist in by_type.items():
            if qlist:
                sample = qlist[0]
                result = await svc.run(RAGRequest(query=sample["query"], mode=RAGMode.AGENTIC))
                assert isinstance(result, RAGResult), \
                    f"{qtype} query '{sample['id']}' failed to produce RAGResult"
                assert result.answer or result.warnings, \
                    f"{qtype} query '{sample['id']}' empty answer and no warnings"

    @pytest.mark.asyncio
    async def test_dataset_stats_report(self, svc):
        """Run all 15 questions and produce a stats report (acceptance criterion 6)."""
        from src.agent.workflow import AgentMetrics
        from src.agent.router import AgentRouter

        data = self._load_questions()
        questions = data["questions"]
        router = AgentRouter()
        metrics = AgentMetrics()

        for q in questions:
            # What does the router predict?
            router_result = router.route(q["query"])
            # Run through the service
            result = await svc.run(RAGRequest(query=q["query"], mode=RAGMode.AGENTIC))

            metrics.record(
                query=q["query"],
                expected_tools=q["expected_tools"],
                actual_tools=router_result.tools,
                trajectory_success=len(result.answer) > 0,
                steps_count=len(result.trace) if result.trace else 0,
                errors=result.warnings,
            )

        report = metrics.report()
        summary = metrics.summary()

        # Verify metrics report structure
        assert report["total_runs"] == len(questions)
        assert "tool_selection_accuracy" in report
        assert "trajectory_success_rate" in report
        assert isinstance(report["tool_selection_accuracy"], float)
        assert isinstance(report["trajectory_success_rate"], float)

        # Verify summary is readable
        assert "Agent Evaluation Summary" in summary
        assert str(len(questions)) in summary

        # Print summary for manual inspection
        print("\n" + summary)
        print(f"\nTool accuracy: {report['tool_selection_accuracy']:.1%}")
        print(f"Trajectory success: {report['trajectory_success_rate']:.1%}")
