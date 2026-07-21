"""
Tests for src/rag/base.py — RAGStrategyBase + Protocol interfaces.
"""

import pytest
from typing import List

from src.models.rag import (
    RAGContext,
    RAGChunk,
    RAGMode,
    RAGSource,
    RAGQuery,
    RAGRequest,
    RAGResponse,
    RAGResult,
    StrategyType,
    TraceStage,
)
from src.rag.base import (
    RAGStrategy,
    RAGStrategyBase,
    RetrieverProtocol,
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RerankerProtocol,
    EmbedderProtocol,
)


# ── Protocol structural conformance ────────────────────────────────────────


class FakeRetriever:
    """Minimal retriever satisfying RetrieverProtocol."""

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RAGContext:
        return RAGContext(
            query=query,
            chunks=[
                RAGChunk(
                    content="test chunk",
                    source=RAGSource(document_id="doc-1"),
                )
            ],
        )

    @property
    def retriever_name(self) -> str:
        return "fake_retriever"


class FakeGenerator:
    """Minimal generator satisfying GeneratorProtocol."""

    async def generate(self, prompt: str, context: str, **kwargs) -> str:
        return f"Answer based on: {context[:50]}"

    async def generate_with_tokens(self, prompt: str, context: str, **kwargs):
        return (f"Answer based on: {context[:50]}", {"prompt": 10, "completion": 5, "total": 15})


class FakeGraphRetriever:
    """Minimal graph retriever satisfying GraphRetrieverProtocol."""

    def graph_search(self, query: str, top_k: int = 5, **kwargs) -> RAGContext:
        return RAGContext(query=query, chunks=[])

    def get_entity(self, entity_id: str):
        return {"id": entity_id, "name": "test"}

    def get_community_report(self, community_id: str):
        return "Community report text"


class TestProtocolConformance:
    """Verify that fake implementations satisfy the protocol interfaces."""

    def test_retriever_protocol(self):
        r = FakeRetriever()
        assert isinstance(r, RetrieverProtocol)
        ctx = r.retrieve("test query", top_k=3)
        assert len(ctx.chunks) == 1
        assert r.retriever_name == "fake_retriever"

    def test_generator_protocol(self):
        g = FakeGenerator()
        assert isinstance(g, GeneratorProtocol)

    @pytest.mark.asyncio
    async def test_generator_async(self):
        g = FakeGenerator()
        result = await g.generate("prompt", "context text here")
        assert "context text" in result

    @pytest.mark.asyncio
    async def test_generator_with_tokens(self):
        g = FakeGenerator()
        answer, tokens = await g.generate_with_tokens("p", "ctx")
        assert tokens["total"] == 15

    def test_graph_retriever_protocol(self):
        gr = FakeGraphRetriever()
        assert isinstance(gr, GraphRetrieverProtocol)
        entity = gr.get_entity("e1")
        assert entity["id"] == "e1"


# ── RAGStrategyBase ───────────────────────────────────────────────────────


class MinimalStrategy(RAGStrategyBase):
    """Minimal concrete strategy for testing the base class."""

    strategy_type = StrategyType.NAIVE

    async def retrieve(self, query: RAGQuery) -> RAGContext:
        return RAGContext(query=query.text, chunks=[])

    async def generate(self, context: RAGContext, query: RAGQuery) -> RAGResponse:
        return RAGResponse(
            query_id=query.query_id,
            answer="test answer",
            strategy=self.strategy_type,
            context=context,
        )

    async def run(self, query: RAGQuery) -> RAGResponse:
        ctx = await self.retrieve(query)
        return await self.generate(ctx, query)


class TestRAGStrategyBase:
    """Tests for the abstract base class."""

    def test_strategy_name(self):
        s = MinimalStrategy()
        assert s.strategy_name == "MinimalStrategy"

    def test_strategy_type(self):
        s = MinimalStrategy()
        assert s.strategy_type == StrategyType.NAIVE

    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        s = MinimalStrategy()
        q = RAGQuery(text="hello", strategy=StrategyType.NAIVE)
        resp = await s.run(q)
        assert resp.answer == "test answer"
        assert resp.strategy == StrategyType.NAIVE
        assert resp.query_id == q.query_id

    @pytest.mark.asyncio
    async def test_pre_retrieve_hook(self):
        s = MinimalStrategy()
        q = RAGQuery(text="original")
        result = await s.pre_retrieve(q)
        assert result.text == "original"

    @pytest.mark.asyncio
    async def test_post_retrieve_hook(self):
        s = MinimalStrategy()
        ctx = RAGContext(query="test", chunks=[])
        q = RAGQuery(text="test", strategy=StrategyType.NAIVE)
        result = await s.post_retrieve(ctx, q)
        assert result.query == "test"

    @pytest.mark.asyncio
    async def test_on_error_hook(self):
        s = MinimalStrategy()
        q = RAGQuery(text="test", strategy=StrategyType.NAIVE)
        resp = await s.on_error(q, ValueError("boom"))
        assert resp.error is not None
        assert "ValueError" in resp.error
        assert "boom" in resp.error


# ═══════════════════════════════════════════════════════════════════════════
# C2 — RAGStrategy Tests
# ═══════════════════════════════════════════════════════════════════════════


class _ConcreteRAG(RAGStrategy):
    """Minimal concrete RAGStrategy for testing the abstract interface."""

    mode: RAGMode = RAGMode.NAIVE

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        # Use injected retriever
        if context.retriever is not None:
            ctx = context.retriever.retrieve(request.query, top_k=5)
            context.chunks = ctx.chunks

        # Use injected LLM
        if context.llm is not None:
            answer = await context.llm.generate(
                prompt="You are a helpful assistant.",
                context=context.combined_text,
            )
        else:
            answer = "no LLM available"

        # Record trace
        self._record(context, TraceStage.RETRIEVE,
                     request.query, f"{len(context.chunks)} chunks", 12.5)

        return RAGResult(
            answer=answer,
            hits=context.chunks,
        )


class _NoDepsRAG(RAGStrategy):
    """RAGStrategy that works without any injected dependencies."""

    mode: RAGMode = RAGMode.ADVANCED

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        return RAGResult(
            answer=f"Echo: {request.query}",
            hits=context.chunks,
        )


class _FailingRAG(RAGStrategy):
    """RAGStrategy that always raises."""

    mode: RAGMode = RAGMode.AGENTIC

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        raise RuntimeError("deliberate failure")


class _HookedRAG(RAGStrategy):
    """RAGStrategy that uses pre/post hooks."""

    mode: RAGMode = RAGMode.MODULAR

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        await self.pre_run(request, context)
        result = RAGResult(answer=request.query.upper(), hits=context.chunks)
        return await self.post_run(request, context, result)

    async def pre_run(self, request: RAGRequest, context: RAGContext) -> None:
        context.metadata["pre_hook_called"] = True  # type: ignore[index]

    async def post_run(
        self, request: RAGRequest, context: RAGContext, result: RAGResult
    ) -> RAGResult:
        result.warnings.append("post_hook_was_here")
        return result


class TestRAGStrategy:
    """Tests for the high-level RAGStrategy abstract interface (C2)."""

    # ── Structure ──────────────────────────────────────────────────────

    def test_is_abstract(self):
        """RAGStrategy itself cannot be instantiated."""
        from abc import ABC
        assert issubclass(RAGStrategy, ABC)

    def test_mode_attribute(self):
        """Base class has mode attribute; subclasses override it."""
        s = _ConcreteRAG()
        assert s.mode == RAGMode.NAIVE

    def test_strategy_name(self):
        s = _ConcreteRAG()
        assert s.strategy_name == "_ConcreteRAG"

    def test_strategy_mode_property(self):
        s = _ConcreteRAG()
        assert s.strategy_mode == RAGMode.NAIVE

    def test_different_modes(self):
        assert _NoDepsRAG().mode == RAGMode.ADVANCED
        assert _FailingRAG().mode == RAGMode.AGENTIC
        assert _HookedRAG().mode == RAGMode.MODULAR

    # ── Dependency injection via context ───────────────────────────────

    @pytest.mark.asyncio
    async def test_run_with_full_deps(self):
        """Strategy uses retriever + LLM from context, never creates own."""
        retriever = FakeRetriever()
        generator = FakeGenerator()
        context = RAGContext(
            query="What is RAG?",
            retriever=retriever,
            llm=generator,
        )
        request = RAGRequest(query="What is RAG?", mode=RAGMode.NAIVE)

        strategy = _ConcreteRAG()
        result = await strategy.run(request, context)

        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0
        assert "test chunk" in result.answer or len(result.hits) > 0

    @pytest.mark.asyncio
    async def test_run_without_deps(self):
        """Strategy should still work when no deps are injected."""
        context = RAGContext(query="What is RAG?")
        request = RAGRequest(query="What is RAG?", mode=RAGMode.ADVANCED)

        strategy = _NoDepsRAG()
        result = await strategy.run(request, context)

        assert result.answer == "Echo: What is RAG?"
        assert result.hits == []

    @pytest.mark.asyncio
    async def test_run_with_retriever_only(self):
        """Strategy uses only the injected retriever."""
        retriever = FakeRetriever()
        context = RAGContext(query="test", retriever=retriever)
        request = RAGRequest(query="test", mode=RAGMode.NAIVE)

        strategy = _ConcreteRAG()
        result = await strategy.run(request, context)

        assert len(result.hits) == 1
        assert result.hits[0].content == "test chunk"

    # ── Hooks ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pre_post_hooks(self):
        """pre_run and post_run hooks are called during execution."""
        context = RAGContext(query="hello")
        request = RAGRequest(query="hello", mode=RAGMode.MODULAR)

        strategy = _HookedRAG()
        result = await strategy.run(request, context)

        assert context.metadata.get("pre_hook_called") is True
        assert "post_hook_was_here" in result.warnings

    @pytest.mark.asyncio
    async def test_pre_run_default_noop(self):
        """Default pre_run does nothing and doesn't crash."""
        strategy = _NoDepsRAG()
        context = RAGContext(query="test")
        request = RAGRequest(query="test", mode=RAGMode.ADVANCED)
        # Should not raise
        await strategy.pre_run(request, context)

    @pytest.mark.asyncio
    async def test_post_run_default_passthrough(self):
        """Default post_run returns result unchanged."""
        strategy = _NoDepsRAG()
        context = RAGContext(query="test")
        request = RAGRequest(query="test", mode=RAGMode.ADVANCED)
        result = RAGResult(answer="unchanged")
        out = await strategy.post_run(request, context, result)
        assert out is result

    # ── Error handling ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_on_error_default(self):
        """Default on_error returns a graceful failure result."""
        strategy = _ConcreteRAG()
        context = RAGContext(query="test")
        request = RAGRequest(query="test", mode=RAGMode.NAIVE)

        result = await strategy.on_error(request, context, RuntimeError("boom"))
        assert result.answer == ""
        assert len(result.warnings) == 1
        assert "RuntimeError" in result.warnings[0]
        assert "boom" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_error_during_run_is_not_caught_by_default(self):
        """RAGStrategy does NOT auto-catch errors — caller must handle."""
        strategy = _FailingRAG()
        context = RAGContext(query="test")
        request = RAGRequest(query="test", mode=RAGMode.AGENTIC)

        with pytest.raises(RuntimeError, match="deliberate failure"):
            await strategy.run(request, context)

    # ── Trace recording ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_trace_recorder_called(self):
        """_record() calls context.trace_recorder when available."""
        recorded: list = []

        def fake_recorder(stage, input_sum, output_sum, duration):
            recorded.append((stage, input_sum, output_sum, duration))

        retriever = FakeRetriever()
        generator = FakeGenerator()
        context = RAGContext(
            query="test",
            retriever=retriever,
            llm=generator,
            trace_recorder=fake_recorder,
        )
        request = RAGRequest(query="test", mode=RAGMode.NAIVE)

        strategy = _ConcreteRAG()
        await strategy.run(request, context)

        assert len(recorded) == 1
        assert recorded[0][0] == TraceStage.RETRIEVE
        assert recorded[0][1] == "test"

    @pytest.mark.asyncio
    async def test_trace_recorder_none_does_not_crash(self):
        """Calling _record with no trace_recorder is safe."""
        context = RAGContext(query="test")
        request = RAGRequest(query="test", mode=RAGMode.NAIVE)

        strategy = _ConcreteRAG()
        result = await strategy.run(request, context)
        assert isinstance(result, RAGResult)

    # ── RAGContext DI fields are excluded from serialization ────────────

    def test_context_di_fields_excluded_from_json(self):
        """llm, retriever, tools, trace_recorder, config are NOT in JSON output."""
        retriever = FakeRetriever()
        generator = FakeGenerator()
        ctx = RAGContext(
            query="test",
            retriever=retriever,
            llm=generator,
            config={"top_k": 10},
        )
        data = ctx.model_dump()
        assert "llm" not in data
        assert "retriever" not in data
        assert "tools" not in data
        assert "trace_recorder" not in data
        assert "config" not in data

    def test_context_di_fields_accessible_in_code(self):
        """DI fields are accessible as Python attributes."""
        retriever = FakeRetriever()
        ctx = RAGContext(query="test", retriever=retriever)
        assert ctx.retriever is retriever
        assert ctx.llm is None
        assert ctx.tools == []

    # ── RAGRequest → RAGQuery conversion ───────────────────────────────

    def test_request_to_query_conversion(self):
        """RAGRequest.to_rag_query() carries options into metadata."""
        request = RAGRequest(
            query="What is Python?",
            mode=RAGMode.AGENTIC,
            kb_id="kb-42",
            session_id="sess-1",
            options={"top_k": 10, "filters": {"lang": "en"}},
        )
        rq = request.to_rag_query()
        assert rq.strategy == StrategyType.AGENTIC
        assert rq.top_k == 10
        assert rq.conversation_id == "sess-1"
        assert rq.metadata["kb_id"] == "kb-42"
        assert rq.metadata["raw_options"]["filters"] == {"lang": "en"}
