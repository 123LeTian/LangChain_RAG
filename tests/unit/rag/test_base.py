"""
Tests for src/rag/base.py — RAGStrategyBase + Protocol interfaces.
"""

import pytest
from typing import List

from src.models.rag import (
    RAGContext,
    RAGChunk,
    RAGSource,
    RAGQuery,
    RAGResponse,
    StrategyType,
)
from src.rag.base import (
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
