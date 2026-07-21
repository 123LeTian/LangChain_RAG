import pytest

from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphRetriever
from src.models.knowledge import ChunkRecord
from src.models.rag import RAGContext, RAGMode, RAGRequest, RAGResult
from src.rag.registry import RAGStrategyRegistry
from src.rag.service import RAGService
from src.rag.strategies.graph_rag import GraphRAGStrategy


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "graph.md", "page": 1, "section": "flow"},
    )


class DummyRetriever:
    retriever_name = "unused_vector"

    def retrieve(self, query: str, top_k: int = 5, **kwargs):
        return RAGContext(query=query, chunks=[])


class DummyLLM:
    async def generate(self, prompt: str, context: str, **kwargs):
        return "Service-level graph answer"


@pytest.mark.asyncio
async def test_graph_rag_strategy_runs_through_rag_service():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    registry = RAGStrategyRegistry()
    registry.register(
        RAGMode.GRAPH,
        GraphRAGStrategy(GraphRetriever(builder.repository)),
    )
    service = RAGService(
        retriever=DummyRetriever(),
        llm=DummyLLM(),
        registry=registry,
    )

    result = await service.run(
        RAGRequest(
            query="RAG",
            kb_id="kb-1",
            mode=RAGMode.GRAPH,
            options={"graph_scope": "local"},
        )
    )

    assert isinstance(result, RAGResult)
    assert result.answer == "Service-level graph answer"
    assert result.citations
    assert result.hits[0].source.metadata["path"]
