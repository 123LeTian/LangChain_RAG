import pytest

import src.models.rag as rag_module

from src.models.knowledge import ChunkRecord
from src.retrieval.adapter import MissingKnowledgeBaseError, VectorRetrieverAdapter
from src.retrieval.embeddings import BaseEmbedder
from src.retrieval.protocols import RetrieverProtocol
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex


class CountingEmbedder(BaseEmbedder):
    def __init__(self):
        self.count = 0

    @property
    def model_name(self):
        return "counting"

    @property
    def dimension(self):
        return 2

    def embed_texts(self, texts):
        self.count += 1
        return [[1.0, 0.0] for _ in texts]


def make_adapter(context_factory=None, with_chunk=True):
    embedder = CountingEmbedder()
    index = InMemoryVectorIndex(embedder)
    if with_chunk:
        chunk = ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            kb_id="kb-1",
            text="adapter content",
            index=3,
            metadata={
                "kb_id": "kb-1",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "filename": "adapter.pdf",
                "source": "adapter.pdf",
                "page": 4,
                "section": "Adapter",
            },
        )
        index.upsert([chunk], [[1.0, 0.0]])
    adapter = VectorRetrieverAdapter(
        VectorRetriever(embedder, index),
        context_factory=context_factory,
    )
    return adapter, embedder


def test_adapter_structurally_satisfies_c_protocol_and_preserves_hits():
    captured = {}

    def context_factory(**kwargs):
        captured.update(kwargs)
        return kwargs

    adapter, embedder = make_adapter(context_factory)

    context = adapter.retrieve(
        "query",
        top_k=1,
        kb_id="kb-1",
        filters={"section": "Adapter"},
    )

    assert isinstance(adapter, RetrieverProtocol)
    assert adapter.retriever_name == "vector"
    assert embedder.count == 1
    hit = context["hits"][0]
    assert hit.chunk.id == "chunk-1"
    assert hit.score == 1.0
    assert hit.rank == 1
    assert hit.metadata["page"] == 4
    assert context["metadata"] == {
        "kb_id": "kb-1",
        "filters": {"section": "Adapter"},
    }


def test_adapter_search_bridges_c_injection_to_b_strategy_once(monkeypatch):
    adapter, embedder = make_adapter()
    wrapped = adapter._retriever
    original_search = wrapped.search
    calls = []

    def recording_search(query, kb_id, top_k=5, filters=None):
        calls.append(
            {
                "query": query,
                "kb_id": kb_id,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return original_search(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
        )

    monkeypatch.setattr(wrapped, "search", recording_search)

    hits = adapter.search(
        query="query",
        kb_id="kb-1",
        top_k=1,
        filters={"section": "Adapter"},
    )

    assert calls == [
        {
            "query": "query",
            "kb_id": "kb-1",
            "top_k": 1,
            "filters": {"section": "Adapter"},
        }
    ]
    assert embedder.count == 1
    assert [hit.chunk.id for hit in hits] == ["chunk-1"]


def test_adapter_default_factory_builds_c_contract_shape(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(rag_module, "RAGSource", Model, raising=False)
    monkeypatch.setattr(rag_module, "RAGChunk", Model, raising=False)
    monkeypatch.setattr(rag_module, "RAGContext", Model, raising=False)
    adapter, _ = make_adapter()

    context = adapter.retrieve("query", kb_id="kb-1")

    assert context.query == "query"
    assert context.retrieval_method == "vector"
    assert context.total_candidates == 1
    assert context.chunks[0].chunk_id == "chunk-1"
    assert context.chunks[0].source.score == 1.0
    assert context.chunks[0].source.metadata["rank"] == 1
    assert context.chunks[0].source.metadata["retriever"] == "vector"


def test_adapter_requires_kb_and_converts_empty_results():
    adapter, _ = make_adapter(lambda **kwargs: kwargs, with_chunk=False)

    with pytest.raises(MissingKnowledgeBaseError, match="kb_id is required"):
        adapter.retrieve("query")
    context = adapter.retrieve("query", kb_id="kb-empty")
    assert context["hits"] == []
    assert context["total_candidates"] == 0
