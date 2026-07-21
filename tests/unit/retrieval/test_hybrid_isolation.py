import pytest

from src.models.knowledge import ChunkRecord
from src.retrieval.embeddings import BaseEmbedder
from src.retrieval.hybrid import HybridRetrievalError, HybridRetriever
from src.retrieval.vector_index import InMemoryVectorIndex


class QueryEmbedder(BaseEmbedder):
    @property
    def model_name(self):
        return "query"

    @property
    def dimension(self):
        return 2

    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def make_chunk(chunk_id, kb_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        kb_id=kb_id,
        text=text,
        index=0,
        metadata={
            "kb_id": kb_id,
            "document_id": f"doc-{chunk_id}",
            "chunk_id": chunk_id,
            "filename": f"{chunk_id}.txt",
            "page": 1,
            "section": "hybrid",
        },
    )


def test_vector_keyword_and_fused_hybrid_results_never_cross_kb():
    embedder = QueryEmbedder()
    index = InMemoryVectorIndex(embedder)
    hybrid = HybridRetriever(embedder, index, alpha=0.5)
    kb_a = [make_chunk(f"a-{i}", "kb-a", "target query") for i in range(5)]
    kb_b = make_chunk("b-target", "kb-b", "target query")
    index.upsert(kb_a + [kb_b], [[1.0, 0.0]] * 5 + [[0.6, 0.8]])
    hybrid.keyword_searcher.add_chunks(kb_a + [kb_b])

    vector_hits = hybrid.vector_retriever.search("target query", "kb-b", top_k=1)
    keyword_hits = hybrid.keyword_searcher.search(
        "target query", kb_id="kb-b", top_k=1
    )
    hybrid_hits = hybrid.search("target query", "kb-b", top_k=1)

    assert [hit.chunk.id for hit in vector_hits] == ["b-target"]
    assert [hit.chunk.id for hit in keyword_hits] == ["b-target"]
    assert [hit.chunk.id for hit in hybrid_hits] == ["b-target"]
    assert all(hit.chunk.kb_id == "kb-b" for hit in hybrid_hits)


def test_hybrid_vector_failure_degrades_to_keyword_results_with_warning(monkeypatch):
    embedder = QueryEmbedder()
    index = InMemoryVectorIndex(embedder)
    hybrid = HybridRetriever(embedder, index, alpha=0.5)
    chunk = make_chunk("keyword", "kb-b", "target query")
    hybrid.keyword_searcher.add_chunks([chunk])

    def fail_vector(*args, **kwargs):
        raise RuntimeError("vector unavailable")

    monkeypatch.setattr(hybrid.vector_retriever, "search", fail_vector)
    hits = hybrid.search("target query", "kb-b", top_k=1)

    assert [hit.chunk.id for hit in hits] == ["keyword"]
    assert any("vector branch degraded" in warning for warning in hybrid.last_warnings)


def test_hybrid_keyword_failure_degrades_to_vector_results_with_warning(monkeypatch):
    embedder = QueryEmbedder()
    index = InMemoryVectorIndex(embedder)
    hybrid = HybridRetriever(embedder, index, alpha=0.5)
    chunk = make_chunk("vector", "kb-b", "target query")
    index.upsert([chunk], [[1.0, 0.0]])

    def fail_keyword(*args, **kwargs):
        raise TimeoutError("keyword unavailable")

    monkeypatch.setattr(hybrid.keyword_searcher, "search", fail_keyword)
    hits = hybrid.search("target query", "kb-b", top_k=1)

    assert [hit.chunk.id for hit in hits] == ["vector"]
    assert any("keyword branch degraded" in warning for warning in hybrid.last_warnings)


def test_hybrid_raises_only_when_both_branches_fail(monkeypatch):
    embedder = QueryEmbedder()
    hybrid = HybridRetriever(embedder, InMemoryVectorIndex(embedder))

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(hybrid.vector_retriever, "search", fail)
    monkeypatch.setattr(hybrid.keyword_searcher, "search", fail)

    with pytest.raises(HybridRetrievalError, match="vector branch degraded"):
        hybrid.search("target query", "kb-b", top_k=1)
