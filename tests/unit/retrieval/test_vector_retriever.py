import math

import pytest

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.retrieval.embeddings import BaseEmbedder, HashEmbedder
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex


def make_chunk(chunk_id="c-1", kb_id="kb-1", category="guide"):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id=kb_id,
        text="RAG retrieval content",
        index=0,
        metadata={
            "kb_id": kb_id,
            "document_id": "doc-1",
            "chunk_id": chunk_id,
            "filename": "guide.pdf",
            "page": 2,
            "section": "Retrieval",
            "category": category,
        },
    )


class RecordingEmbedder(BaseEmbedder):
    def __init__(self):
        self.calls = []

    @property
    def model_name(self):
        return "recording"

    @property
    def dimension(self):
        return 2

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def test_embedding_aliases_are_equal_ordered_finite_and_stable():
    embedder = HashEmbedder(dim=16)
    batch = embedder.embed_texts(["first", "second"])

    assert embedder.embed("first") == embedder.embed_query("first") == batch[0]
    assert batch == embedder.embed_texts(["first", "second"])
    assert batch[0] != batch[1]
    assert embedder.embed_texts([]) == []
    assert all(math.isfinite(value) for vector in batch for value in vector)
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed("")


def test_vector_retriever_embeds_once_returns_standard_hits_and_metadata():
    embedder = RecordingEmbedder()
    index = InMemoryVectorIndex(embedder)
    index.upsert(
        [make_chunk("c-2"), make_chunk("c-1")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    retriever = VectorRetriever(embedder, index)

    hits = retriever.search("RAG", kb_id="kb-1", top_k=2)

    assert embedder.calls == [["RAG"]]
    assert all(isinstance(hit, RetrievalHit) for hit in hits)
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.chunk.id for hit in hits] == ["c-1", "c-2"]
    assert all(hit.retriever == retriever.retriever_name == "vector" for hit in hits)
    for hit in hits:
        assert {
            "kb_id",
            "document_id",
            "chunk_id",
            "filename",
            "page",
            "section",
        } <= hit.metadata.keys()


def test_vector_retriever_filters_empty_results_and_kb_protection():
    embedder = RecordingEmbedder()
    index = InMemoryVectorIndex(embedder)
    index.upsert([make_chunk()], [[1.0, 0.0]])
    retriever = VectorRetriever(embedder, index)

    assert retriever.search(
        "RAG", "kb-1", filters={"category": "missing"}
    ) == []
    with pytest.raises(ValueError, match="cannot override kb_id"):
        retriever.search("RAG", "kb-1", filters={"kb_id": "kb-2"})
    with pytest.raises(ValueError, match="kb_id must be a non-empty string"):
        retriever.search("RAG", "")


def test_embedder_and_vector_index_errors_propagate_without_being_hidden():
    class FailingEmbedder(RecordingEmbedder):
        def embed_texts(self, texts):
            raise RuntimeError("embedding unavailable")

    class FailingIndex(InMemoryVectorIndex):
        def search(self, *args, **kwargs):
            raise RuntimeError("index unavailable")

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        VectorRetriever(FailingEmbedder(), InMemoryVectorIndex(HashEmbedder(2))).search(
            "query", "kb-1"
        )
    with pytest.raises(RuntimeError, match="index unavailable"):
        VectorRetriever(RecordingEmbedder(), FailingIndex(HashEmbedder(2))).search(
            "query", "kb-1"
        )
