import sys

import pytest

from src.models.knowledge import ChunkRecord
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.vector_index import ChromaVectorIndex, InMemoryVectorIndex


def chunk(
    chunk_id: str,
    kb_id: str,
    document_id: str,
    *,
    category: str = "general",
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        kb_id=kb_id,
        document_id=document_id,
        text=chunk_id,
        index=0,
        metadata={
            "kb_id": kb_id,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "filename": f"{document_id}.txt",
            "page": 1,
            "section": category,
            "category": category,
        },
    )


def test_kb_filter_happens_before_top_k_global_high_scores_cannot_crowd_target():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    kb_a = [chunk(f"a-{number:02d}", "kb-a", f"doc-a-{number}") for number in range(10)]
    kb_b = chunk("b-target", "kb-b", "doc-b")
    index.upsert(kb_a + [kb_b], [[1.0, 0.0]] * 10 + [[0.7, 0.7]])

    hits = index.search([1.0, 0.0], kb_id="kb-b", top_k=1)

    assert [hit.chunk.id for hit in hits] == ["b-target"]
    assert all(hit.chunk.kb_id == "kb-b" for hit in hits)


def test_metadata_filters_are_applied_before_ranking():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    high_wrong = chunk("high-wrong", "kb-a", "doc-1", category="wrong")
    lower_right = chunk("lower-right", "kb-a", "doc-2", category="right")
    index.upsert([high_wrong, lower_right], [[1.0, 0.0], [0.6, 0.8]])

    hits = index.search(
        [1.0, 0.0],
        kb_id="kb-a",
        top_k=1,
        filters={"category": "right"},
    )

    assert [hit.chunk.id for hit in hits] == ["lower-right"]


def test_stable_tie_break_uses_chunk_id():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    chunks = [
        chunk("z-last", "kb-a", "doc-z"),
        chunk("a-first", "kb-a", "doc-a"),
        chunk("m-middle", "kb-a", "doc-m"),
    ]
    index.upsert(chunks, [[1.0, 0.0]] * 3)

    first = index.search([1.0, 0.0], "kb-a", top_k=3)
    second = index.search([1.0, 0.0], "kb-a", top_k=3)

    assert [hit.chunk.id for hit in first] == ["a-first", "m-middle", "z-last"]
    assert [hit.chunk.id for hit in second] == [hit.chunk.id for hit in first]


def test_count_and_deletes_are_isolated():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    chunks = [
        chunk("a-1", "kb-a", "doc-shared-a"),
        chunk("a-2", "kb-a", "doc-delete"),
        chunk("b-1", "kb-b", "doc-b"),
    ]
    index.upsert(chunks, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    assert index.count() == 3
    assert index.count("kb-a") == 2
    assert index.count("kb-b") == 1
    assert index.delete_by_document_id("doc-delete") == 1
    assert index.count("kb-a") == 1
    assert index.count("kb-b") == 1
    assert index.delete_by_kb_id("kb-a") == 1
    assert index.count("kb-a") == 0
    assert index.count("kb-b") == 1
    assert [hit.chunk.id for hit in index.search([1.0, 1.0], "kb-b")] == [
        "b-1"
    ]


def test_duplicate_upsert_replaces_instead_of_appending():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    original = chunk("same", "kb-a", "doc-a")
    replacement = chunk("same", "kb-a", "doc-a", category="updated")
    replacement.text = "replacement"
    index.upsert([original], [[1.0, 0.0]])

    index.upsert([replacement], [[0.0, 1.0]])

    assert index.count("kb-a") == 1
    hit = index.search([0.0, 1.0], "kb-a", filters={"category": "updated"})[0]
    assert hit.chunk.text == "replacement"


def test_empty_kb_invalid_top_k_dimension_and_missing_kb_are_explicit():
    index = InMemoryVectorIndex(HashEmbedder(dim=2))
    index.upsert([chunk("a", "kb-a", "doc-a")], [[1.0, 0.0]])

    assert index.search([1.0, 0.0], "kb-empty") == []
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search([1.0, 0.0], "kb-a", top_k=0)
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search([1.0, 0.0, 0.0], "kb-a")
    with pytest.raises(ValueError, match="kb_id must be a non-empty string"):
        index.search([1.0, 0.0], "")
    with pytest.raises(ValueError, match="cannot override kb_id"):
        index.search([1.0, 0.0], "kb-a", filters={"kb_id": "kb-b"})


def test_chroma_search_pushes_kb_and_metadata_filters_into_where_clause():
    calls = {}

    class FakeCollection:
        def get(self, **kwargs):
            calls["get_where"] = kwargs.get("where")
            return {"ids": ["candidate-1", "candidate-2"]}

        def query(self, **kwargs):
            calls.update(kwargs)
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }

    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._collection = FakeCollection()

    assert index.search(
        [1.0, 0.0],
        kb_id="kb-b",
        top_k=2,
        filters={"section": "guide"},
    ) == []
    assert calls["where"] == {
        "$and": [
            {"kb_id": {"$eq": "kb-b"}},
            {"section": {"$eq": "guide"}},
        ]
    }
    assert calls["get_where"] == calls["where"]
    assert calls["n_results"] == 2


def test_chroma_delete_uses_database_filter_and_optional_import_is_lazy(monkeypatch):
    calls = {}

    class FakeCollection:
        def get(self, **kwargs):
            calls["where"] = kwargs["where"]
            return {"ids": ["a-1", "a-2"]}

        def delete(self, **kwargs):
            calls["deleted_ids"] = kwargs["ids"]

    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._collection = FakeCollection()

    assert index.delete_by_kb_id("kb-a") == 2
    assert calls == {
        "where": {"kb_id": {"$eq": "kb-a"}},
        "deleted_ids": ["a-1", "a-2"],
    }

    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError, match="requirements-chroma.txt"):
        ChromaVectorIndex(HashEmbedder(dim=2))
