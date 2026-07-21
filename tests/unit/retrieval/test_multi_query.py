import asyncio

import pytest

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.retrieval.multi_query import (
    MultiQueryRetrievalError,
    merge_retrieval_hits,
    retrieve_queries,
)


def run(coro):
    return asyncio.run(coro)


def hit(chunk_id, score, *, kb_id="kb-b", rank=1):
    chunk = ChunkRecord(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        kb_id=kb_id,
        text=f"text for {chunk_id}",
        index=rank - 1,
        metadata={"kb_id": kb_id, "chunk_id": chunk_id},
    )
    return RetrievalHit(chunk=chunk, score=score, rank=rank)


class MappingRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        value = self.mapping[kwargs["query"]]
        if isinstance(value, Exception):
            raise value
        return value


def test_multi_query_uses_same_kb_stably_deduplicates_and_keeps_highest_score():
    retriever = MappingRetriever(
        {
            "original": [hit("a", 0.8), hit("shared", 0.5, rank=2)],
            "rewrite": [hit("shared", 0.9), hit("b", 0.7, rank=2)],
        }
    )

    report = run(
        retrieve_queries(
            retriever,
            ["original", "rewrite"],
            kb_id="kb-b",
            top_k=3,
            filters={"section": "guide"},
        )
    )

    assert [item.chunk.id for item in report.hits] == ["shared", "a", "b"]
    assert [item.rank for item in report.hits] == [1, 2, 3]
    assert report.hits[0].score == 0.9
    assert report.hits[0].metadata["matched_queries"] == [
        "original",
        "rewrite",
    ]
    assert report.hit_counts == {"original": 2, "rewrite": 2}
    assert {call["kb_id"] for call in retriever.calls} == {"kb-b"}
    assert all(call["filters"] == {"section": "guide"} for call in retriever.calls)


def test_one_query_failure_degrades_while_other_queries_continue():
    retriever = MappingRetriever(
        {
            "broken": RuntimeError("offline branch failed"),
            "working": [hit("ok", 0.7)],
        }
    )

    report = run(
        retrieve_queries(
            retriever,
            ["broken", "working"],
            kb_id="kb-b",
            top_k=2,
        )
    )

    assert [item.chunk.id for item in report.hits] == ["ok"]
    assert len(report.failures) == 1
    assert "RuntimeError" in report.failures[0]


def test_all_query_failures_raise_structured_error():
    retriever = MappingRetriever(
        {"one": RuntimeError("one"), "two": TimeoutError("two")}
    )

    with pytest.raises(MultiQueryRetrievalError, match="RuntimeError"):
        run(
            retrieve_queries(
                retriever,
                ["one", "two"],
                kb_id="kb-b",
                top_k=2,
            )
        )


def test_cross_kb_result_is_a_failed_query_and_never_merged():
    retriever = MappingRetriever({"query": [hit("foreign", 0.9, kb_id="kb-a")]})

    with pytest.raises(MultiQueryRetrievalError, match="another kb"):
        run(
            retrieve_queries(
                retriever,
                ["query"],
                kb_id="kb-b",
                top_k=1,
            )
        )


def test_cross_branch_merge_has_stable_tie_break_and_no_duplicates():
    first = hit("b", 0.8)
    first.metadata["matched_queries"] = ["original"]
    second = hit("a", 0.8)
    second.metadata["matched_queries"] = ["rewrite"]
    duplicate = hit("b", 0.9)
    duplicate.metadata["matched_queries"] = ["hybrid"]

    merged = merge_retrieval_hits([[first, second], [duplicate]], top_k=3)

    assert [item.chunk.id for item in merged] == ["b", "a"]
    assert merged[0].metadata["matched_queries"] == ["original", "hybrid"]
