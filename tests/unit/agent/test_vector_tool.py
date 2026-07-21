import asyncio
from datetime import datetime, timezone
import json

import pytest

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.retrieval.vector_tool import (
    EMPTY_RESULT_SUMMARY,
    VectorSearchTool,
    register_vector_search_tool,
    vector_search,
)
from tests.tool_fakes import (
    FakeToolExecutor,
    FakeToolRegistry,
    FakeToolResult,
)


def run(coro):
    return asyncio.run(coro)


def make_hit(
    chunk_id="chunk-1",
    *,
    kb_id="kb-b",
    score=0.8,
    rank=1,
):
    metadata = {
        "kb_id": kb_id,
        "document_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "filename": "guide.pdf",
        "source": "guide.pdf",
        "page": 2,
        "section": "Vector Search",
        "indexed_at": datetime(2026, 7, 20, tzinfo=timezone.utc),
    }
    return RetrievalHit(
        chunk=ChunkRecord(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            kb_id=kb_id,
            text=f"real text for {chunk_id}",
            index=rank - 1,
            metadata=dict(metadata),
        ),
        score=score,
        rank=rank,
        retriever="vector",
        metadata=dict(metadata),
    )


class RecordingRetriever:
    retriever_name = "vector"

    def __init__(self, hits=None, error=None):
        self.hits = [] if hits is None else hits
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return list(self.hits)


def tool(retriever, **kwargs):
    return VectorSearchTool(
        retriever,
        result_factory=FakeToolResult,
        **kwargs,
    )


def test_vector_tool_schema_name_description_and_openai_shape_are_readable():
    instance = tool(RecordingRetriever())

    assert instance.name == "vector_search"
    assert instance.description
    schema = instance.parameter_schema
    assert schema["required"] == ["query", "kb_id"]
    assert schema["properties"]["top_k"]["maximum"] == 100
    assert schema["properties"]["filters"]["type"] == ["object", "null"]
    function = instance.to_openai_function()
    assert function["function"]["name"] == "vector_search"
    assert function["function"]["parameters"] == schema


def test_sync_invoke_calls_retriever_once_and_preserves_order_and_source_fields():
    retriever = RecordingRetriever(
        [make_hit("first", score=0.9), make_hit("second", score=0.7, rank=2)]
    )
    result = tool(retriever).invoke(
        " vector evidence ",
        " kb-b ",
        top_k=2,
        filters={"section": "Vector Search"},
    )

    assert result.success is True
    assert retriever.calls == [
        {
            "query": "vector evidence",
            "kb_id": "kb-b",
            "top_k": 2,
            "filters": {"section": "Vector Search"},
        }
    ]
    assert [item["chunk_id"] for item in result.data] == ["first", "second"]
    first = result.data[0]
    assert first == {
        "chunk_id": "first",
        "document_id": "doc-first",
        "kb_id": "kb-b",
        "text": "real text for first",
        "content": "real text for first",
        "score": 0.9,
        "rank": 1,
        "retriever": "vector",
        "source": "doc-first",
        "filename": "guide.pdf",
        "page": 2,
        "section": "Vector Search",
        "metadata": {
            "kb_id": "kb-b",
            "document_id": "doc-first",
            "chunk_id": "first",
            "filename": "guide.pdf",
            "source": "guide.pdf",
            "page": 2,
            "section": "Vector Search",
            "indexed_at": "2026-07-20T00:00:00+00:00",
        },
    }
    assert result.result_count == 2
    assert result.duration_ms >= 0
    json.dumps(result.data, ensure_ascii=False, allow_nan=False)


def test_ainvoke_and_c_execute_each_use_the_same_single_core_call():
    async_retriever = RecordingRetriever([make_hit()])
    async_result = run(
        tool(async_retriever).ainvoke("query", "kb-b", top_k=1)
    )
    execute_retriever = RecordingRetriever([make_hit()])
    execute_result = run(
        tool(execute_retriever).execute(
            query="query",
            kb_id="kb-b",
            top_k=1,
        )
    )

    assert async_result.success and execute_result.success
    assert len(async_retriever.calls) == 1
    assert len(execute_retriever.calls) == 1


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"query": "", "kb_id": "kb-b"}, "InvalidToolArgumentsError"),
        ({"query": "query", "kb_id": ""}, "MissingKnowledgeBaseError"),
        (
            {"query": "query", "kb_id": "kb-b", "top_k": 0},
            "InvalidToolArgumentsError",
        ),
        (
            {"query": "query", "kb_id": "kb-b", "top_k": -1},
            "InvalidToolArgumentsError",
        ),
        (
            {"query": "query", "kb_id": "kb-b", "top_k": 101},
            "InvalidToolArgumentsError",
        ),
        (
            {"query": "query", "kb_id": "kb-b", "filters": "bad"},
            "InvalidToolArgumentsError",
        ),
        (
            {
                "query": "query",
                "kb_id": "kb-b",
                "filters": {"kb_id": "kb-a"},
            },
            "InvalidToolArgumentsError",
        ),
    ],
)
def test_invalid_tool_arguments_return_structured_failure_without_retrieval(
    kwargs,
    error_type,
):
    retriever = RecordingRetriever([make_hit()])

    result = tool(retriever).invoke(**kwargs)

    assert result.success is False
    assert result.data is None
    assert f"tool=vector_search; type={error_type}" in result.error
    assert result.retryable is False
    assert result.duration_ms >= 0
    assert retriever.calls == []


def test_configurable_top_k_limit_and_unknown_arguments_are_explicit():
    limited = tool(RecordingRetriever(), max_top_k=10)
    too_large = limited.invoke("query", "kb-b", top_k=11)
    unknown = limited.invoke("query", "kb-b", unsupported=True)

    assert "must not exceed 10" in too_large.error
    assert "unknown tool argument" in unknown.error


def test_same_kb_filter_is_allowed_and_forwarded_without_mutation():
    filters = {"kb_id": "kb-b", "section": "guide"}
    retriever = RecordingRetriever([])

    result = tool(retriever).invoke("query", "kb-b", filters=filters)

    assert result.success is True
    assert retriever.calls[0]["filters"] == filters
    assert filters == {"kb_id": "kb-b", "section": "guide"}


def test_normalized_kb_id_accepts_matching_filter():
    retriever = RecordingRetriever([])

    result = tool(retriever).invoke(
        "query",
        " kb-b ",
        filters={"kb_id": "kb-b"},
    )

    assert result.success is True
    assert retriever.calls[0]["kb_id"] == "kb-b"


def test_missing_optional_citation_fields_are_not_fabricated():
    hit = make_hit()
    hit.chunk.metadata = {"source": "not-a-filename"}
    hit.metadata = {}

    result = tool(RecordingRetriever([hit])).invoke("query", "kb-b")

    assert result.success is True
    assert result.data[0]["filename"] is None
    assert result.data[0]["page"] is None
    assert result.data[0]["section"] is None


def test_empty_search_is_success_with_clear_summary_and_no_fake_citation():
    result = tool(RecordingRetriever([])).invoke("query", "kb-b")

    assert result.success is True
    assert result.data == []
    assert result.result_count == 0
    assert result.result_summary == EMPTY_RESULT_SUMMARY
    assert result.error is None


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (RuntimeError("backend rejected request"), False),
        (TimeoutError("backend timed out"), True),
    ],
)
def test_retriever_error_is_structured_and_retryability_is_preserved(
    error,
    retryable,
):
    result = tool(RecordingRetriever(error=error)).invoke("query", "kb-b")

    assert result.success is False
    assert "tool=vector_search" in result.error
    assert f"type={type(error).__name__}" in result.error
    assert result.retryable is retryable
    assert result.error_type == type(error).__name__


def test_retriever_error_does_not_leak_stack_path_or_api_key():
    error = RuntimeError(
        "Traceback (most recent call last):\n"
        "File C:\\private\\service.py line 12 API_KEY=secret sk-secret123"
    )

    result = tool(RecordingRetriever(error=error)).invoke("query", "kb-b")

    assert result.success is False
    assert "Traceback" not in result.error
    assert "C:\\private" not in result.error
    assert "secret123" not in result.error


def test_cross_kb_or_malformed_hits_become_execution_error():
    cross_kb = tool(
        RecordingRetriever([make_hit("foreign", kb_id="kb-a")])
    ).invoke("query", "kb-b")

    assert cross_kb.success is False
    assert "another kb" in cross_kb.error
    assert cross_kb.data is None


def test_registry_executor_and_functional_entry_point_are_compatible():
    registry = FakeToolRegistry()
    first = register_vector_search_tool(
        registry,
        RecordingRetriever([make_hit("first")]),
        result_factory=FakeToolResult,
    )
    replacement_retriever = RecordingRetriever([make_hit("replacement")])
    replacement = register_vector_search_tool(
        registry,
        replacement_retriever,
        result_factory=FakeToolResult,
    )

    assert registry.get("vector_search") is replacement
    assert registry.get("vector_search") is not first
    result = run(
        FakeToolExecutor(registry).execute(
            "vector_search",
            query="query",
            kb_id="kb-b",
            top_k=1,
        )
    )
    direct = vector_search(
        RecordingRetriever([make_hit("direct")]),
        "query",
        "kb-b",
        result_factory=FakeToolResult,
    )

    assert [item["chunk_id"] for item in result.data] == ["replacement"]
    assert [item["chunk_id"] for item in direct.data] == ["direct"]
    assert len(replacement_retriever.calls) == 1
