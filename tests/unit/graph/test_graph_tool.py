import asyncio
import inspect
import json

import pytest

from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import (
    GraphGlobalSearchResult,
    GraphLocalSearchResult,
    GraphRetriever,
)
from src.graph.tool import GraphSearchTool
from src.models.knowledge import ChunkRecord


def run(coro):
    return asyncio.run(coro)


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "graph.md", "page": 1, "section": "demo"},
    )


def build_retriever():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    return GraphRetriever(builder.repository)


def test_graph_tool_contract_fields_and_async_execute():
    tool = GraphSearchTool(build_retriever())

    assert tool.name == "graph_search"
    assert tool.description
    assert inspect.iscoroutinefunction(tool.execute)
    assert tool.parameter_schema["properties"]["scope"]["enum"] == [
        "local",
        "global",
        "auto",
    ]


def test_local_scope_calls_local_search_and_preserves_traceability():
    tool = GraphSearchTool(build_retriever())

    result = run(tool.execute(query="RAG", kb_id="kb-1", scope="local", top_k=2))

    assert result.success is True
    assert result.tool_name == "graph_search"
    assert result.duration_ms >= 0
    assert result.data["scope"] == "local"
    assert result.data["result_count"] >= 1
    item = result.data["results"][0]
    assert item["graph_scope"] == "local"
    assert item["entity_id"]
    assert item["entity_name"] == "RAG"
    assert item["relationships"]
    assert item["neighbor_entities"]
    assert any(part.startswith("source_chunk:doc-1/") for part in item["path"])
    assert item["source_refs"][0]["document_id"] == "doc-1"
    assert item["source_refs"][0]["chunk_id"] in {"chunk-1", "chunk-2"}
    assert result.data["citations"][0]["document_id"] == "doc-1"
    json.dumps(result.data)


def test_global_scope_returns_community_report_fields():
    tool = GraphSearchTool(build_retriever())

    result = run(
        tool.execute(
            query="overall Retrieval theme",
            kb_id="kb-1",
            scope="global",
            top_k=2,
        )
    )

    assert result.success is True
    assert result.data["scope"] == "global"
    assert result.data["result_count"] >= 1
    item = result.data["results"][0]
    assert item["graph_scope"] == "global"
    assert item["community_id"]
    assert item["report_id"]
    assert item["title"].startswith("Community:")
    assert "contains" in item["summary"]
    assert item["key_entities"]
    assert item["key_relationships"]
    assert item["entity_id"] is None
    assert item["source_refs"][0]["document_id"] == "doc-1"
    assert any(part.startswith("community:") for part in item["path"])
    json.dumps(result.data)


def test_auto_scope_resolves_global_for_summary_and_local_by_default():
    tool = GraphSearchTool(build_retriever())

    global_result = run(
        tool.execute(
            query="summarize the overall Retrieval theme",
            kb_id="kb-1",
            scope="auto",
        )
    )
    local_result = run(tool.execute(query="RAG", kb_id="kb-1", scope="auto"))

    assert global_result.success is True
    assert global_result.data["requested_scope"] == "auto"
    assert global_result.data["scope"] == "global"
    assert local_result.success is True
    assert local_result.data["requested_scope"] == "auto"
    assert local_result.data["scope"] == "local"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"query": "", "kb_id": "kb-1"}, "query must be a non-empty string"),
        ({"query": "RAG", "kb_id": ""}, "kb_id must be a non-empty string"),
        ({"query": "RAG", "kb_id": "kb-1", "scope": "wide"}, "scope must be"),
        ({"query": "RAG", "kb_id": "kb-1", "top_k": 0}, "top_k must be"),
        ({"query": "RAG", "kb_id": "kb-1", "top_k": True}, "top_k must be"),
        ({"query": "RAG", "kb_id": "kb-1", "unknown": "x"}, "unknown tool"),
    ],
)
def test_argument_validation_returns_structured_failure(kwargs, message):
    result = run(GraphSearchTool(build_retriever()).execute(**kwargs))

    assert result.success is False
    assert result.data == {}
    assert result.tool_name == "graph_search"
    assert result.duration_ms >= 0
    assert message in result.error


def test_retriever_exception_returns_structured_failure_without_raising():
    class BrokenRetriever:
        def local_search(self, query, *, kb_id, top_k):
            raise RuntimeError("boom from graph")

    result = run(
        GraphSearchTool(BrokenRetriever()).execute(
            query="RAG",
            kb_id="kb-1",
            scope="local",
        )
    )

    assert result.success is False
    assert result.data == {}
    assert result.tool_name == "graph_search"
    assert "RuntimeError" in result.error
    assert "boom from graph" in result.error


def test_empty_hits_are_successful_with_results_and_warning():
    result = run(
        GraphSearchTool(build_retriever()).execute(
            query="Nonexistent",
            kb_id="kb-1",
            scope="local",
        )
    )

    assert result.success is True
    assert result.data["results"] == []
    assert result.data["result_count"] == 0
    assert result.data["warnings"] == ["no matching graph entities found"]
    json.dumps(result.data)


def test_tool_rejects_missing_retriever_method():
    result = run(
        GraphSearchTool(object()).execute(
            query="RAG",
            kb_id="kb-1",
            scope="local",
        )
    )

    assert result.success is False
    assert result.data == {}
    assert "GraphRetriever.local_search is not configured" in result.error


def test_tool_rejects_wrong_retriever_result_type():
    class WrongRetriever:
        def local_search(self, query, *, kb_id, top_k):
            return GraphGlobalSearchResult(query=query, kb_id=kb_id)

        def global_search(self, query, *, kb_id, top_k):
            return GraphLocalSearchResult(query=query, kb_id=kb_id)

    result = run(
        GraphSearchTool(WrongRetriever()).execute(
            query="RAG",
            kb_id="kb-1",
            scope="local",
        )
    )

    assert result.success is False
    assert "GraphRetriever.local_search must return GraphLocalSearchResult" in result.error
