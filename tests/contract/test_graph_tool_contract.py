import asyncio
import json

from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphRetriever
from src.graph.tool import GraphSearchTool, register_graph_search_tool
from src.models.knowledge import ChunkRecord
from tests.tool_fakes import (
    FakeToolExecutor,
    FakeToolProtocol,
    FakeToolRegistry,
    FakeToolResult,
)


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-contract",
        kb_id="kb-contract",
        text=text,
        index=0,
        metadata={
            "filename": "contract.md",
            "page": 3,
            "section": "Contract",
        },
    )


def build_retriever():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-contract",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    return GraphRetriever(builder.repository)


def test_graph_tool_satisfies_c_protocol_and_exact_result_shape():
    instance = GraphSearchTool(build_retriever(), result_factory=FakeToolResult)

    assert isinstance(instance, FakeToolProtocol)
    result = asyncio.run(
        instance.execute(query="RAG", kb_id="kb-contract", scope="local", top_k=1)
    )

    assert isinstance(result, FakeToolResult)
    assert result.to_dict().keys() == {
        "success",
        "data",
        "error",
        "duration_ms",
        "tool_name",
    }
    assert result.success is True
    assert result.tool_name == "graph_search"
    assert result.duration_ms >= 0
    assert result.data["scope"] == "local"
    assert result.data["results"][0]["entity_name"] == "RAG"
    assert result.data["results"][0]["source_refs"][0]["document_id"] == (
        "doc-contract"
    )
    assert result.data["results"][0]["source_refs"][0]["chunk_id"]
    json.dumps(result.data)


def test_c_registry_and_executor_can_register_and_call_graph_tool():
    registry = FakeToolRegistry()
    registered = register_graph_search_tool(
        registry,
        build_retriever(),
        result_factory=FakeToolResult,
    )

    result = asyncio.run(
        FakeToolExecutor(registry).execute(
            "graph_search",
            query="overall Retrieval theme",
            kb_id="kb-contract",
            scope="global",
            top_k=1,
        )
    )

    assert registry.get("graph_search") is registered
    assert registry.list_names() == ["graph_search"]
    assert result.success is True
    assert result.tool_name == "graph_search"
    assert result.duration_ms >= 0
    assert result.data["scope"] == "global"
    assert result.data["results"][0]["community_id"]
    assert result.data["results"][0]["report_id"]
    assert result.data["citations"][0]["document_id"] == "doc-contract"
    json.dumps(result.data)


def test_registry_duplicate_registration_replaces_graph_tool():
    registry = FakeToolRegistry()
    first = register_graph_search_tool(
        registry,
        build_retriever(),
        result_factory=FakeToolResult,
    )
    second = register_graph_search_tool(
        registry,
        build_retriever(),
        result_factory=FakeToolResult,
    )

    assert first is not second
    assert registry.get("graph_search") is second


def test_graph_tool_failure_result_keeps_exact_shape_for_executor():
    registry = FakeToolRegistry()
    register_graph_search_tool(
        registry,
        build_retriever(),
        result_factory=FakeToolResult,
    )

    result = asyncio.run(
        FakeToolExecutor(registry).execute(
            "graph_search",
            query="",
            kb_id="kb-contract",
            scope="local",
        )
    )

    assert result.to_dict().keys() == {
        "success",
        "data",
        "error",
        "duration_ms",
        "tool_name",
    }
    assert result.success is False
    assert result.data == {}
    assert result.tool_name == "graph_search"
    assert "query must be a non-empty string" in result.error
