import asyncio

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.retrieval.vector_tool import (
    VectorSearchTool,
    register_vector_search_tool,
)
from tests.tool_fakes import (
    FakeToolExecutor,
    FakeToolProtocol,
    FakeToolRegistry,
    FakeToolResult,
)


class Retriever:
    retriever_name = "vector"

    def search(self, *, query, kb_id, top_k, filters):
        chunk = ChunkRecord(
            id="chunk-contract",
            document_id="doc-contract",
            kb_id=kb_id,
            text="contract source text",
            index=0,
            metadata={
                "kb_id": kb_id,
                "document_id": "doc-contract",
                "chunk_id": "chunk-contract",
                "filename": "contract.md",
                "page": 3,
                "section": "Contract",
            },
        )
        return [
            RetrievalHit(
                chunk=chunk,
                score=0.88,
                rank=1,
                retriever="vector",
                metadata=dict(chunk.metadata),
            )
        ]


def test_vector_tool_satisfies_c_protocol_and_exact_result_shape():
    instance = VectorSearchTool(Retriever(), result_factory=FakeToolResult)

    assert isinstance(instance, FakeToolProtocol)
    result = asyncio.run(
        instance.execute(query="contract", kb_id="kb-contract", top_k=1)
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
    assert result.tool_name == "vector_search"
    assert result.data[0]["document_id"] == "doc-contract"
    assert result.data[0]["chunk_id"] == "chunk-contract"
    assert result.data[0]["filename"] == "contract.md"
    assert result.data[0]["page"] == 3
    assert result.data[0]["text"] == "contract source text"


def test_c_registry_and_executor_can_register_and_call_vector_tool():
    registry = FakeToolRegistry()
    registered = register_vector_search_tool(
        registry,
        Retriever(),
        result_factory=FakeToolResult,
    )

    result = asyncio.run(
        FakeToolExecutor(registry).execute(
            "vector_search",
            query="contract",
            kb_id="kb-contract",
            top_k=1,
        )
    )

    assert registry.get("vector_search") is registered
    assert result.success is True
    assert result.tool_name == "vector_search"
    assert result.duration_ms >= 0
