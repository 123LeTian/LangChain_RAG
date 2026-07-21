import asyncio

from src.ingestion.indexer import Indexer
from src.ingestion.splitter import split_document
from src.knowledge.repositories import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
)
from src.knowledge.service import KnowledgeService
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex
from src.retrieval.vector_tool import register_vector_search_tool
from tests.tool_fakes import FakeToolExecutor, FakeToolRegistry, FakeToolResult


def test_vector_tool_offline_executor_integration_preserves_kb_isolation(tmp_path):
    kb_a_paths = []
    for index, text in enumerate(
        (
            "global high score alpha",
            "global high score beta",
            "global high score gamma",
        )
    ):
        path = tmp_path / f"kb-a-{index}.txt"
        path.write_text(text, encoding="utf-8")
        kb_a_paths.append(path)
    kb_b_path = tmp_path / "kb-b-target.txt"
    kb_b_path.write_text(
        "isolated lower target evidence",
        encoding="utf-8",
    )

    service = KnowledgeService(
        InMemoryKnowledgeRepository(),
        InMemoryDocumentRepository(),
        InMemoryChunkRepository(),
    )
    for kb_id in ("kb-a", "kb-b"):
        service.create_knowledge_base(
            owner_id="owner",
            name=kb_id,
            embedding_model="hash",
            kb_id=kb_id,
        )

    embedder = HashEmbedder(dim=16)
    vector_index = InMemoryVectorIndex(embedder)
    indexer = Indexer(
        knowledge_service=service,
        splitter=lambda document: split_document(
            document,
            chunk_size=200,
            chunk_overlap=0,
        ),
        embedder=embedder,
        vector_index=vector_index,
    )
    kb_a_documents = [
        indexer.index_document(
            "kb-a",
            path,
            document_id=f"doc-a-{index}",
        )
        for index, path in enumerate(kb_a_paths)
    ]
    kb_b_document = indexer.index_document(
        "kb-b",
        kb_b_path,
        document_id="doc-b-target",
    )

    registry = FakeToolRegistry()
    register_vector_search_tool(
        registry,
        VectorRetriever(embedder, vector_index),
        result_factory=FakeToolResult,
    )
    result = asyncio.run(
        FakeToolExecutor(registry).execute(
            "vector_search",
            query="global high score alpha",
            kb_id="kb-b",
            top_k=2,
        )
    )

    kb_a_document_ids = {document.id for document in kb_a_documents}
    expected_chunk = service.get_document_chunks(kb_b_document.id)[0]
    assert result.success is True
    assert result.tool_name == "vector_search"
    assert result.result_count == 1
    assert [item["kb_id"] for item in result.data] == ["kb-b"]
    assert [item["document_id"] for item in result.data] == [kb_b_document.id]
    assert all(item["document_id"] not in kb_a_document_ids for item in result.data)
    assert result.data[0]["chunk_id"] == expected_chunk.id
    assert result.data[0]["text"] == expected_chunk.text
    assert result.data[0]["filename"] == kb_b_document.filename
    assert result.data[0]["page"] == expected_chunk.metadata["page"]
    assert result.data[0]["rank"] == 1
    assert result.data[0]["retriever"] == "vector"
    assert result.duration_ms >= 0
