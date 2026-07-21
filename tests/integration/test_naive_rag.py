import asyncio

from src.ingestion.indexer import Indexer
from src.ingestion.splitter import split_document
from src.knowledge.repositories import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
)
from src.knowledge.service import KnowledgeService
from src.rag.strategies.naive import NaiveRAGStrategy
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex
from tests.rag_fakes import (
    FakeContext,
    FakeContractFactory,
    FakeRequest,
    FakeTraceStage,
)


class OfflineGenerator:
    def __init__(self):
        self.calls = []

    async def generate_with_tokens(self, prompt, context, **kwargs):
        self.calls.append({"prompt": prompt, "context": context})
        return "RAG combines retrieval and grounded generation.", {
            "prompt": 12,
            "completion": 7,
            "total": 19,
        }


def test_naive_rag_offline_end_to_end_with_real_indexer_and_vector_retriever(
    tmp_path,
):
    source_a = tmp_path / "private-notes.txt"
    source_b = tmp_path / "rag-guide.txt"
    source_a.write_text("KB-A contains unrelated private notes.", encoding="utf-8")
    source_text = "RAG combines document retrieval with grounded answer generation."
    source_b.write_text(source_text, encoding="utf-8")
    knowledge_repository = InMemoryKnowledgeRepository()
    document_repository = InMemoryDocumentRepository()
    chunk_repository = InMemoryChunkRepository()
    service = KnowledgeService(
        knowledge_repository,
        document_repository,
        chunk_repository,
    )
    service.create_knowledge_base(
        owner_id="owner",
        name="Private Notes",
        embedding_model="hash",
        kb_id="kb-a",
    )
    service.create_knowledge_base(
        owner_id="owner",
        name="RAG Guide",
        embedding_model="hash",
        kb_id="kb-b",
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
    document_a = indexer.index_document(
        "kb-a",
        source_a,
        document_id="doc-private",
    )
    document_b = indexer.index_document(
        "kb-b",
        source_b,
        document_id="doc-rag-guide",
    )
    generator = OfflineGenerator()
    context = FakeContext(
        retriever=VectorRetriever(embedder, vector_index),
        llm=generator,
        metadata={"trace_id": "trace-naive-e2e"},
    )
    request = FakeRequest(
        # HashEmbedder is intentionally non-semantic; an exact query makes this
        # integration test deterministic without weakening the real pipeline.
        query=source_text,
        kb_id="kb-b",
        options={"top_k": 3, "score_threshold": 0.0},
    )
    strategy = NaiveRAGStrategy(contract_factory=FakeContractFactory())

    result = asyncio.run(strategy.run(request, context))

    assert result.answer
    assert generator.calls
    assert all(hit.source.metadata["kb_id"] == "kb-b" for hit in result.hits)
    assert all(hit.source.document_id != document_a.id for hit in result.hits)
    assert [
        (citation.document_id, citation.chunk_id)
        for citation in result.citations
    ] == [(document_b.id, result.hits[0].chunk_id)]
    assert (
        service.get_document_chunks(document_b.id)[0].id
        == result.hits[0].chunk_id
    )
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert result.usage == {"prompt": 12, "completion": 7, "total": 19}
