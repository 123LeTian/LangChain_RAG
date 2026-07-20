import asyncio
from copy import deepcopy

from src.ingestion.indexer import Indexer
from src.ingestion.splitter import split_document
from src.knowledge.repositories import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
)
from src.knowledge.service import KnowledgeService
from src.rag.strategies.advanced import AdvancedRAGStrategy
from src.retrieval.embeddings import BaseEmbedder
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex
from tests.rag_fakes import (
    FakeContext,
    FakeContractFactory,
    FakeRAGMode,
    FakeRequest,
    FakeTraceStage,
)


class OfflineKeywordEmbedder(BaseEmbedder):
    """Small deterministic semantic fake; no model or network is involved."""

    @property
    def model_name(self):
        return "offline-keyword"

    @property
    def dimension(self):
        return 3

    def embed_texts(self, texts):
        result = []
        for text in texts:
            lowered = text.lower()
            vector = [
                1.0 if "rag" in lowered else 0.0,
                1.0
                if any(word in lowered for word in ("retrieval", "grounded"))
                else 0.0,
                1.0 if "private" in lowered else 0.0,
            ]
            if not any(vector):
                vector = [0.1, 0.1, 0.1]
            result.append(vector)
        return result


class OfflineRewriter:
    def __init__(self):
        self.calls = []

    async def rewrite(self, query, *, max_queries):
        self.calls.append((query, max_queries))
        return ["retrieval grounded generation"]


class ReverseReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, hits, top_k=5):
        self.calls.append([hit.chunk.id for hit in hits])
        return list(reversed(deepcopy(hits)))[:top_k]


class OneChunkCompressor:
    def __init__(self):
        self.calls = []

    def compress(self, hits):
        self.calls.append([hit.chunk.id for hit in hits])
        result = deepcopy(hits[:1])
        if result:
            result[0].chunk.text = result[0].chunk.text[:18]
        return result


class OfflineGenerator:
    def __init__(self):
        self.calls = []

    async def generate_with_tokens(self, prompt, context, **kwargs):
        self.calls.append({"prompt": prompt, "context": context})
        return "offline advanced answer", {
            "prompt": 30,
            "completion": 6,
            "total": 36,
        }


def _strategy(options, *, rewriter=None, reranker=None, compressor=None):
    return AdvancedRAGStrategy(
        config=options,
        query_rewriter=rewriter,
        reranker=reranker,
        compressor=compressor,
        contract_factory=FakeContractFactory(),
    )


def test_advanced_rag_offline_end_to_end_and_minimal_ablation(tmp_path):
    private_path = tmp_path / "private.txt"
    overview_path = tmp_path / "overview.txt"
    guide_path = tmp_path / "guide.txt"
    private_path.write_text("Private KB-A material.", encoding="utf-8")
    overview_path.write_text("RAG introduction and overview.", encoding="utf-8")
    guide_path.write_text(
        "RAG combines retrieval with grounded generation.",
        encoding="utf-8",
    )

    knowledge_repository = InMemoryKnowledgeRepository()
    document_repository = InMemoryDocumentRepository()
    chunk_repository = InMemoryChunkRepository()
    service = KnowledgeService(
        knowledge_repository,
        document_repository,
        chunk_repository,
    )
    for kb_id, name in (("kb-a", "Private"), ("kb-b", "RAG")):
        service.create_knowledge_base(
            owner_id="owner",
            name=name,
            embedding_model="offline-keyword",
            kb_id=kb_id,
        )

    embedder = OfflineKeywordEmbedder()
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
    private_document = indexer.index_document(
        "kb-a", private_path, document_id="doc-private"
    )
    overview_document = indexer.index_document(
        "kb-b", overview_path, document_id="doc-overview"
    )
    guide_document = indexer.index_document(
        "kb-b", guide_path, document_id="doc-guide"
    )
    retriever = VectorRetriever(embedder, vector_index)
    base_request = {
        "query": "What is RAG?",
        "kb_id": "kb-b",
        "mode": FakeRAGMode.ADVANCED,
    }

    config_a = {
        "top_k": 2,
        "rewrite_enabled": False,
        "multi_query_enabled": False,
        "hybrid_enabled": False,
        "rerank_enabled": False,
        "compression_enabled": False,
    }
    generator_a = OfflineGenerator()
    result_a = asyncio.run(
        _strategy(config_a).run(
            FakeRequest(**base_request),
            FakeContext(retriever, generator_a),
        )
    )

    config_b = {
        **config_a,
        "rewrite_enabled": True,
    }
    rewriter_b = OfflineRewriter()
    generator_b = OfflineGenerator()
    result_b = asyncio.run(
        _strategy(config_b, rewriter=rewriter_b).run(
            FakeRequest(**base_request),
            FakeContext(retriever, generator_b),
        )
    )

    config_c = {
        **config_b,
        "rerank_enabled": True,
        "compression_enabled": True,
    }
    rewriter_c = OfflineRewriter()
    reranker_c = ReverseReranker()
    compressor_c = OneChunkCompressor()
    generator_c = OfflineGenerator()
    result_c = asyncio.run(
        _strategy(
            config_c,
            rewriter=rewriter_c,
            reranker=reranker_c,
            compressor=compressor_c,
        ).run(
            FakeRequest(**base_request),
            FakeContext(retriever, generator_c),
        )
    )

    private_chunk_id = service.get_document_chunks(private_document.id)[0].id
    overview_chunk_id = service.get_document_chunks(overview_document.id)[0].id
    guide_chunk_id = service.get_document_chunks(guide_document.id)[0].id
    assert [hit.chunk_id for hit in result_a.hits] == [
        overview_chunk_id,
        guide_chunk_id,
    ]
    assert [hit.chunk_id for hit in result_b.hits] == [
        guide_chunk_id,
        overview_chunk_id,
    ]
    assert [hit.chunk_id for hit in result_c.hits] == [overview_chunk_id]
    assert all(hit.chunk_id != private_chunk_id for hit in result_a.hits)
    assert all(hit.chunk_id != private_chunk_id for hit in result_b.hits)
    assert all(hit.chunk_id != private_chunk_id for hit in result_c.hits)
    assert [
        (citation.document_id, citation.chunk_id)
        for citation in result_c.citations
    ] == [(overview_document.id, overview_chunk_id)]
    assert len(generator_c.calls[0]["context"]) < len(generator_b.calls[0]["context"])
    assert [event.stage for event in result_a.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert [event.stage for event in result_b.trace] == [
        FakeTraceStage.REWRITE,
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert [event.stage for event in result_c.trace] == [
        FakeTraceStage.REWRITE,
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.RERANK,
        FakeTraceStage.COMPRESS,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert result_c.usage == {"prompt": 30, "completion": 6, "total": 36}
