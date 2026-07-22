"""Integration tests for unified RAG mode dispatch.

Verifies that RAGService dispatches to the correct strategy for each
of the five RAG modes (naive/advanced/modular/graph/agentic).
"""

import asyncio

from src.models.rag import (
    RAGMode,
    RAGRequest,
    StrategyType,
    TraceStage,
)
from src.rag.registry import RAGStrategyRegistry, set_registry, get_registry
from src.rag.service import RAGService
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.retriever import VectorRetriever
from src.retrieval.vector_index import InMemoryVectorIndex


# ═══════════════════════════════════════════════════════════════════════════
# Fake dependencies
# ═══════════════════════════════════════════════════════════════════════════


class OfflineGenerator:
    """Fake LLM that returns a fixed answer."""

    def __init__(self, answer: str = "RAG combines retrieval and generation."):
        self.answer = answer
        self.calls: list[dict] = []

    async def generate_with_tokens(self, prompt, context, **kwargs):
        self.calls.append({"prompt": prompt, "context": context})
        return self.answer, {"prompt": 12, "completion": 7, "total": 19}

    def invoke(self, text):
        """LangChain-compatible sync invoke."""
        from langchain_core.messages import AIMessage
        self.calls.append({"text": text})
        return AIMessage(content=self.answer)


class OfflineQueryRewriter:
    """Fake query rewriter that returns variants."""

    async def rewrite(self, query: str, max_queries: int = 3):
        return [query, f"{query} - variant 1", f"{query} - variant 2"][:max_queries]


class OfflineReranker:
    """Fake reranker that passes through hits."""

    def rerank(self, query, hits, top_k=5):
        return hits[:top_k]


class OfflineCompressor:
    """Fake compressor that passes through hits."""

    def compress(self, hits):
        return hits


class OfflineGraphRetriever:
    """Fake graph retriever for testing GraphRAGStrategy."""

    def local_search(self, query, kb_id="default", top_k=5):
        from src.graph.retriever import GraphLocalSearchResult
        return GraphLocalSearchResult(
            query=query, kb_id=kb_id, hits=[], warnings=["no graph data"]
        )

    def global_search(self, query, kb_id="default", top_k=5):
        from src.graph.retriever import GraphGlobalSearchResult
        return GraphGlobalSearchResult(
            query=query, kb_id=kb_id, hits=[], warnings=["no graph data"]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_TEST_CHUNK_TEXT = (
    "RAG combines document retrieval with grounded answer generation. "
    "It is a technique used in natural language processing."
)


def _build_test_index():
    """Build an InMemoryVectorIndex with a test chunk for deterministic retrieval."""
    from src.models.schemas import ChunkRecord

    embedder = HashEmbedder(dim=16)
    index = InMemoryVectorIndex(embedder)

    test_chunk = ChunkRecord(
        id="chunk-001",
        document_id="doc-001",
        kb_id="default",
        text=_TEST_CHUNK_TEXT,
        index=0,
        metadata={"filename": "test.txt", "kb_id": "default"},
    )
    index.add_chunks([test_chunk])
    return embedder, index


def _get_service_and_deps(*, graph=None):
    """Build a RAGService + deps and return (service, llm, test_query_text)."""
    embedder, index = _build_test_index()
    retriever = VectorRetriever(embedder, index)
    llm = OfflineGenerator()

    registry = RAGStrategyRegistry()
    set_registry(registry)

    service = RAGService(
        retriever=retriever,
        llm=llm,
        graph=graph,
        registry=registry,
    )

    _register_strategies(retriever, llm, graph)
    return service, llm, _TEST_CHUNK_TEXT


def _register_strategies(retriever, llm, graph):
    """Register all five strategies in the global registry."""
    from src.rag.strategies.naive import NaiveRAGStrategy
    from src.rag.strategies.advanced import AdvancedRAGStrategy
    from src.rag.strategies.modular import ModularRAGStrategy, ModuleConfig
    from src.rag.strategies.graph_rag import GraphRAGStrategy
    from src.rag.strategies.agentic import AgenticRAGStrategy

    registry = get_registry()

    registry.register(RAGMode.NAIVE, NaiveRAGStrategy())

    registry.register(
        RAGMode.ADVANCED,
        AdvancedRAGStrategy(
            query_rewriter=OfflineQueryRewriter(),
            hybrid_retriever=retriever,
            reranker=OfflineReranker(),
            compressor=OfflineCompressor(),
        ),
    )

    registry.register_or_replace(
        RAGMode.MODULAR,
        ModularRAGStrategy(
            config=ModuleConfig(
                rewrite=True, retrieve=True, rerank=True,
                compress=True, verify=False,
            ),
        ),
    )

    registry.register(
        RAGMode.GRAPH,
        GraphRAGStrategy(graph_retriever=graph),
    )

    registry.register_or_replace(
        RAGMode.AGENTIC,
        AgenticRAGStrategy(max_steps=2),
    )


def _make_request(mode, query=_TEST_CHUNK_TEXT, kb_id="default"):
    return RAGRequest(query=query, kb_id=kb_id, mode=mode)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Mode dispatch
# ═══════════════════════════════════════════════════════════════════════════


class TestModeDispatch:
    """Verify that RAGService dispatches to registered strategies by mode."""

    def test_all_five_modes_are_registered(self):
        service, _, _ = _get_service_and_deps(graph=OfflineGraphRetriever())
        mode_values = {m.value for m in service.list_modes()}
        assert mode_values == {"naive", "advanced", "modular", "graph", "agentic"}

    def test_naive_returns_retrieve_and_generate_trace(self):
        service, llm, query_text = _get_service_and_deps()
        request = _make_request(RAGMode.NAIVE, query=query_text)
        result = asyncio.run(service.run(request))

        assert result.answer
        assert llm.calls, "LLM should have been called"
        stages = {e.stage for e in result.trace}
        assert TraceStage.RETRIEVE in stages, "Naive should produce RETRIEVE trace"
        assert TraceStage.GENERATE in stages, "Naive should produce GENERATE trace"

    def test_advanced_returns_rewrite_rerank_compress_trace(self):
        service, llm, query_text = _get_service_and_deps()
        request = _make_request(RAGMode.ADVANCED, query=query_text)
        result = asyncio.run(service.run(request))

        assert result.answer
        stages = {e.stage for e in result.trace}
        assert TraceStage.REWRITE in stages, "Advanced should produce REWRITE trace"
        assert TraceStage.RERANK in stages, "Advanced should produce RERANK trace"
        assert TraceStage.COMPRESS in stages, "Advanced should produce COMPRESS trace"

    def test_modular_returns_trace(self):
        service, llm, query_text = _get_service_and_deps()
        request = _make_request(RAGMode.MODULAR, query=query_text)
        result = asyncio.run(service.run(request))

        assert result.answer
        assert len(result.trace) > 0, "Modular should produce trace events"

    def test_graph_returns_graph_search_trace_or_warning(self):
        graph = OfflineGraphRetriever()
        service, llm, query_text = _get_service_and_deps(graph=graph)
        request = _make_request(RAGMode.GRAPH, query=query_text)
        result = asyncio.run(service.run(request))

        stages = {e.stage for e in result.trace}
        has_graph_search = TraceStage.GRAPH_SEARCH in stages
        has_warning = any(
            "graph" in w.lower() or "no graph" in w.lower()
            for w in result.warnings
        )
        assert has_graph_search or has_warning, (
            f"Graph mode should produce GRAPH_SEARCH trace or graph warning. "
            f"stages={[e.stage.value for e in result.trace]}, warnings={result.warnings}"
        )

    def test_agentic_returns_valid_result(self):
        service, llm, query_text = _get_service_and_deps()
        request = _make_request(RAGMode.AGENTIC, query=query_text)
        result = asyncio.run(service.run(request))

        assert isinstance(result.answer, str)
        assert isinstance(result.warnings, list)
        assert isinstance(result.trace, list)


class TestModeMapping:
    """Verify RAGMode → StrategyType mapping."""

    def test_naive_maps_to_naive_strategy_type(self):
        assert RAGMode.NAIVE.to_strategy_type() == StrategyType.NAIVE

    def test_advanced_maps_to_advanced_strategy_type(self):
        assert RAGMode.ADVANCED.to_strategy_type() == StrategyType.ADVANCED

    def test_modular_maps_to_modular_strategy_type(self):
        assert RAGMode.MODULAR.to_strategy_type() == StrategyType.MODULAR

    def test_graph_maps_to_graph_rag_strategy_type(self):
        assert RAGMode.GRAPH.to_strategy_type() == StrategyType.GRAPH_RAG

    def test_agentic_maps_to_agentic_strategy_type(self):
        assert RAGMode.AGENTIC.to_strategy_type() == StrategyType.AGENTIC


class TestModelConversion:
    """Verify API ↔ internal model conversion functions."""

    def test_internal_request_conversion(self):
        from src.api.unified_rag_service import _to_internal_request
        from src.api.api_models import RAGRequest as APIRAGRequest

        api_req = APIRAGRequest(
            query="test query",
            kb_id="my-kb",
            mode=RAGMode.ADVANCED,
            session_id="sess-1",
            options={"top_k": 10},
        )
        internal = _to_internal_request(api_req)

        assert internal.query == "test query"
        assert internal.kb_id == "my-kb"
        assert internal.mode == RAGMode.ADVANCED
        assert internal.session_id == "sess-1"
        assert internal.options == {"top_k": 10}

    def test_api_result_conversion_preserves_citations(self):
        from src.api.unified_rag_service import _to_api_result
        from src.models.rag import (
            RAGResult as InternalResult,
            RAGCitation,
            RAGChunk,
            RAGSource,
        )

        internal = InternalResult(
            answer="test answer",
            citations=[
                RAGCitation(
                    chunk_id="c1", document_id="d1", text_snippet="source text"
                ),
                RAGCitation(
                    chunk_id="c2", document_id="d2", text_snippet="more text"
                ),
            ],
            hits=[
                RAGChunk(
                    chunk_id="c1",
                    content="source text content",
                    source=RAGSource(document_id="d1", score=0.95),
                ),
            ],
            trace=[],
            usage={"prompt": 10, "completion": 5, "total": 15},
            warnings=[],
        )

        api_result = _to_api_result(internal, RAGMode.NAIVE, "trace-123")

        assert api_result.answer == "test answer"
        assert len(api_result.citations) == 2
        assert api_result.citations[0].document_id == "d1"
        assert api_result.citations[0].chunk_id == "c1"
        assert api_result.citations[0].quote == "source text"
        assert len(api_result.hits) == 1
        assert api_result.hits[0].chunk_id == "c1"
        assert api_result.hits[0].score == 0.95
        assert "detail" in api_result.usage
        assert api_result.mode == RAGMode.NAIVE


class TestKnowledgeBaseIsolation:
    """Regression tests for real KB IDs reaching the unified RAG pipeline."""

    def test_document_specs_preserve_persisted_kb_id(self, tmp_path):
        from src.api.unified_rag_service import _load_document_specs

        docs_dir = tmp_path / "documents"
        data_dir = tmp_path / "data"
        docs_dir.mkdir()
        data_dir.mkdir()
        (docs_dir / "星河计划.txt").write_text("项目负责人：李明", encoding="utf-8")
        (data_dir / "knowledge_bases.json").write_text(
            (
                '{"kbs":[{"id":"kb_star","name":"星河计划"}],'
                '"docs":[{"id":"doc_star","kb_id":"kb_star","filename":"星河计划.txt"}]}'
            ),
            encoding="utf-8",
        )

        specs = _load_document_specs(tmp_path, docs_dir)

        assert len(specs) == 1
        assert specs[0]["path"] == docs_dir / "星河计划.txt"
        assert specs[0]["kb_id"] == "kb_star"
        assert specs[0]["document_id"] == "doc_star"
        assert specs[0]["filename"] == "星河计划.txt"

    def test_document_specs_prefer_storage_path_for_same_filename(self, tmp_path):
        from src.api.unified_rag_service import _load_document_specs

        docs_dir = tmp_path / "documents"
        data_dir = tmp_path / "data"
        docs_dir.mkdir()
        data_dir.mkdir()
        (docs_dir / "kb_alpha").mkdir()
        (docs_dir / "kb_beta").mkdir()
        (docs_dir / "kb_alpha" / "doc_alpha__plan.txt").write_text(
            "owner: Alice", encoding="utf-8"
        )
        (docs_dir / "kb_beta" / "doc_beta__plan.txt").write_text(
            "owner: Bob", encoding="utf-8"
        )
        (data_dir / "knowledge_bases.json").write_text(
            (
                '{"kbs":[{"id":"kb_alpha","name":"A"},{"id":"kb_beta","name":"B"}],'
                '"docs":['
                '{"id":"doc_alpha","kb_id":"kb_alpha","filename":"plan.txt",'
                '"storage_path":"kb_alpha/doc_alpha__plan.txt"},'
                '{"id":"doc_beta","kb_id":"kb_beta","filename":"plan.txt",'
                '"storage_path":"kb_beta/doc_beta__plan.txt"}]}'
            ),
            encoding="utf-8",
        )

        specs = _load_document_specs(tmp_path, docs_dir)

        by_doc = {spec["document_id"]: spec for spec in specs}
        assert by_doc["doc_alpha"]["path"] == docs_dir / "kb_alpha" / "doc_alpha__plan.txt"
        assert by_doc["doc_alpha"]["kb_id"] == "kb_alpha"
        assert by_doc["doc_alpha"]["filename"] == "plan.txt"
        assert by_doc["doc_beta"]["path"] == docs_dir / "kb_beta" / "doc_beta__plan.txt"
        assert by_doc["doc_beta"]["kb_id"] == "kb_beta"
        assert by_doc["doc_beta"]["filename"] == "plan.txt"

    def test_modular_retrieve_passes_current_kb_id_to_search(self):
        from src.models.rag import RAGContext
        from src.models.schemas import ChunkRecord, RetrievalHit
        from src.rag.strategies.modular import ModularRAGStrategy, ModuleConfig

        class SearchRetriever:
            retriever_name = "fake_search"

            def __init__(self):
                self.calls = []

            def search(self, *, query, kb_id, top_k, filters=None):
                self.calls.append(
                    {"query": query, "kb_id": kb_id, "top_k": top_k, "filters": filters}
                )
                return [
                    RetrievalHit(
                        chunk=ChunkRecord(
                            id="chunk-star",
                            document_id="doc-star",
                            kb_id=kb_id,
                            text="项目负责人：李明",
                            index=0,
                            metadata={"filename": "星河计划.txt"},
                        ),
                        score=0.99,
                        rank=1,
                        retriever="fake_search",
                    )
                ]

        class SimpleLLM:
            async def generate(self, prompt, context):
                return context

        retriever = SearchRetriever()
        context = RAGContext(
            query="星河计划负责人是谁",
            llm=SimpleLLM(),
            retriever=retriever,
            metadata={"kb_id": "kb_star"},
            config={},
        )
        request = RAGRequest(
            query="星河计划负责人是谁",
            kb_id="kb_star",
            mode=RAGMode.MODULAR,
        )
        strategy = ModularRAGStrategy(
            ModuleConfig(
                rewrite=False,
                retrieve=True,
                rerank=False,
                compress=False,
                verify=False,
                top_k=3,
            )
        )

        result = asyncio.run(strategy.run(request, context))

        assert retriever.calls == [
            {
                "query": "星河计划负责人是谁",
                "kb_id": "kb_star",
                "top_k": 3,
                "filters": None,
            }
        ]
        assert result.hits[0].content == "项目负责人：李明"
        assert result.citations[0].document_id == "doc-star"
