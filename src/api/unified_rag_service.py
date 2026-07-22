"""Unified RAG API Service — mode-ware dispatch to five RAG strategies.

Replaces the monolith `RealRAGService.query()` (which ignored `request.mode`)
with a strategy-pattern dispatch through `src.rag.service.RAGService`.

Responsibilities:
  - Lazy-init real dependencies (documents, embeddings, retriever, reranker,
    compressor, LLM, graph index)
  - Register all five strategies in the strategy registry
  - Convert API models (src.api.api_models) ↔ internal models (src.models.rag)
  - Build RAGContext with injected dependencies
  - Dispatch to the correct strategy via RAGService.run()
  - Support SSE streaming with trace, detail, chunk, done events
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from src.api.api_models import (
    Citation,
    RAGMode,
    RAGResult,
    RetrievalHit,
    TraceEvent,
    TraceStage,
)


def _load_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _format_generation_input(prompt: str, context: str) -> str:
    context = context or ""
    if context:
        return f"{prompt}\n\nKnowledge base context:\n{context}"
    return prompt


def _unpack_llm_response(raw: Any) -> tuple[str, dict[str, int]]:
    usage: Any = {}
    if isinstance(raw, tuple) and len(raw) == 2:
        raw, usage = raw
    if isinstance(raw, str):
        answer = raw
    elif isinstance(raw, dict):
        answer = raw.get("answer") or raw.get("content") or ""
        usage = raw.get("usage", usage)
    else:
        answer = getattr(raw, "content", "")
        usage = getattr(raw, "usage_metadata", usage)
        if not usage:
            response_metadata = getattr(raw, "response_metadata", {}) or {}
            usage = response_metadata.get("token_usage", {})
    if not isinstance(answer, str):
        answer = str(answer or "")
    return answer.strip(), _normalize_usage(usage)


def _normalize_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[str(key)] = value
    return normalized


def _float_option(options: Any, key: str, default: float) -> float:
    if not isinstance(options, dict):
        return default
    value = options.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_option(options: Any, key: str, default: int) -> int:
    if not isinstance(options, dict):
        return default
    value = options.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class RequestLLMAdapter:
    """Normalize LangChain chat models to the generator protocol used by RAG."""

    def __init__(self, llm: Any, metadata: dict[str, Any] | None = None) -> None:
        self._llm = llm
        self.metadata = metadata or {}

    def invoke(self, text: str) -> Any:
        method = getattr(self._llm, "invoke", None)
        if not callable(method):
            raise TypeError("chat model must expose invoke")
        return method(text)

    async def ainvoke(self, text: str) -> Any:
        method = getattr(self._llm, "ainvoke", None)
        if callable(method):
            return await method(text)
        return await asyncio.to_thread(self.invoke, text)

    async def generate(self, prompt: str, context: str = "", **kwargs: Any) -> str:
        answer, _usage = await self.generate_with_tokens(
            prompt=prompt,
            context=context,
            **kwargs,
        )
        return answer

    async def generate_with_tokens(
        self,
        prompt: str,
        context: str = "",
        **kwargs: Any,
    ) -> tuple[str, dict[str, int]]:
        raw = await self.ainvoke(_format_generation_input(prompt, context))
        answer, usage = _unpack_llm_response(raw)
        return answer, usage


class UnifiedRAGApiService:
    """Mode-aware RAG API service dispatching to strategy implementations.

    Shares document loading / embedding / index building with RealRAGService
    but replaces the monolithic pipeline with per-mode strategy dispatch.
    """

    def __init__(self):
        self._initialized = False
        self._init_lock: asyncio.Lock | None = None

        # Real dependencies (populated by _do_init)
        self.llm = None
        self.hybrid_retriever = None
        self.reranker = None
        self.compressor = None
        self.citation_builder = None
        self.query_rewriter = None

        # Graph dependencies
        self.graph_retriever = None
        self._graph_build_warnings: list[str] = []

        # Strategy-backed RAG service
        self.rag_service = None  # src.rag.service.RAGService

        # Chunks (kept for graph building)
        self._chunks = []

    # ══════════════════════════════════════════════════════════════════════
    # Initialization
    # ══════════════════════════════════════════════════════════════════════

    def _do_init(self):
        """Synchronous init: load docs, embed, index, build graph, create LLM."""
        from src.ingestion import LoaderFactory
        from src.ingestion.splitter import split_documents
        from src.models.schemas import ChunkRecord
        from src.retrieval import (
            HuggingFaceEmbedder,
            InMemoryVectorIndex,
            HybridRetriever,
            SimpleReranker,
            ContextCompressor,
            CitationBuilder,
        )

        project_root = Path(__file__).resolve().parents[2]
        docs_dir = project_root / "documents"
        docs_dir.mkdir(exist_ok=True)

        print("[UnifiedRAG] Loading documents...", flush=True)
        supported = [".txt", ".md", ".markdown", ".pdf", ".docx"]
        files = []
        for ext in supported:
            files.extend(docs_dir.glob(f"**/*{ext}"))

        if not files:
            raise RuntimeError("documents/ is empty - please add documents first")

        documents = []
        for i, fpath in enumerate(files):
            try:
                doc = LoaderFactory.load(
                    str(fpath), kb_id="default", document_id=f"doc-{i:03d}"
                )
                documents.append(doc)
                print(
                    f"[UnifiedRAG]   Loaded: {fpath.name} ({len(doc.text)} chars)",
                    flush=True,
                )
            except Exception as e:
                print(f"[UnifiedRAG]   Failed: {fpath.name}: {e}", flush=True)

        if not documents:
            raise RuntimeError("No documents could be loaded")

        print("[UnifiedRAG] Splitting text...", flush=True)
        chunks_docs = split_documents(documents, chunk_size=1500, chunk_overlap=300)
        print(f"[UnifiedRAG]   {len(chunks_docs)} chunks", flush=True)

        print("[UnifiedRAG] Loading embedding model (bge-small-zh-v1.5)...", flush=True)
        embedder = HuggingFaceEmbedder(model_name="BAAI/bge-small-zh-v1.5")

        def _doc_to_chunk(doc):
            meta = doc.metadata or {}
            return ChunkRecord(
                id=meta.get("chunk_id", ""),
                document_id=meta.get("document_id", ""),
                kb_id=meta.get("kb_id", ""),
                text=doc.page_content,
                index=meta.get("chunk_index", 0),
                metadata=meta,
            )

        chunks = [_doc_to_chunk(d) for d in chunks_docs]
        self._chunks = chunks

        print(
            f"[UnifiedRAG] Embedding {len(chunks)} chunks (this takes a while)...",
            flush=True,
        )
        index = InMemoryVectorIndex(embedder)
        index.add_chunks(chunks)
        print(f"[UnifiedRAG]   Indexed {index.count()} chunks", flush=True)

        self.hybrid_retriever = HybridRetriever(embedder, index, alpha=0.3)
        self.reranker = SimpleReranker()
        self.compressor = ContextCompressor(max_tokens=1000)
        self.citation_builder = CitationBuilder()

        # ── Build graph from chunks ────────────────────────────────────
        self._build_graph(chunks)

        self._initialized = True
        print("[UnifiedRAG] Initialization complete!", flush=True)

    def _build_graph(self, chunks: list):
        """Build graph index from document chunks for GraphRAG strategy."""
        try:
            from src.graph.builder import GraphIndexBuilder
            from src.graph.retriever import GraphRetriever

            print("[UnifiedRAG] Building knowledge graph...", flush=True)
            builder = GraphIndexBuilder()
            result = builder.build_from_chunks(kb_id="default", chunks=chunks)
            self._graph_build_warnings = list(result.warnings)

            self.graph_retriever = GraphRetriever(repository=builder.repository)

            print(
                f"[UnifiedRAG]   Graph: {result.entity_count} entities, "
                f"{result.relationship_count} relationships, "
                f"{result.community_count} communities, "
                f"{result.report_count} reports",
                flush=True,
            )
            if result.warnings:
                for w in result.warnings:
                    print(f"[UnifiedRAG]   Graph warning: {w}", flush=True)
        except Exception as exc:
            self._graph_build_warnings.append(
                f"Graph build failed: {type(exc).__name__}: {exc}"
            )
            print(
                f"[UnifiedRAG] Graph build failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self.graph_retriever = None

    async def _ensure_init(self):
        if self._initialized:
            return
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._initialized:
                return
            self.hybrid_retriever = None
            print(
                "[UnifiedRAG] Starting initialization (in background thread)...",
                flush=True,
            )
            await asyncio.to_thread(self._do_init)

    def _build_rag_service(self, *, llm: Any, query_rewriter: Any):
        """Create the strategy-backed RAGService and register all strategies."""
        from src.rag.service import RAGService
        from src.rag.registry import RAGStrategyRegistry

        registry = RAGStrategyRegistry()

        # Build RAGService with all real dependencies
        rag_service = RAGService(
            retriever=self.hybrid_retriever,
            llm=llm,
            graph=self.graph_retriever,
            registry=registry,
        )

        # Register strategies
        self._register_naive(registry)
        self._register_advanced(registry, query_rewriter)
        self._register_modular(registry)
        self._register_graph(registry)
        self._register_agentic(registry)

        self.llm = llm
        self.query_rewriter = query_rewriter
        self.rag_service = rag_service

        print(
            f"[UnifiedRAG] Registered strategies: {rag_service.list_modes()}",
            flush=True,
        )
        return rag_service

    # ══════════════════════════════════════════════════════════════════════
    # Strategy registration
    # ══════════════════════════════════════════════════════════════════════

    def _register_naive(self, registry):
        from src.rag.strategies.naive import NaiveRAGStrategy

        registry.register_or_replace(RAGMode.NAIVE, NaiveRAGStrategy())

    def _register_advanced(self, registry, query_rewriter):
        from src.rag.strategies.advanced import AdvancedRAGStrategy

        registry.register_or_replace(
            RAGMode.ADVANCED,
            AdvancedRAGStrategy(
                query_rewriter=query_rewriter,
                hybrid_retriever=self.hybrid_retriever,
                reranker=self.reranker,
                compressor=self.compressor,
            ),
        )

    def _register_modular(self, registry):
        from src.rag.strategies.modular import ModularRAGStrategy, ModuleConfig

        # Configure with all modules enabled for a full pipeline
        config = ModuleConfig(
            rewrite=True,
            retrieve=True,
            rerank=True,
            compress=True,
            verify=False,
        )
        registry.register_or_replace(
            RAGMode.MODULAR,
            ModularRAGStrategy(config=config),
        )

    def _register_graph(self, registry):
        from src.rag.strategies.graph_rag import GraphRAGStrategy

        strategy = GraphRAGStrategy(graph_retriever=self.graph_retriever)
        registry.register_or_replace(RAGMode.GRAPH, strategy)

    def _register_agentic(self, registry):
        from src.rag.strategies.agentic import AgenticRAGStrategy

        registry.register_or_replace(
            RAGMode.AGENTIC,
            AgenticRAGStrategy(),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    async def query(self, request: Any) -> RAGResult:
        """Process a RAG query by dispatching to the correct strategy.

        Args:
            request: An api_models.RAGRequest with .query, .mode, .kb_id, .options.

        Returns:
            api_models.RAGResult with answer, citations, hits, trace, usage, warnings.
        """
        mode = request.mode if hasattr(request, "mode") else RAGMode.NAIVE
        query_text = request.query if hasattr(request, "query") else str(request)
        trace_id = uuid.uuid4().hex[:12]

        llm = self._create_llm_for_request(request)
        from src.api.llm_query_rewriter import LLMQueryRewriter
        query_rewriter = LLMQueryRewriter(llm=llm)

        await self._ensure_init()
        rag_service = self._build_rag_service(llm=llm, query_rewriter=query_rewriter)

        # ── Convert API request → internal request ────────────────────
        internal_req = _to_internal_request(request)

        # ── Check if mode is available ────────────────────────────────
        if not rag_service.is_available(mode):
            return RAGResult(
                answer="",
                citations=[],
                hits=[],
                trace=[],
                usage={},
                warnings=[
                    f"Mode '{mode.value}' is not available. "
                    f"Available: {[m.value for m in rag_service.list_modes()]}"
                ],
                mode=mode,
            )

        # ── Dispatch via strategy-backed RAGService ───────────────────
        try:
            internal_result = await rag_service.run(internal_req)
        except Exception as exc:
            return RAGResult(
                answer="",
                citations=[],
                hits=[],
                trace=[
                    TraceEvent(
                        trace_id=trace_id,
                        stage=TraceStage.ERROR,
                        duration_ms=0.0,
                        input_summary=f"mode={mode.value}",
                        output_summary=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                ],
                usage={},
                warnings=[f"RAG pipeline error: {type(exc).__name__}: {exc}"],
                mode=mode,
            )

        # ── Convert internal result → API result ──────────────────────
        result = _to_api_result(internal_result, mode, trace_id)

        # ── Inject graph build warnings when using graph mode ─────────
        if mode == RAGMode.GRAPH and self._graph_build_warnings:
            for w in self._graph_build_warnings:
                if w not in result.warnings:
                    result.warnings.append(f"[graph] {w}")

        return result

    def _create_llm_for_request(self, request: Any) -> RequestLLMAdapter:
        """Create the chat model selected by the frontend for this request."""
        from src.chat.llm_client_factory import LLMClientFactory
        from src.chat.model_runtime import resolve_runtime_config

        options = getattr(request, "options", {}) or {}
        model_config = options.get("model") if isinstance(options, dict) else None
        temperature = _float_option(options, "temperature", 0.3)
        max_tokens = _int_option(options, "max_tokens", 1000)

        factory = LLMClientFactory()
        raw_llm = factory.create_chat_model(
            model_config,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,
        )

        metadata: dict[str, Any] = {}
        if model_config:
            runtime = resolve_runtime_config(model_config)
            metadata = runtime.trace_metadata()
        else:
            metadata = {
                "model_id": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "provider": "default",
                "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            }
        return RequestLLMAdapter(raw_llm, metadata=metadata)

    async def query_stream(self, request: Any):
        """SSE streaming RAG query adapter.

        Runs the full query (strategy doesn't support true streaming yet)
        then yields SSE events compatible with the existing SSE format.
        """
        result = await self.query(request)
        yield result


# ══════════════════════════════════════════════════════════════════════════
# Model conversion helpers
# ══════════════════════════════════════════════════════════════════════════


def _to_internal_request(api_req: Any):
    """Convert API RAGRequest (Pydantic) → internal RAGRequest."""
    from src.models.rag import RAGRequest

    return RAGRequest(
        query=api_req.query if hasattr(api_req, "query") else str(api_req),
        kb_id=api_req.kb_id if hasattr(api_req, "kb_id") else "default",
        mode=api_req.mode if hasattr(api_req, "mode") else RAGMode.NAIVE,
        session_id=api_req.session_id if hasattr(api_req, "session_id") else None,
        options=api_req.options if hasattr(api_req, "options") else {},
    )


def _to_api_result(internal, mode, trace_id: str = "") -> RAGResult:
    """Convert internal RAGResult → API RAGResult with model adaptation.

    Internal models:
      - citations: List[RAGCitation] (chunk_id, document_id, text_snippet)
      - hits: List[RAGChunk] (chunk_id, content, source: RAGSource)
      - trace: List[TraceEvent] (same type in both models)
      - usage: Dict[str, int]

    API models:
      - citations: List[Citation] (document_id, chunk_id, filename, page, quote, score)
      - hits: List[RetrievalHit] (chunk_id, text, score, rank, retriever, metadata)
      - trace: List[TraceEvent] (re-exported from src.models.rag)
      - usage: dict[str, Any]
    """
    # Convert citations
    api_citations = []
    for c in (internal.citations or []):
        api_citations.append(
            Citation(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                filename="unknown",
                quote=c.text_snippet or "",
                score=0.0,
            )
        )

    # Convert hits
    api_hits = []
    for i, h in enumerate(internal.hits or []):
        score = None
        meta = {}
        if hasattr(h, "source") and h.source is not None:
            score = h.source.score
            meta = dict(h.source.metadata or {})

        api_hits.append(
            RetrievalHit(
                chunk_id=h.chunk_id,
                text=(h.content or "")[:200],
                score=score if isinstance(score, (int, float)) else 0.0,
                rank=i + 1,
                retriever=meta.get("retriever", "strategy"),
                metadata=meta,
            )
        )

    # Trace events — same model (re-exported from src.models.rag in api_models)
    api_trace = list(internal.trace or [])

    # Usage — ensure it's a dict
    api_usage: dict[str, Any] = dict(internal.usage or {})

    # Build a detail dict from trace events for SSE compatibility
    api_usage["detail"] = _build_detail(internal, mode, trace_id)

    # Warnings — include any graph build warnings from the service instance
    api_warnings = list(internal.warnings or [])

    return RAGResult(
        answer=internal.answer or "",
        citations=api_citations,
        hits=api_hits,
        trace=api_trace,
        usage=api_usage,
        warnings=api_warnings,
        mode=mode,
    )


def _build_detail(internal, mode, trace_id: str) -> dict[str, Any]:
    """Build a detail dict from the internal result for SSE 'detail' event compatibility."""
    detail: dict[str, Any] = {
        "mode": mode.value if hasattr(mode, "value") else str(mode),
        "trace_id": trace_id,
    }

    # Summarize trace stages
    stages = []
    for ev in (internal.trace or []):
        stage_val = ev.stage.value if hasattr(ev.stage, "value") else str(ev.stage)
        stages.append(
            {
                "stage": stage_val,
                "duration_ms": ev.duration_ms,
                "output_summary": ev.output_summary,
            }
        )
    detail["stages"] = stages
    detail["stage_count"] = len(stages)
    detail["citation_count"] = len(internal.citations or [])
    detail["hit_count"] = len(internal.hits or [])

    return detail


# ══════════════════════════════════════════════════════════════════════════
# Singleton (for dependency injection)
# ══════════════════════════════════════════════════════════════════════════

_instance: UnifiedRAGApiService | None = None


def _get_instance() -> UnifiedRAGApiService | None:
    return _instance


def get_unified_rag_service() -> UnifiedRAGApiService:
    """Return the process-level singleton UnifiedRAGApiService."""
    global _instance
    if _instance is None:
        _instance = UnifiedRAGApiService()
    return _instance
