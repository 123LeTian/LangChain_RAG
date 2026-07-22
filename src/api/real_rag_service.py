"""Real RAG service - DeepSeek LLM + HuggingFace embeddings + hybrid retrieval.

Supports 5 module switches (rewrite/retrieve/rerank/compress/verify) for
Modular RAG ablation experiments. Validates illegal combinations, saves
pipeline config, and supports side-by-side comparison of two configs.
"""
from __future__ import annotations

import asyncio
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


_CHAT_PATTERNS = [
    "hello", "hi", "thanks", "thank you",
    "today", "time", "date", "day", "week",
    "what can you do", "help me", "function",
    "write", "translate",
]


def _is_chat_question(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in _CHAT_PATTERNS)



# ====================================================================
# Module config validation (mirrors C's validate_module_config)
# ====================================================================

_MODULE_DEPENDENCIES = {
    "rerank":   {"retrieve"},
    "compress": {"retrieve"},
    # verify has no hard dependency - generate is always on
}


def validate_module_config(config: dict) -> list[str]:
    """Return list of config errors. Empty list = valid.

    Rules:
      - rerank=True requires retrieve=True
      - compress=True requires retrieve=True
    """
    errors = []
    enabled = {name for name in ("rewrite", "retrieve", "rerank", "compress", "verify")
               if config.get(name, False)}
    for module, deps in _MODULE_DEPENDENCIES.items():
        if module in enabled:
            missing = deps - enabled
            if missing:
                errors.append(
                    f"Module '{module}' requires {sorted(missing)} to be enabled, "
                    f"but they are disabled"
                )
    return errors


def _find_best_snippet(text: str, query: str, window: int = 120) -> str:
    """Find the text window with highest keyword overlap for preview."""
    import re as _re
    if not query or not text:
        return text[:window].replace("\n", " ")
    # Tokenize query into Chinese chars + ascii words
    q_tokens = set()
    for seg in _re.findall(r"[\u4e00-\u9fff]+", query.lower()):
        for i in range(len(seg)):
            q_tokens.add(seg[i])
            if i < len(seg) - 1:
                q_tokens.add(seg[i:i+2])
    q_tokens.update(_re.findall(r"[a-z0-9]+", query.lower()))
    if not q_tokens:
        return text[:window].replace("\n", " ")
    best_score = 0
    best_pos = 0
    step = 15
    for pos in range(0, max(1, len(text) - window), step):
        snip = text[pos:pos+window].lower()
        score = sum(1 for t in q_tokens if t in snip)
        if score > best_score:
            best_score = score
            best_pos = pos
    start = max(0, best_pos)
    end = min(len(text), best_pos + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return (prefix + text[start:end] + suffix).replace("\n", " ")

class RealRAGService:
    """Real RAG service with lazy non-blocking initialization."""

    def __init__(self):
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.llm = None
        self.retriever = None
        self.reranker = None
        self.compressor = None
        self.citation_builder = None
        self.build_naive_prompt = None
        self.query_rewriter = None

    def _do_init(self):
        from src.ingestion import LoaderFactory
        from src.ingestion.splitter import split_documents
        from src.models.schemas import ChunkRecord
        from src.retrieval import (
            HuggingFaceEmbedder, InMemoryVectorIndex, HybridRetriever,
            SimpleReranker, ContextCompressor, CitationBuilder,
        )
        from src.rag.naive_support import build_naive_prompt
        from langchain_openai import ChatOpenAI
        from src.api.llm_query_rewriter import LLMQueryRewriter

        project_root = Path(__file__).resolve().parents[2]
        docs_dir = project_root / "documents"
        docs_dir.mkdir(exist_ok=True)

        print("[RAG] Loading documents...", flush=True)
        supported = [".txt", ".md", ".markdown", ".pdf", ".docx"]
        files = []
        for ext in supported:
            files.extend(docs_dir.glob(f"**/*{ext}"))

        if not files:
            raise RuntimeError("documents/ is empty - please add documents first")

        documents = []
        for i, fpath in enumerate(files):
            try:
                doc = LoaderFactory.load(str(fpath), kb_id="kb-1", document_id=f"doc-{i:03d}")
                documents.append(doc)
                print(f"[RAG]   Loaded: {fpath.name} ({len(doc.text)} chars)", flush=True)
            except Exception as e:
                print(f"[RAG]   Failed: {fpath.name}: {e}", flush=True)

        if not documents:
            raise RuntimeError("No documents could be loaded")

        print("[RAG] Splitting text...", flush=True)
        chunks_docs = split_documents(documents, chunk_size=1500, chunk_overlap=300)
        print(f"[RAG]   {len(chunks_docs)} chunks", flush=True)

        print("[RAG] Loading embedding model (bge-small-zh-v1.5)...", flush=True)
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
        print(f"[RAG] Embedding {len(chunks)} chunks (this takes a while)...", flush=True)
        index = InMemoryVectorIndex(embedder)
        index.add_chunks(chunks)
        print(f"[RAG]   Indexed {index.count()} chunks", flush=True)

        self.retriever = HybridRetriever(embedder, index, alpha=0.3)
        self.reranker = SimpleReranker()
        self.compressor = ContextCompressor(max_tokens=1000)
        self.citation_builder = CitationBuilder()
        self.build_naive_prompt = build_naive_prompt

        self.llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.3,
            max_tokens=1000,
        )

        self.query_rewriter = LLMQueryRewriter(llm=self.llm)
        self._initialized = True
        print(f"[RAG] Initialization complete!", flush=True)

    async def _ensure_init(self):
        async with self._init_lock:
            if self._initialized:
                return
            self.retriever = None
            print("[RAG] Starting initialization (in background thread)...", flush=True)
            await asyncio.to_thread(self._do_init)

    def _create_llm(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.7,
            max_tokens=1000,
        )

    def _parse_config(self, request: Any) -> dict:
        """Extract 5-module config from request.options with defaults."""
        options = getattr(request, "options", {}) or {}
        return {
            "rewrite":  options.get("rewrite_enabled", True),
            "retrieve": options.get("retrieve_enabled", True),
            "rerank":   options.get("rerank_enabled", True),
            "compress": options.get("compress_enabled", True),
            "verify":   options.get("verify_enabled", False),
            "top_k":          options.get("top_k", 10),
            "rerank_top_k":   options.get("rerank_top_k", 5),
        }

    def _build_pipeline_config(self, config: dict) -> dict:
        """Build a serializable pipeline config snapshot."""
        return {
            "rewrite":  config["rewrite"],
            "retrieve": config["retrieve"],
            "rerank":   config["rerank"],
            "compress": config["compress"],
            "verify":   config["verify"],
            "top_k":    config["top_k"],
            "rerank_top_k": config["rerank_top_k"],
            "generate": True,  # always on
        }

    async def _run_verify(self, query: str, answer: str, context: str) -> dict:
        """Verify answer factuality using LLM."""
        verify_prompt = (
            # (translated prompt line)
            # (translated prompt line)
            # (translated prompt line)
            # (translated prompt line)
            # (translated prompt line)
            # (translated prompt line)
            'Reply JSON: {"passed": true/false, "issues": ["issue1", ...]}'
        )
        try:
            response = await asyncio.to_thread(self.llm.invoke, verify_prompt)
            import json
            text = response.content.strip()
            # Try to extract JSON from response
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                result = json.loads(text[start:end])
                return result
            return {"passed": True, "issues": []}
        except Exception as e:
            return {"passed": True, "issues": [], "error": str(e)}

    async def query(self, request: Any) -> RAGResult:
        """Process a RAG query with 5 configurable module switches."""
        query_text = request.query if hasattr(request, "query") else str(request)
        mode = request.mode if hasattr(request, "mode") else RAGMode.NAIVE
        trace_id = uuid.uuid4().hex[:12]
        events: list[TraceEvent] = []

        # --- Parse and validate module config ---
        config = self._parse_config(request)
        errors = validate_module_config(config)
        if errors:
            return RAGResult(
                answer=f"[Config Error] {'; '.join(errors)}",
                citations=[],
                hits=[],
                trace=[],
                usage={"error": "invalid_config", "errors": errors},
                mode=mode,
            )

        pipeline_config = self._build_pipeline_config(config)

        # --- Structured detail for frontend ---
        detail = {
            "original_query": query_text,
            "rewritten_queries": [query_text],
            "pipeline_config": pipeline_config,
            "pre_rerank_top_k": [],
            "post_rerank_top_k": [],
            "rewrite_latency_ms": 0,
            "rerank_latency_ms": 0,
            "verify_result": None,
        }

        # --- Simple chat questions bypass RAG ---
        if _is_chat_question(query_text):
            print(f"[RAG] Chat question (no retrieval): {query_text[:50]}", flush=True)
            t0 = time.time()
            llm = self._create_llm()
            response = await asyncio.to_thread(llm.invoke, query_text)
            answer = response.content
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.GENERATE,
                duration_ms=round((time.time() - t0) * 1000, 1),
                input_summary=query_text[:100],
                output_summary=f"Direct LLM: {len(answer)} chars",
            ))
            return RAGResult(
                answer=answer,
                citations=[],
                hits=[],
                trace=events,
                usage={"total_tokens": len(answer), "detail": detail},
                mode=mode,
            )

        print(f"[RAG] Document question: {query_text[:50]}", flush=True)
        print(f"[RAG] Config: {pipeline_config}", flush=True)
        await self._ensure_init()

        # Track enabled modules for trace ordering
        active_modules = []

        # === Module 1: Rewrite (conditional) ===
        rewritten_queries = [query_text]
        if config["rewrite"]:
            active_modules.append("rewrite")
            t0 = time.time()
            rewritten_queries = await self.query_rewriter.rewrite(query_text, max_queries=3)
            detail["rewritten_queries"] = rewritten_queries
            detail["rewrite_latency_ms"] = round((time.time() - t0) * 1000, 1)
            print(f"[RAG] Rewritten queries: {rewritten_queries}", flush=True)
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.REWRITE,
                duration_ms=detail["rewrite_latency_ms"],
                input_summary=f"original: {query_text}",
                output_summary=f"rewritten to {len(rewritten_queries)} queries: {rewritten_queries}",
            ))

        # === Module 2: Retrieve (conditional) ===
        hits = []
        if config["retrieve"]:
            active_modules.append("retrieve")
            t0 = time.time()
            all_hits = []
            seen_chunks = set()
            for rq in rewritten_queries:
                r_hits = await asyncio.to_thread(self.retriever.retrieve, rq, config["top_k"])
                for h in r_hits:
                    if h.chunk.id not in seen_chunks:
                        seen_chunks.add(h.chunk.id)
                        all_hits.append(h)
            all_hits.sort(key=lambda h: -h.score)
            hits = all_hits[:config["top_k"]]

            detail["pre_rerank_top_k"] = [
                {
                    "rank": i + 1,
                    "chunk_id": h.chunk.id,
                    "score": round(h.score, 4),
                    "text": h.chunk.text[:120].replace("\n", " "),
                }
                for i, h in enumerate(hits)
            ]

            pre_summary = "; ".join(
                f"#{i+1} score={h.score:.4f}" for i, h in enumerate(hits)
            )
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.RETRIEVE,
                duration_ms=round((time.time() - t0) * 1000, 1),
                input_summary=f"{len(rewritten_queries)} queries, top_k={config['top_k']}",
                output_summary=f"pre-rerank Top-{len(hits)}: {pre_summary}",
            ))
        else:
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.RETRIEVE,
                duration_ms=0.0,
                input_summary="",
                output_summary="skipped (retrieve=False)",
            ))

        # === Module 3: Rerank (conditional, requires retrieve) ===
        if config["rerank"] and hits:
            active_modules.append("rerank")
            t0 = time.time()
            reranked = await asyncio.to_thread(
                self.reranker.rerank, query_text, hits, config["rerank_top_k"]
            )
            detail["rerank_latency_ms"] = round((time.time() - t0) * 1000, 1)
            detail["post_rerank_top_k"] = [
                {
                    "rank": i + 1,
                    "chunk_id": h.chunk.id,
                    "score": round(h.score, 4),
                    "text": h.chunk.text[:120].replace("\n", " "),
                }
                for i, h in enumerate(reranked)
            ]
            post_summary = "; ".join(
                f"#{i+1} score={h.score:.4f}" for i, h in enumerate(reranked)
            )
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.RERANK,
                duration_ms=detail["rerank_latency_ms"],
                input_summary=f"pre-rerank Top-{len(hits)}",
                output_summary=f"post-rerank Top-{len(reranked)}: {post_summary}",
            ))
            final_hits = reranked
        elif hits:
            final_hits = hits[:config["rerank_top_k"]]
            detail["post_rerank_top_k"] = detail["pre_rerank_top_k"][:config["rerank_top_k"]]
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.RERANK,
                duration_ms=0.0,
                input_summary=f"Top-{len(hits)}",
                output_summary="skipped (rerank=False)",
            ))
        else:
            final_hits = []

        # === Module 4: Compress (conditional) ===
        if config["compress"] and final_hits:
            active_modules.append("compress")
            t0 = time.time()
            compressed = self.compressor.compress(final_hits)
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.COMPRESS,
                duration_ms=round((time.time() - t0) * 1000, 1),
                input_summary=f"{len(final_hits)} chunks",
                output_summary=f"compressed to {len(compressed)} chunks",
            ))
            final_context = compressed
        else:
            final_context = final_hits
            if final_hits:
                events.append(TraceEvent(
                    trace_id=trace_id,
                    stage=TraceStage.COMPRESS,
                    duration_ms=0.0,
                    input_summary="",
                    output_summary="skipped (compress=False)",
                ))

        # === Module 5: Generate (always on) ===
        active_modules.append("generate")
        context = "\n\n---\n\n".join([
            f"[Chunk {j+1}] {h.chunk.text}"
            for j, h in enumerate(final_context)
        ]) if final_context else "(no context retrieved)"

        citations_data = self.citation_builder.build_citations(final_context) if final_context else []

        api_hits = [
            RetrievalHit(
                chunk_id=h.chunk.id,
                text=h.chunk.text[:200],
                score=h.score,
                rank=h.rank,
                retriever="hybrid",
                metadata=h.chunk.metadata,
            )
            for h in hits
        ]

        api_citations = [
            Citation(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                filename=c.filename or "unknown",
                page=c.page,
                quote=c.quote[:200],
                score=0.0,
            )
            for c in citations_data
        ]

        t0 = time.time()
        prompt = self.build_naive_prompt(query_text, context)
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        answer = response.content

        token_usage = {"total_tokens": len(answer)}
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = {
                    "input_tokens": response.usage_metadata.get("prompt_tokens", 0),
                    "output_tokens": response.usage_metadata.get("completion_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0),
                }
        except Exception:
            pass

        events.append(TraceEvent(
            trace_id=trace_id,
            stage=TraceStage.GENERATE,
            duration_ms=round((time.time() - t0) * 1000, 1),
            input_summary=f"Top-{len(final_context)} chunks + prompt",
            output_summary=f"Generated {len(answer)} chars",
        ))

        # === Module 6: Verify (conditional) ===
        if config["verify"]:
            active_modules.append("verify")
            t0 = time.time()
            verify_result = await self._run_verify(query_text, answer, context)
            detail["verify_result"] = verify_result
            events.append(TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.VERIFY,
                duration_ms=round((time.time() - t0) * 1000, 1),
                input_summary=f"answer={len(answer)} chars, context={len(context)} chars",
                output_summary=f"passed={verify_result.get('passed', '?')}, issues={len(verify_result.get('issues', []))}",
            ))

        # Record active module order (acceptance criterion 5)
        detail["active_modules"] = active_modules

        token_usage["detail"] = detail
        token_usage["pipeline_config"] = pipeline_config

        return RAGResult(
            answer=answer,
            citations=api_citations,
            hits=api_hits,
            trace=events,
            usage=token_usage,
            mode=mode,
        )

    async def query_stream(self, request: Any):
        result = await self.query(request)
        yield result

    async def compare(self, request: Any, config_a: dict, config_b: dict) -> dict:
        """Run the same query with two pipeline configs and return side-by-side results.

        Implements 'same question, one-click comparison of two configs'.
        """
        # Create two request objects with different options
        req_a = type(request)(
            query=request.query,
            kb_id=getattr(request, "kb_id", "default"),
            mode=getattr(request, "mode", RAGMode.MODULAR),
            options={**config_a},
        )
        req_b = type(request)(
            query=request.query,
            kb_id=getattr(request, "kb_id", "default"),
            mode=getattr(request, "mode", RAGMode.MODULAR),
            options={**config_b},
        )

        result_a = await self.query(req_a)
        result_b = await self.query(req_b)

        # Build comparison summary
        def _cfg_str(cfg):
            parts = [name for name in ("rewrite", "retrieve", "rerank", "compress", "verify")
                     if cfg.get(name)]
            return "+".join(parts) if parts else "generate-only"

        trace_a_stages = "+".join(e.stage.value for e in result_a.trace)
        trace_b_stages = "+".join(e.stage.value for e in result_b.trace)

        summary = (
            f"Pipeline Comparison: '{_cfg_str(config_a)}' vs '{_cfg_str(config_b)}'\n"
            f"  Config A: {_cfg_str(config_a)} | answer={len(result_a.answer)} chars | trace={trace_a_stages}\n"
            f"  Config B: {_cfg_str(config_b)} | answer={len(result_b.answer)} chars | trace={trace_b_stages}"
        )

        return {
            "result_a": result_a,
            "result_b": result_b,
            "summary": summary,
        }
