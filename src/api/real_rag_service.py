"""Real RAG service — DeepSeek LLM + HuggingFace embeddings + vector retrieval.

Key design:
  - Lazy init runs in a background thread (non-blocking)
  - Simple chat questions bypass RAG and go straight to DeepSeek
  - Document questions go through full retrieval pipeline
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


# Questions that don't need document retrieval — go straight to LLM
_CHAT_PATTERNS = [
    "你是谁", "你是", "你叫", "你好", "hello", "hi", "谢谢",
    "今天", "时间", "日期", "周几", "星期", "能做什么", "功能",
    "帮我", "write", "translate", "翻译", "写一首", "写一段",
]


def _is_chat_question(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in _CHAT_PATTERNS)


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

    def _do_init(self):
        """Synchronous initialization — runs in a thread."""
        from src.ingestion import LoaderFactory
        from src.ingestion.splitter import split_documents
        from src.models.schemas import ChunkRecord
        from src.retrieval import (
            HuggingFaceEmbedder, InMemoryVectorIndex, HybridRetriever,
            SimpleReranker, ContextCompressor, CitationBuilder,
        )
        from src.rag.naive_support import build_naive_prompt
        from langchain_openai import ChatOpenAI

        project_root = Path(__file__).resolve().parents[2]
        docs_dir = project_root / "documents"
        docs_dir.mkdir(exist_ok=True)

        print("[RAG] Loading documents...", flush=True)
        supported = [".txt", ".md", ".markdown", ".pdf", ".docx"]
        files = []
        for ext in supported:
            files.extend(docs_dir.glob(f"**/*{ext}"))

        if not files:
            raise RuntimeError("documents/ is empty — please add documents first")

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
        chunks_docs = split_documents(documents, chunk_size=1000, chunk_overlap=200)
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

        self.retriever = HybridRetriever(embedder, index, alpha=0.5)
        self.reranker = SimpleReranker()
        self.compressor = ContextCompressor(max_tokens=1000)
        self.citation_builder = CitationBuilder()
        self.build_naive_prompt = build_naive_prompt

        self.llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.1,
            max_tokens=1000,
        )

        self._initialized = True
        print(f"[RAG] Initialization complete!", flush=True)

    async def _ensure_init(self):
        """Initialize in a background thread (non-blocking)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            print("[RAG] Starting initialization (in background thread)...", flush=True)
            await asyncio.to_thread(self._do_init)

    def _create_llm(self):
        """Create a fresh LLM instance for direct chat."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.7,
            max_tokens=1000,
        )

    async def query(self, request: Any) -> RAGResult:
        """Process a RAG query."""
        query_text = request.query if hasattr(request, "query") else str(request)
        mode = request.mode if hasattr(request, "mode") else RAGMode.NAIVE
        trace_id = uuid.uuid4().hex[:12]
        events: list[TraceEvent] = []

        # Simple chat questions — skip RAG, go straight to DeepSeek
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
                usage={"total_tokens": len(answer)},
                mode=mode,
            )

        # Document questions — full RAG pipeline
        print(f"[RAG] Document question: {query_text[:50]}", flush=True)
        await self._ensure_init()

        # 1. Retrieve (in thread to avoid blocking)
        t0 = time.time()
        hits = await asyncio.to_thread(self.retriever.retrieve, query_text, 10)
        events.append(TraceEvent(
            trace_id=trace_id,
            stage=TraceStage.RETRIEVE,
            duration_ms=round((time.time() - t0) * 1000, 1),
            input_summary=query_text[:100],
            output_summary=f"Retrieved {len(hits)} chunks",
        ))

        # 2. Rerank
        t0 = time.time()
        reranked = await asyncio.to_thread(self.reranker.rerank, query_text, hits, 5)
        events.append(TraceEvent(
            trace_id=trace_id,
            stage=TraceStage.RERANK,
            duration_ms=round((time.time() - t0) * 1000, 1),
            input_summary=f"Top-{len(hits)} hits",
            output_summary=f"Reranked to {len(reranked)}",
        ))

        # 3. Compress
        compressed = self.compressor.compress(reranked)

        # 4. Build context
        context = "\n\n---\n\n".join([
            f"[Chunk {j+1}] {h.chunk.text}"
            for j, h in enumerate(compressed)
        ])

        # 5. Citations
        citations_data = self.citation_builder.build_citations(compressed)

        # 6. Build API models
        api_hits = [
            RetrievalHit(
                chunk_id=h.chunk.id,
                text=h.chunk.text[:200],
                score=h.score,
                rank=h.rank,
                retriever="vector",
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

        # 7. Generate with DeepSeek (in thread)
        t0 = time.time()
        prompt = self.build_naive_prompt(query_text, context)
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        answer = response.content
        events.append(TraceEvent(
            trace_id=trace_id,
            stage=TraceStage.GENERATE,
            duration_ms=round((time.time() - t0) * 1000, 1),
            input_summary=f"Top-{len(compressed)} chunks + prompt",
            output_summary=f"Generated {len(answer)} chars",
        ))

        return RAGResult(
            answer=answer,
            citations=api_citations,
            hits=api_hits,
            trace=events,
            usage={"total_tokens": len(answer)},
            mode=mode,
        )

    async def query_stream(self, request: Any):
        result = await self.query(request)
        yield result