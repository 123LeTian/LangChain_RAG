"""
RAG Model — Owner: C (主责), Shared
共享 RAG 契约：定义了 RAG 管道的统一接口和数据模型。
所有 RAG 策略必须实现此契约。

Unified data contract supporting all five RAG paradigms:
  - Naive RAG: simple retrieve → generate
  - Advanced RAG: rerank → rewrite → compress → generate
  - Modular RAG: composable pipeline from swappable modules
  - Graph RAG: knowledge-graph-enhanced retrieval
  - Agentic RAG: multi-step agent-driven reasoning
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator


# ── Strategy Enum ──────────────────────────────────────────────────────────


class StrategyType(str, Enum):
    """Enumerates the five RAG paradigms available through the unified interface."""

    NAIVE = "naive"
    ADVANCED = "advanced"
    MODULAR = "modular"
    GRAPH_RAG = "graph_rag"
    AGENTIC = "agentic"


# ── Status Enum ────────────────────────────────────────────────────────────


class RAGStatus(str, Enum):
    """Pipeline execution status for RAGResponse."""

    SUCCESS = "success"
    PARTIAL = "partial"   # Completed but through fallback
    FAILURE = "failure"   # All strategies exhausted


# ── Source & Chunk ─────────────────────────────────────────────────────────


class RAGSource(BaseModel):
    """Metadata for a retrieved document / chunk source."""

    document_id: str = Field(..., description="Unique document identifier")
    chunk_index: int = Field(default=0, description="Index of the chunk within the document")
    source_path: Optional[str] = Field(default=None, description="Original file path or URL")
    title: Optional[str] = Field(default=None, description="Document title")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    score: Optional[float] = Field(default=None, description="Retrieval relevance score [0, 1]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class RAGChunk(BaseModel):
    """A single retrieved text chunk with its source metadata."""

    chunk_id: str = Field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}",
        description="Unique chunk identifier",
    )
    content: str = Field(..., description="The text content of this chunk")
    source: RAGSource = Field(..., description="Source metadata for this chunk")
    embedding: Optional[List[float]] = Field(
        default=None, repr=False, description="Optional embedding vector (carried for reranking)"
    )


# ── Context ────────────────────────────────────────────────────────────────


class RAGContext(BaseModel):
    """Collection of retrieved chunks forming the augmentation context."""

    chunks: List[RAGChunk] = Field(default_factory=list, description="Retrieved chunks")
    query: str = Field(..., description="The query that produced this context")
    retrieval_method: str = Field(
        default="vector", description="Method used: vector, hybrid, graph, agent"
    )
    retrieval_latency_ms: Optional[float] = Field(
        default=None, description="Retrieval wall-clock time in milliseconds"
    )
    total_candidates: int = Field(default=0, description="Candidates before filtering/reranking")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        """Return all chunk contents concatenated with separators."""
        return "\n\n---\n\n".join(chunk.content for chunk in self.chunks)

    @property
    def source_count(self) -> int:
        """Number of unique source documents."""
        return len({chunk.source.document_id for chunk in self.chunks})


# ── Query ──────────────────────────────────────────────────────────────────


class RAGQuery(BaseModel):
    """Unified query object accepted by every RAG strategy."""

    query_id: str = Field(
        default_factory=lambda: f"q_{uuid.uuid4().hex[:12]}",
        description="Unique query identifier",
    )
    text: str = Field(..., min_length=1, description="Natural-language query text")
    strategy: StrategyType = Field(
        default=StrategyType.NAIVE, description="Which RAG strategy to use"
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters for retrieval"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Multi-turn conversation identifier"
    )
    user_id: Optional[str] = Field(default=None, description="End-user identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query text must not be blank")
        return stripped


# ── Response ───────────────────────────────────────────────────────────────


class RAGCitation(BaseModel):
    """A citation linking generated text back to a source chunk."""

    chunk_id: str = Field(..., description="Referenced chunk ID")
    document_id: str = Field(..., description="Source document ID")
    text_snippet: str = Field(default="", description="Snippet of source text being cited")


class RAGResponse(BaseModel):
    """Unified response object returned by every RAG strategy."""

    response_id: str = Field(
        default_factory=lambda: f"r_{uuid.uuid4().hex[:12]}",
        description="Unique response identifier",
    )
    query_id: str = Field(..., description="Corresponding query identifier")
    answer: str = Field(..., description="Generated answer text")
    strategy: StrategyType = Field(..., description="Strategy that produced this response")
    context: Optional[RAGContext] = Field(
        default=None, description="Retrieval context used for generation"
    )
    citations: List[RAGCitation] = Field(
        default_factory=list, description="Citations linking answer to sources"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Model self-assessed confidence"
    )
    latency_ms: Optional[float] = Field(
        default=None, description="End-to-end wall-clock time in milliseconds"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict, description="LLM token usage: prompt, completion, total"
    )
    trace_id: Optional[str] = Field(
        default=None, description="Trace identifier for observability"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the pipeline failed gracefully"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Streaming Events ──────────────────────────────────────────────────────


class StreamEventType(str, Enum):
    """Stream event type constants for the RAG pipeline."""

    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_END = "retrieval_end"
    RERANK = "rerank"
    GENERATION_START = "generation_start"
    TOKEN = "token"
    SOURCE = "source"
    DONE = "done"
    ERROR = "error"
    AGENT_STEP = "agent_step"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"


class StreamEvent(BaseModel):
    """A single streaming event from the RAG pipeline."""

    event_type: StreamEventType = Field(..., description="Type of the streaming event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    trace_id: Optional[str] = Field(default=None, description="Associated trace identifier")
    timestamp: float = Field(default_factory=time.time)


# ── Pipeline Config ───────────────────────────────────────────────────────


class RAGPipelineConfig(BaseModel):
    """Configuration shared across strategies for the RAG pipeline."""

    strategy: StrategyType = Field(default=StrategyType.NAIVE)
    top_k: int = Field(default=5, ge=1, le=100)
    llm_model: str = Field(default="claude-sonnet-5", description="LLM model identifier")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=8000, ge=1, description="Max tokens for context window")
    rerank_enabled: bool = Field(default=False)
    query_rewrite_enabled: bool = Field(default=False)
    fallback_strategies: List[StrategyType] = Field(
        default_factory=list,
        description="Ordered fallback chain on failure, e.g. [graph_rag, advanced, naive]",
    )
    agent_max_steps: int = Field(default=5, ge=1, le=20, description="Max agent reasoning steps")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# C1 — 共享 RAG 数据契约 (Unified Data Contract)
# ═══════════════════════════════════════════════════════════════════════════


# ── RAGMode ────────────────────────────────────────────────────────────────


class RAGMode(str, Enum):
    """All five RAG paradigms available through the unified orchestration layer.

    This enum complements StrategyType; it uses shorter value names and is the
    preferred mode selector for the API-facing RAGRequest model.
    """

    NAIVE = "naive"
    ADVANCED = "advanced"
    MODULAR = "modular"
    GRAPH = "graph"
    AGENTIC = "agentic"

    def to_strategy_type(self) -> StrategyType:
        """Map RAGMode → StrategyType for internal strategy resolution."""
        _map = {
            RAGMode.NAIVE: StrategyType.NAIVE,
            RAGMode.ADVANCED: StrategyType.ADVANCED,
            RAGMode.MODULAR: StrategyType.MODULAR,
            RAGMode.GRAPH: StrategyType.GRAPH_RAG,
            RAGMode.AGENTIC: StrategyType.AGENTIC,
        }
        return _map[self]


# ── TraceStage ─────────────────────────────────────────────────────────────


class TraceStage(str, Enum):
    """Well-defined stages in a RAG pipeline trace.

    Each stage corresponds to a distinct processing step:
      INTENT       — query intent analysis / routing decision
      REWRITE      — query rewriting / expansion
      RETRIEVE      — vector / hybrid / keyword retrieval
      RERANK        — relevance re-scoring of candidates
      COMPRESS      — context compression / summarization
      GRAPH_SEARCH  — knowledge-graph traversal
      TOOL_CALL     — agent tool invocation
      GENERATE      — LLM answer generation
      VERIFY        — answer verification / factuality check
      COMPLETE      — pipeline finished successfully
      ERROR         — pipeline terminated with error
    """

    INTENT = "intent"
    REWRITE = "rewrite"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    COMPRESS = "compress"
    GRAPH_SEARCH = "graph_search"
    TOOL_CALL = "tool_call"
    GENERATE = "generate"
    VERIFY = "verify"
    COMPLETE = "complete"
    ERROR = "error"


# ── RAGRequest ─────────────────────────────────────────────────────────────


class RAGRequest(BaseModel):
    """Unified request model for the RAG API.

    This is the FastAPI-facing request body accepted by POST /api/chat.
    It maps to the internal RAGQuery after validation.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User's natural-language query",
        examples=["What is the capital of France?"],
    )
    kb_id: Optional[str] = Field(
        default=None,
        description="Knowledge base identifier; uses default KB if omitted",
    )
    mode: RAGMode = Field(
        default=RAGMode.NAIVE,
        description="Which RAG paradigm to use for this request",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session identifier for multi-turn chat",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options: top_k, filters, llm_model, etc.",
        examples=[{"top_k": 10, "llm_model": "claude-sonnet-5"}],
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query must not be blank")
        return stripped

    def to_rag_query(self) -> RAGQuery:
        """Convert this API request into an internal RAGQuery."""
        top_k = self.options.get("top_k", 5)
        filters = self.options.get("filters", {})
        return RAGQuery(
            text=self.query,
            strategy=self.mode.to_strategy_type(),
            top_k=min(max(top_k, 1), 100),
            filters=filters,
            conversation_id=self.session_id,
            metadata={
                "kb_id": self.kb_id,
                "raw_options": self.options,
            },
        )


# ── RAGResult ──────────────────────────────────────────────────────────────


class RAGResult(BaseModel):
    """Unified result model returned by the RAG API.

    Aggregates the answer, supporting citations, retrieval hits, trace events,
    token usage, and any non-fatal warnings.  Designed for direct FastAPI
    response serialization (JSON-compatible).
    """

    answer: str = Field(
        ...,
        description="The generated answer text",
        examples=["The capital of France is Paris."],
    )
    citations: List[RAGCitation] = Field(
        default_factory=list,
        description="Citations linking answer claims to source chunks",
    )
    hits: List[RAGChunk] = Field(
        default_factory=list,
        description="Top retrieval hits used as context for generation",
    )
    trace: List[TraceEvent] = Field(
        default_factory=list,
        description="Ordered trace events recording each pipeline stage",
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="LLM token usage breakdown: prompt, completion, total",
        examples=[{"prompt": 1200, "completion": 300, "total": 1500}],
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g. 'reranking skipped', 'graph unavailable')",
    )

    @classmethod
    def from_rag_response(
        cls,
        response: RAGResponse,
        *,
        trace_events: Optional[List[TraceEvent]] = None,
        warnings: Optional[List[str]] = None,
    ) -> RAGResult:
        """Build a RAGResult from an internal RAGResponse + trace data.

        Args:
            response: The RAGResponse produced by a strategy.
            trace_events: Optional ordered list of TraceEvent records.
            warnings: Optional non-fatal warnings collected during execution.

        Returns:
            A fully populated RAGResult ready for JSON serialization.
        """
        return cls(
            answer=response.answer,
            citations=response.citations,
            hits=response.context.chunks if response.context else [],
            trace=trace_events or [],
            usage=response.token_usage,
            warnings=warnings or [],
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The capital of France is Paris.",
                "citations": [
                    {
                        "chunk_id": "chunk_a1b2c3d4e5f6",
                        "document_id": "doc_paris_facts",
                        "text_snippet": "Paris is the capital and most populous city of France.",
                    }
                ],
                "hits": [
                    {
                        "chunk_id": "chunk_a1b2c3d4e5f6",
                        "content": "Paris is the capital and most populous city of France...",
                        "source": {
                            "document_id": "doc_paris_facts",
                            "chunk_index": 3,
                            "title": "Paris Factsheet",
                            "score": 0.92,
                        },
                    }
                ],
                "trace": [
                    {
                        "trace_id": "trace_abc123",
                        "stage": "retrieve",
                        "started_at": "2026-07-20T10:30:00Z",
                        "duration_ms": 45.2,
                        "input_summary": "What is the capital of France?",
                        "output_summary": "Retrieved 5 chunks from vector index",
                    }
                ],
                "usage": {"prompt": 1200, "completion": 300, "total": 1500},
                "warnings": [],
            }
        }
    }


# ── TraceEvent (Pydantic model, API-facing) ────────────────────────────────


class TraceEvent(BaseModel):
    """A single trace event recording one pipeline stage.

    This is the API-facing trace model (Pydantic) for JSON serialization.
    It differs from the internal ``TraceEvent`` dataclass in ``src/rag/trace.py``
    which is optimized for in-memory span-based observability.

    Fields:
        trace_id:   Unique identifier shared by all events in one request.
        stage:      The pipeline stage this event records.
        started_at: ISO-8601 UTC timestamp when the stage began.
        duration_ms: Wall-clock duration of this stage in milliseconds.
        input_summary:  Short human-readable summary of stage input.
        output_summary: Short human-readable summary of stage output.
    """

    trace_id: str = Field(
        ...,
        description="Shared trace identifier for correlating events",
    )
    stage: TraceStage = Field(
        ...,
        description="Pipeline stage this event corresponds to",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this stage started",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of this stage in milliseconds",
    )
    input_summary: str = Field(
        default="",
        description="Short human-readable summary of what entered this stage",
    )
    output_summary: str = Field(
        default="",
        description="Short human-readable summary of what this stage produced",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "trace_id": "trace_abc123def456",
                "stage": "retrieve",
                "started_at": "2026-07-20T10:30:00.123456Z",
                "duration_ms": 45.2,
                "input_summary": "query='What is the capital of France?', top_k=5",
                "output_summary": "retrieved 5 chunks, total 2340 chars",
            }
        }
    }
