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
from datetime import datetime
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
