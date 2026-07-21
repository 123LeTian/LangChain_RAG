"""API-specific Pydantic models for the FastAPI layer.

These models are self-contained to avoid conflicts with the internal
data models (dataclasses in src.models.knowledge / src.models.schemas).
When the real services are wired in, adapters can convert between these
API models and the internal ones.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# Re-export enums and TraceEvent from main's models (compatible definitions)
from src.models.rag import RAGMode, TraceEvent, TraceStage


# ============================================================================
# Retrieval & Citation models (Pydantic, for API JSON serialization)
# ============================================================================

class RetrievalHit(BaseModel):
    """Unified retrieval result returned by all retrievers."""
    chunk_id: str = Field(..., description="Stable Chunk ID")
    text: str = Field(..., description="Chunk text fragment")
    score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., description="Retrieval rank (1-based)")
    retriever: str = Field(default="vector", description="Source retriever name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class Citation(BaseModel):
    """Citation linking an answer claim to a source chunk."""
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Source Chunk ID")
    filename: str = Field(..., description="Document filename")
    page: Optional[int] = Field(default=None, description="Page number if available")
    quote: str = Field(..., description="Quoted source text")
    score: float = Field(default=0.0, description="Relevance score")


# ============================================================================
# RAG Request / Result (API-facing, with kb_id)
# ============================================================================

class RAGRequest(BaseModel):
    """Unified RAG request accepted by all five modes."""
    query: str = Field(..., min_length=1, description="User query")
    kb_id: str = Field(default="default", description="Target knowledge base ID")
    mode: RAGMode = Field(default=RAGMode.NAIVE, description="RAG mode")
    session_id: Optional[str] = Field(default=None, description="Chat session ID")
    options: dict[str, Any] = Field(default_factory=dict, description="Mode-specific options")


class RAGResult(BaseModel):
    """Unified RAG result returned by all five modes."""
    answer: str = Field(..., description="Final answer text")
    citations: list[Citation] = Field(default_factory=list, description="Citation list")
    hits: list[RetrievalHit] = Field(default_factory=list, description="Retrieval hits")
    trace: list[TraceEvent] = Field(default_factory=list, description="Execution trace")
    usage: dict[str, Any] = Field(default_factory=dict, description="Token usage stats")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")
    mode: RAGMode = Field(..., description="Executed RAG mode")


# ============================================================================
# Knowledge-base models (Pydantic, for API serialization)
# ============================================================================

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class KnowledgeBaseStatus(str, Enum):
    CREATED = "created"
    CREATING = "creating"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Document record for API responses."""
    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:8]}")
    kb_id: str
    filename: str
    type: DocumentType
    checksum: str = ""
    status: DocumentStatus = DocumentStatus.UPLOADED
    chunk_count: int = 0
    size_bytes: int = 0


class KnowledgeBase(BaseModel):
    """Knowledge base record for API responses."""
    id: str
    name: str
    description: str = ""
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.CREATED
    owner_id: str = "user_001"
    doc_count: int = 0
    chunk_count: int = 0