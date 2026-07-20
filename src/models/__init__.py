# Models — Shared data contracts
#
# Owner: C (主责), Shared
#
# Core RAG models (all Pydantic, JSON-serializable, FastAPI-ready):
#   RAGRequest  — unified API request
#   RAGResult   — unified API response
#   RAGQuery    — internal query object
#   RAGResponse — internal response object
#   RAGChunk    — retrieved text chunk
#   RAGCitation — source citation
#   TraceEvent  — API-facing trace event

from src.models.rag import (
    # ── Enums ──
    RAGMode,
    RAGStatus,
    StrategyType,
    TraceStage,
    StreamEventType,
    # ── Core models ──
    RAGChunk,
    RAGCitation,
    RAGContext,
    RAGPipelineConfig,
    RAGQuery,
    RAGRequest,
    RAGResponse,
    RAGResult,
    RAGSource,
    StreamEvent,
    TraceEvent,
)

__all__ = [
    # Enums
    "RAGMode",
    "RAGStatus",
    "StrategyType",
    "TraceStage",
    "StreamEventType",
    # Models
    "RAGChunk",
    "RAGCitation",
    "RAGContext",
    "RAGPipelineConfig",
    "RAGQuery",
    "RAGRequest",
    "RAGResponse",
    "RAGResult",
    "RAGSource",
    "StreamEvent",
    "TraceEvent",
]
