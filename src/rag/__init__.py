# RAG Core — Owner: C (主责), 全组评审
#
# Unified orchestration layer for multi-paradigm RAG.
#
# Exports:
#   - Strategy base class and interfaces
#   - Strategy registry (global singleton + factory)
#   - RAG Service (main orchestration entry point)
#   - Trace system (observability)
#   - All strategies (via strategies/ package)

from src.rag.base import (
    EmbedderProtocol,
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategyBase,
    RerankerProtocol,
    RetrieverProtocol,
)
from src.rag.registry import (
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
    StrategyRegistry,
    get_registry,
    set_registry,
)
from src.rag.service import (
    RAGService,
    StreamEvent,
    StreamEventType,
    create_rag_service,
)
from src.rag.trace import (
    SpanStatus,
    Trace,
    TraceContext,
    TraceEvent,
    TraceEventType,
    TraceSpan,
    TraceStore,
    get_trace_store,
    set_trace_store,
)

__all__ = [
    # Base
    "RAGStrategyBase",
    "RetrieverProtocol",
    "GeneratorProtocol",
    "GraphRetrieverProtocol",
    "RerankerProtocol",
    "EmbedderProtocol",
    # Registry
    "StrategyRegistry",
    "StrategyNotFoundError",
    "StrategyAlreadyRegisteredError",
    "get_registry",
    "set_registry",
    # Service
    "RAGService",
    "create_rag_service",
    "StreamEvent",
    "StreamEventType",
    # Trace
    "Trace",
    "TraceSpan",
    "TraceEvent",
    "TraceEventType",
    "TraceContext",
    "SpanStatus",
    "TraceStore",
    "get_trace_store",
    "set_trace_store",
]
