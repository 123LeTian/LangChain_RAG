# RAG Core — Owner: C (主责), 全组评审
#
# Unified orchestration layer for multi-paradigm RAG.
#
# Exports:
#   - Strategy base class and interfaces
#   - Strategy registry (global singleton)
#   - RAG Service (main orchestration entry point)
#   - Trace system (observability)
#   - All strategies (via strategies/ package)

from src.models.rag import (
    RAGStatus,
    StreamEvent,
    StreamEventType,
)
from src.rag.base import (
    EmbedderProtocol,
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategy,
    RAGStrategyBase,
    RerankerProtocol,
    RetrieverProtocol,
)
from src.rag.registry import (
    RAGStrategyRegistry,
    RegistryError,
    StrategyAlreadyRegisteredError,
    StrategyNotRegisteredError,
    get_registry,
    set_registry,
)
from src.rag.service import (
    ExecutionCancelledError,
    ExecutionTimeoutError,
    RAGService,
    RAGServiceError,
    StrategyNotAvailableError,
)
from src.rag.trace import (
    SpanStatus,
    Trace,
    TraceContext,
    TraceEvent,
    TraceEventType,
    TraceRecorder,
    TraceSpan,
    TraceStore,
    get_recorder,
    get_trace_store,
    set_recorder,
    set_trace_store,
)

__all__ = [
    # Base
    "RAGStrategy",
    "RAGStrategyBase",
    "RetrieverProtocol",
    "GeneratorProtocol",
    "GraphRetrieverProtocol",
    "RerankerProtocol",
    "EmbedderProtocol",
    # Registry
    "RAGStrategyRegistry",
    "RegistryError",
    "StrategyNotRegisteredError",
    "StrategyAlreadyRegisteredError",
    "get_registry",
    "set_registry",
    # Service
    "RAGService",
    "RAGServiceError",
    "StrategyNotAvailableError",
    "ExecutionTimeoutError",
    "ExecutionCancelledError",
    "StreamEvent",
    "StreamEventType",
    "RAGStatus",
    # Trace
    "Trace",
    "TraceSpan",
    "TraceEvent",
    "TraceEventType",
    "TraceContext",
    "SpanStatus",
    "TraceStore",
    "TraceRecorder",
    "get_trace_store",
    "set_trace_store",
    "get_recorder",
    "set_recorder",
]
