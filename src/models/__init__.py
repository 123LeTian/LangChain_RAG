# Models — 原模型 + 共享 RAG 契约

from .evaluation import (
    AgentMetrics,
    AnswerMetrics,
    CitationMetrics,
    EvaluationResult,
    EvaluationSample,
    GraphMetrics,
    MetricsSnapshot,
    RetrievalMetrics,
    SystemMetrics,
)
from .knowledge import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    KnowledgeBase,
    KnowledgeBaseStatus,
)
from .rag import (
    Citation,
    RAGMode,
    RAGRequest,
    RAGResult,
    RetrievalHit,
    TraceEvent,
    TraceStage,
)

__all__ = [
    # RAG 契约
    "RAGMode",
    "TraceStage",
    "RAGRequest",
    "RAGResult",
    "RetrievalHit",
    "Citation",
    "TraceEvent",
    # 知识库模型
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "DocumentRecord",
    "DocumentStatus",
    "DocumentType",
    "ChunkRecord",
    # 评测模型
    "EvaluationSample",
    "EvaluationResult",
    "MetricsSnapshot",
    "RetrievalMetrics",
    "AnswerMetrics",
    "CitationMetrics",
    "SystemMetrics",
    "GraphMetrics",
    "AgentMetrics",
]
