"""Public B1 knowledge-domain API."""

from src.models.knowledge import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
)

from .exceptions import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    DuplicateChunkError,
    DuplicateDocumentError,
    DuplicateKnowledgeError,
    InvalidStatusTransitionError,
    KnowledgeDomainError,
    KnowledgeNotFoundError,
    VectorCleanupError,
)
from .repositories import (
    ChunkRepository,
    DocumentRepository,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
    KnowledgeRepository,
)
from .service import KnowledgeService, VectorCleanup
from .state_machine import (
    validate_document_transition,
    validate_knowledge_base_transition,
)

__all__ = [
    "ChunkNotFoundError",
    "ChunkRecord",
    "ChunkRepository",
    "DocumentNotFoundError",
    "DocumentRecord",
    "DocumentRepository",
    "DocumentStatus",
    "DuplicateChunkError",
    "DuplicateDocumentError",
    "DuplicateKnowledgeError",
    "InMemoryChunkRepository",
    "InMemoryDocumentRepository",
    "InMemoryKnowledgeRepository",
    "InvalidStatusTransitionError",
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "KnowledgeDomainError",
    "KnowledgeNotFoundError",
    "KnowledgeRepository",
    "KnowledgeService",
    "VectorCleanup",
    "VectorCleanupError",
    "validate_document_transition",
    "validate_knowledge_base_transition",
]
