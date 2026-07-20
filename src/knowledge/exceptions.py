"""Domain exceptions for knowledge storage and orchestration."""


class KnowledgeDomainError(Exception):
    """Base class for knowledge-domain failures."""


class KnowledgeNotFoundError(KnowledgeDomainError):
    def __init__(self, kb_id: str):
        super().__init__(f"Knowledge base '{kb_id}' was not found")


class DocumentNotFoundError(KnowledgeDomainError):
    def __init__(self, document_id: str):
        super().__init__(f"Document '{document_id}' was not found")


class ChunkNotFoundError(KnowledgeDomainError):
    def __init__(self, chunk_id: str):
        super().__init__(f"Chunk '{chunk_id}' was not found")


class DuplicateKnowledgeError(KnowledgeDomainError):
    def __init__(self, kb_id: str):
        super().__init__(f"Knowledge base '{kb_id}' already exists")


class DuplicateDocumentError(KnowledgeDomainError):
    def __init__(self, identifier: str):
        super().__init__(f"Document '{identifier}' already exists")


class DuplicateChunkError(KnowledgeDomainError):
    def __init__(self, chunk_id: str):
        super().__init__(f"Chunk '{chunk_id}' already exists")


class InvalidStatusTransitionError(KnowledgeDomainError):
    def __init__(self, resource: str, current: str, target: str):
        super().__init__(
            f"Invalid {resource} status transition: '{current}' -> '{target}'"
        )


class VectorCleanupError(KnowledgeDomainError):
    def __init__(self, resource_id: str):
        super().__init__(f"Vector cleanup failed for '{resource_id}'")


__all__ = [
    "ChunkNotFoundError",
    "DocumentNotFoundError",
    "DuplicateChunkError",
    "DuplicateDocumentError",
    "DuplicateKnowledgeError",
    "InvalidStatusTransitionError",
    "KnowledgeDomainError",
    "KnowledgeNotFoundError",
    "VectorCleanupError",
]
