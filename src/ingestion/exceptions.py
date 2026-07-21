"""Domain errors raised by the offline ingestion pipeline."""


class IngestionError(Exception):
    """Base class for deterministic ingestion failures."""


class EmptyContentError(IngestionError):
    """Raised when normalization or parsing produces no usable text."""

    def __init__(self, source: str = "document") -> None:
        super().__init__(f"{source} contains no usable text")


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when no loader is registered for a file extension."""

    def __init__(self, suffix: str) -> None:
        super().__init__(f"Unsupported document type: '{suffix or '<none>'}'")


class EmbeddingValidationError(IngestionError):
    """Raised when an embedder returns malformed or mismatched vectors."""


class DocumentLoadError(IngestionError):
    """Raised when a supported document cannot be parsed."""

    def __init__(self, document_id: str, cause: Exception) -> None:
        self.document_id = document_id
        super().__init__(f"Failed to load document '{document_id}': {cause}")


class IndexingOperationError(IngestionError):
    """Raised when an indexing stage fails after a document was registered."""

    def __init__(self, document_id: str, stage: str, cause: Exception) -> None:
        self.document_id = document_id
        self.stage = stage
        super().__init__(
            f"Indexing document '{document_id}' failed during {stage}: {cause}"
        )


class IndexingRollbackError(IngestionError):
    """Raised when both indexing and its compensating cleanup fail."""

    def __init__(self, document_id: str, failures: list[str]) -> None:
        self.document_id = document_id
        details = "; ".join(failures)
        super().__init__(f"Rollback failed for document '{document_id}': {details}")


class ReindexDocumentMismatchError(IngestionError):
    """Raised when a document does not belong to the requested knowledge base."""

    def __init__(self, document_id: str, kb_id: str) -> None:
        super().__init__(
            f"Document '{document_id}' does not belong to knowledge base '{kb_id}'"
        )


__all__ = [
    "DocumentLoadError",
    "EmbeddingValidationError",
    "EmptyContentError",
    "IndexingOperationError",
    "IndexingRollbackError",
    "IngestionError",
    "ReindexDocumentMismatchError",
    "UnsupportedDocumentTypeError",
]
