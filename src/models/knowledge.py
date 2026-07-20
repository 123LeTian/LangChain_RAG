"""Shared knowledge-domain models used by ingestion, storage, and APIs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union


class KnowledgeBaseStatus(str, Enum):
    """Lifecycle states for a knowledge base."""

    CREATED = "created"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentStatus(str, Enum):
    """Lifecycle states for a document."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


StatusType = TypeVar("StatusType", bound=Enum)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _coerce_status(value: Union[str, StatusType], enum_type: Type[StatusType]) -> StatusType:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _coerce_datetime(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class KnowledgeBase:
    """Knowledge-base metadata and lifecycle state."""

    id: str
    owner_id: str
    name: str
    embedding_model: str
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.CREATED
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_status(self.status, KnowledgeBaseStatus)
        self.created_at = _coerce_datetime(self.created_at)
        self.updated_at = _coerce_datetime(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "status": self.status.value,
            "embedding_model": self.embedding_model,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        """Create a knowledge base from serialized data."""

        return cls(**data)


@dataclass
class DocumentRecord:
    """Document metadata plus loader text, checksum, and lifecycle state."""

    id: str
    kb_id: str
    filename: str
    file_type: str
    text: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PARSED
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.status = _coerce_status(self.status, DocumentStatus)
        self.created_at = _coerce_datetime(self.created_at)
        self.updated_at = _coerce_datetime(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "id": self.id,
            "kb_id": self.kb_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "text": self.text,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
            "status": self.status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Create a document from serialized data."""

        return cls(**data)


@dataclass
class ChunkRecord:
    """Smallest retrievable unit with source-preserving metadata."""

    id: str
    document_id: str
    kb_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.created_at = _coerce_datetime(self.created_at)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "id": self.id,
            "document_id": self.document_id,
            "kb_id": self.kb_id,
            "text": self.text,
            "index": self.index,
            "metadata": dict(self.metadata),
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create a chunk from serialized data."""

        return cls(**data)


__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "DocumentStatus",
    "KnowledgeBase",
    "KnowledgeBaseStatus",
    "utc_now",
]
