"""Repository protocols and thread-safe in-memory B1 implementations."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from threading import RLock
from typing import Dict, Iterable, List, Optional, Set

from src.models.knowledge import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
    utc_now,
)

from .exceptions import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    DuplicateChunkError,
    DuplicateDocumentError,
    DuplicateKnowledgeError,
    KnowledgeNotFoundError,
)
from .state_machine import (
    validate_document_transition,
    validate_knowledge_base_transition,
)


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _failure_message(status: object, failed_status: object, message: Optional[str]) -> Optional[str]:
    if status == failed_status:
        if not message or not message.strip():
            raise ValueError("error_message is required when status is failed")
        return message.strip()[:500]
    return None


class KnowledgeRepository(ABC):
    """Storage contract for knowledge-base metadata."""

    @abstractmethod
    def create(self, knowledge_base: KnowledgeBase) -> KnowledgeBase: ...

    @abstractmethod
    def get(self, kb_id: str) -> KnowledgeBase: ...

    @abstractmethod
    def list(self, owner_id: Optional[str] = None) -> List[KnowledgeBase]: ...

    @abstractmethod
    def update(self, knowledge_base: KnowledgeBase) -> KnowledgeBase: ...

    @abstractmethod
    def update_status(
        self,
        kb_id: str,
        status: KnowledgeBaseStatus,
        error_message: Optional[str] = None,
    ) -> KnowledgeBase: ...

    @abstractmethod
    def delete(self, kb_id: str) -> KnowledgeBase: ...


class DocumentRepository(ABC):
    """Storage contract for document metadata."""

    @abstractmethod
    def create(self, document: DocumentRecord) -> DocumentRecord: ...

    @abstractmethod
    def get(self, document_id: str) -> DocumentRecord: ...

    @abstractmethod
    def list_by_kb_id(self, kb_id: str) -> List[DocumentRecord]: ...

    @abstractmethod
    def find_by_checksum(
        self, kb_id: str, checksum: str
    ) -> Optional[DocumentRecord]: ...

    @abstractmethod
    def update(self, document: DocumentRecord) -> DocumentRecord: ...

    @abstractmethod
    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> DocumentRecord: ...

    @abstractmethod
    def delete(self, document_id: str) -> DocumentRecord: ...


class ChunkRepository(ABC):
    """Storage contract for document chunks."""

    @abstractmethod
    def batch_create(self, chunks: Iterable[ChunkRecord]) -> List[ChunkRecord]: ...

    @abstractmethod
    def get(self, chunk_id: str) -> ChunkRecord: ...

    @abstractmethod
    def list_by_document_id(self, document_id: str) -> List[ChunkRecord]: ...

    @abstractmethod
    def list_by_kb_id(self, kb_id: str) -> List[ChunkRecord]: ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int: ...

    @abstractmethod
    def delete_by_kb_id(self, kb_id: str) -> int: ...


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Thread-safe in-memory knowledge-base repository."""

    def __init__(self) -> None:
        self._items: Dict[str, KnowledgeBase] = {}
        self._lock = RLock()

    def create(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        _require_identifier(knowledge_base.id, "knowledge_base.id")
        with self._lock:
            if knowledge_base.id in self._items:
                raise DuplicateKnowledgeError(knowledge_base.id)
            stored = deepcopy(knowledge_base)
            self._items[stored.id] = stored
            return deepcopy(stored)

    def get(self, kb_id: str) -> KnowledgeBase:
        with self._lock:
            try:
                return deepcopy(self._items[kb_id])
            except KeyError as exc:
                raise KnowledgeNotFoundError(kb_id) from exc

    def list(self, owner_id: Optional[str] = None) -> List[KnowledgeBase]:
        with self._lock:
            items = self._items.values()
            if owner_id is not None:
                items = [item for item in items if item.owner_id == owner_id]
            return deepcopy(sorted(items, key=lambda item: (item.created_at, item.id)))

    def update(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        with self._lock:
            current = self._items.get(knowledge_base.id)
            if current is None:
                raise KnowledgeNotFoundError(knowledge_base.id)
            validate_knowledge_base_transition(current.status, knowledge_base.status)
            message = _failure_message(
                knowledge_base.status,
                KnowledgeBaseStatus.FAILED,
                knowledge_base.error_message,
            )
            stored = replace(
                deepcopy(knowledge_base),
                created_at=current.created_at,
                updated_at=utc_now(),
                error_message=message,
            )
            self._items[stored.id] = stored
            return deepcopy(stored)

    def update_status(
        self,
        kb_id: str,
        status: KnowledgeBaseStatus,
        error_message: Optional[str] = None,
    ) -> KnowledgeBase:
        target = KnowledgeBaseStatus(status)
        with self._lock:
            current = self._items.get(kb_id)
            if current is None:
                raise KnowledgeNotFoundError(kb_id)
            validate_knowledge_base_transition(current.status, target)
            if current.status == target and error_message is None:
                return deepcopy(current)
            message = _failure_message(target, KnowledgeBaseStatus.FAILED, error_message)
            stored = replace(
                current,
                status=target,
                error_message=message,
                updated_at=utc_now(),
            )
            self._items[kb_id] = stored
            return deepcopy(stored)

    def delete(self, kb_id: str) -> KnowledgeBase:
        with self._lock:
            try:
                return deepcopy(self._items.pop(kb_id))
            except KeyError as exc:
                raise KnowledgeNotFoundError(kb_id) from exc


class InMemoryDocumentRepository(DocumentRepository):
    """Thread-safe document repository with per-knowledge-base deduplication."""

    def __init__(self) -> None:
        self._items: Dict[str, DocumentRecord] = {}
        self._lock = RLock()

    def _find_checksum_locked(
        self,
        kb_id: str,
        checksum: str,
        exclude_id: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        for document in self._items.values():
            if (
                document.kb_id == kb_id
                and document.checksum == checksum
                and document.id != exclude_id
            ):
                return document
        return None

    def create(self, document: DocumentRecord) -> DocumentRecord:
        _require_identifier(document.id, "document.id")
        _require_identifier(document.kb_id, "document.kb_id")
        _require_identifier(document.checksum, "document.checksum")
        with self._lock:
            if document.id in self._items:
                raise DuplicateDocumentError(document.id)
            duplicate = self._find_checksum_locked(document.kb_id, document.checksum)
            if duplicate is not None:
                raise DuplicateDocumentError(
                    f"checksum:{document.kb_id}:{document.checksum}"
                )
            stored = deepcopy(document)
            self._items[stored.id] = stored
            return deepcopy(stored)

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            try:
                return deepcopy(self._items[document_id])
            except KeyError as exc:
                raise DocumentNotFoundError(document_id) from exc

    def list_by_kb_id(self, kb_id: str) -> List[DocumentRecord]:
        with self._lock:
            items = [item for item in self._items.values() if item.kb_id == kb_id]
            return deepcopy(sorted(items, key=lambda item: (item.created_at, item.id)))

    def find_by_checksum(
        self, kb_id: str, checksum: str
    ) -> Optional[DocumentRecord]:
        with self._lock:
            item = self._find_checksum_locked(kb_id, checksum)
            return deepcopy(item) if item is not None else None

    def update(self, document: DocumentRecord) -> DocumentRecord:
        with self._lock:
            current = self._items.get(document.id)
            if current is None:
                raise DocumentNotFoundError(document.id)
            duplicate = self._find_checksum_locked(
                document.kb_id,
                document.checksum,
                exclude_id=document.id,
            )
            if duplicate is not None:
                raise DuplicateDocumentError(
                    f"checksum:{document.kb_id}:{document.checksum}"
                )
            validate_document_transition(current.status, document.status)
            message = _failure_message(
                document.status,
                DocumentStatus.FAILED,
                document.error_message,
            )
            stored = replace(
                deepcopy(document),
                created_at=current.created_at,
                updated_at=utc_now(),
                error_message=message,
            )
            self._items[stored.id] = stored
            return deepcopy(stored)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> DocumentRecord:
        target = DocumentStatus(status)
        with self._lock:
            current = self._items.get(document_id)
            if current is None:
                raise DocumentNotFoundError(document_id)
            validate_document_transition(current.status, target)
            if current.status == target and error_message is None:
                return deepcopy(current)
            message = _failure_message(target, DocumentStatus.FAILED, error_message)
            stored = replace(
                current,
                status=target,
                error_message=message,
                updated_at=utc_now(),
            )
            self._items[document_id] = stored
            return deepcopy(stored)

    def delete(self, document_id: str) -> DocumentRecord:
        with self._lock:
            try:
                return deepcopy(self._items.pop(document_id))
            except KeyError as exc:
                raise DocumentNotFoundError(document_id) from exc


class InMemoryChunkRepository(ChunkRepository):
    """Thread-safe chunk repository with atomic batch validation."""

    def __init__(self) -> None:
        self._items: Dict[str, ChunkRecord] = {}
        self._lock = RLock()

    def batch_create(self, chunks: Iterable[ChunkRecord]) -> List[ChunkRecord]:
        candidates = list(chunks)
        seen: Set[str] = set()
        for chunk in candidates:
            if not isinstance(chunk, ChunkRecord):
                raise TypeError("chunks must contain ChunkRecord objects")
            _require_identifier(chunk.id, "chunk.id")
            _require_identifier(chunk.document_id, "chunk.document_id")
            _require_identifier(chunk.kb_id, "chunk.kb_id")
            if chunk.id in seen:
                raise DuplicateChunkError(chunk.id)
            seen.add(chunk.id)

        with self._lock:
            for chunk_id in seen:
                if chunk_id in self._items:
                    raise DuplicateChunkError(chunk_id)
            stored = [deepcopy(chunk) for chunk in candidates]
            for chunk in stored:
                self._items[chunk.id] = chunk
            return deepcopy(stored)

    def get(self, chunk_id: str) -> ChunkRecord:
        with self._lock:
            try:
                return deepcopy(self._items[chunk_id])
            except KeyError as exc:
                raise ChunkNotFoundError(chunk_id) from exc

    def list_by_document_id(self, document_id: str) -> List[ChunkRecord]:
        with self._lock:
            items = [
                item for item in self._items.values() if item.document_id == document_id
            ]
            return deepcopy(sorted(items, key=lambda item: (item.index, item.id)))

    def list_by_kb_id(self, kb_id: str) -> List[ChunkRecord]:
        with self._lock:
            items = [item for item in self._items.values() if item.kb_id == kb_id]
            return deepcopy(
                sorted(items, key=lambda item: (item.document_id, item.index, item.id))
            )

    def delete_by_document_id(self, document_id: str) -> int:
        with self._lock:
            ids = [
                item.id for item in self._items.values() if item.document_id == document_id
            ]
            for chunk_id in ids:
                del self._items[chunk_id]
            return len(ids)

    def delete_by_kb_id(self, kb_id: str) -> int:
        with self._lock:
            ids = [item.id for item in self._items.values() if item.kb_id == kb_id]
            for chunk_id in ids:
                del self._items[chunk_id]
            return len(ids)


__all__ = [
    "ChunkRepository",
    "DocumentRepository",
    "InMemoryChunkRepository",
    "InMemoryDocumentRepository",
    "InMemoryKnowledgeRepository",
    "KnowledgeRepository",
]
