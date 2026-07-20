"""Application service coordinating B1 knowledge repositories."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import uuid4

from src.models.knowledge import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
    utc_now,
)

from .exceptions import DuplicateDocumentError, VectorCleanupError
from .repositories import ChunkRepository, DocumentRepository, KnowledgeRepository
from .state_machine import (
    validate_document_transition,
    validate_knowledge_base_transition,
)


class VectorCleanup(Protocol):
    """Optional future VectorIndex cleanup dependency."""

    def delete_by_document_id(self, document_id: str) -> None: ...

    def delete_by_kb_id(self, kb_id: str) -> None: ...


IdFactory = Callable[[str], str]


def _default_id_factory(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class KnowledgeService:
    """Coordinate knowledge, document, and chunk lifecycle operations."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        vector_cleanup: Optional[VectorCleanup] = None,
        id_factory: IdFactory = _default_id_factory,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.vector_cleanup = vector_cleanup
        self._id_factory = id_factory

    def create_knowledge_base(
        self,
        owner_id: str,
        name: str,
        embedding_model: str,
        metadata: Optional[Dict[str, Any]] = None,
        kb_id: Optional[str] = None,
    ) -> KnowledgeBase:
        """Create a knowledge base in the CREATED state."""

        knowledge_base = KnowledgeBase(
            id=kb_id or self._id_factory("kb"),
            owner_id=owner_id,
            name=name,
            embedding_model=embedding_model,
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        return self.knowledge_repository.create(knowledge_base)

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Return one knowledge base or raise a domain not-found error."""

        return self.knowledge_repository.get(kb_id)

    def list_knowledge_bases(
        self, owner_id: Optional[str] = None
    ) -> List[KnowledgeBase]:
        """List knowledge bases, optionally filtered by owner."""

        return self.knowledge_repository.list(owner_id=owner_id)

    def update_knowledge_base(
        self,
        kb_id: str,
        name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """Update editable knowledge-base metadata without changing its state."""

        current = self.knowledge_repository.get(kb_id)
        updated = replace(
            current,
            name=name if name is not None else current.name,
            embedding_model=(
                embedding_model
                if embedding_model is not None
                else current.embedding_model
            ),
            metadata=(deepcopy(metadata) if metadata is not None else current.metadata),
            updated_at=utc_now(),
        )
        return self.knowledge_repository.update(updated)

    def update_knowledge_base_status(
        self,
        kb_id: str,
        status: KnowledgeBaseStatus,
        error_message: Optional[str] = None,
    ) -> KnowledgeBase:
        """Validate and apply a knowledge-base lifecycle transition."""

        current = self.knowledge_repository.get(kb_id)
        target = KnowledgeBaseStatus(status)
        validate_knowledge_base_transition(current.status, target)
        return self.knowledge_repository.update_status(kb_id, target, error_message)

    def delete_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Delete a knowledge base and cascade through documents and chunks."""

        knowledge_base = self.knowledge_repository.get(kb_id)
        self._cleanup_vectors_for_kb(kb_id)
        self.chunk_repository.delete_by_kb_id(kb_id)
        for document in self.document_repository.list_by_kb_id(kb_id):
            self.document_repository.delete(document.id)
        self.knowledge_repository.delete(kb_id)
        return knowledge_base

    def register_document(
        self,
        kb_id: str,
        filename: str,
        file_type: str,
        checksum: str,
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> DocumentRecord:
        """Register an uploaded document, deduplicated within its knowledge base."""

        self.knowledge_repository.get(kb_id)
        duplicate = self.document_repository.find_by_checksum(kb_id, checksum)
        if duplicate is not None:
            raise DuplicateDocumentError(f"checksum:{kb_id}:{checksum}")
        document = DocumentRecord(
            id=document_id or self._id_factory("doc"),
            kb_id=kb_id,
            filename=filename,
            file_type=file_type,
            text=text,
            checksum=checksum,
            status=DocumentStatus.UPLOADED,
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        return self.document_repository.create(document)

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return a registered document."""

        return self.document_repository.get(document_id)

    def list_documents(self, kb_id: str) -> List[DocumentRecord]:
        """List documents belonging to one knowledge base."""

        self.knowledge_repository.get(kb_id)
        return self.document_repository.list_by_kb_id(kb_id)

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> DocumentRecord:
        """Validate and apply a document lifecycle transition."""

        current = self.document_repository.get(document_id)
        target = DocumentStatus(status)
        validate_document_transition(current.status, target)
        return self.document_repository.update_status(
            document_id,
            target,
            error_message,
        )

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Delete a document and cascade through its chunks."""

        document = self.document_repository.get(document_id)
        self._cleanup_vectors_for_document(document_id)
        self.chunk_repository.delete_by_document_id(document_id)
        self.document_repository.delete(document_id)
        return document

    def get_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        """Return chunks for an existing document in stable index order."""

        self.document_repository.get(document_id)
        return self.chunk_repository.list_by_document_id(document_id)

    def _cleanup_vectors_for_document(self, document_id: str) -> None:
        if self.vector_cleanup is None:
            return
        try:
            self.vector_cleanup.delete_by_document_id(document_id)
        except Exception as exc:
            raise VectorCleanupError(document_id) from exc

    def _cleanup_vectors_for_kb(self, kb_id: str) -> None:
        if self.vector_cleanup is None:
            return
        try:
            self.vector_cleanup.delete_by_kb_id(kb_id)
        except Exception as exc:
            raise VectorCleanupError(kb_id) from exc


__all__ = ["KnowledgeService", "VectorCleanup"]
