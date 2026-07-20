"""Centralized lifecycle validation for knowledge bases and documents."""

from typing import Dict, Set, TypeVar

from src.models.knowledge import DocumentStatus, KnowledgeBaseStatus

from .exceptions import InvalidStatusTransitionError


KNOWLEDGE_BASE_TRANSITIONS: Dict[KnowledgeBaseStatus, Set[KnowledgeBaseStatus]] = {
    KnowledgeBaseStatus.CREATED: {KnowledgeBaseStatus.INDEXING},
    KnowledgeBaseStatus.INDEXING: {
        KnowledgeBaseStatus.READY,
        KnowledgeBaseStatus.FAILED,
    },
    KnowledgeBaseStatus.READY: set(),
    KnowledgeBaseStatus.FAILED: set(),
}

DOCUMENT_TRANSITIONS: Dict[DocumentStatus, Set[DocumentStatus]] = {
    DocumentStatus.UPLOADED: {DocumentStatus.PARSING},
    DocumentStatus.PARSING: {DocumentStatus.PARSED, DocumentStatus.FAILED},
    DocumentStatus.PARSED: {DocumentStatus.INDEXING},
    DocumentStatus.INDEXING: {DocumentStatus.INDEXED, DocumentStatus.FAILED},
    DocumentStatus.INDEXED: set(),
    DocumentStatus.FAILED: set(),
}

Status = TypeVar("Status", KnowledgeBaseStatus, DocumentStatus)


def _validate_transition(
    current: Status,
    target: Status,
    allowed: Dict[Status, Set[Status]],
    resource: str,
) -> None:
    if current == target:
        return
    if target not in allowed[current]:
        raise InvalidStatusTransitionError(resource, current.value, target.value)


def validate_knowledge_base_transition(
    current: KnowledgeBaseStatus,
    target: KnowledgeBaseStatus,
) -> None:
    """Validate a knowledge-base transition; repeated state is idempotent."""

    current_status = KnowledgeBaseStatus(current)
    target_status = KnowledgeBaseStatus(target)
    _validate_transition(
        current_status,
        target_status,
        KNOWLEDGE_BASE_TRANSITIONS,
        "knowledge base",
    )


def validate_document_transition(
    current: DocumentStatus,
    target: DocumentStatus,
) -> None:
    """Validate a document transition; repeated state is idempotent."""

    current_status = DocumentStatus(current)
    target_status = DocumentStatus(target)
    _validate_transition(
        current_status,
        target_status,
        DOCUMENT_TRANSITIONS,
        "document",
    )


__all__ = [
    "DOCUMENT_TRANSITIONS",
    "KNOWLEDGE_BASE_TRANSITIONS",
    "validate_document_transition",
    "validate_knowledge_base_transition",
]
