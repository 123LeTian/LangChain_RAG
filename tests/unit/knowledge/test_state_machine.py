"""State-machine tests for B1 knowledge resources."""

import pytest

from src.knowledge import (
    DocumentStatus,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
    InvalidStatusTransitionError,
    KnowledgeBaseStatus,
    validate_document_transition,
    validate_knowledge_base_transition,
)

from .test_repositories import make_document, make_kb


def test_knowledge_base_legal_transitions_and_idempotency():
    validate_knowledge_base_transition(
        KnowledgeBaseStatus.CREATED, KnowledgeBaseStatus.INDEXING
    )
    validate_knowledge_base_transition(
        KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.READY
    )
    validate_knowledge_base_transition(
        KnowledgeBaseStatus.READY, KnowledgeBaseStatus.READY
    )


def test_knowledge_base_illegal_and_terminal_transitions():
    with pytest.raises(InvalidStatusTransitionError, match="created.*ready"):
        validate_knowledge_base_transition(
            KnowledgeBaseStatus.CREATED, KnowledgeBaseStatus.READY
        )
    with pytest.raises(InvalidStatusTransitionError, match="failed.*indexing"):
        validate_knowledge_base_transition(
            KnowledgeBaseStatus.FAILED, KnowledgeBaseStatus.INDEXING
        )


def test_document_legal_transitions_and_idempotency():
    path = [
        DocumentStatus.UPLOADED,
        DocumentStatus.PARSING,
        DocumentStatus.PARSED,
        DocumentStatus.INDEXING,
        DocumentStatus.INDEXED,
    ]
    for current, target in zip(path, path[1:]):
        validate_document_transition(current, target)
    validate_document_transition(DocumentStatus.INDEXED, DocumentStatus.INDEXED)


def test_document_illegal_and_terminal_transitions():
    with pytest.raises(InvalidStatusTransitionError, match="uploaded.*indexed"):
        validate_document_transition(
            DocumentStatus.UPLOADED, DocumentStatus.INDEXED
        )
    with pytest.raises(InvalidStatusTransitionError, match="failed.*parsing"):
        validate_document_transition(DocumentStatus.FAILED, DocumentStatus.PARSING)


def test_failed_status_stores_error_summary():
    knowledge_repository = InMemoryKnowledgeRepository()
    document_repository = InMemoryDocumentRepository()
    knowledge_repository.create(make_kb())
    document_repository.create(make_document())

    knowledge_repository.update_status("kb-1", KnowledgeBaseStatus.INDEXING)
    failed_kb = knowledge_repository.update_status(
        "kb-1", KnowledgeBaseStatus.FAILED, "embedding service failed"
    )
    document_repository.update_status("doc-1", DocumentStatus.PARSING)
    failed_document = document_repository.update_status(
        "doc-1", DocumentStatus.FAILED, "invalid document"
    )

    assert failed_kb.error_message == "embedding service failed"
    assert failed_document.error_message == "invalid document"
    assert (
        knowledge_repository.update_status("kb-1", KnowledgeBaseStatus.FAILED)
        == failed_kb
    )
    assert (
        document_repository.update_status("doc-1", DocumentStatus.FAILED)
        == failed_document
    )


def test_failed_status_requires_an_error_summary():
    repository = InMemoryDocumentRepository()
    repository.create(make_document())
    repository.update_status("doc-1", DocumentStatus.PARSING)

    with pytest.raises(ValueError, match="error_message"):
        repository.update_status("doc-1", DocumentStatus.FAILED)
