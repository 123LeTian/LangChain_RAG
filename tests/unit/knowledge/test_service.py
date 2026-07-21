"""Offline orchestration tests for KnowledgeService."""

from itertools import count

import pytest

from src.knowledge import (
    ChunkRecord,
    DocumentNotFoundError,
    DocumentStatus,
    DuplicateDocumentError,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
    InvalidStatusTransitionError,
    KnowledgeBaseStatus,
    KnowledgeNotFoundError,
    KnowledgeService,
    VectorCleanupError,
)


class RecordingVectorCleanup:
    def __init__(self, fail=False):
        self.fail = fail
        self.document_ids = []
        self.kb_ids = []

    def delete_by_document_id(self, document_id):
        self.document_ids.append(document_id)
        if self.fail:
            raise RuntimeError("backend unavailable")

    def delete_by_kb_id(self, kb_id):
        self.kb_ids.append(kb_id)
        if self.fail:
            raise RuntimeError("backend unavailable")


def make_service(cleanup=None):
    sequence = count(1)
    knowledge_repository = InMemoryKnowledgeRepository()
    document_repository = InMemoryDocumentRepository()
    chunk_repository = InMemoryChunkRepository()
    service = KnowledgeService(
        knowledge_repository,
        document_repository,
        chunk_repository,
        vector_cleanup=cleanup,
        id_factory=lambda prefix: f"{prefix}-{next(sequence)}",
    )
    return service, knowledge_repository, document_repository, chunk_repository


def create_kb(service, owner_id="owner-1"):
    return service.create_knowledge_base(
        owner_id=owner_id,
        name="Project Knowledge",
        embedding_model="hash-embedder",
    )


def register_document(service, kb_id, checksum="checksum-1"):
    return service.register_document(
        kb_id=kb_id,
        filename="guide.txt",
        file_type="txt",
        checksum=checksum,
        text="guide content",
        metadata={"source": "documents/guide.txt"},
    )


def add_chunks(chunk_repository, document, count_value=2):
    return chunk_repository.batch_create(
        [
            ChunkRecord(
                id=f"{document.id}:chunk-{index}",
                document_id=document.id,
                kb_id=document.kb_id,
                text=f"chunk {index}",
                index=index,
                metadata={"filename": document.filename},
            )
            for index in range(count_value)
        ]
    )


def test_service_create_get_list_and_update_knowledge_base():
    service, _, _, _ = make_service()
    first = create_kb(service)
    create_kb(service, owner_id="owner-2")

    assert first.id == "kb-1"
    assert first.status == KnowledgeBaseStatus.CREATED
    assert service.get_knowledge_base(first.id).name == "Project Knowledge"
    assert [item.id for item in service.list_knowledge_bases("owner-1")] == [
        first.id
    ]

    updated = service.update_knowledge_base(
        first.id,
        name="Updated Knowledge",
        metadata={"language": "zh"},
    )
    assert updated.name == "Updated Knowledge"
    assert updated.metadata == {"language": "zh"}


def test_service_status_methods_validate_transitions():
    service, _, _, _ = make_service()
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)

    indexing = service.update_knowledge_base_status(
        knowledge_base.id, KnowledgeBaseStatus.INDEXING
    )
    parsing = service.update_document_status(document.id, DocumentStatus.PARSING)

    assert indexing.status == KnowledgeBaseStatus.INDEXING
    assert parsing.status == DocumentStatus.PARSING
    with pytest.raises(InvalidStatusTransitionError):
        service.update_document_status(document.id, DocumentStatus.INDEXED)


def test_service_register_document_checks_kb_and_checksum_scope():
    service, _, _, _ = make_service()
    first_kb = create_kb(service)
    second_kb = create_kb(service, owner_id="owner-2")
    first = register_document(service, first_kb.id, checksum="same")

    with pytest.raises(DuplicateDocumentError, match="checksum"):
        register_document(service, first_kb.id, checksum="same")

    second = register_document(service, second_kb.id, checksum="same")
    assert first.kb_id != second.kb_id
    assert first.status == DocumentStatus.UPLOADED
    assert second.status == DocumentStatus.UPLOADED


def test_service_rejects_document_for_missing_knowledge_base():
    service, _, _, _ = make_service()

    with pytest.raises(KnowledgeNotFoundError, match="missing"):
        register_document(service, "missing")


def test_service_lists_documents_and_gets_chunks():
    service, _, _, chunk_repository = make_service()
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)
    add_chunks(chunk_repository, document)

    assert [item.id for item in service.list_documents(knowledge_base.id)] == [
        document.id
    ]
    assert [item.index for item in service.get_document_chunks(document.id)] == [0, 1]

    with pytest.raises(DocumentNotFoundError, match="missing"):
        service.get_document_chunks("missing")


def test_delete_document_cascades_chunks_and_vector_cleanup():
    cleanup = RecordingVectorCleanup()
    service, _, document_repository, chunk_repository = make_service(cleanup)
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)
    add_chunks(chunk_repository, document)

    deleted = service.delete_document(document.id)

    assert deleted.id == document.id
    assert cleanup.document_ids == [document.id]
    assert chunk_repository.list_by_document_id(document.id) == []
    with pytest.raises(DocumentNotFoundError):
        document_repository.get(document.id)


def test_delete_knowledge_base_cascades_documents_chunks_and_cleanup():
    cleanup = RecordingVectorCleanup()
    service, knowledge_repository, document_repository, chunk_repository = make_service(
        cleanup
    )
    knowledge_base = create_kb(service)
    first = register_document(service, knowledge_base.id, checksum="first")
    second = register_document(service, knowledge_base.id, checksum="second")
    add_chunks(chunk_repository, first)
    add_chunks(chunk_repository, second)

    deleted = service.delete_knowledge_base(knowledge_base.id)

    assert deleted.id == knowledge_base.id
    assert cleanup.kb_ids == [knowledge_base.id]
    assert document_repository.list_by_kb_id(knowledge_base.id) == []
    assert chunk_repository.list_by_kb_id(knowledge_base.id) == []
    with pytest.raises(KnowledgeNotFoundError):
        knowledge_repository.get(knowledge_base.id)


def test_service_without_vector_cleanup_remains_fully_usable():
    service, _, _, chunk_repository = make_service(cleanup=None)
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)
    add_chunks(chunk_repository, document)

    service.delete_document(document.id)
    service.delete_knowledge_base(knowledge_base.id)


def test_vector_cleanup_failure_aborts_document_delete_without_data_loss():
    cleanup = RecordingVectorCleanup(fail=True)
    service, _, document_repository, chunk_repository = make_service(cleanup)
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)
    add_chunks(chunk_repository, document)

    with pytest.raises(VectorCleanupError, match=document.id):
        service.delete_document(document.id)

    assert document_repository.get(document.id).id == document.id
    assert len(chunk_repository.list_by_document_id(document.id)) == 2


def test_vector_cleanup_failure_aborts_kb_delete_without_data_loss():
    cleanup = RecordingVectorCleanup(fail=True)
    service, knowledge_repository, document_repository, chunk_repository = make_service(
        cleanup
    )
    knowledge_base = create_kb(service)
    document = register_document(service, knowledge_base.id)
    add_chunks(chunk_repository, document)

    with pytest.raises(VectorCleanupError, match=knowledge_base.id):
        service.delete_knowledge_base(knowledge_base.id)

    assert knowledge_repository.get(knowledge_base.id).id == knowledge_base.id
    assert document_repository.get(document.id).id == document.id
    assert len(chunk_repository.list_by_kb_id(knowledge_base.id)) == 2
