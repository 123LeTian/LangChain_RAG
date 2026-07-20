"""Offline tests for B1 in-memory repositories."""

from dataclasses import replace
from datetime import timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.knowledge import (
    ChunkNotFoundError,
    ChunkRecord,
    DocumentNotFoundError,
    DocumentRecord,
    DocumentStatus,
    DuplicateChunkError,
    DuplicateDocumentError,
    DuplicateKnowledgeError,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
    KnowledgeBase,
    KnowledgeBaseStatus,
    KnowledgeNotFoundError,
)
from src.ingestion.models import DocumentRecord as IngestionDocumentRecord
from src.models.schemas import ChunkRecord as SchemaChunkRecord


def make_kb(kb_id="kb-1", owner_id="owner-1", name="Knowledge"):
    return KnowledgeBase(
        id=kb_id,
        owner_id=owner_id,
        name=name,
        embedding_model="hash-embedder",
    )


def make_document(
    document_id="doc-1",
    kb_id="kb-1",
    checksum="checksum-1",
    status=DocumentStatus.UPLOADED,
):
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        filename=f"{document_id}.txt",
        file_type="txt",
        text="document text",
        checksum=checksum,
        status=status,
    )


def make_chunk(chunk_id, document_id="doc-1", kb_id="kb-1", index=0):
    return ChunkRecord(
        id=chunk_id,
        document_id=document_id,
        kb_id=kb_id,
        text=f"chunk {index}",
        index=index,
        metadata={
            "filename": f"{document_id}.txt",
            "page": 1,
            "section": "intro",
            "source": f"documents/{document_id}.txt",
        },
    )


def test_shared_model_import_paths_are_identical():
    assert IngestionDocumentRecord is DocumentRecord
    assert SchemaChunkRecord is ChunkRecord


def test_models_use_utc_serializable_status_and_independent_metadata():
    first = make_kb("kb-first")
    second = make_kb("kb-second")
    first.metadata["tag"] = "first"

    assert second.metadata == {}
    assert first.created_at.tzinfo == timezone.utc
    assert first.to_dict()["status"] == "created"
    assert first.to_dict()["created_at"].endswith("+00:00")


def test_knowledge_repository_crud_returns_detached_objects():
    repository = InMemoryKnowledgeRepository()
    created = repository.create(make_kb())
    created.metadata["outside"] = True

    stored = repository.get("kb-1")
    assert stored.metadata == {}

    updated = repository.update(replace(stored, name="Updated"))
    assert updated.name == "Updated"
    assert repository.list()[0].name == "Updated"

    deleted = repository.delete("kb-1")
    assert deleted.id == "kb-1"
    with pytest.raises(KnowledgeNotFoundError, match="kb-1"):
        repository.get("kb-1")


def test_knowledge_repository_rejects_duplicate_id_and_filters_owner():
    repository = InMemoryKnowledgeRepository()
    repository.create(make_kb("kb-2", "owner-2"))
    repository.create(make_kb("kb-1", "owner-1"))

    with pytest.raises(DuplicateKnowledgeError, match="kb-1"):
        repository.create(make_kb("kb-1", "owner-3"))

    assert [item.id for item in repository.list("owner-1")] == ["kb-1"]
    first_order = [item.id for item in repository.list()]
    assert set(first_order) == {"kb-1", "kb-2"}
    assert [item.id for item in repository.list()] == first_order


def test_knowledge_repository_missing_update_and_delete_are_explicit():
    repository = InMemoryKnowledgeRepository()

    with pytest.raises(KnowledgeNotFoundError, match="missing"):
        repository.update(make_kb("missing"))
    with pytest.raises(KnowledgeNotFoundError, match="missing"):
        repository.delete("missing")


def test_document_repository_crud_and_kb_listing():
    repository = InMemoryDocumentRepository()
    first = repository.create(make_document("doc-2"))
    repository.create(make_document("doc-1", checksum="checksum-2"))
    repository.create(make_document("other", "kb-2", "checksum-1"))
    first.metadata["outside"] = True

    assert repository.get("doc-2").metadata == {}
    first_order = [item.id for item in repository.list_by_kb_id("kb-1")]
    assert set(first_order) == {"doc-1", "doc-2"}
    assert [item.id for item in repository.list_by_kb_id("kb-1")] == first_order

    updated = repository.update(replace(repository.get("doc-2"), filename="new.txt"))
    assert updated.filename == "new.txt"
    assert repository.delete("doc-2").id == "doc-2"
    with pytest.raises(DocumentNotFoundError, match="doc-2"):
        repository.get("doc-2")


def test_document_checksum_is_unique_only_inside_one_kb():
    repository = InMemoryDocumentRepository()
    repository.create(make_document("doc-1", "kb-1", "same"))

    with pytest.raises(DuplicateDocumentError, match="checksum:kb-1:same"):
        repository.create(make_document("doc-2", "kb-1", "same"))

    other = repository.create(make_document("doc-3", "kb-2", "same"))
    assert other.kb_id == "kb-2"
    assert repository.find_by_checksum("kb-1", "same").id == "doc-1"
    assert repository.find_by_checksum("kb-3", "same") is None


def test_document_repository_rejects_duplicate_and_missing_ids():
    repository = InMemoryDocumentRepository()
    repository.create(make_document())

    with pytest.raises(DuplicateDocumentError, match="doc-1"):
        repository.create(make_document(checksum="different"))
    with pytest.raises(DocumentNotFoundError, match="missing"):
        repository.update(make_document("missing"))
    with pytest.raises(DocumentNotFoundError, match="missing"):
        repository.delete("missing")


def test_chunk_repository_batch_query_order_and_source_metadata():
    repository = InMemoryChunkRepository()
    repository.batch_create(
        [
            make_chunk("chunk-2", index=2),
            make_chunk("chunk-0", index=0),
            make_chunk("chunk-1", index=1),
            make_chunk("other", "doc-2", "kb-2", 0),
        ]
    )

    chunks = repository.list_by_document_id("doc-1")
    assert [chunk.id for chunk in chunks] == ["chunk-0", "chunk-1", "chunk-2"]
    assert set(chunks[0].metadata) >= {"filename", "page", "section", "source"}
    assert [chunk.id for chunk in repository.list_by_kb_id("kb-2")] == ["other"]


def test_chunk_repository_returns_detached_objects_and_explicit_missing_error():
    repository = InMemoryChunkRepository()
    created = repository.batch_create([make_chunk("chunk-1")])[0]
    created.metadata["outside"] = True

    assert "outside" not in repository.get("chunk-1").metadata
    with pytest.raises(ChunkNotFoundError, match="missing"):
        repository.get("missing")


def test_chunk_batch_validation_is_atomic_for_duplicate_ids():
    repository = InMemoryChunkRepository()

    with pytest.raises(DuplicateChunkError, match="duplicate"):
        repository.batch_create(
            [make_chunk("duplicate"), make_chunk("duplicate", index=1)]
        )

    assert repository.list_by_kb_id("kb-1") == []


def test_chunk_delete_operations_preserve_kb_isolation():
    repository = InMemoryChunkRepository()
    repository.batch_create(
        [
            make_chunk("doc-1-a", "doc-1", "kb-1", 0),
            make_chunk("doc-1-b", "doc-1", "kb-1", 1),
            make_chunk("doc-2-a", "doc-2", "kb-1", 0),
            make_chunk("other", "doc-3", "kb-2", 0),
        ]
    )

    assert repository.delete_by_document_id("doc-1") == 2
    assert [item.id for item in repository.list_by_kb_id("kb-1")] == ["doc-2-a"]
    assert repository.delete_by_kb_id("kb-1") == 1
    assert [item.id for item in repository.list_by_kb_id("kb-2")] == ["other"]


def test_in_memory_chunk_repository_supports_concurrent_writes():
    repository = InMemoryChunkRepository()

    def create_one(index):
        repository.batch_create([make_chunk(f"chunk-{index}", index=index)])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(create_one, range(40)))

    assert len(repository.list_by_kb_id("kb-1")) == 40
