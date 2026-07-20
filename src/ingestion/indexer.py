"""Offline orchestration for loading, normalizing, splitting, and indexing."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from langchain_core.documents import Document

from src.knowledge.exceptions import DuplicateDocumentError
from src.knowledge.service import KnowledgeService
from src.models.knowledge import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    KnowledgeBaseStatus,
)
from src.retrieval.embeddings import BaseEmbedder
from src.retrieval.vector_index import BaseVectorIndex

from .exceptions import (
    DocumentLoadError,
    EmbeddingValidationError,
    EmptyContentError,
    IndexingOperationError,
    IndexingRollbackError,
    IngestionError,
    ReindexDocumentMismatchError,
    UnsupportedDocumentTypeError,
)
from .loaders import LoaderFactory
from .normalizer import TextNormalizer
from .splitter import split_document


PathLike = Union[str, Path]
Splitter = Callable[[DocumentRecord], Sequence[Document]]


class Indexer:
    """Coordinate the complete B2 ingestion lifecycle using injected dependencies."""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        loader_registry=LoaderFactory,
        normalizer: Optional[TextNormalizer] = None,
        splitter: Splitter = split_document,
        embedder: Optional[BaseEmbedder] = None,
        vector_index: Optional[BaseVectorIndex] = None,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        if vector_index is None:
            raise ValueError("vector_index is required")
        self.knowledge_service = knowledge_service
        self.loader_registry = loader_registry
        self.normalizer = normalizer or TextNormalizer()
        self.splitter = splitter
        self.embedder = embedder
        self.vector_index = vector_index

    def index_document(
        self,
        kb_id: str,
        file_path: PathLike,
        *,
        document_id: Optional[str] = None,
    ) -> DocumentRecord:
        """Index one supported file and return its final document record."""

        path, file_bytes, checksum, loader = self._prepare_file(file_path)
        self.knowledge_service.get_knowledge_base(kb_id)
        duplicate = self.knowledge_service.document_repository.find_by_checksum(
            kb_id, checksum
        )
        if duplicate is not None:
            raise DuplicateDocumentError(f"checksum:{kb_id}:{checksum}")

        self._ensure_knowledge_base_indexing(kb_id)
        document = self.knowledge_service.register_document(
            kb_id=kb_id,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            checksum=checksum,
            metadata={
                "file_size": len(file_bytes),
                "filename": path.name,
                "source": path.name,
            },
            document_id=document_id,
        )

        stage = "loading"
        try:
            self.knowledge_service.update_document_status(
                document.id, DocumentStatus.PARSING
            )
            loaded = loader.load(path, kb_id=kb_id, document_id=document.id)
            if loaded.checksum != checksum:
                raise IngestionError(
                    f"Checksum changed while reading document '{document.id}'"
                )

            stage = "normalization"
            normalized = self.normalizer.normalize_document(loaded)
            current = self.knowledge_service.get_document(document.id)
            parsed = replace(
                current,
                filename=normalized.filename,
                file_type=normalized.file_type,
                text=normalized.text,
                checksum=checksum,
                metadata=deepcopy(normalized.metadata),
                status=DocumentStatus.PARSED,
                error_message=None,
            )
            parsed = self.knowledge_service.document_repository.update(parsed)

            stage = "splitting"
            split_documents = self._split(parsed)
            if not split_documents:
                raise EmptyContentError(path.name)
            chunks = self._make_chunks(parsed, split_documents)
            self.knowledge_service.chunk_repository.batch_create(chunks)

            self.knowledge_service.update_document_status(
                document.id, DocumentStatus.INDEXING
            )
            stage = "embedding"
            vectors = self._embed(chunks)

            stage = "vector write"
            self.vector_index.upsert(chunks, vectors)

            indexed = self.knowledge_service.update_document_status(
                document.id, DocumentStatus.INDEXED
            )
            self._mark_knowledge_base_ready(kb_id)
            return indexed
        except Exception as exc:
            rollback_failures = self._rollback(document.id, kb_id, exc)
            if rollback_failures:
                raise IndexingRollbackError(document.id, rollback_failures) from exc
            if isinstance(exc, IngestionError):
                raise
            if stage == "loading":
                raise DocumentLoadError(document.id, exc) from exc
            raise IndexingOperationError(document.id, stage, exc) from exc

    def reindex_document(
        self,
        kb_id: str,
        document_id: str,
        file_path: PathLike,
    ) -> DocumentRecord:
        """Replace one document's chunks and vectors using the supplied source file."""

        current = self.knowledge_service.get_document(document_id)
        if current.kb_id != kb_id:
            raise ReindexDocumentMismatchError(document_id, kb_id)

        _, _, checksum, _ = self._prepare_file(file_path)
        duplicate = self.knowledge_service.document_repository.find_by_checksum(
            kb_id, checksum
        )
        if duplicate is not None and duplicate.id != document_id:
            raise DuplicateDocumentError(f"checksum:{kb_id}:{checksum}")

        try:
            self.vector_index.delete_by_document_id(document_id)
            self.knowledge_service.chunk_repository.delete_by_document_id(document_id)
            self.knowledge_service.document_repository.delete(document_id)
        except Exception as exc:
            raise IndexingOperationError(
                document_id, "reindex cleanup", exc
            ) from exc

        return self.index_document(
            kb_id,
            file_path,
            document_id=document_id,
        )

    def _prepare_file(self, file_path: PathLike):
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document file was not found: {path}")
        try:
            loader = self.loader_registry.get_loader(path)
        except ValueError as exc:
            raise UnsupportedDocumentTypeError(path.suffix.lower()) from exc
        file_bytes = path.read_bytes()
        if not file_bytes:
            raise EmptyContentError(path.name)
        checksum = hashlib.sha256(file_bytes).hexdigest()
        return path, file_bytes, checksum, loader

    def _split(self, document: DocumentRecord) -> List[Document]:
        splitter = self.splitter
        if callable(splitter):
            return list(splitter(document))
        method = getattr(splitter, "split_document", None)
        if callable(method):
            return list(method(document))
        raise TypeError("splitter must be callable or expose split_document()")

    def _make_chunks(
        self,
        document: DocumentRecord,
        split_documents: Sequence[Document],
    ) -> List[ChunkRecord]:
        chunks = []
        for index, split in enumerate(split_documents):
            text = split.page_content
            if not isinstance(text, str) or not text.strip():
                raise EmptyContentError(f"chunk {index}")
            chunk_id = f"{document.id}:chunk-{index}"
            metadata = deepcopy(split.metadata)
            metadata.update(
                {
                    "kb_id": document.kb_id,
                    "document_id": document.id,
                    "chunk_id": chunk_id,
                    "filename": document.filename,
                    "source": metadata.get("source", document.filename),
                    "page": metadata.get("page", document.metadata.get("page", 1)),
                    "section": metadata.get(
                        "section", document.metadata.get("section", "")
                    ),
                    "chunk_index": index,
                }
            )
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document.id,
                    kb_id=document.kb_id,
                    text=text,
                    index=index,
                    metadata=metadata,
                )
            )
        return chunks

    def _embed(self, chunks: Sequence[ChunkRecord]) -> List[List[float]]:
        raw_vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if len(raw_vectors) != len(chunks):
            raise EmbeddingValidationError(
                f"Embedding count mismatch: expected {len(chunks)}, got {len(raw_vectors)}"
            )

        expected_dimension = self.embedder.dimension
        vectors: List[List[float]] = []
        for index, raw_vector in enumerate(raw_vectors):
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingValidationError(
                    f"Embedding {index} contains non-numeric values"
                ) from exc
            if len(vector) != expected_dimension:
                raise EmbeddingValidationError(
                    f"Embedding {index} dimension mismatch: "
                    f"expected {expected_dimension}, got {len(vector)}"
                )
            if not vector or not all(math.isfinite(value) for value in vector):
                raise EmbeddingValidationError(
                    f"Embedding {index} must contain finite values"
                )
            vectors.append(vector)
        return vectors

    def _ensure_knowledge_base_indexing(self, kb_id: str) -> None:
        knowledge_base = self.knowledge_service.get_knowledge_base(kb_id)
        if knowledge_base.status == KnowledgeBaseStatus.CREATED:
            self.knowledge_service.update_knowledge_base_status(
                kb_id, KnowledgeBaseStatus.INDEXING
            )
        elif knowledge_base.status not in {
            KnowledgeBaseStatus.INDEXING,
            KnowledgeBaseStatus.READY,
        }:
            # Delegate creation of the established domain transition error.
            self.knowledge_service.update_knowledge_base_status(
                kb_id, KnowledgeBaseStatus.INDEXING
            )

    def _mark_knowledge_base_ready(self, kb_id: str) -> None:
        knowledge_base = self.knowledge_service.get_knowledge_base(kb_id)
        if knowledge_base.status == KnowledgeBaseStatus.INDEXING:
            self.knowledge_service.update_knowledge_base_status(
                kb_id, KnowledgeBaseStatus.READY
            )

    def _rollback(
        self,
        document_id: str,
        kb_id: str,
        cause: Exception,
    ) -> List[str]:
        failures: List[str] = []
        try:
            self.vector_index.delete_by_document_id(document_id)
        except Exception as exc:
            failures.append(f"vector cleanup: {exc}")
        try:
            self.knowledge_service.chunk_repository.delete_by_document_id(document_id)
        except Exception as exc:
            failures.append(f"chunk cleanup: {exc}")

        summary = str(cause).strip() or cause.__class__.__name__
        try:
            current = self.knowledge_service.get_document(document_id)
            if current.status == DocumentStatus.UPLOADED:
                current = self.knowledge_service.update_document_status(
                    document_id, DocumentStatus.PARSING
                )
            if current.status == DocumentStatus.PARSED:
                current = self.knowledge_service.update_document_status(
                    document_id, DocumentStatus.INDEXING
                )
            if current.status in {DocumentStatus.PARSING, DocumentStatus.INDEXING}:
                self.knowledge_service.update_document_status(
                    document_id,
                    DocumentStatus.FAILED,
                    error_message=summary,
                )
        except Exception as exc:
            failures.append(f"document status: {exc}")

        try:
            knowledge_base = self.knowledge_service.get_knowledge_base(kb_id)
            if knowledge_base.status == KnowledgeBaseStatus.INDEXING:
                self.knowledge_service.update_knowledge_base_status(
                    kb_id,
                    KnowledgeBaseStatus.FAILED,
                    error_message=summary,
                )
        except Exception as exc:
            failures.append(f"knowledge-base status: {exc}")
        return failures


__all__ = ["Indexer"]
