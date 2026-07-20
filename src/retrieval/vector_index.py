"""Vector-index protocol and isolated in-memory/Chroma implementations."""

from abc import ABC, abstractmethod
from copy import deepcopy
import math
from threading import RLock
from typing import Any, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.models.schemas import ChunkRecord, RetrievalHit

from .embeddings import BaseEmbedder


MetadataFilters = Optional[Mapping[str, Any]]


@runtime_checkable
class VectorIndexProtocol(Protocol):
    """Formal storage contract required by B2 indexing and B3 retrieval."""

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        kb_id: str,
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]: ...

    def delete_by_document_id(self, document_id: str) -> int: ...

    def delete_by_kb_id(self, kb_id: str) -> int: ...

    def count(self, kb_id: Optional[str] = None) -> int: ...

    def clear(self) -> None: ...


class BaseVectorIndex(ABC):
    """Backward-compatible ABC implementing the formal vector-index semantics."""

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Embed and add chunks using the configured embedder."""

    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert or replace chunks and their precomputed vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        kb_id: str,
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        """Search only candidates belonging to one knowledge base."""

    @abstractmethod
    def search_all(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        """Explicit management/legacy search across all knowledge bases."""

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int: ...

    @abstractmethod
    def delete_by_kb_id(self, kb_id: str) -> int: ...

    @abstractmethod
    def count(self, kb_id: Optional[str] = None) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _normalize_filters(kb_id: Optional[str], filters: MetadataFilters) -> dict:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError("filters must be a mapping or None")
    normalized = dict(filters)
    requested_kb = normalized.pop("kb_id", kb_id)
    if kb_id is not None and requested_kb != kb_id:
        raise ValueError("filters cannot override kb_id")
    return normalized


def _validate_batch(
    chunks: Sequence[ChunkRecord],
    vectors: Sequence[Sequence[float]],
) -> List[List[float]]:
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique within an upsert batch")

    result: List[List[float]] = []
    dimension: Optional[int] = None
    for index, (chunk, raw_vector) in enumerate(zip(chunks, vectors)):
        _validate_identifier(chunk.id, "chunk.id")
        _validate_identifier(chunk.kb_id, "chunk.kb_id")
        _validate_identifier(chunk.document_id, "chunk.document_id")
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Vector {index} contains non-numeric values") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Vectors must contain finite values and be non-empty")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("All vectors must have one consistent dimension")
        result.append(vector)
    return result


def _stored_chunk(chunk: ChunkRecord) -> ChunkRecord:
    stored = deepcopy(chunk)
    metadata = deepcopy(stored.metadata)
    protected = {
        "kb_id": stored.kb_id,
        "document_id": stored.document_id,
        "chunk_id": stored.id,
        "chunk_index": stored.index,
    }
    for key, expected in protected.items():
        actual = metadata.get(key, expected)
        if actual != expected:
            raise ValueError(f"Chunk metadata '{key}' conflicts with its canonical value")
        metadata[key] = expected
    stored.metadata = metadata
    return stored


def _matches_filters(chunk: ChunkRecord, filters: Mapping[str, Any]) -> bool:
    return all(chunk.metadata.get(key) == value for key, value in filters.items())


def _query_array(query_vector: Sequence[float], dimension: Optional[int]) -> np.ndarray:
    try:
        query = np.asarray([float(value) for value in query_vector], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("query_vector must contain numeric values") from exc
    if query.ndim != 1 or not len(query) or not np.all(np.isfinite(query)):
        raise ValueError("query_vector must be a non-empty finite one-dimensional vector")
    if dimension is not None and len(query) != dimension:
        raise ValueError(
            f"Query vector dimension mismatch: expected {dimension}, got {len(query)}"
        )
    return query


def _hit(chunk: ChunkRecord, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=deepcopy(chunk),
        score=float(score),
        rank=rank,
        retriever="vector",
        metadata=deepcopy(chunk.metadata),
    )


class InMemoryVectorIndex(BaseVectorIndex):
    """Thread-safe index that filters candidates before similarity and Top-K."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._chunks: List[ChunkRecord] = []
        self._vectors: List[np.ndarray] = []
        self._dimension: Optional[int] = None
        self._lock = RLock()

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.upsert(chunks, vectors)

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        candidates = [_stored_chunk(chunk) for chunk in chunks]
        candidate_vectors = _validate_batch(candidates, vectors)
        if not candidates:
            return
        dimension = len(candidate_vectors[0])
        arrays = [np.asarray(vector, dtype=np.float32) for vector in candidate_vectors]

        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dimension}, got {dimension}"
                )
            incoming_ids = {chunk.id for chunk in candidates}
            retained = [
                (chunk, vector)
                for chunk, vector in zip(self._chunks, self._vectors)
                if chunk.id not in incoming_ids
            ]
            self._chunks = [chunk for chunk, _ in retained] + candidates
            self._vectors = [vector for _, vector in retained] + arrays
            self._dimension = dimension

    def search(
        self,
        query_vector: Sequence[float],
        kb_id: str,
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        _validate_identifier(kb_id, "kb_id")
        return self._search(query_vector, kb_id, top_k, filters)

    def search_all(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        return self._search(query_vector, None, top_k, filters)

    def _search(
        self,
        query_vector: Sequence[float],
        kb_id: Optional[str],
        top_k: int,
        filters: MetadataFilters,
    ) -> List[RetrievalHit]:
        _validate_top_k(top_k)
        normalized_filters = _normalize_filters(kb_id, filters)
        with self._lock:
            dimension = self._dimension
            entries = [
                (deepcopy(chunk), vector.copy())
                for chunk, vector in zip(self._chunks, self._vectors)
                if (kb_id is None or chunk.kb_id == kb_id)
                and _matches_filters(chunk, normalized_filters)
            ]
        query = _query_array(query_vector, dimension)
        if not entries:
            return []

        query_norm = np.linalg.norm(query)
        scored = []
        for chunk, vector in entries:
            denominator = np.linalg.norm(vector) * query_norm
            score = float(vector @ query / denominator) if denominator else 0.0
            scored.append((score, chunk.id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _hit(chunk, score, rank)
            for rank, (score, _, chunk) in enumerate(scored[:top_k])
        ]

    def delete_by_document_id(self, document_id: str) -> int:
        _validate_identifier(document_id, "document_id")
        return self._delete(lambda chunk: chunk.document_id == document_id)

    def delete_by_kb_id(self, kb_id: str) -> int:
        _validate_identifier(kb_id, "kb_id")
        return self._delete(lambda chunk: chunk.kb_id == kb_id)

    def _delete(self, predicate) -> int:
        with self._lock:
            retained = [
                (chunk, vector)
                for chunk, vector in zip(self._chunks, self._vectors)
                if not predicate(chunk)
            ]
            deleted = len(self._chunks) - len(retained)
            self._chunks = [chunk for chunk, _ in retained]
            self._vectors = [vector for _, vector in retained]
            if not self._chunks:
                self._dimension = None
            return deleted

    def count(self, kb_id: Optional[str] = None) -> int:
        if kb_id is not None:
            _validate_identifier(kb_id, "kb_id")
        with self._lock:
            return sum(1 for chunk in self._chunks if kb_id is None or chunk.kb_id == kb_id)

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._vectors = []
            self._dimension = None


def _chroma_where(kb_id: Optional[str], filters: MetadataFilters) -> Optional[dict]:
    normalized = _normalize_filters(kb_id, filters)
    conditions = []
    if kb_id is not None:
        conditions.append({"kb_id": {"$eq": kb_id}})
    conditions.extend({key: {"$eq": value}} for key, value in normalized.items())
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ChromaVectorIndex(BaseVectorIndex):
    """Optional Chroma index using database-side ``where`` isolation."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str = "default",
        persist_path: Optional[str] = None,
    ):
        self.embedder = embedder
        self._collection_name = collection_name
        self._persist_path = persist_path
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorIndex 需要可选依赖 'chromadb'；"
                "请执行: pip install -r requirements-chroma.txt"
            ) from exc
        if self._persist_path:
            self._client = chromadb.PersistentClient(path=self._persist_path)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.upsert(chunks, vectors)

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        candidates = [_stored_chunk(chunk) for chunk in chunks]
        candidate_vectors = _validate_batch(candidates, vectors)
        if not candidates:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in candidates],
            embeddings=candidate_vectors,
            documents=[chunk.text for chunk in candidates],
            metadatas=[deepcopy(chunk.metadata) for chunk in candidates],
        )

    def search(
        self,
        query_vector: Sequence[float],
        kb_id: str,
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        _validate_identifier(kb_id, "kb_id")
        return self._search(query_vector, kb_id, top_k, filters)

    def search_all(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: MetadataFilters = None,
    ) -> List[RetrievalHit]:
        return self._search(query_vector, None, top_k, filters)

    def _search(
        self,
        query_vector: Sequence[float],
        kb_id: Optional[str],
        top_k: int,
        filters: MetadataFilters,
    ) -> List[RetrievalHit]:
        _validate_top_k(top_k)
        query = _query_array(query_vector, None).tolist()
        where = _chroma_where(kb_id, filters)
        get_options = {"where": where} if where is not None else {}
        candidate_ids = self._collection.get(**get_options).get("ids", [])
        if not candidate_ids:
            return []
        query_options = {
            "query_embeddings": [query],
            # Query all already-filtered candidates so equal-score ties can be
            # ordered deterministically by chunk ID before applying Top-K.
            "n_results": len(candidate_ids),
        }
        if where is not None:
            query_options["where"] = where
        results = self._collection.query(**query_options)
        scored = []
        for chunk_id, distance, text, metadata in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            chunk = ChunkRecord(
                id=chunk_id,
                document_id=metadata["document_id"],
                kb_id=metadata["kb_id"],
                text=text,
                index=metadata.get("chunk_index", 0),
                metadata=deepcopy(metadata),
            )
            scored.append((1.0 - float(distance), chunk.id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _hit(chunk, score, rank)
            for rank, (score, _, chunk) in enumerate(scored[:top_k])
        ]

    def delete_by_document_id(self, document_id: str) -> int:
        _validate_identifier(document_id, "document_id")
        return self._delete_where({"document_id": {"$eq": document_id}})

    def delete_by_kb_id(self, kb_id: str) -> int:
        _validate_identifier(kb_id, "kb_id")
        return self._delete_where({"kb_id": {"$eq": kb_id}})

    def _delete_where(self, where: dict) -> int:
        existing = self._collection.get(where=where)
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self, kb_id: Optional[str] = None) -> int:
        if kb_id is None:
            return self._collection.count()
        _validate_identifier(kb_id, "kb_id")
        result = self._collection.get(where={"kb_id": {"$eq": kb_id}})
        return len(result.get("ids", []))

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )


__all__ = [
    "BaseVectorIndex",
    "ChromaVectorIndex",
    "InMemoryVectorIndex",
    "MetadataFilters",
    "VectorIndexProtocol",
]
