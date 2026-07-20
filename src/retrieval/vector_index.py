"""Vector indexes used by retrieval and the offline ingestion pipeline."""

from abc import ABC, abstractmethod
from copy import deepcopy
from threading import RLock
from typing import List, Optional, Sequence

import numpy as np

from src.models.schemas import ChunkRecord, RetrievalHit

from .embeddings import BaseEmbedder


class BaseVectorIndex(ABC):
    """Backward-compatible vector-index contract."""

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Embed and add chunks using the index's configured embedder."""

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Write precomputed vectors; concrete B2-capable indexes override this."""

        raise NotImplementedError("This vector index does not support precomputed upsert")

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete one document's vectors; concrete B2-capable indexes override this."""

        raise NotImplementedError("This vector index does not support document deletion")

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[RetrievalHit]:
        """Search for the most similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed chunks."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all indexed chunks."""


def _validate_batch(
    chunks: Sequence[ChunkRecord],
    vectors: Sequence[Sequence[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )
    if not chunks:
        return
    dimensions = {len(vector) for vector in vectors}
    if 0 in dimensions or len(dimensions) != 1:
        raise ValueError("All vectors must have one consistent non-zero dimension")


def _vector_metadata(chunk: ChunkRecord) -> dict:
    metadata = deepcopy(chunk.metadata)
    metadata.setdefault("kb_id", chunk.kb_id)
    metadata.setdefault("document_id", chunk.document_id)
    metadata.setdefault("chunk_id", chunk.id)
    metadata.setdefault("chunk_index", chunk.index)
    return {key: value for key, value in metadata.items() if value is not None}


class InMemoryVectorIndex(BaseVectorIndex):
    """Thread-safe NumPy-backed vector index for development and offline tests."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._chunks: List[ChunkRecord] = []
        self._vectors: List[np.ndarray] = []
        self._lock = RLock()

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.upsert(chunks, vectors)

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        candidates = list(chunks)
        candidate_vectors = [list(vector) for vector in vectors]
        _validate_batch(candidates, candidate_vectors)
        if len({chunk.id for chunk in candidates}) != len(candidates):
            raise ValueError("Chunk IDs must be unique within an upsert batch")

        new_vectors = [np.asarray(vector, dtype=np.float32) for vector in candidate_vectors]
        if any(not np.all(np.isfinite(vector)) for vector in new_vectors):
            raise ValueError("Vectors must contain only finite numeric values")

        with self._lock:
            incoming_ids = {chunk.id for chunk in candidates}
            retained = [
                (chunk, vector)
                for chunk, vector in zip(self._chunks, self._vectors)
                if chunk.id not in incoming_ids
            ]
            self._chunks = [deepcopy(chunk) for chunk, _ in retained]
            self._vectors = [vector.copy() for _, vector in retained]
            self._chunks.extend(deepcopy(candidates))
            self._vectors.extend(vector.copy() for vector in new_vectors)

    def delete_by_document_id(self, document_id: str) -> int:
        with self._lock:
            retained = [
                (chunk, vector)
                for chunk, vector in zip(self._chunks, self._vectors)
                if chunk.document_id != document_id
            ]
            deleted = len(self._chunks) - len(retained)
            self._chunks = [chunk for chunk, _ in retained]
            self._vectors = [vector for _, vector in retained]
            return deleted

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[RetrievalHit]:
        with self._lock:
            if not self._chunks:
                return []
            chunks = deepcopy(self._chunks)
            vectors = [vector.copy() for vector in self._vectors]

        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(vectors, dtype=np.float32)
        scores = matrix @ query / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-8
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(
                chunk=chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever="vector",
            )
            for rank, index in enumerate(top_indices)
        ]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._vectors = []


class ChromaVectorIndex(BaseVectorIndex):
    """Optional Chroma index; importing this module does not require Chroma."""

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
            raise ImportError("请安装 chromadb: pip install chromadb") from exc
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
        candidates = list(chunks)
        candidate_vectors = [list(vector) for vector in vectors]
        _validate_batch(candidates, candidate_vectors)
        if not candidates:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in candidates],
            embeddings=candidate_vectors,
            documents=[chunk.text for chunk in candidates],
            metadatas=[_vector_metadata(chunk) for chunk in candidates],
        )

    def delete_by_document_id(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[RetrievalHit]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        hits = []
        for rank, (chunk_id, distance, text, metadata) in enumerate(
            zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ):
            chunk = ChunkRecord(
                id=chunk_id,
                document_id=metadata.get("document_id", ""),
                kb_id=metadata.get("kb_id", ""),
                text=text,
                index=metadata.get("chunk_index", metadata.get("index", 0)),
                metadata=metadata,
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=1.0 - distance,
                    rank=rank,
                    retriever="vector",
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )


__all__ = ["BaseVectorIndex", "ChromaVectorIndex", "InMemoryVectorIndex"]
