"""向量索引模块。"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from src.models.schemas import ChunkRecord, RetrievalHit
from .embeddings import BaseEmbedder


class BaseVectorIndex(ABC):
    """向量索引抽象基类。"""

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """添加 Chunk 到索引。"""
        ...

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[RetrievalHit]:
        """搜索最相关的 Chunk。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回索引中的 Chunk 数量。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""
        ...


class InMemoryVectorIndex(BaseVectorIndex):
    """内存向量索引（基于 NumPy 余弦相似度）。
    适合开发测试，生产环境请用 ChromaVectorIndex。
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._chunks: List[ChunkRecord] = []
        self._vectors: List[np.ndarray] = []

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        for chunk, vec in zip(chunks, vectors):
            self._chunks.append(chunk)
            self._vectors.append(np.array(vec, dtype=np.float32))

    def search(self, query_vector: List[float], top_k: int = 5) -> List[RetrievalHit]:
        if not self._chunks:
            return []
        query = np.array(query_vector, dtype=np.float32)
        matrix = np.array(self._vectors, dtype=np.float32)
        # 余弦相似度
        scores = matrix @ query / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-8
        )
        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        hits = []
        for rank, idx in enumerate(top_indices):
            hits.append(RetrievalHit(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                rank=rank,
                retriever="vector",
            ))
        return hits

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._vectors = []


class ChromaVectorIndex(BaseVectorIndex):
    """Chroma 向量索引（生产环境推荐）。"""

    def __init__(self, embedder: BaseEmbedder, collection_name: str = "default", persist_path: str = None):
        self.embedder = embedder
        self._collection_name = collection_name
        self._persist_path = persist_path
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
        if self._persist_path:
            self._client = chromadb.PersistentClient(path=self._persist_path)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )

    def search(self, query_vector: List[float], top_k: int = 5) -> List[RetrievalHit]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        hits = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for rank, (cid, dist, doc, meta) in enumerate(zip(ids, distances, documents, metadatas)):
            chunk = ChunkRecord(id=cid, document_id=meta.get("document_id", ""), kb_id=meta.get("kb_id", ""), text=doc, index=meta.get("index", 0), metadata=meta)
            hits.append(RetrievalHit(chunk=chunk, score=1.0 - dist, rank=rank, retriever="vector"))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
