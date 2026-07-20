"""KnowledgeRepository 协议：管理知识库的统一接口。"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, runtime_checkable
from src.models.schemas import ChunkRecord, RetrievalHit
from .embeddings import BaseEmbedder
from .vector_index import BaseVectorIndex


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Structural copy of C's frozen synchronous retrieval contract."""

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> Any:
        """Return C's RAGContext for one query."""
        ...

    @property
    def retriever_name(self) -> str:
        """Human-readable retriever name used by C's trace layer."""
        ...


class KnowledgeRepository(ABC):
    """知识库仓库抽象接口。"""

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        """添加 Chunk 到知识库，返回添加数量。"""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """搜索知识库。"""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """按 ID 获取 Chunk。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回 Chunk 数量。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空知识库。"""
        ...


class SimpleKnowledgeRepository(KnowledgeRepository):
    """简单知识库实现：内存存储 + 向量索引。"""

    def __init__(self, embedder: BaseEmbedder, index: BaseVectorIndex):
        self.embedder = embedder
        self.index = index
        self._chunks: dict[str, ChunkRecord] = {}

    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        count = 0
        for chunk in chunks:
            if chunk.id not in self._chunks:
                self._chunks[chunk.id] = chunk
                count += 1
        self.index.add_chunks([c for c in chunks if c.embedding is None or True])
        return count

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        query_vec = self.embedder.embed_query(query)
        return self.index.search_all(query_vec, top_k=top_k)

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = {}
        self.index.clear()


__all__ = [
    "KnowledgeRepository",
    "RetrieverProtocol",
    "SimpleKnowledgeRepository",
]
