"""共享数据模型 — Owner: Shared"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .knowledge import ChunkRecord


@dataclass
class RetrievalHit:
    """检索命中结果。"""
    chunk: ChunkRecord
    score: float
    rank: int
    retriever: str = "vector"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(), "score": self.score,
            "rank": self.rank, "retriever": self.retriever, "metadata": self.metadata,
        }


@dataclass
class Citation:
    """引用来源。"""
    document_id: str
    chunk_id: str
    filename: str
    page: Optional[int] = None
    quote: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id, "chunk_id": self.chunk_id,
            "filename": self.filename, "page": self.page, "quote": self.quote, "score": self.score,
        }
