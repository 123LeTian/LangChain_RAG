"""检索器模块。"""
from typing import List, Optional
from src.models.schemas import ChunkRecord, RetrievalHit
from .embeddings import BaseEmbedder
from .vector_index import BaseVectorIndex


class Retriever:
    """Top-K 检索器：把查询文本变成向量，从向量索引中找最相关的 Chunk。"""

    def __init__(self, embedder: BaseEmbedder, index: BaseVectorIndex):
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """检索最相关的 Top-K 个 Chunk。"""
        query_vector = self.embedder.embed_query(query)
        hits = self.index.search(query_vector, top_k=top_k)
        return hits

    def retrieve_with_filter(self, query: str, top_k: int = 5, kb_id: str = None) -> List[RetrievalHit]:
        """检索并按知识库过滤。"""
        hits = self.retrieve(query, top_k=top_k)
        if kb_id:
            hits = [h for h in hits if h.chunk.kb_id == kb_id]
        return hits
