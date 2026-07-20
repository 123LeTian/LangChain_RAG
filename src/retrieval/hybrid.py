"""混合检索模块：向量检索 + 关键词检索。"""
from typing import List
from src.models.schemas import ChunkRecord, RetrievalHit
from .vector_index import BaseVectorIndex
from .embeddings import BaseEmbedder
import re


class KeywordSearcher:
    """简单的关键词检索（基于 TF 匹配）。"""

    def __init__(self):
        self._chunks: List[ChunkRecord] = []

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        self._chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in self._chunks:
            text_words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append(RetrievalHit(chunk=chunk, score=score, rank=0, retriever="keyword"))
        scored.sort(key=lambda h: h.score, reverse=True)
        for rank, hit in enumerate(scored[:top_k]):
            hit.rank = rank
        return scored[:top_k]

    def clear(self) -> None:
        self._chunks = []


class HybridRetriever:
    """混合检索器：向量 + 关键词，加权融合。"""

    def __init__(self, embedder: BaseEmbedder, index: BaseVectorIndex, alpha: float = 0.7):
        self.embedder = embedder
        self.index = index
        self.keyword_searcher = KeywordSearcher()
        self.alpha = alpha  # 向量权重

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        self.index.add_chunks(chunks)
        self.keyword_searcher.add_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        # 向量检索
        query_vec = self.embedder.embed_query(query)
        vec_hits = self.index.search(query_vec, top_k=top_k * 2)
        # 关键词检索
        kw_hits = self.keyword_searcher.search(query, top_k=top_k * 2)
        # 融合
        merged = {}
        for hit in vec_hits:
            merged[hit.chunk.id] = RetrievalHit(
                chunk=hit.chunk, score=self.alpha * hit.score, rank=0, retriever="hybrid"
            )
        for hit in kw_hits:
            if hit.chunk.id in merged:
                merged[hit.chunk.id].score += (1 - self.alpha) * hit.score
            else:
                merged[hit.chunk.id] = RetrievalHit(
                    chunk=hit.chunk, score=(1 - self.alpha) * hit.score, rank=0, retriever="hybrid"
                )
        result = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
        for rank, hit in enumerate(result):
            hit.rank = rank
        return result
