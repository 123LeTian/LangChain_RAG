"""Isolated vector + keyword hybrid retrieval."""

from copy import deepcopy
import re
from typing import Any, Dict, List, Optional

from src.models.schemas import ChunkRecord, RetrievalHit

from .embeddings import BaseEmbedder
from .retriever import VectorRetriever
from .vector_index import BaseVectorIndex


def _metadata_matches(chunk: ChunkRecord, filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    return all(chunk.metadata.get(key) == value for key, value in filters.items())


class KeywordSearcher:
    """Small deterministic keyword searcher with pre-ranking isolation."""

    def __init__(self):
        self._chunks: List[ChunkRecord] = []

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        existing = {chunk.id: chunk for chunk in self._chunks}
        existing.update({chunk.id: deepcopy(chunk) for chunk in chunks})
        self._chunks = list(existing.values())

    def search(
        self,
        query: str,
        top_k: int = 5,
        kb_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and kb_id is not None:
            if filters.get("kb_id", kb_id) != kb_id:
                raise ValueError("filters cannot override kb_id")
            filters = {key: value for key, value in filters.items() if key != "kb_id"}
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        # Isolation and filters are applied before scoring and Top-K.
        for chunk in self._chunks:
            if kb_id is not None and chunk.kb_id != kb_id:
                continue
            if not _metadata_matches(chunk, filters):
                continue
            text_words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score, chunk.id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(
                chunk=deepcopy(chunk),
                score=score,
                rank=rank,
                retriever="keyword",
                metadata=deepcopy(chunk.metadata),
            )
            for rank, (score, _, chunk) in enumerate(scored[:top_k])
        ]

    def clear(self) -> None:
        self._chunks = []


class HybridRetriever:
    """Weighted hybrid retriever with strict B3 search and legacy retrieve."""

    retriever_name: str = "hybrid"

    def __init__(
        self,
        embedder: BaseEmbedder,
        index: BaseVectorIndex,
        alpha: float = 0.7,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.embedder = embedder
        self.index = index
        self.vector_retriever = VectorRetriever(embedder, index)
        self.keyword_searcher = KeywordSearcher()
        self.alpha = alpha

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        self.index.add_chunks(chunks)
        self.keyword_searcher.add_chunks(chunks)

    def search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        vec_hits = self.vector_retriever.search(
            query,
            kb_id=kb_id,
            top_k=top_k * 2,
            filters=filters,
        )
        kw_hits = self.keyword_searcher.search(
            query,
            kb_id=kb_id,
            top_k=top_k * 2,
            filters=filters,
        )
        return self._merge(vec_hits, kw_hits, top_k, rank_start=1)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """Legacy explicit all-KB path retained for existing callers."""

        query_vector = self.embedder.embed_query(query)
        vec_hits = self.index.search_all(query_vector, top_k=top_k * 2)
        kw_hits = self.keyword_searcher.search(query, top_k=top_k * 2)
        return self._merge(vec_hits, kw_hits, top_k, rank_start=0)

    def _merge(
        self,
        vector_hits: List[RetrievalHit],
        keyword_hits: List[RetrievalHit],
        top_k: int,
        rank_start: int,
    ) -> List[RetrievalHit]:
        merged: Dict[str, RetrievalHit] = {}
        for hit in vector_hits:
            merged[hit.chunk.id] = RetrievalHit(
                chunk=deepcopy(hit.chunk),
                score=self.alpha * hit.score,
                rank=0,
                retriever=self.retriever_name,
                metadata=deepcopy(hit.metadata),
            )
        for hit in keyword_hits:
            if hit.chunk.id in merged:
                merged[hit.chunk.id].score += (1 - self.alpha) * hit.score
            else:
                merged[hit.chunk.id] = RetrievalHit(
                    chunk=deepcopy(hit.chunk),
                    score=(1 - self.alpha) * hit.score,
                    rank=0,
                    retriever=self.retriever_name,
                    metadata=deepcopy(hit.metadata),
                )
        result = sorted(
            merged.values(),
            key=lambda hit: (-hit.score, hit.chunk.id),
        )[:top_k]
        for rank, hit in enumerate(result, start=rank_start):
            hit.rank = rank
        return result


__all__ = ["HybridRetriever", "KeywordSearcher"]
