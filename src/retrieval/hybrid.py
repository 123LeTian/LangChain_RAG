"""Isolated vector + keyword hybrid retrieval with Chinese trigram support."""

from copy import deepcopy
from contextvars import ContextVar
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


def _tokenize_zh(text: str) -> set:
    """Tokenize Chinese+mixed text into unigrams, bigrams, and trigrams."""
    tokens = set()
    raw_words = re.findall(r"[\w]+", text.lower())
    for word in raw_words:
        ascii_part = re.findall(r"[a-z0-9]+", word)
        chinese_part = re.findall(r"[\u4e00-\u9fff]+", word)
        tokens.update(ascii_part)
        for seg in chinese_part:
            n = len(seg)
            for i in range(n):
                tokens.add(seg[i])
                if i < n - 1:
                    tokens.add(seg[i:i+2])
                if i < n - 2:
                    tokens.add(seg[i:i+3])
    return tokens


class KeywordSearcher:
    """Chinese-aware keyword searcher with trigram matching + phrase boost."""

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

        query_tokens = _tokenize_zh(query)
        query_lower = query.lower()
        # Extract Chinese segments for phrase matching
        zh_segs = re.findall(r"[\u4e00-\u9fff]+", query)

        scored = []
        for chunk in self._chunks:
            if kb_id is not None and chunk.kb_id != kb_id:
                continue
            if not _metadata_matches(chunk, filters):
                continue
            text_lower = chunk.text.lower()
            text_tokens = _tokenize_zh(chunk.text)

            # Token overlap score
            overlap = len(query_tokens & text_tokens)
            if overlap == 0 and not any(seg in text_lower for seg in zh_segs):
                continue

            base_score = overlap / max(len(query_tokens), 1)

            # Phrase boost: exact query substring or long Chinese segment match
            phrase_boost = 0.0
            if query_lower in text_lower:
                phrase_boost = 0.5
            else:
                for seg in zh_segs:
                    if len(seg) >= 3 and seg in text_lower:
                        phrase_boost = max(phrase_boost, 0.25)
                    elif len(seg) >= 2 and seg in text_lower:
                        phrase_boost = max(phrase_boost, 0.12)

            final_score = base_score + phrase_boost
            scored.append((final_score, chunk.id, chunk))

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


class HybridRetrievalError(RuntimeError):
    """Raised when both vector and keyword hybrid branches fail."""


class HybridRetriever:
    """Weighted hybrid retriever with Chinese-aware keyword search."""

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
        self._warning_context: ContextVar[tuple[str, ...]] = ContextVar(
            f"hybrid_warnings_{id(self)}",
            default=(),
        )

    @property
    def last_warnings(self) -> List[str]:
        return list(self._warning_context.get())

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
        warnings = []
        failures = []
        try:
            vec_hits = self.vector_retriever.search(
                query, kb_id=kb_id, top_k=top_k * 2, filters=filters,
            )
        except Exception as exc:
            vec_hits = []
            failures.append(exc)
            warnings.append(self._branch_warning("vector", exc))
        try:
            kw_hits = self.keyword_searcher.search(
                query, kb_id=kb_id, top_k=top_k * 2, filters=filters,
            )
        except Exception as exc:
            kw_hits = []
            failures.append(exc)
            warnings.append(self._branch_warning("keyword", exc))
        self._warning_context.set(tuple(warnings))
        if len(failures) == 2:
            raise HybridRetrievalError("; ".join(warnings))
        return self._merge(vec_hits, kw_hits, top_k, rank_start=1)

    @staticmethod
    def _branch_warning(branch: str, error: Exception) -> str:
        detail = str(error).replace("\n", " ").strip()[:160]
        suffix = f": {detail}" if detail else ""
        return f"hybrid {branch} branch degraded ({type(error).__name__}){suffix}"

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """Legacy all-KB path."""
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
        result = sorted(merged.values(), key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]
        for rank, hit in enumerate(result, start=rank_start):
            hit.rank = rank
        return result


__all__ = ["HybridRetrievalError", "HybridRetriever", "KeywordSearcher"]