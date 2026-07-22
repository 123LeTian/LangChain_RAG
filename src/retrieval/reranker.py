"""Reranker module with Chinese-aware keyword matching.

Key improvement: extracts the most important Chinese key phrases from the query
and gives massive boosts when they appear as contiguous substrings in the chunk.
This prevents common single-character trigrams from flooding results.
"""
from abc import ABC, abstractmethod
import re
from typing import List
from src.models.schemas import RetrievalHit


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


def _extract_key_phrases(query: str) -> List[str]:
    """Extract the most important Chinese substrings from the query.

    Returns phrases sorted by length (longest first), which are most
    discriminative for matching specific financial terms.
    """
    zh_segs = re.findall(r"[\u4e00-\u9fff]+", query)
    phrases = []
    for seg in zh_segs:
        n = len(seg)
        if n <= 2:
            phrases.append(seg)
            continue
        for length in range(n, 1, -1):
            for i in range(n - length + 1):
                sub = seg[i:i+length]
                if len(sub) >= 2:
                    phrases.append(sub)
    seen = set()
    result = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _phrase_boost(query: str, text_lower: str) -> float:
    """Calculate phrase boost based on contiguous key-phrase matches.

    Longer matches get exponentially higher boosts:
    - 6+ char match: 0.45
    - 4-5 char match: 0.30
    - 3 char match: 0.18
    - 2 char match: 0.08
    """
    phrases = _extract_key_phrases(query)
    best_boost = 0.0
    for phrase in phrases:
        plen = len(phrase)
        if phrase in text_lower:
            if plen >= 6:
                best_boost = max(best_boost, 0.45)
            elif plen >= 4:
                best_boost = max(best_boost, 0.30)
            elif plen >= 3:
                best_boost = max(best_boost, 0.18)
            elif plen >= 2:
                best_boost = max(best_boost, 0.08)
    return best_boost


class BaseReranker(ABC):
    """Reranker abstract base."""

    @abstractmethod
    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        ...


class SimpleReranker(BaseReranker):
    """Reranker with Chinese key-phrase matching.

    Scoring (total ~1.0):
      - Vector score: 0.30 (original embedding similarity)
      - Key-phrase contiguous boost: up to 0.45 (longest match wins)
      - Trigram overlap: up to 0.25 (supplementary)
    """

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        query_tokens = _tokenize_zh(query)
        scored = []
        for hit in hits:
            text_lower = hit.chunk.text.lower()
            pboost = _phrase_boost(query, text_lower)
            text_tokens = _tokenize_zh(hit.chunk.text)
            overlap = len(query_tokens & text_tokens)
            match_ratio = overlap / max(len(query_tokens), 1)
            new_score = hit.score * 0.30 + pboost + match_ratio * 0.25
            scored.append((new_score, hit))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for rank, (score, hit) in enumerate(scored[:top_k]):
            hit.score = score
            hit.rank = rank
            hit.retriever = "rerank"
            result.append(hit)
        return result


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker (requires sentence-transformers)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("pip install sentence-transformers")
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        model = self._get_model()
        pairs = [(query, h.chunk.text) for h in hits]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        result = []
        for rank, (score, hit) in enumerate(ranked[:top_k]):
            hit.score = float(score)
            hit.rank = rank
            hit.retriever = "rerank"
            result.append(hit)
        return result
