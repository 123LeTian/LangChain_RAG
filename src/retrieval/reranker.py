"""Reranker module with Chinese-aware keyword matching."""
from abc import ABC, abstractmethod
import re
from typing import List
from src.models.schemas import RetrievalHit


def _tokenize_zh(text: str) -> set:
    """Tokenize Chinese+mixed text into meaningful units.

    Uses trigrams for better precision than bigrams:
    "归母净利润" -> {归, 母, 净, 利, 润, 归母, 母净, 净利, 利润, 归母净, 母净利, 净利润}
    """
    tokens = set()
    raw_words = re.findall(r"[\w]+", text.lower())
    for word in raw_words:
        ascii_part = re.findall(r"[a-z0-9]+", word)
        chinese_part = re.findall(r"[\u4e00-\u9fff]+", word)
        tokens.update(ascii_part)
        for seg in chinese_part:
            n = len(seg)
            for i in range(n):
                # unigram
                tokens.add(seg[i])
                # bigram
                if i < n - 1:
                    tokens.add(seg[i:i+2])
                # trigram (key for precision)
                if i < n - 2:
                    tokens.add(seg[i:i+3])
    return tokens


def _find_best_snippet(text: str, query_tokens: set, window: int = 100) -> str:
    """Find the text window with highest token overlap for preview."""
    if not query_tokens or not text:
        return text[:120]
    # Score each position by counting matched token starts
    best_score = 0
    best_pos = 0
    step = 20
    for pos in range(0, max(1, len(text) - window), step):
        snippet = text[pos:pos + window]
        snip_tokens = _tokenize_zh(snippet)
        score = len(query_tokens & snip_tokens)
        if score > best_score:
            best_score = score
            best_pos = pos
    start = max(0, best_pos)
    end = min(len(text), best_pos + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ") + suffix


class BaseReranker(ABC):
    """Reranker abstract base."""

    @abstractmethod
    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        ...


class SimpleReranker(BaseReranker):
    """Simple reranker with Chinese trigram matching + exact phrase boost.

    Scoring:
      - Exact phrase match: +0.3 (huge boost)
      - Trigram overlap ratio: up to 0.3
      - Original vector score: 0.4
    """

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        query_tokens = _tokenize_zh(query)
        scored = []
        for hit in hits:
            text_lower = hit.chunk.text.lower()
            # Exact phrase boost
            phrase_boost = 0.0
            if query.lower() in text_lower:
                phrase_boost = 0.3
            # Also check if key 3+ char substrings appear
            else:
                # Check for important substrings of length 3+
                chinese_segs = re.findall(r"[\u4e00-\u9fff]+", query)
                for seg in chinese_segs:
                    if len(seg) >= 3 and seg in text_lower:
                        phrase_boost = max(phrase_boost, 0.15)
                    elif len(seg) >= 2 and seg in text_lower:
                        phrase_boost = max(phrase_boost, 0.08)

            # Token overlap
            text_tokens = _tokenize_zh(hit.chunk.text)
            overlap = len(query_tokens & text_tokens)
            match_ratio = overlap / max(len(query_tokens), 1)

            # Final score: vector + keyword + phrase boost
            new_score = hit.score * 0.4 + match_ratio * 0.3 + phrase_boost

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