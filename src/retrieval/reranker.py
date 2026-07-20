"""重排器模块：对初步检索结果用更精细的模型重新排序。"""
from abc import ABC, abstractmethod
from typing import List
from src.models.schemas import RetrievalHit


class BaseReranker(ABC):
    """重排器抽象基类。"""

    @abstractmethod
    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        """对检索结果重新排序。"""
        ...


class SimpleReranker(BaseReranker):
    """简单重排器：基于关键词匹配加分。无需外部模型。"""

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 5) -> List[RetrievalHit]:
        query_words = set(query.lower().split())
        scored = []
        for hit in hits:
            text_words = set(hit.chunk.text.lower().split())
            overlap = len(query_words & text_words)
            new_score = hit.score + overlap * 0.1
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
    """交叉编码器重排器（需要 sentence-transformers）。"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("请安装: pip install sentence-transformers")
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
