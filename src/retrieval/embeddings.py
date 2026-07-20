"""文本向量化模块。"""
from abc import ABC, abstractmethod
from typing import List
import hashlib
import numpy as np


class BaseEmbedder(ABC):
    """Embedding 抽象基类。"""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """把一组文本变成向量。"""
        ...

    def embed_query(self, query: str) -> List[float]:
        """把单个查询变成向量。"""
        return self.embed_texts([query])[0]


class HashEmbedder(BaseEmbedder):
    """哈希向量器（无需外部依赖，仅用于测试和演示）。
    生产环境请用 OpenAIEmbedder 或 HuggingFaceEmbedder。
    """

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "hash-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, ch in enumerate(h):
                if i >= self._dim:
                    break
                vec[i] = (ord(ch) - 48) / 48.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embedding 向量器。"""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self._model = model
        self._api_key = api_key
        self._client = None
        self._dim = 1536

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
            import os
            key = self._api_key or os.getenv("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        resp = client.embeddings.create(model=self._model, input=texts)
        self._dim = len(resp.data[0].embedding)
        return [d.embedding for d in resp.data]
