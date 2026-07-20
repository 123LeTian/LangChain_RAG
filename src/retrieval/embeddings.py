"""文本向量化模块。"""
from abc import ABC, abstractmethod
from typing import List, Optional
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


class HuggingFaceEmbedder(BaseEmbedder):
    """延迟加载的 HuggingFace 本地向量化器。

    ``sentence-transformers`` 仅在第一次对非空文本执行向量化时导入，
    因此基础检索功能不需要安装该可选依赖。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._dim = 512

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "HuggingFaceEmbedder 需要可选依赖 'sentence-transformers'；"
                    "请执行: pip install -r requirements-bge.txt"
                ) from exc

            model_options = {}
            if self._device is not None:
                model_options["device"] = self._device
            self._model = SentenceTransformer(self._model_name, **model_options)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(embedding) for embedding in embeddings]

    def embed_query(self, query: str) -> List[float]:
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

