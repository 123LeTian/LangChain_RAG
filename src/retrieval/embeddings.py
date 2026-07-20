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


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace 本地向量化器（中文推荐 bge-small-zh-v1.5）。
    首次运行自动下载模型，之后纯本地推理。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dim = 512

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("请安装: pip install sentence-transformers")
            print(f"  正在加载 Embedding 模型 {self._model_name}...")
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"  模型加载完成，维度: {self._dim}")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()

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


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace 本地向量化器（推荐中文用 bge-small-zh）。
    首次运行会自动下载模型，之后纯本地推理，不调 API。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dim = 512

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("请安装: pip install sentence-transformers")
            print(f"  正在加载 Embedding 模型 {self._model_name}（首次需下载）...")
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"  模型加载完成，维度: {self._dim}")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()

