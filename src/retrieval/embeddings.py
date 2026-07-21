"""Embedding adapters with a shared, offline-safe compatibility contract."""

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List, Optional, Sequence

import numpy as np


def _validate_texts(texts: List[str]) -> None:
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a string")
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be empty")


def _validated_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> List[List[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    result: List[List[float]] = []
    dimension: Optional[int] = None
    for index, raw_vector in enumerate(vectors):
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Embedding {index} contains non-numeric values") from exc
        if not vector:
            raise ValueError(f"Embedding {index} must not be empty")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Embedding {index} must contain only finite values")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("Embedding vectors must have one consistent dimension")
        result.append(vector)
    return result


class BaseEmbedder(ABC):
    """Common embedding contract used by indexing and retrieval."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch while preserving its input order."""

    def embed(self, text: str) -> List[float]:
        """Embed one non-empty text using the batch implementation."""

        _validate_texts([text])
        return self.embed_texts([text])[0]

    def embed_query(self, text: str) -> List[float]:
        """Compatibility alias for :meth:`embed`."""

        return self.embed(text)


class HuggingFaceEmbedder(BaseEmbedder):
    """Lazily initialized local sentence-transformers embedder."""

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
        _validate_texts(texts)
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        raw_vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        vectors = _validated_vectors(raw_vectors, len(texts))
        self._dim = len(vectors[0])
        return vectors


class HashEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder for tests and demos."""

    def __init__(self, dim: int = 256):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "hash-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        _validate_texts(texts)
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.zeros(self._dim, dtype=np.float32)
            for index in range(self._dim):
                byte = digest[index % len(digest)]
                vector[index] = (float(byte) - 127.5) / 127.5
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.tolist())
        return _validated_vectors(vectors, len(texts)) if vectors else []


class OpenAIEmbedder(BaseEmbedder):
    """Lazily initialized OpenAI embedding adapter."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
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
            except ImportError as exc:
                raise ImportError("请安装 openai: pip install openai") from exc
            import os

            key = self._api_key or os.getenv("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        _validate_texts(texts)
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=texts)
        vectors = _validated_vectors(
            [item.embedding for item in response.data],
            len(texts),
        )
        self._dim = len(vectors[0])
        return vectors


__all__ = [
    "BaseEmbedder",
    "HashEmbedder",
    "HuggingFaceEmbedder",
    "OpenAIEmbedder",
]
