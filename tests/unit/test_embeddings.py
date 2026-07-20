"""Embedding 适配器的离线行为测试。"""

import ast
import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest

from src.retrieval import HashEmbedder, HuggingFaceEmbedder, OpenAIEmbedder
from src.retrieval import embeddings as embeddings_module


def test_embeddings_module_imports_without_optional_dependencies():
    module = importlib.import_module("src.retrieval.embeddings")

    assert module.HashEmbedder is HashEmbedder
    assert module.HuggingFaceEmbedder is HuggingFaceEmbedder
    assert module.OpenAIEmbedder is OpenAIEmbedder


def test_huggingface_embedder_has_one_authoritative_definition():
    source_path = Path(embeddings_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HuggingFaceEmbedder"
    ]

    assert len(definitions) == 1


def test_huggingface_embedder_constructor_is_lazy(monkeypatch):
    real_import = builtins.__import__

    def reject_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise AssertionError("constructor must not import sentence-transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sentence_transformers)
    embedder = HuggingFaceEmbedder(
        model_name="local/test-model",
        device="cpu",
        batch_size=8,
    )

    assert embedder.model_name == "local/test-model"
    assert embedder.device == "cpu"
    assert embedder.batch_size == 8
    assert embedder._model is None


def test_huggingface_embedder_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = HuggingFaceEmbedder()

    with pytest.raises(ImportError, match="sentence-transformers"):
        embedder.embed_texts(["离线测试"])


def test_huggingface_embedder_passes_device_and_batch_size(monkeypatch):
    calls = {}

    class FakeEmbeddings(list):
        def tolist(self):
            return list(self)

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["init_options"] = kwargs

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["encode_options"] = kwargs
            return FakeEmbeddings([[1.0, 0.0, 0.0] for _ in texts])

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = HuggingFaceEmbedder(
        model_name="local/test-model",
        device="cpu",
        batch_size=2,
    )
    vectors = embedder.embed_texts(["第一段", "第二段"])

    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert embedder.dimension == 3
    assert calls == {
        "model_name": "local/test-model",
        "init_options": {"device": "cpu"},
        "texts": ["第一段", "第二段"],
        "encode_options": {"batch_size": 2, "normalize_embeddings": True},
    }


def test_hash_and_openai_public_constructors_remain_compatible():
    hash_embedder = HashEmbedder(dim=16)
    openai_embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="not-used")

    assert len(hash_embedder.embed_query("offline")) == 16
    assert openai_embedder.model_name == "text-embedding-3-small"
    assert openai_embedder.dimension == 1536
    assert openai_embedder._client is None


def test_optional_embedders_accept_empty_batch_without_initializing_clients():
    huggingface = HuggingFaceEmbedder()
    openai = OpenAIEmbedder(api_key="not-used")

    assert huggingface.embed_texts([]) == []
    assert openai.embed_texts([]) == []
    assert huggingface._model is None
    assert openai._client is None
