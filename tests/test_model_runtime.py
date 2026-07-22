from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.chat.llm_client_factory import LLMClientFactory
from src.chat.model_registry import ChatModel, ModelRegistry
from src.chat.model_runtime import ModelRuntimeError, resolve_runtime_config


def test_registry_default_model_resolves(tmp_path):
    model = ModelRegistry(custom_path=tmp_path / "custom_models.json").default_model()

    assert model.id == "mock-chat"
    assert model.provider == "mock"


def test_missing_api_key_returns_clear_error(monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    model = ChatModel(
        id="deepseek-missing",
        provider="deepseek",
        display_name="DeepSeek Missing",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="MISSING_PROVIDER_KEY",
        enabled=True,
    )

    with pytest.raises(ModelRuntimeError) as exc_info:
        resolve_runtime_config(model)

    message = str(exc_info.value)
    assert "deepseek-missing" in message
    assert "MISSING_PROVIDER_KEY" in message
    assert "secret-value" not in message


def test_disabled_model_cannot_resolve_for_stream():
    model = ChatModel(
        id="gpt-disabled",
        provider="openai",
        display_name="GPT Disabled",
        model_name="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        enabled=False,
    )

    with pytest.raises(ModelRuntimeError) as exc_info:
        resolve_runtime_config(model, require_stream=True)

    assert "未启用" in str(exc_info.value)


def test_ollama_model_allows_missing_api_key():
    model = ChatModel(
        id="ollama-qwen",
        provider="ollama",
        display_name="Local Qwen",
        model_name="qwen2.5",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        enabled=True,
    )

    runtime = resolve_runtime_config(model, require_stream=True)

    assert runtime.api_key is None
    assert runtime.base_url == "http://localhost:11434/v1"


def test_factory_passes_runtime_model_to_chat_openai(monkeypatch):
    calls = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("FAKE_DEEPSEEK_KEY", "secret-value")
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    model = ChatModel(
        id="deepseek-reasoner",
        provider="deepseek",
        display_name="DeepSeek Reasoner",
        model_name="deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
        api_key_env="FAKE_DEEPSEEK_KEY",
        enabled=True,
    )

    LLMClientFactory().create_streaming_chat_model(
        model,
        temperature=0.2,
        max_tokens=300,
    )

    assert calls == [
        {
            "model": "deepseek-reasoner",
            "api_key": "secret-value",
            "base_url": "https://api.deepseek.com/v1",
            "temperature": 0.2,
            "max_tokens": 300,
            "streaming": True,
        }
    ]


def test_public_model_response_does_not_expose_api_key_fields():
    payload = str(ModelRegistry().public_response())

    assert "api_key" not in payload.lower()
    assert "DEEPSEEK_API_KEY" not in payload
