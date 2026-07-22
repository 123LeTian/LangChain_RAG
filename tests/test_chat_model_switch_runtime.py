from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.api_models import RAGMode, RAGRequest, RAGResult
from src.api.app import app
from src.api.dependencies import (
    get_chat_application_service,
    get_chat_message_service,
    get_chat_session_service,
)
from src.api.real_rag_service import RealRAGService
from src.chat.chat_application_service import ChatApplicationService
from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.rag_gateway import RAGGateway
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


class RecordingRAGService:
    def __init__(self):
        self.requests: List[RAGRequest] = []

    async def query(self, request: RAGRequest) -> RAGResult:
        self.requests.append(request)
        return RAGResult(
            answer="runtime answer",
            citations=[],
            hits=[],
            trace=[],
            usage={},
            mode=request.mode,
        )


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def runtime_client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    memory_service = MemoryService(message_service)
    registry = ModelRegistry(custom_path=tmp_path / "custom_models.json")
    registry.create_model({
        "id": "deepseek-reasoner",
        "provider": "deepseek",
        "display_name": "DeepSeek Reasoner",
        "model_name": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "enabled": True,
    })
    registry.create_model({
        "id": "gpt-4o",
        "provider": "openai",
        "display_name": "GPT-4o",
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "enabled": False,
    })
    rag_service = RecordingRAGService()
    app_service = ChatApplicationService(
        session_service,
        message_service,
        memory_service,
        RAGGateway(rag_service),
        registry,
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as test_client:
        yield test_client, rag_service

    app.dependency_overrides.clear()


def test_chat_stream_request_model_reaches_rag_request_options(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={
            "question": "hello",
            "model_id": "deepseek-reasoner",
            "temperature": 0.3,
        },
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    options = rag_service.requests[-1].options
    model_config = options["model"]
    assert model_config["id"] == "deepseek-reasoner"
    assert model_config["model_name"] == "deepseek-reasoner"
    assert options["temperature"] == 0.3
    assert events[-1]["type"] == "done"


def test_chat_stream_uses_session_model_when_request_omits_model(runtime_client):
    client, rag_service = runtime_client
    session = client.post(
        "/api/chat/sessions",
        json={"title": "chat", "model_id": "deepseek-reasoner"},
    ).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello"},
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    assert rag_service.requests[-1].options["model"]["id"] == "deepseek-reasoner"


def test_chat_stream_uses_default_model_without_request_or_session_model(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello"},
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    assert not rag_service.requests
    assert events[-1]["content"] == "您好，我是本地默认模型，不会回答任何问题，只用于测试，请在模型配置页面添加您的模型"
    assert events[-1]["model_id"] == "mock-chat"


def test_chat_rag_request_uses_original_question_not_history_context(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    for question in ["First question", "Second question"]:
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": question, "model_id": "deepseek-reasoner"},
        ) as response:
            assert response.status_code == 200
            _read_sse(response)

    request = rag_service.requests[-1]
    assert request.query == "Second question"
    assert "First question" not in request.query
    assert "最近历史" not in request.query


def test_missing_key_stream_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    RealRAGServiceDouble = type(
        "RealRAGService",
        (),
        {
            "__module__": "src.api.real_rag_service",
            "query": lambda self, _request: None,
        },
    )
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    registry = ModelRegistry(custom_path=tmp_path / "custom_missing_key_models.json")
    registry.create_model({
        "id": "deepseek-chat",
        "provider": "deepseek",
        "display_name": "DeepSeek Chat",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "enabled": True,
    })
    app_service = ChatApplicationService(
        session_service,
        message_service,
        MemoryService(message_service),
        RAGGateway(RealRAGServiceDouble()),
        registry,
    )
    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as client:
        session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": "hello", "model_id": "deepseek-chat"},
        ) as response:
            assert response.status_code == 200
            events = _read_sse(response)

    app.dependency_overrides.clear()
    assert events[-1]["type"] == "error"
    assert "DEEPSEEK_API_KEY" in events[-1]["message"]
    messages = message_service.list_messages(session["id"])
    assert [message.role for message in messages] == ["user"]


def test_disabled_model_stream_returns_validation_error(runtime_client):
    client, _ = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello", "model_id": "gpt-4o"},
    )

    assert response.status_code == 422


def test_model_trace_contains_model_id_provider_but_not_secret(runtime_client):
    client, _ = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello", "model_id": "deepseek-reasoner"},
    ) as response:
        events = _read_sse(response)

    trace_events = [event for event in events if event["type"] == "trace"]
    trace_text = json.dumps(trace_events, ensure_ascii=False)
    assert "deepseek-reasoner" in trace_text
    assert "deepseek" in trace_text
    assert "DEEPSEEK_API_KEY" not in trace_text
    assert "secret-value" not in trace_text


@pytest.mark.asyncio
async def test_unified_rag_service_uses_request_model_config(monkeypatch):
    calls: List[Dict[str, Any]] = []

    class FakeRuntimeLLM:
        async def ainvoke(self, text: str):
            calls.append({"invoke_text": text})
            return type(
                "FakeResponse",
                (),
                {
                    "content": "selected model answer",
                    "usage_metadata": {"input_tokens": 3, "output_tokens": 2},
                },
            )()

    def fake_create_chat_model(self, model_config=None, **kwargs):
        calls.append({"model_config": model_config, "kwargs": kwargs})
        return FakeRuntimeLLM()

    from src.api.api_models import RAGRequest
    from src.api.unified_rag_service import UnifiedRAGApiService
    from src.chat.llm_client_factory import LLMClientFactory

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("FRONTEND_MODEL_KEY", "front-secret")
    monkeypatch.setattr(
        LLMClientFactory,
        "create_chat_model",
        fake_create_chat_model,
    )

    service = UnifiedRAGApiService()
    adapter = service._create_llm_for_request(
        RAGRequest(
            query="What is RAG?",
            mode=RAGMode.NAIVE,
            options={
                "temperature": 0.42,
                "model": {
                    "id": "front-model",
                    "provider": "openai",
                    "model_name": "gpt-front",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "FRONTEND_MODEL_KEY",
                    "enabled": True,
                },
            },
        )
    )

    answer, usage = await adapter.generate_with_tokens("Prompt", "Context")

    assert answer == "selected model answer"
    assert usage == {"input_tokens": 3, "output_tokens": 2}
    assert calls[0]["model_config"]["id"] == "front-model"
    assert calls[0]["model_config"]["api_key_env"] == "FRONTEND_MODEL_KEY"
    assert calls[0]["kwargs"]["temperature"] == 0.42
    assert calls[0]["kwargs"]["max_tokens"] == 1000
    assert adapter.metadata == {
        "model_id": "front-model",
        "provider": "openai",
        "model_name": "gpt-front",
    }


@pytest.mark.asyncio
async def test_old_rag_request_without_model_config_uses_default_deepseek_path(monkeypatch):
    class FakeResponse:
        content = "runtime answer"

    calls: List[Dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def invoke(self, _prompt: str) -> FakeResponse:
            return FakeResponse()

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-real")
    service = RealRAGService()

    result = await service.query(
        RAGRequest(query="hello", mode=RAGMode.NAIVE, options={})
    )

    assert result.answer == "runtime answer"
    assert calls[-1]["model"] == "deepseek-chat"
    assert calls[-1]["base_url"] == "https://api.deepseek.com/v1"
