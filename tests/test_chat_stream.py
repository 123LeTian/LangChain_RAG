from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_application_service,
    get_chat_message_service,
    get_chat_session_service,
)
from src.chat.chat_application_service import ChatApplicationService
from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


class FakeRAGGateway:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.message_count_at_call = 0
        self._message_service = None

    def bind_message_service(self, message_service: MessageService) -> None:
        self._message_service = message_service

    async def stream(self, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        self.calls.append(kwargs)
        if self._message_service is not None:
            self.message_count_at_call = len(
                self._message_service.list_messages(kwargs["session_id"])
            )
        trace = [{"stage": "retrieve", "output_summary": "fake trace"}]
        citations = [{"document_id": "doc-1", "chunk_id": "chunk-1"}]
        yield {"type": "trace", "trace": trace}
        yield {"type": "chunk", "content": "Mock "}
        yield {"type": "chunk", "content": "answer"}
        yield {"type": "citation", "citations": citations}
        yield {
            "type": "done",
            "content": "Mock answer",
            "citations": citations,
            "trace": trace,
            "latency_ms": 25,
            "prompt_tokens": 11,
            "completion_tokens": 5,
        }


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    memory_service = MemoryService(message_service)
    model_registry = ModelRegistry(custom_path=tmp_path / "custom_models.json")
    gateway = FakeRAGGateway()
    gateway.bind_message_service(message_service)
    app_service = ChatApplicationService(
        session_service,
        message_service,
        memory_service,
        gateway,
        model_registry,
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as test_client:
        yield test_client, gateway

    app.dependency_overrides.clear()


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _create_session(test_client: TestClient, title: str = "新对话") -> Dict[str, Any]:
    response = test_client.post("/api/chat/sessions", json={"title": title})
    assert response.status_code == 201
    return response.json()


def _stream(test_client: TestClient, session_id: str, question: str) -> List[Dict[str, Any]]:
    with test_client.stream(
        "POST",
        f"/api/chat/sessions/{session_id}/stream",
        json={"question": question, "rag_mode": "naive"},
    ) as response:
        assert response.status_code == 200
        return _read_sse(response)


def test_missing_session_stream_returns_404(client):
    test_client, _ = client

    response = test_client.post(
        "/api/chat/sessions/missing/stream",
        json={"question": "hello"},
    )

    assert response.status_code == 404


def test_empty_question_fails_validation(client):
    test_client, _ = client
    session = _create_session(test_client)

    response = test_client.post(
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_user_message_is_saved_before_rag_call(client):
    test_client, gateway = client
    session = _create_session(test_client)

    _stream(test_client, session["id"], "What is RAG?")

    assert gateway.message_count_at_call == 1
    messages = test_client.get(f"/api/chat/sessions/{session['id']}/messages").json()
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is RAG?"


def test_successful_stream_saves_assistant_message(client):
    test_client, _ = client
    session = _create_session(test_client)

    events = _stream(test_client, session["id"], "What is RAG?")

    messages = test_client.get(f"/api/chat/sessions/{session['id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Mock answer"
    assert events[-1]["type"] == "done"
    assert events[-1]["message_id"] == messages[1]["id"]


def test_assistant_message_saves_citations_and_trace(client):
    test_client, _ = client
    session = _create_session(test_client)

    _stream(test_client, session["id"], "What is RAG?")

    messages = test_client.get(f"/api/chat/sessions/{session['id']}/messages").json()
    assistant = messages[1]
    assert assistant["citations"] == [{"document_id": "doc-1", "chunk_id": "chunk-1"}]
    assert assistant["trace"] == [{"stage": "retrieve", "output_summary": "fake trace"}]
    assert assistant["prompt_tokens"] == 11
    assert assistant["completion_tokens"] == 5
    assert assistant["latency_ms"] == 25


def test_default_title_is_updated_from_first_question(client):
    test_client, _ = client
    session = _create_session(test_client)
    question = "什么是检索增强生成系统，它适合解决什么问题？"

    _stream(test_client, session["id"], question)

    updated = test_client.get(f"/api/chat/sessions/{session['id']}").json()
    assert updated["title"] == question[:20]


def test_second_stream_loads_recent_history(client):
    test_client, gateway = client
    session = _create_session(test_client)

    _stream(test_client, session["id"], "First question")
    _stream(test_client, session["id"], "Second question")

    second_call = gateway.calls[-1]
    enhanced = second_call["context_enhanced_question"]
    assert "以下是当前会话的最近历史：" in enhanced
    assert "User: First question" in enhanced
    assert "Assistant: Mock answer" in enhanced
    assert "当前问题：\nSecond question" in enhanced
    assert "User: Second question" not in enhanced


def test_stream_passes_rag_options_to_gateway(client):
    test_client, gateway = client
    session = _create_session(test_client)

    with test_client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={
            "question": "What is RAG?",
            "rag_mode": "advanced",
            "knowledge_base_id": "kb-1",
            "top_k": 3,
            "score_threshold": 0.2,
            "temperature": 0.1,
        },
    ) as response:
        assert response.status_code == 200
        _read_sse(response)

    call = gateway.calls[-1]
    assert call["rag_mode"] == "advanced"
    assert call["knowledge_base_id"] == "kb-1"
    assert call["top_k"] == 3
    assert call["score_threshold"] == 0.2
    assert call["temperature"] == 0.1


def test_stream_passes_request_model_id_to_gateway(client):
    test_client, gateway = client
    session = _create_session(test_client)

    with test_client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "What is RAG?", "model_id": "mock-chat"},
    ) as response:
        assert response.status_code == 200
        _read_sse(response)

    assert gateway.calls[-1]["model_id"] == "mock-chat"


def test_stream_uses_session_model_when_request_omits_model(client):
    test_client, gateway = client
    session = _create_session(test_client)
    response = test_client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": "mock-chat"},
    )
    assert response.status_code == 200

    _stream(test_client, session["id"], "What is RAG?")

    assert gateway.calls[-1]["model_id"] == "mock-chat"


def test_stream_uses_default_model_when_session_has_no_model(client):
    test_client, gateway = client
    session = _create_session(test_client)

    _stream(test_client, session["id"], "What is RAG?")

    assert gateway.calls[-1]["model_id"] == "deepseek-chat"


def test_original_rag_stream_route_still_registered(client):
    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/rag/query/stream" in paths
