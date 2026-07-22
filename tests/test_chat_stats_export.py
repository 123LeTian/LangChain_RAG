from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_export_service,
    get_chat_message_service,
    get_chat_session_service,
    get_chat_token_service,
)
from src.chat.export_service import ExportService
from src.chat.message_service import MessageService
from src.chat.session_service import SessionService
from src.chat.token_service import TokenService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    token_service = TokenService(store)
    export_service = ExportService(session_service, message_service, token_service)

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_token_service] = lambda: token_service
    app.dependency_overrides[get_chat_export_service] = lambda: export_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _session(client: TestClient, title: str = "chat"):
    return client.post("/api/chat/sessions", json={"title": title}).json()


def _message(
    client: TestClient,
    session_id: str,
    role: str,
    content: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    citations=None,
    trace=None,
    latency_ms: int = 0,
):
    payload = {
        "role": role,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
    }
    if citations is not None:
        payload["citations"] = citations
    if trace is not None:
        payload["trace"] = trace
    return client.post(f"/api/chat/sessions/{session_id}/messages", json=payload).json()


def test_global_stats_counts_sessions_messages_and_tokens(client):
    first = _session(client, "first")
    second = _session(client, "second")
    _message(client, first["id"], "user", "hello", prompt_tokens=3)
    _message(client, first["id"], "assistant", "answer", completion_tokens=7)
    _message(client, second["id"], "assistant", "answer", prompt_tokens=2, completion_tokens=5)

    response = client.get("/api/chat/stats")

    assert response.status_code == 200
    assert response.json() == {
        "sessions_count": 2,
        "messages_count": 3,
        "prompt_tokens": 5,
        "completion_tokens": 12,
        "total_tokens": 17,
    }


def test_session_stats(client):
    session = _session(client)
    _message(client, session["id"], "user", "q", prompt_tokens=4)
    _message(client, session["id"], "assistant", "a", completion_tokens=9)

    response = client.get(f"/api/chat/sessions/{session['id']}/stats")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session["id"],
        "prompt_tokens": 4,
        "completion_tokens": 9,
        "total_tokens": 13,
        "messages_count": 2,
    }


def test_empty_session_stats_are_zero(client):
    session = _session(client)

    response = client.get(f"/api/chat/sessions/{session['id']}/stats")

    assert response.status_code == 200
    assert response.json()["messages_count"] == 0
    assert response.json()["total_tokens"] == 0


def test_assistant_message_token_fields_are_counted(client):
    session = _session(client)
    _message(
        client,
        session["id"],
        "assistant",
        "a",
        prompt_tokens=11,
        completion_tokens=17,
    )

    stats = client.get(f"/api/chat/sessions/{session['id']}/stats").json()

    assert stats["prompt_tokens"] == 11
    assert stats["completion_tokens"] == 17


def test_export_existing_session(client):
    session = _session(client, "RAG 技术介绍")
    _message(client, session["id"], "user", "什么是 RAG？", prompt_tokens=6)
    _message(
        client,
        session["id"],
        "assistant",
        "RAG 是检索增强生成。",
        completion_tokens=12,
        citations=[{"filename": "doc.md", "chunk_id": "chunk-1", "score": 0.91, "quote": "RAG definition"}],
        trace=[{"stage": "retrieve"}, {"stage": "generate"}],
        latency_ms=88,
    )

    response = client.get(f"/api/chat/sessions/{session['id']}/export")

    assert response.status_code == 200
    data = response.json()
    assert data["filename"].endswith(".md")
    assert "# RAG 技术介绍" in data["content"]
    assert "## User" in data["content"]
    assert "什么是 RAG？" in data["content"]
    assert "## Assistant" in data["content"]
    assert "RAG 是检索增强生成。" in data["content"]
    assert "### 引用来源" in data["content"]
    assert "doc.md / chunk chunk-1" in data["content"]
    assert "阶段数: 2" in data["content"]


def test_export_missing_session_returns_404(client):
    response = client.get("/api/chat/sessions/missing/export")

    assert response.status_code == 404


def test_export_empty_session(client):
    session = _session(client, "empty")

    response = client.get(f"/api/chat/sessions/{session['id']}/export")

    assert response.status_code == 200
    assert "暂无消息" in response.json()["content"]


def test_export_redacts_sensitive_fields(client):
    session = _session(client, "secret")
    _message(
        client,
        session["id"],
        "assistant",
        "api_key_env=DEEPSEEK_API_KEY and api_key=abc123",
    )

    content = client.get(f"/api/chat/sessions/{session['id']}/export").json()["content"]

    assert "DEEPSEEK_API_KEY" not in content
    assert "api_key_env" not in content
    assert "abc123" not in content


def test_export_filename_is_safe(client):
    session = _session(client, 'bad:/\\*?"<>| name')

    filename = client.get(f"/api/chat/sessions/{session['id']}/export").json()["filename"]

    assert filename.endswith(".md")
    assert not re.search(r'[<>:"/\\|?*]', filename)
