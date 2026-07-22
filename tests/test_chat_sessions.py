from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_message_service,
    get_chat_session_service,
)
from src.chat.message_service import MessageService
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service

    with TestClient(app) as test_client:
        yield test_client, store

    app.dependency_overrides.clear()


def test_create_session(client):
    test_client, _ = client

    response = test_client.post(
        "/api/chat/sessions",
        json={
            "title": "新对话",
            "model_id": "deepseek-chat",
            "preset_id": None,
            "rag_mode": "advanced",
            "knowledge_base_id": None,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["id"].startswith("session_")
    assert data["title"] == "新对话"
    assert data["model_id"] == "deepseek-chat"
    assert data["rag_mode"] == "advanced"
    assert data["total_prompt_tokens"] == 0
    assert data["total_completion_tokens"] == 0
    assert "T" in data["created_at"]
    assert "T" in data["updated_at"]


def test_create_session_accepts_empty_body(client):
    test_client, _ = client

    response = test_client.post("/api/chat/sessions")

    assert response.status_code == 201
    assert response.json()["title"] == "新对话"


def test_list_sessions_orders_by_updated_at_desc(client):
    test_client, _ = client

    first = test_client.post("/api/chat/sessions", json={"title": "first"}).json()
    second = test_client.post("/api/chat/sessions", json={"title": "second"}).json()
    test_client.post(
        f"/api/chat/sessions/{first['id']}/messages",
        json={"role": "user", "content": "hello"},
    )

    response = test_client.get("/api/chat/sessions")

    assert response.status_code == 200
    sessions = response.json()
    assert [item["id"] for item in sessions] == [first["id"], second["id"]]


def test_get_single_session(client):
    test_client, _ = client
    created = test_client.post("/api/chat/sessions", json={"title": "details"}).json()

    response = test_client.get(f"/api/chat/sessions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert response.json()["title"] == "details"


def test_rename_session(client):
    test_client, _ = client
    created = test_client.post("/api/chat/sessions", json={"title": "old"}).json()

    response = test_client.patch(
        f"/api/chat/sessions/{created['id']}",
        json={"title": "新的标题"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "新的标题"
    assert data["updated_at"] >= created["updated_at"]


def test_update_session_config_fields(client):
    test_client, _ = client
    created = test_client.post(
        "/api/chat/sessions",
        json={
            "title": "config",
            "model_id": "mock-chat",
            "preset_id": "default-assistant",
            "rag_mode": "naive",
            "knowledge_base_id": "kb_1",
        },
    ).json()

    response = test_client.patch(
        f"/api/chat/sessions/{created['id']}",
        json={
            "model_id": "mock-chat",
            "preset_id": None,
            "rag_mode": "advanced",
            "knowledge_base_id": None,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "config"
    assert data["model_id"] == "mock-chat"
    assert data["preset_id"] is None
    assert data["rag_mode"] == "advanced"
    assert data["knowledge_base_id"] is None
    assert data["updated_at"] >= created["updated_at"]


def test_delete_session(client):
    test_client, _ = client
    created = test_client.post("/api/chat/sessions", json={"title": "delete me"}).json()

    response = test_client.delete(f"/api/chat/sessions/{created['id']}")

    assert response.status_code == 204
    assert test_client.get(f"/api/chat/sessions/{created['id']}").status_code == 404


def test_save_message_updates_session_totals(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "role": "user",
            "content": "什么是 RAG？",
            "citations": [{"chunk_id": "chunk-1"}],
            "trace": [{"stage": "save"}],
            "prompt_tokens": 7,
            "completion_tokens": 0,
            "latency_ms": 12,
        },
    )

    assert response.status_code == 201
    message = response.json()
    assert message["id"].startswith("message_")
    assert message["session_id"] == session["id"]
    assert message["role"] == "user"
    assert message["content"] == "什么是 RAG？"
    assert message["citations"] == [{"chunk_id": "chunk-1"}]
    assert message["trace"] == [{"stage": "save"}]

    updated_session = test_client.get(f"/api/chat/sessions/{session['id']}").json()
    assert updated_session["total_prompt_tokens"] == 7
    assert updated_session["total_completion_tokens"] == 0
    assert updated_session["updated_at"] >= session["updated_at"]


def test_load_messages(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()
    test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"role": "system", "content": "rules"},
    )
    test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"role": "assistant", "content": "answer"},
    )

    response = test_client.get(f"/api/chat/sessions/{session['id']}/messages")

    assert response.status_code == 200
    assert [item["role"] for item in response.json()] == ["system", "assistant"]
    assert [item["content"] for item in response.json()] == ["rules", "answer"]


def test_delete_session_cascades_messages(client):
    test_client, store = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()
    test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"role": "user", "content": "hello"},
    )

    assert len(store.list_messages(session["id"])) == 1
    response = test_client.delete(f"/api/chat/sessions/{session['id']}")

    assert response.status_code == 204
    assert store.list_messages(session["id"]) == []


def test_missing_session_returns_404(client):
    test_client, _ = client

    assert test_client.get("/api/chat/sessions/missing").status_code == 404
    assert (
        test_client.post(
            "/api/chat/sessions/missing/messages",
            json={"role": "user", "content": "hello"},
        ).status_code
        == 404
    )


def test_empty_message_save_fails(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"role": "user", "content": "   "},
    )

    assert response.status_code == 422


def test_invalid_role_fails(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = test_client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"role": "tool", "content": "hello"},
    )

    assert response.status_code == 422


def test_sqlite_schema_is_created_on_first_call(client):
    test_client, store = client

    assert test_client.get("/api/chat/sessions").status_code == 200

    with sqlite3.connect(store.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"chat_sessions", "chat_messages"}.issubset(tables)
