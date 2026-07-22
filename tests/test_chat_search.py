from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_message_service,
    get_chat_search_service,
    get_chat_session_service,
)
from src.chat.message_service import MessageService
from src.chat.search_service import SearchService
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    search_service = SearchService(store)

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_search_service] = lambda: search_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _session(client: TestClient, title: str):
    return client.post("/api/chat/sessions", json={"title": title}).json()


def _message(client: TestClient, session_id: str, role: str, content: str):
    return client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"role": role, "content": content},
    ).json()


def test_empty_search_query_returns_validation_error(client):
    response = client.get("/api/chat/search", params={"q": "   "})

    assert response.status_code == 422


def test_search_user_message(client):
    session = _session(client, "RAG intro")
    _message(client, session["id"], "user", "Please explain retrieval augmented generation")

    response = client.get("/api/chat/search", params={"q": "retrieval"})

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["role"] == "user"
    assert data["items"][0]["session_title"] == "RAG intro"


def test_search_assistant_message(client):
    session = _session(client, "answer session")
    _message(client, session["id"], "assistant", "RAG combines retrieval with generation.")

    response = client.get("/api/chat/search", params={"q": "combines"})

    assert response.status_code == 200
    assert response.json()["items"][0]["role"] == "assistant"


def test_search_supports_session_filter(client):
    first = _session(client, "first")
    second = _session(client, "second")
    _message(client, first["id"], "user", "shared keyword from first")
    _message(client, second["id"], "user", "shared keyword from second")

    response = client.get(
        "/api/chat/search",
        params={"q": "shared", "session_id": second["id"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["session_id"] == second["id"]


def test_search_supports_role_filter(client):
    session = _session(client, "roles")
    _message(client, session["id"], "user", "needle from user")
    _message(client, session["id"], "assistant", "needle from assistant")

    response = client.get(
        "/api/chat/search",
        params={"q": "needle", "role": "assistant"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["role"] == "assistant"


def test_search_snippet_is_short_and_escaped(client):
    session = _session(client, "html")
    _message(client, session["id"], "user", "<script>alert('x')</script> " + ("keyword " * 40))

    response = client.get("/api/chat/search", params={"q": "keyword"})

    snippet = response.json()["items"][0]["snippet"]
    assert len(snippet) <= 120
    assert "<script>" not in snippet
    assert "&lt;script&gt;" in snippet or "keyword" in snippet


def test_search_results_order_by_created_at_desc(client):
    session = _session(client, "order")
    first = _message(client, session["id"], "user", "ordered keyword first")
    second = _message(client, session["id"], "assistant", "ordered keyword second")

    response = client.get("/api/chat/search", params={"q": "ordered"})

    items = response.json()["items"]
    assert [item["message_id"] for item in items] == [second["id"], first["id"]]
