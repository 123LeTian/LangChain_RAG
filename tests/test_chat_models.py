from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_application_service,
    get_chat_message_service,
    get_chat_session_service,
    get_model_registry,
)
from src.chat.chat_application_service import ChatApplicationService
from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.rag_gateway import RAGGateway
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


class DummyRAGService:
    async def query(self, request):
        raise AssertionError("model tests should not call RAG")


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    memory_service = MemoryService(message_service)
    registry = ModelRegistry()
    app_service = ChatApplicationService(
        session_service,
        message_service,
        memory_service,
        RAGGateway(DummyRAGService()),
        registry,
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service
    app.dependency_overrides[get_model_registry] = lambda: registry

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_list_chat_models_returns_enabled_models(client):
    response = client.get("/api/chat/models")

    assert response.status_code == 200
    data = response.json()
    assert data["default_model_id"] == "deepseek-chat"
    ids = [model["id"] for model in data["models"]]
    assert "deepseek-chat" in ids
    assert "deepseek-reasoner" in ids
    assert "gpt-4o" not in ids


def test_list_chat_models_does_not_expose_api_keys(client):
    response = client.get("/api/chat/models")

    assert response.status_code == 200
    payload = response.text
    assert "api_key" not in payload.lower()
    assert "DEEPSEEK_API_KEY" not in payload
    assert "OPENAI_API_KEY" not in payload


def test_list_chat_models_has_one_default_model(client):
    response = client.get("/api/chat/models")

    defaults = [model for model in response.json()["models"] if model["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == "deepseek-chat"


def test_update_session_model(client):
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": "deepseek-reasoner"},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "deepseek-reasoner"
    assert client.get(f"/api/chat/sessions/{session['id']}").json()["model_id"] == "deepseek-reasoner"


def test_update_missing_session_model_returns_404(client):
    response = client.patch(
        "/api/chat/sessions/missing/model",
        json={"model_id": "deepseek-chat"},
    )

    assert response.status_code == 404


def test_update_unknown_model_returns_validation_error(client):
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": "missing-model"},
    )

    assert response.status_code == 422


def test_update_disabled_model_returns_validation_error(client):
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": "gpt-4o"},
    )

    assert response.status_code == 422
