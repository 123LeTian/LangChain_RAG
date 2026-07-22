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
    registry = ModelRegistry(custom_path=tmp_path / "custom_models.json")
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
    assert data["default_model_id"] == "mock-chat"
    ids = [model["id"] for model in data["models"]]
    assert ids == ["mock-chat"]


def test_list_chat_models_does_not_expose_api_keys(client):
    response = client.get("/api/chat/models")

    assert response.status_code == 200
    payload = response.text
    assert "api_key" not in payload.lower()
    assert "DEEPSEEK_API_KEY" not in payload
    assert "OPENAI_API_KEY" not in payload


def test_model_management_does_not_expose_api_key_env(client, monkeypatch):
    monkeypatch.setenv("CUSTOM_MODEL_KEY", "secret-value")
    response = client.post(
        "/api/chat/models",
        json={
            "provider": "openai",
            "display_name": "Custom OpenAI",
            "model_name": "custom-openai",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "CUSTOM_MODEL_KEY",
        },
    )

    assert response.status_code == 201
    payload = client.get("/api/chat/models/manage").text
    assert "secret-value" not in payload
    assert "CUSTOM_MODEL_KEY" not in payload
    assert "api_key" not in payload.lower()
    assert "key_configured" in payload


def test_list_chat_models_has_one_default_model(client):
    response = client.get("/api/chat/models")

    defaults = [model for model in response.json()["models"] if model["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == "mock-chat"


def test_custom_model_crud_and_default_model(client):
    created = client.post(
        "/api/chat/models",
        json={
            "provider": "ollama",
            "display_name": "Ollama Qwen",
            "model_name": "custom-qwen-local",
            "base_url": "http://localhost:11434/v1",
        },
    )
    assert created.status_code == 201
    model_id = created.json()["id"]

    updated = client.patch(
        f"/api/chat/models/{model_id}",
        json={"display_name": "Ollama Qwen Local"},
    )
    assert updated.status_code == 200
    assert updated.json()["display_name"] == "Ollama Qwen Local"

    default_response = client.patch(
        "/api/chat/models/default",
        json={"model_id": model_id},
    )
    assert default_response.status_code == 200
    assert default_response.json()["id"] == model_id
    assert client.get("/api/chat/models").json()["default_model_id"] == model_id

    delete_response = client.delete(f"/api/chat/models/{model_id}")
    assert delete_response.status_code == 204
    ids = [model["id"] for model in client.get("/api/chat/models/manage").json()["models"]]
    assert model_id not in ids


def test_builtin_model_is_read_only(client):
    response = client.delete("/api/chat/models/mock-chat")

    assert response.status_code == 422


def test_model_connection_reports_missing_key(client, monkeypatch):
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)
    created = client.post(
        "/api/chat/models",
        json={
            "provider": "openai",
            "display_name": "Missing Key Model",
            "model_name": "missing-key-model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "MISSING_MODEL_KEY",
        },
    ).json()
    response = client.post(f"/api/chat/models/{created['id']}/test")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "缺少" in data["message"]


def test_local_model_connection_does_not_require_key(client):
    created = client.post(
        "/api/chat/models",
        json={
            "provider": "ollama",
            "display_name": "Local Model",
            "model_name": "local-model",
            "base_url": "http://localhost:11434/v1",
        },
    ).json()

    response = client.post(f"/api/chat/models/{created['id']}/test")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_update_session_model(client):
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
    created = client.post(
        "/api/chat/models",
        json={
            "provider": "ollama",
            "display_name": "Local Test Model",
            "model_name": "local-test-model",
            "base_url": "http://localhost:11434/v1",
        },
    ).json()

    response = client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": created["id"]},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == created["id"]
    assert client.get(f"/api/chat/sessions/{session['id']}").json()["model_id"] == created["id"]


def test_update_missing_session_model_returns_404(client):
    response = client.patch(
        "/api/chat/sessions/missing/model",
        json={"model_id": "mock-chat"},
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
    created = client.post(
        "/api/chat/models",
        json={
            "provider": "ollama",
            "display_name": "Disabled Local Model",
            "model_name": "disabled-local-model",
            "base_url": "http://localhost:11434/v1",
            "enabled": False,
        },
    ).json()

    response = client.patch(
        f"/api/chat/sessions/{session['id']}/model",
        json={"model_id": created["id"]},
    )

    assert response.status_code == 422
