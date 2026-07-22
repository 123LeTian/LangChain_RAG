from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_chat_message_service,
    get_chat_session_service,
    get_preset_service,
)
from src.chat.message_service import MessageService
from src.chat.preset_service import DEFAULT_PRESET_ID, PresetService
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


@pytest.fixture
def client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    preset_service = PresetService(store)

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_preset_service] = lambda: preset_service

    with TestClient(app) as test_client:
        yield test_client, store

    app.dependency_overrides.clear()


def test_list_presets_returns_system_presets(client):
    test_client, _ = client

    response = test_client.get("/api/chat/presets")

    assert response.status_code == 200
    data = response.json()
    ids = [preset["id"] for preset in data["presets"]]
    assert ids == [DEFAULT_PRESET_ID, "rag-evidence", "engineering-copilot"]


def test_list_presets_has_default_and_hides_system_prompt(client):
    test_client, _ = client

    data = test_client.get("/api/chat/presets").json()
    default = next(
        preset for preset in data["presets"] if preset["id"] == data["default_preset_id"]
    )

    assert data["default_preset_id"] == DEFAULT_PRESET_ID
    assert default["is_default"] is True
    assert default["owner_type"] == "system"
    assert "system_prompt" not in default


def test_create_user_preset(client):
    test_client, _ = client

    response = test_client.post(
        "/api/chat/presets",
        json={
            "name": "Paper Helper",
            "description": "For papers",
            "system_prompt": "Use academic style.",
            "rag_prompt_hint": "Prefer definitions and conclusions.",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["id"].startswith("preset_")
    assert data["owner_type"] == "user"
    assert data["name"] == "Paper Helper"
    assert data["system_prompt"] == "Use academic style."


def test_update_user_preset(client):
    test_client, _ = client
    created = test_client.post(
        "/api/chat/presets",
        json={"name": "Draft", "system_prompt": "Draft answers."},
    ).json()

    response = test_client.patch(
        f"/api/chat/presets/{created['id']}",
        json={"name": "Updated", "system_prompt": "Updated prompt."},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Updated"
    assert response.json()["system_prompt"] == "Updated prompt."


def test_delete_user_preset(client):
    test_client, _ = client
    created = test_client.post(
        "/api/chat/presets",
        json={"name": "Temp", "system_prompt": "Temporary prompt."},
    ).json()

    response = test_client.delete(f"/api/chat/presets/{created['id']}")

    assert response.status_code == 204
    ids = [preset["id"] for preset in test_client.get("/api/chat/presets").json()["presets"]]
    assert created["id"] not in ids


def test_cannot_delete_system_preset(client):
    test_client, _ = client

    response = test_client.delete("/api/chat/presets/rag-evidence")

    assert response.status_code == 422


def test_cannot_update_system_preset(client):
    test_client, _ = client

    response = test_client.patch(
        "/api/chat/presets/rag-evidence",
        json={"name": "Changed"},
    )

    assert response.status_code == 422


def test_update_session_preset(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = test_client.patch(
        f"/api/chat/sessions/{session['id']}/preset",
        json={"preset_id": "rag-evidence"},
    )

    assert response.status_code == 200
    assert response.json()["preset_id"] == "rag-evidence"


def test_missing_preset_returns_404(client):
    test_client, _ = client
    session = test_client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = test_client.patch(
        f"/api/chat/sessions/{session['id']}/preset",
        json={"preset_id": "missing-preset"},
    )

    assert response.status_code == 404


def test_deleting_active_user_preset_resets_session_to_default(client):
    test_client, _ = client
    preset = test_client.post(
        "/api/chat/presets",
        json={"name": "Temp", "system_prompt": "Temporary prompt."},
    ).json()
    session = test_client.post(
        "/api/chat/sessions",
        json={"title": "chat", "preset_id": preset["id"]},
    ).json()

    response = test_client.delete(f"/api/chat/presets/{preset['id']}")

    assert response.status_code == 204
    updated = test_client.get(f"/api/chat/sessions/{session['id']}").json()
    assert updated["preset_id"] == DEFAULT_PRESET_ID


def test_empty_user_preset_fields_fail_validation(client):
    test_client, _ = client

    response = test_client.post(
        "/api/chat/presets",
        json={"name": "   ", "system_prompt": "   "},
    )

    assert response.status_code == 422
