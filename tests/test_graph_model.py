from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from src.api.app import app
from src.api.dependencies import get_model_registry
from src.chat.model_registry import ModelRegistry


@pytest.fixture
def graph_client(tmp_path, monkeypatch):
    from src.api.routes import knowledge as knowledge_routes

    registry = ModelRegistry(
        custom_path=tmp_path / "custom_models.json",
        secret_path=tmp_path / ".env.test",
    )
    monkeypatch.setattr(knowledge_routes, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        knowledge_routes,
        "_DATA_FILE",
        tmp_path / "data" / "knowledge_bases.json",
    )
    monkeypatch.setattr(
        knowledge_routes,
        "_MOCK_KBS",
        [
            {
                "id": "kb_graph",
                "name": "Graph KB",
                "description": "test kb",
                "status": "ready",
                "owner_id": "user_001",
                "doc_count": 0,
                "chunk_count": 0,
            }
        ],
    )
    monkeypatch.setattr(knowledge_routes, "_MOCK_DOCS", [])
    monkeypatch.setattr(knowledge_routes, "_MOCK_JOBS", {})

    app.dependency_overrides[get_model_registry] = lambda: registry
    with TestClient(app) as client:
        yield client, registry, knowledge_routes
    app.dependency_overrides.clear()


def test_update_graph_model_persists_safe_model_metadata(graph_client):
    client, _, knowledge_routes = graph_client

    response = client.patch("/api/graphs/kb_graph/model", json={"model_id": "mock-chat"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["kb_id"] == "kb_graph"
    assert payload["model"]["id"] == "mock-chat"
    assert knowledge_routes._MOCK_KBS[0]["graph_model_id"] == "mock-chat"
    assert "api_key" not in response.text.lower()


def test_get_graph_uses_selected_model_and_does_not_expose_secrets(graph_client):
    client, _, knowledge_routes = graph_client

    response = client.get("/api/graphs/kb_graph?model_id=mock-chat")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["id"] == "mock-chat"
    assert payload["build"]["model_id"] == "mock-chat"
    assert knowledge_routes._MOCK_KBS[0]["graph_model_id"] == "mock-chat"
    assert "api_key" not in response.text.lower()
    assert "api_key_env" not in response.text.lower()


def test_get_graph_rejects_unknown_model(graph_client):
    client, _, _ = graph_client

    response = client.get("/api/graphs/kb_graph?model_id=missing-model")

    assert response.status_code == 422
    assert "missing-model" in response.text


def test_get_graph_builds_nodes_from_uploaded_document(graph_client):
    client, _, _ = graph_client

    upload = client.post(
        "/api/knowledge-bases/kb_graph/documents",
        files={
            "file": (
                "graph.txt",
                b"RAG depends on Retrieval. Retrieval contains Vector Search.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 201

    response = client.get("/api/graphs/kb_graph?model_id=mock-chat")

    assert response.status_code == 200
    payload = response.json()
    assert payload["build"]["mode"] == "rule_based"
    assert payload["build"]["entity_count"] >= 3
    assert payload["build"]["relationship_count"] >= 2
    assert {node["name"] for node in payload["nodes"]} >= {
        "RAG",
        "Retrieval",
        "Vector Search",
    }


def test_get_graph_calls_selected_non_mock_model(graph_client, monkeypatch):
    client, _, _ = graph_client
    calls = []

    class FakeLLM:
        def invoke(self, prompt):
            calls.append(prompt)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {"name": "Model Extracted Entity", "type": "concept"},
                            {"name": "Selected Graph Model", "type": "model"},
                        ],
                        "relationships": [
                            {
                                "source": "Selected Graph Model",
                                "target": "Model Extracted Entity",
                                "relation": "extracts",
                            }
                        ],
                    }
                )
            )

    def fake_create_chat_model(self, model_config=None, **kwargs):
        assert model_config.id == "custom-graph-model"
        assert kwargs["temperature"] == 0.0
        return FakeLLM()

    monkeypatch.setattr(
        "src.api.routes.graph.LLMClientFactory.create_chat_model",
        fake_create_chat_model,
    )
    created = client.post(
        "/api/chat/models",
        json={
            "id": "custom-graph-model",
            "provider": "deepseek",
            "display_name": "Graph DeepSeek",
            "model_name": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "secret-value",
        },
    )
    assert created.status_code == 201
    upload = client.post(
        "/api/knowledge-bases/kb_graph/documents",
        files={
            "file": (
                "graph.txt",
                b"RAG depends on Retrieval.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 201

    response = client.get("/api/graphs/kb_graph?model_id=custom-graph-model")

    assert response.status_code == 200
    payload = response.json()
    assert calls
    assert payload["model"]["id"] == "custom-graph-model"
    assert payload["build"]["mode"] == "llm"
    assert payload["build"]["relationship_count"] == 1
    assert {node["name"] for node in payload["nodes"]} == {
        "Model Extracted Entity",
        "Selected Graph Model",
    }
    assert "secret-value" not in response.text


def test_llm_graph_normalizes_entity_types(graph_client, monkeypatch):
    client, _, _ = graph_client

    class FakeLLM:
        def invoke(self, _prompt):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {"name": "星河计划", "type": "project"},
                            {"name": "李明", "type": "project"},
                            {"name": "128万元", "type": "project"},
                            {"name": "2026年10月15日", "type": "concept"},
                        ],
                        "relationships": [
                            {"source": "李明", "target": "星河计划", "relation": "负责"},
                            {"source": "星河计划", "target": "128万元", "relation": "预算为"},
                            {"source": "星河计划", "target": "2026年10月15日", "relation": "验收日期为"},
                        ],
                    }
                )
            )

    monkeypatch.setattr(
        "src.api.routes.graph.LLMClientFactory.create_chat_model",
        lambda *_args, **_kwargs: FakeLLM(),
    )
    created = client.post(
        "/api/chat/models",
        json={
            "id": "typed-graph-model",
            "provider": "deepseek",
            "display_name": "Typed Graph Model",
            "model_name": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "secret-value",
        },
    )
    assert created.status_code == 201
    upload = client.post(
        "/api/knowledge-bases/kb_graph/documents",
        files={"file": ("graph.txt", b"graph text", "text/plain")},
    )
    assert upload.status_code == 201

    response = client.get("/api/graphs/kb_graph?model_id=typed-graph-model")

    assert response.status_code == 200
    types = {node["name"]: node["type"] for node in response.json()["nodes"]}
    assert types["星河计划"] == "project"
    assert types["李明"] == "person"
    assert types["128万元"] == "metric"
    assert types["2026年10月15日"] == "metric"
