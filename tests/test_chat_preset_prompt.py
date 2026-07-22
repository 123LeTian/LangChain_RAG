from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.api_models import RAGMode, RAGResult
from src.api.app import app
from src.api.dependencies import (
    get_chat_application_service,
    get_chat_message_service,
    get_chat_session_service,
    get_preset_service,
)
from src.chat.chat_application_service import ChatApplicationService
from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.preset_service import DEFAULT_PRESET_ID, PresetService
from src.chat.rag_gateway import RAGGateway
from src.chat.schemas import ChatPreset
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


class FakeRAGGateway:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def stream(self, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        self.calls.append(kwargs)
        yield {"type": "chunk", "content": "ok"}
        yield {
            "type": "done",
            "content": "ok",
            "citations": [],
            "trace": [],
            "latency_ms": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def preset_client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    preset_service = PresetService(store)
    gateway = FakeRAGGateway()
    app_service = ChatApplicationService(
        session_service,
        message_service,
        MemoryService(message_service),
        gateway,
        ModelRegistry(),
        preset_service,
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_preset_service] = lambda: preset_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as test_client:
        yield test_client, gateway

    app.dependency_overrides.clear()


def test_stream_request_preset_reaches_rag_gateway(preset_client):
    client, gateway = preset_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "What is RAG?", "preset_id": "teacher"},
    ) as response:
        assert response.status_code == 200
        _read_sse(response)

    assert gateway.calls[-1]["preset_id"] == "teacher"
    assert gateway.calls[-1]["preset_config"].name == "耐心导师"


def test_stream_uses_session_preset_when_request_omits_preset(preset_client):
    client, gateway = preset_client
    session = client.post(
        "/api/chat/sessions",
        json={"title": "chat", "preset_id": "translator"},
    ).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "What is RAG?"},
    ) as response:
        assert response.status_code == 200
        _read_sse(response)

    assert gateway.calls[-1]["preset_id"] == "translator"


def test_stream_uses_default_preset_without_request_or_session_preset(preset_client):
    client, gateway = preset_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "What is RAG?"},
    ) as response:
        assert response.status_code == 200
        _read_sse(response)

    assert gateway.calls[-1]["preset_id"] == DEFAULT_PRESET_ID


@pytest.mark.asyncio
async def test_rag_gateway_trace_does_not_include_full_prompt():
    long_prompt = "secret-style-" * 40
    preset = ChatPreset(
        id="user-preset",
        name="User Preset",
        system_prompt=long_prompt,
        rag_prompt_hint="short hint",
        owner_type="user",
    )

    class FakeRAGService:
        async def query(self, _request):
            return RAGResult(
                answer="answer",
                citations=[],
                hits=[],
                trace=[],
                usage={},
                mode=RAGMode.NAIVE,
            )

    events = [
        event
        async for event in RAGGateway(FakeRAGService()).stream(
            session_id="session-1",
            original_question="q",
            context_enhanced_question="q",
            preset_id=preset.id,
            preset_config=preset,
        )
    ]

    trace_text = json.dumps([event for event in events if event["type"] == "trace"])
    assert "user-preset" in trace_text
    assert "User Preset" in trace_text
    assert long_prompt not in trace_text


@pytest.mark.asyncio
async def test_rag_gateway_carries_preset_metadata_without_polluting_rag_query():
    preset = ChatPreset(
        id="user-preset",
        name="User Preset",
        system_prompt="Answer with conclusions first.",
        rag_prompt_hint="Separate evidence from uncertainty.",
        owner_type="user",
    )
    requests = []

    class FakeRAGService:
        async def query(self, request):
            requests.append(request)
            return RAGResult(
                answer="answer",
                citations=[],
                hits=[],
                trace=[],
                usage={},
                mode=RAGMode.NAIVE,
            )

    events = [
        event
        async for event in RAGGateway(FakeRAGService()).stream(
            session_id="session-1",
            original_question="What is RAG?",
            context_enhanced_question="history should not pollute retrieval",
            preset_id=preset.id,
            preset_config=preset,
        )
    ]

    assert requests[-1].query == "What is RAG?"
    assert requests[-1].options["preset"]["id"] == "user-preset"
    assert events[-1]["type"] == "done"
