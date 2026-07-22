from __future__ import annotations

import asyncio
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
from src.chat.model_runtime import ModelRuntimeError
from src.chat.rag_gateway import RAGGateway
from src.chat.retry_policy import ChatRetryExhaustedError, RetryPolicy
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore


@pytest.mark.asyncio
async def test_timeout_error_is_retried():
    calls = 0
    policy = RetryPolicy(timeout_seconds=0.01, max_retries=2, backoff_seconds=0)

    async def operation():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise TimeoutError("temporary timeout")
        return "ok"

    result, retry_count = await policy.run(operation)

    assert result == "ok"
    assert retry_count == 1
    assert calls == 2


@pytest.mark.asyncio
async def test_temporary_network_error_is_retried():
    calls = 0
    policy = RetryPolicy(max_retries=2, backoff_seconds=0)

    async def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("network reset")
        return "ok"

    result, retry_count = await policy.run(operation)

    assert result == "ok"
    assert retry_count == 2
    assert calls == 3


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried():
    calls = 0
    policy = RetryPolicy(max_retries=2, backoff_seconds=0)

    async def operation():
        nonlocal calls
        calls += 1
        raise RuntimeError("authentication failed")

    with pytest.raises(ChatRetryExhaustedError) as exc_info:
        await policy.run(operation)

    assert exc_info.value.retry_count == 0
    assert calls == 1


@pytest.mark.asyncio
async def test_missing_api_key_error_is_not_retried():
    calls = 0
    policy = RetryPolicy(max_retries=2, backoff_seconds=0)

    async def operation():
        nonlocal calls
        calls += 1
        raise ModelRuntimeError("模型 deepseek-chat 缺少环境变量 DEEPSEEK_API_KEY")

    with pytest.raises(ChatRetryExhaustedError) as exc_info:
        await policy.run(operation)

    assert exc_info.value.retry_count == 0
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_exhaustion_reports_retry_count():
    policy = RetryPolicy(max_retries=2, backoff_seconds=0)

    async def operation():
        raise TimeoutError("still slow")

    with pytest.raises(ChatRetryExhaustedError) as exc_info:
        await policy.run(operation)

    assert exc_info.value.retry_count == 2


class FailingRAGService:
    async def query(self, _request):
        raise TimeoutError("slow")


def _test_model_registry(tmp_path) -> ModelRegistry:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  - id: local-test",
                "    provider: local",
                "    display_name: Local Test",
                "    model_name: local-test",
                "    base_url: null",
                "    api_key_env: null",
                "    description: Test-only non-mock model.",
                "    supports_stream: true",
                "    enabled: true",
                "    is_default: true",
            ]
        ),
        encoding="utf-8",
    )
    return ModelRegistry(
        config_path,
        custom_path=tmp_path / "custom_models.json",
        secret_path=tmp_path / ".env.test",
    )


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            import json

            events.append(json.loads(line[len("data: ") :]))
    return events


def test_failed_chat_stream_keeps_user_message(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    gateway = RAGGateway(
        FailingRAGService(),
        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0),
    )
    app_service = ChatApplicationService(
        session_service,
        message_service,
        MemoryService(message_service),
        gateway,
        _test_model_registry(tmp_path),
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as client:
        session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": "hello"},
        ) as response:
            assert response.status_code == 200
            events = _read_sse(response)

    app.dependency_overrides.clear()
    assert events[-1]["type"] == "error"
    assert events[-1]["retry_count"] == 2
    assert events[-1]["request_id"]
    messages = message_service.list_messages(session["id"])
    assert [message.role for message in messages] == ["user"]
