from __future__ import annotations

import json
from typing import Any, Dict, List

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
from src.chat.rag_gateway import RAGGateway
from src.chat.retry_policy import RetryPolicy
from src.chat.session_service import SessionService
from src.chat.structured_logger import REDACTED, StructuredLogger
from src.chat_storage.sqlite_chat_store import SQLiteChatStore
from src.api.api_models import RAGMode, RAGResult


class SuccessRAGService:
    async def query(self, request):
        return RAGResult(
            answer="ok",
            citations=[],
            hits=[],
            trace=[],
            usage={"prompt_tokens": 3, "completion_tokens": 4},
            mode=RAGMode.NAIVE,
        )


class ErrorRAGService:
    async def query(self, request):
        raise RuntimeError("500 DEEPSEEK_API_KEY failed with api_key=secret-value")


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _client(tmp_path, rag_service):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    logger = StructuredLogger(
        app_log_path=tmp_path / "app.jsonl",
        error_log_path=tmp_path / "error.jsonl",
    )
    gateway = RAGGateway(
        rag_service,
        retry_policy=RetryPolicy(max_retries=0, backoff_seconds=0),
        logger=logger,
    )
    app_service = ChatApplicationService(
        session_service,
        message_service,
        MemoryService(message_service),
        gateway,
        ModelRegistry(),
        logger=logger,
    )
    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service
    return TestClient(app), logger


def test_successful_chat_stream_writes_app_log(tmp_path):
    client, logger = _client(tmp_path, SuccessRAGService())
    try:
        session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": "hello", "rag_mode": "advanced"},
        ) as response:
            events = _read_sse(response)
    finally:
        app.dependency_overrides.clear()

    record = json.loads(logger.app_log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "chat_stream_done"
    assert record["request_id"] == events[-1]["request_id"]
    assert record["session_id"] == session["id"]
    assert record["model_id"] == "deepseek-chat"
    assert record["rag_mode"] == "advanced"


def test_failed_chat_stream_writes_error_log_and_redacts(tmp_path):
    client, logger = _client(tmp_path, ErrorRAGService())
    try:
        session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": "hello"},
        ) as response:
            events = _read_sse(response)
    finally:
        app.dependency_overrides.clear()

    text = logger.error_log_path.read_text(encoding="utf-8")
    record = json.loads(text.splitlines()[-1])
    assert record["event"] == "chat_stream_error"
    assert record["request_id"] == events[-1]["request_id"]
    assert record["session_id"] == session["id"]
    assert "DEEPSEEK_API_KEY" not in text
    assert "secret-value" not in text
    assert REDACTED in text


def test_sensitive_fields_are_redacted(tmp_path):
    logger = StructuredLogger(
        app_log_path=tmp_path / "app.jsonl",
        error_log_path=tmp_path / "error.jsonl",
    )

    logger.info(
        "redaction_test",
        api_key="real-key",
        nested={"Authorization": "Bearer token", "safe": "value"},
    )

    text = logger.app_log_path.read_text(encoding="utf-8")
    assert "real-key" not in text
    assert "Bearer token" not in text
    assert REDACTED in text


def test_log_write_failure_does_not_break_flow(tmp_path):
    logger = StructuredLogger(
        app_log_path=tmp_path,
        error_log_path=tmp_path,
    )

    logger.info("should_not_raise", api_key="secret")
    logger.error("should_not_raise", password="secret")
