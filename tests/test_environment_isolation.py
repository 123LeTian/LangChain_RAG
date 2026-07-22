from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from scripts.build_check import get_build_log_dir
from src.api.app import app
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore
from src.config.logging_config import get_app_log_path, get_error_log_path
from src.config.runtime_config import get_runtime_config, reset_runtime_config


def _reset_env(monkeypatch, env: str):
    for key in [
        "APP_ENV",
        "CHAT_DB_PATH",
        "APP_LOG_PATH",
        "ERROR_LOG_PATH",
        "BUILD_LOG_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APP_ENV", env)
    reset_runtime_config()
    return get_runtime_config(force_reload=True)


def test_dev_test_prod_storage_paths_are_isolated(monkeypatch):
    dev = _reset_env(monkeypatch, "dev")
    assert "data/dev/chat/chat.db" in dev.chat_db_path.as_posix()
    assert "data/dev/knowledge" in dev.knowledge_dir.as_posix()
    assert "data/dev/vector" in dev.vector_dir.as_posix()

    test = _reset_env(monkeypatch, "test")
    assert "data/test/chat/chat.db" in test.chat_db_path.as_posix()
    assert "data/test/knowledge" in test.knowledge_dir.as_posix()
    assert "data/test/vector" in test.vector_dir.as_posix()

    prod = _reset_env(monkeypatch, "prod")
    assert "data/prod/chat/chat.db" in prod.chat_db_path.as_posix()
    assert "data/prod/knowledge" in prod.knowledge_dir.as_posix()
    assert "data/prod/vector" in prod.vector_dir.as_posix()


def test_logs_and_build_logs_are_isolated_by_env(monkeypatch):
    config = _reset_env(monkeypatch, "test")

    assert get_app_log_path() == config.app_log_path
    assert get_error_log_path() == config.error_log_path
    assert get_build_log_dir() == config.build_log_dir
    assert "logs/test/app.jsonl" in get_app_log_path().as_posix()
    assert "logs/test/build" in get_build_log_dir().as_posix()


def test_test_session_write_does_not_touch_dev_database(monkeypatch, tmp_path):
    dev_db = tmp_path / "data" / "dev" / "chat" / "chat.db"
    test_db = tmp_path / "data" / "test" / "chat" / "chat.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CHAT_DB_PATH", str(test_db))
    reset_runtime_config()

    store = SQLiteChatStore(get_runtime_config(force_reload=True).chat_db_path)
    service = SessionService(store)
    session = service.create_session()

    assert session.id.startswith("session_")
    assert test_db.exists()
    assert not dev_db.exists()
    with sqlite3.connect(test_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
    assert count == 1


def test_runtime_config_api_does_not_leak_secret_values(monkeypatch):
    _reset_env(monkeypatch, "prod")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "prod-secret-value")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:secret@localhost/db")
    reset_runtime_config()

    with TestClient(app) as client:
        response = client.get("/api/config/runtime")

    assert response.status_code == 200
    text = response.text
    assert "prod-secret-value" not in text
    assert "DATABASE_URL" not in text
    assert "postgres://" not in text
    assert "secret" not in text.lower()
    assert response.json()["env"] == "prod"

