from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api.app import app
from src.chat.model_registry import ModelRegistry
from src.config.runtime_config import get_runtime_config, reset_runtime_config


def _clear_runtime_env(monkeypatch):
    for key in [
        "APP_ENV",
        "CHAT_DB_PATH",
        "KNOWLEDGE_DIR",
        "VECTOR_DIR",
        "EXPORT_DIR",
        "APP_LOG_PATH",
        "ERROR_LOG_PATH",
        "BUILD_LOG_DIR",
        "MODELS_CONFIG_PATH",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    reset_runtime_config()


def test_default_env_is_dev(monkeypatch):
    _clear_runtime_env(monkeypatch)

    config = get_runtime_config(force_reload=True)

    assert config.env == "dev"
    assert config.chat_db_path.as_posix().endswith("data/dev/chat/chat.db")
    assert config.app_log_path.as_posix().endswith("logs/dev/app.jsonl")


def test_app_env_test_loads_test_paths(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "test")

    config = get_runtime_config(force_reload=True)

    assert config.env == "test"
    assert "data/test/chat/chat.db" in config.chat_db_path.as_posix()
    assert "logs/test/app.jsonl" in config.app_log_path.as_posix()
    assert "logs/test/build" in config.build_log_dir.as_posix()


def test_app_env_prod_loads_prod_paths(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")

    config = get_runtime_config(force_reload=True)

    assert config.env == "prod"
    assert "data/prod/chat/chat.db" in config.chat_db_path.as_posix()
    assert "logs/prod/error.jsonl" in config.error_log_path.as_posix()


def test_system_environment_overrides_yaml(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    custom_db = tmp_path / "custom" / "chat.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CHAT_DB_PATH", str(custom_db))
    monkeypatch.setenv("RETRY_MAX_RETRIES", "7")

    config = get_runtime_config(force_reload=True)

    assert config.chat_db_path == custom_db
    assert config.retry_max_retries == 7


def test_missing_config_files_have_fallback(monkeypatch, tmp_path):
    import src.config.runtime_config as runtime_config

    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(runtime_config, "PROJECT_ROOT", tmp_path)

    config = runtime_config.get_runtime_config(force_reload=True)

    assert config.env == "dev"
    assert config.app_name == "LangChain RAG"
    assert config.chat_db_path == tmp_path / "data" / "dev" / "chat" / "chat.db"


def test_public_config_does_not_expose_secrets(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "real-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "another-secret")

    payload = get_runtime_config(force_reload=True).public_dict()
    text = json.dumps(payload, ensure_ascii=False)

    assert "real-secret-value" not in text
    assert "another-secret" not in text
    assert "api_key" not in text.lower()
    assert "secret" not in text.lower()
    assert "token" not in text.lower()


def test_model_registry_uses_runtime_model_config_path(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    model_config = tmp_path / "models.yaml"
    model_config.write_text(
        """
models:
  - id: custom-default
    provider: ollama
    display_name: Custom Default
    model_name: qwen2.5
    base_url: http://localhost:11434/v1
    api_key_env: null
    enabled: true
    is_default: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(model_config))
    reset_runtime_config()

    registry = ModelRegistry(custom_path=tmp_path / "custom_models.json")

    assert registry.config_path == model_config
    assert registry.default_model().id == "custom-default"


def test_runtime_config_api_is_safe(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "real-secret-value")
    reset_runtime_config()

    with TestClient(app) as client:
        response = client.get("/api/config/runtime")

    assert response.status_code == 200
    data = response.json()
    text = response.text
    assert data["env"] == "test"
    assert "data/test/chat/chat.db" in data["storage"]["chat_db_path"]
    assert "real-secret-value" not in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "api_key" not in text.lower()

