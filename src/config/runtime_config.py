"""Runtime configuration with dev/test/prod isolation."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_ENVS = {"dev", "test", "prod"}
SENSITIVE_RESPONSE_KEYS = ("key", "secret", "token", "password", "authorization", "cookie")


@dataclass(frozen=True)
class RuntimeConfig:
    env: str
    app_name: str
    version: str
    project_root: Path
    raw: Mapping[str, Any]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / self.env

    @property
    def chat_db_path(self) -> Path:
        return self._path("storage.chat_db_path", f"data/{self.env}/chat/chat.db")

    @property
    def knowledge_dir(self) -> Path:
        return self._path("storage.knowledge_dir", f"data/{self.env}/knowledge")

    @property
    def vector_dir(self) -> Path:
        return self._path("storage.vector_dir", f"data/{self.env}/vector")

    @property
    def export_dir(self) -> Path:
        return self._path("storage.export_dir", f"data/{self.env}/exports")

    @property
    def app_log_path(self) -> Path:
        return self._path("logging.app_log_path", f"logs/{self.env}/app.jsonl")

    @property
    def error_log_path(self) -> Path:
        return self._path("logging.error_log_path", f"logs/{self.env}/error.jsonl")

    @property
    def build_log_dir(self) -> Path:
        return self._path("logging.build_log_dir", f"logs/{self.env}/build")

    @property
    def model_config_path(self) -> Path:
        return self._path("models.config_path", "config/models.yaml")

    @property
    def retry_timeout_seconds(self) -> float:
        return float(self._get("retry.timeout_seconds", 60))

    @property
    def retry_max_retries(self) -> int:
        return int(self._get("retry.max_retries", 2))

    @property
    def retry_backoff_seconds(self) -> float:
        return float(self._get("retry.backoff_seconds", 1.5))

    def get_model_env_var(self, name: str) -> Optional[str]:
        return os.getenv(name)

    def public_dict(self) -> dict[str, Any]:
        return {
            "env": self.env,
            "app_name": self.app_name,
            "version": self.version,
            "storage": {
                "chat_db_path": self._display_path(self.chat_db_path),
                "knowledge_dir": self._display_path(self.knowledge_dir),
                "vector_dir": self._display_path(self.vector_dir),
            },
            "logging": {
                "app_log_path": self._display_path(self.app_log_path),
                "error_log_path": self._display_path(self.error_log_path),
            },
        }

    def _path(self, dotted_key: str, default: str) -> Path:
        value = str(self._get(dotted_key, default))
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path)


_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config(*, force_reload: bool = False) -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None or force_reload:
        _runtime_config = _load_runtime_config()
    return _runtime_config


def reset_runtime_config() -> None:
    global _runtime_config
    _runtime_config = None


def get_data_dir() -> Path:
    return get_runtime_config().data_dir


def get_chat_db_path() -> Path:
    return get_runtime_config().chat_db_path


def get_knowledge_dir() -> Path:
    return get_runtime_config().knowledge_dir


def get_vector_dir() -> Path:
    return get_runtime_config().vector_dir


def get_log_dir() -> Path:
    return get_runtime_config().app_log_path.parent


def get_model_env_var(name: str) -> Optional[str]:
    return get_runtime_config().get_model_env_var(name)


def _load_runtime_config() -> RuntimeConfig:
    base = _load_yaml(PROJECT_ROOT / "config.yaml")
    env = _normalize_env(os.getenv("APP_ENV") or _deep_get(base, "runtime.env", "dev"))
    env_config = _load_yaml(PROJECT_ROOT / f"config.{env}.yaml")
    merged = _deep_merge(base, env_config)
    _load_env_files_for_env(env)
    _apply_environment_overrides(merged)
    env = _normalize_env(os.getenv("APP_ENV") or _deep_get(merged, "runtime.env", env))
    merged.setdefault("runtime", {})["env"] = env
    app = merged.get("app") if isinstance(merged.get("app"), Mapping) else {}
    return RuntimeConfig(
        env=env,
        app_name=str(app.get("name") or "LangChain RAG"),
        version=str(app.get("version") or "0.1.0"),
        project_root=PROJECT_ROOT,
        raw=merged,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_env_files_for_env(env: str) -> None:
    for path in [PROJECT_ROOT / f".env.{env}", PROJECT_ROOT / ".env"]:
        if not path.exists():
            continue
        for key, value in _parse_env_file(path).items():
            os.environ.setdefault(key, value)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    mapping = {
        "APP_ENV": ("runtime", "env"),
        "APP_NAME": ("app", "name"),
        "APP_VERSION": ("app", "version"),
        "CHAT_DB_PATH": ("storage", "chat_db_path"),
        "KNOWLEDGE_DIR": ("storage", "knowledge_dir"),
        "VECTOR_DIR": ("storage", "vector_dir"),
        "EXPORT_DIR": ("storage", "export_dir"),
        "APP_LOG_PATH": ("logging", "app_log_path"),
        "ERROR_LOG_PATH": ("logging", "error_log_path"),
        "BUILD_LOG_DIR": ("logging", "build_log_dir"),
        "MODELS_CONFIG_PATH": ("models", "config_path"),
        "CHAT_MEMORY_WINDOW": ("chat", "memory_window"),
        "RETRY_TIMEOUT_SECONDS": ("retry", "timeout_seconds"),
        "RETRY_MAX_RETRIES": ("retry", "max_retries"),
        "RETRY_BACKOFF_SECONDS": ("retry", "backoff_seconds"),
    }
    for env_name, path in mapping.items():
        if env_name in os.environ:
            _deep_set(config, path, _parse_scalar(os.environ[env_name]))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _deep_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _deep_set(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_env(env: str) -> str:
    normalized = str(env or "dev").strip().lower()
    return normalized if normalized in VALID_ENVS else "dev"


__all__ = [
    "RuntimeConfig",
    "get_chat_db_path",
    "get_data_dir",
    "get_knowledge_dir",
    "get_log_dir",
    "get_model_env_var",
    "get_runtime_config",
    "get_vector_dir",
    "reset_runtime_config",
]
