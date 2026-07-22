"""Model shelf configuration for the chat platform."""

from __future__ import annotations

import json
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel

from src.config.runtime_config import get_runtime_config


class ChatModel(BaseModel):
    id: str
    provider: str
    display_name: str
    model_name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    description: str = ""
    supports_stream: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    enabled: bool = True
    is_default: bool = False
    is_builtin: bool = True

    def public_dict(self) -> Dict[str, Any]:
        key_required = self.provider.lower() not in {"ollama", "local", "mock"}
        key_configured = bool(self.api_key_env and os.getenv(self.api_key_env))
        return {
            "id": self.id,
            "provider": self.provider,
            "display_name": self.display_name,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "description": self.description,
            "supports_stream": self.supports_stream,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "enabled": self.enabled,
            "is_default": self.is_default,
            "is_builtin": self.is_builtin,
            "key_required": key_required,
            "key_configured": key_configured,
            "key_scope": "not_required" if not key_required else "independent",
        }


class ModelNotFoundError(Exception):
    """Raised when a model is missing or disabled."""


class ModelReadOnlyError(Exception):
    """Raised when trying to mutate a built-in model."""


class ModelConnectionError(Exception):
    """Sanitized model connection failure suitable for API responses."""


ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def model_api_key_env_name(model_name: str, base_url: str = "") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper() or "CUSTOM"
    identity = f"{base_url.strip().rstrip('/').casefold()}\n{model_name.strip()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8].upper()
    return f"MODEL_{slug[:80]}_{digest}_API_KEY"


class ModelRegistry:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        custom_path: Optional[Path] = None,
        secret_path: Optional[Path] = None,
    ):
        runtime = get_runtime_config()
        self.config_path = config_path or get_runtime_config().model_config_path
        self.custom_path = custom_path or runtime.data_dir / "chat" / "custom_models.json"
        self.secret_path = secret_path or runtime.project_root / f".env.{runtime.env}"
        self._models = self._load_models()

    def list_models(self) -> List[ChatModel]:
        default_id = self._resolve_default_model_id()
        return [
            model.model_copy(update={"is_default": model.id == default_id})
            for model in self._models
        ]

    def list_enabled(self) -> List[ChatModel]:
        return [model for model in self.list_models() if model.enabled]

    def get_enabled(self, model_id: str) -> ChatModel:
        for model in self.list_enabled():
            if model.id == model_id and model.enabled:
                return model
        raise ModelNotFoundError(model_id)

    def default_model(self) -> ChatModel:
        enabled = [model for model in self._models if model.enabled]
        custom_default = self._custom_default_model_id()
        if custom_default:
            for model in enabled:
                if model.id == custom_default:
                    return model.model_copy(update={"is_default": True})
        for model in enabled:
            if model.is_default:
                return model.model_copy(update={"is_default": True})
        if enabled:
            return enabled[0].model_copy(update={"is_default": True})
        return self._default_models()[0]

    def public_response(self) -> Dict[str, Any]:
        default_model = self.default_model()
        return {
            "models": [model.public_dict() for model in self.list_enabled()],
            "default_model_id": default_model.id,
        }

    def public_management_response(self) -> Dict[str, Any]:
        default_model = self.default_model()
        models = self.list_models()
        configured_keys = sum(
            1
            for model in models
            if model.provider.lower() not in {"ollama", "local", "mock"}
            and model.api_key_env
            and os.getenv(model.api_key_env)
        )
        required_keys = sum(
            1
            for model in models
            if model.provider.lower() not in {"ollama", "local", "mock"}
        )
        return {
            "models": [model.public_dict() for model in models],
            "default_model_id": default_model.id,
            "key_metrics": {
                "configured": configured_keys,
                "required": required_keys,
                "missing": max(required_keys - configured_keys, 0),
                "independent": required_keys,
            },
        }

    def create_model(self, payload: Any) -> ChatModel:
        data = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else dict(payload)
        api_key = self._optional_secret(data.pop("api_key", None))
        data["id"] = data.get("id") or f"custom-{uuid.uuid4().hex[:12]}"
        if api_key and not data.get("api_key_env"):
            data["api_key_env"] = model_api_key_env_name(
                str(data.get("model_name") or ""),
                str(data.get("base_url") or ""),
            )
        data["is_builtin"] = False
        data["is_default"] = False
        model = ChatModel(**data)
        self._validate_custom_model(model)
        if any(item.id == model.id for item in self._models):
            raise ValueError("Model id already exists")
        if any(item.model_name == model.model_name and item.provider == model.provider for item in self._models):
            raise ValueError("Model already exists")
        if api_key and model.api_key_env:
            self._persist_api_key(model.api_key_env, api_key)
        custom = self._load_custom_payload()
        custom["models"] = [*custom.get("models", []), model.model_dump()]
        self._write_custom_payload(custom)
        self._models = self._load_models()
        return self.get_model(model.id)

    def update_model(self, model_id: str, payload: Any) -> ChatModel:
        model = self.get_model(model_id)
        if model.is_builtin:
            raise ModelReadOnlyError(model_id)
        changes = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else dict(payload)
        api_key = self._optional_secret(changes.pop("api_key", None))
        candidate = model.model_dump()
        candidate.update(changes)
        candidate["id"] = model.id
        candidate["is_builtin"] = False
        if api_key and not candidate.get("api_key_env"):
            candidate["api_key_env"] = model_api_key_env_name(
                str(candidate.get("model_name") or ""),
                str(candidate.get("base_url") or ""),
            )
        updated = ChatModel(**candidate)
        self._validate_custom_model(updated)
        if api_key and updated.api_key_env:
            self._persist_api_key(updated.api_key_env, api_key)
        custom = self._load_custom_payload()
        custom["models"] = [
            updated.model_dump() if item.get("id") == model_id else item
            for item in custom.get("models", [])
            if isinstance(item, dict)
        ]
        self._write_custom_payload(custom)
        self._models = self._load_models()
        return self.get_model(model_id)

    def delete_model(self, model_id: str) -> None:
        model = self.get_model(model_id)
        if model.is_builtin:
            raise ModelReadOnlyError(model_id)
        custom = self._load_custom_payload()
        custom["models"] = [
            item for item in custom.get("models", [])
            if isinstance(item, dict) and item.get("id") != model_id
        ]
        if custom.get("default_model_id") == model_id:
            custom["default_model_id"] = None
        self._write_custom_payload(custom)
        self._models = self._load_models()

    def set_default_model(self, model_id: str) -> ChatModel:
        model = self.get_enabled(model_id)
        custom = self._load_custom_payload()
        custom["default_model_id"] = model.id
        self._write_custom_payload(custom)
        self._models = self._load_models()
        return self.default_model()

    def test_connection(self, model_id: str) -> Dict[str, Any]:
        model = self.get_enabled(model_id)
        provider = model.provider.lower()
        if provider in {"ollama", "local", "mock"}:
            return {"ok": True, "message": "本地模型无需 API Key", "model_id": model.id}
        if not model.api_key_env or not os.getenv(model.api_key_env):
            return {
                "ok": False,
                "message": f"模型 {model.id} 缺少独立密钥配置",
                "model_id": model.id,
            }
        if not model.base_url:
            return {"ok": False, "message": "模型缺少 Base URL", "model_id": model.id}
        try:
            self._probe_openai_compatible(model)
            return {"ok": True, "message": "连接测试通过", "model_id": model.id}
        except Exception as exc:
            return {
                "ok": False,
                "message": _sanitize_connection_error(exc),
                "model_id": model.id,
            }

    def discover_model_ids(self, base_url: str, api_key: str) -> Dict[str, Any]:
        clean_base_url = base_url.strip().rstrip("/")
        parsed = urlparse(clean_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("API Base URL must be http(s)")
        clean_api_key = api_key.strip()
        if not clean_api_key:
            raise ValueError("API Key cannot be empty")
        try:
            return {
                "models": self._fetch_model_ids(clean_base_url, clean_api_key),
                "base_url": clean_base_url,
            }
        except Exception as exc:
            raise ValueError(_sanitize_connection_error(exc)) from exc

    def _load_models(self) -> List[ChatModel]:
        if self.config_path.exists():
            data = self._load_yaml_like(self.config_path)
            models = data.get("models") or []
            builtins = [
                ChatModel(**{**item, "is_builtin": True})
                for item in models
                if isinstance(item, dict)
            ] if isinstance(models, list) else []
        else:
            builtins = []
        builtins = builtins or self._default_models()
        custom_payload = self._load_custom_payload()
        custom = [
            ChatModel(**{**item, "is_builtin": False, "is_default": False})
            for item in custom_payload.get("models", [])
            if isinstance(item, dict)
        ]
        return [*builtins, *custom]

    def _load_yaml_like(self, path: Path) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return self._parse_simple_models_yaml(path.read_text(encoding="utf-8"))

    def _parse_simple_models_yaml(self, text: str) -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "models:":
                continue
            if line.startswith("- "):
                if current is not None:
                    models.append(current)
                current = {}
                key, value = line[2:].split(":", 1)
                current[key.strip()] = self._parse_scalar(value.strip())
                continue
            if current is not None and ":" in line:
                key, value = line.split(":", 1)
                current[key.strip()] = self._parse_scalar(value.strip())
        if current is not None:
            models.append(current)
        return {"models": models}

    def _parse_scalar(self, value: str) -> Any:
        if value == "null":
            return None
        if value == "true":
            return True
        if value == "false":
            return False
        return value

    def _default_models(self) -> List[ChatModel]:
        return [
            ChatModel(
                id="mock-chat",
                provider="mock",
                display_name="Mock Chat",
                model_name="mock-chat",
                base_url=None,
                api_key_env=None,
                description="Local mock model for testing custom model creation.",
                supports_stream=True,
                enabled=True,
                is_default=True,
                is_builtin=True,
            )
        ]

    def _load_custom_payload(self) -> Dict[str, Any]:
        if not self.custom_path.exists():
            return {"models": [], "default_model_id": None}
        try:
            data = json.loads(self.custom_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"models": [], "default_model_id": None}
        except Exception:
            return {"models": [], "default_model_id": None}

    def _write_custom_payload(self, payload: Dict[str, Any]) -> None:
        self.custom_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.custom_path.with_suffix(self.custom_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.custom_path)

    def _custom_default_model_id(self) -> Optional[str]:
        value = self._load_custom_payload().get("default_model_id")
        return value if isinstance(value, str) and value else None

    def _resolve_default_model_id(self) -> str:
        custom_default = self._custom_default_model_id()
        enabled = [model for model in self._models if model.enabled]
        if custom_default and any(model.id == custom_default for model in enabled):
            return custom_default
        for model in enabled:
            if model.is_default:
                return model.id
        if enabled:
            return enabled[0].id
        return self._default_models()[0].id

    def _validate_custom_model(self, model: ChatModel) -> None:
        if model.is_builtin:
            raise ValueError("Custom model cannot be builtin")
        provider = model.provider.strip().lower()
        if not provider:
            raise ValueError("Provider is required")
        if model.api_key_env and not ENV_NAME_PATTERN.fullmatch(model.api_key_env):
            raise ValueError("API key env name is invalid")
        if model.base_url:
            parsed = urlparse(model.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Base URL must be http(s)")

    def _optional_secret(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _persist_api_key(self, env_name: str, api_key: str) -> None:
        if not ENV_NAME_PATTERN.fullmatch(env_name):
            raise ValueError("API key env name is invalid")
        secret = api_key.strip()
        if not secret:
            raise ValueError("API Key cannot be empty")
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        lines = (
            self.secret_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if self.secret_path.exists()
            else []
        )
        replacement = f"{env_name}={secret}\n"
        matched = False
        next_lines: list[str] = []
        for line in lines:
            if re.match(rf"^\s*{re.escape(env_name)}\s*=", line):
                if not matched:
                    next_lines.append(replacement)
                    matched = True
                continue
            next_lines.append(line)
        if not matched:
            if next_lines and not next_lines[-1].endswith(("\n", "\r")):
                next_lines[-1] += "\n"
            next_lines.append(replacement)
        temp = self.secret_path.with_suffix(self.secret_path.suffix + ".tmp")
        try:
            temp.write_text("".join(next_lines), encoding="utf-8")
            temp.replace(self.secret_path)
        finally:
            if temp.exists():
                temp.unlink()
        os.environ[env_name] = secret

    def get_model(self, model_id: str) -> ChatModel:
        for model in self.list_models():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(model_id)

    def _probe_openai_compatible(self, model: ChatModel) -> None:
        request = Request(
            f"{str(model.base_url).rstrip('/')}/models",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {os.getenv(str(model.api_key_env), '')}",
            },
            method="GET",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310 - URL is user configured.
            response.read(1)

    def _fetch_model_ids(self, base_url: str, api_key: str) -> List[str]:
        request = Request(
            f"{base_url.rstrip('/')}/models",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="GET",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - URL is user configured.
            payload = json.load(response)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list):
            raise ValueError("Invalid model list response")
        model_ids: List[str] = []
        for item in data:
            model_id = item.get("id") if isinstance(item, Mapping) else None
            clean_id = model_id.strip() if isinstance(model_id, str) else ""
            if clean_id and clean_id not in model_ids:
                model_ids.append(clean_id)
        if not model_ids:
            raise ValueError("Model list is empty")
        return model_ids


def _sanitize_connection_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret)=\S+", r"\1=<redacted>", text)
    return f"连接测试失败：{text[:180]}"


__all__ = [
    "ChatModel",
    "ModelConnectionError",
    "ModelNotFoundError",
    "ModelReadOnlyError",
    "ModelRegistry",
    "model_api_key_env_name",
]
