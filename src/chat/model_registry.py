"""Model shelf configuration for the chat platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.config.runtime_config import get_runtime_config


class ChatModel(BaseModel):
    id: str
    provider: str
    display_name: str
    model_name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    supports_stream: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    enabled: bool = True
    is_default: bool = False

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "display_name": self.display_name,
            "model_name": self.model_name,
            "supports_stream": self.supports_stream,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "enabled": self.enabled,
            "is_default": self.is_default,
        }


class ModelNotFoundError(Exception):
    """Raised when a model is missing or disabled."""


class ModelRegistry:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or get_runtime_config().model_config_path
        self._models = self._load_models()

    def list_enabled(self) -> List[ChatModel]:
        return [model for model in self._models if model.enabled]

    def get_enabled(self, model_id: str) -> ChatModel:
        for model in self._models:
            if model.id == model_id and model.enabled:
                return model
        raise ModelNotFoundError(model_id)

    def default_model(self) -> ChatModel:
        enabled = self.list_enabled()
        for model in enabled:
            if model.is_default:
                return model
        if enabled:
            return enabled[0]
        return self._default_models()[0]

    def public_response(self) -> Dict[str, Any]:
        default_model = self.default_model()
        return {
            "models": [model.public_dict() for model in self.list_enabled()],
            "default_model_id": default_model.id,
        }

    def _load_models(self) -> List[ChatModel]:
        if not self.config_path.exists():
            return self._default_models()

        data = self._load_yaml_like(self.config_path)
        models = data.get("models") or []
        if not isinstance(models, list):
            return self._default_models()

        parsed = [ChatModel(**item) for item in models if isinstance(item, dict)]
        return parsed or self._default_models()

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
                id="deepseek-chat",
                provider="deepseek",
                display_name="DeepSeek Chat",
                model_name="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key_env="DEEPSEEK_API_KEY",
                supports_stream=True,
                enabled=True,
                is_default=True,
            )
        ]


__all__ = ["ChatModel", "ModelNotFoundError", "ModelRegistry"]
