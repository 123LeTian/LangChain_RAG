"""Runtime model configuration for chat-driven RAG calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.chat.model_registry import ChatModel
from src.config.runtime_config import get_model_env_var


SUPPORTED_PROVIDERS = {"deepseek", "openai", "qwen", "ollama", "local"}
LOCAL_PROVIDERS = {"ollama", "local"}


class ModelRuntimeError(Exception):
    """Raised when a chat model cannot be used at runtime."""


@dataclass(frozen=True)
class ModelRuntimeConfig:
    id: str
    provider: str
    model_name: str
    base_url: Optional[str]
    api_key_env: Optional[str]
    api_key: Optional[str]
    supports_stream: bool = True

    def __repr__(self) -> str:
        return (
            "ModelRuntimeConfig("
            f"id={self.id!r}, provider={self.provider!r}, "
            f"model_name={self.model_name!r}, base_url={self.base_url!r}, "
            f"api_key_env={self.api_key_env!r}, api_key=<redacted>, "
            f"supports_stream={self.supports_stream!r})"
        )

    def trace_metadata(self) -> dict[str, str]:
        return {
            "model_id": self.id,
            "provider": self.provider,
            "model_name": self.model_name,
        }


def resolve_runtime_config(
    model_config: ChatModel | Mapping[str, Any],
    *,
    require_stream: bool = False,
) -> ModelRuntimeConfig:
    """Resolve a public/internal model config into an LLM runtime config."""
    values = _model_values(model_config)
    model_id = str(values.get("id") or "").strip()
    provider = str(values.get("provider") or "").strip().lower()
    model_name = str(values.get("model_name") or "").strip()
    base_url = _optional_str(values.get("base_url"))
    api_key_env = _optional_str(values.get("api_key_env"))
    supports_stream = bool(values.get("supports_stream", True))
    enabled = bool(values.get("enabled", True))

    if not model_id:
        raise ModelRuntimeError("模型配置缺少 id")
    if not enabled:
        raise ModelRuntimeError(f"模型 {model_id} 未启用")
    if provider not in SUPPORTED_PROVIDERS:
        raise ModelRuntimeError(f"模型 {model_id} 使用了不支持的 provider: {provider}")
    if not model_name:
        raise ModelRuntimeError(f"模型 {model_id} 缺少 model_name")
    if require_stream and not supports_stream:
        raise ModelRuntimeError(f"模型 {model_id} 不支持流式输出")

    api_key: Optional[str] = None
    if api_key_env:
        api_key = get_model_env_var(api_key_env)
        if not api_key and provider not in LOCAL_PROVIDERS:
            raise ModelRuntimeError(f"模型 {model_id} 缺少环境变量 {api_key_env}")
    elif provider not in LOCAL_PROVIDERS:
        raise ModelRuntimeError(f"模型 {model_id} 缺少 API Key 环境变量配置")

    return ModelRuntimeConfig(
        id=model_id,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        supports_stream=supports_stream,
    )


def _model_values(model_config: ChatModel | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(model_config, Mapping):
        return model_config
    if hasattr(model_config, "model_dump"):
        return model_config.model_dump()
    if hasattr(model_config, "dict"):
        return model_config.dict()
    raise ModelRuntimeError("无效的模型配置")


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "LOCAL_PROVIDERS",
    "ModelRuntimeConfig",
    "ModelRuntimeError",
    "SUPPORTED_PROVIDERS",
    "resolve_runtime_config",
]
