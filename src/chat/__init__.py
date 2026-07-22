"""Chat session and message services."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ChatApplicationService": ("src.chat.chat_application_service", "ChatApplicationService"),
    "ChatMessage": ("src.chat.schemas", "ChatMessage"),
    "ChatMessageCreate": ("src.chat.schemas", "ChatMessageCreate"),
    "ChatMessageRole": ("src.chat.schemas", "ChatMessageRole"),
    "ChatModel": ("src.chat.model_registry", "ChatModel"),
    "ChatPreset": ("src.chat.schemas", "ChatPreset"),
    "ChatPresetCreate": ("src.chat.schemas", "ChatPresetCreate"),
    "ChatPresetUpdate": ("src.chat.schemas", "ChatPresetUpdate"),
    "ChatSession": ("src.chat.schemas", "ChatSession"),
    "ChatSessionCreate": ("src.chat.schemas", "ChatSessionCreate"),
    "ChatSessionModelUpdate": ("src.chat.schemas", "ChatSessionModelUpdate"),
    "ChatSessionPresetUpdate": ("src.chat.schemas", "ChatSessionPresetUpdate"),
    "ChatSessionUpdate": ("src.chat.schemas", "ChatSessionUpdate"),
    "ChatStreamRequest": ("src.chat.schemas", "ChatStreamRequest"),
    "LLMClientFactory": ("src.chat.llm_client_factory", "LLMClientFactory"),
    "MemoryService": ("src.chat.memory_service", "MemoryService"),
    "MessageService": ("src.chat.message_service", "MessageService"),
    "ModelRegistry": ("src.chat.model_registry", "ModelRegistry"),
    "ModelRuntimeConfig": ("src.chat.model_runtime", "ModelRuntimeConfig"),
    "ModelRuntimeError": ("src.chat.model_runtime", "ModelRuntimeError"),
    "PresetService": ("src.chat.preset_service", "PresetService"),
    "RAGGateway": ("src.chat.rag_gateway", "RAGGateway"),
    "SessionService": ("src.chat.session_service", "SessionService"),
    "resolve_runtime_config": ("src.chat.model_runtime", "resolve_runtime_config"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
