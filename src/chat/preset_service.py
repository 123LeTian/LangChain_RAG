"""Prompt preset service for system and local user presets."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chat.schemas import (
    ChatPreset,
    ChatPresetCreate,
    ChatPresetUpdate,
    utc_now,
)
from src.chat_storage.chat_store import ChatStore


DEFAULT_PRESET_ID = "default-assistant"
LOCAL_USER_ID = "local-user"


class ChatPresetNotFoundError(Exception):
    """Raised when a prompt preset does not exist."""


class ChatPresetReadOnlyError(Exception):
    """Raised when trying to modify a system preset."""


class PresetService:
    def __init__(
        self,
        store: ChatStore,
        config_path: Optional[Path] = None,
        *,
        default_preset_id: str = DEFAULT_PRESET_ID,
    ):
        self._store = store
        project_root = Path(__file__).resolve().parents[2]
        self.config_path = config_path or project_root / "config" / "presets.yaml"
        self.default_preset_id = default_preset_id
        self._system_presets = self._load_system_presets()

    def list_presets(self) -> List[ChatPreset]:
        return [*self._system_presets, *self._store.list_user_presets()]

    def public_response(self) -> Dict[str, Any]:
        return {
            "presets": [
                preset.public_dict(include_prompt=preset.owner_type == "user")
                for preset in self.list_presets()
            ],
            "default_preset_id": self.default_preset().id,
        }

    def default_preset(self) -> ChatPreset:
        try:
            return self.resolve(self.default_preset_id)
        except ChatPresetNotFoundError:
            if self._system_presets:
                return self._system_presets[0]
            raise

    def resolve(self, preset_id: Optional[str]) -> ChatPreset:
        target = preset_id or self.default_preset_id
        for preset in self._system_presets:
            if preset.id == target:
                return preset
        user_preset = self._store.get_user_preset(target)
        if user_preset is not None:
            return user_preset
        raise ChatPresetNotFoundError(target)

    def create_user_preset(self, request: ChatPresetCreate) -> ChatPreset:
        now = utc_now()
        preset = ChatPreset(
            id=f"preset_{uuid.uuid4().hex}",
            name=request.name,
            description=request.description,
            category=getattr(request, "category", "自定义") or "自定义",
            system_prompt=request.system_prompt,
            rag_prompt_hint=request.rag_prompt_hint,
            owner_type="user",
            is_default=False,
            created_at=now,
            updated_at=now,
        )
        return self._store.create_preset(preset)

    def update_user_preset(
        self,
        preset_id: str,
        request: ChatPresetUpdate,
    ) -> ChatPreset:
        if self._is_system_preset(preset_id):
            raise ChatPresetReadOnlyError(preset_id)
        preset = self._store.update_user_preset(
            preset_id,
            name=request.name,
            description=request.description,
            category=request.category,
            system_prompt=request.system_prompt,
            rag_prompt_hint=request.rag_prompt_hint,
        )
        if preset is None:
            raise ChatPresetNotFoundError(preset_id)
        return preset

    def delete_user_preset(self, preset_id: str) -> None:
        if self._is_system_preset(preset_id):
            raise ChatPresetReadOnlyError(preset_id)
        deleted = self._store.delete_user_preset(preset_id)
        if not deleted:
            raise ChatPresetNotFoundError(preset_id)
        self._store.replace_session_preset(preset_id, self.default_preset().id)

    def _is_system_preset(self, preset_id: str) -> bool:
        return any(preset.id == preset_id for preset in self._system_presets)

    def _load_system_presets(self) -> List[ChatPreset]:
        if not self.config_path.exists():
            return self._default_system_presets()
        data = self._load_yaml_like(self.config_path)
        raw_presets = data.get("presets") or []
        presets = [
            ChatPreset(**{**item, "owner_type": "system"})
            for item in raw_presets
            if isinstance(item, dict)
        ]
        return presets or self._default_system_presets()

    def _load_yaml_like(self, path: Path) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"presets": []}

    def _default_system_presets(self) -> List[ChatPreset]:
        return [
            ChatPreset(
                id=DEFAULT_PRESET_ID,
                name="默认助手",
                description="保持当前默认回答风格。",
                category="通用",
                system_prompt="你是一名可靠的问答助手。回答应清晰、准确、自然。",
                rag_prompt_hint="保持默认风格，必要时引用知识库证据。",
                owner_type="system",
                is_default=True,
            )
        ]


__all__ = [
    "DEFAULT_PRESET_ID",
    "LOCAL_USER_ID",
    "ChatPresetNotFoundError",
    "ChatPresetReadOnlyError",
    "PresetService",
]
