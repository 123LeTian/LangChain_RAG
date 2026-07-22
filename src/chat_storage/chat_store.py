"""Storage contract for chat sessions and messages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.chat.schemas import ChatMessage, ChatPreset, ChatSession


UNSET = object()


class ChatStore(ABC):
    @abstractmethod
    def create_session(self, session: ChatSession) -> ChatSession: ...

    @abstractmethod
    def list_sessions(self) -> List[ChatSession]: ...

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[ChatSession]: ...

    @abstractmethod
    def update_session(
        self,
        session_id: str,
        *,
        title: Any = UNSET,
        model_id: Any = UNSET,
        preset_id: Any = UNSET,
        rag_mode: Any = UNSET,
        knowledge_base_id: Any = UNSET,
    ) -> Optional[ChatSession]: ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool: ...

    @abstractmethod
    def create_message(self, message: ChatMessage) -> ChatMessage: ...

    @abstractmethod
    def list_messages(self, session_id: str) -> List[ChatMessage]: ...

    @abstractmethod
    def create_preset(self, preset: ChatPreset) -> ChatPreset: ...

    @abstractmethod
    def list_user_presets(self) -> List[ChatPreset]: ...

    @abstractmethod
    def get_user_preset(self, preset_id: str) -> Optional[ChatPreset]: ...

    @abstractmethod
    def update_user_preset(
        self,
        preset_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        rag_prompt_hint: Optional[str] = None,
    ) -> Optional[ChatPreset]: ...

    @abstractmethod
    def delete_user_preset(self, preset_id: str) -> bool: ...

    @abstractmethod
    def replace_session_preset(self, preset_id: str, replacement_preset_id: str) -> None: ...

    @abstractmethod
    def search_messages(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]: ...

    @abstractmethod
    def global_chat_stats(self) -> Dict[str, int]: ...

    @abstractmethod
    def session_chat_stats(self, session_id: str) -> Optional[Dict[str, int]]: ...
