"""Business operations for chat sessions."""

from __future__ import annotations

import uuid
from typing import List, Optional

from src.chat.schemas import ChatSession, ChatSessionCreate, ChatSessionUpdate, utc_now
from src.chat_storage.chat_store import UNSET, ChatStore


class ChatSessionNotFoundError(Exception):
    """Raised when a chat session does not exist."""


class SessionService:
    def __init__(self, store: ChatStore):
        self._store = store

    def create_session(self, request: Optional[ChatSessionCreate] = None) -> ChatSession:
        payload = request or ChatSessionCreate()
        now = utc_now()
        session = ChatSession(
            id=f"session_{uuid.uuid4().hex}",
            title=payload.title,
            model_id=payload.model_id,
            preset_id=payload.preset_id,
            rag_mode=payload.rag_mode,
            knowledge_base_id=payload.knowledge_base_id,
            created_at=now,
            updated_at=now,
        )
        return self._store.create_session(session)

    def list_sessions(self) -> List[ChatSession]:
        return self._store.list_sessions()

    def get_session(self, session_id: str) -> ChatSession:
        session = self._store.get_session(session_id)
        if session is None:
            raise ChatSessionNotFoundError(session_id)
        return session

    def update_session(self, session_id: str, request: ChatSessionUpdate) -> ChatSession:
        fields = request.model_fields_set
        session = self._store.update_session(
            session_id,
            title=request.title if "title" in fields else UNSET,
            model_id=request.model_id if "model_id" in fields else UNSET,
            preset_id=request.preset_id if "preset_id" in fields else UNSET,
            rag_mode=request.rag_mode if "rag_mode" in fields else UNSET,
            knowledge_base_id=(
                request.knowledge_base_id
                if "knowledge_base_id" in fields
                else UNSET
            ),
        )
        if session is None:
            raise ChatSessionNotFoundError(session_id)
        return session

    def update_model(self, session_id: str, model_id: str) -> ChatSession:
        session = self._store.update_session(session_id, model_id=model_id)
        if session is None:
            raise ChatSessionNotFoundError(session_id)
        return session

    def update_preset(self, session_id: str, preset_id: str) -> ChatSession:
        session = self._store.update_session(session_id, preset_id=preset_id)
        if session is None:
            raise ChatSessionNotFoundError(session_id)
        return session

    def delete_session(self, session_id: str) -> None:
        deleted = self._store.delete_session(session_id)
        if not deleted:
            raise ChatSessionNotFoundError(session_id)
