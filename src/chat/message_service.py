"""Business operations for chat messages."""

from __future__ import annotations

import uuid
from typing import List

from src.chat.schemas import ChatMessage, ChatMessageCreate, utc_now
from src.chat.session_service import ChatSessionNotFoundError
from src.chat_storage.chat_store import ChatStore


class MessageService:
    def __init__(self, store: ChatStore):
        self._store = store

    def list_messages(self, session_id: str) -> List[ChatMessage]:
        if self._store.get_session(session_id) is None:
            raise ChatSessionNotFoundError(session_id)
        return self._store.list_messages(session_id)

    def save_message(self, session_id: str, request: ChatMessageCreate) -> ChatMessage:
        if self._store.get_session(session_id) is None:
            raise ChatSessionNotFoundError(session_id)

        message = ChatMessage(
            id=f"message_{uuid.uuid4().hex}",
            session_id=session_id,
            role=request.role,
            content=request.content,
            citations=request.citations,
            trace=request.trace,
            prompt_tokens=request.prompt_tokens,
            completion_tokens=request.completion_tokens,
            latency_ms=request.latency_ms,
            created_at=utc_now(),
        )
        return self._store.create_message(message)
