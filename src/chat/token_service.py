"""Token statistics for chat sessions."""

from __future__ import annotations

from src.chat.schemas import ChatSessionStats, ChatStats
from src.chat.session_service import ChatSessionNotFoundError
from src.chat_storage.chat_store import ChatStore


class TokenService:
    def __init__(self, store: ChatStore):
        self._store = store

    def global_stats(self) -> ChatStats:
        stats = self._store.global_chat_stats()
        prompt_tokens = stats["prompt_tokens"]
        completion_tokens = stats["completion_tokens"]
        return ChatStats(
            sessions_count=stats["sessions_count"],
            messages_count=stats["messages_count"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def session_stats(self, session_id: str) -> ChatSessionStats:
        stats = self._store.session_chat_stats(session_id)
        if stats is None:
            raise ChatSessionNotFoundError(session_id)
        prompt_tokens = stats["prompt_tokens"]
        completion_tokens = stats["completion_tokens"]
        return ChatSessionStats(
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            messages_count=stats["messages_count"],
        )


__all__ = ["TokenService"]
