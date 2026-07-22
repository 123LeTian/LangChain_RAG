"""History message search for chat sessions."""

from __future__ import annotations

import html
from typing import Optional

from src.chat.schemas import ChatSearchItem, ChatSearchResponse
from src.chat_storage.chat_store import ChatStore


class ChatSearchValidationError(Exception):
    """Raised when a search request is invalid."""


class SearchService:
    def __init__(self, store: ChatStore):
        self._store = store

    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ChatSearchResponse:
        stripped = query.strip()
        if not stripped:
            raise ChatSearchValidationError("q cannot be empty")
        if role is not None and role not in {"user", "assistant", "system"}:
            raise ChatSearchValidationError("role is invalid")

        safe_limit = min(max(limit, 1), 100)
        safe_offset = max(offset, 0)
        rows, total = self._store.search_messages(
            stripped,
            session_id=session_id,
            role=role,
            limit=safe_limit,
            offset=safe_offset,
        )
        return ChatSearchResponse(
            items=[
                ChatSearchItem(
                    session_id=row["session_id"],
                    session_title=row["session_title"],
                    message_id=row["message_id"],
                    role=row["role"],
                    snippet=self._snippet(row["content"], stripped),
                    created_at=row["created_at"],
                )
                for row in rows
            ],
            total=total,
        )

    def _snippet(self, content: str, query: str, max_length: int = 120) -> str:
        text = " ".join((content or "").split())
        index = text.lower().find(query.lower())
        if index < 0:
            raw = text[:max_length]
        else:
            start = max(index - 40, 0)
            end = min(start + max_length, len(text))
            start = max(end - max_length, 0)
            raw = text[start:end]
            if start > 0:
                raw = f"...{raw}"
            if end < len(text):
                raw = f"{raw}..."
        return html.escape(raw, quote=False)[:max_length]


__all__ = ["ChatSearchValidationError", "SearchService"]
