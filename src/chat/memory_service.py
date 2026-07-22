"""Conversation memory formatting for session-aware chat."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.chat.message_service import MessageService
from src.chat.schemas import ChatMessage


_ROLE_LABELS: Dict[str, str] = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
}


class MemoryService:
    def __init__(self, message_service: MessageService, max_messages: int = 10):
        self._message_service = message_service
        self._max_messages = max_messages

    def recent_messages(
        self,
        session_id: str,
        *,
        exclude_message_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        messages = self._message_service.list_messages(session_id)
        if exclude_message_id is not None:
            messages = [message for message in messages if message.id != exclude_message_id]
        return messages[-self._max_messages :]

    def build_context_enhanced_question(
        self,
        session_id: str,
        question: str,
        *,
        exclude_message_id: Optional[str] = None,
    ) -> str:
        history = self.recent_messages(
            session_id,
            exclude_message_id=exclude_message_id,
        )
        if not history:
            return question

        lines = ["以下是当前会话的最近历史："]
        for message in history:
            label = _ROLE_LABELS.get(message.role, message.role)
            lines.append(f"{label}: {message.content}")
        lines.extend(["", "当前问题：", question])
        return "\n".join(lines)
