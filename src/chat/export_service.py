"""Markdown export for chat sessions."""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.chat.message_service import MessageService
from src.chat.schemas import ChatExportResponse
from src.chat.session_service import SessionService
from src.chat.token_service import TokenService


class ExportService:
    def __init__(
        self,
        session_service: SessionService,
        message_service: MessageService,
        token_service: TokenService,
    ):
        self._session_service = session_service
        self._message_service = message_service
        self._token_service = token_service

    def export_session(self, session_id: str) -> ChatExportResponse:
        session = self._session_service.get_session(session_id)
        messages = self._message_service.list_messages(session_id)
        stats = self._token_service.session_stats(session_id)
        title = self._clean_markdown_text(session.title)
        filename = f"{self._safe_filename(title or session.id)}.md"

        lines = [
            f"# {title or 'Chat Session'}",
            "",
            f"- 会话 ID: `{session.id}`",
            f"- 模型: {session.model_id or '-'}",
            f"- 预设: {session.preset_id or '-'}",
            f"- RAG 模式: {session.rag_mode or '-'}",
            f"- 知识库: {session.knowledge_base_id or '-'}",
            f"- 创建时间: {session.created_at.isoformat()}",
            f"- 更新时间: {session.updated_at.isoformat()}",
            f"- Prompt Tokens: {stats.prompt_tokens}",
            f"- Completion Tokens: {stats.completion_tokens}",
            "",
            "---",
            "",
        ]

        if not messages:
            lines.extend(["暂无消息。", ""])
        else:
            for message in messages:
                lines.extend(self._message_lines(message))

        return ChatExportResponse(filename=filename, content="\n".join(lines).strip() + "\n")

    def _message_lines(self, message: Any) -> list[str]:
        role = "User" if message.role == "user" else "Assistant" if message.role == "assistant" else "System"
        lines = [
            f"## {role}",
            "",
            self._redact_sensitive(self._clean_markdown_text(message.content or "")),
            "",
        ]
        if message.role == "assistant":
            citations = message.citations or []
            trace = message.trace or []
            if citations:
                lines.extend(["### 引用来源", ""])
                for index, citation in enumerate(citations[:10], start=1):
                    lines.append(f"{index}. {self._citation_summary(citation)}")
                lines.append("")
            lines.extend([
                "### 执行摘要",
                "",
                f"- 耗时: {message.latency_ms or 0}ms",
                f"- 引用条数: {len(citations)}",
                f"- 阶段数: {len(trace)}",
                f"- Prompt Tokens: {message.prompt_tokens}",
                f"- Completion Tokens: {message.completion_tokens}",
                "",
            ])
        return lines

    def _citation_summary(self, citation: Any) -> str:
        if not isinstance(citation, dict):
            return self._clean_markdown_text(str(citation))
        filename = citation.get("filename") or citation.get("document_id") or "unknown"
        chunk_id = citation.get("chunk_id") or "-"
        score = citation.get("score")
        score_text = f" / score {score}" if isinstance(score, (int, float)) else ""
        quote = citation.get("quote") or citation.get("text") or ""
        quote_text = f" / {quote[:80]}" if quote else ""
        return self._clean_markdown_text(f"{filename} / chunk {chunk_id}{score_text}{quote_text}")

    def _safe_filename(self, title: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" ._")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return (cleaned[:80] or "chat-session").strip()

    def _clean_markdown_text(self, text: str) -> str:
        return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    def _redact_sensitive(self, text: str) -> str:
        patterns: Iterable[str] = [
            r"(?i)api_key_env\s*[:=]\s*[\w.-]+",
            r"(?i)api[_-]?key\s*[:=]\s*[\w.-]+",
            r"\b[A-Z][A-Z0-9_]*API_KEY\b",
        ]
        redacted = text
        for pattern in patterns:
            redacted = re.sub(pattern, "[redacted]", redacted)
        return redacted


__all__ = ["ExportService"]
