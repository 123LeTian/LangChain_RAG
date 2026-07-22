"""Application service for session-aware chat streaming."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Dict

from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.preset_service import PresetService
from src.chat.preset_service import ChatPresetNotFoundError
from src.chat.rag_gateway import RAGGateway
from src.chat.structured_logger import StructuredLogger
from src.chat.schemas import (
    DEFAULT_CHAT_TITLE,
    ChatMessageCreate,
    ChatSessionUpdate,
    ChatStreamRequest,
)
from src.chat.session_service import ChatSessionNotFoundError, SessionService


ChatStreamEvent = Dict[str, Any]


class ChatApplicationService:
    def __init__(
        self,
        session_service: SessionService,
        message_service: MessageService,
        memory_service: MemoryService,
        rag_gateway: RAGGateway,
        model_registry: ModelRegistry,
        preset_service: PresetService | None = None,
        logger: StructuredLogger | None = None,
    ):
        self._session_service = session_service
        self._message_service = message_service
        self._memory_service = memory_service
        self._rag_gateway = rag_gateway
        self._model_registry = model_registry
        self._preset_service = preset_service
        self._logger = logger or StructuredLogger()

    async def stream_session(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> AsyncIterator[ChatStreamEvent]:
        session = self._session_service.get_session(session_id)
        started_at = time.monotonic()
        request_id = uuid.uuid4().hex
        model = self._resolve_model(request.model_id or session.model_id)
        preset = self._resolve_preset(request.preset_id or session.preset_id)

        if request.model_id and request.model_id != session.model_id:
            session = self._session_service.update_model(session_id, model.id)
        if preset is not None and request.preset_id and request.preset_id != session.preset_id:
            session = self._session_service.update_preset(session_id, preset.id)

        user_message = self._message_service.save_message(
            session_id,
            ChatMessageCreate(role="user", content=request.question),
        )

        if session.title == DEFAULT_CHAT_TITLE:
            self._session_service.update_session(
                session_id,
                ChatSessionUpdate(title=self._make_title(request.question)),
            )

        enhanced_question = self._memory_service.build_context_enhanced_question(
            session_id,
            request.question,
            exclude_message_id=user_message.id,
        )

        assistant_content = ""
        citations = []
        trace = []
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0

        async for event in self._rag_gateway.stream(
            session_id=session_id,
            original_question=request.question,
            context_enhanced_question=enhanced_question,
            rag_mode=request.rag_mode or session.rag_mode,
            knowledge_base_id=request.knowledge_base_id or session.knowledge_base_id,
            model_id=model.id,
            model_config=model,
            preset_id=preset.id if preset is not None else None,
            preset_config=preset,
            request_id=request_id,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
            rewrite_enabled=request.rewrite_enabled,
            retrieve_enabled=request.retrieve_enabled,
            rerank_enabled=request.rerank_enabled,
            compress_enabled=request.compress_enabled,
            verify_enabled=request.verify_enabled,
        ):
            event_type = event.get("type")
            if event_type == "chunk":
                assistant_content += event.get("content", "")
                yield event
            elif event_type == "trace":
                trace = event.get("trace") or []
                yield event
            elif event_type == "citation":
                citations = event.get("citations") or []
                yield event
            elif event_type == "done":
                assistant_content = event.get("content", assistant_content)
                citations = event.get("citations", citations) or []
                trace = event.get("trace", trace) or []
                prompt_tokens = int(event.get("prompt_tokens") or 0)
                completion_tokens = int(event.get("completion_tokens") or 0)
                latency_ms = int(
                    event.get("latency_ms")
                    or ((time.monotonic() - started_at) * 1000)
                )
                assistant_message = self._message_service.save_message(
                    session_id,
                    ChatMessageCreate(
                        role="assistant",
                        content=assistant_content or "",
                        citations=citations,
                        trace=trace,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                    ),
                )
                done_event = {
                    "type": "done",
                    "request_id": request_id,
                    "message_id": assistant_message.id,
                    "session_id": session_id,
                    "content": assistant_content,
                    "citations": citations,
                    "trace": trace,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model_id": model.id,
                    "retry_count": int(event.get("retry_count") or 0),
                }
                self._logger.info(
                    "chat_stream_done",
                    request_id=request_id,
                    session_id=session_id,
                    message_id=assistant_message.id,
                    model_id=model.id,
                    provider=getattr(model, "provider", None),
                    rag_mode=request.rag_mode or session.rag_mode,
                    knowledge_base_id=request.knowledge_base_id or session.knowledge_base_id,
                    preset_id=preset.id if preset is not None else None,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=int(event.get("retry_count") or 0),
                    status="success",
                )
                yield done_event
            elif event_type == "error":
                error_event = {
                    **event,
                    "request_id": event.get("request_id") or request_id,
                    "retry_count": int(event.get("retry_count") or 0),
                }
                self._logger.error(
                    "chat_stream_error",
                    request_id=error_event["request_id"],
                    session_id=session_id,
                    model_id=model.id,
                    provider=getattr(model, "provider", None),
                    rag_mode=request.rag_mode or session.rag_mode,
                    knowledge_base_id=request.knowledge_base_id or session.knowledge_base_id,
                    preset_id=preset.id if preset is not None else None,
                    error_type="ChatStreamError",
                    message=error_event.get("message", ""),
                    retry_count=error_event["retry_count"],
                    status="error",
                )
                yield error_event
                return
            else:
                yield event

    def ensure_session(self, session_id: str) -> None:
        self._session_service.get_session(session_id)

    def ensure_stream_model(self, session_id: str, model_id: str | None) -> None:
        session = self._session_service.get_session(session_id)
        self._resolve_model(model_id or session.model_id)

    def ensure_stream_options(
        self,
        session_id: str,
        *,
        model_id: str | None = None,
        preset_id: str | None = None,
    ) -> None:
        session = self._session_service.get_session(session_id)
        self._resolve_model(model_id or session.model_id)
        self._resolve_preset(preset_id or session.preset_id)

    def _resolve_model(self, model_id: str | None):
        if model_id:
            return self._model_registry.get_enabled(model_id)
        return self._model_registry.default_model()

    def _resolve_preset(self, preset_id: str | None):
        if self._preset_service is None:
            return None
        try:
            return self._preset_service.resolve(preset_id)
        except ChatPresetNotFoundError:
            return self._preset_service.default_preset()

    def _make_title(self, question: str) -> str:
        stripped = question.strip()
        if self._contains_cjk(stripped):
            return stripped[:20]
        return stripped[:40]

    def _contains_cjk(self, value: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in value)


__all__ = ["ChatApplicationService", "ChatSessionNotFoundError"]
