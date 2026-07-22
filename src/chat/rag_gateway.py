"""Adapter between chat application code and the existing RAG service.

Supports three modes:
  - Mock models → canned local response
  - Chat-only (no documents or RAG unavailable) → direct LLM via selected model
  - Full RAG (documents present) → query rewrite + retrieve + rerank + generate
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.api.api_models import RAGMode, RAGRequest
from src.chat.model_runtime import resolve_runtime_config
from src.chat.retry_policy import ChatRetryExhaustedError, RetryPolicy
from src.chat.structured_logger import StructuredLogger


ChatStreamEvent = Dict[str, Any]
MOCK_MODEL_ANSWER = "您好，我是本地默认模型，不会回答任何问题，只用于测试，请在模型配置页面添加您的模型"


class RAGGateway:
    def __init__(
        self,
        rag_service: Any,
        retry_policy: Optional[RetryPolicy] = None,
        logger: Optional[StructuredLogger] = None,
    ):
        self._rag_service = rag_service
        self._retry_policy = retry_policy or RetryPolicy()
        self._logger = logger or StructuredLogger()

    async def stream(
        self,
        *,
        session_id: str,
        original_question: str,
        context_enhanced_question: str,
        rag_mode: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        model_id: Optional[str] = None,
        model_config: Optional[Any] = None,
        preset_id: Optional[str] = None,
        preset_config: Optional[Any] = None,
        request_id: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        started_at = time.monotonic()
        try:
            if self._is_mock_model(model_config):
                async for event in self._stream_mock_model(
                    model_id=model_id,
                    model_config=model_config,
                    request_id=request_id,
                    started_at=started_at,
                ):
                    yield event
                return

            self._validate_model_for_real_rag(model_config)
            try:
                request = self._build_request(
                    session_id=session_id,
                    query=original_question,
                    rag_mode=rag_mode,
                    knowledge_base_id=knowledge_base_id,
                    model_id=model_id,
                    model_config=model_config,
                    preset_id=preset_id,
                    preset_config=preset_config,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    temperature=temperature,
                )
                result, retry_count = await self._retry_policy.run(
                    lambda: self._rag_service.query(request),
                    logger=self._logger,
                    log_payload={
                        "request_id": request_id,
                        "session_id": session_id,
                        "model_id": model_id,
                        "rag_mode": rag_mode,
                        "knowledge_base_id": knowledge_base_id,
                        "preset_id": preset_id,
                    },
                )
                answer = getattr(result, "answer", "") or ""
                trace = self._serialize_list(getattr(result, "trace", []))
                citations = self._serialize_list(getattr(result, "citations", []))
                usage = getattr(result, "usage", {}) or {}
            except ChatRetryExhaustedError:
                raise  # let outer except handler yield error event
            except Exception as rag_error:
                # RAG unavailable (e.g. no documents) — fall back to direct chat
                rag_msg = str(rag_error)
                print(f"[RAG-Gateway] RAG unavailable, falling back to direct chat: {rag_msg[:120]}", flush=True)
                answer, trace, citations, usage, retry_count = await self._direct_chat(
                    model_config=model_config,
                    model_id=model_id,
                    preset_config=preset_config,
                    preset_id=preset_id,
                    question=original_question,
                    fallback_reason=rag_msg[:200],
                )

            if model_id:
                provider = getattr(model_config, "provider", None)
                model_name = getattr(model_config, "model_name", None)
                trace.insert(0, {
                    "stage": "model",
                    "duration_ms": 0,
                    "input_summary": "chat model selection",
                    "output_summary": f"Using model {model_id} via {provider or 'default'}",
                    "metadata": {
                        "model_id": model_id,
                        "provider": provider,
                        "model_name": model_name,
                    },
                })
            if preset_id:
                preset_name = getattr(preset_config, "name", None)
                trace.insert(0, {
                    "stage": "preset",
                    "duration_ms": 0,
                    "input_summary": "chat preset selection",
                    "output_summary": f"Using preset {preset_name or preset_id}",
                    "metadata": {
                        "preset_id": preset_id,
                        "preset_name": preset_name,
                    },
                })

            if trace:
                yield {"type": "trace", "trace": trace}

            chunk_size = 40
            for index in range(0, len(answer), chunk_size):
                yield {"type": "chunk", "content": answer[index : index + chunk_size]}

            if citations:
                yield {"type": "citation", "citations": citations}

            yield {
                "type": "done",
                "content": answer,
                "citations": citations,
                "trace": trace,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "prompt_tokens": self._usage_value(usage, "prompt", "prompt_tokens"),
                "completion_tokens": self._usage_value(
                    usage,
                    "completion",
                    "completion_tokens",
                ),
                "model_id": model_id,
                "request_id": request_id,
                "retry_count": retry_count,
            }
        except ChatRetryExhaustedError as exc:
            yield {
                "type": "error",
                "message": self._friendly_error_message(exc.original),
                "request_id": request_id,
                "retry_count": exc.retry_count,
            }
        except Exception as exc:
            yield {
                "type": "error",
                "message": self._friendly_error_message(exc),
                "request_id": request_id,
                "retry_count": 0,
            }

    def _build_request(
        self,
        *,
        session_id: str,
        query: str,
        rag_mode: Optional[str],
        knowledge_base_id: Optional[str],
        model_id: Optional[str],
        model_config: Optional[Any],
        preset_id: Optional[str],
        preset_config: Optional[Any],
        top_k: Optional[int],
        score_threshold: Optional[float],
        temperature: Optional[float],
    ) -> RAGRequest:
        options: Dict[str, Any] = {}
        if top_k is not None:
            options["top_k"] = top_k
        if score_threshold is not None:
            options["score_threshold"] = score_threshold
        if temperature is not None:
            options["temperature"] = temperature
        if query:
            options["chat_original_question"] = query
        if model_id is not None:
            options["model_id"] = model_id
        if model_config is not None:
            options["model"] = {
                "id": getattr(model_config, "id", model_id),
                "provider": getattr(model_config, "provider", None),
                "model_name": getattr(model_config, "model_name", None),
                "base_url": getattr(model_config, "base_url", None),
                "api_key_env": getattr(model_config, "api_key_env", None),
            }
        if preset_id is not None:
            options["preset_id"] = preset_id
        if preset_config is not None:
            options["preset"] = {
                "id": getattr(preset_config, "id", preset_id),
                "name": getattr(preset_config, "name", None),
                "system_prompt": getattr(preset_config, "system_prompt", None),
                "rag_prompt_hint": getattr(preset_config, "rag_prompt_hint", None),
                "owner_type": getattr(preset_config, "owner_type", None),
            }

        return RAGRequest(
            query=query,
            kb_id=knowledge_base_id or "default",
            mode=RAGMode(rag_mode) if rag_mode else RAGMode.NAIVE,
            session_id=session_id,
            options=options,
        )

    def _serialize_list(self, items: Any) -> List[Any]:
        result: List[Any] = []
        for item in items or []:
            if hasattr(item, "model_dump"):
                result.append(item.model_dump(mode="json"))
            elif hasattr(item, "dict"):
                result.append(item.dict())
            else:
                result.append(item)
        return result

    def _usage_value(self, usage: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return 0

    def _friendly_error_message(self, exc: Exception) -> str:
        if isinstance(exc, TimeoutError) or type(exc).__name__ in {"TimeoutError", "ReadTimeout", "ConnectTimeout"}:
            return "模型调用超时，请稍后重试"
        return str(exc)

    def _is_mock_model(self, model_config: Optional[Any]) -> bool:
        return str(getattr(model_config, "provider", "") or "").lower() == "mock"

    async def _stream_mock_model(
        self,
        *,
        model_id: Optional[str],
        model_config: Optional[Any],
        request_id: Optional[str],
        started_at: float,
    ) -> AsyncIterator[ChatStreamEvent]:
        provider = getattr(model_config, "provider", "mock")
        model_name = getattr(model_config, "model_name", model_id or "mock-chat")
        trace = [{
            "stage": "model",
            "duration_ms": 0,
            "input_summary": "chat model selection",
            "output_summary": f"Using mock model {model_id or model_name}",
            "metadata": {
                "model_id": model_id or model_name,
                "provider": provider,
                "model_name": model_name,
            },
        }]
        yield {"type": "trace", "trace": trace}
        yield {"type": "chunk", "content": MOCK_MODEL_ANSWER}
        yield {
            "type": "done",
            "content": MOCK_MODEL_ANSWER,
            "citations": [],
            "trace": trace,
            "latency_ms": int((time.monotonic() - started_at) * 1000),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_id": model_id or model_name,
            "request_id": request_id,
            "retry_count": 0,
        }

    async def _direct_chat(
        self,
        *,
        model_config: Optional[Any],
        model_id: Optional[str],
        preset_config: Optional[Any],
        preset_id: Optional[str],
        question: str,
        fallback_reason: str,
    ) -> tuple:
        """Direct LLM chat without RAG — used when documents are unavailable."""
        from langchain_openai import ChatOpenAI

        runtime = resolve_runtime_config(model_config)
        system_prompt = getattr(preset_config, "system_prompt", None) or ""
        rag_hint = getattr(preset_config, "rag_prompt_hint", None) or ""

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))

        # Note: user may just want a pure chat, no RAG context needed
        messages.append(("user", question))

        llm = ChatOpenAI(
            model=runtime.model_name,
            api_key=runtime.api_key,
            base_url=runtime.base_url or "https://api.openai.com/v1",
            temperature=0.7,
            max_tokens=2000,
        )

        t0 = time.monotonic()
        try:
            response = await asyncio.to_thread(llm.invoke, messages)
            answer = response.content or ""
        except Exception as chat_error:
            answer = f"抱歉，模型调用失败：{chat_error}。请检查 API Key 配置或切换其他模型。"

        elapsed = int((time.monotonic() - t0) * 1000)

        trace = [{
            "stage": "direct_chat",
            "duration_ms": elapsed,
            "input_summary": question[:100],
            "output_summary": f"Direct LLM answer ({len(answer)} chars); fallback: {fallback_reason[:120]}",
            "metadata": {
                "model_id": model_id or runtime.id,
                "provider": runtime.provider,
                "model_name": runtime.model_name,
                "fallback_reason": fallback_reason[:200],
            },
        }]

        usage = {
            "prompt_tokens": len(question) // 2,
            "completion_tokens": len(answer) // 2,
        }
        return answer, trace, [], usage, 0

    def _validate_model_for_real_rag(self, model_config: Optional[Any]) -> None:
        if model_config is None:
            return
        rag_type = type(self._rag_service)
        is_real_rag = (
            rag_type.__name__ == "RealRAGService"
            and rag_type.__module__ == "src.api.real_rag_service"
        )
        if is_real_rag:
            resolve_runtime_config(model_config)
