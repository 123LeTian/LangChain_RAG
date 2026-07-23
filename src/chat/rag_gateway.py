"""Adapter between chat application code and the existing RAG service."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.api.api_models import RAGMode, RAGRequest
from src.chat.model_runtime import resolve_runtime_config
from src.chat.retry_policy import ChatRetryExhaustedError, RetryPolicy
from src.chat.structured_logger import StructuredLogger


ChatStreamEvent = Dict[str, Any]
DIRECT_RAG_MODE = "direct"
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
        rerank_top_k: Optional[int] = None,
        graph_scope: Optional[str] = None,
        score_threshold: Optional[float] = None,
        max_steps: Optional[int] = None,
        agent_vector_enabled: Optional[bool] = None,
        agent_graph_enabled: Optional[bool] = None,
        temperature: Optional[float] = None,
        rewrite_enabled: Optional[bool] = None,
        retrieve_enabled: Optional[bool] = None,
        rerank_enabled: Optional[bool] = None,
        compress_enabled: Optional[bool] = None,
        verify_enabled: Optional[bool] = None,
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

            if self._is_model_identity_question(original_question):
                async for event in self._stream_model_identity(
                    model_id=model_id,
                    model_config=model_config,
                    request_id=request_id,
                    started_at=started_at,
                ):
                    yield event
                return

            # Direct chat mode streams through the selected LLM without retrieval.
            if self._is_direct_mode(rag_mode):
                async for event in self._stream_direct_llm(
                    original_question=original_question,
                    model_config=model_config,
                    model_id=model_id,
                    request_id=request_id,
                    started_at=started_at,
                ):
                    yield event
                return

            self._validate_model_for_real_rag(model_config)
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
                rerank_top_k=rerank_top_k,
                graph_scope=graph_scope,
                score_threshold=score_threshold,
                max_steps=max_steps,
                agent_vector_enabled=agent_vector_enabled,
                agent_graph_enabled=agent_graph_enabled,
                temperature=temperature,
                rewrite_enabled=rewrite_enabled,
                retrieve_enabled=retrieve_enabled,
                rerank_enabled=rerank_enabled,
                compress_enabled=compress_enabled,
                verify_enabled=verify_enabled,
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
            citations = self._serialize_list(getattr(result, "citations", []))
            usage = getattr(result, "usage", {}) or {}

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


    async def _stream_direct_llm(
        self,
        *,
        original_question: str,
        model_config: Optional[Any],
        model_id: Optional[str],
        request_id: Optional[str],
        started_at: float,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream a direct LLM response without RAG retrieval (daily chat)."""
        from src.chat.llm_client_factory import LLMClientFactory

        factory = LLMClientFactory()
        llm = factory.create_chat_model(
            model_config,
            temperature=0.7,
            max_tokens=1000,
            streaming=True,
        )

        provider = getattr(model_config, "provider", None) or "default"
        model_name = getattr(model_config, "model_name", None) or model_id or "default"

        trace = [{
            "stage": "model",
            "duration_ms": 0,
            "input_summary": "direct chat (no RAG)",
            "output_summary": f"Streaming via {model_id or model_name}",
            "metadata": {
                "model_id": model_id or model_name,
                "provider": provider,
                "model_name": model_name,
            },
        }]
        yield {"type": "trace", "trace": trace}

        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            for chunk in llm.stream(original_question):
                delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                if delta:
                    full_content += delta
                    yield {"type": "chunk", "content": delta}

            try:
                if hasattr(llm, "get_last_response_metadata"):
                    meta = llm.get_last_response_metadata()
                    usage = meta.get("usage", {}) if meta else {}
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
            except Exception:
                pass
            if not completion_tokens:
                completion_tokens = len(full_content)

            yield {
                "type": "done",
                "content": full_content,
                "citations": [],
                "trace": trace,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_id": model_id or model_name,
                "request_id": request_id,
                "retry_count": 0,
            }
        except Exception as exc:
            yield {
                "type": "error",
                "message": str(exc),
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
        rerank_top_k: Optional[int],
        graph_scope: Optional[str],
        score_threshold: Optional[float],
        max_steps: Optional[int],
        agent_vector_enabled: Optional[bool],
        agent_graph_enabled: Optional[bool],
        temperature: Optional[float],
        rewrite_enabled: Optional[bool],
        retrieve_enabled: Optional[bool],
        rerank_enabled: Optional[bool],
        compress_enabled: Optional[bool],
        verify_enabled: Optional[bool],
    ) -> RAGRequest:
        options: Dict[str, Any] = {}
        if top_k is not None:
            options["top_k"] = top_k
        if rerank_top_k is not None:
            options["rerank_top_k"] = rerank_top_k
        if graph_scope is not None:
            options["graph_scope"] = graph_scope
        if score_threshold is not None:
            options["score_threshold"] = score_threshold
        if max_steps is not None:
            options["max_steps"] = max_steps
        if agent_vector_enabled is not None:
            options["agent_vector_enabled"] = agent_vector_enabled
        if agent_graph_enabled is not None:
            options["agent_graph_enabled"] = agent_graph_enabled
        if temperature is not None:
            options["temperature"] = temperature
        for key, value in {
            "rewrite_enabled": rewrite_enabled,
            "retrieve_enabled": retrieve_enabled,
            "rerank_enabled": rerank_enabled,
            "compress_enabled": compress_enabled,
            "verify_enabled": verify_enabled,
        }.items():
            if value is not None:
                options[key] = value
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

    def _is_direct_mode(self, rag_mode: Optional[str]) -> bool:
        mode = str(rag_mode or "").strip().lower()
        return mode in {DIRECT_RAG_MODE, "none", "no_rag", "chat"}

    def _is_model_identity_question(self, question: str) -> bool:
        text = (question or "").strip().lower().replace("？", "?")
        compact = "".join(text.split())
        if not compact:
            return False
        identity_patterns = (
            "你是什么模型",
            "你是哪个模型",
            "你用的什么模型",
            "现在是什么模型",
            "当前是什么模型",
            "当前模型",
            "什么模型",
            "whichmodel",
            "whatmodel",
            "modelareyou",
        )
        return len(compact) <= 40 and any(pattern in compact for pattern in identity_patterns)

    async def _stream_model_identity(
        self,
        *,
        model_id: Optional[str],
        model_config: Optional[Any],
        request_id: Optional[str],
        started_at: float,
    ) -> AsyncIterator[ChatStreamEvent]:
        provider = getattr(model_config, "provider", None) or "default"
        model_name = getattr(model_config, "model_name", None) or model_id or "default"
        display_name = getattr(model_config, "display_name", None) or model_id or model_name
        answer = (
            f"当前会话正在使用 {display_name}。"
            f"模型 ID：{model_id or model_name}；Provider：{provider}；Model Name：{model_name}。"
        )
        trace = [{
            "stage": "model",
            "duration_ms": 0,
            "input_summary": "model identity question",
            "output_summary": f"Answered from current chat model config: {model_id or model_name}",
            "metadata": {
                "model_id": model_id or model_name,
                "provider": provider,
                "model_name": model_name,
            },
        }]
        yield {"type": "trace", "trace": trace}
        yield {"type": "chunk", "content": answer}
        yield {
            "type": "done",
            "content": answer,
            "citations": [],
            "trace": trace,
            "latency_ms": int((time.monotonic() - started_at) * 1000),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_id": model_id or model_name,
            "request_id": request_id,
            "retry_count": 0,
        }

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
