"""Retry policy for chat stream RAG calls."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional, TypeVar

from src.chat.model_runtime import ModelRuntimeError


T = TypeVar("T")


class ChatRetryExhaustedError(Exception):
    def __init__(self, original: Exception, retry_count: int):
        super().__init__(str(original))
        self.original = original
        self.retry_count = retry_count


@dataclass
class RetryPolicy:
    timeout_seconds: float = 60
    max_retries: int = 2
    backoff_seconds: float = 1.5
    retryable_errors: list[str] = field(
        default_factory=lambda: [
            "TimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectionError",
            "ServiceUnavailable",
            "RateLimitError",
            "TooManyRequests",
            "InternalServerError",
        ]
    )

    async def run(
        self,
        operation: Callable[[], Awaitable[T] | T],
        *,
        logger: Any = None,
        log_payload: Optional[dict[str, Any]] = None,
    ) -> tuple[T, int]:
        attempts = 0
        while True:
            try:
                result = operation()
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=self.timeout_seconds)
                return result, attempts
            except Exception as exc:
                if not self.is_retryable(exc) or attempts >= self.max_retries:
                    raise ChatRetryExhaustedError(exc, attempts) from exc
                attempts += 1
                if logger is not None:
                    logger.warning(
                        "chat_stream_retry",
                        **(log_payload or {}),
                        error_type=type(exc).__name__,
                        message=str(exc),
                        retry_count=attempts,
                    )
                await asyncio.sleep(self.backoff_seconds * attempts)

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, ModelRuntimeError):
            return False
        text = f"{type(exc).__name__} {str(exc)}".lower()
        non_retryable = [
            "api key",
            "api_key",
            "missing",
            "缺少环境变量",
            "authentication",
            "unauthorized",
            "invalid_api_key",
            "permission",
            "validation",
            "bad request",
            "401",
            "403",
            "422",
        ]
        if any(item in text for item in non_retryable):
            return False
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        retry_markers = [item.lower() for item in self.retryable_errors]
        return any(marker in text for marker in retry_markers) or any(
            code in text for code in ["429", "500", "502", "503", "504"]
        )


__all__ = ["ChatRetryExhaustedError", "RetryPolicy"]
