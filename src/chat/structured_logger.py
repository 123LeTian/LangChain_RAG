"""Small JSONL structured logger with secret redaction."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.config.logging_config import get_app_log_path, get_error_log_path


SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "set-cookie",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "deepseek_api_key",
    "openai_api_key",
    "dashscope_api_key",
}
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"\b(DEEPSEEK_API_KEY|OPENAI_API_KEY|DASHSCOPE_API_KEY)\b", re.I),
    re.compile(r"(?i)(api[_-]?key|authorization|access_token|refresh_token|secret|password)\s*[:=]\s*[\w.\-]+"),
]
REDACTED = "***REDACTED***"


class StructuredLogger:
    def __init__(
        self,
        *,
        app_log_path: Path | None = None,
        error_log_path: Path | None = None,
    ):
        self.app_log_path = app_log_path or get_app_log_path()
        self.error_log_path = error_log_path or get_error_log_path()

    def info(self, event: str, **payload: Any) -> None:
        self._write(self.app_log_path, "INFO", event, payload)

    def warning(self, event: str, **payload: Any) -> None:
        self._write(self.app_log_path, "WARNING", event, payload)

    def error(self, event: str, **payload: Any) -> None:
        self._write(self.error_log_path, "ERROR", event, payload)

    def _write(self, path: Path, level: str, event: str, payload: Mapping[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "event": event,
                **self._redact(payload),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def _redact(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            redacted = {}
            for key, item in value.items():
                key_text = str(key)
                if self._is_sensitive_key(key_text):
                    redacted[key_text] = REDACTED
                else:
                    redacted[key_text] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            redacted = value
            for pattern in SENSITIVE_VALUE_PATTERNS:
                redacted = pattern.sub(REDACTED, redacted)
            return redacted
        return value

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        return normalized in SENSITIVE_KEYS or any(part in normalized for part in SENSITIVE_KEYS)


__all__ = ["REDACTED", "SENSITIVE_KEYS", "StructuredLogger"]
