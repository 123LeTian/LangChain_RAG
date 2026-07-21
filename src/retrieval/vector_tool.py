"""C-compatible, KB-isolated vector search tool (B6)."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from enum import Enum
from importlib import import_module
import math
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.models.schemas import RetrievalHit

from .adapter import MissingKnowledgeBaseError

try:
    from src.agent.tools import BaseTool as _CBaseTool
except ImportError:
    class _CBaseTool:  # type: ignore[no-redef]
        """Structural fallback until C's frozen BaseTool is merged."""


TOOL_NAME = "vector_search"
TOOL_DESCRIPTION = (
    "在指定知识库中进行语义向量检索，适用于事实、定义和文本证据查询。"
)
EMPTY_RESULT_SUMMARY = "指定知识库中未检索到相关内容"


class VectorSearchToolError(Exception):
    """Base error for vector tool configuration or execution failures."""

    retryable: bool = False


class InvalidToolArgumentsError(VectorSearchToolError):
    """Raised when tool arguments violate the public parameter schema."""


class VectorSearchExecutionError(VectorSearchToolError):
    """Raised when the retriever or hit conversion fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


ResultFactory = Callable[..., Any]


class CToolResultAdapter:
    """Lazily construct C's canonical ToolResult without duplicating it."""

    def __init__(self, result_factory: Optional[ResultFactory] = None) -> None:
        self._result_factory = result_factory

    def build(self, **fields: Any) -> Any:
        if self._result_factory is not None:
            return self._result_factory(**fields)
        module = import_module("src.agent.tools")
        result_type = getattr(module, "ToolResult", None)
        if result_type is None:
            raise VectorSearchToolError(
                "C ToolResult is unavailable on this branch; merge C's frozen "
                "tool contract or inject result_factory"
            )
        return result_type(**fields)


class VectorSearchTool(_CBaseTool):
    """Invoke only an injected ``VectorRetriever.search`` implementation."""

    name = TOOL_NAME
    description = TOOL_DESCRIPTION

    def __init__(
        self,
        retriever: Optional[Any],
        *,
        max_top_k: int = 100,
        result_factory: Optional[ResultFactory] = None,
    ) -> None:
        if (
            not isinstance(max_top_k, int)
            or isinstance(max_top_k, bool)
            or max_top_k <= 0
        ):
            raise ValueError("max_top_k must be a positive integer")
        self._retriever = retriever
        self._max_top_k = max_top_k
        self._results = CToolResultAdapter(result_factory)

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        """Return a detached JSON Schema readable by C and LLM tool adapters."""

        return deepcopy(self._parameter_schema())

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The semantic vector search query.",
                },
                "kb_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Required isolated knowledge-base identifier.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_top_k,
                    "default": 5,
                    "description": "Maximum number of results.",
                },
                "filters": {
                    "type": ["object", "null"],
                    "description": "Optional metadata equality filters.",
                    "additionalProperties": True,
                },
            },
            "required": ["query", "kb_id"],
            "additionalProperties": False,
        }

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameter_schema,
            },
        }

    def invoke(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous compatibility entry point; performs one retriever call."""

        return self._invoke_core(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
            extra=kwargs,
        )

    async def ainvoke(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Async compatibility entry point without duplicating core execution."""

        return self._invoke_core(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
            extra=kwargs,
        )

    async def execute(
        self,
        query: str = "",
        kb_id: str = "",
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """C ToolProtocol entry point; always returns a canonical ToolResult."""

        return await self.ainvoke(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
            **kwargs,
        )

    def _invoke_core(
        self,
        *,
        query: Any,
        kb_id: Any,
        top_k: Any,
        filters: Any,
        extra: Mapping[str, Any],
    ) -> Any:
        started = time.perf_counter()
        argument_summary = self._argument_summary(query, kb_id, top_k, filters)
        try:
            query, kb_id, top_k, filters = self._validate_arguments(
                query,
                kb_id,
                top_k,
                filters,
                extra,
            )
        except Exception as exc:
            return self._error_result(
                exc,
                started=started,
                argument_summary=argument_summary,
                retryable=False,
            )

        if self._retriever is None or not callable(
            getattr(self._retriever, "search", None)
        ):
            return self._error_result(
                VectorSearchExecutionError(
                    "VectorRetriever.search is not configured",
                    retryable=False,
                ),
                started=started,
                argument_summary=argument_summary,
                retryable=False,
            )

        try:
            hits = self._retriever.search(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
                filters=filters,
            )
            items = self._convert_hits(hits, kb_id)
        except Exception as exc:
            retryable = isinstance(
                exc,
                (ConnectionError, OSError, TimeoutError),
            )
            wrapped = VectorSearchExecutionError(
                _safe_error_summary(exc),
                retryable=retryable,
            )
            return self._error_result(
                wrapped,
                started=started,
                argument_summary=argument_summary,
                retryable=retryable,
                source_error_type=type(exc).__name__,
            )

        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        summary = (
            f"在知识库 {kb_id} 中检索到 {len(items)} 条相关内容"
            if items
            else EMPTY_RESULT_SUMMARY
        )
        result = self._results.build(
            success=True,
            data=items,
            error=None,
            duration_ms=duration_ms,
            tool_name=self.name,
        )
        return _attach_trace_metadata(
            result,
            argument_summary=argument_summary,
            result_count=len(items),
            result_summary=summary,
            error_type=None,
            retryable=False,
        )

    def _validate_arguments(
        self,
        query: Any,
        kb_id: Any,
        top_k: Any,
        filters: Any,
        extra: Mapping[str, Any],
    ) -> tuple[str, str, int, Optional[dict]]:
        if extra:
            unknown = ", ".join(sorted(str(key) for key in extra))
            raise InvalidToolArgumentsError(f"unknown tool argument(s): {unknown}")
        if not isinstance(query, str) or not query.strip():
            raise InvalidToolArgumentsError(
                "query must be a non-empty string"
            )
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise MissingKnowledgeBaseError()
        normalized_kb_id = kb_id.strip()
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidToolArgumentsError("top_k must be a positive integer")
        if top_k > self._max_top_k:
            raise InvalidToolArgumentsError(
                f"top_k must not exceed {self._max_top_k}"
            )
        if filters is not None and not isinstance(filters, dict):
            raise InvalidToolArgumentsError("filters must be a dict or None")
        if (
            filters is not None
            and filters.get("kb_id", normalized_kb_id) != normalized_kb_id
        ):
            raise InvalidToolArgumentsError(
                "filters cannot override the required kb_id"
            )
        return (
            query.strip(),
            normalized_kb_id,
            top_k,
            deepcopy(filters),
        )

    @staticmethod
    def _convert_hits(hits: Any, kb_id: str) -> List[Dict[str, Any]]:
        if not isinstance(hits, list):
            raise VectorSearchExecutionError(
                "VectorRetriever.search must return a list"
            )
        items = []
        for hit in hits:
            if not isinstance(hit, RetrievalHit):
                raise VectorSearchExecutionError(
                    "VectorRetriever.search must return RetrievalHit objects"
                )
            if hit.chunk.kb_id != kb_id:
                raise VectorSearchExecutionError(
                    f"retriever returned chunk '{hit.chunk.id}' from another kb"
                )
            if not hit.chunk.id or not hit.chunk.document_id:
                raise VectorSearchExecutionError(
                    "retrieval hit is missing citation source identifiers"
                )
            if (
                not isinstance(hit.score, (int, float))
                or isinstance(hit.score, bool)
                or not math.isfinite(float(hit.score))
            ):
                raise VectorSearchExecutionError(
                    f"retrieval hit '{hit.chunk.id}' has an invalid score"
                )
            if (
                not isinstance(hit.rank, int)
                or isinstance(hit.rank, bool)
                or hit.rank <= 0
            ):
                raise VectorSearchExecutionError(
                    f"retrieval hit '{hit.chunk.id}' has an invalid rank"
                )
            metadata = deepcopy(hit.chunk.metadata)
            metadata.update(deepcopy(hit.metadata))
            filename = metadata.get("filename") or None
            retriever_name = (
                hit.retriever
                or getattr(hit, "retriever_name", None)
                or "vector"
            )
            items.append(
                {
                    "chunk_id": hit.chunk.id,
                    "document_id": hit.chunk.document_id,
                    "kb_id": hit.chunk.kb_id,
                    "text": hit.chunk.text,
                    "content": hit.chunk.text,
                    "score": float(hit.score),
                    "rank": hit.rank,
                    "retriever": retriever_name,
                    "source": hit.chunk.document_id,
                    "filename": _json_safe(filename),
                    "page": _json_safe(metadata.get("page")),
                    "section": _json_safe(metadata.get("section")),
                    "metadata": _json_safe(metadata),
                }
            )
        return items

    def _error_result(
        self,
        error: Exception,
        *,
        started: float,
        argument_summary: str,
        retryable: bool,
        source_error_type: Optional[str] = None,
    ) -> Any:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        error_type = source_error_type or type(error).__name__
        summary = _safe_error_summary(error)
        error_text = (
            f"tool={self.name}; type={error_type}; "
            f"retryable={str(retryable).lower()}; summary={summary}"
        )
        result = self._results.build(
            success=False,
            data=None,
            error=error_text,
            duration_ms=duration_ms,
            tool_name=self.name,
        )
        return _attach_trace_metadata(
            result,
            argument_summary=argument_summary,
            result_count=0,
            result_summary="vector search failed",
            error_type=error_type,
            retryable=retryable,
        )

    @staticmethod
    def _argument_summary(
        query: Any,
        kb_id: Any,
        top_k: Any,
        filters: Any,
    ) -> str:
        query_chars = len(query) if isinstance(query, str) else "invalid"
        filter_keys = (
            sorted(str(key) for key in filters)
            if isinstance(filters, dict)
            else []
        )
        safe_kb_id = kb_id if isinstance(kb_id, str) else "invalid"
        return (
            f"query_chars={query_chars}, kb_id={safe_kb_id!r}, "
            f"top_k={top_k!r}, filter_keys={filter_keys!r}"
        )


def register_vector_search_tool(
    registry: Any,
    retriever: Any,
    *,
    max_top_k: int = 100,
    result_factory: Optional[ResultFactory] = None,
) -> VectorSearchTool:
    """Register an instance using C's replacement-on-duplicate semantics."""

    register = getattr(registry, "register", None)
    if not callable(register):
        raise TypeError("registry must expose register(tool)")
    tool = VectorSearchTool(
        retriever,
        max_top_k=max_top_k,
        result_factory=result_factory,
    )
    register(tool)
    return tool


def vector_search(
    retriever: Any,
    query: str,
    kb_id: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
    *,
    max_top_k: int = 100,
    result_factory: Optional[ResultFactory] = None,
) -> Any:
    """Functional synchronous entry point backed by ``VectorSearchTool``."""

    return VectorSearchTool(
        retriever,
        max_top_k=max_top_k,
        result_factory=result_factory,
    ).invoke(
        query=query,
        kb_id=kb_id,
        top_k=top_k,
        filters=filters,
    )


def _attach_trace_metadata(
    result: Any,
    *,
    argument_summary: str,
    result_count: int,
    result_summary: str,
    error_type: Optional[str],
    retryable: bool,
) -> Any:
    """Attach optional executor-facing fields without changing C's frozen model."""

    values = {
        "argument_summary": argument_summary,
        "result_count": result_count,
        "result_summary": result_summary,
        "error_type": error_type,
        "retryable": retryable,
    }
    for name, value in values.items():
        try:
            setattr(result, name, value)
        except (AttributeError, TypeError):
            pass
    return result


def _safe_error_summary(error: Exception) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    if "Traceback (most recent call last)" in message:
        message = message.split("Traceback (most recent call last)", 1)[0]
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", message)
    message = re.sub(
        r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(r"[A-Za-z]:\\[^\s]+", "<path>", message)
    message = re.sub(r"/(?:[^\s/]+/)+[^\s]+", "<path>", message)
    return message[:240] or "vector search failed"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "CToolResultAdapter",
    "EMPTY_RESULT_SUMMARY",
    "InvalidToolArgumentsError",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "VectorSearchExecutionError",
    "VectorSearchTool",
    "VectorSearchToolError",
    "register_vector_search_tool",
    "vector_search",
]
