"""C-compatible graph search tool backed by D4 GraphRetriever."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from enum import Enum
from importlib import import_module
import math
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - pydantic is a project dependency
    BaseModel = ()  # type: ignore[assignment]

try:
    from src.agent.tools import BaseTool as _CBaseTool
except ImportError:
    class _CBaseTool:  # type: ignore[no-redef]
        """Structural fallback until C's frozen BaseTool is available."""

from src.graph.retriever import (
    GraphGlobalSearchResult,
    GraphLocalSearchResult,
    resolve_graph_scope,
)


TOOL_NAME = "graph_search"
TOOL_DESCRIPTION = (
    "Search the knowledge graph using local entity-relation search or global "
    "community-report search."
)
EMPTY_LOCAL_WARNING = "no matching graph entities found"
EMPTY_GLOBAL_WARNING = "no matching community reports found"

ResultFactory = Callable[..., Any]


class GraphSearchToolError(Exception):
    """Base error for graph tool configuration or execution failures."""

    retryable: bool = False


class InvalidGraphToolArgumentsError(GraphSearchToolError):
    """Raised when tool arguments violate the public parameter schema."""


class GraphSearchExecutionError(GraphSearchToolError):
    """Raised when the retriever or result conversion fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


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
            raise GraphSearchToolError(
                "C ToolResult is unavailable on this branch; merge C's frozen "
                "tool contract or inject result_factory"
            )
        return result_type(**fields)


class GraphSearchTool(_CBaseTool):
    """Expose GraphRetriever local/global search through C's ToolProtocol."""

    name = TOOL_NAME
    description = TOOL_DESCRIPTION

    def __init__(
        self,
        graph_retriever: Optional[Any],
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
        self._graph_retriever = graph_retriever
        self._max_top_k = max_top_k
        self._results = CToolResultAdapter(result_factory)

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        """Return a detached JSON Schema readable by C and LLM adapters."""

        return deepcopy(self._parameter_schema())

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The graph search query.",
                },
                "kb_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Required isolated knowledge-base identifier.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["local", "global", "auto"],
                    "default": "auto",
                    "description": "Graph search scope.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_top_k,
                    "default": 5,
                    "description": "Maximum number of graph results.",
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
        scope: str = "auto",
        top_k: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Synchronous compatibility entry point."""

        return self._invoke_core(
            query=query,
            kb_id=kb_id,
            scope=scope,
            top_k=top_k,
            extra=kwargs,
        )

    async def ainvoke(
        self,
        query: str,
        kb_id: str,
        scope: str = "auto",
        top_k: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Async compatibility entry point."""

        return self._invoke_core(
            query=query,
            kb_id=kb_id,
            scope=scope,
            top_k=top_k,
            extra=kwargs,
        )

    async def execute(
        self,
        query: str = "",
        kb_id: str = "",
        scope: str = "auto",
        top_k: int = 5,
        **kwargs: Any,
    ) -> Any:
        """C ToolProtocol entry point; always returns a canonical ToolResult."""

        return await self.ainvoke(
            query=query,
            kb_id=kb_id,
            scope=scope,
            top_k=top_k,
            **kwargs,
        )

    def _invoke_core(
        self,
        *,
        query: Any,
        kb_id: Any,
        scope: Any,
        top_k: Any,
        extra: Mapping[str, Any],
    ) -> Any:
        started = time.perf_counter()
        argument_summary = self._argument_summary(query, kb_id, scope, top_k)
        try:
            query, kb_id, requested_scope, resolved_scope, top_k = (
                self._validate_arguments(query, kb_id, scope, top_k, extra)
            )
        except Exception as exc:
            return self._error_result(
                exc,
                started=started,
                argument_summary=argument_summary,
                retryable=False,
            )

        method_name = "global_search" if resolved_scope == "global" else "local_search"
        search = getattr(self._graph_retriever, method_name, None)
        if self._graph_retriever is None or not callable(search):
            return self._error_result(
                GraphSearchExecutionError(
                    f"GraphRetriever.{method_name} is not configured",
                    retryable=False,
                ),
                started=started,
                argument_summary=argument_summary,
                retryable=False,
            )

        try:
            raw_result = search(query, kb_id=kb_id, top_k=top_k)
            data = self._convert_result(
                raw_result,
                query=query,
                kb_id=kb_id,
                requested_scope=requested_scope,
                resolved_scope=resolved_scope,
            )
        except Exception as exc:
            retryable = isinstance(exc, (ConnectionError, OSError, TimeoutError))
            wrapped = GraphSearchExecutionError(
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
        result = self._results.build(
            success=True,
            data=data,
            error=None,
            duration_ms=duration_ms,
            tool_name=self.name,
        )
        return _attach_trace_metadata(
            result,
            argument_summary=argument_summary,
            result_count=data["result_count"],
            result_summary=(
                f"graph {resolved_scope} search returned "
                f"{data['result_count']} result(s)"
            ),
            error_type=None,
            retryable=False,
        )

    def _validate_arguments(
        self,
        query: Any,
        kb_id: Any,
        scope: Any,
        top_k: Any,
        extra: Mapping[str, Any],
    ) -> tuple[str, str, str, str, int]:
        if extra:
            unknown = ", ".join(sorted(str(key) for key in extra))
            raise InvalidGraphToolArgumentsError(
                f"unknown tool argument(s): {unknown}"
            )
        if not isinstance(query, str) or not query.strip():
            raise InvalidGraphToolArgumentsError(
                "query must be a non-empty string"
            )
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise InvalidGraphToolArgumentsError(
                "kb_id must be a non-empty string"
            )
        if not isinstance(scope, str) or not scope.strip():
            raise InvalidGraphToolArgumentsError(
                "scope must be one of: local, global, auto"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidGraphToolArgumentsError("top_k must be a positive integer")
        if top_k > self._max_top_k:
            raise InvalidGraphToolArgumentsError(
                f"top_k must not exceed {self._max_top_k}"
            )
        normalized_query = query.strip()
        normalized_kb_id = kb_id.strip()
        requested_scope = scope.strip().lower()
        try:
            resolved_scope = resolve_graph_scope(normalized_query, requested_scope)
        except ValueError as exc:
            raise InvalidGraphToolArgumentsError(str(exc)) from exc
        return (
            normalized_query,
            normalized_kb_id,
            requested_scope,
            resolved_scope,
            top_k,
        )

    def _convert_result(
        self,
        raw_result: Any,
        *,
        query: str,
        kb_id: str,
        requested_scope: str,
        resolved_scope: str,
    ) -> Dict[str, Any]:
        if resolved_scope == "global":
            if not isinstance(raw_result, GraphGlobalSearchResult):
                raise GraphSearchExecutionError(
                    "GraphRetriever.global_search must return GraphGlobalSearchResult"
                )
            items = [
                self._global_hit_to_item(hit, query=query)
                for hit in raw_result.hits
            ]
        else:
            if not isinstance(raw_result, GraphLocalSearchResult):
                raise GraphSearchExecutionError(
                    "GraphRetriever.local_search must return GraphLocalSearchResult"
                )
            items = [self._local_hit_to_item(hit) for hit in raw_result.hits]

        citations = _citations_from_results(items)
        warnings = list(raw_result.warnings)
        if not items and not warnings:
            warnings = [
                EMPTY_GLOBAL_WARNING
                if resolved_scope == "global"
                else EMPTY_LOCAL_WARNING
            ]

        data = {
            "query": query,
            "kb_id": kb_id,
            "scope": resolved_scope,
            "requested_scope": requested_scope,
            "result_count": len(items),
            "results": items,
            "citations": citations,
            "warnings": _json_safe(warnings),
        }
        return _json_safe(data)

    @staticmethod
    def _local_hit_to_item(hit: Any) -> Dict[str, Any]:
        return {
            "score": _json_safe(hit.score),
            "graph_scope": "local",
            "entity_id": hit.entity_id,
            "entity_name": hit.entity_name,
            "community_id": None,
            "report_id": None,
            "title": hit.entity_name,
            "summary": hit.matched_text,
            "relationships": [_model_dump(item) for item in hit.relationships],
            "neighbor_entities": [_model_dump(item) for item in hit.neighbor_entities],
            "path": list(hit.path),
            "source_refs": [_source_ref_to_dict(ref) for ref in hit.source_refs],
            "key_entities": [],
            "key_relationships": [],
            "metadata": _json_safe(hit.metadata),
        }

    @staticmethod
    def _global_hit_to_item(hit: Any, *, query: str) -> Dict[str, Any]:
        source_refs = [_source_ref_to_dict(ref) for ref in hit.source_refs]
        path = [f"query:{query}", f"community:{hit.community_id}", f"report:{hit.report_id}"]
        if source_refs:
            first = source_refs[0]
            path.append(f"source_chunk:{first['document_id']}/{first['chunk_id']}")
        return {
            "score": _json_safe(hit.score),
            "graph_scope": "global",
            "entity_id": None,
            "entity_name": None,
            "community_id": hit.community_id,
            "report_id": hit.report_id,
            "title": hit.title,
            "summary": hit.summary,
            "relationships": [],
            "neighbor_entities": [],
            "path": path,
            "source_refs": source_refs,
            "key_entities": _json_safe(list(hit.key_entities)),
            "key_relationships": _json_safe(list(hit.key_relationships)),
            "metadata": _json_safe(hit.metadata),
        }

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
            data={},
            error=error_text,
            duration_ms=duration_ms,
            tool_name=self.name,
        )
        return _attach_trace_metadata(
            result,
            argument_summary=argument_summary,
            result_count=0,
            result_summary="graph search failed",
            error_type=error_type,
            retryable=retryable,
        )

    @staticmethod
    def _argument_summary(query: Any, kb_id: Any, scope: Any, top_k: Any) -> str:
        query_chars = len(query) if isinstance(query, str) else "invalid"
        safe_kb_id = kb_id if isinstance(kb_id, str) else "invalid"
        safe_scope = scope if isinstance(scope, str) else "invalid"
        return (
            f"query_chars={query_chars}, kb_id={safe_kb_id!r}, "
            f"scope={safe_scope!r}, top_k={top_k!r}"
        )


def register_graph_search_tool(
    registry: Any,
    graph_retriever: Any,
    *,
    max_top_k: int = 100,
    result_factory: Optional[ResultFactory] = None,
) -> GraphSearchTool:
    """Register an instance using C's replacement-on-duplicate semantics."""

    register = getattr(registry, "register", None)
    if not callable(register):
        raise TypeError("registry must expose register(tool)")
    tool = GraphSearchTool(
        graph_retriever,
        max_top_k=max_top_k,
        result_factory=result_factory,
    )
    register(tool)
    return tool


def graph_search(
    graph_retriever: Any,
    query: str,
    kb_id: str,
    scope: str = "auto",
    top_k: int = 5,
    *,
    max_top_k: int = 100,
    result_factory: Optional[ResultFactory] = None,
) -> Any:
    """Functional synchronous entry point backed by GraphSearchTool."""

    return GraphSearchTool(
        graph_retriever,
        max_top_k=max_top_k,
        result_factory=result_factory,
    ).invoke(query=query, kb_id=kb_id, scope=scope, top_k=top_k)


def _source_ref_to_dict(ref: Any) -> Dict[str, Any]:
    value = _model_dump(ref)
    document_id = value.get("document_id")
    chunk_id = value.get("chunk_id")
    if not document_id or not chunk_id:
        raise GraphSearchExecutionError(
            "graph source reference is missing document_id or chunk_id"
        )
    return {
        "document_id": _json_safe(document_id),
        "chunk_id": _json_safe(chunk_id),
        "filename": _json_safe(value.get("filename")),
        "page": _json_safe(value.get("page")),
        "section": _json_safe(value.get("section")),
        "quote": _json_safe(value.get("quote", "")),
        "metadata": _json_safe(value.get("metadata", {})),
    }


def _citations_from_results(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen = set()
    for item in items:
        refs = item.get("source_refs") or []
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
            continue
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            key = (ref.get("document_id"), ref.get("chunk_id"))
            if key in seen or not key[0] or not key[1]:
                continue
            seen.add(key)
            citations.append(
                {
                    "document_id": ref.get("document_id"),
                    "chunk_id": ref.get("chunk_id"),
                    "filename": ref.get("filename"),
                    "page": ref.get("page"),
                    "section": ref.get("section"),
                    "quote": ref.get("quote", ""),
                }
            )
    return _json_safe(citations)


def _model_dump(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise GraphSearchExecutionError(
        f"graph result contains unsupported object: {type(value).__name__}"
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
    return message[:240] or "graph search failed"


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
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    return str(value)


__all__ = [
    "CToolResultAdapter",
    "EMPTY_GLOBAL_WARNING",
    "EMPTY_LOCAL_WARNING",
    "GraphSearchExecutionError",
    "GraphSearchTool",
    "GraphSearchToolError",
    "InvalidGraphToolArgumentsError",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "graph_search",
    "register_graph_search_tool",
]
