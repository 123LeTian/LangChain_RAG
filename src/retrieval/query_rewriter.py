"""Dependency-injected query rewriting helpers for Advanced RAG."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Iterable, List, Protocol, Union


RewriteCallable = Callable[
    [str],
    Union[Iterable[str], Awaitable[Iterable[str]]],
]


class QueryRewriterProtocol(Protocol):
    """Structural contract accepted by :class:`AdvancedRAGStrategy`."""

    async def rewrite(
        self,
        query: str,
        *,
        max_queries: int = 3,
    ) -> List[str]: ...


def normalize_queries(
    original_query: str,
    rewritten_queries: Iterable[str] | None,
    *,
    max_queries: int,
) -> List[str]:
    """Keep the original query and stable, non-empty, unique rewrites."""

    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("original_query must be a non-empty string")
    if not isinstance(max_queries, int) or isinstance(max_queries, bool):
        raise TypeError("max_queries must be an integer")
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    normalized = [original_query.strip()]
    seen = {normalized[0]}
    for candidate in rewritten_queries or []:
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
        if len(normalized) >= max_queries:
            break
    return normalized


class QueryRewriter:
    """Adapt an injected sync or async rewrite callable without creating an LLM."""

    def __init__(self, rewrite_callable: RewriteCallable) -> None:
        if not callable(rewrite_callable):
            raise TypeError("rewrite_callable must be callable")
        self._rewrite_callable = rewrite_callable

    async def rewrite(
        self,
        query: str,
        *,
        max_queries: int = 3,
    ) -> List[str]:
        raw: Any = self._rewrite_callable(query)
        if inspect.isawaitable(raw):
            raw = await raw
        if raw is None:
            raw = []
        if isinstance(raw, str) or not isinstance(raw, Iterable):
            raise TypeError("query rewrite result must be an iterable of strings")
        return normalize_queries(query, raw, max_queries=max_queries)


async def rewrite_queries(
    rewriter: Any,
    query: str,
    *,
    max_queries: int,
) -> List[str]:
    """Call any compatible injected rewriter and normalize its result."""

    method = getattr(rewriter, "rewrite", None)
    if not callable(method):
        raise TypeError("query_rewriter must expose rewrite()")
    raw = method(query, max_queries=max_queries)
    if inspect.isawaitable(raw):
        raw = await raw
    if raw is None:
        raw = []
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise TypeError("query rewrite result must be an iterable of strings")
    return normalize_queries(query, raw, max_queries=max_queries)


__all__ = [
    "QueryRewriter",
    "QueryRewriterProtocol",
    "normalize_queries",
    "rewrite_queries",
]
