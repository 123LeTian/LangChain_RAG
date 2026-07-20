"""Stable, KB-isolated multi-query retrieval and result fusion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.models.schemas import RetrievalHit


class MultiQueryRetrievalError(RuntimeError):
    """Raised when every query retrieval attempt fails."""


@dataclass(frozen=True)
class QueryRetrievalReport:
    hits: List[RetrievalHit]
    hit_counts: Dict[str, int]
    failures: List[str] = field(default_factory=list)


def _validate_hits(hits: Any, kb_id: str) -> List[RetrievalHit]:
    if not isinstance(hits, list):
        raise TypeError("retriever.search() must return a list")
    for hit in hits:
        if not isinstance(hit, RetrievalHit):
            raise TypeError("retriever.search() must return RetrievalHit objects")
        if hit.chunk.kb_id != kb_id:
            raise ValueError(
                f"retriever returned chunk '{hit.chunk.id}' from another kb"
            )
    return hits


def _matched_queries(hit: RetrievalHit) -> List[str]:
    raw = hit.metadata.get("matched_queries", [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [query for query in raw if isinstance(query, str)]
    return []


def merge_retrieval_hits(
    hit_groups: Iterable[Sequence[RetrievalHit]],
    *,
    top_k: int,
) -> List[RetrievalHit]:
    """Deduplicate by chunk ID, retaining the highest score and stable order."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    merged: Dict[str, Dict[str, Any]] = {}
    first_seen = 0
    for group in hit_groups:
        for hit in group:
            chunk_id = hit.chunk.id
            queries = _matched_queries(hit)
            entry = merged.get(chunk_id)
            if entry is None:
                copied = deepcopy(hit)
                entry = {
                    "hit": copied,
                    "first_seen": first_seen,
                    "matched_queries": [],
                }
                merged[chunk_id] = entry
                first_seen += 1
            elif hit.score > entry["hit"].score:
                copied = deepcopy(hit)
                entry["hit"] = copied
            for query in queries:
                if query not in entry["matched_queries"]:
                    entry["matched_queries"].append(query)

    ordered = sorted(
        merged.values(),
        key=lambda entry: (
            -entry["hit"].score,
            entry["first_seen"],
            entry["hit"].chunk.id,
        ),
    )[:top_k]
    result: List[RetrievalHit] = []
    for rank, entry in enumerate(ordered, start=1):
        hit = entry["hit"]
        hit.metadata = deepcopy(hit.metadata)
        hit.metadata["matched_queries"] = list(entry["matched_queries"])
        hit.metadata.setdefault("source_rank", hit.rank)
        hit.rank = rank
        result.append(hit)
    return result


async def retrieve_queries(
    retriever: Any,
    queries: Sequence[str],
    *,
    kb_id: str,
    top_k: int,
    filters: Mapping[str, Any] | None = None,
) -> QueryRetrievalReport:
    """Search each query independently; partial failures do not abort the rest."""

    search = getattr(retriever, "search", None)
    if not callable(search):
        raise MultiQueryRetrievalError("retriever must expose search()")

    hit_groups: List[List[RetrievalHit]] = []
    hit_counts: Dict[str, int] = {}
    failures: List[str] = []
    successful_queries = 0
    for query in queries:
        try:
            raw = search(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
                filters=dict(filters or {}),
            )
            if inspect.isawaitable(raw):
                raw = await raw
            hits = _validate_hits(raw, kb_id)
            successful_queries += 1
            hit_counts[query] = len(hits)
            copied_hits = []
            for hit in hits:
                copied = deepcopy(hit)
                copied.metadata = deepcopy(copied.metadata)
                copied.metadata["matched_queries"] = [query]
                copied_hits.append(copied)
            hit_groups.append(copied_hits)
        except Exception as exc:
            detail = str(exc).replace("\n", " ").strip()[:160]
            failures.append(
                f"query retrieval failed ({type(exc).__name__}): {detail}"
            )

    if successful_queries == 0:
        detail = "; ".join(failures) or "no query was attempted"
        raise MultiQueryRetrievalError(detail)
    return QueryRetrievalReport(
        hits=merge_retrieval_hits(hit_groups, top_k=top_k),
        hit_counts=hit_counts,
        failures=failures,
    )


__all__ = [
    "MultiQueryRetrievalError",
    "QueryRetrievalReport",
    "merge_retrieval_hits",
    "retrieve_queries",
]
