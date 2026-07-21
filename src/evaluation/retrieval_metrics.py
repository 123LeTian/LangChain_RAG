"""Retrieval metrics for RAG evaluation."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.evaluation.dataset import ExpectedSource


SOURCE_KEYS = ("document_id", "chunk_id", "filename")


def hit_at_k(
    hits: Sequence[Any],
    expected_sources: Sequence[Any],
    *,
    k: int,
) -> float:
    """Return 1.0 if any expected source appears in the top-k hits."""

    if k <= 0 or not hits or not expected_sources:
        return 0.0
    top_hits = list(hits)[:k]
    for hit in top_hits:
        if any(source_matches(expected, hit) for expected in expected_sources):
            return 1.0
    return 0.0


def mrr(hits: Sequence[Any], expected_sources: Sequence[Any]) -> float:
    """Return reciprocal rank for the first hit matching any expected source."""

    if not hits or not expected_sources:
        return 0.0
    for index, hit in enumerate(hits, start=1):
        if any(source_matches(expected, hit) for expected in expected_sources):
            return 1.0 / index
    return 0.0


def reciprocal_rank(hits: Sequence[Any], expected_sources: Sequence[Any]) -> float:
    """Alias for MRR at the single-query level."""

    return mrr(hits, expected_sources)


def source_matches(expected: Any, candidate: Any) -> bool:
    """Return whether a returned hit/citation points at an expected source.

    Matching is identifier based. At least one shared identifier must match, and
    any other shared identifiers must not conflict. This lets callers match by
    document_id, chunk_id, or filename without requiring every runner to expose
    every possible source field.
    """

    expected_fields = _extract_source_fields(expected)
    candidate_fields = _extract_source_fields(candidate)
    matches = 0

    for key in SOURCE_KEYS:
        expected_value = expected_fields.get(key)
        candidate_value = candidate_fields.get(key)
        if expected_value is None or candidate_value is None:
            continue
        if expected_value != candidate_value:
            return False
        matches += 1

    return matches > 0


def _extract_source_fields(value: Any) -> Dict[str, str | None]:
    if isinstance(value, ExpectedSource):
        data: Dict[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        data = _object_to_mapping(value)

    nested_candidates = []
    if isinstance(data.get("metadata"), Mapping):
        nested_candidates.append(dict(data["metadata"]))
    if isinstance(data.get("source"), Mapping):
        nested_candidates.append(dict(data["source"]))
    if isinstance(data.get("chunk"), Mapping):
        chunk_data = dict(data["chunk"])
        nested_candidates.append(chunk_data)
        if isinstance(chunk_data.get("metadata"), Mapping):
            nested_candidates.append(dict(chunk_data["metadata"]))

    chunk_obj = getattr(value, "chunk", None)
    source_obj = getattr(value, "source", None)
    if chunk_obj is not None:
        nested_candidates.append(_object_to_mapping(chunk_obj))
        nested_source = getattr(chunk_obj, "source", None)
        if nested_source is not None:
            nested_candidates.append(_object_to_mapping(nested_source))
    if source_obj is not None:
        nested_candidates.append(_object_to_mapping(source_obj))

    merged: Dict[str, Any] = {}
    for nested in reversed(nested_candidates):
        merged.update(nested)
    merged.update(data)

    chunk_mapping = data.get("chunk")
    if not isinstance(chunk_mapping, Mapping) and chunk_obj is not None:
        chunk_mapping = _object_to_mapping(chunk_obj)

    return {
        "document_id": _normalize_identifier(
            _first_present(merged, "document_id", "doc_id")
        ),
        "chunk_id": _normalize_identifier(
            _first_present(merged, "chunk_id", "id")
            or (
                _first_present(chunk_mapping, "id", "chunk_id")
                if isinstance(chunk_mapping, Mapping)
                else None
            )
        ),
        "filename": _normalize_filename(
            _first_present(
                merged,
                "filename",
                "source_path",
                "source",
                "path",
                "title",
            )
        ),
    }


def _object_to_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {
        name: getattr(value, name)
        for name in (
            "document_id",
            "chunk_id",
            "filename",
            "source_path",
            "source",
            "title",
            "id",
            "chunk",
        )
        if hasattr(value, name)
    }


def _first_present(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _normalize_filename(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    return PurePath(text).name.lower()


__all__ = [
    "hit_at_k",
    "mrr",
    "reciprocal_rank",
    "source_matches",
]
