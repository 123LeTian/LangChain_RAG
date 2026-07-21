"""Graph-mode metrics for GraphRAG evaluation runs."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def _hit_has_graph_trace(hit: Any) -> bool:
    if not isinstance(hit, dict):
        return False
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if hit.get("graph_source_trace") or metadata.get("graph_source_trace"):
        return True
    if hit.get("graph_scope") or metadata.get("graph_scope"):
        return True
    if str(hit.get("retriever", "")).startswith("graph"):
        return True
    return False


def _hit_has_graph_path(hit: Any) -> bool:
    if not isinstance(hit, dict):
        return False
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    path = hit.get("graph_path") or metadata.get("graph_path")
    return isinstance(path, list) and len(path) > 0


def _hit_uses_community_report(hit: Any) -> bool:
    if not isinstance(hit, dict):
        return False
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if hit.get("community_report_used") or metadata.get("community_report_used"):
        return True
    if hit.get("graph_scope") == "global" or metadata.get("graph_scope") == "global":
        return True
    return False


def graph_source_trace_rate(per_sample_results: Sequence[Dict[str, Any]]) -> float:
    if not per_sample_results:
        return 0.0
    traced = 0
    for result in per_sample_results:
        hits = result.get("hits") or []
        if any(_hit_has_graph_trace(hit) for hit in hits):
            traced += 1
    return round(traced / len(per_sample_results), 6)


def graph_path_presence_rate(per_sample_results: Sequence[Dict[str, Any]]) -> float:
    if not per_sample_results:
        return 0.0
    with_path = 0
    for result in per_sample_results:
        hits = result.get("hits") or []
        if any(_hit_has_graph_path(hit) for hit in hits):
            with_path += 1
    return round(with_path / len(per_sample_results), 6)


def community_report_usage_rate(
    per_sample_results: Sequence[Dict[str, Any]],
) -> float:
    if not per_sample_results:
        return 0.0
    used = 0
    for result in per_sample_results:
        hits = result.get("hits") or []
        if any(_hit_uses_community_report(hit) for hit in hits):
            used += 1
    return round(used / len(per_sample_results), 6)


def summarize_graph_metrics(
    per_sample_results: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    return {
        "graph_source_trace_rate": graph_source_trace_rate(per_sample_results),
        "graph_path_presence_rate": graph_path_presence_rate(per_sample_results),
        "community_report_usage_rate": community_report_usage_rate(
            per_sample_results
        ),
    }


__all__ = [
    "community_report_usage_rate",
    "graph_path_presence_rate",
    "graph_source_trace_rate",
    "summarize_graph_metrics",
]
