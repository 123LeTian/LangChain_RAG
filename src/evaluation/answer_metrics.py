"""Rule-based answer quality metrics for RAG evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from src.evaluation.dataset import EvaluationSample

_REFUSAL_HINTS = (
    "insufficient",
    "cannot",
    "can't",
    "refuse",
    "unable",
    "no evidence",
    "clarification",
    "avoid fabricating",
    "hallucinat",
    "do not know",
    "don't know",
    "not enough",
)


def _normalize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 3]


def answer_point_coverage(answer: str, answer_points: Sequence[str]) -> float:
    """Fraction of answer points whose keywords appear in the answer."""

    if not answer_points:
        return 0.0
    answer_tokens = set(_normalize_keywords(answer or ""))
    if not answer_tokens:
        return 0.0
    covered = 0
    for point in answer_points:
        point_tokens = _normalize_keywords(point)
        if not point_tokens:
            continue
        hits = sum(1 for token in point_tokens if token in answer_tokens)
        if hits / len(point_tokens) >= 0.5:
            covered += 1
    return round(covered / len(answer_points), 6)


def citation_presence_rate(per_sample_results: Sequence[Dict[str, Any]]) -> float:
    """Share of samples with at least one citation."""

    if not per_sample_results:
        return 0.0
    with_citations = sum(
        1 for result in per_sample_results if result.get("citations")
    )
    return round(with_citations / len(per_sample_results), 6)


def unanswerable_refusal_rate(
    samples: Sequence[EvaluationSample],
    per_sample_results: Sequence[Dict[str, Any]],
) -> float | None:
    """Refusal rate on samples tagged ``unanswerable``; null if none present."""

    by_id = {sample.id: sample for sample in samples}
    unanswerable_ids = [
        sample.id for sample in samples if "unanswerable" in sample.tags
    ]
    if not unanswerable_ids:
        return None
    refused = 0
    for result in per_sample_results:
        if result.get("id") not in unanswerable_ids:
            continue
        answer = str(result.get("answer") or "").lower()
        sample = by_id[result["id"]]
        if any(hint in answer for hint in _REFUSAL_HINTS):
            refused += 1
            continue
        for point in sample.answer_points:
            if any(hint in point.lower() for hint in _REFUSAL_HINTS):
                if answer_point_coverage(answer, [point]) >= 1.0:
                    refused += 1
                    break
    return round(refused / len(unanswerable_ids), 6)


def enrich_sample_answer_metrics(
    sample: EvaluationSample,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach per-sample answer metrics to an existing result dict."""

    answer = str(result.get("answer") or "")
    enriched = dict(result)
    enriched["tags"] = list(sample.tags)
    enriched["answer_point_coverage"] = answer_point_coverage(
        answer, sample.answer_points
    )
    enriched["citation_present"] = bool(result.get("citations"))
    return enriched


__all__ = [
    "answer_point_coverage",
    "citation_presence_rate",
    "enrich_sample_answer_metrics",
    "unanswerable_refusal_rate",
]
