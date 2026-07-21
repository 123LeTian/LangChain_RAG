"""Structured report generation for RAG evaluation runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from src.evaluation.dataset import EvaluationSample


def build_evaluation_report(
    *,
    dataset_path: str,
    runner_type: str,
    samples: Sequence[EvaluationSample],
    per_sample_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON-serializable report for a completed evaluation run."""

    sample_count = len(samples)
    failed = [
        result["id"]
        for result in per_sample_results
        if result.get("error")
    ]
    return {
        "dataset_path": dataset_path,
        "runner_type": runner_type,
        "sample_count": sample_count,
        "tag_distribution": _tag_distribution(samples),
        "hit_at_1": _average(per_sample_results, "hit_at_1"),
        "hit_at_3": _average(per_sample_results, "hit_at_3"),
        "hit_at_5": _average(per_sample_results, "hit_at_5"),
        "mrr": _average(per_sample_results, "reciprocal_rank"),
        "average_latency_ms": _average(per_sample_results, "latency_ms"),
        "failed_samples": failed,
        "per_sample_results": list(per_sample_results),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report(report: Dict[str, Any], path: str | Path) -> Path:
    """Write an evaluation report as pretty JSON."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _tag_distribution(samples: Sequence[EvaluationSample]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(sample.tags)
    return dict(sorted(counter.items()))


def _average(results: Sequence[Dict[str, Any]], field: str) -> float:
    values = [
        float(result[field])
        for result in results
        if isinstance(result.get(field), (int, float))
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


__all__ = ["build_evaluation_report", "write_report"]
