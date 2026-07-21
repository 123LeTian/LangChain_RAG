"""Structured report generation for RAG evaluation runs."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.evaluation.dataset import EvaluationSample
from src.evaluation.regression import RegressionThresholds, regression_summary, run_regression_checks


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


def build_final_evaluation_report(
    *,
    dataset_path: str,
    modes: Sequence[str],
    samples: Sequence[EvaluationSample],
    per_mode_results: Sequence[Dict[str, Any]],
    regression_thresholds: RegressionThresholds | None = None,
) -> Dict[str, Any]:
    """Build the D6 multi-mode final evaluation report payload."""

    runner_summary = {
        mode_result.get("mode", ""): {
            "runner_type": mode_result.get("runner_type"),
            "status": mode_result.get("status", "ok"),
            "is_mock": str(mode_result.get("runner_type", "")).startswith("mock_"),
        }
        for mode_result in per_mode_results
    }
    mode_summaries = [
        _mode_summary_row(mode_result) for mode_result in per_mode_results
    ]
    mock_relaxed = all(
        str(item.get("runner_type", "")).startswith("mock_")
        for item in per_mode_results
        if item.get("status", "ok") == "ok"
    )
    thresholds = regression_thresholds or RegressionThresholds(
        mock_relaxed=mock_relaxed
    )
    report_without_regression = {
        "dataset_path": dataset_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modes": list(modes),
        "sample_count": len(samples),
        "tag_distribution": _tag_distribution(samples),
        "runner_summary": runner_summary,
        "mode_summaries": mode_summaries,
        "comparisons": _build_comparisons(per_mode_results),
        "tag_breakdown": _tag_breakdown(samples, per_mode_results),
        "per_mode_results": list(per_mode_results),
        "risks": _collect_risks(per_mode_results),
    }
    checks = run_regression_checks(
        report_without_regression, thresholds=thresholds
    )
    report_without_regression["regression_checks"] = regression_summary(checks)
    return report_without_regression


def write_markdown_report(report: Dict[str, Any], path: str | Path) -> Path:
    """Write the human-readable final evaluation report."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _render_markdown(report)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_csv_summary(report: Dict[str, Any], path: str | Path) -> Path:
    """Write a compact CSV comparison table for presentation."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "runner_type",
        "status",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "average_latency_ms",
        "p95_latency_ms",
        "average_token_count",
        "citation_presence_rate",
        "answer_point_coverage",
        "failure_count",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("mode_summaries") or []:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def _mode_summary_row(mode_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": mode_result.get("mode"),
        "runner_type": mode_result.get("runner_type"),
        "status": mode_result.get("status", "ok"),
        "sample_count": mode_result.get("sample_count"),
        "hit_at_1": mode_result.get("hit_at_1"),
        "hit_at_3": mode_result.get("hit_at_3"),
        "hit_at_5": mode_result.get("hit_at_5"),
        "mrr": mode_result.get("mrr"),
        "average_latency_ms": mode_result.get("average_latency_ms"),
        "p50_latency_ms": mode_result.get("p50_latency_ms"),
        "p95_latency_ms": mode_result.get("p95_latency_ms"),
        "average_token_count": mode_result.get("average_token_count"),
        "citation_presence_rate": mode_result.get("citation_presence_rate"),
        "answer_point_coverage": mode_result.get("answer_point_coverage"),
        "failure_count": mode_result.get("failure_count"),
        "warning_count": mode_result.get("warning_count"),
    }


def _build_comparisons(
    per_mode_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_mode = {item.get("mode"): item for item in per_mode_results}
    naive = by_mode.get("naive") or {}
    advanced = by_mode.get("advanced") or {}
    modular = by_mode.get("modular") or {}
    graph = by_mode.get("graph") or {}
    agentic = by_mode.get("agentic") or {}
    return {
        "naive_vs_advanced": _delta_pair(naive, advanced),
        "modular_vs_naive": _delta_pair(naive, modular),
        "graph_vs_advanced": _delta_pair(advanced, graph),
        "agentic_vs_modular": _delta_pair(modular, agentic),
    }


def _delta_pair(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Dict[str, Any]:
    fields = ("hit_at_1", "mrr", "average_latency_ms", "average_token_count")
    delta: Dict[str, Any] = {}
    for field in fields:
        left_val = left.get(field)
        right_val = right.get(field)
        if isinstance(left_val, (int, float)) and isinstance(
            right_val, (int, float)
        ):
            delta[field] = round(float(right_val) - float(left_val), 6)
    return delta


def _tag_breakdown(
    samples: Sequence[EvaluationSample],
    per_mode_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    sample_tags: Dict[str, List[str]] = {
        sample.id: list(sample.tags) for sample in samples
    }
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for mode_result in per_mode_results:
        mode = str(mode_result.get("mode"))
        if mode_result.get("status") in {"failed", "unavailable"}:
            continue
        tag_hits: Dict[str, List[float]] = defaultdict(list)
        for result in mode_result.get("per_sample_results") or []:
            for tag in sample_tags.get(result.get("id"), []):
                if isinstance(result.get("hit_at_1"), (int, float)):
                    tag_hits[tag].append(float(result["hit_at_1"]))
        for tag, values in tag_hits.items():
            breakdown[tag][mode] = {
                "hit_at_1": round(sum(values) / len(values), 6),
                "sample_count": len(values),
            }
    return dict(sorted(breakdown.items()))


def _collect_risks(per_mode_results: Sequence[Dict[str, Any]]) -> List[str]:
    risks: List[str] = []
    if any(
        str(item.get("runner_type", "")).startswith("mock_")
        for item in per_mode_results
    ):
        risks.append(
            "One or more modes used mock runners; quality numbers are "
            "contract/smoke checks only, not production quality estimates."
        )
    if any(item.get("status") in {"failed", "unavailable"} for item in per_mode_results):
        risks.append(
            "Some modes failed or were unavailable; comparisons involving "
            "those modes should be treated as incomplete."
        )
    agentic = next(
        (item for item in per_mode_results if item.get("mode") == "agentic"),
        None,
    )
    if agentic and str(agentic.get("runner_type", "")).startswith("mock_"):
        risks.append(
            "Agentic mode used a placeholder mock runner; tool metrics do "
            "not validate a real agent workflow."
        )
    return risks


def _render_markdown(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = [
        "# RAG Final Evaluation Report",
        "",
        "## 数据集说明",
        f"- Dataset: `{report.get('dataset_path')}`",
        f"- Samples: {report.get('sample_count')}",
        f"- Tags: {report.get('tag_distribution')}",
        "",
        "## 运行环境说明",
        f"- Created at (UTC): {report.get('created_at')}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Runner summary: {json.dumps(report.get('runner_summary'), ensure_ascii=False)}",
        "",
        "> **Note:** Mock runners validate metrics and report schema offline. "
        "They do not represent live LLM retrieval/generation quality.",
        "",
        "## 五模式指标对比表",
        "",
        _markdown_table(
            ["Mode", "Runner", "Status", "Hit@1", "Hit@3", "MRR", "Latency(ms)", "Tokens", "Citations"],
            [
                [
                    row.get("mode"),
                    row.get("runner_type"),
                    row.get("status"),
                    row.get("hit_at_1"),
                    row.get("hit_at_3"),
                    row.get("mrr"),
                    row.get("average_latency_ms"),
                    row.get("average_token_count"),
                    row.get("citation_presence_rate"),
                ]
                for row in report.get("mode_summaries") or []
            ],
        ),
        "",
        "## 按标签对比表",
        "",
    ]
    tag_lines = ["| Tag | Mode | Hit@1 | Count |", "| --- | --- | ---: | ---: |"]
    for tag, modes in (report.get("tag_breakdown") or {}).items():
        for mode, stats in sorted(modes.items()):
            tag_lines.append(
                f"| {tag} | {mode} | {stats.get('hit_at_1')} | {stats.get('sample_count')} |"
            )
    lines.extend(tag_lines)
    lines.extend(
        [
            "",
            "## Naive vs Advanced 结论",
            _comparison_conclusion(
                report,
                "naive_vs_advanced",
                "Advanced relative to Naive",
            ),
            "",
            "## Modular 模块结论",
            _comparison_conclusion(
                report,
                "modular_vs_naive",
                "Modular relative to Naive",
            ),
            "",
            "## GraphRAG 适用问题结论",
            _graph_conclusion(report),
            "",
            "## Agentic 工具选择结论",
            _agentic_conclusion(report),
            "",
            "## 质量/耗时/Token 权衡",
            _tradeoff_conclusion(report),
            "",
            "## 失败样例摘要",
            _failure_summary(report),
            "",
            "## 剩余风险",
        ]
    )
    for risk in report.get("risks") or []:
        lines.append(f"- {risk}")
    regression = report.get("regression_checks") or {}
    lines.extend(
        [
            "",
            "## 回归检查",
            f"- Passed: {regression.get('passed')} "
            f"({regression.get('passed_count')}/{regression.get('total_count')})",
            "",
            "## 后续改进建议",
            "- Replace mock runners with configured RAGService runs when API keys and KB data are available.",
            "- Add LLM-as-judge only after deterministic offline metrics remain stable.",
            "- Extend agentic evaluation once the production agent workflow is wired to the unified runner.",
        ]
    )
    return lines


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _comparison_conclusion(
    report: Dict[str, Any], key: str, label: str
) -> str:
    delta = (report.get("comparisons") or {}).get(key) or {}
    if not delta:
        return f"{label}: insufficient data (mode missing or unavailable)."
    return (
        f"{label}: ΔHit@1={delta.get('hit_at_1')}, ΔMRR={delta.get('mrr')}, "
        f"ΔLatency={delta.get('average_latency_ms')} ms, "
        f"ΔTokens={delta.get('average_token_count')}."
    )


def _graph_conclusion(report: Dict[str, Any]) -> str:
    graph = next(
        (item for item in report.get("per_mode_results") or [] if item.get("mode") == "graph"),
        None,
    )
    if not graph or graph.get("status") != "ok":
        return "Graph mode unavailable; GraphRAG conclusions cannot be drawn from this run."
    relation = (report.get("tag_breakdown") or {}).get("relation", {}).get("graph")
    multi_hop = (report.get("tag_breakdown") or {}).get("multi_hop", {}).get("graph")
    path_rate = graph.get("graph_path_presence_rate")
    return (
        "Graph mode is intended for relation/multi-hop style questions. "
        f"relation Hit@1={relation.get('hit_at_1') if relation else 'n/a'}, "
        f"multi_hop Hit@1={multi_hop.get('hit_at_1') if multi_hop else 'n/a'}, "
        f"graph_path_presence_rate={path_rate}."
    )


def _agentic_conclusion(report: Dict[str, Any]) -> str:
    agentic = next(
        (
            item
            for item in report.get("per_mode_results") or []
            if item.get("mode") == "agentic"
        ),
        None,
    )
    if not agentic or agentic.get("status") != "ok":
        return "Agentic mode unavailable; tool-selection conclusions cannot be drawn."
    metrics = agentic.get("agent_metrics") or {}
    route = (report.get("tag_breakdown") or {}).get("agent_route", {}).get("agentic")
    return (
        f"Agentic mock run tool_selection_accuracy="
        f"{metrics.get('tool_selection_accuracy')}, "
        f"trajectory_success_rate={metrics.get('trajectory_success_rate')}, "
        f"agent_route Hit@1={route.get('hit_at_1') if route else 'n/a'}."
    )


def _tradeoff_conclusion(report: Dict[str, Any]) -> str:
    rows = report.get("mode_summaries") or []
    if not rows:
        return "No mode summaries available."
    slowest = max(rows, key=lambda row: float(row.get("average_latency_ms") or 0))
    cheapest = min(rows, key=lambda row: float(row.get("average_token_count") or 0))
    best_mrr = max(rows, key=lambda row: float(row.get("mrr") or 0))
    return (
        f"Best MRR in this run: {best_mrr.get('mode')} ({best_mrr.get('mrr')}). "
        f"Lowest token average: {cheapest.get('mode')} "
        f"({cheapest.get('average_token_count')}). "
        f"Highest latency: {slowest.get('mode')} "
        f"({slowest.get('average_latency_ms')} ms)."
    )


def _failure_summary(report: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for mode_result in report.get("per_mode_results") or []:
        failed = mode_result.get("failed_samples") or []
        if failed:
            chunks.append(
                f"- {mode_result.get('mode')}: {', '.join(failed[:5])}"
                + (" ..." if len(failed) > 5 else "")
            )
    return "\n".join(chunks) if chunks else "No failed samples in this run."


__all__ = [
    "build_evaluation_report",
    "build_final_evaluation_report",
    "write_csv_summary",
    "write_markdown_report",
    "write_report",
]
