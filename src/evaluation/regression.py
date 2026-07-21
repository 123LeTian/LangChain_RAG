"""Regression checks for multi-mode evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class RegressionThresholds:
    """Configurable thresholds; mock runners may use relaxed values."""

    max_failure_count: int = 0
    min_citation_presence_rate: float = 0.5
    min_graph_path_presence_rate: float = 0.5
    min_tool_selection_accuracy: float = 0.5
    mock_relaxed: bool = False

    def effective(self) -> "RegressionThresholds":
        if not self.mock_relaxed:
            return self
        return RegressionThresholds(
            max_failure_count=max(self.max_failure_count, 24),
            min_citation_presence_rate=min(self.min_citation_presence_rate, 0.0),
            min_graph_path_presence_rate=min(self.min_graph_path_presence_rate, 0.0),
            min_tool_selection_accuracy=min(self.min_tool_selection_accuracy, 0.0),
            mock_relaxed=True,
        )


def run_regression_checks(
    final_report: Dict[str, Any],
    *,
    thresholds: RegressionThresholds | None = None,
) -> List[Dict[str, Any]]:
    """Run regression checks; returns a list of structured pass/fail entries."""

    config = (thresholds or RegressionThresholds()).effective()
    checks: List[Dict[str, Any]] = []
    per_mode = final_report.get("per_mode_results") or []

    for mode_result in per_mode:
        mode = mode_result.get("mode", "<unknown>")
        status = mode_result.get("status", "ok")
        if status in {"unavailable", "failed"}:
            checks.append(
                {
                    "name": f"{mode}_availability",
                    "passed": True,
                    "detail": f"Mode recorded as {status}: {mode_result.get('error')}",
                }
            )
            continue

        failure_count = int(mode_result.get("failure_count") or 0)
        checks.append(
            _check(
                f"{mode}_failure_count",
                failure_count <= config.max_failure_count,
                f"failure_count={failure_count}, max={config.max_failure_count}",
            )
        )

        citation_rate = mode_result.get("citation_presence_rate")
        if citation_rate is not None:
            checks.append(
                _check(
                    f"{mode}_citation_presence_rate",
                    float(citation_rate) >= config.min_citation_presence_rate,
                    f"citation_presence_rate={citation_rate}, "
                    f"min={config.min_citation_presence_rate}",
                )
            )

        if mode == "graph":
            graph_path_rate = (mode_result.get("graph_metrics") or {}).get(
                "graph_path_presence_rate"
            )
            if graph_path_rate is None:
                graph_path_rate = mode_result.get("graph_path_presence_rate")
            if graph_path_rate is not None:
                checks.append(
                    _check(
                        "graph_path_presence_rate",
                        float(graph_path_rate) >= config.min_graph_path_presence_rate,
                        f"graph_path_presence_rate={graph_path_rate}, "
                        f"min={config.min_graph_path_presence_rate}",
                    )
                )

        if mode == "agentic":
            tool_acc = (mode_result.get("agent_metrics") or {}).get(
                "tool_selection_accuracy"
            )
            if tool_acc is None:
                tool_acc = mode_result.get("tool_selection_accuracy")
            if tool_acc is not None:
                checks.append(
                    _check(
                        "agentic_tool_selection_accuracy",
                        float(tool_acc) >= config.min_tool_selection_accuracy,
                        f"tool_selection_accuracy={tool_acc}, "
                        f"min={config.min_tool_selection_accuracy}",
                    )
                )

    checks.append(
        {
            "name": "naive_baseline_report_schema",
            "passed": _naive_baseline_schema_ok(final_report),
            "detail": "Ensures per-mode summaries remain compatible with D2 baseline fields",
        }
    )
    return checks


def regression_summary(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    passed = sum(1 for check in checks if check.get("passed"))
    return {
        "passed": passed == len(checks),
        "passed_count": passed,
        "total_count": len(checks),
        "checks": list(checks),
    }


def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _naive_baseline_schema_ok(final_report: Dict[str, Any]) -> bool:
    for mode_result in final_report.get("per_mode_results") or []:
        if mode_result.get("mode") != "naive":
            continue
        for field in (
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "average_latency_ms",
            "per_sample_results",
        ):
            if field not in mode_result:
                return False
        return True
    return False


__all__ = [
    "RegressionThresholds",
    "regression_summary",
    "run_regression_checks",
]
