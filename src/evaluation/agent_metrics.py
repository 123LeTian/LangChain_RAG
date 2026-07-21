"""Agent-mode metrics for agentic RAG evaluation runs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


_TOOL_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*(?:_search|_tool))\b")


def infer_expected_tools(answer_points: Sequence[str]) -> List[str]:
    """Heuristic expected tools from evaluation answer points (offline, reproducible)."""

    tools: List[str] = []
    text = " ".join(answer_points).lower()
    if "vector_search" in text or "vector search" in text:
        tools.append("vector_search")
    if "graph tool" in text or "graph_search" in text or "graph search" in text:
        tools.append("graph_search")
    if "agent route" in text or "agentic" in text:
        tools.append("route_agent")
    for match in _TOOL_PATTERN.findall(text):
        if match not in tools:
            tools.append(match)
    return tools


def tool_selection_accuracy(
    per_sample_results: Sequence[Dict[str, Any]],
    *,
    samples_by_id: Dict[str, Any] | None = None,
) -> float | None:
    """Fraction of samples where selected tools match inferred expectations."""

    scored = 0
    correct = 0
    for result in per_sample_results:
        agent = result.get("agent") or {}
        selected = agent.get("tools_selected") or result.get("tools_selected")
        expected = agent.get("expected_tools")
        if expected is None and samples_by_id is not None:
            sample = samples_by_id.get(result.get("id"))
            if sample is not None:
                expected = infer_expected_tools(sample.answer_points)
        if not expected:
            continue
        if not isinstance(selected, list):
            selected = []
        scored += 1
        if set(selected) & set(expected):
            correct += 1
    if scored == 0:
        return None
    return round(correct / scored, 6)


def trajectory_success_rate(per_sample_results: Sequence[Dict[str, Any]]) -> float | None:
    agent_samples = [
        result
        for result in per_sample_results
        if (result.get("agent") or {}).get("trajectory_success") is not None
    ]
    if not agent_samples:
        return None
    successes = sum(
        1
        for result in agent_samples
        if (result.get("agent") or {}).get("trajectory_success")
    )
    return round(successes / len(agent_samples), 6)


def average_tool_steps(per_sample_results: Sequence[Dict[str, Any]]) -> float | None:
    steps = [
        (result.get("agent") or {}).get("tool_steps")
        for result in per_sample_results
    ]
    numeric = [value for value in steps if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def summarize_agent_metrics(
    per_sample_results: Sequence[Dict[str, Any]],
    *,
    samples_by_id: Dict[str, Any] | None = None,
) -> Dict[str, float | None]:
    return {
        "tool_selection_accuracy": tool_selection_accuracy(
            per_sample_results, samples_by_id=samples_by_id
        ),
        "trajectory_success_rate": trajectory_success_rate(per_sample_results),
        "average_tool_steps": average_tool_steps(per_sample_results),
    }


__all__ = [
    "average_tool_steps",
    "infer_expected_tools",
    "summarize_agent_metrics",
    "tool_selection_accuracy",
    "trajectory_success_rate",
]
