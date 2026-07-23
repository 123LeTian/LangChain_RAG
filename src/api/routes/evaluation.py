"""
Evaluation Routes — Owner: A
REST 端点：启动评测与查看评测报告。
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.dependencies import EvalRunnerDep
from src.api.errors import NotFoundError
from src.models.evaluation import (
    AgentMetrics,
    AnswerMetrics,
    CitationMetrics,
    EvaluationResult,
    GraphMetrics,
    MetricsSnapshot,
    RetrievalMetrics,
    SystemMetrics,
)
from src.api.api_models import RAGMode

router = APIRouter(prefix="/api/evaluations", tags=["Evaluation"])


# ============================================================================
# Mock 数据 — 五模式评测对比表
# ============================================================================


class EvaluationRunRequest(BaseModel):
    kb_id: str = "kb_001"
    modes: list[str] = Field(default_factory=lambda: ["naive", "advanced", "modular", "graph", "agentic"])
    model_id: Optional[str] = None
    sample_limit: Optional[int] = Field(default=5, ge=1)


_EVALUATION_RUNS: dict[str, dict[str, object]] = {}

def _mock_result(mode: RAGMode, hit3: float, mrr: float, lat: float) -> dict:
    """构造单个模式的 Mock 评测结果"""
    return EvaluationResult(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        mode=mode,
        sample_count=25,
        latency_ms=lat,
        token_usage={"prompt": 8000, "completion": 3000, "total": 11000},
        metrics=MetricsSnapshot(
            retrieval=RetrievalMetrics(hit_at_1=hit3 * 0.6, hit_at_3=hit3, hit_at_5=hit3 + 0.05, hit_at_10=hit3 + 0.1, mrr=mrr),
            answer=AnswerMetrics(coverage=0.72, faithfulness=0.85, relevance=0.80),
            citation=CitationMetrics(precision=0.78, recall=0.65),
            system=SystemMetrics(first_token_latency_ms=lat * 0.7, total_latency_ms=lat, total_tokens=11000, estimated_cost_usd=0.03),
            graph=GraphMetrics(relation_hit_rate=0.0, local_search_success=0.0, global_search_success=0.0) if mode == RAGMode.GRAPH else None,
            agent=AgentMetrics(tool_selection_accuracy=0.0, trajectory_success_rate=0.0, avg_steps=0.0) if mode == RAGMode.AGENTIC else None,
        ),
        details=[],
    ).model_dump()


_MOCK_RESULTS = [
    _mock_result(RAGMode.NAIVE, hit3=0.72, mrr=0.58, lat=1200.0),
    _mock_result(RAGMode.ADVANCED, hit3=0.80, mrr=0.66, lat=2100.0),
    _mock_result(RAGMode.MODULAR, hit3=0.78, mrr=0.63, lat=1950.0),
    _mock_result(RAGMode.GRAPH, hit3=0.65, mrr=0.52, lat=3200.0),
    _mock_result(RAGMode.AGENTIC, hit3=0.76, mrr=0.60, lat=4500.0),
]


def _mock_results_for_modes(
    run_id: str,
    modes: list[str],
    model_id: str | None = None,
    sample_limit: int | None = None,
) -> list[dict]:
    mock_by_mode = {
        str(getattr(result["mode"], "value", result["mode"])): result
        for result in _MOCK_RESULTS
    }
    requested_modes = modes or list(mock_by_mode)
    results: list[dict] = []
    for mode in requested_modes:
        if mode not in mock_by_mode:
            continue
        item = dict(mock_by_mode[mode])
        item["run_id"] = run_id
        item["model_id"] = model_id
        if sample_limit:
            item["sample_count"] = sample_limit
        results.append(item)
    return results


def _evaluation_result_from_mode_report(
    *,
    run_id: str,
    model_id: str | None,
    mode_result: dict[str, Any],
) -> dict:
    mode = str(mode_result.get("mode") or "")
    sample_count = int(mode_result.get("sample_count") or 0)
    latency_ms = float(mode_result.get("average_latency_ms") or 0.0)
    token_total = _total_tokens(mode_result, sample_count)

    result = EvaluationResult(
        run_id=run_id,
        mode=RAGMode(mode),
        sample_count=sample_count,
        latency_ms=latency_ms,
        token_usage={"prompt": 0, "completion": 0, "total": token_total},
        metrics=MetricsSnapshot(
            retrieval=RetrievalMetrics(
                hit_at_1=float(mode_result.get("hit_at_1") or 0.0),
                hit_at_3=float(mode_result.get("hit_at_3") or 0.0),
                hit_at_5=float(mode_result.get("hit_at_5") or 0.0),
                hit_at_10=float(mode_result.get("hit_at_10") or mode_result.get("hit_at_5") or 0.0),
                mrr=float(mode_result.get("mrr") or 0.0),
            ),
            answer=AnswerMetrics(
                coverage=float(mode_result.get("answer_point_coverage") or 0.0),
                faithfulness=0.0,
                relevance=0.0,
            ),
            citation=CitationMetrics(
                precision=float(mode_result.get("citation_presence_rate") or 0.0),
                recall=float(mode_result.get("citation_presence_rate") or 0.0),
            ),
            system=SystemMetrics(
                first_token_latency_ms=float(mode_result.get("p50_latency_ms") or latency_ms),
                total_latency_ms=latency_ms,
                total_tokens=token_total,
                estimated_cost_usd=0.0,
            ),
            graph=_graph_metrics(mode_result) if mode == RAGMode.GRAPH.value else None,
            agent=_agent_metrics(mode_result) if mode == RAGMode.AGENTIC.value else None,
        ),
        details=list(mode_result.get("per_sample_results") or []),
    ).model_dump()
    result["model_id"] = model_id
    return result


def _evaluation_results_from_report(report: dict[str, Any]) -> list[dict]:
    run_id = str(report.get("run_id") or "")
    model_id = report.get("model_id")
    results: list[dict] = []
    for mode_result in report.get("per_mode_results") or []:
        try:
            results.append(
                _evaluation_result_from_mode_report(
                    run_id=run_id,
                    model_id=str(model_id) if model_id else None,
                    mode_result=dict(mode_result),
                )
            )
        except ValueError:
            continue
    return results


def _total_tokens(mode_result: dict[str, Any], sample_count: int) -> int:
    total = 0
    for sample in mode_result.get("per_sample_results") or []:
        usage = sample.get("token_usage") if isinstance(sample, dict) else None
        if not isinstance(usage, dict):
            continue
        sample_total = usage.get("total") or usage.get("total_tokens")
        if isinstance(sample_total, (int, float)):
            total += int(sample_total)
            continue
        prompt = usage.get("prompt") or usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion") or usage.get("completion_tokens") or usage.get("output_tokens") or 0
        if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
            total += int(prompt or 0) + int(completion or 0)
    if total:
        return total
    return int(round(float(mode_result.get("average_token_count") or 0.0) * sample_count))


def _graph_metrics(mode_result: dict[str, Any]) -> GraphMetrics:
    graph_metrics = mode_result.get("graph_metrics") or {}
    return GraphMetrics(
        relation_hit_rate=float(graph_metrics.get("relation_hit_rate") or mode_result.get("relation_hit_rate") or 0.0),
        local_search_success=float(graph_metrics.get("local_search_success") or mode_result.get("local_search_success") or 0.0),
        global_search_success=float(graph_metrics.get("global_search_success") or mode_result.get("global_search_success") or 0.0),
    )


def _agent_metrics(mode_result: dict[str, Any]) -> AgentMetrics:
    agent_metrics = mode_result.get("agent_metrics") or {}
    return AgentMetrics(
        tool_selection_accuracy=float(agent_metrics.get("tool_selection_accuracy") or mode_result.get("tool_selection_accuracy") or 0.0),
        trajectory_success_rate=float(agent_metrics.get("trajectory_success_rate") or mode_result.get("trajectory_success_rate") or 0.0),
        avg_steps=float(agent_metrics.get("avg_steps") or mode_result.get("avg_steps") or 0.0),
    )


# ============================================================================
# 路由
# ============================================================================

@router.post("/run", status_code=202)
async def run_evaluation(
    request: EvaluationRunRequest,
    eval_runner: EvalRunnerDep = None,  # type: ignore
) -> dict:
    """启动五模式评测对比，返回评测运行 ID"""
    if eval_runner is not None:
        return await eval_runner.run_evaluation(request)

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    _EVALUATION_RUNS[run_id] = {
        "kb_id": request.kb_id,
        "modes": list(request.modes),
        "model_id": request.model_id,
        "sample_limit": request.sample_limit,
    }
    return {
        "run_id": run_id,
        "status": "running",
        "kb_id": request.kb_id,
        "modes": request.modes,
        "model_id": request.model_id,
        "sample_limit": request.sample_limit,
        "message": f"评测已启动，将在 {len(request.modes)} 种模式下运行",
    }


@router.get("/{run_id}")
async def get_evaluation_result(
    run_id: str,
    eval_runner: EvalRunnerDep = None,  # type: ignore
) -> list[dict]:
    """获取评测报告（五模式对比结果）"""
    if eval_runner is not None:
        try:
            report = await eval_runner.get_result(run_id)
        except KeyError as exc:
            raise NotFoundError(f"Evaluation run {run_id} not found") from exc
        return _evaluation_results_from_report(report)

    config = _EVALUATION_RUNS.get(run_id)
    if config is None:
        raise NotFoundError(f"Evaluation run {run_id} not found")
    requested_modes = [str(mode) for mode in config.get("modes", []) if str(mode)]
    model_id = config.get("model_id")
    sample_limit = config.get("sample_limit")
    return _mock_results_for_modes(
        run_id,
        requested_modes,
        str(model_id) if model_id else None,
        int(sample_limit) if sample_limit else None,
    )
