"""
Evaluation Routes — Owner: A
REST 端点：启动评测与查看评测报告。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

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
from src.models.rag import RAGMode

router = APIRouter(prefix="/api/evaluations", tags=["Evaluation"])


# ============================================================================
# Mock 数据 — 五模式评测对比表
# ============================================================================

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


# ============================================================================
# 路由
# ============================================================================

@router.post("/run", status_code=202)
async def run_evaluation(
    kb_id: str = "kb_001",
    modes: list[str] = ["naive", "advanced", "modular", "graph", "agentic"],
    eval_runner: EvalRunnerDep = None,  # type: ignore
) -> dict:
    """启动五模式评测对比，返回评测运行 ID"""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    return {
        "run_id": run_id,
        "status": "running",
        "kb_id": kb_id,
        "modes": modes,
        "message": f"评测已启动，将在 {len(modes)} 种模式下运行",
    }


@router.get("/{run_id}")
async def get_evaluation_result(
    run_id: str,
    eval_runner: EvalRunnerDep = None,  # type: ignore
) -> list[dict]:
    """获取评测报告（五模式对比结果）"""
    return _MOCK_RESULTS
