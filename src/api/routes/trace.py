"""
Trace Routes — Owner: A
REST 端点：执行轨迹查询。
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import TraceRepoDep
from src.api.errors import NotFoundError
from src.models.rag import TraceEvent, TraceStage

router = APIRouter(prefix="/api/traces", tags=["Trace"])


# ============================================================================
# Mock 数据（与 chat.py 中 _MOCK_TRACE 结构一致）
# ============================================================================

_MOCK_TRACE = [
    TraceEvent(
        trace_id="mock_trace_1",
        stage=TraceStage.REWRITE,
        duration_ms=120.5,
        input_summary="原始问题：该产品最高频率？",
        output_summary="改写后：该产品支持的最高运行频率是多少？",
    ),
    TraceEvent(
        trace_id="mock_trace_1",
        stage=TraceStage.RETRIEVE,
        duration_ms=45.2,
        input_summary="改写后问题",
        output_summary="召回 10 个 chunks，Top-1 相似度 0.94",
    ),
    TraceEvent(
        trace_id="mock_trace_1",
        stage=TraceStage.RERANK,
        duration_ms=88.0,
        input_summary="Top-10 召回结果",
        output_summary="重排后 Top-3：chunk_0003(0.94) chunk_0007(0.87) chunk_0012(0.82)",
    ),
    TraceEvent(
        trace_id="mock_trace_1",
        stage=TraceStage.GENERATE,
        duration_ms=1500.0,
        input_summary="Top-3 Chunks + System Prompt",
        output_summary="生成完成，共 120 tokens",
    ),
    TraceEvent(
        trace_id="mock_trace_1",
        stage=TraceStage.COMPLETE,
        duration_ms=0.5,
        input_summary="",
        output_summary="总耗时 1754.2ms，总 Token 440",
    ),
]


# ============================================================================
# 路由
# ============================================================================

@router.get("/{trace_id}")
async def get_trace(
    trace_id: str,
    trace_repo: TraceRepoDep = None,  # type: ignore
) -> list[dict]:
    """获取指定 trace_id 的完整执行轨迹"""
    # ===== Mock 模式 =====
    if trace_id.startswith("mock_"):
        return [e.model_dump() for e in _MOCK_TRACE]
    raise NotFoundError(f"Trace {trace_id} 不存在")
