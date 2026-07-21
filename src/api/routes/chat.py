"""
Chat Routes — Owner: A
REST 端点：RAG 查询（非流式）与 SSE 流式查询。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse  # type: ignore[import]

from src.api.dependencies import RAGServiceDep
from src.api.errors import RAGError, TimeoutError, ValidationError
from src.models.rag import (
    Citation,
    RAGMode,
    RAGRequest,
    RAGResult,
    RetrievalHit,
    TraceEvent,
    TraceStage,
)

router = APIRouter(prefix="/api/rag", tags=["Chat"])


# ============================================================================
# Mock 数据 — 后端 RAG 服务未就绪时前端可独立开发
# ============================================================================

_MOCK_CITATIONS = [
    Citation(
        document_id="doc_0001",
        chunk_id="chunk_0003",
        filename="产品手册.pdf",
        page=12,
        quote="该产品支持最高 3.2GHz 的频率运行...",
        score=0.94,
    ),
    Citation(
        document_id="doc_0001",
        chunk_id="chunk_0007",
        filename="产品手册.pdf",
        page=15,
        quote="能耗比达到行业平均水平的 1.5 倍...",
        score=0.87,
    ),
]

_MOCK_HITS = [
    RetrievalHit(chunk_id="chunk_0003", text="该产品支持最高 3.2GHz 的频率运行，在标准测试中…", score=0.94, rank=1, retriever="vector"),
    RetrievalHit(chunk_id="chunk_0007", text="能耗比达到行业平均水平的 1.5 倍，得益于…", score=0.87, rank=2, retriever="vector"),
    RetrievalHit(chunk_id="chunk_0012", text="散热设计使得芯片在满载时温度…", score=0.82, rank=3, retriever="vector"),
]

_MOCK_TRACE = [
    TraceEvent(trace_id="mock_trace_1", stage=TraceStage.REWRITE, duration_ms=120.5, input_summary="原始问题", output_summary="改写：该产品最高频率？"),
    TraceEvent(trace_id="mock_trace_1", stage=TraceStage.RETRIEVE, duration_ms=45.2, input_summary="改写后问题", output_summary="召回 10 个 chunks"),
    TraceEvent(trace_id="mock_trace_1", stage=TraceStage.RERANK, duration_ms=88.0, input_summary="Top-10 召回", output_summary="重排后 Top-3"),
    TraceEvent(trace_id="mock_trace_1", stage=TraceStage.GENERATE, duration_ms=1500.0, input_summary="Top-3 Chunks + Prompt", output_summary="流式生成 120 tokens"),
]


def _build_mock_result(query: str, mode: RAGMode) -> RAGResult:
    """构建用于前端开发的 Mock RAGResult"""
    trace_id = f"mock_{uuid.uuid4().hex[:8]}"
    return RAGResult(
        answer=f"根据知识库中的信息，针对您的问题「{query[:30]}…」，以下是基于检索到的相关文档片段生成的回答。\n\n"
               f"该产品支持最高 3.2GHz 的运行频率，能耗比达到行业平均水平的 1.5 倍。"
               f"在标准测试条件下，该芯片的散热系统可在满载时维持温度在 85°C 以下。",
        citations=_MOCK_CITATIONS,
        hits=_MOCK_HITS,
        trace=[e.model_copy(update={"trace_id": trace_id}) for e in _MOCK_TRACE],
        usage={"prompt_tokens": 320, "completion_tokens": 120, "total_tokens": 440},
        mode=mode,
    )


# ============================================================================
# 路由
# ============================================================================

@router.post("/query", response_model=RAGResult)
async def rag_query(
    request: RAGRequest,
    rag_service: RAGServiceDep,
) -> RAGResult:
    """非流式 RAG 查询 — 等待完整结果后返回，便于测试"""
    # ===== 参数校验 =====
    if not request.query.strip():
        raise ValidationError("查询文本不能为空")
    if request.mode == RAGMode.MODULAR and request.options.get("modules") is None:
        raise ValidationError("Modular 模式必须指定 modules 配置")

    # ===== Mock 模式：返回固定数据（真实服务就绪后切换） =====
    return _build_mock_result(request.query, request.mode)


@router.post("/query/stream")
async def rag_query_stream(
    request: RAGRequest,
    rag_service: RAGServiceDep,
):
    """SSE 流式 RAG 查询 — 同时发送追踪事件与生成 Token"""
    import asyncio

    # ===== 参数校验 =====
    if not request.query.strip():
        raise ValidationError("查询文本不能为空")

    # ===== Mock SSE 流：模拟真实 Trace + Chunk + Done 事件顺序 =====
    async def mock_event_generator():
        trace_id = f"mock_{uuid.uuid4().hex[:8]}"
        started = time.time()

        # 1) 发送 Trace 事件
        for stage in [TraceStage.REWRITE, TraceStage.RETRIEVE, TraceStage.RERANK]:
            await asyncio.sleep(0.2)  # 模拟阶段耗时
            elapsed = (time.time() - started) * 1000
            event_data = json.dumps(
                {
                    "trace_id": trace_id,
                    "stage": stage.value,
                    "elapsed_ms": round(elapsed, 1),
                    "output_summary": f"{stage.value} 阶段完成",
                },
                ensure_ascii=False,
            )
            yield {"event": "trace", "data": event_data}

        # 2) 流式发送生成 Token（chunk 事件）
        mock_tokens = ["根据", "知识库", "中的", "信息，", "该产品", "支持", "最高", "3.2GHz", "的", "频率。"]
        for token in mock_tokens:
            await asyncio.sleep(0.08)
            yield {"event": "chunk", "data": json.dumps({"delta": token}, ensure_ascii=False)}

        # 3) 发送完成事件（done）
        done_data = json.dumps(
            {
                "trace_id": trace_id,
                "citations": [c.model_dump() for c in _MOCK_CITATIONS],
                "usage": {"prompt_tokens": 320, "completion_tokens": 120, "total_tokens": 440},
            },
            ensure_ascii=False,
        )
        yield {"event": "done", "data": done_data}

    return EventSourceResponse(mock_event_generator())
