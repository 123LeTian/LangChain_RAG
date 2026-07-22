"""
Chat Routes - RAG query with real DeepSeek + 5-module pipeline.
SSE streaming with trace, detail, chunk events + comparison endpoint.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import RAGServiceDep
from src.api.errors import ValidationError
from src.api.api_models import RAGRequest, RAGResult, TraceStage

router = APIRouter(prefix="/api/rag", tags=["Chat"])


class CompareRequest(BaseModel):
    """Request for side-by-side config comparison."""
    query: str
    kb_id: str = "default"
    mode: str = "modular"
    config_a: dict = Field(default_factory=dict)
    config_b: dict = Field(default_factory=dict)


@router.post("/query", response_model=RAGResult)
async def rag_query(
    request: RAGRequest,
    rag_service: RAGServiceDep,
) -> RAGResult:
    """Non-streaming RAG query."""
    if not request.query.strip():
        raise ValidationError("query cannot be empty")
    try:
        return await rag_service.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def rag_query_stream(
    request: RAGRequest,
    rag_service: RAGServiceDep,
):
    """SSE streaming RAG query - trace, detail, then answer chunks."""
    if not request.query.strip():
        raise ValidationError("query cannot be empty")

    async def event_generator():
        trace_id = uuid.uuid4().hex[:12]

        yield {
            "event": "trace",
            "data": json.dumps({
                "trace_id": trace_id,
                "stage": "intent",
                "elapsed_ms": 0,
                "output_summary": "Processing query...",
            }, ensure_ascii=False),
        }

        try:
            result = await rag_service.query(request)

            for ev in result.trace:
                yield {
                    "event": "trace",
                    "data": json.dumps({
                        "trace_id": ev.trace_id,
                        "stage": ev.stage.value,
                        "elapsed_ms": ev.duration_ms,
                        "input_summary": ev.input_summary,
                        "output_summary": ev.output_summary,
                    }, ensure_ascii=False),
                }

            detail = result.usage.get("detail", {})
            if detail:
                yield {
                    "event": "detail",
                    "data": json.dumps(detail, ensure_ascii=False),
                }

            answer = result.answer
            chunk_size = 40
            for i in range(0, len(answer), chunk_size):
                yield {
                    "event": "chunk",
                    "data": json.dumps({"delta": answer[i:i+chunk_size]}, ensure_ascii=False),
                }

            yield {
                "event": "done",
                "data": json.dumps({
                    "trace_id": trace_id,
                    "citations": [c.model_dump() for c in result.citations],
                    "usage": {k: v for k, v in result.usage.items() if k != "detail"},
                }, ensure_ascii=False),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("/compare/stream")
async def rag_compare_stream(
    req: CompareRequest,
    rag_service: RAGServiceDep,
):
    """SSE streaming comparison - runs same query with two configs."""
    if not req.query.strip():
        raise ValidationError("query cannot be empty")

    async def event_generator():
        trace_id = uuid.uuid4().hex[:12]

        yield {
            "event": "start",
            "data": json.dumps({
                "trace_id": trace_id,
                "config_a": req.config_a,
                "config_b": req.config_b,
            }, ensure_ascii=False),
        }

        try:
            # Build two RAGRequest objects
            from src.api.api_models import RAGRequest, RAGMode
            mode_val = RAGMode.MODULAR
            for m in RAGMode:
                if m.value == req.mode:
                    mode_val = m
                    break

            request_a = RAGRequest(
                query=req.query, kb_id=req.kb_id, mode=mode_val, options=req.config_a
            )
            request_b = RAGRequest(
                query=req.query, kb_id=req.kb_id, mode=mode_val, options=req.config_b
            )

            result_a = await rag_service.query(request_a)

            yield {
                "event": "config_a",
                "data": json.dumps({
                    "answer": result_a.answer,
                    "citations": [c.model_dump() for c in result_a.citations],
                    "trace": [{
                        "stage": e.stage.value,
                        "duration_ms": e.duration_ms,
                        "output_summary": e.output_summary,
                    } for e in result_a.trace],
                    "detail": result_a.usage.get("detail", {}),
                    "usage": {k: v for k, v in result_a.usage.items() if k != "detail"},
                }, ensure_ascii=False),
            }

            result_b = await rag_service.query(request_b)

            yield {
                "event": "config_b",
                "data": json.dumps({
                    "answer": result_b.answer,
                    "citations": [c.model_dump() for c in result_b.citations],
                    "trace": [{
                        "stage": e.stage.value,
                        "duration_ms": e.duration_ms,
                        "output_summary": e.output_summary,
                    } for e in result_b.trace],
                    "detail": result_b.usage.get("detail", {}),
                    "usage": {k: v for k, v in result_b.usage.items() if k != "detail"},
                }, ensure_ascii=False),
            }

            yield {
                "event": "done",
                "data": json.dumps({"trace_id": trace_id}, ensure_ascii=False),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())