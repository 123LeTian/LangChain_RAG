"""
Chat Routes — RAG query with real DeepSeek + vector retrieval.
SSE streaming sends immediate feedback so the frontend doesn't appear stuck.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import RAGServiceDep
from src.api.errors import ValidationError
from src.api.api_models import RAGRequest, RAGResult, TraceStage

router = APIRouter(prefix="/api/rag", tags=["Chat"])


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
    """SSE streaming RAG query — sends immediate feedback, then results."""
    if not request.query.strip():
        raise ValidationError("query cannot be empty")

    async def event_generator():
        trace_id = uuid.uuid4().hex[:12]

        # 1) Immediate feedback — frontend knows we're working
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

            # 2) Send trace events
            for ev in result.trace:
                yield {
                    "event": "trace",
                    "data": json.dumps({
                        "trace_id": ev.trace_id,
                        "stage": ev.stage.value,
                        "elapsed_ms": ev.duration_ms,
                        "output_summary": ev.output_summary,
                    }, ensure_ascii=False),
                }

            # 3) Send answer in chunks
            answer = result.answer
            chunk_size = 40
            for i in range(0, len(answer), chunk_size):
                yield {
                    "event": "chunk",
                    "data": json.dumps({"delta": answer[i:i+chunk_size]}, ensure_ascii=False),
                }

            # 4) Done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "trace_id": trace_id,
                    "citations": [c.model_dump() for c in result.citations],
                    "usage": result.usage,
                }, ensure_ascii=False),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())