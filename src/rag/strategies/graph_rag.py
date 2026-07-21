"""GraphRAG strategy using D4 local/global graph search."""

from __future__ import annotations

import inspect
import time
from datetime import datetime, timezone
from typing import Any, Iterable, List, Sequence
from uuid import uuid4

from src.graph.retriever import (
    GraphGlobalSearchHit,
    GraphLocalSearchResult,
    GraphRetriever,
    GraphSearchHit,
)
from src.models.graph import GraphSourceRef
from src.models.rag import (
    RAGChunk,
    RAGCitation,
    RAGContext,
    RAGMode,
    RAGRequest,
    RAGResult,
    RAGSource,
    TraceEvent,
    TraceStage,
)
from src.rag.base import RAGStrategy


GLOBAL_HINTS = {
    "整体",
    "总体",
    "主题",
    "趋势",
    "概括",
    "总结",
    "summarize",
    "summary",
    "overall",
    "global",
    "theme",
    "trend",
    "overview",
}

REFUSAL_ANSWER = "知识图谱中未找到足够依据，无法基于图谱回答该问题。"


class GraphRAGStrategy(RAGStrategy):
    """Run GraphRAG local or global search and return a canonical RAGResult."""

    mode: RAGMode = RAGMode.GRAPH

    def __init__(self, graph_retriever: GraphRetriever | None = None) -> None:
        self.graph_retriever = graph_retriever

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        trace_id = str(
            (context.metadata or {}).get("trace_id") or f"trace_{uuid4().hex[:16]}"
        )
        traces: List[TraceEvent] = []
        warnings: List[str] = []
        started = time.perf_counter()

        try:
            graph_retriever = self._resolve_graph_retriever(context)
            kb_id = self._resolve_kb_id(request, context)
            top_k = self._resolve_top_k(request)
            scope = self._resolve_scope(request)

            graph_started = time.perf_counter()
            if scope == "global":
                global_result = graph_retriever.global_search(
                    request.query, kb_id=kb_id, top_k=top_k
                )
                local_result = None
                warnings.extend(global_result.warnings)
                hits = _chunks_from_global(global_result.hits)
                source_refs = _source_refs_from_global(global_result.hits)
                graph_output = (
                    f"scope=global, reports={len(global_result.hits)}, "
                    f"citations={len(source_refs)}"
                )
            else:
                local_result = graph_retriever.local_search(
                    request.query, kb_id=kb_id, top_k=top_k
                )
                global_result = None
                warnings.extend(local_result.warnings)
                hits = _chunks_from_local(local_result.hits)
                source_refs = _source_refs_from_local(local_result.hits)
                relationship_count = sum(
                    len(hit.relationships) for hit in local_result.hits
                )
                graph_output = (
                    f"scope=local, entities={len(local_result.hits)}, "
                    f"relationships={relationship_count}, citations={len(source_refs)}"
                )

            traces.append(
                self._trace_event(
                    context,
                    trace_id,
                    TraceStage.GRAPH_SEARCH,
                    started_perf=graph_started,
                    input_summary=(
                        f"query_chars={len(request.query)}, scope={scope}, "
                        f"top_k={top_k}, kb_id={kb_id}"
                    ),
                    output_summary=graph_output,
                )
            )

            citations = _citations_from_source_refs(source_refs)
            generation_started = time.perf_counter()
            if not citations:
                answer = REFUSAL_ANSWER
            elif scope == "global" and global_result is not None:
                answer = _global_template_answer(request.query, global_result.hits)
            elif local_result is not None:
                answer = _local_template_answer(request.query, local_result.hits)
            else:
                answer = REFUSAL_ANSWER

            answer, usage, generation_warnings = await self._maybe_generate_with_llm(
                context,
                request,
                answer,
                hits,
            )
            warnings.extend(generation_warnings)

            traces.append(
                self._trace_event(
                    context,
                    trace_id,
                    TraceStage.GENERATE,
                    started_perf=generation_started,
                    input_summary=(
                        f"context_chunks={len(hits)}, citation_count={len(citations)}"
                    ),
                    output_summary=f"answer_chars={len(answer)}, usage_present={bool(usage)}",
                )
            )

            complete_started = time.perf_counter()
            total_ms = (time.perf_counter() - started) * 1000.0
            traces.append(
                self._trace_event(
                    context,
                    trace_id,
                    TraceStage.COMPLETE,
                    started_perf=complete_started,
                    input_summary=f"scope={scope}",
                    output_summary=(
                        f"hits={len(hits)}, citations={len(citations)}, "
                        f"warnings={len(warnings)}, total_ms={total_ms:.3f}"
                    ),
                )
            )
            return RAGResult(
                answer=answer,
                citations=citations,
                hits=hits,
                trace=traces,
                usage=usage,
                warnings=warnings,
            )
        except Exception as exc:
            sanitized = str(exc).replace("\n", " ").strip()[:200]
            traces.append(
                self._trace_event(
                    context,
                    trace_id,
                    TraceStage.ERROR,
                    started_perf=time.perf_counter(),
                    input_summary="stage=graph_rag",
                    output_summary=f"error={type(exc).__name__}",
                )
            )
            return RAGResult(
                answer=REFUSAL_ANSWER,
                citations=[],
                hits=[],
                trace=traces,
                usage={},
                warnings=[f"GraphRAG failed: {type(exc).__name__}: {sanitized}"],
            )

    def _resolve_graph_retriever(self, context: RAGContext) -> GraphRetriever:
        configured = (context.config or {}).get("graph_retriever")
        if isinstance(configured, GraphRetriever):
            return configured
        if self.graph_retriever is not None:
            return self.graph_retriever
        raise ValueError("GraphRAGStrategy requires a GraphRetriever")

    @staticmethod
    def _resolve_kb_id(request: RAGRequest, context: RAGContext) -> str:
        kb_id = request.kb_id or (context.metadata or {}).get("kb_id")
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise ValueError("GraphRAGStrategy requires request.kb_id")
        return kb_id.strip()

    @staticmethod
    def _resolve_top_k(request: RAGRequest) -> int:
        value = (request.options or {}).get("top_k", 5)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("top_k must be a positive integer")
        return min(value, 100)

    @staticmethod
    def _resolve_scope(request: RAGRequest) -> str:
        raw_scope = str((request.options or {}).get("graph_scope", "auto")).lower()
        if raw_scope not in {"local", "global", "auto"}:
            raise ValueError("graph_scope must be one of: local, global, auto")
        if raw_scope != "auto":
            return raw_scope
        query_lower = request.query.lower()
        if any(hint in query_lower for hint in GLOBAL_HINTS):
            return "global"
        return "local"

    async def _maybe_generate_with_llm(
        self,
        context: RAGContext,
        request: RAGRequest,
        fallback_answer: str,
        hits: Sequence[RAGChunk],
    ) -> tuple[str, dict[str, int], list[str]]:
        llm = context.llm
        if llm is None or not hits:
            return fallback_answer, {}, []
        prompt = (
            "Answer using only the provided graph evidence. "
            "Keep source-grounded claims concise."
        )
        graph_context = "\n".join(chunk.content for chunk in hits)
        try:
            if callable(getattr(llm, "generate_with_tokens", None)):
                output = llm.generate_with_tokens(prompt=prompt, context=graph_context)
                if inspect.isawaitable(output):
                    output = await output
                answer, usage = output
                if isinstance(answer, str) and answer.strip():
                    return answer.strip(), dict(usage or {}), []
            if callable(getattr(llm, "generate", None)):
                output = llm.generate(prompt=prompt, context=graph_context)
                if inspect.isawaitable(output):
                    output = await output
                if isinstance(output, str) and output.strip():
                    return output.strip(), {}, []
        except Exception as exc:
            return (
                fallback_answer,
                {},
                [f"graph generation fallback: {type(exc).__name__}"],
            )
        return fallback_answer, {}, []

    def _trace_event(
        self,
        context: RAGContext,
        trace_id: str,
        stage: TraceStage,
        *,
        started_perf: float,
        input_summary: str,
        output_summary: str,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=trace_id,
            stage=stage,
            started_at=datetime.now(timezone.utc),
            duration_ms=round(max(0.0, (time.perf_counter() - started_perf) * 1000.0), 3),
            input_summary=input_summary,
            output_summary=output_summary,
        )
        recorder = context.trace_recorder
        try:
            if callable(getattr(recorder, "record", None)):
                recorder.record(trace_id, event)
            elif callable(recorder):
                recorder(stage, input_summary, output_summary, event.duration_ms)
        except Exception:
            pass
        return event


def _chunks_from_local(hits: Sequence[GraphSearchHit]) -> List[RAGChunk]:
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        ref = hit.source_refs[0] if hit.source_refs else None
        chunks.append(
            RAGChunk(
                chunk_id=ref.chunk_id if ref else f"graph_local_{hit.entity_id}",
                content=_local_hit_context(hit),
                source=RAGSource(
                    document_id=ref.document_id if ref else hit.entity_id,
                    chunk_index=0,
                    source_path=ref.filename if ref else None,
                    title=(ref.section or ref.filename if ref else hit.entity_name),
                    page=ref.page if ref else None,
                    score=hit.score,
                    metadata={
                        "graph_scope": "local",
                        "entity_id": hit.entity_id,
                        "entity_name": hit.entity_name,
                        "relationship_id": hit.relationships[0].id
                        if hit.relationships
                        else None,
                        "path": list(hit.path),
                        "rank": rank,
                        "score_reason": hit.metadata.get("score_reason"),
                    },
                ),
            )
        )
    return chunks


def _chunks_from_global(hits: Sequence[GraphGlobalSearchHit]) -> List[RAGChunk]:
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        ref = hit.source_refs[0] if hit.source_refs else None
        chunks.append(
            RAGChunk(
                chunk_id=ref.chunk_id if ref else f"graph_global_{hit.report_id}",
                content=hit.summary,
                source=RAGSource(
                    document_id=ref.document_id if ref else hit.community_id,
                    chunk_index=0,
                    source_path=ref.filename if ref else None,
                    title=hit.title,
                    page=ref.page if ref else None,
                    score=hit.score,
                    metadata={
                        "graph_scope": "global",
                        "community_id": hit.community_id,
                        "report_id": hit.report_id,
                        "rank": rank,
                        "score_reason": hit.metadata.get("score_reason"),
                    },
                ),
            )
        )
    return chunks


def _citations_from_source_refs(source_refs: Sequence[GraphSourceRef]) -> List[RAGCitation]:
    citations = []
    seen = set()
    for ref in source_refs:
        key = (ref.document_id, ref.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            RAGCitation(
                document_id=ref.document_id,
                chunk_id=ref.chunk_id,
                text_snippet=ref.quote,
            )
        )
    return citations


def _source_refs_from_local(hits: Sequence[GraphSearchHit]) -> List[GraphSourceRef]:
    return _dedupe_source_refs([ref for hit in hits for ref in hit.source_refs])


def _source_refs_from_global(
    hits: Sequence[GraphGlobalSearchHit],
) -> List[GraphSourceRef]:
    return _dedupe_source_refs([ref for hit in hits for ref in hit.source_refs])


def _dedupe_source_refs(refs: Iterable[GraphSourceRef]) -> List[GraphSourceRef]:
    seen = set()
    unique = []
    for ref in refs:
        key = (ref.document_id, ref.chunk_id, ref.quote)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


def _local_template_answer(query: str, hits: Sequence[GraphSearchHit]) -> str:
    lines = [f"Graph local search found {len(hits)} relevant entity result(s)."]
    for hit in hits:
        relationships = "; ".join(
            relationship.description or relationship.relation_type
            for relationship in hit.relationships[:3]
        )
        neighbors = ", ".join(entity.name for entity in hit.neighbor_entities[:3])
        lines.append(
            f"- {hit.entity_name}: relationships={relationships or 'none'}; "
            f"neighbors={neighbors or 'none'}."
        )
    return "\n".join(lines)


def _global_template_answer(query: str, hits: Sequence[GraphGlobalSearchHit]) -> str:
    lines = [f"Graph global search found {len(hits)} community report(s)."]
    for hit in hits:
        entities = ", ".join(hit.key_entities[:5])
        relationships = "; ".join(hit.key_relationships[:3])
        lines.append(
            f"- {hit.title}: {hit.summary} "
            f"Key entities={entities or 'none'}; "
            f"key relationships={relationships or 'none'}."
        )
    return "\n".join(lines)


def _local_hit_context(hit: GraphSearchHit) -> str:
    relationships = "; ".join(
        relationship.description or relationship.relation_type
        for relationship in hit.relationships
    )
    neighbors = ", ".join(entity.name for entity in hit.neighbor_entities)
    return (
        f"Entity: {hit.entity_name}. "
        f"Relationships: {relationships or 'none'}. "
        f"Neighbors: {neighbors or 'none'}. "
        f"Path: {' -> '.join(hit.path)}."
    )


__all__ = ["GraphRAGStrategy", "REFUSAL_ANSWER"]
