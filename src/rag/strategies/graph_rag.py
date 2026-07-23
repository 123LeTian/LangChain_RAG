"""GraphRAG strategy using D4 local/global graph search."""

from __future__ import annotations

import inspect
import re
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
                selected_global_hits = _select_global_hits_for_query(
                    request.query,
                    global_result.hits,
                )
                hits = _chunks_from_global(selected_global_hits)
                source_refs = _source_refs_from_global(
                    request.query,
                    selected_global_hits,
                )
                graph_output = (
                    f"scope=global, reports={len(selected_global_hits)}, "
                    f"raw_reports={len(global_result.hits)}, "
                    f"citations={len(source_refs)}"
                )
            else:
                local_result = graph_retriever.local_search(
                    request.query, kb_id=kb_id, top_k=top_k
                )
                global_result = None
                warnings.extend(local_result.warnings)
                selected_local_hits = _select_local_hits_for_query(
                    request.query,
                    local_result.hits,
                )
                selected_local_hits = _focus_local_hits_for_query(
                    request.query,
                    selected_local_hits,
                )
                hits = _chunks_from_local(selected_local_hits)
                source_refs = _source_refs_from_local(selected_local_hits)
                relationship_count = sum(
                    len(hit.relationships) for hit in selected_local_hits
                )
                graph_output = (
                    f"scope=local, entities={len(selected_local_hits)}, "
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
                answer = _global_template_answer(request.query, selected_global_hits)
            elif local_result is not None:
                answer = _local_template_answer(request.query, selected_local_hits)
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
        chinese_global_hints = {
            "\u6574\u4f53",
            "\u603b\u4f53",
            "\u4e3b\u9898",
            "\u8d8b\u52bf",
            "\u6982\u62ec",
            "\u603b\u7ed3",
            "\u91cd\u8981\u4e8b\u9879",
            "\u7ecf\u8425\u91cd\u70b9",
            "\u7ecf\u8425\u4f18\u52bf",
            "\u95ee\u9898\u4e3b\u7ebf",
            "\u4e3b\u7ebf",
            "\u677f\u5757",
            "\u4e2d\u5fc3\u677f\u5757",
            "\u903b\u8f91",
            "\u5f52\u7eb3",
            "\u652f\u6491",
            "\u957f\u671f\u53d1\u5c55",
            "\u8bf4\u660e",
            "\u4f53\u73b0",
            "\u610f\u5473",
            "\u6210\u679c",
            "\u5f71\u54cd",
            "\u8868\u660e",
        }
        if any(hint in query_lower for hint in chinese_global_hints):
            return "global"
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
            "You are answering a GraphRAG question. "
            "Use only the provided graph evidence.\n"
            "Answer requirements:\n"
            "1. Start with the direct answer, not with retrieval details.\n"
            "2. Then explain the relationship between the entities in plain language.\n"
            "For global, overview, important-matter, or theme questions, synthesize "
            "the provided reports into evidence-supported themes even if the exact "
            "question wording is not present in the evidence. Do not refuse solely "
            "because a label such as 'important matters' is absent; treat it as a "
            "request for the main themes found in the provided reports.\n"
            "3. For is_a evidence, explain it as a definition or identity: "
            "'X is the concrete meaning/content of Y' or 'Y is reflected as X'; "
            "do not say 'the two are the same relationship'.\n"
            "4. For contains evidence, explain it as an inclusion/composition relationship.\n"
            "5. For relates_to or depends_on evidence, explain it as support, influence, "
            "or dependency only when the evidence supports that wording.\n"
            "6. If the evidence is insufficient for part of the question, say so briefly "
            "and do not invent facts.\n"
            "7. If the question is Chinese, answer in Chinese.\n\n"
            f"Question:\n{request.query}"
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
            if callable(getattr(llm, "ainvoke", None)):
                output = llm.ainvoke(_graph_generation_prompt(prompt, graph_context))
                if inspect.isawaitable(output):
                    output = await output
                answer, usage = _unpack_llm_response(output)
                if answer:
                    return answer, usage, []
            if callable(getattr(llm, "invoke", None)):
                output = llm.invoke(_graph_generation_prompt(prompt, graph_context))
                if inspect.isawaitable(output):
                    output = await output
                answer, usage = _unpack_llm_response(output)
                if answer:
                    return answer, usage, []
            if callable(getattr(llm, "generate", None)):
                output = llm.generate(prompt=prompt, context=graph_context)
                if inspect.isawaitable(output):
                    output = await output
                answer, usage = _unpack_llm_response(output)
                if answer:
                    return answer, usage, []
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


def _select_global_hits_for_query(
    query: str,
    hits: Sequence[GraphGlobalSearchHit],
) -> List[GraphGlobalSearchHit]:
    if not hits:
        return []
    if _query_allows_shareholder_topics(query):
        return list(hits)

    scored_hits = [
        (hit, _global_hit_context_score(query, hit))
        for hit in hits
    ]
    selected = [
        hit
        for hit, score in sorted(
            scored_hits,
            key=lambda item: (item[1], item[0].score),
            reverse=True,
        )
        if score > 0
    ]
    return selected or list(hits)


def _global_hit_context_score(query: str, hit: GraphGlobalSearchHit) -> float:
    labels = [
        hit.title,
        hit.summary,
        *hit.key_entities,
        *hit.key_relationships,
    ]
    text = " ".join(str(label or "") for label in labels)
    score = 0.0
    score += sum(3.0 for marker in _global_context_strong_markers() if marker in text)
    score -= sum(4.0 for marker in _global_context_weak_markers() if marker in text)

    for topic in _global_topic_labels(hit):
        if not _is_weak_graph_label(topic):
            score += 1.0
    for relationship in hit.key_relationships:
        if _relationship_phrase(relationship):
            score += 1.0
    for term in _query_focus_terms(query):
        if term and term in text:
            score += 0.8

    if _has_noise_marker(text):
        score -= 6.0
    if "\u5e74\u5ea6\u62a5\u544a\u6458\u8981" in text and score <= 0:
        score -= 1.0
    return score


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
    query: str,
    hits: Sequence[GraphGlobalSearchHit],
) -> List[GraphSourceRef]:
    refs = _dedupe_source_refs([ref for hit in hits for ref in hit.source_refs])
    if not refs:
        return []

    scored_refs = [
        (ref, _global_source_ref_score(query, ref))
        for ref in refs
    ]
    quality_refs = [
        ref
        for ref, score in sorted(
            scored_refs,
            key=lambda item: item[1],
            reverse=True,
        )
        if score > 0
    ]
    return quality_refs or refs


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


def _global_source_ref_score(query: str, ref: GraphSourceRef) -> float:
    text = " ".join(
        [
            str(ref.section or ""),
            str(ref.filename or ""),
            str(ref.quote or ""),
        ]
    )
    score = 0.0
    strong_markers = _global_context_strong_markers()
    weak_markers = _global_context_weak_markers()
    score += sum(3.0 for marker in strong_markers if marker in text)
    score -= sum(4.0 for marker in weak_markers if marker in text)

    for term in _query_focus_terms(query):
        if term and term in text:
            score += 1.0

    if "\u5e74\u5ea6\u62a5\u544a\u6458\u8981" in text and score <= 0:
        score -= 1.0
    if len(str(ref.quote or "").strip()) < 30:
        score -= 1.0
    return score


def _global_context_strong_markers() -> set[str]:
    return {
        "\u54c1\u8d28",
        "\u8d28\u91cf",
        "\u54c1\u724c",
        "\u5e02\u573a",
        "\u5de5\u827a",
        "\u751f\u4ea7",
        "\u9500\u552e",
        "\u6e20\u9053",
        "\u7ecf\u8425",
        "\u7ade\u4e89\u4f18\u52bf",
        "\u4e3b\u8981\u4e1a\u52a1",
        "\u6838\u5fc3\u4ea7\u533a",
        "\u8305\u53f0\u9152",
        "\u9ad8\u8d28\u91cf\u53d1\u5c55",
        "\u6570\u5b57\u5316",
    }


def _global_context_weak_markers() -> set[str]:
    return {
        "\u672a\u77e5\u672a\u77e5",
        "\u524d\u5341\u540d\u80a1\u4e1c",
        "\u80a1\u4e1c\u60c5\u51b5",
        "\u80a1\u4e1c\u5173\u8054\u5173\u7cfb",
        "\u666e\u901a\u80a1\u80a1\u4e1c",
        "\u6301\u80a1\u60c5\u51b5",
        "\u8868\u51b3\u6743",
        "\u4e00\u81f4\u884c\u52a8",
        "\u4e00\u81f4\u884c\u52a8\u4eba",
        "\u5b63\u5ea6\u6570\u636e",
        "\u516c\u53f8\u4ee3\u7801",
        "\u516c\u53f8\u7b80\u79f0",
        "\u91cd\u8981\u63d0\u793a",
        "sse.com.cn",
        "\u6295\u8d44\u8005\u5e94\u5f53",
    }


def _query_allows_shareholder_topics(query: str) -> bool:
    text = str(query or "")
    return any(
        marker in text
        for marker in {
            "\u80a1\u4e1c",
            "\u6301\u80a1",
            "\u8868\u51b3\u6743",
            "\u4e00\u81f4\u884c\u52a8",
            "\u5173\u8054\u5173\u7cfb",
        }
    )


def _graph_generation_prompt(prompt: str, graph_context: str) -> str:
    return f"{prompt}\n\nGraph evidence:\n{graph_context}"


def _unpack_llm_response(raw: Any) -> tuple[str, dict[str, int]]:
    usage: Any = {}
    if isinstance(raw, tuple) and len(raw) == 2:
        raw, usage = raw
    if isinstance(raw, str):
        answer = raw
    elif isinstance(raw, dict):
        answer = raw.get("answer") or raw.get("content") or ""
        usage = raw.get("usage", usage)
    else:
        answer = getattr(raw, "content", "")
        usage = getattr(raw, "usage_metadata", usage)
        if not usage:
            response_metadata = getattr(raw, "response_metadata", {}) or {}
            usage = response_metadata.get("token_usage", {})
    if isinstance(answer, list):
        answer = " ".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in answer
        )
    if not isinstance(answer, str):
        raise TypeError("graph generator answer must be a string")
    normalized_usage = {
        str(key): int(value)
        for key, value in dict(usage or {}).items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    return answer.strip(), normalized_usage


def _clean_graph_label(value: Any) -> str:
    text = str(value or "").replace("Community:", "").strip()
    text = text.strip(" `\"'.,;:()[]{}")
    text = " ".join(text.split())
    text = re.sub(r"^(?:com\s+)+", "", text, flags=re.I)
    if "\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1" in text and len(text) > 20:
        return "\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1"
    return text


def _is_weak_graph_label(value: str) -> bool:
    text = _clean_graph_label(value)
    weak_labels = {
        "\u4e00",
        "\u4e8c",
        "\u4e09",
        "\u56db",
        "\u4e94",
        "\u516d",
        "\u4e03",
        "\u516b",
        "\u4e5d",
        "\u5341",
        "\u5f52",
        "\u4e0a\u5e02",
        "\u4e0a\u5e02\u516c\u53f8\u80a1\u4e1c\u7684",
    }
    if len(text) <= 1 or text in weak_labels:
        return True
    if text.endswith("\u7684") and len(text) <= 12:
        return True
    return False


def _dedupe_texts(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = _clean_graph_label(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _global_topic_labels(hit: GraphGlobalSearchHit) -> List[str]:
    title_labels = _dedupe_texts(_clean_graph_label(hit.title).split(","))
    entity_labels = _dedupe_texts(hit.key_entities)
    labels = [
        label
        for label in [*title_labels, *entity_labels]
        if not _is_weak_graph_label(label)
    ]
    return _dedupe_texts(labels)


def _is_displayable_global_hit(hit: GraphGlobalSearchHit) -> bool:
    topics = _global_topic_labels(hit)
    relationships = [
        _relationship_phrase(relationship)
        for relationship in hit.key_relationships
    ]
    return bool(topics) and any(
        not _is_weak_graph_label(relationship) for relationship in relationships
    )


def _relationship_phrase(value: str) -> str:
    text = _clean_graph_label(value)
    relation_labels = {
        " is_a ": "\u4f53\u73b0\u4e3a",
        " belongs_to ": "\u5f52\u5c5e\u4e8e",
        " contains ": "\u5305\u542b",
        " next_step ": "\u4e0b\u4e00\u6b65\u662f",
        " depends_on ": "\u4f9d\u8d56",
        " relates_to ": "\u76f8\u5173\u4e8e",
        " calls ": "\u8c03\u7528",
    }
    for marker, verb in relation_labels.items():
        if marker not in text:
            continue
        source, target = text.split(marker, 1)
        source = _clean_graph_label(source)
        target = _clean_graph_label(target)
        if _is_weak_graph_label(target):
            return ""
        if _is_weak_graph_label(source) and target:
            return target
        if source and target:
            return f"{source} {verb} {target}"
    return text


def _select_local_hits_for_query(
    query: str,
    hits: Sequence[GraphSearchHit],
) -> List[GraphSearchHit]:
    if not hits:
        return []
    displayable_hits = [
        hit for hit in hits if not _is_noisy_local_hit(hit, query)
    ]
    if not displayable_hits:
        displayable_hits = list(hits)

    intent = _local_query_intent(query)
    if not intent:
        return _dedupe_local_hits_by_relationship(displayable_hits, query)

    if "process" in intent:
        process_hits = [
            hit
            for hit in displayable_hits
            if _local_hit_is_process_container(hit)
        ]
        if process_hits:
            return process_hits

    if "business" in intent:
        business_hits = [
            hit
            for hit in displayable_hits
            if _local_hit_has_business_signal(hit)
        ]
        if business_hits:
            return _dedupe_local_hits_by_relationship(business_hits, query)

    if "channel" not in intent and "process" not in intent:
        focused_hits = _select_focused_local_hits(query, displayable_hits)
        if focused_hits:
            return focused_hits

    selected = [hit for hit in displayable_hits if _local_hit_matches_intent(hit, intent)]
    if selected:
        return _dedupe_local_hits_by_relationship(selected, query)
    return _dedupe_local_hits_by_relationship(displayable_hits, query)


def _focus_local_hits_for_query(
    query: str,
    hits: Sequence[GraphSearchHit],
) -> List[GraphSearchHit]:
    focused_hits = []
    for hit in hits:
        relationships = _focused_relationships_for_hit(query, hit)
        neighbors = _focused_neighbors_for_relationships(hit, relationships)
        focused_hits.append(
            hit.model_copy(
                update={
                    "relationships": relationships,
                    "neighbor_entities": neighbors,
                }
            )
        )
    return focused_hits


def _local_query_intent(query: str) -> set[str]:
    text = str(query or "").lower()
    intent = set()
    if any(marker in text for marker in {"\u6d41\u7a0b", "\u6b65\u9aa4", "\u73af\u8282", "\u5148\u540e", "process", "step"}):
        intent.add("process")
    if any(marker in text for marker in {"\u6e20\u9053", "\u4e3b\u4f53", "\u9500\u552e", "channel"}):
        intent.add("channel")
    if any(marker in text for marker in {"\u4e3b\u8981\u4e1a\u52a1", "\u4e1a\u52a1", "\u751f\u4ea7\u4e0e\u9500\u552e", "business"}):
        intent.add("business")
    if any(marker in text for marker in {"\u5305\u62ec", "\u5305\u542b", "\u54ea\u4e9b", "include", "contain"}):
        intent.add("contains")
    if any(marker in text for marker in {"\u5173\u7cfb", "\u4e4b\u95f4", "\u5148\u540e", "relationship", "relation"}):
        intent.add("relationship")
    return intent


def _local_hit_matches_intent(hit: GraphSearchHit, intent: set[str]) -> bool:
    entity = _clean_graph_label(hit.entity_name).lower()
    relation_types = {relationship.relation_type for relationship in hit.relationships}
    descriptions = " ".join(
        _clean_graph_label(relationship.description or relationship.relation_type)
        for relationship in hit.relationships
    ).lower()
    if "process" in intent:
        return (
            "\u6d41\u7a0b" in entity
            or "process" in entity
            or "next_step" in relation_types
            or "\u6d41\u7a0b" in descriptions
        )
    if "channel" in intent:
        return (
            "\u6e20\u9053" in entity
            or "channel" in entity
            or "\u6e20\u9053" in descriptions
        )
    if "business" in intent:
        return _local_hit_has_business_signal(hit)
    if "contains" in intent:
        return "contains" in relation_types
    if "relationship" in intent:
        return bool(hit.relationships)
    return True


def _select_focused_local_hits(
    query: str,
    hits: Sequence[GraphSearchHit],
) -> List[GraphSearchHit]:
    scored = [
        (hit, _local_focus_score(query, hit))
        for hit in hits
    ]
    max_score = max((score for _, score in scored), default=0.0)
    if max_score < 2.0:
        return []
    selected = [
        hit
        for hit, score in scored
        if score >= max(2.0, max_score * 0.55)
    ]
    return _dedupe_local_hits_by_relationship(selected, query)


def _local_focus_score(query: str, hit: GraphSearchHit) -> float:
    terms = _query_focus_terms(query)
    if not terms:
        return 0.0
    query_lower = str(query or "").lower()
    entity = _clean_graph_label(hit.entity_name).lower()
    descriptions = " ".join(
        _clean_graph_label(relationship.description or relationship.relation_type)
        for relationship in hit.relationships
    ).lower()
    neighbors = " ".join(
        _clean_graph_label(entity.name)
        for entity in hit.neighbor_entities
    ).lower()
    score = 0.0
    if entity and entity in query_lower:
        score += 4.0
    for term in terms:
        term_lower = term.lower()
        if term_lower in entity:
            score += 2.5
        if term_lower in descriptions:
            score += 1.5
        if term_lower in neighbors:
            score += 0.8
    if _is_noisy_graph_label(entity):
        score -= 4.0
    return score


def _query_focus_terms(query: str) -> List[str]:
    text = str(query or "")
    terms = []
    terms.extend(re.findall(r"[\u201c\"]([^\"\u201d]{2,30})[\u201d\"]", text))
    terms.extend(
        re.findall(
            (
                r"[\u4e00-\u9fffA-Za-z0-9]{2,30}?"
                r"(?:\u6e20\u9053|\u6d41\u7a0b|\u73af\u8282|\u7279\u5f81|"
                r"\u4e1a\u52a1|\u4ea7\u54c1|\u4ea7\u533a|\u8425\u9500\u7f51\u7edc|"
                r"\u751f\u4ea7\u4e0e\u9500\u552e|\u54c1\u8d28)"
            ),
            text,
        )
    )
    split_parts = re.split(
        (
            r"[\s\u3001\uff0c,。？！?；;]+|"
            r"\u662f\u4ec0\u4e48|\u6709\u54ea\u4e9b|\u54ea\u4e9b|"
            r"\u5305\u62ec|\u4e0e|\u548c|\u7684|\u5b83\u4eec|\u8fd9\u4e9b|"
            r"\u4e4b\u95f4|\u5173\u7cfb|\u5206\u522b"
        ),
        text,
    )
    terms.extend(split_parts)
    stop_terms = {
        "",
        "\u8d35\u5dde\u8305\u53f0",
        "\u516c\u53f8",
        "2025",
        "2025\u5e74",
        "\u5e74\u5ea6",
        "\u5e74\u5ea6\u62a5\u544a",
        "\u62a5\u544a\u6458\u8981",
        "\u4ec0\u4e48",
        "\u54ea\u4e9b",
        "\u5173\u7cfb",
    }
    cleaned = []
    seen = set()
    for term in terms:
        item = _clean_graph_label(term)
        if len(item) < 2 or len(item) > 30 or item in stop_terms:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _dedupe_local_hits_by_relationship(
    hits: Sequence[GraphSearchHit],
    query: str,
) -> List[GraphSearchHit]:
    result = []
    seen_relationship_sets = set()
    for hit in sorted(
        hits,
        key=lambda item: (
            _local_focus_score(query, item),
            item.score,
            len(item.relationships),
        ),
        reverse=True,
    ):
        relationship_key = tuple(
            sorted(
                _clean_graph_label(relationship.description or relationship.relation_type)
                for relationship in hit.relationships
            )
        )
        if relationship_key and relationship_key in seen_relationship_sets:
            continue
        if relationship_key:
            seen_relationship_sets.add(relationship_key)
        result.append(hit)
    return result


def _focused_relationships_for_hit(
    query: str,
    hit: GraphSearchHit,
) -> List[Any]:
    clean_relationships = [
        relationship
        for relationship in hit.relationships
        if not _has_noise_marker(relationship.description or relationship.relation_type)
    ]
    if not clean_relationships:
        return []

    intent = _local_query_intent(query)
    entity = _clean_graph_label(hit.entity_name)
    terms = _query_focus_terms(query)

    if "process" in intent and _local_hit_is_process_container(hit):
        source_edges = [
            relationship
            for relationship in clean_relationships
            if (
                relationship.relation_type == "contains"
                and _relationship_source_target(relationship)[0] == entity
            )
        ]
        return _dedupe_relationships(source_edges or clean_relationships)

    scored: list[tuple[Any, float]] = []
    for relationship in clean_relationships:
        score = _relationship_focus_score(
            query=query,
            relationship=relationship,
            entity_name=entity,
            terms=terms,
            intent=intent,
        )
        if score > 0:
            scored.append((relationship, score))

    if scored:
        max_score = max(score for _, score in scored)
        selected = [
            relationship
            for relationship, score in sorted(
                scored,
                key=lambda item: item[1],
                reverse=True,
            )
            if score >= max(1.0, max_score * 0.45)
        ]
        return _dedupe_relationships(selected)

    return _dedupe_relationships(clean_relationships)


def _relationship_focus_score(
    *,
    query: str,
    relationship: Any,
    entity_name: str,
    terms: Sequence[str],
    intent: set[str],
) -> float:
    source, target = _relationship_source_target(relationship)
    description = _clean_graph_label(
        getattr(relationship, "description", "")
        or getattr(relationship, "relation_type", "")
    )
    relation_type = str(getattr(relationship, "relation_type", "") or "")
    haystack = " ".join([source, target, description]).lower()
    query_lower = str(query or "").lower()
    score = 0.0

    if entity_name and entity_name in {source, target}:
        score += 0.8
    if source and source.lower() in query_lower:
        score += 2.0
    if target and target.lower() in query_lower:
        score += 2.0
    for term in terms:
        term_lower = term.lower()
        if term_lower in haystack:
            score += 1.5
        if source and term_lower in source.lower():
            score += 1.0
        if target and term_lower in target.lower():
            score += 1.0

    if "contains" in intent and relation_type == "contains":
        score += 1.0
    if "relationship" in intent:
        score += 0.5
    if "business" in intent and _text_has_business_signal(haystack):
        score += 3.0
    if "channel" in intent and ("\u6e20\u9053" in haystack or "channel" in haystack):
        score += 2.0
    if "process" in intent and (
        "\u6d41\u7a0b" in haystack
        or relation_type == "next_step"
        or "process" in haystack
    ):
        score += 2.0
    return score


def _focused_neighbors_for_relationships(
    hit: GraphSearchHit,
    relationships: Sequence[Any],
) -> List[Any]:
    if not relationships:
        return []
    allowed = set()
    entity = _clean_graph_label(hit.entity_name)
    for relationship in relationships:
        source, target = _relationship_source_target(relationship)
        if source and source != entity:
            allowed.add(source)
        if target and target != entity:
            allowed.add(target)

    neighbors = []
    seen = set()
    for neighbor in hit.neighbor_entities:
        name = _clean_graph_label(getattr(neighbor, "name", ""))
        if not name or name in seen:
            continue
        if _is_noisy_entity_label(name):
            continue
        if allowed and name not in allowed:
            continue
        seen.add(name)
        neighbors.append(neighbor)
    return neighbors


def _relationship_source_target(relationship: Any) -> tuple[str, str]:
    description = _clean_graph_label(
        getattr(relationship, "description", "")
        or getattr(relationship, "relation_type", "")
    )
    relation_type = str(getattr(relationship, "relation_type", "") or "")
    markers = [
        f" {relation_type} " if relation_type else "",
        " is_a ",
        " belongs_to ",
        " contains ",
        " next_step ",
        " depends_on ",
        " relates_to ",
        " related_to ",
        " calls ",
    ]
    for marker in markers:
        if marker and marker in description:
            source, target = description.split(marker, 1)
            return _clean_graph_label(source), _clean_graph_label(target)
    return "", ""


def _dedupe_relationships(relationships: Sequence[Any]) -> List[Any]:
    result = []
    seen = set()
    for relationship in relationships:
        key = (
            getattr(relationship, "id", None),
            _clean_graph_label(
                getattr(relationship, "description", "")
                or getattr(relationship, "relation_type", "")
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(relationship)
    return result


def _is_noisy_local_hit(hit: GraphSearchHit, query: str) -> bool:
    query_text = str(query or "")
    if "\u80a1\u4e1c" in query_text or "\u4e00\u81f4\u884c\u52a8" in query_text:
        return False
    entity_label = _clean_graph_label(hit.entity_name)
    if _is_noisy_entity_label(entity_label):
        return True
    return any(
        _has_noise_marker(relationship.description or relationship.relation_type)
        for relationship in hit.relationships
    )


def _is_noisy_graph_label(value: Any) -> bool:
    text = _clean_graph_label(value)
    if not text:
        return True
    if _has_noise_marker(text):
        return True
    if len(text) > 80:
        return True
    return False


def _is_noisy_entity_label(value: Any) -> bool:
    text = _clean_graph_label(value)
    if not text:
        return True
    if _has_noise_marker(text):
        return True
    if len(text) > 80 and "\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1" not in text:
        return True
    return False


def _has_noise_marker(value: Any) -> bool:
    text = _clean_graph_label(value)
    noise_markers = {
        "\u672a\u77e5\u672a\u77e5",
        "\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb",
        "\u4e00\u81f4\u884c\u52a8",
        "\u524d\u5341\u540d\u80a1\u4e1c",
        "\u80a1\u4e1c\u60c5\u51b5",
        "\u5e74\u5ea6\u62a5\u544a\u6458\u8981\u4e09",
    }
    return any(marker in text for marker in noise_markers)


def _local_hit_is_process_container(hit: GraphSearchHit) -> bool:
    entity = _clean_graph_label(hit.entity_name).lower()
    relation_types = {relationship.relation_type for relationship in hit.relationships}
    contains_steps = _targets_for_relationships(
        hit.relationships,
        "contains",
        source=_clean_graph_label(hit.entity_name),
    )
    return (
        ("\u6d41\u7a0b" in entity or "process" in entity)
        and "contains" in relation_types
        and len(contains_steps) >= 2
    )


def _local_hit_has_business_signal(hit: GraphSearchHit) -> bool:
    text = " ".join(
        [
            _clean_graph_label(hit.entity_name),
            *[
                _clean_graph_label(relationship.description or relationship.relation_type)
                for relationship in hit.relationships
            ],
            *[_clean_graph_label(entity.name) for entity in hit.neighbor_entities],
        ]
    )
    return _text_has_business_signal(text)


def _text_has_business_signal(value: Any) -> bool:
    text = str(value or "")
    return any(
        marker in text
        for marker in {
            "\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1",
            "\u4e3b\u8981\u4e1a\u52a1",
            "\u8305\u53f0\u9152\u53ca\u7cfb\u5217\u9152",
            "\u751f\u4ea7\u4e0e\u9500\u552e",
        }
    )


def _local_template_answer(query: str, hits: Sequence[GraphSearchHit]) -> str:
    lines = [
        (
            "\u6839\u636e\u77e5\u8bc6\u56fe\u8c31\u7684\u5c40\u90e8"
            f"\u68c0\u7d22\uff0c\u627e\u5230 {len(hits)} "
            "\u4e2a\u76f8\u5173\u5b9e\u4f53\u6216\u5173\u7cfb\uff1a"
        )
    ]
    sequence_line = _local_process_sequence_line(query, hits)
    if sequence_line:
        lines.append(sequence_line)
    for index, hit in enumerate(hits, start=1):
        entity = _clean_graph_label(hit.entity_name)
        relationships = _dedupe_texts(
            _relationship_phrase(relationship.description or relationship.relation_type)
            for relationship in hit.relationships[:8]
        )
        neighbors = _dedupe_texts(entity.name for entity in hit.neighbor_entities[:8])
        relation_text = (
            "\uff1b".join(relationships)
            if relationships
            else "\u6682\u672a\u68c0\u7d22\u5230\u660e\u786e\u5173\u7cfb"
        )
        neighbor_text = (
            "\u3001".join(neighbors)
            if neighbors
            else "\u6682\u65e0\u660e\u786e\u76f8\u5173\u5b9e\u4f53"
        )
        lines.append(
            f"{index}. {entity}\uff1a\u5173\u7cfb\u7ebf\u7d22\uff1a"
            f"{relation_text}\uff1b\u76f8\u5173\u5b9e\u4f53\uff1a{neighbor_text}\u3002"
        )
    return "\n".join(lines)


def _local_process_sequence_line(query: str, hits: Sequence[GraphSearchHit]) -> str:
    query_text = str(query or "").lower()
    process_intent = any(
        marker in query_text
        for marker in {
            "\u6d41\u7a0b",
            "\u6b65\u9aa4",
            "\u73af\u8282",
            "\u5148\u540e",
            "process",
            "step",
        }
    )
    if not process_intent:
        return ""

    for hit in hits:
        entity = _clean_graph_label(hit.entity_name)
        if "\u6d41\u7a0b" not in entity and "process" not in entity.lower():
            continue
        steps = _targets_for_relationships(hit.relationships, "contains", source=entity)
        if len(steps) >= 2:
            return (
                "\u6d41\u7a0b\u987a\u5e8f\uff1a"
                + " -> ".join(steps)
                + "\u3002"
            )

    next_edges: list[tuple[str, str]] = []
    for hit in hits:
        next_edges.extend(_edges_for_relationships(hit.relationships, "next_step"))
    sequence = _sequence_from_edges(next_edges)
    if len(sequence) >= 2:
        return "\u6d41\u7a0b\u987a\u5e8f\uff1a" + " -> ".join(sequence) + "\u3002"
    return ""


def _targets_for_relationships(
    relationships: Sequence[Any],
    relation_type: str,
    *,
    source: str | None = None,
) -> List[str]:
    targets = []
    seen = set()
    marker = f" {relation_type} "
    for relationship in relationships:
        description = _clean_graph_label(
            getattr(relationship, "description", "") or getattr(relationship, "relation_type", "")
        )
        if marker not in description:
            continue
        left, right = description.split(marker, 1)
        left = _clean_graph_label(left)
        right = _clean_graph_label(right)
        if source is not None and left != source:
            continue
        if not right or right in seen:
            continue
        seen.add(right)
        targets.append(right)
    return targets


def _edges_for_relationships(
    relationships: Sequence[Any],
    relation_type: str,
) -> List[tuple[str, str]]:
    edges = []
    marker = f" {relation_type} "
    for relationship in relationships:
        description = _clean_graph_label(
            getattr(relationship, "description", "") or getattr(relationship, "relation_type", "")
        )
        if marker not in description:
            continue
        left, right = description.split(marker, 1)
        left = _clean_graph_label(left)
        right = _clean_graph_label(right)
        if left and right:
            edges.append((left, right))
    return edges


def _sequence_from_edges(edges: Sequence[tuple[str, str]]) -> List[str]:
    if not edges:
        return []
    next_by_source = {}
    targets = set()
    for source, target in edges:
        next_by_source.setdefault(source, target)
        targets.add(target)
    starts = [source for source in next_by_source if source not in targets]
    current = starts[0] if starts else edges[0][0]
    sequence = [current]
    seen = {current}
    while current in next_by_source:
        current = next_by_source[current]
        if current in seen:
            break
        seen.add(current)
        sequence.append(current)
    return sequence


def _global_template_answer(query: str, hits: Sequence[GraphGlobalSearchHit]) -> str:
    display_hits = [hit for hit in hits if _is_displayable_global_hit(hit)]
    if not display_hits:
        display_hits = list(hits)
    lines = [
        (
            "\u6839\u636e\u77e5\u8bc6\u56fe\u8c31\u7684\u5168\u5c40"
            f"\u68c0\u7d22\uff0c\u8be5\u95ee\u9898\u6d89\u53ca {len(display_hits)} "
            "\u7ec4\u4e3b\u9898\u793e\u533a\uff0c\u53ef\u5f52\u7eb3\u4e3a\uff1a"
        )
    ]
    for index, hit in enumerate(display_hits, start=1):
        topics = _global_topic_labels(hit)
        topic_text = "\u3001".join(topics[:4]) or f"\u4e3b\u9898 {index}"
        relationships = _dedupe_texts(
            _relationship_phrase(relationship)
            for relationship in hit.key_relationships[:4]
        )
        relation_text = (
            "\uff1b".join(relationships)
            if relationships
            else "\u6682\u672a\u68c0\u7d22\u5230\u660e\u786e\u5173\u7cfb"
        )
        lines.append(
            f"{index}. {topic_text}\uff1a\u56fe\u8c31\u5173\u7cfb\u663e\u793a"
            f"{relation_text}\u3002"
        )
    lines.append(
        "\u4ee5\u4e0a\u4e3b\u9898\u5171\u540c\u6784\u6210\u4e86\u516c\u53f8"
        "\u91cd\u8981\u4e8b\u9879\u7684\u4e3b\u8981\u5173\u7cfb\u7f51\uff1a"
        "\u54c1\u8d28\u3001\u54c1\u724c\u3001\u5e02\u573a\u3001\u5de5\u827a"
        "\u548c\u8d44\u672c\u5e02\u573a\u8868\u73b0\u7b49\u4e3b\u9898\u76f8\u4e92"
        "\u652f\u6491\u3002"
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
