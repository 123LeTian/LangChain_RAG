"""Formal dependency-injected Advanced RAG strategy (B5)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import inspect
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from uuid import uuid4

from src.models.schemas import RetrievalHit
from src.rag.c_contract import resolve_advanced_mode
from src.rag.naive_support import (
    GeneratorAdapter,
    REFUSAL_ANSWER,
    build_advanced_instruction,
    build_bounded_context,
)
from src.rag.strategies.naive import (
    NaiveRAGStrategy,
    _TraceWriter,
    _mode_value,
    _string_warnings,
    build_naive_citations,
)
from src.retrieval.multi_query import (
    MultiQueryRetrievalError,
    merge_retrieval_hits,
    retrieve_queries,
)
from src.retrieval.query_rewriter import rewrite_queries

if TYPE_CHECKING:
    from src.models.rag import RAGContext, RAGRequest, RAGResult


class AdvancedRAGError(Exception):
    """Base error for Advanced RAG validation and orchestration."""


class AdvancedRAGValidationError(AdvancedRAGError):
    """Raised for an invalid request or configuration."""


class AdvancedRAGExecutionError(AdvancedRAGError):
    """Raised for malformed injected component output."""


def _top_ids(hits: Sequence[RetrievalHit], limit: int = 5) -> str:
    return ",".join(hit.chunk.id for hit in hits[:limit]) or "none"


def _text_chars(hits: Sequence[RetrievalHit]) -> int:
    return sum(len(hit.chunk.text) for hit in hits)


def _query_summary(queries: Sequence[str], limit: int = 160) -> str:
    return repr(
        [query if len(query) <= limit else f"{query[:limit]}..." for query in queries]
    )


def _passes_score_threshold(score: float, threshold: float) -> bool:
    return round(float(score), 2) >= threshold


def _dependency(primary: Any, context: Any, name: str) -> Any:
    return primary if primary is not None else getattr(context, name, None)


def _is_synthesis_query(query: str) -> bool:
    normalized = "".join(str(query or "").lower().split())
    keywords = (
        "为什么",
        "为何",
        "如何",
        "怎么",
        "怎样",
        "关系",
        "共同",
        "支撑",
        "体现",
        "区别",
        "分类",
        "联动",
        "串起来",
        "串联",
        "归纳",
        "总结",
        "meaning",
        "why",
        "how",
        "relationship",
        "support",
    )
    return any(keyword in normalized for keyword in keywords)


def _ensure_synthesis_context(
    compressed_hits: Sequence[RetrievalHit],
    source_hits: Sequence[RetrievalHit],
    *,
    min_hits: int = 3,
) -> List[RetrievalHit]:
    expanded = list(compressed_hits)
    if len(expanded) >= min_hits:
        return expanded

    seen = {
        (hit.chunk.document_id, hit.chunk.id)
        for hit in expanded
    }
    for hit in source_hits:
        key = (hit.chunk.document_id, hit.chunk.id)
        if key in seen:
            continue
        expanded.append(deepcopy(hit))
        seen.add(key)
        if len(expanded) >= min_hits:
            break
    return expanded


class AdvancedRAGStrategy(NaiveRAGStrategy):
    """Rewrite, retrieve, fuse, rerank, compress, generate, and cite."""

    mode = resolve_advanced_mode()

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        query_rewriter: Optional[Any] = None,
        hybrid_retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        compressor: Optional[Any] = None,
        default_top_k: int = 5,
        default_score_threshold: float = 0.0,
        default_max_context_chars: int = 8000,
        default_max_queries: int = 3,
        refusal_answer: str = REFUSAL_ANSWER,
        contract_factory: Optional[Any] = None,
    ) -> None:
        super().__init__(
            config=config,
            default_top_k=default_top_k,
            default_score_threshold=default_score_threshold,
            default_max_context_chars=default_max_context_chars,
            refusal_answer=refusal_answer,
            contract_factory=contract_factory,
        )
        self._defaults.update(
            {
                "max_queries": default_max_queries,
                "rewrite_enabled": True,
                "multi_query_enabled": True,
                "hybrid_enabled": True,
                "rerank_enabled": True,
                "compression_enabled": True,
            }
        )
        self._query_rewriter = query_rewriter
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._compressor = compressor

    async def run(
        self,
        request: RAGRequest,
        context: RAGContext,
    ) -> RAGResult:
        settings = self._validate_and_resolve_advanced(request, context)
        query = settings["query"]
        kb_id = settings["kb_id"]
        top_k = settings["top_k"]
        filters = settings["filters"]

        context_metadata = getattr(context, "metadata", {}) or {}
        trace_id = str(
            context_metadata.get("trace_id") or f"trace_{uuid4().hex[:16]}"
        )
        trace = _TraceWriter(context, self._factory, trace_id)
        overall_started_perf = time.perf_counter()
        warnings = _string_warnings(context_metadata.get("warnings"))
        warnings.extend(_string_warnings(settings.get("warnings")))

        queries = [query]
        if settings["rewrite_enabled"]:
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            rewriter = _dependency(
                self._query_rewriter,
                context,
                "query_rewriter",
            )
            try:
                if rewriter is None:
                    raise AdvancedRAGExecutionError(
                        "query rewriter is not configured"
                    )
                queries = await rewrite_queries(
                    rewriter,
                    query,
                    max_queries=settings["max_queries"],
                )
                rewrite_output = (
                    f"success=true, query_count={len(queries)}, "
                    f"queries={_query_summary(queries)}"
                )
            except Exception as exc:
                warning = self._degraded_warning("rewrite", exc)
                warnings.append(warning)
                queries = [query]
                rewrite_output = (
                    f"success=false, fallback=original_query, "
                    f"error={type(exc).__name__}"
                )
            trace.emit(
                "REWRITE",
                started_at,
                started_perf,
                (
                    f"original_query={_query_summary([query])}, "
                    f"max_queries={settings['max_queries']}"
                ),
                rewrite_output,
            )

        if settings["multi_query_enabled"]:
            retrieval_queries = queries
        elif settings["rewrite_enabled"] and len(queries) > 1:
            # Rewrite-only mode uses the first alternative while retaining the
            # original query in the rewrite plan and trace.
            retrieval_queries = [queries[1]]
        else:
            retrieval_queries = [query]

        retrieve_started_at = datetime.now(timezone.utc)
        retrieve_started_perf = time.perf_counter()
        reports = []
        report_labels: List[str] = []
        retrieval_errors: List[Exception] = []

        retriever = getattr(context, "retriever", None)
        try:
            report = await retrieve_queries(
                retriever,
                retrieval_queries,
                kb_id=kb_id,
                top_k=top_k,
                filters=filters,
            )
            reports.append(report)
            report_labels.append("vector")
            warnings.extend(f"vector {failure}" for failure in report.failures)
        except Exception as exc:
            retrieval_errors.append(exc)
            warnings.append(self._degraded_warning("vector retrieval", exc))

        if settings["hybrid_enabled"]:
            hybrid = _dependency(
                self._hybrid_retriever,
                context,
                "hybrid_retriever",
            )
            try:
                if hybrid is None:
                    raise AdvancedRAGExecutionError(
                        "hybrid retriever is not configured"
                    )
                report = await retrieve_queries(
                    hybrid,
                    retrieval_queries,
                    kb_id=kb_id,
                    top_k=top_k,
                    filters=filters,
                )
                reports.append(report)
                report_labels.append("hybrid")
                warnings.extend(
                    f"hybrid {failure}" for failure in report.failures
                )
                warnings.extend(
                    _string_warnings(getattr(hybrid, "last_warnings", None))
                )
            except Exception as exc:
                retrieval_errors.append(exc)
                warnings.append(self._degraded_warning("hybrid retrieval", exc))

        if not reports:
            exc = MultiQueryRetrievalError(
                "; ".join(str(error) for error in retrieval_errors)
                or "all retrieval branches failed"
            )
            trace.emit(
                "RETRIEVE",
                retrieve_started_at,
                retrieve_started_perf,
                (
                    f"kb_id={kb_id}, requested_top_k={top_k}, "
                    f"queries={_query_summary(retrieval_queries)}"
                ),
                f"failed: {type(exc).__name__}",
            )
            return self._error_result(
                trace,
                stage="retrieve",
                error=exc,
                used_hits=[],
                warnings=warnings,
            )

        fused_hits = merge_retrieval_hits(
            [report.hits for report in reports],
            top_k=top_k,
        )
        self._validate_hits(fused_hits, kb_id)
        raw_count = sum(sum(report.hit_counts.values()) for report in reports)
        count_summary = ";".join(
            f"{label}={report.hit_counts}"
            for label, report in zip(report_labels, reports)
        )
        trace.emit(
            "RETRIEVE",
            retrieve_started_at,
            retrieve_started_perf,
            (
                f"kb_id={kb_id}, requested_top_k={top_k}, "
                f"queries={_query_summary(retrieval_queries)}"
            ),
            (
                f"query_hits={count_summary}, pre_fusion_hits={raw_count}, "
                f"post_fusion_hits={len(fused_hits)}, "
                f"score_threshold={settings['score_threshold']:.2f}"
            ),
        )

        if not fused_hits:
            reason = (
                "retriever returned no hits"
            )
            warnings.append(reason)
            return self._refusal_result(
                trace=trace,
                overall_started_perf=overall_started_perf,
                warnings=warnings,
            )

        final_hits = fused_hits
        if settings["rerank_enabled"]:
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            before = _top_ids(final_hits)
            reranker = _dependency(self._reranker, context, "reranker")
            try:
                if reranker is None or not callable(
                    getattr(reranker, "rerank", None)
                ):
                    raise AdvancedRAGExecutionError(
                        "reranker must expose rerank()"
                    )
                raw = reranker.rerank(
                    query,
                    deepcopy(final_hits),
                    top_k=settings["rerank_top_k"],
                )
                if inspect.isawaitable(raw):
                    raw = await raw
                final_hits = self._normalize_stage_hits(
                    raw,
                    fused_hits,
                    kb_id,
                    "reranker",
                )
                rerank_status = "success=true"
            except Exception as exc:
                warnings.append(self._degraded_warning("rerank", exc))
                final_hits = deepcopy(fused_hits)
                rerank_status = (
                    f"success=false, fallback=pre_rerank, "
                    f"error={type(exc).__name__}"
                )
            trace.emit(
                "RERANK",
                started_at,
                started_perf,
                f"before_top={before}",
                f"{rerank_status}, after_top={_top_ids(final_hits)}",
            )

        score_threshold = settings["score_threshold"]
        if score_threshold > 0:
            before_score = len(final_hits)
            final_hits = [
                hit
                for hit in final_hits
                if _passes_score_threshold(hit.score, score_threshold)
            ]
            dropped_by_score = before_score - len(final_hits)
            if dropped_by_score:
                warnings.append(
                    f"{dropped_by_score} retrieval hit(s) were below score_threshold"
                )
        if not final_hits:
            warnings.append("all hits were below score_threshold")
            return self._refusal_result(
                trace=trace,
                overall_started_perf=overall_started_perf,
                warnings=warnings,
            )

        selection = None
        if settings["compression_enabled"]:
            started_at = datetime.now(timezone.utc)
            started_perf = time.perf_counter()
            before_chars = _text_chars(final_hits)
            before_hits = deepcopy(final_hits)
            compressor = _dependency(self._compressor, context, "compressor")
            try:
                if compressor is None or not callable(
                    getattr(compressor, "compress", None)
                ):
                    raise AdvancedRAGExecutionError(
                        "compressor must expose compress()"
                    )
                raw = compressor.compress(deepcopy(final_hits))
                if inspect.isawaitable(raw):
                    raw = await raw
                final_hits = self._normalize_stage_hits(
                    raw,
                    before_hits,
                    kb_id,
                    "compressor",
                )
                if _is_synthesis_query(query):
                    final_hits = self._normalize_stage_hits(
                        _ensure_synthesis_context(final_hits, before_hits),
                        before_hits,
                        kb_id,
                        "synthesis_context",
                    )
                compression_status = "success=true"
            except Exception as exc:
                warnings.append(self._degraded_warning("compression", exc))
                final_hits = before_hits
                compression_status = (
                    f"success=false, fallback=uncompressed, "
                    f"error={type(exc).__name__}"
                )
            selection = build_bounded_context(
                final_hits,
                settings["max_context_chars"],
            )
            warnings.extend(selection.warnings)
            trace.emit(
                "COMPRESS",
                started_at,
                started_perf,
                f"before_chars={before_chars}, before_hits={len(before_hits)}",
                (
                    f"{compression_status}, after_chars={_text_chars(final_hits)}, "
                    f"after_hits={len(final_hits)}, "
                    f"bounded_context_chars={len(selection.text)}"
                ),
            )

        if selection is None:
            selection = build_bounded_context(
                final_hits,
                settings["max_context_chars"],
            )
            warnings.extend(selection.warnings)

        if not selection.used_hits:
            warnings.append("compression or context budget admitted no chunks")
            return self._refusal_result(
                trace=trace,
                overall_started_perf=overall_started_perf,
                warnings=warnings,
            )

        generate_started_at = datetime.now(timezone.utc)
        generate_started_perf = time.perf_counter()
        try:
            generator = getattr(context, "llm", None)
            if generator is None:
                raise AdvancedRAGValidationError("context.llm is required")
            generation = await GeneratorAdapter(generator).generate(
                prompt=build_advanced_instruction(query),
                context=selection.text,
                **settings["generator_options"],
            )
        except Exception as exc:
            trace.emit(
                "GENERATE",
                generate_started_at,
                generate_started_perf,
                (
                    f"context_chunks={len(selection.used_hits)}, "
                    f"context_chars={len(selection.text)}"
                ),
                f"failed: {type(exc).__name__}",
            )
            return self._error_result(
                trace,
                stage="generate",
                error=exc,
                used_hits=selection.used_hits,
                warnings=warnings,
            )

        trace.emit(
            "GENERATE",
            generate_started_at,
            generate_started_perf,
            (
                f"context_chunks={len(selection.used_hits)}, "
                f"context_chars={len(selection.text)}"
            ),
            (
                f"success=true, answer_chars={len(generation.answer)}, "
                f"usage_present={bool(generation.usage)}"
            ),
        )

        citations, citation_warnings = build_naive_citations(
            selection.used_hits,
            self._factory,
        )
        warnings.extend(citation_warnings)
        result_hits = self._rag_chunks(selection.used_hits)
        complete_started_at = datetime.now(timezone.utc)
        complete_started_perf = time.perf_counter()
        total_ms = max(
            0.0,
            (time.perf_counter() - overall_started_perf) * 1000.0,
        )
        trace.emit(
            "COMPLETE",
            complete_started_at,
            complete_started_perf,
            f"answer_chars={len(generation.answer)}",
            (
                f"citations={len(citations)}, total_ms={total_ms:.3f}, "
                f"warnings={len(warnings)}"
            ),
        )
        return self._factory.result(
            answer=generation.answer,
            citations=citations,
            hits=result_hits,
            trace=list(trace.events),
            usage=deepcopy(generation.usage),
            warnings=warnings,
        )

    def _validate_and_resolve_advanced(
        self,
        request: Any,
        context: Any,
    ) -> Dict[str, Any]:
        query = getattr(request, "query", None)
        if not isinstance(query, str) or not query.strip():
            raise AdvancedRAGValidationError(
                "request.query must be a non-empty string"
            )
        query = query.strip()
        kb_id = getattr(request, "kb_id", None)
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise AdvancedRAGValidationError(
                "request.kb_id must be a non-empty string"
            )
        kb_id = kb_id.strip()
        if _mode_value(getattr(request, "mode", self.mode)) != "advanced":
            raise AdvancedRAGValidationError("request.mode must be advanced")

        context_config = getattr(context, "config", {}) or {}
        request_options = getattr(request, "options", {}) or {}
        if not isinstance(context_config, Mapping):
            raise AdvancedRAGValidationError("context.config must be a mapping")
        if not isinstance(request_options, Mapping):
            raise AdvancedRAGValidationError("request.options must be a mapping")
        settings = dict(self._defaults)
        settings.update(context_config)
        settings.update(self._strategy_config)
        settings.update(request_options)

        if settings.get("kb_id", kb_id) != kb_id:
            raise AdvancedRAGValidationError(
                "options cannot override request.kb_id"
            )
        self._positive_integer(settings.get("top_k"), "top_k")
        self._positive_integer(settings.get("max_queries"), "max_queries")
        self._positive_integer(
            settings.get("max_context_chars"),
            "max_context_chars",
        )
        rerank_top_k = settings.get("rerank_top_k", settings["top_k"])
        self._positive_integer(rerank_top_k, "rerank_top_k")
        threshold = settings.get("score_threshold")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise AdvancedRAGValidationError(
                "score_threshold must be between 0 and 1"
            )

        switches = (
            "rewrite_enabled",
            "multi_query_enabled",
            "hybrid_enabled",
            "rerank_enabled",
            "compression_enabled",
        )
        for switch in switches:
            if not isinstance(settings.get(switch), bool):
                raise AdvancedRAGValidationError(f"{switch} must be a boolean")

        filters = settings.get("filters") or {}
        if not isinstance(filters, Mapping):
            raise AdvancedRAGValidationError("filters must be a mapping")
        if filters.get("kb_id", kb_id) != kb_id:
            raise AdvancedRAGValidationError(
                "filters cannot override request.kb_id"
            )
        filters = {key: value for key, value in filters.items() if key != "kb_id"}
        generator_options = settings.get("generator_options") or {}
        if not isinstance(generator_options, Mapping):
            raise AdvancedRAGValidationError(
                "generator_options must be a mapping"
            )
        return {
            "query": query,
            "kb_id": kb_id,
            "top_k": settings["top_k"],
            "rerank_top_k": rerank_top_k,
            "score_threshold": float(threshold),
            "max_context_chars": settings["max_context_chars"],
            "max_queries": settings["max_queries"],
            "filters": dict(filters),
            "generator_options": dict(generator_options),
            "warnings": settings.get("warnings"),
            **{switch: settings[switch] for switch in switches},
        }

    @staticmethod
    def _positive_integer(value: Any, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AdvancedRAGValidationError(
                f"{name} must be a positive integer"
            )

    def _normalize_stage_hits(
        self,
        hits: Any,
        source_hits: Sequence[RetrievalHit],
        kb_id: str,
        component: str,
    ) -> List[RetrievalHit]:
        if not isinstance(hits, list):
            raise AdvancedRAGExecutionError(
                f"{component} must return a list"
            )
        self._validate_hits(hits, kb_id)
        sources = {hit.chunk.id: hit for hit in source_hits}
        normalized = []
        seen = set()
        for rank, output in enumerate(hits, start=1):
            source = sources.get(output.chunk.id)
            if source is None:
                raise AdvancedRAGExecutionError(
                    f"{component} returned unknown chunk '{output.chunk.id}'"
                )
            if output.chunk.document_id != source.chunk.document_id:
                raise AdvancedRAGExecutionError(
                    f"{component} changed document_id for chunk "
                    f"'{output.chunk.id}'"
                )
            if output.chunk.id in seen:
                continue
            seen.add(output.chunk.id)
            hit = deepcopy(output)
            chunk_metadata = deepcopy(source.chunk.metadata)
            chunk_metadata.update(deepcopy(hit.chunk.metadata))
            hit.chunk.metadata = chunk_metadata
            metadata = deepcopy(source.metadata)
            metadata.update(deepcopy(hit.metadata))
            metadata.setdefault(f"pre_{component}_rank", source.rank)
            hit.metadata = metadata
            hit.rank = len(normalized) + 1
            normalized.append(hit)
        return normalized

    @staticmethod
    def _degraded_warning(stage: str, error: Exception) -> str:
        detail = str(error).replace("\n", " ").strip()[:180]
        suffix = f": {detail}" if detail else ""
        return f"{stage} degraded ({type(error).__name__}){suffix}"


__all__ = [
    "AdvancedRAGError",
    "AdvancedRAGExecutionError",
    "AdvancedRAGStrategy",
    "AdvancedRAGValidationError",
]
