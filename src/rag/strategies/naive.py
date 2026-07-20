"""Formal dependency-injected Naive RAG strategy (B4)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from uuid import uuid4

from src.models.schemas import RetrievalHit
from src.rag.c_contract import CContractFactory, resolve_naive_mode
from src.rag.naive_support import (
    GeneratorAdapter,
    REFUSAL_ANSWER,
    UsedHit,
    build_bounded_context,
    build_citation_specs,
    build_naive_instruction,
)

if TYPE_CHECKING:
    from src.models.rag import RAGContext, RAGRequest, RAGResult

try:
    from src.rag.base import RAGStrategy as _CStrategyBase
except ImportError:
    class _CStrategyBase:  # type: ignore[no-redef]
        """Temporary structural base until C's frozen contract is merged."""


class NaiveRAGError(Exception):
    """Base error for Naive RAG validation and orchestration."""


class NaiveRAGValidationError(NaiveRAGError):
    """Raised for invalid requests or strategy configuration."""


class NaiveRAGExecutionError(NaiveRAGError):
    """Raised internally for malformed dependency results."""


def _mode_value(mode: Any) -> Any:
    return getattr(mode, "value", mode)


def _string_warnings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [item for item in value if isinstance(item, str)]
    return []


class _TraceWriter:
    """Create API TraceEvents and forward them to C-compatible recorders."""

    def __init__(
        self,
        context: Any,
        factory: Any,
        trace_id: str,
    ) -> None:
        self.context = context
        self.factory = factory
        self.trace_id = trace_id
        self.events: List[Any] = []

    def emit(
        self,
        stage_name: str,
        started_at: datetime,
        started_perf: float,
        input_summary: str,
        output_summary: str,
    ) -> Any:
        stage = self.factory.stage(stage_name)
        duration_ms = max(0.0, (time.perf_counter() - started_perf) * 1000.0)
        event = self.factory.trace_event(
            trace_id=self.trace_id,
            stage=stage,
            started_at=started_at,
            duration_ms=round(duration_ms, 3),
            input_summary=input_summary,
            output_summary=output_summary,
        )
        self.events.append(event)

        recorder = getattr(self.context, "trace_recorder", None)
        if recorder is not None:
            try:
                if callable(recorder):
                    recorder(stage, input_summary, output_summary, duration_ms)
                elif callable(getattr(recorder, "record", None)):
                    recorder.record(self.trace_id, event)
            except Exception:
                # C's contract explicitly treats observability as best effort.
                pass
        return event


class NaiveRAGStrategy(_CStrategyBase):
    """Retrieve once, build bounded context, generate once, and cite sources."""

    mode = resolve_naive_mode()

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        default_top_k: int = 5,
        default_score_threshold: float = 0.0,
        default_max_context_chars: int = 8000,
        refusal_answer: str = REFUSAL_ANSWER,
        contract_factory: Optional[Any] = None,
    ) -> None:
        self._defaults = {
            "top_k": default_top_k,
            "score_threshold": default_score_threshold,
            "max_context_chars": default_max_context_chars,
        }
        self._strategy_config = dict(config or {})
        self._refusal_answer = refusal_answer
        self._factory = contract_factory or CContractFactory()

    async def run(
        self,
        request: RAGRequest,
        context: RAGContext,
    ) -> RAGResult:
        """Execute C's ``run(request, context) -> RAGResult`` contract."""

        settings = self._validate_and_resolve(request, context)
        query = settings["query"]
        kb_id = settings["kb_id"]
        top_k = settings["top_k"]
        threshold = settings["score_threshold"]
        filters = settings["filters"]
        max_context_chars = settings["max_context_chars"]

        context_metadata = getattr(context, "metadata", {}) or {}
        trace_id = str(context_metadata.get("trace_id") or f"trace_{uuid4().hex[:16]}")
        trace = _TraceWriter(context, self._factory, trace_id)
        overall_started_perf = time.perf_counter()
        warnings = _string_warnings(context_metadata.get("warnings"))
        warnings.extend(_string_warnings(settings.get("warnings")))

        retrieve_started_at = datetime.now(timezone.utc)
        retrieve_started_perf = time.perf_counter()
        retrieve_input = (
            f"query_chars={len(query)}, kb_id={kb_id}, requested_top_k={top_k}"
        )
        try:
            retriever = getattr(context, "retriever", None)
            if retriever is None or not callable(getattr(retriever, "search", None)):
                raise NaiveRAGValidationError(
                    "context.retriever must expose VectorRetriever.search()"
                )
            raw_hits = retriever.search(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
                filters=filters,
            )
            self._validate_hits(raw_hits, kb_id)
        except Exception as exc:
            trace.emit(
                "RETRIEVE",
                retrieve_started_at,
                retrieve_started_perf,
                retrieve_input,
                f"failed: {type(exc).__name__}",
            )
            return self._error_result(
                trace,
                stage="retrieve",
                error=exc,
                used_hits=[],
                warnings=warnings,
            )

        filtered_hits = [hit for hit in raw_hits if hit.score >= threshold]
        dropped_by_score = len(raw_hits) - len(filtered_hits)
        if dropped_by_score:
            warnings.append(
                f"{dropped_by_score} retrieval hit(s) were below score_threshold"
            )
        selection = build_bounded_context(filtered_hits, max_context_chars)
        warnings.extend(selection.warnings)
        trace.emit(
            "RETRIEVE",
            retrieve_started_at,
            retrieve_started_perf,
            retrieve_input,
            (
                f"requested_top_k={top_k}, raw_hits={len(raw_hits)}, "
                f"score_filtered_hits={len(filtered_hits)}, "
                f"context_chunks={len(selection.used_hits)}"
            ),
        )

        if not selection.used_hits:
            if not raw_hits:
                reason = "retriever returned no hits"
            elif not filtered_hits:
                reason = "all hits were below score_threshold"
            else:
                reason = "context budget admitted no chunks"
            warnings.append(reason)
            return self._refusal_result(
                trace=trace,
                overall_started_perf=overall_started_perf,
                warnings=warnings,
            )

        generation_started_at = datetime.now(timezone.utc)
        generation_started_perf = time.perf_counter()
        generation_input = (
            f"context_chunks={len(selection.used_hits)}, "
            f"context_chars={len(selection.text)}"
        )
        try:
            generator = getattr(context, "llm", None)
            if generator is None:
                raise NaiveRAGValidationError("context.llm is required")
            generation = await GeneratorAdapter(generator).generate(
                prompt=build_naive_instruction(query),
                context=selection.text,
                **settings["generator_options"],
            )
        except Exception as exc:
            trace.emit(
                "GENERATE",
                generation_started_at,
                generation_started_perf,
                generation_input,
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
            generation_started_at,
            generation_started_perf,
            generation_input,
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
        total_ms = max(0.0, (time.perf_counter() - overall_started_perf) * 1000.0)
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

    def _validate_and_resolve(self, request: Any, context: Any) -> Dict[str, Any]:
        query = getattr(request, "query", None)
        if not isinstance(query, str) or not query.strip():
            raise NaiveRAGValidationError("request.query must be a non-empty string")
        query = query.strip()
        kb_id = getattr(request, "kb_id", None)
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise NaiveRAGValidationError("request.kb_id must be a non-empty string")
        kb_id = kb_id.strip()
        mode = getattr(request, "mode", self.mode)
        if _mode_value(mode) != "naive":
            raise NaiveRAGValidationError("request.mode must be naive")

        context_config = getattr(context, "config", {}) or {}
        request_options = getattr(request, "options", {}) or {}
        if not isinstance(context_config, Mapping):
            raise NaiveRAGValidationError("context.config must be a mapping")
        if not isinstance(request_options, Mapping):
            raise NaiveRAGValidationError("request.options must be a mapping")
        settings = dict(self._defaults)
        settings.update(context_config)
        settings.update(self._strategy_config)
        settings.update(request_options)

        option_kb_id = settings.get("kb_id", kb_id)
        if option_kb_id != kb_id:
            raise NaiveRAGValidationError("options cannot override request.kb_id")
        top_k = settings.get("top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise NaiveRAGValidationError("top_k must be a positive integer")
        threshold = settings.get("score_threshold")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise NaiveRAGValidationError(
                "score_threshold must be between 0 and 1"
            )
        max_context_chars = settings.get("max_context_chars")
        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars <= 0
        ):
            raise NaiveRAGValidationError(
                "max_context_chars must be a positive integer"
            )
        filters = settings.get("filters") or {}
        if not isinstance(filters, Mapping):
            raise NaiveRAGValidationError("filters must be a mapping")
        if filters.get("kb_id", kb_id) != kb_id:
            raise NaiveRAGValidationError("filters cannot override request.kb_id")
        filters = {key: value for key, value in filters.items() if key != "kb_id"}
        generator_options = settings.get("generator_options") or {}
        if not isinstance(generator_options, Mapping):
            raise NaiveRAGValidationError("generator_options must be a mapping")
        return {
            "query": query,
            "kb_id": kb_id,
            "top_k": top_k,
            "score_threshold": float(threshold),
            "max_context_chars": max_context_chars,
            "filters": dict(filters),
            "generator_options": dict(generator_options),
            "warnings": settings.get("warnings"),
        }

    @staticmethod
    def _validate_hits(hits: Any, kb_id: str) -> None:
        if not isinstance(hits, list):
            raise NaiveRAGExecutionError("retriever.search() must return a list")
        for hit in hits:
            if not isinstance(hit, RetrievalHit):
                raise NaiveRAGExecutionError(
                    "retriever.search() must return RetrievalHit objects"
                )
            if hit.chunk.kb_id != kb_id:
                raise NaiveRAGExecutionError(
                    f"retriever returned chunk '{hit.chunk.id}' from another kb"
                )

    def _rag_chunks(self, used_hits: Sequence[UsedHit]) -> List[Any]:
        chunks = []
        for used in used_hits:
            hit = used.hit
            metadata = deepcopy(hit.chunk.metadata)
            metadata.update(deepcopy(hit.metadata))
            metadata.update(
                {
                    "rank": hit.rank,
                    "retriever": hit.retriever,
                    "context_chars_used": len(used.used_text),
                    "context_truncated": used.truncated,
                }
            )
            filename = metadata.get("filename") or metadata.get("source") or None
            source = self._factory.source(
                document_id=hit.chunk.document_id,
                chunk_index=hit.chunk.index,
                source_path=metadata.get("source") or filename or None,
                title=metadata.get("section") or filename,
                page=metadata.get("page"),
                score=hit.score,
                metadata=metadata,
            )
            chunks.append(
                self._factory.chunk(
                    chunk_id=hit.chunk.id,
                    content=hit.chunk.text,
                    source=source,
                    embedding=deepcopy(hit.chunk.embedding),
                )
            )
        return chunks

    def _refusal_result(
        self,
        trace: _TraceWriter,
        overall_started_perf: float,
        warnings: List[str],
    ) -> Any:
        complete_started_at = datetime.now(timezone.utc)
        complete_started_perf = time.perf_counter()
        total_ms = max(0.0, (time.perf_counter() - overall_started_perf) * 1000.0)
        trace.emit(
            "COMPLETE",
            complete_started_at,
            complete_started_perf,
            "generation_skipped=true",
            f"citations=0, total_ms={total_ms:.3f}, warnings={len(warnings)}",
        )
        return self._factory.result(
            answer=self._refusal_answer,
            citations=[],
            hits=[],
            trace=list(trace.events),
            usage={},
            warnings=warnings,
        )

    def _error_result(
        self,
        trace: _TraceWriter,
        stage: str,
        error: Exception,
        used_hits: Sequence[UsedHit],
        warnings: List[str],
    ) -> Any:
        sanitized = str(error).replace("\n", " ").strip()[:200]
        warning = f"{stage} failed: {type(error).__name__}"
        if sanitized:
            warning = f"{warning}: {sanitized}"
        warnings = list(warnings) + [warning]
        error_started_at = datetime.now(timezone.utc)
        error_started_perf = time.perf_counter()
        trace.emit(
            "ERROR",
            error_started_at,
            error_started_perf,
            f"stage={stage}",
            f"error={type(error).__name__}",
        )
        return self._factory.result(
            answer="处理请求时发生错误，请稍后重试。",
            citations=[],
            hits=self._rag_chunks(used_hits),
            trace=list(trace.events),
            usage={},
            warnings=warnings,
        )


def build_naive_citations(
    used_hits: Sequence[UsedHit],
    contract_factory: Any,
) -> tuple[List[Any], List[str]]:
    """Publicly testable builder that emits C's canonical RAGCitation objects."""

    specs, warnings = build_citation_specs(used_hits)
    return (
        [
            contract_factory.citation(
                document_id=spec.document_id,
                chunk_id=spec.chunk_id,
                text_snippet=spec.quote,
            )
            for spec in specs
        ],
        warnings,
    )


__all__ = [
    "NaiveRAGError",
    "NaiveRAGExecutionError",
    "NaiveRAGStrategy",
    "NaiveRAGValidationError",
    "build_naive_citations",
]
