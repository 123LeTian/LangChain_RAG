"""
Modular RAG Strategy — Owner: C
Composable, configurable RAG pipeline built from swappable modules.

The Modular RAG paradigm differs from Advanced RAG: instead of hardwiring
an enhanced pipeline, it composes modules dynamically.  Each module is
an independent "processing step".  Users toggle modules on/off via
:class:`ModuleConfig` to create custom pipelines.

Supported modules:
  rewrite   — query rewriting / expansion
  retrieve  — vector / hybrid / graph retrieval
  rerank    — relevance re-scoring of candidates
  compress  — context truncation for token budget
  verify    — answer factuality / quality check (generate is always on)

Pipeline execution order:
  Query → Rewrite → Retrieve → Rerank → Compress → Generate → Verify

Design rules:
  - Modules are MOCK implementations — no dependency on B or D code.
  - Real retriever / LLM / reranker are injected via RAGContext at runtime.
  - TraceRecorder logs every module's start/end with real execution order.
  - Config validation catches illegal switch combinations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
from src.rag.trace import get_recorder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ModuleConfig — Pipeline switch configuration
# ═══════════════════════════════════════════════════════════════════════════


class ModuleConfig(BaseModel):
    """Toggle-based pipeline configuration.

    Each field is a boolean switch.  Generate is always on (implied).
    Illegal combinations (e.g. ``rerank=True`` with ``retrieve=False``)
    are rejected at construction time.

    Example::

        config = ModuleConfig(rewrite=False, retrieve=True, rerank=True,
                              compress=False, verify=True)
        errors = ModularRAGStrategy.validate_config(config)
        assert len(errors) == 0
    """

    rewrite: bool = Field(
        default=False,
        description="Enable query rewriting / expansion before retrieval",
    )
    retrieve: bool = Field(
        default=True,
        description="Enable document retrieval (vector / hybrid / graph)",
    )
    rerank: bool = Field(
        default=False,
        description="Enable relevance re-scoring of retrieved chunks",
    )
    compress: bool = Field(
        default=False,
        description="Enable context compression to fit token budget",
    )
    verify: bool = Field(
        default=False,
        description="Enable answer verification / factuality check after generation",
    )

    # Per-module options
    top_k: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")
    compress_max_tokens: int = Field(default=4000, ge=100, description="Max tokens for compressed context")
    rerank_top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to keep after reranking")

    model_config = {"extra": "forbid"}


# ── Validation ──────────────────────────────────────────────────────────


# Mapping: module → set of modules it depends on (must all be enabled)
_MODULE_DEPENDENCIES: Dict[str, set] = {
    "rerank":   {"retrieve"},
    "compress": {"retrieve"},
    "verify":   set(),  # no hard dependency — generate is always on
}


def validate_module_config(config: ModuleConfig) -> List[str]:
    """Return a list of configuration errors.  Empty list means valid.

    Rules enforced:
      - ``rerank=True`` requires ``retrieve=True``  (nothing to rerank)
      - ``compress=True`` requires ``retrieve=True`` (nothing to compress)
    """
    errors: List[str] = []
    enabled = {name for name in ("rewrite", "retrieve", "rerank", "compress", "verify")
               if getattr(config, name)}

    for module, deps in _MODULE_DEPENDENCIES.items():
        if module in enabled:
            missing = deps - enabled
            if missing:
                errors.append(
                    f"Module '{module}' requires {sorted(missing)} to be enabled, "
                    f"but they are disabled"
                )
    return errors


# ═══════════════════════════════════════════════════════════════════════════
# Mock Modules — self-contained, no dependency on B or D implementations
# ═══════════════════════════════════════════════════════════════════════════


class _MockRewrite:
    """Mock: rewrites the query by lowercasing and trimming."""

    name = "rewrite"

    async def run(self, ctx: RAGContext, *, query: str, **__kw: Any) -> RAGContext:
        rewritten = query.strip()
        ctx.metadata["rewritten_query"] = rewritten
        ctx.metadata["original_query"] = query
        return ctx


class _MockRetrieve:
    """Mock: returns hard-coded chunks or calls injected retriever."""

    name = "retrieve"

    async def run(self, ctx: RAGContext, *, query: str, top_k: int = 5, **__kw: Any) -> RAGContext:
        # Try the real retriever first (injected via context)
        if ctx.retriever is not None:
            try:
                result = ctx.retriever.retrieve(query=query, top_k=top_k)
                ctx.chunks = list(result.chunks)
                ctx.total_candidates = len(ctx.chunks)
                ctx.retrieval_method = "vector"
                return ctx
            except Exception:
                pass

        # Mock fallback: generate fake chunks
        mock_chunks = [
            RAGChunk(
                content=f"[Mock chunk {i}] This is simulated content for query: {query}",
                source=RAGSource(document_id=f"mock-doc-{i}", chunk_index=i, score=0.9 - i * 0.1),
            )
            for i in range(min(top_k, 3))
        ]
        ctx.chunks = mock_chunks
        ctx.total_candidates = len(mock_chunks)
        ctx.retrieval_method = "mock"
        return ctx


class _MockRerank:
    """Mock: sorts chunks by score descending, keeps top-k."""

    name = "rerank"

    async def run(self, ctx: RAGContext, *, query: str,
                  rerank_top_k: int = 5, **__kw: Any) -> RAGContext:
        if not ctx.chunks:
            return ctx

        sorted_chunks = sorted(
            ctx.chunks,
            key=lambda c: c.source.score or 0.0,
            reverse=True,
        )
        ctx.chunks = sorted_chunks[:rerank_top_k]
        ctx.metadata["reranked"] = True
        ctx.metadata["rerank_top_k"] = rerank_top_k
        return ctx


class _MockCompress:
    """Mock: truncates chunks to fit a character budget."""

    name = "compress"

    async def run(self, ctx: RAGContext, *, query: str,
                  compress_max_tokens: int = 4000, **__kw: Any) -> RAGContext:
        chars_per_token = 3.5
        max_chars = int(compress_max_tokens * chars_per_token)
        total = 0
        kept: List[RAGChunk] = []

        for chunk in ctx.chunks:
            if total + len(chunk.content) <= max_chars:
                kept.append(chunk)
                total += len(chunk.content)
            else:
                remaining = max_chars - total
                if remaining > 200:
                    kept.append(RAGChunk(
                        content=chunk.content[:remaining] + "...",
                        source=chunk.source,
                    ))
                break

        ctx.metadata["compressed"] = True
        ctx.metadata["original_chunk_count"] = len(ctx.chunks)
        ctx.metadata["kept_chunk_count"] = len(kept)
        ctx.chunks = kept
        return ctx


class _MockGenerate:
    """Mock: generates an answer using the injected LLM or a mock fallback."""

    name = "generate"

    async def run(self, ctx: RAGContext, *, query: str, **__kw: Any) -> str:
        if ctx.llm is not None:
            try:
                context_text = ctx.combined_text if ctx.chunks else "No context available."
                answer = await ctx.llm.generate(
                    prompt="You are a helpful assistant. Answer based on the provided context.",
                    context=context_text,
                )
                return answer
            except Exception:
                pass

        if ctx.chunks:
            return f"[Mock answer] Based on {len(ctx.chunks)} retrieved chunks, the answer to '{query}' is: ..."
        return f"[Mock answer] No context available for query: '{query}'"


class _MockVerify:
    """Mock: checks if the answer looks reasonable."""

    name = "verify"

    async def run(self, ctx: RAGContext, *, answer: str, **__kw: Any) -> Dict[str, Any]:
        issues: List[str] = []

        if not answer or len(answer.strip()) < 10:
            issues.append("Answer is too short")
        if "[Mock answer]" in answer:
            issues.append("Answer is mock-generated (no LLM)")
        if not ctx.chunks and "no context" not in answer.lower():
            issues.append("Answer generated without retrieval context")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "answer_length": len(answer),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ModularRAGStrategy — the concrete strategy
# ═══════════════════════════════════════════════════════════════════════════


class ModularRAGStrategy(RAGStrategy):
    """Configurable modular RAG pipeline.

    Implements the :class:`RAGStrategy` interface.  Pipeline behaviour
    is controlled by :class:`ModuleConfig`.  Multiple named configurations
    can coexist and be switched at runtime.

    Usage::

        strategy = ModularRAGStrategy()
        strategy.set_config(ModuleConfig(rewrite=False, retrieve=True,
                                          rerank=True, compress=False, verify=True))

        result = await strategy.run(
            RAGRequest(query="What is RAG?", mode=RAGMode.MODULAR),
            context,  # injected by RAGService
        )
    """

    mode: RAGMode = RAGMode.MODULAR

    # ── Module execution order (positional) ──────────────────────────────

    _ORDER = ("rewrite", "retrieve", "rerank", "compress", "generate", "verify")

    # ── Constructor ─────────────────────────────────────────────────────

    def __init__(self, config: Optional[ModuleConfig] = None) -> None:
        self._config = config or ModuleConfig()
        self._mock_modules = {
            "rewrite":  _MockRewrite(),
            "retrieve": _MockRetrieve(),
            "rerank":   _MockRerank(),
            "compress": _MockCompress(),
            "generate": _MockGenerate(),
            "verify":   _MockVerify(),
        }
        # Named preset configurations
        self._presets: Dict[str, ModuleConfig] = {}

    # ── Config management ───────────────────────────────────────────────

    @property
    def config(self) -> ModuleConfig:
        """Return the current active configuration."""
        return self._config

    def set_config(self, config: ModuleConfig) -> None:
        """Replace the active configuration.

        Does NOT validate — call :meth:`validate_config` first if needed.
        """
        self._config = config

    def save_preset(self, name: str, config: ModuleConfig) -> None:
        """Save a named pipeline configuration preset.

        Args:
            name:   Preset name (e.g. ``"fast"``, ``"thorough"``).
            config: The ModuleConfig to save.
        """
        self._presets[name] = config

    def load_preset(self, name: str) -> ModuleConfig:
        """Load a previously saved preset.

        Raises:
            KeyError: If *name* is not a saved preset.
        """
        if name not in self._presets:
            available = list(self._presets) or ["(none)"]
            raise KeyError(
                f"Preset '{name}' not found. Available: {', '.join(available)}"
            )
        return self._presets[name]

    def apply_preset(self, name: str) -> None:
        """Load and apply a saved preset as the active config."""
        self._config = self.load_preset(name)

    def list_presets(self) -> List[str]:
        """Return all saved preset names."""
        return list(self._presets)

    @staticmethod
    def validate_config(config: ModuleConfig) -> List[str]:
        """Validate a :class:`ModuleConfig` and return a list of errors.

        Returns:
            Empty list if the configuration is valid; otherwise a list of
            human-readable error messages.
        """
        return validate_module_config(config)

    # ── RAGStrategy.run ─────────────────────────────────────────────────

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        """Execute the modular pipeline.

        Execution order (fixed):
            Query → Rewrite → Retrieve → Rerank → Compress → Generate → Verify

        Each enabled module records its start/end via ``context.trace_recorder``.

        The active pipeline config is saved into ``context.metadata["pipeline_config"]``
        so it can be inspected after the run (acceptance criterion 3).
        """
        config = self._config
        warnings: List[str] = []

        # ── Save active config for post-run inspection ─────────────
        context.metadata["pipeline_config"] = {
            "rewrite": config.rewrite,
            "retrieve": config.retrieve,
            "rerank": config.rerank,
            "compress": config.compress,
            "verify": config.verify,
            "top_k": config.top_k,
            "rerank_top_k": config.rerank_top_k,
            "compress_max_tokens": config.compress_max_tokens,
        }

        # ── Validate config ──────────────────────────────────────────
        errors = validate_module_config(config)
        if errors:
            return RAGResult(
                answer="",
                warnings=[f"Invalid pipeline configuration: {'; '.join(errors)}"],
            )

        # ── Get trace recorder ───────────────────────────────────────
        recorder = context.trace_recorder
        if recorder is None:
            recorder = get_recorder()

        trace_id = context.metadata.get("trace_id", f"trace_modular_{id(request)}")
        recorder.start_trace(trace_id)

        query = request.query
        answer = ""

        # ── Execute modules in fixed order ───────────────────────────
        for module_name in self._ORDER:
            if module_name == "generate":
                # Generate is always on (not configurable)
                pass
            elif not getattr(config, module_name, False):
                continue  # Module disabled

            module = self._mock_modules[module_name]

            # Start trace
            recorder.start(
                trace_id,
                _stage_for_module(module_name),
                input_summary=f"query={query[:80]}" if module_name != "verify" else f"answer={answer[:80]}",
            )

            try:
                if module_name == "generate":
                    answer = await module.run(context, query=query)
                    # Store answer in context for downstream modules
                    context.metadata["generated_answer"] = answer
                    recorder.end(
                        trace_id, _stage_for_module(module_name),
                        output_summary=f"answer={len(answer)} chars",
                    )
                elif module_name == "verify":
                    result = await module.run(context, answer=answer)
                    recorder.end(
                        trace_id, _stage_for_module(module_name),
                        output_summary=f"passed={result['passed']}, issues={len(result['issues'])}",
                    )
                    if not result["passed"]:
                        warnings.extend(result["issues"])
                else:
                    # Pass config parameters through
                    extra = {}
                    if module_name == "retrieve":
                        extra["top_k"] = config.top_k
                    elif module_name == "rerank":
                        extra["rerank_top_k"] = config.rerank_top_k
                    elif module_name == "compress":
                        extra["compress_max_tokens"] = config.compress_max_tokens

                    context = await module.run(context, query=query, **extra)
                    recorder.end(
                        trace_id, _stage_for_module(module_name),
                        output_summary=_summarise_module_output(module_name, context),
                    )
            except Exception as exc:
                logger.error("Module '%s' failed: %s", module_name, exc)
                recorder.end(
                    trace_id, _stage_for_module(module_name),
                    output_summary=f"error={exc}",
                )
                warnings.append(f"Module '{module_name}' failed: {exc}")

        # ── Build result ─────────────────────────────────────────────
        recorder.record(
            trace_id,
            TraceEvent(
                trace_id=trace_id,
                stage=TraceStage.COMPLETE,
                started_at=datetime.now(timezone.utc),
                duration_ms=0.0,
                input_summary="pipeline finished",
                output_summary=f"answer={len(answer)} chars",
            ),
        )

        # ── Build citations from retrieved chunks ───────────────────────
        citations = _build_citations(context.chunks)

        return RAGResult(
            answer=answer,
            citations=citations,
            hits=context.chunks,
            trace=recorder.get_trace(trace_id),
            warnings=warnings,
        )

    # ── Compare (acceptance criterion 4) ───────────────────────────────

    async def compare(
        self,
        request: RAGRequest,
        context: RAGContext,
        config_a: ModuleConfig,
        config_b: ModuleConfig,
        *,
        labels: tuple = ("A", "B"),
    ) -> Dict[str, Any]:
        """Run the same query with two pipeline configs and return side-by-side results.

        Implements "相同问题可一键对比两套配置" — run the same query through
        two different pipeline configurations and compare the outputs.

        Args:
            request:   The shared query request.
            context:   The shared RAG context (dependencies are reused).
            config_a:  First pipeline configuration.
            config_b:  Second pipeline configuration.
            labels:    Display labels for the two configs (default ``("A", "B")``).

        Returns:
            Dict with ``result_a``, ``result_b`` (the two :class:`RAGResult`),
            ``trace_a``, ``trace_b``, and a ``summary`` string for display.
        """
        # Snapshot original context state so we can restore it
        _original_chunks = list(context.chunks)
        _original_metadata = dict(context.metadata)
        _original_config = self._config

        try:
            # ── Run with config A ──────────────────────────────────────
            self.set_config(config_a)
            context.chunks = []
            context.metadata = {**_original_metadata}
            result_a = await self.run(request, context)
            trace_a = list(result_a.trace)

            # ── Run with config B ──────────────────────────────────────
            self.set_config(config_b)
            context.chunks = []
            context.metadata = {**_original_metadata}
            result_b = await self.run(request, context)
            trace_b = list(result_b.trace)

            # ── Build summary ──────────────────────────────────────────
            summary = self._build_comparison_summary(
                config_a, config_b, result_a, result_b, labels
            )

            return {
                "result_a": result_a,
                "result_b": result_b,
                "trace_a": trace_a,
                "trace_b": trace_b,
                "summary": summary,
            }
        finally:
            # Restore original state
            self._config = _original_config
            context.chunks = _original_chunks
            context.metadata = _original_metadata

    async def compare_presets(
        self,
        request: RAGRequest,
        context: RAGContext,
        preset_a: str,
        preset_b: str,
    ) -> Dict[str, Any]:
        """Like :meth:`compare`, but takes two preset names.

        Raises:
            KeyError: If either preset is not found.
        """
        cfg_a = self.load_preset(preset_a)
        cfg_b = self.load_preset(preset_b)
        return await self.compare(
            request, context, cfg_a, cfg_b, labels=(preset_a, preset_b)
        )

    @staticmethod
    def _build_comparison_summary(
        config_a: ModuleConfig,
        config_b: ModuleConfig,
        result_a: RAGResult,
        result_b: RAGResult,
        labels: tuple,
    ) -> str:
        """Build a human-readable side-by-side comparison table."""
        def _cfg_str(cfg: ModuleConfig) -> str:
            parts = [name for name in ("rewrite", "retrieve", "rerank", "compress", "verify")
                     if getattr(cfg, name)]
            return "+".join(parts) if parts else "generate-only"

        lines = [
            "═════════════════════════════════════════════",
            f"  Pipeline Comparison: '{labels[0]}' vs '{labels[1]}'",
            "═════════════════════════════════════════════",
            "",
            f"  {'':>12} | {labels[0]:<20} | {labels[1]:<20}",
            f"  {'Pipeline':>12} | {_cfg_str(config_a):<20} | {_cfg_str(config_b):<20}",
            f"  {'top_k':>12} | {config_a.top_k:<20} | {config_b.top_k:<20}",
            f"  {'answer_len':>12} | {len(result_a.answer):<20} | {len(result_b.answer):<20}",
            f"  {'hits':>12} | {len(result_a.hits):<20} | {len(result_b.hits):<20}",
            f"  {'warnings':>12} | {len(result_a.warnings):<20} | {len(result_b.warnings):<20}",
        ]

        if result_a.trace and result_b.trace:
            trace_a_stages = "+".join(e.stage.value for e in result_a.trace)
            trace_b_stages = "+".join(e.stage.value for e in result_b.trace)
            lines.extend([
                f"  {'trace':>12} | {trace_a_stages:<20} | {trace_b_stages:<20}",
            ])

        lines.append("")
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────


def _stage_for_module(module_name: str) -> TraceStage:
    """Map a module name to the corresponding TraceStage."""
    _map = {
        "rewrite":  TraceStage.REWRITE,
        "retrieve": TraceStage.RETRIEVE,
        "rerank":   TraceStage.RERANK,
        "compress": TraceStage.COMPRESS,
        "generate": TraceStage.GENERATE,
        "verify":   TraceStage.VERIFY,
    }
    return _map.get(module_name, TraceStage.ERROR)


def _summarise_module_output(module_name: str, ctx: RAGContext) -> str:
    """Produce a short human-readable summary of module output."""
    if module_name == "rewrite":
        rewritten = ctx.metadata.get("rewritten_query", "")
        return f"rewritten_query={rewritten[:80]}"
    elif module_name == "retrieve":
        return f"{len(ctx.chunks)} chunks, method={ctx.retrieval_method}"
    elif module_name == "rerank":
        return f"{len(ctx.chunks)} chunks after reranking"
    elif module_name == "compress":
        return f"{len(ctx.chunks)} chunks, {sum(len(c.content) for c in ctx.chunks)} chars"
    return f"done"


def _build_citations(chunks: List[RAGChunk]) -> List[RAGCitation]:
    """Build RAGCitation list from retrieved chunks."""
    citations: List[RAGCitation] = []
    for chunk in chunks:
        if chunk.chunk_id and chunk.source.document_id:
            citations.append(RAGCitation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.source.document_id,
                text_snippet=chunk.content[:200] if chunk.content else "",
            ))
    return citations


# ── Auto-register ─────────────────────────────────────────────────────────


try:
    from src.rag.registry import get_registry
    _reg = get_registry()
    if not _reg.is_registered(RAGMode.MODULAR):
        _reg.register(RAGMode.MODULAR, ModularRAGStrategy())
except Exception:
    pass
