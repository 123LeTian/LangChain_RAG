"""
Tests for src/rag/strategies/modular.py — ModularRAGStrategy (configurable pipeline).
"""

import pytest

from src.models.rag import (
    RAGChunk,
    RAGContext,
    RAGMode,
    RAGRequest,
    RAGResult,
    RAGSource,
    TraceStage,
)
from src.rag.strategies.modular import (
    ModuleConfig,
    ModularRAGStrategy,
    validate_module_config,
)
from src.rag.trace import TraceRecorder


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_context(query: str = "What is RAG?") -> RAGContext:
    """Build a minimal RAGContext for testing."""
    return RAGContext(query=query)


def _make_request(query: str = "What is RAG?") -> RAGRequest:
    return RAGRequest(query=query, mode=RAGMode.MODULAR)


# ── ModuleConfig Tests ─────────────────────────────────────────────────────


class TestModuleConfig:
    """Tests for ModuleConfig — the pipeline switch configuration."""

    def test_default_config(self):
        cfg = ModuleConfig()
        assert cfg.rewrite is False
        assert cfg.retrieve is True
        assert cfg.rerank is False
        assert cfg.compress is False
        assert cfg.verify is False

    def test_full_config(self):
        cfg = ModuleConfig(
            rewrite=True, retrieve=True, rerank=True,
            compress=True, verify=True,
        )
        assert cfg.rewrite is True
        assert cfg.rerank is True

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ModuleConfig(unknown_field=True)  # type: ignore[call-arg]


class TestConfigValidation:
    """1b. Illegal pipeline configurations are rejected."""

    def test_valid_default(self):
        cfg = ModuleConfig()
        assert validate_module_config(cfg) == []

    def test_valid_full(self):
        cfg = ModuleConfig(rewrite=True, retrieve=True, rerank=True,
                           compress=True, verify=True)
        assert validate_module_config(cfg) == []

    def test_valid_retrieve_only(self):
        cfg = ModuleConfig(retrieve=True, rerank=False, compress=False)
        assert validate_module_config(cfg) == []

    def test_rerank_without_retrieve_invalid(self):
        cfg = ModuleConfig(retrieve=False, rerank=True)
        errors = validate_module_config(cfg)
        assert len(errors) >= 1
        assert any("rerank" in e.lower() for e in errors)
        assert any("retrieve" in e.lower() for e in errors)

    def test_compress_without_retrieve_invalid(self):
        cfg = ModuleConfig(retrieve=False, compress=True)
        errors = validate_module_config(cfg)
        assert len(errors) >= 1
        assert any("compress" in e.lower() for e in errors)

    def test_rewrite_alone_is_valid(self):
        """Rewrite doesn't depend on retrieve — it can be used alone."""
        cfg = ModuleConfig(rewrite=True, retrieve=False)
        assert validate_module_config(cfg) == []

    def test_verify_alone_is_valid(self):
        """Verify depends on generate (always on), not on retrieve."""
        cfg = ModuleConfig(verify=True, retrieve=False)
        assert validate_module_config(cfg) == []


# ── ModularRAGStrategy Tests — Normal Pipeline ────────────────────────────


class TestModularRAGNormalPipeline:
    """1a. Normal pipeline configurations execute successfully."""

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    @pytest.mark.asyncio
    async def test_retrieve_only_pipeline(self):
        """Simplest valid pipeline: retrieve only."""
        self.strategy.set_config(ModuleConfig(retrieve=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0
        assert len(result.hits) > 0
        assert result.warnings == []

    @pytest.mark.asyncio
    async def test_rewrite_retrieve_pipeline(self):
        """Rewrite + Retrieve pipeline."""
        self.strategy.set_config(ModuleConfig(rewrite=True, retrieve=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert isinstance(result, RAGResult)
        assert "rewritten_query" in ctx.metadata

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """All modules enabled: rewrite, retrieve, rerank, compress, verify."""
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=True,
            compress=True, verify=True,
        ))
        ctx = _make_context()
        req = _make_request("What is machine learning?")

        result = await self.strategy.run(req, ctx)
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_rerank(self):
        """Retrieve + Rerank pipeline."""
        self.strategy.set_config(ModuleConfig(retrieve=True, rerank=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert ctx.metadata.get("reranked") is True

    @pytest.mark.asyncio
    async def test_pipeline_with_verify(self):
        """Verify module checks answer quality."""
        self.strategy.set_config(ModuleConfig(retrieve=True, verify=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert isinstance(result, RAGResult)


class TestModularRAGIllegalPipeline:
    """1b. Illegal pipeline configurations return warnings."""

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    @pytest.mark.asyncio
    async def test_rerank_without_retrieve_returns_warning(self):
        self.strategy.set_config(ModuleConfig(retrieve=False, rerank=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert result.answer == ""
        assert len(result.warnings) >= 1
        assert any("rerank" in w.lower() for w in result.warnings)
        assert any("retrieve" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_compress_without_retrieve_returns_warning(self):
        self.strategy.set_config(ModuleConfig(retrieve=False, compress=True))
        ctx = _make_context()
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        assert result.answer == ""
        assert len(result.warnings) >= 1


class TestModularRAGTraceRecording:
    """TraceRecorder integration tests."""

    def setup_method(self):
        self.strategy = ModularRAGStrategy()
        self.recorder = TraceRecorder()

    @pytest.mark.asyncio
    async def test_trace_records_execution_order(self):
        """Trace events are in the correct execution order."""
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=True, verify=True,
        ))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder  # Inject recorder
        ctx.metadata["trace_id"] = "t_order_test"
        req = _make_request()

        await self.strategy.run(req, ctx)

        events = self.recorder.get_trace("t_order_test")
        assert len(events) >= 3  # At least retrieve, verify, complete

        # Verify order: rewrite → retrieve → rerank → verify
        stage_order = [e.stage for e in events]
        # Check that rewrite comes before retrieve
        if TraceStage.REWRITE in stage_order and TraceStage.RETRIEVE in stage_order:
            assert stage_order.index(TraceStage.REWRITE) < stage_order.index(TraceStage.RETRIEVE)
        # Check that rerank comes before verify
        if TraceStage.RERANK in stage_order and TraceStage.VERIFY in stage_order:
            assert stage_order.index(TraceStage.RERANK) < stage_order.index(TraceStage.VERIFY)

    @pytest.mark.asyncio
    async def test_disabled_modules_not_in_trace(self):
        """Disabled modules should NOT appear in the trace."""
        self.strategy.set_config(ModuleConfig(retrieve=True, rerank=False, compress=False, verify=False))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_disabled"
        req = _make_request()

        await self.strategy.run(req, ctx)

        events = self.recorder.get_trace("t_disabled")
        stages = {e.stage for e in events}
        assert TraceStage.RERANK not in stages
        assert TraceStage.COMPRESS not in stages
        assert TraceStage.VERIFY not in stages
        assert TraceStage.RETRIEVE in stages  # Enabled

    @pytest.mark.asyncio
    async def test_trace_events_have_duration(self):
        """Every trace event has a non-negative duration."""
        self.strategy.set_config(ModuleConfig(retrieve=True))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_duration"
        req = _make_request()

        await self.strategy.run(req, ctx)

        events = self.recorder.get_trace("t_duration")
        for evt in events:
            assert evt.duration_ms >= 0, f"Negative duration for {evt.stage.value}"


class TestModularRAGPresets:
    """Multiple named pipeline configuration support."""

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    def test_save_and_load_preset(self):
        cfg = ModuleConfig(retrieve=True, rerank=True, verify=True)
        self.strategy.save_preset("thorough", cfg)
        loaded = self.strategy.load_preset("thorough")
        assert loaded.rerank is True
        assert loaded.verify is True

    def test_apply_preset(self):
        cfg = ModuleConfig(rewrite=True, retrieve=True, compress=True)
        self.strategy.save_preset("compressed", cfg)
        self.strategy.apply_preset("compressed")
        assert self.strategy.config.compress is True
        assert self.strategy.config.rewrite is True

    def test_list_presets(self):
        assert self.strategy.list_presets() == []
        self.strategy.save_preset("fast", ModuleConfig(rewrite=False, retrieve=True))
        self.strategy.save_preset("full", ModuleConfig(rewrite=True, retrieve=True, rerank=True, verify=True))
        assert "fast" in self.strategy.list_presets()
        assert "full" in self.strategy.list_presets()

    def test_load_missing_preset_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            self.strategy.load_preset("nonexistent")

    @pytest.mark.asyncio
    async def test_preset_changes_behavior(self):
        """Different presets produce different trace contents."""
        recorder = TraceRecorder()
        ctx = _make_context()
        ctx.trace_recorder = recorder

        # Fast: retrieve only
        self.strategy.save_preset("fast", ModuleConfig(retrieve=True, rerank=False))
        self.strategy.apply_preset("fast")
        ctx.metadata["trace_id"] = "t_fast"
        await self.strategy.run(_make_request(), ctx)

        # Thorough: retrieve + rerank + verify
        self.strategy.save_preset("thorough", ModuleConfig(retrieve=True, rerank=True, verify=True))
        self.strategy.apply_preset("thorough")
        ctx.metadata["trace_id"] = "t_thorough"
        await self.strategy.run(_make_request(), ctx)

        fast_stages = {e.stage for e in recorder.get_trace("t_fast")}
        thorough_stages = {e.stage for e in recorder.get_trace("t_thorough")}

        assert len(thorough_stages) > len(fast_stages)
        assert TraceStage.RERANK in thorough_stages
        assert TraceStage.RERANK not in fast_stages


class TestModularRAGStrategyInterface:
    """Verify ModularRAGStrategy conforms to RAGStrategy."""

    def test_mode_is_modular(self):
        s = ModularRAGStrategy()
        assert s.mode == RAGMode.MODULAR
        assert s.strategy_mode == RAGMode.MODULAR

    def test_validate_config_static_method(self):
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, rerank=True)
        )
        assert len(errors) >= 1

    def test_config_property(self):
        s = ModularRAGStrategy()
        assert isinstance(s.config, ModuleConfig)

    def test_set_config(self):
        s = ModularRAGStrategy()
        new_cfg = ModuleConfig(rewrite=True, retrieve=True)
        s.set_config(new_cfg)
        assert s.config.rewrite is True
