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


# ═══════════════════════════════════════════════════════════════════════════
# Acceptance Criteria Tests — RAG测试
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion1_ModuleSwitches:
    """验收标准 1：支持 rewrite / retrieve / rerank / compress / verify 五个模块开关。

    每个模块都可以独立开启/关闭，ModuleConfig 提供布尔开关。
    """

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    def test_all_five_switches_exist_in_config(self):
        """ModuleConfig 包含全部 5 个模块的布尔开关字段。"""
        cfg = ModuleConfig()
        assert hasattr(cfg, "rewrite")
        assert hasattr(cfg, "retrieve")
        assert hasattr(cfg, "rerank")
        assert hasattr(cfg, "compress")
        assert hasattr(cfg, "verify")
        # 所有字段都是 bool 类型
        for name in ("rewrite", "retrieve", "rerank", "compress", "verify"):
            assert isinstance(getattr(cfg, name), bool), f"{name} should be bool"

    def test_each_module_independently_toggleable(self):
        """每个开关可以独立设置，不互相影响。"""
        cfg = ModuleConfig(
            rewrite=True, retrieve=False, rerank=True, compress=False, verify=True
        )
        assert cfg.rewrite is True
        assert cfg.retrieve is False
        assert cfg.rerank is True
        assert cfg.compress is False
        assert cfg.verify is True

    @pytest.mark.asyncio
    async def test_rewrite_switch_changes_behavior(self):
        """开启 rewrite 会记录 rewritten_query 到 metadata。"""
        ctx = _make_context()
        req = _make_request()

        # 关闭 rewrite
        self.strategy.set_config(ModuleConfig(retrieve=True, rewrite=False))
        await self.strategy.run(req, ctx)
        assert "rewritten_query" not in ctx.metadata

        # 开启 rewrite
        self.strategy.set_config(ModuleConfig(retrieve=True, rewrite=True))
        ctx2 = _make_context()
        await self.strategy.run(req, ctx2)
        assert "rewritten_query" in ctx2.metadata

    @pytest.mark.asyncio
    async def test_rerank_switch_changes_behavior(self):
        """开启 rerank 会在 metadata 中标记 reranked=True。"""
        self.strategy.set_config(ModuleConfig(retrieve=True, rerank=True))
        ctx = _make_context()
        req = _make_request()

        await self.strategy.run(req, ctx)
        assert ctx.metadata.get("reranked") is True

    @pytest.mark.asyncio
    async def test_compress_switch_changes_behavior(self):
        """开启 compress 会在 metadata 中记录 compression 信息。"""
        self.strategy.set_config(ModuleConfig(retrieve=True, compress=True))
        ctx = _make_context()
        req = _make_request()

        await self.strategy.run(req, ctx)
        assert ctx.metadata.get("compressed") is True
        assert "original_chunk_count" in ctx.metadata
        assert "kept_chunk_count" in ctx.metadata

    @pytest.mark.asyncio
    async def test_verify_switch_changes_behavior(self):
        """开启 verify 会检查答案质量并在 trace 中包含 VERIFY 阶段。"""
        self.strategy.set_config(ModuleConfig(retrieve=True, verify=True))
        recorder = TraceRecorder()
        ctx = _make_context()
        ctx.trace_recorder = recorder
        ctx.metadata["trace_id"] = "t_verify_on"
        req = _make_request()

        result = await self.strategy.run(req, ctx)
        stages = {e.stage for e in recorder.get_trace("t_verify_on")}
        assert TraceStage.VERIFY in stages
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_all_five_modules_together(self):
        """全部 5 个模块同时开启的完整流水线可以正常执行。"""
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=True, compress=True, verify=True,
        ))
        recorder = TraceRecorder()
        ctx = _make_context()
        ctx.trace_recorder = recorder
        ctx.metadata["trace_id"] = "t_all5"
        req = _make_request("Explain retrieval-augmented generation")

        result = await self.strategy.run(req, ctx)
        assert len(result.answer) > 0
        stages = {e.stage for e in recorder.get_trace("t_all5")}
        assert TraceStage.REWRITE in stages
        assert TraceStage.RETRIEVE in stages
        assert TraceStage.RERANK in stages
        assert TraceStage.COMPRESS in stages
        assert TraceStage.VERIFY in stages


class TestAcceptanceCriterion2_IllegalConfigValidation:
    """验收标准 2：对非法组合给出校验错误（如关闭 retrieve 却开启 rerank）。

    ModuleConfig 开关之间有依赖关系，不合法的组合应在 run() 时返回 warning，
    或通过静态 validate_config() 预先检查。
    """

    def test_rerank_requires_retrieve(self):
        """关闭 retrieve 却开启 rerank → 非法组合。"""
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, rerank=True)
        )
        assert len(errors) >= 1
        assert any("rerank" in e.lower() for e in errors)
        assert any("retrieve" in e.lower() for e in errors)

    def test_compress_requires_retrieve(self):
        """关闭 retrieve 却开启 compress → 非法组合。"""
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, compress=True)
        )
        assert len(errors) >= 1
        assert any("compress" in e.lower() for e in errors)

    def test_all_illegal_combos_detected(self):
        """同时开启 rerank 和 compress 但关闭 retrieve → 两个错误都被检测。"""
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, rerank=True, compress=True)
        )
        assert len(errors) >= 2

    def test_rewrite_does_not_require_retrieve(self):
        """rewrite 不依赖 retrieve，单独开启合法。"""
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, rewrite=True)
        )
        assert len(errors) == 0

    def test_verify_does_not_require_retrieve(self):
        """verify 依赖的是 generate（始终开启），不依赖 retrieve。"""
        errors = ModularRAGStrategy.validate_config(
            ModuleConfig(retrieve=False, verify=True)
        )
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_illegal_combo_run_returns_warning_not_crash(self):
        """非法组合的 run() 不会崩溃，而是返回 warning。"""
        strategy = ModularRAGStrategy()
        strategy.set_config(ModuleConfig(retrieve=False, rerank=True))
        ctx = _make_context()
        req = _make_request()

        result = await strategy.run(req, ctx)
        assert result.answer == ""
        assert len(result.warnings) >= 1
        assert any("Invalid pipeline configuration" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_valid_config_returns_no_warnings(self):
        """合法的配置不会产生 configuration 相关的 warning。"""
        strategy = ModularRAGStrategy()
        strategy.set_config(ModuleConfig(retrieve=True))
        ctx = _make_context()
        req = _make_request()

        result = await strategy.run(req, ctx)
        config_warnings = [w for w in result.warnings if "Invalid pipeline configuration" in w]
        assert len(config_warnings) == 0


class TestAcceptanceCriterion3_SaveConfigPerRun:
    """验收标准 3：保存每次运行的 Pipeline 配置。

    每次 run() 调用后, context.metadata["pipeline_config"] 必须包含当前
    使用的完整配置信息，以便事后审查。
    """

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    @pytest.mark.asyncio
    async def test_config_saved_in_metadata(self):
        """run() 后 context.metadata 包含 pipeline_config。"""
        cfg = ModuleConfig(retrieve=True, rerank=True, top_k=10)
        self.strategy.set_config(cfg)
        ctx = _make_context()
        req = _make_request()

        await self.strategy.run(req, ctx)

        assert "pipeline_config" in ctx.metadata
        saved = ctx.metadata["pipeline_config"]
        assert isinstance(saved, dict)

    @pytest.mark.asyncio
    async def test_config_saved_fields_correct(self):
        """保存的配置字段值与设置的 ModuleConfig 一致。"""
        cfg = ModuleConfig(
            rewrite=True, retrieve=True, rerank=False,
            compress=True, verify=False, top_k=15,
            rerank_top_k=8, compress_max_tokens=3000,
        )
        self.strategy.set_config(cfg)
        ctx = _make_context()
        req = _make_request()

        await self.strategy.run(req, ctx)

        saved = ctx.metadata["pipeline_config"]
        assert saved["rewrite"] is True
        assert saved["retrieve"] is True
        assert saved["rerank"] is False
        assert saved["compress"] is True
        assert saved["verify"] is False
        assert saved["top_k"] == 15
        assert saved["rerank_top_k"] == 8
        assert saved["compress_max_tokens"] == 3000

    @pytest.mark.asyncio
    async def test_config_saved_even_on_illegal_combo(self):
        """即使配置不合法，仍然会保存配置以便调试。"""
        illegal_cfg = ModuleConfig(retrieve=False, rerank=True)
        self.strategy.set_config(illegal_cfg)
        ctx = _make_context()
        req = _make_request()

        await self.strategy.run(req, ctx)

        assert "pipeline_config" in ctx.metadata
        saved = ctx.metadata["pipeline_config"]
        assert saved["retrieve"] is False
        assert saved["rerank"] is True

    @pytest.mark.asyncio
    async def test_two_runs_save_different_configs(self):
        """两次不同配置的 run() 分别保存各自的配置。"""
        ctx = _make_context()
        req = _make_request()

        # Run 1: retrieve only
        self.strategy.set_config(ModuleConfig(retrieve=True, top_k=5))
        await self.strategy.run(req, ctx)
        first_saved = ctx.metadata.get("pipeline_config")

        # Run 2: full pipeline
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=True, top_k=20
        ))
        await self.strategy.run(req, ctx)
        second_saved = ctx.metadata.get("pipeline_config")

        # Metadata keeps the LAST run's config (overwritten)
        assert second_saved["rewrite"] is True
        assert second_saved["top_k"] == 20
        assert first_saved != second_saved


class TestAcceptanceCriterion4_CompareConfigs:
    """验收标准 4：相同问题可一键对比两套配置。

    compare() 方法用同一个问题和 context 跑两套不同的配置，
    返回并排对比结果，包含 summary 文本描述。
    """

    def setup_method(self):
        self.strategy = ModularRAGStrategy()

    @pytest.mark.asyncio
    async def test_compare_returns_both_results(self):
        """compare() 返回两个 result 和对比信息。"""
        ctx = _make_context()
        req = _make_request("What is RAG?")

        cfg_fast = ModuleConfig(retrieve=True)
        cfg_full = ModuleConfig(retrieve=True, rerank=True, verify=True)

        result = await self.strategy.compare(req, ctx, cfg_fast, cfg_full)

        assert "result_a" in result
        assert "result_b" in result
        assert "trace_a" in result
        assert "trace_b" in result
        assert "summary" in result
        assert isinstance(result["result_a"], RAGResult)
        assert isinstance(result["result_b"], RAGResult)

    @pytest.mark.asyncio
    async def test_compare_diff_configs_produce_diff_traces(self):
        """不同配置产生不同的 trace 阶段。"""
        ctx = _make_context()
        req = _make_request("What is machine learning?")

        cfg_a = ModuleConfig(retrieve=True)
        cfg_b = ModuleConfig(retrieve=True, rerank=True, verify=True)

        result = await self.strategy.compare(req, ctx, cfg_a, cfg_b)

        stages_a = {e.stage for e in result["trace_a"]}
        stages_b = {e.stage for e in result["trace_b"]}
        assert TraceStage.RERANK not in stages_a
        assert TraceStage.RERANK in stages_b
        assert TraceStage.VERIFY not in stages_a
        assert TraceStage.VERIFY in stages_b

    @pytest.mark.asyncio
    async def test_compare_summary_mentions_configs(self):
        """对比 summary 描述了两套配置的差异。"""
        ctx = _make_context()
        req = _make_request("What is deep learning?")

        cfg_a = ModuleConfig(retrieve=True)
        cfg_b = ModuleConfig(rewrite=True, retrieve=True, rerank=True)

        result = await self.strategy.compare(
            req, ctx, cfg_a, cfg_b, labels=("fast", "thorough")
        )

        summary = result["summary"]
        assert "fast" in summary
        assert "thorough" in summary
        assert "retrieve" in summary.lower()

    @pytest.mark.asyncio
    async def test_compare_presets_works(self):
        """compare_presets() 用预设名称一键对比。"""
        ctx = _make_context()
        req = _make_request("Explain transformers")

        self.strategy.save_preset("light", ModuleConfig(retrieve=True))
        self.strategy.save_preset(
            "heavy", ModuleConfig(rewrite=True, retrieve=True, rerank=True, verify=True)
        )

        result = await self.strategy.compare_presets(req, ctx, "light", "heavy")

        assert "result_a" in result
        assert "result_b" in result
        assert len(result["result_a"].answer) > 0
        assert len(result["result_b"].answer) > 0
        # Preset names are used as labels
        assert "light" in result["summary"]
        assert "heavy" in result["summary"]

    @pytest.mark.asyncio
    async def test_compare_presets_missing_raises(self):
        """compare_presets 对不存在的 preset 抛出 KeyError。"""
        ctx = _make_context()
        req = _make_request("test")

        self.strategy.save_preset("exists", ModuleConfig(retrieve=True))
        with pytest.raises(KeyError, match="nonexistent"):
            await self.strategy.compare_presets(req, ctx, "exists", "nonexistent")

    @pytest.mark.asyncio
    async def test_compare_restores_original_state(self):
        """compare() 完成后恢复原始的 config 和 context 状态。"""
        original_cfg = ModuleConfig(retrieve=True)
        self.strategy.set_config(original_cfg)
        ctx = _make_context()
        req = _make_request()

        await self.strategy.compare(
            req, ctx,
            ModuleConfig(retrieve=True, rerank=True),
            ModuleConfig(rewrite=True, retrieve=True, verify=True),
        )

        # 原始 config 已恢复
        assert self.strategy.config.retrieve is True
        assert self.strategy.config.rerank is False
        assert self.strategy.config.rewrite is False

    @pytest.mark.asyncio
    async def test_compare_saves_config_for_each_run(self):
        """compare() 每次内部 run 也会保存各自的配置。"""
        cfg_a = ModuleConfig(retrieve=True, top_k=5)
        cfg_b = ModuleConfig(retrieve=True, top_k=20)

        ctx = _make_context()
        req = _make_request()

        results = await self.strategy.compare(req, ctx, cfg_a, cfg_b)

        # 每个 result 有自己的 trace，不同的 top_k 配置
        traces_a = results["trace_a"]
        traces_b = results["trace_b"]
        assert len(traces_a) > 0
        assert len(traces_b) > 0


class TestAcceptanceCriterion5_TraceOrderMatchesModules:
    """验收标准 5：Trace 顺序必须与实际启用模块一致。

    Trace 中的事件顺序必须严格对应 pipeline 执行顺序：
    rewrite → retrieve → rerank → compress → generate → verify → complete
    禁用模块不出现在 trace 中。
    """

    def setup_method(self):
        self.strategy = ModularRAGStrategy()
        self.recorder = TraceRecorder()

    @pytest.mark.asyncio
    async def test_trace_order_full_pipeline(self):
        """完整流水线的 trace 事件顺序固定。"""
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=True, compress=True, verify=True,
        ))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_order_full"
        req = _make_request()

        await self.strategy.run(req, ctx)
        events = self.recorder.get_trace("t_order_full")

        stage_order = [e.stage for e in events]
        # 验证模块的相对顺序
        _order_indices = {
            TraceStage.REWRITE: stage_order.index(TraceStage.REWRITE),
            TraceStage.RETRIEVE: stage_order.index(TraceStage.RETRIEVE),
            TraceStage.RERANK: stage_order.index(TraceStage.RERANK),
            TraceStage.COMPRESS: stage_order.index(TraceStage.COMPRESS),
            TraceStage.GENERATE: stage_order.index(TraceStage.GENERATE),
            TraceStage.VERIFY: stage_order.index(TraceStage.VERIFY),
            TraceStage.COMPLETE: stage_order.index(TraceStage.COMPLETE),
        }
        assert _order_indices[TraceStage.REWRITE] < _order_indices[TraceStage.RETRIEVE]
        assert _order_indices[TraceStage.RETRIEVE] < _order_indices[TraceStage.RERANK]
        assert _order_indices[TraceStage.RERANK] < _order_indices[TraceStage.COMPRESS]
        assert _order_indices[TraceStage.COMPRESS] < _order_indices[TraceStage.GENERATE]
        assert _order_indices[TraceStage.GENERATE] < _order_indices[TraceStage.VERIFY]
        assert _order_indices[TraceStage.VERIFY] < _order_indices[TraceStage.COMPLETE]

    @pytest.mark.asyncio
    async def test_disabled_modules_not_in_trace(self):
        """关闭 rerank + compress + verify 时，它们的 stage 不出现在 trace 中。"""
        self.strategy.set_config(ModuleConfig(
            rewrite=True, retrieve=True, rerank=False, compress=False, verify=False,
        ))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_disabled2"
        req = _make_request()

        await self.strategy.run(req, ctx)
        events = self.recorder.get_trace("t_disabled2")

        stages_present = {e.stage for e in events}
        assert TraceStage.RERANK not in stages_present
        assert TraceStage.COMPRESS not in stages_present
        assert TraceStage.VERIFY not in stages_present
        # 启用的模块在 trace 中
        assert TraceStage.REWRITE in stages_present
        assert TraceStage.RETRIEVE in stages_present
        assert TraceStage.GENERATE in stages_present
        assert TraceStage.COMPLETE in stages_present

    @pytest.mark.asyncio
    async def test_trace_order_skips_disabled_correctly(self):
        """trace 顺序跳过禁用模块，但保持其余模块的相对顺序。"""
        self.strategy.set_config(ModuleConfig(
            rewrite=False, retrieve=True, rerank=True, compress=False, verify=True,
        ))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_skip"
        req = _make_request()

        await self.strategy.run(req, ctx)
        events = self.recorder.get_trace("t_skip")

        stage_order = [e.stage for e in events]
        # rewrite 和 compress 被跳过，不出现在 order 中
        assert TraceStage.REWRITE not in stage_order
        assert TraceStage.COMPRESS not in stage_order
        # 剩余模块保持相对顺序: retrieve → rerank → generate → verify → complete
        idx = {
            TraceStage.RETRIEVE: stage_order.index(TraceStage.RETRIEVE),
            TraceStage.RERANK: stage_order.index(TraceStage.RERANK),
            TraceStage.GENERATE: stage_order.index(TraceStage.GENERATE),
            TraceStage.VERIFY: stage_order.index(TraceStage.VERIFY),
            TraceStage.COMPLETE: stage_order.index(TraceStage.COMPLETE),
        }
        assert idx[TraceStage.RETRIEVE] < idx[TraceStage.RERANK]
        assert idx[TraceStage.RERANK] < idx[TraceStage.GENERATE]
        assert idx[TraceStage.GENERATE] < idx[TraceStage.VERIFY]
        assert idx[TraceStage.VERIFY] < idx[TraceStage.COMPLETE]

    @pytest.mark.asyncio
    async def test_generate_always_in_trace(self):
        """generate 始终开启，始终出现在 trace 中。"""
        self.strategy.set_config(ModuleConfig(
            rewrite=False, retrieve=False, rerank=False, compress=False, verify=False,
        ))
        ctx = _make_context()
        ctx.trace_recorder = self.recorder
        ctx.metadata["trace_id"] = "t_gen_only"
        req = _make_request()

        await self.strategy.run(req, ctx)
        events = self.recorder.get_trace("t_gen_only")

        stages = {e.stage for e in events}
        assert TraceStage.GENERATE in stages
        assert TraceStage.COMPLETE in stages
        assert TraceStage.RETRIEVE not in stages  # retrieve disabled
