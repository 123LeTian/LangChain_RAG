"""
Tests for src/rag/registry.py — RAGStrategyRegistry (C3).
"""

import pytest

from src.models.rag import RAGMode, RAGContext, RAGRequest, RAGResult
from src.rag.base import RAGStrategy
from src.rag.registry import (
    RAGStrategyRegistry,
    RegistryError,
    StrategyNotRegisteredError,
    StrategyAlreadyRegisteredError,
    get_registry,
    set_registry,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_strategy(strategy_mode: RAGMode, name: str = "Test") -> RAGStrategy:
    """Factory: create a minimal RAGStrategy instance for testing."""

    class _S(RAGStrategy):
        mode = strategy_mode

        async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
            return RAGResult(answer=f"answer from {name}")

    _s = _S()
    _s.__class__.__name__ = name
    return _s


# ── Tests ──────────────────────────────────────────────────────────────────


class TestRAGStrategyRegistry:
    """Tests for the RAGStrategyRegistry (RAGMode-keyed, instance-based)."""

    def setup_method(self):
        self.registry = RAGStrategyRegistry()

    # ── Registration ─────────────────────────────────────────────────

    def test_register_and_get(self):
        strat = _make_strategy(RAGMode.NAIVE, "Naive")
        self.registry.register(RAGMode.NAIVE, strat)
        assert self.registry.is_registered(RAGMode.NAIVE)

        got = self.registry.get(RAGMode.NAIVE)
        assert got is strat

    def test_register_duplicate_raises(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE, "S1"))
        with pytest.raises(StrategyAlreadyRegisteredError):
            self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE, "S2"))

    def test_register_mode_mismatch_raises(self):
        strat = _make_strategy(RAGMode.NAIVE)
        with pytest.raises(ValueError, match="mode mismatch"):
            self.registry.register(RAGMode.AGENTIC, strat)

    def test_register_invalid_type_raises(self):
        with pytest.raises(TypeError, match="RAGMode"):
            self.registry.register("naive", _make_strategy(RAGMode.NAIVE))

    def test_replace_overwrites(self):
        s1 = _make_strategy(RAGMode.NAIVE, "v1")
        s2 = _make_strategy(RAGMode.NAIVE, "v2")
        self.registry.register(RAGMode.NAIVE, s1)
        self.registry.replace(RAGMode.NAIVE, s2)
        assert self.registry.get(RAGMode.NAIVE) is s2

    def test_register_or_replace(self):
        s1 = _make_strategy(RAGMode.NAIVE, "v1")
        s2 = _make_strategy(RAGMode.NAIVE, "v2")
        self.registry.register_or_replace(RAGMode.NAIVE, s1)
        self.registry.register_or_replace(RAGMode.NAIVE, s2)
        assert self.registry.get(RAGMode.NAIVE) is s2

    # ── Lookup ────────────────────────────────────────────────────────

    def test_get_unregistered_raises(self):
        with pytest.raises(StrategyNotRegisteredError) as exc:
            self.registry.get(RAGMode.ADVANCED)
        assert "advanced" in str(exc.value)

    def test_get_or_none_returns_none(self):
        assert self.registry.get_or_none(RAGMode.GRAPH) is None

    def test_get_or_none_returns_strategy(self):
        strat = _make_strategy(RAGMode.MODULAR)
        self.registry.register(RAGMode.MODULAR, strat)
        assert self.registry.get_or_none(RAGMode.MODULAR) is strat

    # ── Introspection ─────────────────────────────────────────────────

    def test_list_modes(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        self.registry.register(RAGMode.MODULAR, _make_strategy(RAGMode.MODULAR))
        modes = self.registry.list_modes()
        assert RAGMode.NAIVE in modes
        assert RAGMode.MODULAR in modes
        assert len(modes) == 2

    def test_list_modes_str(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        names = self.registry.list_modes_str()
        assert "naive" in names

    def test_registered_count(self):
        assert self.registry.registered_count == 0
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        assert self.registry.registered_count == 1

    def test_contains(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        assert RAGMode.NAIVE in self.registry
        assert RAGMode.GRAPH not in self.registry

    # ── Management ────────────────────────────────────────────────────

    def test_unregister(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        assert len(self.registry) == 1
        self.registry.unregister(RAGMode.NAIVE)
        assert len(self.registry) == 0
        assert not self.registry.is_registered(RAGMode.NAIVE)

    def test_unregister_unregistered_raises(self):
        with pytest.raises(StrategyNotRegisteredError):
            self.registry.unregister(RAGMode.AGENTIC)

    def test_clear(self):
        self.registry.register(RAGMode.NAIVE, _make_strategy(RAGMode.NAIVE))
        self.registry.register(RAGMode.MODULAR, _make_strategy(RAGMode.MODULAR))
        self.registry.clear()
        assert self.registry.registered_count == 0

    def test_all_five_modes(self):
        """Can register all five RAG paradigms."""
        for mode in RAGMode:
            self.registry.register(mode, _make_strategy(mode))
        assert self.registry.registered_count == 5
        for mode in RAGMode:
            assert self.registry.is_registered(mode)


class TestGlobalRegistry:
    """Tests for the global registry singleton."""

    def teardown_method(self):
        set_registry(RAGStrategyRegistry())

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_set_registry_replaces(self):
        old = get_registry()
        new = RAGStrategyRegistry()
        set_registry(new)
        assert get_registry() is new
        assert get_registry() is not old
