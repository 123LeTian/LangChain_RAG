"""
Tests for src/rag/registry.py — StrategyRegistry.
"""

import pytest

from src.models.rag import StrategyType, RAGQuery, RAGContext, RAGResponse
from src.rag.base import RAGStrategyBase
from src.rag.registry import (
    StrategyRegistry,
    StrategyNotFoundError,
    StrategyAlreadyRegisteredError,
    get_registry,
    set_registry,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_strategy(name: str, stype: StrategyType) -> type:
    """Factory: create a minimal strategy class for testing."""

    class _S(RAGStrategyBase):
        strategy_type = stype

        async def retrieve(self, query: RAGQuery) -> RAGContext:
            return RAGContext(query=query.text)

        async def generate(self, context: RAGContext, query: RAGQuery) -> RAGResponse:
            return RAGResponse(
                query_id=query.query_id, answer=name, strategy=stype
            )

        async def run(self, query: RAGQuery) -> RAGResponse:
            return RAGResponse(
                query_id=query.query_id, answer=name, strategy=stype
            )

    _S.__name__ = name
    _S.__qualname__ = name
    return _S


# ── Tests ──────────────────────────────────────────────────────────────────


class TestStrategyRegistry:
    """Tests for the strategy registry."""

    def setup_method(self):
        self.registry = StrategyRegistry()

    def test_register_and_get(self):
        Cls = _make_strategy("TestNaive", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        assert self.registry.is_registered(StrategyType.NAIVE)

        strategy = self.registry.get(StrategyType.NAIVE)
        assert strategy.strategy_type == StrategyType.NAIVE

    def test_register_duplicate_raises(self):
        Cls = _make_strategy("TestNaive", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        with pytest.raises(StrategyAlreadyRegisteredError):
            self.registry.register(StrategyType.NAIVE, Cls)

    def test_get_unregistered_raises(self):
        with pytest.raises(StrategyNotFoundError) as exc:
            self.registry.get(StrategyType.NAIVE)
        assert "naive" in str(exc.value)

    def test_list_names(self):
        Cls1 = _make_strategy("S1", StrategyType.NAIVE)
        Cls2 = _make_strategy("S2", StrategyType.MODULAR)
        self.registry.register(StrategyType.NAIVE, Cls1)
        self.registry.register(StrategyType.MODULAR, Cls2)
        names = self.registry.list_names()
        assert "naive" in names
        assert "modular" in names
        assert len(names) == 2

    def test_list_types(self):
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        types = self.registry.list_types()
        assert StrategyType.NAIVE in types

    def test_default_strategy_explicit(self):
        Cls = _make_strategy("S_Naive", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls, set_default=True)
        default = self.registry.get_default()
        assert default.strategy_type == StrategyType.NAIVE

    def test_default_strategy_first_registered(self):
        Cls = _make_strategy("S_Modular", StrategyType.MODULAR)
        self.registry.register(StrategyType.MODULAR, Cls)
        default = self.registry.get_default()
        assert default.strategy_type == StrategyType.MODULAR

    def test_unregister(self):
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        assert len(self.registry) == 1
        self.registry.unregister(StrategyType.NAIVE)
        assert len(self.registry) == 0
        assert not self.registry.is_registered(StrategyType.NAIVE)

    def test_unregister_unregistered_raises(self):
        with pytest.raises(StrategyNotFoundError):
            self.registry.unregister(StrategyType.NAIVE)

    def test_set_default(self):
        Cls1 = _make_strategy("S1", StrategyType.NAIVE)
        Cls2 = _make_strategy("S2", StrategyType.MODULAR)
        self.registry.register(StrategyType.NAIVE, Cls1)
        self.registry.register(StrategyType.MODULAR, Cls2)
        self.registry.set_default(StrategyType.MODULAR)
        default = self.registry.get_default()
        assert default.strategy_type == StrategyType.MODULAR

    def test_set_default_unregistered_raises(self):
        with pytest.raises(StrategyNotFoundError):
            self.registry.set_default(StrategyType.NAIVE)

    def test_clear(self):
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        self.registry.clear()
        assert len(self.registry) == 0

    def test_contains(self):
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        assert StrategyType.NAIVE in self.registry
        assert StrategyType.AGENTIC not in self.registry

    def test_decorator_registration(self):
        registry = StrategyRegistry()

        @registry.register(StrategyType.MODULAR)
        class DecoratedStrategy(RAGStrategyBase):
            strategy_type = StrategyType.MODULAR

            async def retrieve(self, query): pass
            async def generate(self, context, query): pass
            async def run(self, query): pass

        assert registry.is_registered(StrategyType.MODULAR)

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            self.registry.register(StrategyType.NAIVE, dict)  # type: ignore[arg-type]

    def test_get_metadata(self):
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(
            StrategyType.NAIVE, Cls,
            metadata={"version": "1.0", "author": "C"},
        )
        meta = self.registry.get_metadata(StrategyType.NAIVE)
        assert meta["version"] == "1.0"
        assert meta["author"] == "C"

    def test_lazy_instance(self):
        """Verify that instances are created lazily on first get()."""
        Cls = _make_strategy("S", StrategyType.NAIVE)
        self.registry.register(StrategyType.NAIVE, Cls)
        instance1 = self.registry.get(StrategyType.NAIVE)
        instance2 = self.registry.get(StrategyType.NAIVE)
        # Same singleton instance
        assert instance1 is instance2


class TestGlobalRegistry:
    """Tests for the global registry singleton."""

    def teardown_method(self):
        # Reset global registry after each test
        set_registry(StrategyRegistry())

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_set_registry_replaces(self):
        old = get_registry()
        new = StrategyRegistry()
        set_registry(new)
        assert get_registry() is new
        assert get_registry() is not old
