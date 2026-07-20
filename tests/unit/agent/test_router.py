"""
Tests for src/agent/router.py — QueryRouter, CircuitBreaker, FallbackChain.
"""

import asyncio
import pytest

from src.models.rag import StrategyType
from src.agent.router import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackChain,
    FallbackEntry,
    QueryRouter,
    RouteDecision,
    default_fallback_chain,
    create_router_and_fallback,
)


# ── CircuitBreaker Tests ──────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for the CircuitBreaker."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open

    def test_failure_threshold_validation(self):
        with pytest.raises(ValueError):
            CircuitBreaker("test", failure_threshold=0)
        with pytest.raises(ValueError):
            CircuitBreaker("test", recovery_timeout=0)

    @pytest.mark.asyncio
    async def test_closed_to_open(self):
        cb = CircuitBreaker("test", failure_threshold=2)

        async def fail():
            raise RuntimeError("boom")

        # First failure
        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.CLOSED

        # Second failure → OPEN
        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_rejects(self):
        cb = CircuitBreaker("test", failure_threshold=1)

        async def fail():
            raise RuntimeError("boom")

        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.OPEN

        async def ok():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(ok)

    @pytest.mark.asyncio
    async def test_success_resets_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        async def fail():
            raise RuntimeError("boom")

        async def ok():
            return "success"

        # One failure
        try:
            await cb.call(fail)
        except RuntimeError:
            pass

        # One success resets counter
        result = await cb.call(ok)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_transition(self):
        """Test OPEN → HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)

        async def fail():
            raise RuntimeError("boom")

        # Open the circuit
        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.05)

        async def ok():
            return "recovered"

        # Should now be HALF_OPEN and allow one probe
        result = await cb.call(ok)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)

        async def fail():
            raise RuntimeError("boom")

        # Open the circuit
        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.05)

        # HALF_OPEN probe fails → re-opens
        try:
            await cb.call(fail)
        except RuntimeError:
            pass
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        # Force open via internal state (for test)
        cb._state = CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        cb = CircuitBreaker("test")
        stats = cb.stats
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0


# ── QueryRouter Tests ──────────────────────────────────────────────────────


class TestQueryRouter:
    """Tests for the QueryRouter."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_graph_keywords_route_to_graph_rag(self):
        d = self.router.route("How are entities connected in the knowledge graph hierarchy?")
        assert d.strategy == StrategyType.GRAPH_RAG, f"Expected GRAPH_RAG, got {d.strategy}"
        assert d.confidence > 0.5

    def test_agentic_keywords_route_to_agentic(self):
        d = self.router.route("Compare and contrast transformers vs RNNs for sequence modeling")
        assert d.strategy == StrategyType.AGENTIC, f"Expected AGENTIC, got {d.strategy}"

    def test_short_factual_routes_to_naive(self):
        d = self.router.route("What is machine learning?")
        assert d.strategy == StrategyType.NAIVE, f"Expected NAIVE, got {d.strategy}"

    def test_default_routes_to_advanced(self):
        d = self.router.route("Tell me about the history of artificial intelligence research")
        assert d.strategy == StrategyType.ADVANCED, f"Expected ADVANCED, got {d.strategy}"

    def test_relationship_keyword(self):
        d = self.router.route("What is the relationship between mitochondria and ATP production?")
        assert d.strategy == StrategyType.GRAPH_RAG

    def test_advantage_keyword_with_sufficient_words(self):
        d = self.router.route("Discuss the advantages and disadvantages of microservices architecture")
        assert d.strategy == StrategyType.AGENTIC

    def test_returns_route_decision(self):
        d = self.router.route("test query")
        assert isinstance(d, RouteDecision)
        assert isinstance(d.strategy, StrategyType)
        assert 0.0 <= d.confidence <= 1.0
        assert isinstance(d.reasoning, str)


# ── FallbackChain Tests ────────────────────────────────────────────────────


class TestFallbackChain:
    """Tests for the FallbackChain."""

    def test_default_fallback_chain(self):
        chain = default_fallback_chain()
        entries = chain.entries
        assert len(entries) == 3
        assert entries[0].strategy == StrategyType.GRAPH_RAG
        assert entries[1].strategy == StrategyType.ADVANCED
        assert entries[2].strategy == StrategyType.NAIVE

    def test_fallback_chain_add(self):
        chain = FallbackChain()
        chain.add(StrategyType.NAIVE)
        chain.add(StrategyType.ADVANCED)
        assert len(chain.entries) == 2

    def test_fallback_chain_skip_open_circuit(self):
        import time
        cb_open = CircuitBreaker("test", failure_threshold=1)
        # Force open and set last_failure_time to now (so recovery timeout hasn't elapsed)
        cb_open._state = CircuitState.OPEN
        cb_open._last_failure_time = time.monotonic()
        chain = FallbackChain([
            FallbackEntry(StrategyType.GRAPH_RAG, cb_open),
        ])
        # Should be skipped because circuit is open
        assert cb_open.is_open

    def test_factory_function(self):
        router, chain = create_router_and_fallback()
        assert isinstance(router, QueryRouter)
        assert isinstance(chain, FallbackChain)

    def test_get_stats(self):
        chain = default_fallback_chain()
        stats = chain.get_stats()
        assert len(stats) == 3
        assert all("name" in s for s in stats)
        assert all("state" in s for s in stats)
