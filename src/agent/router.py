"""
Agent Router — Owner: C
Query router that decides retrieval path (vector, graph, hybrid, direct)
and provides error degradation through Circuit Breaker + Fallback Chain patterns.

Components:
  - QueryRouter: Heuristic-based query → strategy routing.
  - CircuitBreaker: Protect external services from cascading failures.
  - FallbackChain: Ordered list of fallback strategies with circuit protection.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.models.rag import StrategyType

logger = logging.getLogger(__name__)


# ── Circuit Breaker ───────────────────────────────────────────────────────


class CircuitState(str, Enum):
    """Circuit breaker states (standard three-state model)."""

    CLOSED = "closed"         # Normal — calls pass through
    OPEN = "open"             # Failing — calls rejected immediately
    HALF_OPEN = "half_open"   # Testing — one probe call allowed


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Circuit breaker '{name}' is OPEN — rejecting call")
        self.name = name


class CircuitBreaker:
    """Circuit breaker for external service calls.

    Protects downstream services (vector DB, graph DB, LLM) from cascading
    failures. After `failure_threshold` consecutive failures, the circuit
    opens and rejects calls for `recovery_timeout` seconds. Then it moves
    to HALF_OPEN — one probe call is allowed; success closes the circuit,
    failure re-opens it.

    Thread-safe.

    Usage:
        cb = CircuitBreaker("graph_db", failure_threshold=3, recovery_timeout=30)
        try:
            result = await cb.call(lambda: graph.search("query"))
        except CircuitBreakerOpenError:
            # Circuit is open — use fallback
            ...
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_requests: int = 1,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_requests = half_open_max_requests

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_requests: int = 0
        self._lock = threading.Lock()

        # Statistics
        self.total_calls: int = 0
        self.total_failures: int = 0
        self.total_rejections: int = 0

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic transition from OPEN → HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info("Circuit '%s' transitioning OPEN → HALF_OPEN", self.name)
        return self._state

    @property
    def is_open(self) -> bool:
        """Shortcut: is the circuit currently rejecting calls?"""
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
        }

    # ── Core ───────────────────────────────────────────────────────────

    async def call(self, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        """Execute a call through the circuit breaker.

        Args:
            coro_factory: Async callable (factory) that returns the operation result.

        Returns:
            The result of the coro_factory.

        Raises:
            CircuitBreakerOpenError: If the circuit is OPEN.
            The original exception if the call fails in CLOSED or HALF_OPEN state.
        """
        current_state = self.state
        self.total_calls += 1

        # OPEN → reject immediately
        if current_state == CircuitState.OPEN:
            self.total_rejections += 1
            logger.warning("Circuit '%s' is OPEN — rejecting call", self.name)
            raise CircuitBreakerOpenError(self.name)

        # HALF_OPEN → allow limited probe requests
        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_requests >= self._half_open_max_requests:
                    self.total_rejections += 1
                    raise CircuitBreakerOpenError(self.name)
                self._half_open_requests += 1

        # Execute
        try:
            result = await coro_factory()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        """Called when a call succeeds."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit '%s' HALF_OPEN → CLOSED (probe succeeded)", self.name)
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Called when a call fails."""
        with self._lock:
            self._failure_count += 1
            self.total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' HALF_OPEN → OPEN (probe failed)", self.name
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' CLOSED → OPEN (%d consecutive failures)",
                    self.name,
                    self._failure_count,
                )

    def reset(self) -> None:
        """Force the circuit back to CLOSED (manual intervention)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_requests = 0
            logger.info("Circuit '%s' manually reset to CLOSED", self.name)


# ── Query Router ──────────────────────────────────────────────────────────


@dataclass
class RouteDecision:
    """Result of query routing analysis."""

    strategy: StrategyType
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""


class QueryRouter:
    """Heuristic query router: analyzes query text to suggest the best strategy.

    This is a lightweight, rule-based router. It can be replaced with an
    LLM-based classifier for more sophisticated routing.

    Routing rules (in priority order):
      1. Entity / relationship → GRAPH_RAG (most specific)
      2. Complex / analytical → AGENTIC
      3. Short factual → NAIVE
      4. Default → ADVANCED
    """

    # Keywords strongly suggesting each strategy
    _AGENT_KEYWORDS: frozenset = frozenset({
        "compare", "contrast", "analyze", "explain",
        "step by step", "break down", "evaluate", "assess", "synthesize",
        "what if", "pros and cons", "advantages", "disadvantages",
    })

    _GRAPH_KEYWORDS: frozenset = frozenset({
        "relationship", "connection", "network", "entity", "community",
        "hierarchy", "related to", "linked to", "depend", "structure",
        "organization", "taxonomy", "ontology",
    })

    _SIMPLE_INDICATORS: frozenset = frozenset({
        "what is", "who is", "when was", "where is", "define",
    })

    def route(self, query_text: str) -> RouteDecision:
        """Analyze a query and return the recommended strategy.

        Args:
            query_text: The raw user query string.

        Returns:
            RouteDecision with recommended strategy and confidence.
        """
        q = query_text.lower().strip()
        word_count = len(q.split())

        # 1. Entity / relationship → Graph RAG (checked first — most specific)
        graph_hits = sum(1 for kw in self._GRAPH_KEYWORDS if kw in q)
        if graph_hits >= 1:
            confidence = min(0.55 + graph_hits * 0.15, 0.9)
            return RouteDecision(
                strategy=StrategyType.GRAPH_RAG,
                confidence=confidence,
                reasoning=f"Entity/relationship query matched {graph_hits} graph keywords",
            )

        # 2. Multi-step / analytical → Agentic
        agent_hits = sum(1 for kw in self._AGENT_KEYWORDS if kw in q)
        if agent_hits >= 1 and word_count > 5:
            confidence = min(0.6 + agent_hits * 0.15, 0.95)
            return RouteDecision(
                strategy=StrategyType.AGENTIC,
                confidence=confidence,
                reasoning=f"Complex/analytical query matched {agent_hits} agentic keywords",
            )

        # 3. Short factual → Naive
        simple_hits = sum(1 for kw in self._SIMPLE_INDICATORS if kw in q)
        if simple_hits >= 1 and word_count <= 8:
            return RouteDecision(
                strategy=StrategyType.NAIVE,
                confidence=0.7,
                reasoning="Short factual query — naive retrieval sufficient",
            )

        # 4. Default → Advanced
        return RouteDecision(
            strategy=StrategyType.ADVANCED,
            confidence=0.5,
            reasoning="Default to advanced RAG",
        )


# ── Fallback Chain ────────────────────────────────────────────────────────


@dataclass
class FallbackEntry:
    """One link in the fallback chain."""

    strategy: StrategyType
    circuit_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(name="default")
    )


class FallbackChain:
    """Ordered chain of fallback strategies with per-strategy circuit breakers.

    When the primary strategy fails, the FallbackChain iterates through
    alternative strategies in priority order. Each strategy has its own
    CircuitBreaker — if a strategy's circuit is open, it's skipped entirely.

    Usage:
        chain = FallbackChain([
            FallbackEntry(StrategyType.GRAPH_RAG, CircuitBreaker("graph")),
            FallbackEntry(StrategyType.ADVANCED, CircuitBreaker("advanced")),
            FallbackEntry(StrategyType.NAIVE, CircuitBreaker("naive")),
        ])

        response = await chain.execute(
            query,
            create_strategy=lambda st: registry.get(st),
        )
    """

    def __init__(
        self,
        entries: Optional[List[FallbackEntry]] = None,
    ) -> None:
        self._entries: List[FallbackEntry] = list(entries) if entries else []

    def add(
        self,
        strategy: StrategyType,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> FallbackChain:
        """Append a fallback entry. Returns self for chaining."""
        cb = circuit_breaker or CircuitBreaker(name=strategy.value)
        self._entries.append(FallbackEntry(strategy=strategy, circuit_breaker=cb))
        return self

    @property
    def entries(self) -> List[FallbackEntry]:
        return list(self._entries)

    async def execute(
        self,
        query: Any,  # RAGQuery
        create_strategy: Callable[[StrategyType], Any],  # → RAGStrategyBase
        *,
        skip_strategies: Optional[set] = None,
    ) -> Any:  # RAGResponse
        """Execute the fallback chain.

        Iterates through fallback strategies. For each:
          1. Skips strategies already tried (in skip_strategies).
          2. Checks the circuit breaker — skips if OPEN.
          3. Creates the strategy and calls its run() method via the breaker.

        Args:
            query: The RAGQuery to execute.
            create_strategy: Factory: StrategyType → RAGStrategyBase instance.
            skip_strategies: Strategy types to skip (the already-failed ones).

        Returns:
            RAGResponse from the first successful fallback.

        Raises:
            RuntimeError: If ALL fallback strategies are exhausted.
        """
        skipped = skip_strategies or set()
        errors: List[str] = []

        for entry in self._entries:
            if entry.strategy in skipped:
                continue

            # Check circuit breaker
            if entry.circuit_breaker.is_open:
                logger.info(
                    "Skipping fallback '%s' — circuit is OPEN", entry.strategy.value
                )
                errors.append(f"{entry.strategy.value}: circuit open")
                continue

            try:
                strategy = create_strategy(entry.strategy)
                result = await entry.circuit_breaker.call(
                    lambda s=strategy, q=query: s.run(q)
                )
                # Success — annotate result and return
                if hasattr(result, "metadata"):
                    result.metadata["fallback_used"] = True
                    result.metadata["fallback_strategy"] = entry.strategy.value
                if errors and hasattr(result, "error"):
                    result.error = (
                        result.error or ""
                        + f" (degraded via {entry.strategy.value}; "
                        + f"skipped: {'; '.join(errors)})"
                    )
                return result

            except CircuitBreakerOpenError:
                errors.append(f"{entry.strategy.value}: circuit open")
                continue
            except Exception as exc:
                logger.warning("Fallback '%s' failed: %s", entry.strategy.value, exc)
                errors.append(f"{entry.strategy.value}: {exc}")
                continue

        raise RuntimeError(
            f"All fallback strategies exhausted. Errors: {'; '.join(errors)}"
        )

    def get_stats(self) -> List[Dict[str, Any]]:
        """Return circuit breaker stats for each entry."""
        return [e.circuit_breaker.stats for e in self._entries]


# ── Default configurations ────────────────────────────────────────────────


def default_fallback_chain() -> FallbackChain:
    """Create the standard fallback chain: Graph → Advanced → Naive.

    Returns:
        Configured FallbackChain ready for use.
    """
    return (
        FallbackChain()
        .add(
            StrategyType.GRAPH_RAG,
            CircuitBreaker(name="graph_rag", failure_threshold=3, recovery_timeout=30.0),
        )
        .add(
            StrategyType.ADVANCED,
            CircuitBreaker(name="advanced", failure_threshold=5, recovery_timeout=60.0),
        )
        .add(
            StrategyType.NAIVE,
            CircuitBreaker(name="naive", failure_threshold=5, recovery_timeout=60.0),
        )
    )


def create_router_and_fallback() -> tuple[QueryRouter, FallbackChain]:
    """Factory: create a QueryRouter and FallbackChain pair.

    Returns:
        (QueryRouter, FallbackChain) tuple.
    """
    return QueryRouter(), default_fallback_chain()
