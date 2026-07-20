"""
Agent Router — Owner: C
Tool selection for agentic RAG.

Two routers:
  1. AgentRouter — deterministic, keyword-based (CURRENT)
  2. QueryRouter (legacy) — heuristic strategy-level router

The AgentRouter classifies queries into tool sets using keyword matching:
  - Fact questions          → ["vector_search"]
  - Entity / relationship   → ["graph_search"]
  - Document summary        → ["document_summary"]
  - Complex / analytical    → ["vector_search", "graph_search"]

Also provides CircuitBreaker + FallbackChain for error degradation (unchanged).
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


# ═══════════════════════════════════════════════════════════════════════════
# AgentRouter — deterministic tool selection by query type
# ═══════════════════════════════════════════════════════════════════════════


class QueryCategory(str, Enum):
    """Categories used by AgentRouter to classify queries."""

    FACT = "fact"
    ENTITY = "entity"
    DOCUMENT_SUMMARY = "document_summary"
    COMPLEX = "complex"


@dataclass
class RouterResult:
    """Output of the AgentRouter."""

    category: QueryCategory
    tools: List[str]
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""


class AgentRouter:
    """Deterministic, keyword-based tool selector.

    Rules (checked in order):
      1. Document summary keywords → ["document_summary"]
      2. Entity / relationship keywords → ["graph_search"]
      3. Complex / comparison keywords → ["vector_search", "graph_search"]
      4. Fact keywords OR default → ["vector_search"]

    Usage::

        router = AgentRouter()
        result = router.route("What is the relationship between A and B?")
        assert result.tools == ["graph_search"]
    """

    # ── Keyword sets ──────────────────────────────────────────────────

    _FACT_KEYWORDS: frozenset = frozenset({
        "what is", "who is", "when was", "where is", "define",
        "how many", "how much", "which", "list",
    })

    _ENTITY_KEYWORDS: frozenset = frozenset({
        "relationship", "connection", "network", "entity", "community",
        "hierarchy", "related to", "linked to", "depend", "structure",
        "organization", "taxonomy", "ontology", "graph",
    })

    _SUMMARY_KEYWORDS: frozenset = frozenset({
        "summarize", "summary", "summarise", "overview", "abstract",
        "brief", "tldr", "recap", "outline", "condense",
    })

    _COMPLEX_KEYWORDS: frozenset = frozenset({
        "compare", "contrast", "analyze", "evaluate", "assess",
        "synthesize", "pros and cons", "advantages", "disadvantages",
        "difference between", "similarities", "versus", "vs",
        "step by step", "break down", "explain why", "explain how",
    })

    # ── Routing ───────────────────────────────────────────────────────

    def route(self, query: str) -> RouterResult:
        """Classify *query* and return the recommended tool set.

        Args:
            query: Raw user query string.

        Returns:
            RouterResult with category, tool list, and confidence.
        """
        q = query.lower().strip()

        # 1. Document summary
        summary_hits = sum(1 for kw in self._SUMMARY_KEYWORDS if kw in q)
        if summary_hits >= 1:
            return RouterResult(
                category=QueryCategory.DOCUMENT_SUMMARY,
                tools=["document_summary"],
                confidence=min(0.7 + summary_hits * 0.1, 0.95),
                reasoning=f"Matched {summary_hits} summary keyword(s)",
            )

        # 2. Entity / relationship → graph_search
        entity_hits = sum(1 for kw in self._ENTITY_KEYWORDS if kw in q)
        if entity_hits >= 1:
            return RouterResult(
                category=QueryCategory.ENTITY,
                tools=["graph_search"],
                confidence=min(0.6 + entity_hits * 0.15, 0.9),
                reasoning=f"Matched {entity_hits} entity/relationship keyword(s)",
            )

        # 3. Complex / analytical → vector_search + graph_search
        complex_hits = sum(1 for kw in self._COMPLEX_KEYWORDS if kw in q)
        if complex_hits >= 1:
            return RouterResult(
                category=QueryCategory.COMPLEX,
                tools=["vector_search", "graph_search"],
                confidence=min(0.65 + complex_hits * 0.1, 0.9),
                reasoning=f"Matched {complex_hits} complex/analytical keyword(s)",
            )

        # 4. Fact / default → vector_search
        fact_hits = sum(1 for kw in self._FACT_KEYWORDS if kw in q)
        return RouterResult(
            category=QueryCategory.FACT,
            tools=["vector_search"],
            confidence=0.55 + fact_hits * 0.1,
            reasoning=f"Default to vector_search (matched {fact_hits} fact keyword(s))",
        )


# ═══════════════════════════════════════════════════════════════════════════
# CircuitBreaker + FallbackChain (unchanged, kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Circuit breaker '{name}' is OPEN — rejecting call")
        self.name = name


class CircuitBreaker:
    """Circuit breaker for external service calls (unchanged)."""

    def __init__(self, name: str = "default", failure_threshold: int = 5,
                 recovery_timeout: float = 60.0, half_open_max_requests: int = 1) -> None:
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
        self.total_calls: int = 0
        self.total_failures: int = 0
        self.total_rejections: int = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info("Circuit '%s' OPEN → HALF_OPEN", self.name)
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name, "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
        }

    async def call(self, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        current_state = self.state
        self.total_calls += 1
        if current_state == CircuitState.OPEN:
            self.total_rejections += 1
            raise CircuitBreakerOpenError(self.name)
        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_requests >= self._half_open_max_requests:
                    self.total_rejections += 1
                    raise CircuitBreakerOpenError(self.name)
                self._half_open_requests += 1
        try:
            result = await coro_factory()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self.total_failures += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED and self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_requests = 0


@dataclass
class FallbackEntry:
    strategy: StrategyType
    circuit_breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="default"))


class FallbackChain:
    def __init__(self, entries: Optional[List[FallbackEntry]] = None) -> None:
        self._entries: List[FallbackEntry] = list(entries) if entries else []

    def add(self, strategy: StrategyType, circuit_breaker: Optional[CircuitBreaker] = None) -> FallbackChain:
        cb = circuit_breaker or CircuitBreaker(name=strategy.value)
        self._entries.append(FallbackEntry(strategy=strategy, circuit_breaker=cb))
        return self

    @property
    def entries(self) -> List[FallbackEntry]:
        return list(self._entries)

    async def execute(self, query: Any, create_strategy: Callable[[StrategyType], Any],
                      *, skip_strategies: Optional[set] = None) -> Any:
        skipped = skip_strategies or set()
        errors: List[str] = []
        for entry in self._entries:
            if entry.strategy in skipped:
                continue
            if entry.circuit_breaker.is_open:
                errors.append(f"{entry.strategy.value}: circuit open")
                continue
            try:
                strategy = create_strategy(entry.strategy)
                result = await entry.circuit_breaker.call(lambda s=strategy, q=query: s.run(q))
                if hasattr(result, "metadata"):
                    result.metadata["fallback_used"] = True
                    result.metadata["fallback_strategy"] = entry.strategy.value
                return result
            except CircuitBreakerOpenError:
                errors.append(f"{entry.strategy.value}: circuit open")
            except Exception as exc:
                errors.append(f"{entry.strategy.value}: {exc}")
        raise RuntimeError(f"All fallback strategies exhausted. Errors: {'; '.join(errors)}")

    def get_stats(self) -> List[Dict[str, Any]]:
        return [e.circuit_breaker.stats for e in self._entries]


def default_fallback_chain() -> FallbackChain:
    return (FallbackChain()
            .add(StrategyType.GRAPH_RAG, CircuitBreaker(name="graph_rag", failure_threshold=3, recovery_timeout=30.0))
            .add(StrategyType.ADVANCED, CircuitBreaker(name="advanced", failure_threshold=5, recovery_timeout=60.0))
            .add(StrategyType.NAIVE, CircuitBreaker(name="naive", failure_threshold=5, recovery_timeout=60.0)))


def create_router_and_fallback() -> tuple:
    return QueryRouter(), default_fallback_chain()


# ── Legacy QueryRouter (kept for backward compat) ─────────────────────────


@dataclass
class RouteDecision:
    strategy: StrategyType
    confidence: float
    reasoning: str = ""


class QueryRouter:
    """Legacy heuristic query router for strategy selection (unchanged)."""
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
        q = query_text.lower().strip()
        word_count = len(q.split())
        graph_hits = sum(1 for kw in self._GRAPH_KEYWORDS if kw in q)
        if graph_hits >= 1:
            return RouteDecision(strategy=StrategyType.GRAPH_RAG,
                                confidence=min(0.55 + graph_hits * 0.15, 0.9),
                                reasoning=f"Matched {graph_hits} graph keywords")
        agent_hits = sum(1 for kw in self._AGENT_KEYWORDS if kw in q)
        if agent_hits >= 1 and word_count > 5:
            return RouteDecision(strategy=StrategyType.AGENTIC,
                                confidence=min(0.6 + agent_hits * 0.15, 0.95),
                                reasoning=f"Matched {agent_hits} agentic keywords")
        simple_hits = sum(1 for kw in self._SIMPLE_INDICATORS if kw in q)
        if simple_hits >= 1 and word_count <= 8:
            return RouteDecision(strategy=StrategyType.NAIVE, confidence=0.7,
                                reasoning="Short factual query")
        return RouteDecision(strategy=StrategyType.ADVANCED, confidence=0.5,
                            reasoning="Default to advanced RAG")
