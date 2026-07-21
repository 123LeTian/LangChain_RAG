"""
RAG Registry — Owner: C
Strategy registry for managing RAGStrategy instances keyed by RAGMode.

Supports:
  - Dynamic registration of pre-built RAGStrategy instances
  - Singleton management (one strategy per RAGMode)
  - Unregistered-mode exception handling
  - Backward compatibility with StrategyType via RAGMode mapping
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.models.rag import RAGMode

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────


class RegistryError(Exception):
    """Base exception for all registry errors."""


class StrategyNotRegisteredError(RegistryError):
    """Raised when requesting a strategy for an unregistered RAGMode."""

    def __init__(self, mode: RAGMode) -> None:
        self.mode = mode
        super().__init__(
            f"No strategy registered for mode '{mode.value}'. "
            f"Available modes: [no strategies registered]"
        )

    @classmethod
    def with_available(cls, mode: RAGMode, available: List[str]) -> StrategyNotRegisteredError:
        inst = cls.__new__(cls)
        inst.mode = mode
        inst.available = available
        msg = (
            f"No strategy registered for mode '{mode.value}'. "
            f"Available: {', '.join(available) if available else '(none)'}"
        )
        super(StrategyNotRegisteredError, inst).__init__(msg)
        return inst


class StrategyAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a mode that already has a strategy."""

    def __init__(self, mode: RAGMode) -> None:
        self.mode = mode
        super().__init__(
            f"Strategy already registered for mode '{mode.value}'. "
            f"Use unregister() first or call replace()."
        )


# ── Registry ──────────────────────────────────────────────────────────────


class RAGStrategyRegistry:
    """Central registry for RAGStrategy instances.

    Strategies are keyed by :class:`RAGMode`.  Each mode holds exactly ONE
    strategy instance (singleton per mode).  Instances are registered
    pre-built — the registry does NOT create or configure strategies.

    Usage::

        registry = RAGStrategyRegistry()

        # Register a pre-built strategy
        registry.register(RAGMode.NAIVE, naive_strategy)

        # Retrieve and execute
        strategy = registry.get(RAGMode.NAIVE)
        result = await strategy.run(request, context)

        # Replace an existing strategy
        registry.replace(RAGMode.NAIVE, new_strategy)

        # Check availability
        if registry.is_registered(RAGMode.GRAPH):
            ...
    """

    def __init__(self) -> None:
        self._strategies: Dict[RAGMode, Any] = {}  # RAGMode → RAGStrategy

    # ── Registration ──────────────────────────────────────────────────

    def register(self, mode: RAGMode, strategy: Any) -> None:
        """Register a strategy instance for the given mode.

        Args:
            mode:     The RAG paradigm this strategy implements.
            strategy: A concrete RAGStrategy instance (must have ``mode``
                      attribute matching *mode*).

        Raises:
            StrategyAlreadyRegisteredError: If *mode* already has a strategy.
                Use :meth:`replace` to overwrite.
            ValueError: If *strategy.mode* does not match *mode*.
        """
        if not isinstance(mode, RAGMode):
            raise TypeError(f"mode must be a RAGMode, got {type(mode).__name__}")

        if mode in self._strategies:
            raise StrategyAlreadyRegisteredError(mode)

        # Validate that the strategy's mode matches
        actual_mode = getattr(strategy, "mode", None)
        if actual_mode is not None and actual_mode != mode:
            raise ValueError(
                f"Strategy mode mismatch: registry key={mode.value}, "
                f"strategy.mode={actual_mode.value if isinstance(actual_mode, RAGMode) else actual_mode}"
            )

        self._strategies[mode] = strategy
        name = getattr(strategy, "strategy_name", strategy.__class__.__name__)
        logger.info("Registered strategy: %s -> %s", mode.value, name)

    def replace(self, mode: RAGMode, strategy: Any) -> None:
        """Replace the strategy for *mode*, overwriting any existing one.

        Unlike :meth:`register`, this does NOT raise if the mode is already
        registered — it silently replaces.
        """
        self._strategies[mode] = strategy
        name = getattr(strategy, "strategy_name", strategy.__class__.__name__)
        logger.info("Replaced strategy: %s -> %s", mode.value, name)

    def register_or_replace(self, mode: RAGMode, strategy: Any) -> None:
        """Register if not present, otherwise replace."""
        if mode in self._strategies:
            self.replace(mode, strategy)
        else:
            self.register(mode, strategy)

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, mode: RAGMode) -> Any:
        """Return the strategy instance for *mode*.

        Args:
            mode: The RAGMode to look up.

        Returns:
            The registered RAGStrategy instance.

        Raises:
            StrategyNotRegisteredError: If *mode* is not registered.
        """
        if not isinstance(mode, RAGMode):
            raise TypeError(f"mode must be a RAGMode, got {type(mode).__name__}")

        if mode not in self._strategies:
            raise StrategyNotRegisteredError.with_available(
                mode, self.list_modes_str()
            )
        return self._strategies[mode]

    def get_or_none(self, mode: RAGMode) -> Optional[Any]:
        """Return the strategy for *mode*, or None if not registered."""
        return self._strategies.get(mode)

    # ── Introspection ─────────────────────────────────────────────────

    def is_registered(self, mode: RAGMode) -> bool:
        """Check if a strategy is registered for *mode*."""
        return mode in self._strategies

    def list_modes(self) -> List[RAGMode]:
        """Return all registered RAGMode values."""
        return list(self._strategies.keys())

    def list_modes_str(self) -> List[str]:
        """Return registered mode names as strings (for error messages)."""
        return [m.value for m in self._strategies]

    @property
    def registered_count(self) -> int:
        """Number of registered strategies."""
        return len(self._strategies)

    # ── Management ────────────────────────────────────────────────────

    def unregister(self, mode: RAGMode) -> None:
        """Remove the strategy for *mode*.

        Raises:
            StrategyNotRegisteredError: If *mode* is not registered.
        """
        if mode not in self._strategies:
            raise StrategyNotRegisteredError.with_available(
                mode, self.list_modes_str()
            )
        del self._strategies[mode]
        logger.info("Unregistered strategy for mode: %s", mode.value)

    def clear(self) -> None:
        """Remove ALL registered strategies."""
        count = len(self._strategies)
        self._strategies.clear()
        logger.info("Cleared all %d registered strategies", count)

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, mode: RAGMode) -> bool:
        return self.is_registered(mode)


# ── Global singleton ─────────────────────────────────────────────────────

_default_registry: Optional[RAGStrategyRegistry] = None


def get_registry() -> RAGStrategyRegistry:
    """Return the global (process-level) RAGStrategyRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RAGStrategyRegistry()
    return _default_registry


def set_registry(registry: RAGStrategyRegistry) -> None:
    """Replace the global registry (e.g. for testing)."""
    global _default_registry
    _default_registry = registry


# ── Backward compatibility aliases ───────────────────────────────────────

# These aliases let existing code that imports StrategyRegistry,
# StrategyNotFoundError, StrategyAlreadyRegisteredError continue to work.
# They map to the new RAGMode-based equivalents.

# Old name → new name
StrategyRegistry = RAGStrategyRegistry  # type: ignore[misc]
StrategyNotFoundError = StrategyNotRegisteredError  # type: ignore[misc]
# StrategyAlreadyRegisteredError — same name, same behavior
