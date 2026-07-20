"""
RAG Registry — Owner: C
Strategy registry for dynamic lookup and plugin-style registration of RAG strategies.

Supports:
  - Manual registration via register()
  - Decorator-based registration
  - Auto-discovery of strategies in the strategies/ package
  - Lazy initialization (strategies are instantiated on first use)
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from src.models.rag import StrategyType
from src.rag.base import RAGStrategyBase

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────


class RegistryError(Exception):
    """Base exception for registry errors."""


class StrategyNotFoundError(RegistryError):
    """Raised when a requested strategy name is not registered."""

    def __init__(self, name: str, available: List[str]) -> None:
        super().__init__(
            f"Strategy '{name}' not found. Available: {', '.join(available)}"
        )
        self.name = name
        self.available = available


class StrategyAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a duplicate strategy name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Strategy '{name}' is already registered")
        self.name = name


# ── Entry ─────────────────────────────────────────────────────────────────


class _StrategyEntry:
    """Internal entry holding a strategy class and its (optionally lazy) instance."""

    __slots__ = ("strategy_type", "cls", "instance", "config", "metadata")

    def __init__(
        self,
        strategy_type: StrategyType,
        cls: Type[RAGStrategyBase],
        config: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy_type = strategy_type
        self.cls = cls
        self.instance: Optional[RAGStrategyBase] = None
        self.config = config
        self.metadata = metadata or {}

    def get_instance(self) -> RAGStrategyBase:
        """Return the singleton instance, creating it lazily if needed."""
        if self.instance is None:
            self.instance = self.cls(config=self.config)
        return self.instance


# ── Registry ──────────────────────────────────────────────────────────────


class StrategyRegistry:
    """Central registry for RAG strategies.

    Strategies are keyed by their StrategyType enum value.

    Usage:
        registry = StrategyRegistry()

        # Manual registration
        registry.register(StrategyType.NAIVE, NaiveRAG)

        # Decorator registration
        @registry.register(StrategyType.MODULAR)
        class MyModularRAG(RAGStrategyBase):
            ...

        # Lookup
        strategy = registry.get(StrategyType.NAIVE)
        response = await strategy.run(query)
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _StrategyEntry] = {}
        self._default_strategy: Optional[StrategyType] = None

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        strategy_type: StrategyType,
        cls: Optional[Type[RAGStrategyBase]] = None,
        *,
        config: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        set_default: bool = False,
    ) -> Callable[[Type[RAGStrategyBase]], Type[RAGStrategyBase]]:
        """Register a strategy class.

        Can be used as a direct call or as a decorator.

        Args:
            strategy_type: Enum key for this strategy.
            cls: The strategy class (when called directly).
            config: Optional configuration passed to the constructor.
            metadata: Arbitrary key-value metadata.
            set_default: If True, make this the default strategy.

        Returns:
            Decorator when called without cls, otherwise the class unchanged.

        Raises:
            StrategyAlreadyRegisteredError: If strategy_type is already registered.
            TypeError: If cls does not subclass RAGStrategyBase.
        """
        key = strategy_type.value

        def _register(c: Type[RAGStrategyBase]) -> Type[RAGStrategyBase]:
            if not issubclass(c, RAGStrategyBase):
                raise TypeError(
                    f"{c.__name__} must subclass RAGStrategyBase"
                )
            if key in self._entries:
                raise StrategyAlreadyRegisteredError(key)
            self._entries[key] = _StrategyEntry(
                strategy_type=strategy_type,
                cls=c,
                config=config,
                metadata=metadata,
            )
            if set_default:
                self._default_strategy = strategy_type
            logger.info("Registered strategy: %s → %s", key, c.__name__)
            return c

        if cls is not None:
            return _register(cls)
        return _register

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, strategy_type: StrategyType) -> RAGStrategyBase:
        """Return the singleton instance for the given strategy type.

        Args:
            strategy_type: Enum key.

        Returns:
            The strategy instance (created lazily on first access).

        Raises:
            StrategyNotFoundError: If the strategy type is not registered.
        """
        key = strategy_type.value
        entry = self._entries.get(key)
        if entry is None:
            raise StrategyNotFoundError(key, self.list_names())
        return entry.get_instance()

    def get_default(self) -> RAGStrategyBase:
        """Return the default strategy instance.

        If no default was explicitly set, returns the first registered strategy.

        Raises:
            StrategyNotFoundError: If no strategies are registered at all.
        """
        if self._default_strategy is not None:
            return self.get(self._default_strategy)
        if not self._entries:
            raise StrategyNotFoundError("<default>", [])
        first_key = next(iter(self._entries))
        return self._entries[first_key].get_instance()

    # ── Introspection ─────────────────────────────────────────────────

    def list_names(self) -> List[str]:
        """Return all registered strategy names."""
        return list(self._entries.keys())

    def list_types(self) -> List[StrategyType]:
        """Return all registered strategy types."""
        return [entry.strategy_type for entry in self._entries.values()]

    def is_registered(self, strategy_type: StrategyType) -> bool:
        """Check if a strategy type is registered."""
        return strategy_type.value in self._entries

    def get_metadata(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Return metadata for a registered strategy."""
        entry = self._entries.get(strategy_type.value)
        if entry is None:
            raise StrategyNotFoundError(strategy_type.value, self.list_names())
        return dict(entry.metadata)

    # ── Management ────────────────────────────────────────────────────

    def unregister(self, strategy_type: StrategyType) -> None:
        """Remove a strategy from the registry.

        Raises:
            StrategyNotFoundError: If not registered.
        """
        key = strategy_type.value
        if key not in self._entries:
            raise StrategyNotFoundError(key, self.list_names())
        del self._entries[key]
        if self._default_strategy == strategy_type:
            self._default_strategy = None
        logger.info("Unregistered strategy: %s", key)

    def set_default(self, strategy_type: StrategyType) -> None:
        """Set the default strategy (must already be registered)."""
        if strategy_type.value not in self._entries:
            raise StrategyNotFoundError(strategy_type.value, self.list_names())
        self._default_strategy = strategy_type

    def clear(self) -> None:
        """Remove all registered strategies."""
        self._entries.clear()
        self._default_strategy = None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, strategy_type: StrategyType) -> bool:
        return self.is_registered(strategy_type)

    # ── Auto-discovery ────────────────────────────────────────────────

    def discover_all(self, package: str = "src.rag.strategies") -> int:
        """Auto-discover and register all RAGStrategyBase subclasses in a package.

        Scans the given package for classes that subclass RAGStrategyBase
        and have a ``strategy_type`` attribute (the StrategyType enum value).

        Args:
            package: Dotted package path to scan.

        Returns:
            Number of newly discovered strategies registered.
        """
        try:
            module = importlib.import_module(package)
        except ImportError as exc:
            logger.warning("Cannot auto-discover strategies from %s: %s", package, exc)
            return 0

        registered_count = 0
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, RAGStrategyBase) or obj is RAGStrategyBase:
                continue
            strategy_type = getattr(obj, "strategy_type", None)
            if strategy_type is None:
                continue
            if not isinstance(strategy_type, StrategyType):
                continue
            if self.is_registered(strategy_type):
                continue
            try:
                self.register(strategy_type, obj)
                registered_count += 1
            except RegistryError as exc:
                logger.warning("Skipping %s: %s", name, exc)

        if registered_count:
            logger.info(
                "Auto-discovered %d strategy(s) from %s", registered_count, package
            )
        return registered_count


# ── Global singleton ─────────────────────────────────────────────────────


_default_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Return the global (process-level) strategy registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = StrategyRegistry()
    return _default_registry


def set_registry(registry: StrategyRegistry) -> None:
    """Replace the global registry (e.g. for testing)."""
    global _default_registry
    _default_registry = registry
