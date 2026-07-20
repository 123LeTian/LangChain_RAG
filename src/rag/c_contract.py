"""Lazy bridge to C's frozen RAG models without redefining them."""

from importlib import import_module
from typing import Any


class CContractUnavailableError(RuntimeError):
    """Raised when C's shared model commit is not present on this branch."""


class CContractFactory:
    """Instantiate C-owned models through their canonical import path."""

    def _models(self):
        module = import_module("src.models.rag")
        required = (
            "RAGChunk",
            "RAGCitation",
            "RAGMode",
            "RAGResult",
            "RAGSource",
            "TraceEvent",
            "TraceStage",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise CContractUnavailableError(
                "C's frozen RAG contract is not available on this branch; "
                f"missing: {', '.join(missing)}"
            )
        return module

    @property
    def naive_mode(self) -> Any:
        """Return C's canonical ``RAGMode.NAIVE`` value."""

        return self._models().RAGMode.NAIVE

    @property
    def advanced_mode(self) -> Any:
        """Return C's canonical ``RAGMode.ADVANCED`` value."""

        return self._models().RAGMode.ADVANCED

    def stage(self, name: str) -> Any:
        """Return a canonical C ``TraceStage`` member by enum name."""

        return getattr(self._models().TraceStage, name)

    def trace_event(self, **fields: Any) -> Any:
        return self._models().TraceEvent(**fields)

    def source(self, **fields: Any) -> Any:
        return self._models().RAGSource(**fields)

    def chunk(self, **fields: Any) -> Any:
        return self._models().RAGChunk(**fields)

    def citation(self, **fields: Any) -> Any:
        return self._models().RAGCitation(**fields)

    def result(self, **fields: Any) -> Any:
        return self._models().RAGResult(**fields)


def resolve_naive_mode() -> Any:
    """Resolve C's enum when present and a registry-compatible value otherwise."""

    try:
        return CContractFactory().naive_mode
    except CContractUnavailableError:
        return "naive"


def resolve_advanced_mode() -> Any:
    """Resolve C's advanced enum with a registry-compatible fallback."""

    try:
        return CContractFactory().advanced_mode
    except CContractUnavailableError:
        return "advanced"


__all__ = [
    "CContractFactory",
    "CContractUnavailableError",
    "resolve_advanced_mode",
    "resolve_naive_mode",
]
