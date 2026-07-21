"""Exact-shape test doubles for C's not-yet-merged tool contracts."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class FakeToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tool_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tool_name": self.tool_name,
        }


@runtime_checkable
class FakeToolProtocol(Protocol):
    name: str
    description: str

    async def execute(self, **kwargs: Any) -> FakeToolResult: ...


class FakeToolRegistry:
    """Mirrors C's replacement-on-duplicate registry semantics."""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def list_names(self):
        return list(self._tools)

    def __contains__(self, name):
        return name in self._tools


class FakeToolExecutor:
    """Minimal executor with C's async call and result shape."""

    def __init__(self, registry):
        self.registry = registry

    async def execute(self, tool_name, **kwargs):
        result = await self.registry.get(tool_name).execute(**kwargs)
        result.tool_name = tool_name
        return result


__all__ = [
    "FakeToolExecutor",
    "FakeToolProtocol",
    "FakeToolRegistry",
    "FakeToolResult",
]
