"""
Agent Tools — Owner: C
Tool definitions for agentic RAG.

Provides:
  - BaseTool: abstract base for all agent tools.
  - Concrete tools: VectorSearchTool, GraphSearchTool, CalculatorTool.
  - ToolRegistry: central registry for tool lookup.
  - ToolExecutor: executes tools with timeout, error handling, and retry.

All tools return a ToolResult for uniform error handling.
"""

from __future__ import annotations

import asyncio
import logging
import math
import operator
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.models.rag import RAGChunk, RAGContext, RAGSource
from src.rag.base import GraphRetrieverProtocol, RetrieverProtocol

logger = logging.getLogger(__name__)


# ── Tool Result ───────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Uniform result from any tool execution."""

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


# ── Base Tool ─────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """Abstract base class for agent tools.

    Each tool has a name, description, and parameter schema (for LLM function calling).
    Subclasses implement execute().
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        Returns ToolResult (never raises — errors are captured in the result).
        """
        ...

    def to_openai_function(self) -> Dict[str, Any]:
        """Serialize to OpenAI-compatible function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameter_schema(),
            },
        }

    def _parameter_schema(self) -> Dict[str, Any]:
        """Override to provide a JSON Schema for the tool's parameters."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }


# ── Concrete Tools ────────────────────────────────────────────────────────


class VectorSearchTool(BaseTool):
    """Search the vector database for semantically relevant documents."""

    name: str = "vector_search"
    description: str = (
        "Search the document knowledge base using semantic similarity. "
        "Best for factual lookups, definitions, and finding relevant passages."
    )

    def __init__(self, retrieval: RetrieverProtocol) -> None:
        self._retrieval = retrieval

    async def execute(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> ToolResult:
        t0 = time.perf_counter()
        try:
            context = self._retrieval.retrieve(query=query, top_k=top_k, **kwargs)
            data = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "source": c.source.document_id,
                    "score": c.source.score,
                }
                for c in context.chunks
            ]
            return ToolResult(
                success=True,
                data=data,
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )
        except Exception as exc:
            logger.error("VectorSearchTool failed: %s", exc)
            return ToolResult(
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }


class GraphSearchTool(BaseTool):
    """Search the knowledge graph for entities and relationships."""

    name: str = "graph_search"
    description: str = (
        "Search the knowledge graph for entities, relationships, and community summaries. "
        "Best for questions about connections between entities, hierarchies, or network structure."
    )

    def __init__(self, graph: GraphRetrieverProtocol) -> None:
        self._graph = graph

    async def execute(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> ToolResult:
        t0 = time.perf_counter()
        try:
            context = self._graph.graph_search(query=query, top_k=top_k, **kwargs)
            data = {
                "chunks": [
                    {"chunk_id": c.chunk_id, "content": c.content}
                    for c in context.chunks
                ],
            }
            return ToolResult(
                success=True,
                data=data,
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )
        except Exception as exc:
            logger.error("GraphSearchTool failed: %s", exc)
            return ToolResult(
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The entity or relationship query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        }


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression. "
        "Supports basic arithmetic (+, -, *, /, **), and functions: abs, round, min, max, sqrt."
    )

    # Safe namespace for eval
    _SAFE_NAMESPACE: Dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
    }

    async def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        try:
            # Restricted eval — no builtins, no attribute access
            result = eval(
                expression,
                {"__builtins__": {}},
                dict(self._SAFE_NAMESPACE),
            )
            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Invalid expression: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.name,
            )

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'.",
                },
            },
            "required": ["expression"],
        }


class DirectAnswerTool(BaseTool):
    """Answer directly from the LLM's own knowledge (no retrieval)."""

    name: str = "direct_answer"
    description: str = (
        "Answer a question using the LLM's built-in knowledge without document retrieval. "
        "Best for general knowledge questions that don't require specific documents."
    )

    async def execute(self, question: str, **kwargs: Any) -> ToolResult:
        # This tool is special: it delegates to the LLM generator directly.
        # The actual generation is handled by the workflow, not here.
        # We return a marker that tells the workflow to generate.
        return ToolResult(
            success=True,
            data={"question": question, "action": "generate_directly"},
            tool_name=self.name,
        )

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer from built-in knowledge.",
                },
            },
            "required": ["question"],
        }


# ── Tool Registry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for agent tools.

    Tools are registered by name and can be looked up for execution or
    serialized as OpenAI/Anthropic-compatible function definitions.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {self.list_names()}")
        return self._tools[name]

    def list_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, str]]:
        """Return lightweight tool metadata for display."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Serialize all tools to OpenAI function-calling format."""
        return [t.to_openai_function() for t in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Serialize all tools to Anthropic tool-use format."""
        tools = []
        for tool in self._tools.values():
            func_def = tool.to_openai_function()["function"]
            tools.append({
                "name": func_def["name"],
                "description": func_def["description"],
                "input_schema": func_def["parameters"],
            })
        return tools

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── Tool Executor ─────────────────────────────────────────────────────────


class ToolExecutor:
    """Executes tools with timeout, retry, and uniform error handling.

    Usage:
        registry = ToolRegistry()
        registry.register(VectorSearchTool(retriever))
        executor = ToolExecutor(registry, default_timeout=30.0)
        result = await executor.execute("vector_search", query="What is RAG?", top_k=5)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_retries: int = 1,
        retry_delay: float = 0.5,
    ) -> None:
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with timeout and retry.

        Args:
            tool_name: Registered tool name.
            **kwargs: Arguments to pass to the tool's execute() method.

        Returns:
            ToolResult (always — errors are captured, not raised).
        """
        tool = self._registry.get(tool_name)  # May raise KeyError

        last_result: Optional[ToolResult] = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs),
                    timeout=self._default_timeout,
                )
                result.tool_name = tool_name
                return result
            except asyncio.TimeoutError:
                last_result = ToolResult(
                    success=False,
                    error=f"Tool '{tool_name}' timed out after {self._default_timeout}s",
                    tool_name=tool_name,
                )
            except Exception as exc:
                last_result = ToolResult(
                    success=False,
                    error=f"Tool '{tool_name}' error: {exc}",
                    tool_name=tool_name,
                )

            if attempt < self._max_retries:
                logger.warning(
                    "Tool '%s' attempt %d/%d failed, retrying in %.1fs",
                    tool_name,
                    attempt + 1,
                    self._max_retries + 1,
                    self._retry_delay,
                )
                await asyncio.sleep(self._retry_delay)

        return last_result or ToolResult(
            success=False,
            error=f"Tool '{tool_name}' failed after {self._max_retries + 1} attempts",
            tool_name=tool_name,
        )

    async def execute_batch(
        self, calls: List[tuple[str, Dict[str, Any]]]
    ) -> List[ToolResult]:
        """Execute multiple tool calls concurrently.

        Args:
            calls: List of (tool_name, kwargs) tuples.

        Returns:
            List of ToolResult, one per call, in the same order.
        """
        tasks = [self.execute(name, **kwargs) for name, kwargs in calls]
        return await asyncio.gather(*tasks)


# ── Default tool set factory ──────────────────────────────────────────────


def create_default_tools(
    retrieval: RetrieverProtocol,
    graph: Optional[GraphRetrieverProtocol] = None,
) -> ToolRegistry:
    """Create a ToolRegistry pre-populated with the standard tool set.

    Args:
        retrieval: The retrieval protocol implementation (from Owner: B).
        graph: Optional graph retriever (from Owner: D).

    Returns:
        Configured ToolRegistry with all default tools.
    """
    registry = ToolRegistry()
    registry.register(VectorSearchTool(retrieval))
    registry.register(CalculatorTool())
    registry.register(DirectAnswerTool())
    if graph is not None:
        registry.register(GraphSearchTool(graph))
    return registry
