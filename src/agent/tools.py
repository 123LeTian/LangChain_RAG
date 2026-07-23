"""
Agent Tools — Owner: C
Unified tool protocol interfaces for agentic RAG.

Provides:
  - ToolProtocol: what every tool must implement
  - ToolResult: uniform execution result
  - Four tool interfaces: vector_search, graph_search, document_summary, answer_verify
  - ToolRegistry + ToolExecutor: lookup and execution with timeout/retry

IMPORTANT — C only defines INTERFACES here.  Concrete implementations come from:
  - Vector Tool  → Owner B (src/retrieval/)
  - Graph Tool   → Owner D (src/graph/)
  - Document Summary Tool → Owner B
  - Answer Verify Tool     → Owner C (can be provided here or by others)

All tools are accessed via ToolRegistry by name; strategies never import
concrete implementations directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import operator
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

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


# ── Tool Protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class ToolProtocol(Protocol):
    """Unified protocol that every agent tool MUST satisfy.

    Each tool has:
      - a unique ``name`` (used as registry key)
      - a ``description`` (shown to the LLM / router)
      - an ``execute(**kwargs) -> ToolResult`` method (async, never raises)
    """

    name: str
    description: str

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.  MUST return ToolResult (errors captured, not raised)."""
        ...


# ── Abstract Base Tool ────────────────────────────────────────────────────


class BaseTool(ABC):
    """Convenience base class implementing ToolProtocol.

    Subclasses must provide ``name``, ``description``, and ``execute()``.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
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
        return {"type": "object", "properties": {}, "required": []}


# ═══════════════════════════════════════════════════════════════════════════
# Four Tool Interfaces (Protocols — implementations from B / D / C)
# ═══════════════════════════════════════════════════════════════════════════


# ── 1. vector_search (Owner: B) ──────────────────────────────────────────


class VectorSearchTool(BaseTool):
    """Search the vector database for semantically relevant documents.

    Concrete implementation provided by Owner B (src/retrieval/).
    This class wraps a ``RetrieverProtocol`` instance injected at construction.
    """

    name: str = "vector_search"
    description: str = (
        "Search the document knowledge base using semantic similarity. "
        "Best for factual lookups, definitions, and finding relevant passages."
    )

    def __init__(self, retrieval: Optional[RetrieverProtocol] = None) -> None:
        self._retrieval = retrieval

    async def execute(self, query: str = "", top_k: int = 5, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        if self._retrieval is None:
            return ToolResult(success=False, error="No retriever configured", tool_name=self.name)
        try:
            context = self._retrieval.retrieve(query=query, top_k=top_k, **kwargs)
            data = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "source": c.source.document_id,
                    "score": c.source.score,
                    "filename": c.source.source_path,
                    "page": c.source.page,
                    "metadata": c.source.metadata,
                }
                for c in context.chunks
            ]
            return ToolResult(success=True, data=data,
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)
        except Exception as exc:
            logger.error("vector_search failed: %s", exc)
            return ToolResult(success=False, error=str(exc),
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
            },
            "required": ["query"],
        }


# ── 2. graph_search (Owner: D) ───────────────────────────────────────────


class GraphSearchTool(BaseTool):
    """Search the knowledge graph for entities and relationships.

    Concrete implementation provided by Owner D (src/graph/).
    This class wraps a ``GraphRetrieverProtocol`` instance injected at construction.
    """

    name: str = "graph_search"
    description: str = (
        "Search the knowledge graph for entities, relationships, and community summaries. "
        "Best for questions about connections between entities, hierarchies, or network structure."
    )

    def __init__(self, graph: Optional[GraphRetrieverProtocol] = None) -> None:
        self._graph = graph

    async def execute(self, query: str = "", top_k: int = 5, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        if self._graph is None:
            return ToolResult(success=False, error="No graph retriever configured", tool_name=self.name)
        try:
            context = self._graph.graph_search(query=query, top_k=top_k, **kwargs)
            data = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "source": c.source.document_id,
                        "score": c.source.score,
                        "filename": c.source.source_path,
                        "page": c.source.page,
                        "metadata": c.source.metadata,
                    }
                    for c in context.chunks
                ],
            }
            return ToolResult(success=True, data=data,
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)
        except Exception as exc:
            logger.error("graph_search failed: %s", exc)
            return ToolResult(success=False, error=str(exc),
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The entity or relationship query."},
                "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
            },
            "required": ["query"],
        }


# ── 3. document_summary (Owner: B) ───────────────────────────────────────


class DocumentSummaryTool(BaseTool):
    """Generate a summary of a specific document or set of chunks.

    Concrete implementation provided by Owner B (src/retrieval/ or src/ingestion/).
    This tool takes retrieved chunks and produces a condensed summary.
    """

    name: str = "document_summary"
    description: str = (
        "Summarize the content of one or more documents. "
        "Best when the user asks for a summary, overview, or abstract of documents."
    )

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    async def execute(self, chunks: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        if not chunks:
            return ToolResult(success=False, error="No chunks provided to summarise", tool_name=self.name)
        if self._llm is None:
            # No LLM: return concatenated text as "summary"
            combined = "\n\n".join(c.get("content", "") for c in chunks)
            return ToolResult(success=True, data={"summary": combined[:2000]},
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)
        try:
            text = "\n\n".join(c.get("content", "") for c in chunks)
            summary = await self._llm.generate(
                prompt="Summarize the following documents concisely.",
                context=text,
            )
            return ToolResult(success=True, data={"summary": summary},
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc),
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chunks": {"type": "array", "description": "List of chunks to summarise."},
            },
            "required": ["chunks"],
        }


# ── 4. answer_verify (Owner: C) ─────────────────────────────────────────


class AnswerVerifyTool(BaseTool):
    """Verify the factual accuracy and completeness of a generated answer.

    Owner: C.  Checks the answer against the provided context chunks.
    In production this may use an NLI model or a second LLM pass.
    """

    name: str = "answer_verify"
    description: str = (
        "Verify whether an answer is factually supported by the provided context. "
        "Returns a verification result with issues found."
    )

    async def execute(self, answer: str = "", chunks: Optional[List[Dict[str, Any]]] = None,
                      **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        issues: List[str] = []

        if not answer or len(answer.strip()) < 10:
            issues.append("Answer is too short or empty")
        if not chunks:
            issues.append("No context chunks provided for verification")

        # Simple heuristic checks (production would use NLI / LLM judge)
        if chunks and answer:
            context_words = set()
            for c in chunks:
                context_words.update(c.get("content", "").lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(answer_words & context_words)
            if overlap < 3 and len(answer_words) > 10:
                issues.append("Low word overlap between answer and context")

        return ToolResult(
            success=len(issues) == 0,
            data={"passed": len(issues) == 0, "issues": issues, "answer_length": len(answer)},
            duration_ms=(time.perf_counter() - t0) * 1000,
            tool_name=self.name,
        )

    def _parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The generated answer to verify."},
                "chunks": {"type": "array", "description": "Context chunks used for generation."},
            },
            "required": ["answer"],
        }


# ── Tool Registry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for agent tools keyed by name.

    Tools are registered as instances (not classes).  Lookup is by string name.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {self.list_names()}")
        return self._tools[name]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        return [t.to_openai_function() for t in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
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
    """Executes tools with timeout, retry, and uniform error handling."""

    def __init__(self, registry: ToolRegistry, default_timeout: float = 30.0,
                 max_retries: int = 1, retry_delay: float = 0.5) -> None:
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self._registry.get(tool_name)
        last_result: Optional[ToolResult] = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await asyncio.wait_for(tool.execute(**kwargs), timeout=self._default_timeout)
                result.tool_name = tool_name
                return result
            except asyncio.TimeoutError:
                last_result = ToolResult(success=False, error=f"Tool '{tool_name}' timed out", tool_name=tool_name)
            except Exception as exc:
                last_result = ToolResult(success=False, error=f"Tool '{tool_name}': {exc}", tool_name=tool_name)
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay)

        return last_result or ToolResult(success=False, error=f"Tool '{tool_name}' failed", tool_name=tool_name)

    async def execute_batch(self, calls: List[tuple]) -> List[ToolResult]:
        tasks = [self.execute(name, **kwargs) for name, kwargs in calls]
        return await asyncio.gather(*tasks)


# ── Legacy tools (kept for backward compatibility) ────────────────────────


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely (legacy)."""
    name: str = "calculator"
    description: str = "Evaluate a mathematical expression."
    _SAFE_NAMESPACE: Dict[str, Any] = {"abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "add": operator.add, "sub": operator.sub,
        "mul": operator.mul, "truediv": operator.truediv,
        "floordiv": operator.floordiv, "mod": operator.mod}

    async def execute(self, expression: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        try:
            result = eval(expression, {"__builtins__": {}}, dict(self._SAFE_NAMESPACE))
            return ToolResult(success=True, data={"expression": expression, "result": result},
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)
        except Exception as exc:
            return ToolResult(success=False, error=f"Invalid: {exc}",
                            duration_ms=(time.perf_counter() - t0) * 1000, tool_name=self.name)

    def _parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression."}}, "required": ["expression"]}


class DirectAnswerTool(BaseTool):
    """Answer directly from LLM knowledge (legacy)."""
    name: str = "direct_answer"
    description: str = "Answer without document retrieval."

    async def execute(self, question: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"question": question, "action": "generate_directly"}, tool_name=self.name)

    def _parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"question": {"type": "string", "description": "Question."}}, "required": ["question"]}


# ── Factory ───────────────────────────────────────────────────────────────


def create_default_tools(
    retrieval: Optional[RetrieverProtocol] = None,
    graph: Optional[GraphRetrieverProtocol] = None,
    llm: Any = None,
) -> ToolRegistry:
    """Create a ToolRegistry pre-populated with the four standard tools."""
    registry = ToolRegistry()
    registry.register(VectorSearchTool(retrieval))
    registry.register(GraphSearchTool(graph))
    registry.register(DocumentSummaryTool(llm))
    registry.register(AnswerVerifyTool())
    return registry
