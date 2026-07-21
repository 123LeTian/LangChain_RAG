"""
Agentic RAG Strategy — Owner: C
Agent-driven RAG with tool use, multi-step reasoning, and dynamic retrieval decisions.

This strategy implements the RAGStrategy (NEW) interface and delegates to the
finite-state AgentWorkflow.  Dependencies (retriever, LLM, tools) are read
from RAGContext — never created by the strategy.

Workflow:
  START -> Intent Detection -> Tool Selection -> Tool Execution -> Generate -> Verify -> END

Error degradation:
  - graph_search fails -> auto-fallback to vector_search
  - All tools fail -> structured error, LLM-only generation
  - Stack traces are NEVER sent to LLM
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.agent.state import AgentRunState
from src.agent.tools import (
    AnswerVerifyTool,
    DocumentSummaryTool,
    GraphSearchTool,
    ToolRegistry,
    VectorSearchTool,
    create_default_tools,
)
from src.agent.workflow import AgentWorkflow
from src.models.rag import (
    RAGCitation,
    RAGChunk,
    RAGContext,
    RAGMode,
    RAGRequest,
    RAGResult,
    RAGSource,
    TraceStage,
)
from src.rag.base import (
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategy,
    RetrieverProtocol,
)
from src.rag.registry import get_registry

logger = logging.getLogger(__name__)


class AgenticRAGStrategy(RAGStrategy):
    """Agent-driven RAG using the finite-state AgentWorkflow.

    Implements the NEW RAGStrategy interface — ``run(request, context) -> RAGResult``.
    Dependencies (retriever, LLM, tools, graph) are injected via *context*, not
    constructor arguments.

    The strategy:
      1. Builds tools from context (retriever -> VectorSearchTool, etc.)
      2. Creates an AgentRunState from the RAGRequest
      3. Runs the finite-state AgentWorkflow
      4. Converts the final AgentRunState into a RAGResult with citations

    Usage via RAGService::

        service = RAGService(retriever=..., llm=..., graph=...)
        result = await service.run(RAGRequest(query="What is RAG?", mode=RAGMode.AGENTIC))
    """

    mode: RAGMode = RAGMode.AGENTIC

    def __init__(self, max_steps: int = 4) -> None:
        self._max_steps = max_steps

    # ── RAGStrategy.run — THE entry point ───────────────────────────────

    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        """Execute the agentic RAG pipeline.

        Args:
            request: The user query + mode + options from the API layer.
            context:  Injected dependencies (retriever, LLM, tools, trace_recorder).

        Returns:
            RAGResult with answer, citations, hits, trace, and warnings.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # ── Build tools from context dependencies ───────────────────────
        tool_registry = self._build_tools(context)

        # ── Create workflow and state ───────────────────────────────────
        workflow = AgentWorkflow(max_steps=self._max_steps)
        state = AgentRunState(query=request.query)

        # Inject tools into context for the workflow
        context.tools = list(tool_registry._tools.values()) if hasattr(tool_registry, '_tools') else []

        # ── Record trace start ──────────────────────────────────────────
        trace_id = context.metadata.get("trace_id", f"trace_agentic_{id(request)}")

        # ── Execute the finite-state workflow ───────────────────────────
        try:
            state = await workflow.run(state, context)
        except Exception as exc:
            logger.exception("AgentWorkflow crashed")
            return RAGResult(
                answer="",
                warnings=[f"AgentWorkflow error: {exc}"],
            )

        # ── Build result from state ─────────────────────────────────────
        return self._build_result(state, context, trace_id, t0, warnings)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_tools(self, context: RAGContext) -> ToolRegistry:
        """Build a ToolRegistry from context dependencies.

        Follows the protocol: reads retriever/llm/graph from context,
        never creates its own instances.
        """
        retrieval: Optional[RetrieverProtocol] = getattr(context, "retriever", None)
        llm: Any = getattr(context, "llm", None)
        graph: Optional[GraphRetrieverProtocol] = getattr(context, "graph", None)

        registry = ToolRegistry()

        if retrieval is not None:
            registry.register(VectorSearchTool(retrieval))
        if graph is not None:
            registry.register(GraphSearchTool(graph))
        if llm is not None:
            registry.register(DocumentSummaryTool(llm))
        registry.register(AnswerVerifyTool())

        return registry

    def _build_result(
        self,
        state: AgentRunState,
        context: RAGContext,
        trace_id: str,
        t0: float,
        warnings: List[str],
    ) -> RAGResult:
        """Convert AgentRunState → RAGResult with citations and trace."""
        answer = state.final_answer or ""
        errors = list(state.errors)

        # ── Build citations from tool results ───────────────────────────
        citations: List[RAGCitation] = []
        hits: List[RAGChunk] = []

        for key in ("vector_search", "graph_search"):
            val = state.tool_results.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        chunk_id = item.get("chunk_id", "")
                        doc_id = item.get("source", item.get("document_id", ""))
                        content = item.get("content", "")
                        score = item.get("score")

                        if chunk_id and doc_id:
                            citations.append(RAGCitation(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                text_snippet=content[:200] if content else "",
                            ))

                        hits.append(RAGChunk(
                            chunk_id=chunk_id or f"chunk_{len(hits)}",
                            content=content,
                            source=RAGSource(
                                document_id=doc_id or "unknown",
                                score=score,
                            ),
                        ))
            elif isinstance(val, dict) and val.get("chunks"):
                for chunk in val["chunks"]:
                    if isinstance(chunk, dict):
                        chunk_id = chunk.get("chunk_id", "")
                        doc_id = chunk.get("source", chunk.get("document_id", ""))
                        content = chunk.get("content", "")
                        if chunk_id and doc_id:
                            citations.append(RAGCitation(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                text_snippet=content[:200] if content else "",
                            ))
                        hits.append(RAGChunk(
                            chunk_id=chunk_id or f"chunk_{len(hits)}",
                            content=content,
                            source=RAGSource(document_id=doc_id or "unknown"),
                        ))

        # Use state.citations if available (from _generate method)
        if not citations and state.citations:
            for c in state.citations:
                citations.append(RAGCitation(
                    chunk_id=c.get("chunk_id", ""),
                    document_id=c.get("document_id", ""),
                    text_snippet=c.get("text_snippet", "")[:200],
                ))
                hits.append(RAGChunk(
                    chunk_id=c.get("chunk_id", ""),
                    content=c.get("text_snippet", ""),
                    source=RAGSource(document_id=c.get("document_id", "")),
                ))

        # ── Record trace ────────────────────────────────────────────────
        recorder = context.trace_recorder
        if recorder is not None:
            try:
                self._record(context, TraceStage.INTENT,
                             request_summary(state.query),
                             f"tools={state.selected_tools}",
                             (time.perf_counter() - t0) * 1000)
            except Exception:
                pass

        # ── Collect warnings ────────────────────────────────────────────
        if state.status == "failed":
            warnings.extend(errors)
        if state.tool_results.get("__all_tools_failed__"):
            warnings.append("All tools failed — answer from LLM knowledge only")
        degradation_notes = [e for e in errors if "degrad" in e.lower()]
        warnings.extend(degradation_notes)

        return RAGResult(
            answer=answer,
            citations=citations,
            hits=hits,
            warnings=warnings,
            usage=_extract_usage(state),
        )


# ── Helpers ────────────────────────────────────────────────────────────────


def _extract_usage(state: AgentRunState) -> Dict[str, int]:
    """Extract approximate token usage from state metadata."""
    answer_len = len(state.final_answer or "")
    # Rough estimate: 1 token ≈ 4 chars
    completion_tokens = max(1, answer_len // 4)
    prompt_tokens = max(1, len(state.query) // 4 + len(state.selected_tools) * 50)
    return {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": prompt_tokens + completion_tokens,
    }


def request_summary(query: str) -> str:
    return f"query={query[:80]}"


# ── Auto-register ─────────────────────────────────────────────────────────


try:
    _reg = get_registry()
    if not _reg.is_registered(RAGMode.AGENTIC):
        _reg.register(RAGMode.AGENTIC, AgenticRAGStrategy())
except Exception:
    pass  # Registration will be handled by explicit setup
