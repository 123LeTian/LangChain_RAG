"""
Agent Workflow — Owner: C
Finite-state agent for agentic RAG.

Workflow:
  START → Intent Detection → Tool Selection → Tool Execution → Generate → Verify → END

Key constraints:
  - max_steps = 4 (default) — prevents infinite loops
  - Each step increments current_step; when max_steps is reached, workflow terminates
  - Tools are selected deterministically by AgentRouter
  - Tools are executed via ToolExecutor (from context)

Also preserves legacy LangGraph-based workflow for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph

from src.agent.state import (
    AgentAction,
    AgentRunState,
    AgentState,
    AgentStatus,
    AgentStep,
    is_finished,
    set_answer,
    set_error,
)
from src.agent.router import AgentRouter, RouterResult
from src.agent.tools import ToolExecutor, ToolRegistry
from src.models.rag import RAGContext, RAGQuery, RAGResponse, RAGSource, StrategyType
from src.rag.base import GeneratorProtocol, GraphRetrieverProtocol, RetrieverProtocol

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# AgentWorkflow — finite-state agent (CURRENT)
# ═══════════════════════════════════════════════════════════════════════════


class AgentWorkflow:
    """Finite-state agent that executes the following steps:

    START → Intent Detection → Tool Selection → Tool Execution → Generate → Verify → END

    The workflow is bounded by ``max_steps`` (default 4).  Each tool execution
    counts as one step.  When the step budget is exhausted the agent forces
    generate → verify → end.

    Usage::

        workflow = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is RAG?")
        state = await workflow.run(state, context)
        print(state.final_answer)
    """

    def __init__(self, max_steps: int = 4) -> None:
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2 (generate + verify)")
        self.max_steps = max_steps
        self._router = AgentRouter()

    # ── Public API ─────────────────────────────────────────────────────

    async def run(self, state: AgentRunState, context: RAGContext) -> AgentRunState:
        """Execute the full agent workflow.

        Args:
            state:   Initial or resumed AgentRunState.
            context: RAGContext with injected dependencies (llm, tools, retriever).

        Returns:
            The final AgentRunState with answer and status.
        """
        state.max_steps = self.max_steps

        try:
            # Step 1: Intent Detection
            state = await self._intent_detection(state)

            # Step 2: Tool Selection
            state = await self._tool_selection(state)

            # Step 3: Tool Execution (counts as 1+ steps)
            state = await self._tool_execution(state, context)

            # Step 4: Generate
            state = await self._generate(state, context)

            # Step 5: Verify
            state = await self._verify(state, context)

        except Exception as exc:
            logger.exception("Agent workflow failed")
            state.set_failed(str(exc))

        # Ensure terminal state
        if not state.is_finished:
            if state.final_answer:
                state.status = "completed"
            else:
                state.set_failed("Workflow ended without answer")

        return state

    # ── Step implementations ───────────────────────────────────────────

    async def _intent_detection(self, state: AgentRunState) -> AgentRunState:
        """Classify the query intent and record it."""
        state.record_step("intent_detection")
        result = self._router.route(state.query)
        # Store intent metadata (available to downstream steps)
        state.tool_results["__intent__"] = {
            "category": result.category.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
        return state

    async def _tool_selection(self, state: AgentRunState) -> AgentRunState:
        """Select tools based on the classified intent."""
        if state.is_max_steps_exceeded:
            return state

        state.record_step("tool_selection")
        result = self._router.route(state.query)
        state.selected_tools = list(result.tools)
        return state

    async def _tool_execution(self, state: AgentRunState, context: RAGContext) -> AgentRunState:
        """Execute each selected tool in order, one step per tool.

        Stops early if max_steps is reached.
        """
        executor = _get_executor(context)
        if executor is None:
            state.errors.append("No ToolExecutor available in context")
            return state

        for tool_name in state.selected_tools:
            if state.is_max_steps_exceeded:
                state.errors.append(f"Step budget exhausted before executing '{tool_name}'")
                break

            state.record_step(f"tool_execution:{tool_name}")

            # Build tool arguments from query + context
            kwargs: Dict[str, Any] = {"query": state.query}
            if tool_name == "document_summary" and state.tool_results.get("vector_search"):
                kwargs["chunks"] = state.tool_results["vector_search"]
            if tool_name == "answer_verify":
                kwargs["answer"] = state.final_answer or ""
                all_chunks = []
                for key in ("vector_search", "graph_search"):
                    val = state.tool_results.get(key)
                    if isinstance(val, list):
                        all_chunks.extend(val)
                kwargs["chunks"] = all_chunks or None

            try:
                result = await executor.execute(tool_name, **kwargs)
                state.tool_results[tool_name] = result.data if result.success else None
                if not result.success:
                    state.errors.append(f"Tool '{tool_name}' failed: {result.error}")
            except Exception as exc:
                state.errors.append(f"Tool '{tool_name}' exception: {exc}")
                state.tool_results[tool_name] = None

        return state

    async def _generate(self, state: AgentRunState, context: RAGContext) -> AgentRunState:
        """Generate the final answer from tool results."""
        if state.is_max_steps_exceeded:
            state.set_failed("Step budget exhausted before generation")
            return state

        state.record_step("generate")

        # Collect context from tool results
        context_parts: List[str] = []
        for key in ("vector_search", "graph_search", "document_summary"):
            val = state.tool_results.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        context_parts.append(item.get("content", "") or item.get("summary", ""))
            elif isinstance(val, dict):
                context_parts.append(val.get("summary", "") or str(val))
            elif isinstance(val, str):
                context_parts.append(val)

        combined_context = "\n\n".join(context_parts) if context_parts else "No context available."

        # Generate via LLM or fallback
        if context.llm is not None:
            try:
                answer = await context.llm.generate(
                    prompt="You are a helpful assistant. Answer the query based on the provided context.",
                    context=combined_context,
                )
            except Exception as exc:
                answer = f"[Generation failed: {exc}]"
        else:
            answer = f"[No LLM] Query: {state.query}. Context: {len(context_parts)} sources available."

        state.set_answer(answer)
        return state

    async def _verify(self, state: AgentRunState, context: RAGContext) -> AgentRunState:
        """Verify the generated answer."""
        if state.is_max_steps_exceeded:
            return state

        state.record_step("verify")

        executor = _get_executor(context)
        if executor is None or "answer_verify" not in _get_registry_names(context):
            return state  # Skip verification if not available

        all_chunks = []
        for key in ("vector_search", "graph_search"):
            val = state.tool_results.get(key)
            if isinstance(val, list):
                all_chunks.extend(val)

        try:
            result = await executor.execute(
                "answer_verify",
                answer=state.final_answer or "",
                chunks=all_chunks or None,
            )
            state.tool_results["answer_verify"] = result.data if result.success else None
            if not result.success:
                state.errors.append(f"Verification failed: {result.error}")
        except Exception as exc:
            state.errors.append(f"Verification exception: {exc}")

        return state


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_executor(context: RAGContext) -> Optional[ToolExecutor]:
    """Extract ToolExecutor from context if available."""
    tools = getattr(context, "tools", None)
    if tools is None:
        return None

    if isinstance(tools, ToolExecutor):
        return tools

    if isinstance(tools, ToolRegistry):
        return ToolExecutor(tools)

    # Try to build from list of BaseTool
    if isinstance(tools, list):
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        return ToolExecutor(registry)

    return None


def _get_registry_names(context: RAGContext) -> List[str]:
    """Get available tool names from context."""
    tools = getattr(context, "tools", None)
    if tools is None:
        return []
    if isinstance(tools, ToolExecutor):
        return tools._registry.list_names()
    if isinstance(tools, ToolRegistry):
        return tools.list_names()
    if isinstance(tools, list):
        return [getattr(t, "name", "") for t in tools]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Legacy LangGraph workflow (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════


async def plan_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]
    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1, "action": AgentAction.PLAN.value,
        "thought": f"Planning: {query[:100]}", "tool_name": None, "tool_input": None,
        "tool_output": None, "observation": None, "error": None, "duration_ms": 0.0, "timestamp": "",
    }
    t0 = time.perf_counter()
    if llm:
        try:
            plan_text = await llm.generate(
                prompt="You are a RAG planning agent. Produce a step-by-step plan.",
                context=f"Query: {query}",
            )
            plan_steps = [line.strip("- 0123456789. ") for line in plan_text.split("\n") if line.strip()]
        except Exception:
            plan_steps = _default_plan(query)
    else:
        plan_steps = _default_plan(query)
    step["observation"] = f"Plan: {len(plan_steps)} steps"
    step["duration_ms"] = (time.perf_counter() - t0) * 1000
    steps = list(state.get("steps", [])); steps.append(step)
    return {"plan": plan_steps, "current_step_index": 0, "steps": steps,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "next_action": AgentAction.RETRIEVE.value, "status": AgentStatus.RUNNING.value}


async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    retrieval: Optional[RetrieverProtocol] = state.get("retrieval")  # type: ignore[assignment]
    graph: Optional[GraphRetrieverProtocol] = state.get("graph")  # type: ignore[assignment]
    t0 = time.perf_counter(); all_chunks: List[Dict[str, Any]] = []; entities: List[Dict[str, Any]] = []; reports: List[Dict[str, Any]] = []; error_msg = None
    if retrieval:
        try:
            context = retrieval.retrieve(query=query, top_k=10)
            all_chunks = [{"chunk_id": c.chunk_id, "content": c.content, "source": c.source.document_id, "score": c.source.score} for c in context.chunks]
        except Exception as exc:
            error_msg = f"Vector retrieval: {exc}"
    if graph:
        try:
            graph_context = graph.graph_search(query=query, top_k=5)
            for chunk in graph_context.chunks:
                all_chunks.append({"chunk_id": chunk.chunk_id, "content": chunk.content, "source": chunk.source.document_id, "score": chunk.source.score})
        except Exception as exc:
            pass
    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1, "action": AgentAction.RETRIEVE.value,
        "thought": f"Retrieving: {query[:100]}", "tool_name": "vector_search",
        "tool_input": {"query": query, "top_k": 10},
        "tool_output": f"{len(all_chunks)} chunks, {len(entities)} entities",
        "observation": f"Got {len(all_chunks)} vector + {len(entities)} graph results",
        "error": error_msg, "duration_ms": (time.perf_counter() - t0) * 1000, "timestamp": "",
    }
    steps = list(state.get("steps", [])); steps.append(step)
    return {"retrieved_chunks": list(state.get("retrieved_chunks", [])) + all_chunks,
            "graph_entities": list(state.get("graph_entities", [])) + entities,
            "community_reports": list(state.get("community_reports", [])) + reports,
            "steps": steps, "iteration_count": state.get("iteration_count", 0) + 1,
            "next_action": AgentAction.REASON.value}


async def reason_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]; chunks = state.get("retrieved_chunks", [])
    llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]
    t0 = time.perf_counter()
    context_text = "\n\n".join(f"[{i+1}] {c.get('content', '')[:500]}" for i, c in enumerate(chunks[:10]))
    reasoning_output = ""
    if llm and context_text:
        try:
            reasoning_output = await llm.generate(prompt="Analyze the retrieved information.", context=f"Query: {query}\n\n{context_text}")
        except Exception as exc:
            reasoning_output = f"[Reasoning unavailable: {exc}]"
    else:
        reasoning_output = f"Retrieved {len(chunks)} chunks for: {query}"
    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1, "action": AgentAction.REASON.value,
        "thought": "Analyzing retrieved information", "tool_name": None, "tool_input": None,
        "tool_output": reasoning_output[:500], "observation": f"Reasoned about {len(chunks)} chunks",
        "error": None, "duration_ms": (time.perf_counter() - t0) * 1000, "timestamp": "",
    }
    steps = list(state.get("steps", [])); steps.append(step)
    rc = list(state.get("reasoning_chain", [])); rc.append(reasoning_output)
    return {"reasoning_chain": rc, "steps": steps, "iteration_count": state.get("iteration_count", 0) + 1, "next_action": AgentAction.GENERATE.value}


async def tool_call_node(state: AgentState) -> Dict[str, Any]:
    tool_name = state.get("pending_tool"); tool_args = state.get("pending_tool_args") or {}
    executor: Optional[ToolExecutor] = state.get("tool_executor")  # type: ignore[assignment]
    if not tool_name:
        return {"next_action": AgentAction.GENERATE.value}
    t0 = time.perf_counter(); error_msg = None; tool_output = None
    if executor:
        result = await executor.execute(tool_name, **tool_args)
        tool_output = str(result.data) if result.success else result.error
        error_msg = result.error if not result.success else None
    else:
        error_msg = f"No ToolExecutor for '{tool_name}'"
    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1, "action": AgentAction.TOOL_CALL.value,
        "thought": f"Calling: {tool_name}", "tool_name": tool_name, "tool_input": tool_args,
        "tool_output": tool_output, "observation": f"Tool '{tool_name}' done",
        "error": error_msg, "duration_ms": (time.perf_counter() - t0) * 1000, "timestamp": "",
    }
    steps = list(state.get("steps", [])); steps.append(step)
    tr = dict(state.get("tool_results", {})); tr[tool_name] = tool_output
    return {"tool_results": tr, "pending_tool": None, "pending_tool_args": None,
            "steps": steps, "iteration_count": state.get("iteration_count", 0) + 1,
            "next_action": AgentAction.GENERATE.value}


async def generate_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]; llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]
    t0 = time.perf_counter()
    context_text = "\n\n".join(f"### Source {i+1} [{c.get('source', '?')}]\n{c.get('content', '')[:1000]}" for i, c in enumerate(state.get("retrieved_chunks", [])[:15]))
    if llm:
        try:
            answer = await llm.generate(prompt="Answer based on the context.", context=context_text or "No context.")
        except Exception as exc:
            answer = f"Error: {exc}"
    else:
        answer = _no_llm_summary(query, state.get("retrieved_chunks", []))
    citations: List[Dict[str, str]] = [{"chunk_id": c.get("chunk_id", ""), "document_id": c.get("source", ""), "snippet": (c.get("content", "") or "")[:200]} for c in state.get("retrieved_chunks", [])[:5]]
    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1, "action": AgentAction.GENERATE.value,
        "thought": "Generating answer", "tool_name": None, "tool_input": None,
        "tool_output": answer[:500], "observation": f"Answer: {len(answer)} chars",
        "error": None, "duration_ms": (time.perf_counter() - t0) * 1000, "timestamp": "",
    }
    steps = list(state.get("steps", [])); steps.append(step)
    return {"answer": answer, "citations": citations, "steps": steps,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "next_action": AgentAction.FINISH.value, "status": AgentStatus.COMPLETED.value}


async def reflect_node(state: AgentState) -> Dict[str, Any]:
    iteration = state.get("iteration_count", 0); answer = state.get("answer", "")
    if iteration >= state.get("max_iterations", 10):
        return {"answer": answer or "Unable to answer within step budget.",
                "status": AgentStatus.COMPLETED.value, "next_action": AgentAction.FINISH.value}
    if answer and len(answer.strip()) > 20:
        return {"status": AgentStatus.COMPLETED.value, "next_action": AgentAction.FINISH.value}
    return {"next_action": AgentAction.RETRIEVE.value, "iteration_count": iteration + 1}


def should_continue(state: AgentState) -> Literal["plan", "retrieve", "reason", "tool_call", "generate", "reflect", "__end__"]:
    if state.get("fatal_error"): return "__end__"
    if state.get("status", "") in (AgentStatus.COMPLETED.value, AgentStatus.FAILED.value): return "__end__"
    na = state.get("next_action", "")
    am: Dict[str, str] = {AgentAction.PLAN.value: "plan", AgentAction.RETRIEVE.value: "retrieve",
        AgentAction.REASON.value: "reason", AgentAction.TOOL_CALL.value: "tool_call",
        AgentAction.GENERATE.value: "generate", AgentAction.REFLECT.value: "reflect",
        AgentAction.FINISH.value: "__end__"}
    if na in am: return am[na]  # type: ignore[return-value]
    if state.get("answer"): return "reflect"
    if not state.get("retrieved_chunks"): return "retrieve"
    if not state.get("reasoning_chain"): return "reason"
    return "generate"


def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    for name, node in [("plan", plan_node), ("retrieve", retrieve_node), ("reason", reason_node),
                        ("tool_call", tool_call_node), ("generate", generate_node), ("reflect", reflect_node)]:
        workflow.add_node(name, node)
    workflow.set_entry_point("plan")
    for name in ("plan", "retrieve", "reason", "tool_call", "generate", "reflect"):
        workflow.add_conditional_edges(name, should_continue)
    return workflow


_compiled_graph = None


def get_agent_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph().compile()
    return _compiled_graph


def _default_plan(query: str) -> List[str]:
    return [f"Retrieve documents related to: {query}", "Analyze retrieved information", "Generate answer based on context"]


def _no_llm_summary(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return f"I couldn't find any relevant information to answer: {query}"
    top = sorted(chunks, key=lambda c: c.get("score", 0) or 0, reverse=True)[:3]
    excerpts = "\n\n".join(c.get("content", "")[:300] for c in top)
    return f"Based on the retrieved documents:\n\n{excerpts}"


# Legacy helper exports (kept for backward compatibility with tests)
def _format_context_for_reasoning(state: AgentState) -> str:
    parts = [f"[{i+1}] (Source: {c.get('source', '?')})\n{c.get('content', '')[:500]}" for i, c in enumerate(state.get("retrieved_chunks", [])[:10])]
    for e in state.get("graph_entities", [])[:5]:
        parts.append(f"[Entity] {e}")
    return "\n\n".join(parts) if parts else ""


def _format_context_for_generation(state: AgentState) -> str:
    parts = []
    rc = state.get("reasoning_chain", [])
    if rc: parts.append("### Analysis\n" + "\n".join(rc[-3:]))
    for i, c in enumerate(state.get("retrieved_chunks", [])[:15]):
        parts.append(f"### Source {i+1} [{c.get('source', '?')}] (score: {c.get('score', 'N/A')})\n{c.get('content', '')[:1000]}")
    entities = state.get("graph_entities", [])
    if entities: parts.append("### Entities\n" + "\n".join(str(e) for e in entities[:5]))
    return "\n\n".join(parts) if parts else "No context available."


def _heuristic_reasoning(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks: return f"No information retrieved for query: {query}"
    return f"Retrieved {len(chunks)} chunks for query: {query}. Sufficient information appears to be available."
