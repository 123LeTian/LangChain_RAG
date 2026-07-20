"""
Agent Workflow — Owner: C
LangGraph workflow definition for multi-step agentic RAG reasoning.

The workflow is a state machine with these nodes:
  plan → retrieve → reason → [tool_call] → generate → reflect

Conditional edges route the agent based on its current state.
The agent can loop (reflect → plan) up to max_iterations times.

Design:
  - Nodes are pure async functions: (AgentState) → Partial<AgentState>.
  - The graph compiles into a Runnable that ainvoke() / astream() executes.
  - Dependencies (retrieval, llm, graph, tools) are bound into the state
    before invocation, so nodes access them via state keys.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph

from src.agent.state import (
    AgentAction,
    AgentState,
    AgentStatus,
    AgentStep,
    is_finished,
    set_answer,
    set_error,
)
from src.agent.tools import (
    CalculatorTool,
    DirectAnswerTool,
    GraphSearchTool,
    ToolExecutor,
    ToolRegistry,
    VectorSearchTool,
)
from src.models.rag import RAGContext, RAGQuery, RAGResponse, RAGSource, StrategyType
from src.rag.base import GeneratorProtocol, GraphRetrieverProtocol, RetrieverProtocol

logger = logging.getLogger(__name__)


# ── Node Implementations ──────────────────────────────────────────────────


async def plan_node(state: AgentState) -> Dict[str, Any]:
    """Analyze the query and produce a step-by-step plan.

    Uses the LLM to decompose the query into actionable steps.
    If the LLM is unavailable, falls back to a simple default plan.
    """
    query = state["query"]
    llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]

    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1,
        "action": AgentAction.PLAN.value,
        "thought": f"Planning how to answer: {query[:100]}",
        "tool_name": None,
        "tool_input": None,
        "tool_output": None,
        "observation": None,
        "error": None,
        "duration_ms": 0.0,
        "timestamp": "",
    }

    t0 = time.perf_counter()

    if llm:
        try:
            prompt = (
                "You are a RAG planning agent. Given a user query, produce a step-by-step plan "
                "to answer it. Each step should be one of: retrieve information, reason about "
                "findings, or generate the answer. Output one step per line.\n\n"
                f"Query: {query}\n\nPlan:"
            )
            plan_text = await llm.generate(prompt=prompt, context="")
            plan_steps = [
                line.strip("- 0123456789. ") for line in plan_text.split("\n") if line.strip()
            ]
        except Exception as exc:
            logger.warning("LLM plan generation failed, using default plan: %s", exc)
            plan_steps = _default_plan(query)
    else:
        plan_steps = _default_plan(query)

    step["observation"] = f"Generated plan with {len(plan_steps)} steps"
    step["duration_ms"] = (time.perf_counter() - t0) * 1000

    steps = list(state.get("steps", []))
    steps.append(step)

    return {
        "plan": plan_steps,
        "current_step_index": 0,
        "steps": steps,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "next_action": AgentAction.RETRIEVE.value,
        "status": AgentStatus.RUNNING.value,
    }


async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Execute retrieval using the available tools.

    Performs vector/hybrid search and optionally graph search.
    Results are accumulated into the state.
    """
    query = state["query"]
    retrieval: Optional[RetrieverProtocol] = state.get("retrieval")  # type: ignore[assignment]
    graph: Optional[GraphRetrieverProtocol] = state.get("graph")  # type: ignore[assignment]

    t0 = time.perf_counter()
    all_chunks: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    error_msg: Optional[str] = None

    # Vector search
    if retrieval:
        try:
            context = retrieval.retrieve(query=query, top_k=10)
            all_chunks = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "source": c.source.document_id,
                    "score": c.source.score,
                }
                for c in context.chunks
            ]
        except Exception as exc:
            logger.error("Vector retrieval failed: %s", exc)
            error_msg = f"Vector retrieval: {exc}"

    # Graph search
    if graph:
        try:
            graph_context = graph.graph_search(query=query, top_k=5)
            # Graph results may include entity data in metadata
            for chunk in graph_context.chunks:
                if chunk.source.metadata:
                    if "entity" in chunk.source.metadata:
                        entities.append(chunk.source.metadata)
                    if "community_report" in chunk.source.metadata:
                        reports.append(chunk.source.metadata)
            # Also add graph chunks
            for chunk in graph_context.chunks:
                all_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source.document_id,
                    "score": chunk.source.score,
                })
        except Exception as exc:
            logger.error("Graph retrieval failed: %s", exc)
            # Non-fatal — continue with vector results only

    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1,
        "action": AgentAction.RETRIEVE.value,
        "thought": f"Retrieving information for: {query[:100]}",
        "tool_name": "vector_search",
        "tool_input": {"query": query, "top_k": 10},
        "tool_output": f"Retrieved {len(all_chunks)} chunks, {len(entities)} entities",
        "observation": f"Got {len(all_chunks)} vector + {len(entities)} graph results",
        "error": error_msg,
        "duration_ms": (time.perf_counter() - t0) * 1000,
        "timestamp": "",
    }

    steps = list(state.get("steps", []))
    steps.append(step)

    accumulated_chunks = list(state.get("retrieved_chunks", [])) + all_chunks
    accumulated_entities = list(state.get("graph_entities", [])) + entities
    accumulated_reports = list(state.get("community_reports", [])) + reports

    return {
        "retrieved_chunks": accumulated_chunks,
        "graph_entities": accumulated_entities,
        "community_reports": accumulated_reports,
        "steps": steps,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "next_action": AgentAction.REASON.value,
    }


async def reason_node(state: AgentState) -> Dict[str, Any]:
    """Analyze retrieved information and synthesize insights.

    This is the agent's "thinking" step — it reviews what was retrieved
    and decides if more information is needed or if we can proceed to generation.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]

    t0 = time.perf_counter()
    context_text = _format_context_for_reasoning(state)

    reasoning_output = ""
    if llm and context_text:
        try:
            prompt = (
                "You are a reasoning agent. Analyze the retrieved information below "
                "and determine if it's sufficient to answer the query. "
                "If information is missing, specify what else is needed. "
                "If sufficient, synthesize the key points.\n\n"
                f"Query: {query}\n\n"
                f"Retrieved Information:\n{context_text}\n\n"
                "Analysis:"
            )
            reasoning_output = await llm.generate(prompt=prompt, context="")
        except Exception as exc:
            logger.warning("LLM reasoning failed: %s", exc)
            reasoning_output = f"[Reasoning unavailable: {exc}]"
    else:
        reasoning_output = _heuristic_reasoning(query, chunks)

    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1,
        "action": AgentAction.REASON.value,
        "thought": "Analyzing retrieved information",
        "tool_name": None,
        "tool_input": None,
        "tool_output": reasoning_output[:500],
        "observation": f"Reasoned about {len(chunks)} chunks",
        "error": None,
        "duration_ms": (time.perf_counter() - t0) * 1000,
        "timestamp": "",
    }

    steps = list(state.get("steps", []))
    steps.append(step)

    reasoning_chain = list(state.get("reasoning_chain", []))
    reasoning_chain.append(reasoning_output)

    return {
        "reasoning_chain": reasoning_chain,
        "steps": steps,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "next_action": AgentAction.GENERATE.value,
    }


async def tool_call_node(state: AgentState) -> Dict[str, Any]:
    """Execute a pending tool call and collect the result."""
    tool_name = state.get("pending_tool")
    tool_args = state.get("pending_tool_args") or {}
    executor: Optional[ToolExecutor] = state.get("tool_executor")  # type: ignore[assignment]

    if not tool_name:
        return {
            "next_action": AgentAction.GENERATE.value,
        }

    t0 = time.perf_counter()
    error_msg: Optional[str] = None
    tool_output: Optional[str] = None

    if executor:
        result = await executor.execute(tool_name, **tool_args)
        tool_output = str(result.data) if result.success else result.error
        error_msg = result.error if not result.success else None
    else:
        error_msg = f"No ToolExecutor available for '{tool_name}'"

    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1,
        "action": AgentAction.TOOL_CALL.value,
        "thought": f"Calling tool: {tool_name}",
        "tool_name": tool_name,
        "tool_input": tool_args,
        "tool_output": tool_output,
        "observation": f"Tool '{tool_name}' completed",
        "error": error_msg,
        "duration_ms": (time.perf_counter() - t0) * 1000,
        "timestamp": "",
    }

    steps = list(state.get("steps", []))
    steps.append(step)

    tool_results = dict(state.get("tool_results", {}))
    tool_results[tool_name] = tool_output

    return {
        "tool_results": tool_results,
        "pending_tool": None,
        "pending_tool_args": None,
        "steps": steps,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "next_action": AgentAction.GENERATE.value,
    }


async def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer from retrieved + reasoned context."""
    query = state["query"]
    llm: Optional[GeneratorProtocol] = state.get("llm")  # type: ignore[assignment]

    t0 = time.perf_counter()

    # Build the context string
    context_text = _format_context_for_generation(state)

    if llm:
        try:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based ONLY on "
                "the provided context. If the context doesn't contain enough information, "
                "say so clearly. Always cite specific sources when possible."
            )
            answer = await llm.generate(prompt=system_prompt, context=context_text)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"I encountered an error while generating the answer: {exc}"
    else:
        # No LLM available — return a summary of retrieved information
        answer = _no_llm_summary(query, state.get("retrieved_chunks", []))

    duration = (time.perf_counter() - t0) * 1000

    # Build citations from retrieved chunks
    citations: List[Dict[str, str]] = []
    for chunk in state.get("retrieved_chunks", [])[:5]:
        citations.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("source", ""),
            "snippet": (chunk.get("content", "") or "")[:200],
        })

    step: AgentStep = {
        "step_number": state.get("iteration_count", 0) + 1,
        "action": AgentAction.GENERATE.value,
        "thought": "Generating final answer",
        "tool_name": None,
        "tool_input": None,
        "tool_output": answer[:500],
        "observation": f"Generated answer ({len(answer)} chars)",
        "error": None,
        "duration_ms": duration,
        "timestamp": "",
    }

    steps = list(state.get("steps", []))
    steps.append(step)

    return {
        "answer": answer,
        "citations": citations,
        "steps": steps,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "next_action": AgentAction.FINISH.value,
        "status": AgentStatus.COMPLETED.value,
    }


async def reflect_node(state: AgentState) -> Dict[str, Any]:
    """Reflect on the quality of the answer and decide next steps.

    This is the metacognition step: should we loop back for more retrieval
    or is the answer good enough?

    Returns:
        State update with next_action set appropriately.
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 10)
    answer = state.get("answer", "")

    # Hard stop on max iterations
    if iteration >= max_iter:
        if not answer:
            answer = "I was unable to fully answer this question within the available steps."
        return {
            "answer": answer,
            "status": AgentStatus.COMPLETED.value,
            "next_action": AgentAction.FINISH.value,
        }

    # If we have a non-empty answer, we're done
    if answer and len(answer.strip()) > 20:
        return {
            "status": AgentStatus.COMPLETED.value,
            "next_action": AgentAction.FINISH.value,
        }

    # No good answer yet — loop back to retrieval with expanded query
    return {
        "next_action": AgentAction.RETRIEVE.value,
        "iteration_count": iteration + 1,
    }


# ── Conditional Edge ──────────────────────────────────────────────────────


def should_continue(state: AgentState) -> Literal[
    "plan", "retrieve", "reason", "tool_call", "generate", "reflect", "__end__"
]:
    """Route to the next node based on current state.

    This is the core routing function. It looks at next_action and status
    to determine where the state machine should go next.

    Returns:
        The name of the next node, or "__end__" to terminate.
    """
    # Check terminal conditions first
    if state.get("fatal_error"):
        return "__end__"

    status = state.get("status", "")
    if status in (AgentStatus.COMPLETED.value, AgentStatus.FAILED.value):
        return "__end__"

    # Follow the explicit next_action if set
    next_action = state.get("next_action", "")
    action_map: Dict[str, str] = {
        AgentAction.PLAN.value: "plan",
        AgentAction.RETRIEVE.value: "retrieve",
        AgentAction.REASON.value: "reason",
        AgentAction.TOOL_CALL.value: "tool_call",
        AgentAction.GENERATE.value: "generate",
        AgentAction.REFLECT.value: "reflect",
        AgentAction.FINISH.value: "__end__",
    }
    if next_action in action_map:
        return action_map[next_action]  # type: ignore[return-value]

    # Fallback: if we have an answer, go to reflect; otherwise retrieve
    if state.get("answer"):
        return "reflect"
    if not state.get("retrieved_chunks"):
        return "retrieve"
    if not state.get("reasoning_chain"):
        return "reason"
    return "generate"


# ── Graph Construction ────────────────────────────────────────────────────


def build_agent_graph() -> StateGraph:
    """Build and return the agentic RAG StateGraph.

    The graph has nodes for each agent action and conditional edges
    that route between them based on state.

    Returns:
        A StateGraph instance (not yet compiled). Call .compile() to use.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)

    # Entry point
    workflow.set_entry_point("plan")

    # Conditional edges from each active node
    workflow.add_conditional_edges("plan", should_continue)
    workflow.add_conditional_edges("retrieve", should_continue)
    workflow.add_conditional_edges("reason", should_continue)
    workflow.add_conditional_edges("tool_call", should_continue)
    workflow.add_conditional_edges("generate", should_continue)
    workflow.add_conditional_edges("reflect", should_continue)

    # Terminal: reflect → END when status is completed
    # (handled by should_continue returning "__end__")

    return workflow


# ── Compiled graph (singleton, lazily built) ──────────────────────────────

_compiled_graph = None


def get_agent_graph() -> Any:
    """Return the compiled agent graph, building it lazily if needed."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_agent_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


# ── Helpers ───────────────────────────────────────────────────────────────


def _default_plan(query: str) -> List[str]:
    """Generate a simple default plan without LLM."""
    return [
        f"Retrieve documents related to: {query}",
        "Analyze retrieved information",
        "Generate answer based on context",
    ]


def _format_context_for_reasoning(state: AgentState) -> str:
    """Format retrieved chunks for the reasoning step."""
    parts: List[str] = []
    for i, chunk in enumerate(state.get("retrieved_chunks", [])[:10]):
        content = chunk.get("content", "")[:500]
        src = chunk.get("source", "unknown")
        parts.append(f"[{i + 1}] (Source: {src})\n{content}")
    for entity in state.get("graph_entities", [])[:5]:
        parts.append(f"[Entity] {entity}")
    return "\n\n".join(parts) if parts else ""


def _format_context_for_generation(state: AgentState) -> str:
    """Format all accumulated context for final answer generation."""
    parts: List[str] = []

    # Reasoning chain
    reasoning = state.get("reasoning_chain", [])
    if reasoning:
        parts.append("### Analysis\n" + "\n".join(reasoning[-3:]))  # Last 3 reasoning steps

    # Retrieved chunks
    for i, chunk in enumerate(state.get("retrieved_chunks", [])[:15]):
        content = chunk.get("content", "")[:1000]
        src = chunk.get("source", "unknown")
        score = chunk.get("score", "N/A")
        parts.append(f"### Source {i + 1} [{src}] (score: {score})\n{content}")

    # Graph entities
    entities = state.get("graph_entities", [])
    if entities:
        parts.append("### Entities\n" + "\n".join(str(e) for e in entities[:5]))

    return "\n\n".join(parts) if parts else "No context available."


def _heuristic_reasoning(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Simple heuristic reasoning when LLM is unavailable."""
    if not chunks:
        return f"No information retrieved for query: {query}"
    total_chars = sum(len(c.get("content", "")) for c in chunks)
    return (
        f"Retrieved {len(chunks)} chunks ({total_chars} total chars) for query: {query}. "
        f"Sufficient information appears to be available."
    )


def _no_llm_summary(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate a basic summary when no LLM is available."""
    if not chunks:
        return f"I couldn't find any relevant information to answer: {query}"
    top_chunks = sorted(chunks, key=lambda c: c.get("score", 0) or 0, reverse=True)[:3]
    excerpts = "\n\n".join(c.get("content", "")[:300] for c in top_chunks)
    return (
        f"Based on the retrieved documents, here's the most relevant information:\n\n{excerpts}"
    )
