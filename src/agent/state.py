"""
Agent State — Owner: C
State management for agentic RAG workflows.

Defines the state schema, reducer functions, and state machine transitions
used by the LangGraph-based agentic RAG workflow.

Design:
  - AgentState is a TypedDict compatible with LangGraph's StateGraph.
  - State transitions follow: plan → retrieve → reason → [tool_call] → generate → reflect.
  - All accumulated data (chunks, entities, reasoning steps) lives in the state.
  - Helper functions produce initial state and extract safe summaries for streaming.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph.message import add_messages


# ── Enums ──────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Agent execution status."""

    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"  # Completed but with fallback


class AgentAction(str, Enum):
    """Actions the agent can take at each step."""

    PLAN = "plan"
    RETRIEVE = "retrieve"
    REASON = "reason"
    TOOL_CALL = "tool_call"
    GENERATE = "generate"
    REFLECT = "reflect"
    FINISH = "finish"


# ── Agent Step ─────────────────────────────────────────────────────────────


class AgentStep(TypedDict, total=False):
    """Record of a single agent reasoning step."""

    step_number: int
    action: str  # AgentAction value
    thought: str  # The agent's reasoning for this step
    tool_name: Optional[str]  # Tool called (if action == tool_call)
    tool_input: Optional[Dict[str, Any]]  # Arguments to the tool
    tool_output: Optional[str]  # Result from the tool
    observation: Optional[str]  # What the agent observed
    error: Optional[str]  # Error message if step failed
    duration_ms: float
    timestamp: str  # ISO datetime


# ── Main Agent State ───────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """Complete state for an agentic RAG execution.

    This TypedDict is compatible with LangGraph's StateGraph. Each node
    reads from and writes to specific keys. The graph framework merges
    returned dicts using the reducers defined on each key.

    Key groups:
      - Core: query, plan, status
      - Accumulated data: chunks, entities, reports, reasoning_chain
      - Tool execution: tool_results, pending_tool, pending_tool_args
      - Control flow: iteration_count, max_iterations, current_step_index
      - Error handling: errors, fatal_error
      - Result: answer, citations
      - Messages: LangGraph's annotated message list (add_messages reducer)
    """

    # ── Core ──────────────────────────────────────────────────────────
    query: str
    original_query: str
    plan: List[str]  # Sequence of action descriptions
    status: str  # AgentStatus value

    # ── Accumulated Data ──────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]  # Chunks from vector/hybrid search
    graph_entities: List[Dict[str, Any]]  # Entities from graph search
    community_reports: List[Dict[str, Any]]  # Community summary reports
    reasoning_chain: List[str]  # Chain-of-thought trace
    steps: List[AgentStep]  # Record of agent reasoning steps

    # ── Control Flow ──────────────────────────────────────────────────
    current_step_index: int  # Pointer into plan list
    iteration_count: int  # Total agent iterations so far
    max_iterations: int  # Hard stop at this count (default 10)
    next_action: str  # AgentAction value for routing

    # ── Tool Execution ────────────────────────────────────────────────
    tool_results: Dict[str, Any]  # tool_name → tool output
    pending_tool: Optional[str]  # Tool name currently being executed
    pending_tool_args: Optional[Dict[str, Any]]  # Arguments for pending tool

    # ── Error Handling ────────────────────────────────────────────────
    errors: List[str]  # Non-fatal errors (logged but execution continues)
    fatal_error: Optional[str]  # Fatal error (triggers immediate termination)
    fallback_used: bool  # Whether fallback was activated

    # ── Result ────────────────────────────────────────────────────────
    answer: Optional[str]  # Final generated answer
    citations: List[Dict[str, str]]  # Source citations: {chunk_id, document_id, snippet}

    # ── LangGraph Integration ─────────────────────────────────────────
    messages: Annotated[Sequence, add_messages]  # type: ignore[valid-type]


# ── State factory ──────────────────────────────────────────────────────────


def initial_state(
    query: str,
    max_iterations: int = 10,
    **overrides: Any,
) -> AgentState:
    """Create a fresh AgentState for a new agent run.

    Args:
        query: The user's natural-language query.
        max_iterations: Maximum number of agent reasoning loops.
        **overrides: Additional state key-value pairs to set.

    Returns:
        A new AgentState dict ready for LangGraph invocation.
    """
    state: AgentState = {
        "query": query,
        "original_query": query,
        "plan": [],
        "status": AgentStatus.RUNNING.value,
        "retrieved_chunks": [],
        "graph_entities": [],
        "community_reports": [],
        "reasoning_chain": [],
        "steps": [],
        "current_step_index": 0,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "next_action": AgentAction.PLAN.value,
        "tool_results": {},
        "pending_tool": None,
        "pending_tool_args": None,
        "errors": [],
        "fatal_error": None,
        "fallback_used": False,
        "answer": None,
        "citations": [],
        "messages": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ── State helpers ──────────────────────────────────────────────────────────


def state_snapshot(state: AgentState) -> Dict[str, Any]:
    """Extract a safe-to-serialize summary of the current state.

    Used for streaming events — avoids leaking internal objects.
    """
    return {
        "status": state.get("status"),
        "query": state.get("query", "")[:100],
        "current_step_index": state.get("current_step_index", 0),
        "plan_length": len(state.get("plan", [])),
        "chunks_count": len(state.get("retrieved_chunks", [])),
        "entities_count": len(state.get("graph_entities", [])),
        "reasoning_steps": len(state.get("reasoning_chain", [])),
        "iteration_count": state.get("iteration_count", 0),
        "has_answer": state.get("answer") is not None,
        "tool_count": len(state.get("tool_results", {})),
        "error_count": len(state.get("errors", [])),
        "fatal_error": state.get("fatal_error"),
        "next_action": state.get("next_action"),
    }


def is_finished(state: AgentState) -> bool:
    """Check if the agent has reached a terminal state."""
    status = state.get("status", "")
    if status in (AgentStatus.COMPLETED.value, AgentStatus.FAILED.value, AgentStatus.DEGRADED.value):
        return True
    if state.get("fatal_error"):
        return True
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return True
    return False


def append_step(state: AgentState, step: AgentStep) -> AgentState:
    """Append a reasoning step and return updated state."""
    steps = list(state.get("steps", []))
    steps.append(step)
    return {**state, "steps": steps, "iteration_count": state.get("iteration_count", 0) + 1}  # type: ignore[typeddict-item]


def set_error(state: AgentState, error: str, fatal: bool = False) -> AgentState:
    """Record an error in the state."""
    errors = list(state.get("errors", []))
    errors.append(error)
    update: Dict[str, Any] = {"errors": errors}
    if fatal:
        update["fatal_error"] = error
        update["status"] = AgentStatus.FAILED.value
    return {**state, **update}  # type: ignore[typeddict-item]


def set_answer(
    state: AgentState, answer: str, citations: Optional[List[Dict[str, str]]] = None
) -> AgentState:
    """Set the final answer and mark completed."""
    return {
        **state,  # type: ignore[typeddict-item]
        "answer": answer,
        "citations": citations or [],
        "status": AgentStatus.COMPLETED.value,
    }
