"""
Agent State — Owner: C
State management for agentic RAG workflows.

Two state models:
  1. AgentRunState — lightweight finite-state agent (CURRENT, used by AgentWorkflow)
  2. AgentState (Legacy) — LangGraph TypedDict (kept for backward compatibility)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TypedDict

# LangGraph is optional — only needed for the legacy AgentState TypedDict.
# The CURRENT AgentRunState does NOT depend on it.
try:
    from langgraph.graph.message import add_messages
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    add_messages = None  # type: ignore[assignment]
    _LANGGRAPH_AVAILABLE = False


# ── Enums ──────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Agent execution status."""

    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"


class AgentAction(str, Enum):
    """Actions the agent can take at each step."""

    PLAN = "plan"
    RETRIEVE = "retrieve"
    REASON = "reason"
    TOOL_CALL = "tool_call"
    GENERATE = "generate"
    REFLECT = "reflect"
    FINISH = "finish"


# ═══════════════════════════════════════════════════════════════════════════
# AgentRunState — current finite-state agent model
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AgentRunState:
    """Lightweight state for the finite-state agent workflow.

    Fields:
        query:          Original user query.
        selected_tools: Tools chosen by the router for this query.
        tool_results:   Accumulated results keyed by tool name.
        steps:          Ordered list of completed workflow steps.
        final_answer:   Generated answer (set after generate step).
        max_steps:      Hard limit on workflow steps (default 4, no infinite loops).
        current_step:   Current step counter.
        status:         Current execution status.
        errors:         Non-fatal errors collected during execution.
    """

    query: str = ""
    selected_tools: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    citations: List[Dict[str, str]] = field(default_factory=list)
    max_steps: int = 4
    current_step: int = 0
    status: str = "running"  # running | completed | failed
    errors: List[str] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        """Has the agent reached a terminal state?"""
        return self.status in ("completed", "failed")

    @property
    def is_max_steps_exceeded(self) -> bool:
        """Has the agent exceeded its step budget?"""
        return self.current_step > self.max_steps

    def record_step(self, step_name: str) -> None:
        """Record a completed workflow step."""
        self.steps.append(step_name)
        self.current_step += 1

    def set_answer(self, answer: str) -> None:
        """Set the final answer and mark completed."""
        self.final_answer = answer
        self.status = "completed"

    def set_failed(self, error: str) -> None:
        """Mark the agent as failed."""
        self.errors.append(error)
        self.status = "failed"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "query": self.query,
            "selected_tools": self.selected_tools,
            "tool_results": self.tool_results,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "citations": self.citations,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "status": self.status,
            "errors": self.errors,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AgentState (Legacy) — LangGraph TypedDict for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


class AgentStep(TypedDict, total=False):
    """Record of a single agent reasoning step (legacy)."""

    step_number: int
    action: str
    thought: str
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    tool_output: Optional[str]
    observation: Optional[str]
    error: Optional[str]
    duration_ms: float
    timestamp: str


class AgentState(TypedDict, total=False):
    """Complete state for an agentic RAG execution (legacy LangGraph TypedDict)."""

    query: str
    original_query: str
    plan: List[str]
    status: str
    retrieved_chunks: List[Dict[str, Any]]
    graph_entities: List[Dict[str, Any]]
    community_reports: List[Dict[str, Any]]
    reasoning_chain: List[str]
    steps: List[AgentStep]
    current_step_index: int
    iteration_count: int
    max_iterations: int
    next_action: str
    tool_results: Dict[str, Any]
    pending_tool: Optional[str]
    pending_tool_args: Optional[Dict[str, Any]]
    errors: List[str]
    fatal_error: Optional[str]
    fallback_used: bool
    answer: Optional[str]
    citations: List[Dict[str, str]]
    messages: Sequence  # type: ignore[valid-type]
    # When langgraph is available this becomes Annotated[Sequence, add_messages]


# ── Legacy helpers ──────────────────────────────────────────────────────────


def initial_state(query: str, max_iterations: int = 10, **overrides: Any) -> AgentState:
    """Create a fresh AgentState for a new agent run (legacy)."""
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


def state_snapshot(state: AgentState) -> Dict[str, Any]:
    """Extract a safe-to-serialize summary of the current state (legacy)."""
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
    """Check if the agent has reached a terminal state (legacy)."""
    status = state.get("status", "")
    if status in (AgentStatus.COMPLETED.value, AgentStatus.FAILED.value, AgentStatus.DEGRADED.value):
        return True
    if state.get("fatal_error"):
        return True
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return True
    return False


def append_step(state: AgentState, step: AgentStep) -> AgentState:
    """Append a reasoning step and return updated state (legacy)."""
    steps = list(state.get("steps", []))
    steps.append(step)
    return {**state, "steps": steps, "iteration_count": state.get("iteration_count", 0) + 1}  # type: ignore[typeddict-item]


def set_error(state: AgentState, error: str, fatal: bool = False) -> AgentState:
    """Record an error in the state (legacy)."""
    errors = list(state.get("errors", []))
    errors.append(error)
    update: Dict[str, Any] = {"errors": errors}
    if fatal:
        update["fatal_error"] = error
        update["status"] = AgentStatus.FAILED.value
    return {**state, **update}  # type: ignore[typeddict-item]


def set_answer(state: AgentState, answer: str, citations: Optional[List[Dict[str, str]]] = None) -> AgentState:
    """Set the final answer and mark completed (legacy)."""
    return {**state, "answer": answer, "citations": citations or [], "status": AgentStatus.COMPLETED.value}  # type: ignore[typeddict-item]
