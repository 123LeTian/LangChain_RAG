"""
Tests for src/agent/state.py — AgentState, state helpers.
"""

import pytest

from src.agent.state import (
    AgentAction,
    AgentState,
    AgentStatus,
    AgentStep,
    append_step,
    initial_state,
    is_finished,
    set_answer,
    set_error,
    state_snapshot,
)


class TestAgentState:
    """Tests for AgentState creation and manipulation."""

    def test_initial_state_defaults(self):
        state = initial_state("What is RAG?")
        assert state["query"] == "What is RAG?"
        assert state["original_query"] == "What is RAG?"
        assert state["plan"] == []
        assert state["status"] == AgentStatus.RUNNING.value
        assert state["retrieved_chunks"] == []
        assert state["graph_entities"] == []
        assert state["community_reports"] == []
        assert state["reasoning_chain"] == []
        assert state["steps"] == []
        assert state["current_step_index"] == 0
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 10
        assert state["next_action"] == AgentAction.PLAN.value
        assert state["tool_results"] == {}
        assert state["pending_tool"] is None
        assert state["pending_tool_args"] is None
        assert state["errors"] == []
        assert state["fatal_error"] is None
        assert state["fallback_used"] is False
        assert state["answer"] is None
        assert state["citations"] == []
        assert state["messages"] == []

    def test_initial_state_custom_max_iterations(self):
        state = initial_state("query", max_iterations=5)
        assert state["max_iterations"] == 5

    def test_initial_state_overrides(self):
        state = initial_state("query", plan=["step1", "step2"])
        assert state["plan"] == ["step1", "step2"]

    def test_is_finished_running(self):
        state = initial_state("query")
        assert not is_finished(state)

    def test_is_finished_completed(self):
        state = initial_state("query")
        state["status"] = AgentStatus.COMPLETED.value
        assert is_finished(state)

    def test_is_finished_failed(self):
        state = initial_state("query")
        state["status"] = AgentStatus.FAILED.value
        assert is_finished(state)

    def test_is_finished_degraded(self):
        state = initial_state("query")
        state["status"] = AgentStatus.DEGRADED.value
        assert is_finished(state)

    def test_is_finished_fatal_error(self):
        state = initial_state("query")
        state["fatal_error"] = "Something went wrong"
        assert is_finished(state)

    def test_is_finished_max_iterations(self):
        state = initial_state("query", max_iterations=3)
        state["iteration_count"] = 3
        assert is_finished(state)

    def test_state_snapshot(self):
        state = initial_state("What is the capital of France?")
        snap = state_snapshot(state)
        assert snap["query"] == "What is the capital of France?"
        assert snap["chunks_count"] == 0
        assert snap["error_count"] == 0
        assert snap["has_answer"] is False

    def test_state_snapshot_with_data(self):
        state = initial_state("query")
        state["retrieved_chunks"] = [{"content": "chunk1"}, {"content": "chunk2"}]
        state["errors"] = ["err1"]
        snap = state_snapshot(state)
        assert snap["chunks_count"] == 2
        assert snap["error_count"] == 1


class TestStateHelpers:
    """Tests for state mutation helpers."""

    def test_append_step(self):
        state = initial_state("query")
        step: AgentStep = {
            "step_number": 1,
            "action": AgentAction.RETRIEVE.value,
            "thought": "test thought",
            "tool_name": None,
            "tool_input": None,
            "tool_output": None,
            "observation": None,
            "error": None,
            "duration_ms": 10.0,
            "timestamp": "",
        }
        new_state = append_step(state, step)
        assert len(new_state["steps"]) == 1
        assert new_state["iteration_count"] == 1

    def test_set_error_non_fatal(self):
        state = initial_state("query")
        new_state = set_error(state, "minor issue", fatal=False)
        assert len(new_state["errors"]) == 1
        assert new_state["errors"][0] == "minor issue"
        assert new_state["fatal_error"] is None
        assert new_state["status"] == AgentStatus.RUNNING.value

    def test_set_error_fatal(self):
        state = initial_state("query")
        new_state = set_error(state, "major failure", fatal=True)
        assert new_state["fatal_error"] == "major failure"
        assert new_state["status"] == AgentStatus.FAILED.value

    def test_set_answer(self):
        state = initial_state("query")
        new_state = set_answer(
            state,
            "The answer is 42.",
            citations=[{"chunk_id": "c1", "document_id": "d1", "snippet": "..."}],
        )
        assert new_state["answer"] == "The answer is 42."
        assert len(new_state["citations"]) == 1
        assert new_state["status"] == AgentStatus.COMPLETED.value
