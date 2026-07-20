"""
Tests for src/agent/workflow.py — Agent workflow nodes and graph construction.
"""

import pytest

from src.agent.state import (
    AgentAction,
    AgentState,
    AgentStatus,
    initial_state,
)
from src.agent.workflow import (
    build_agent_graph,
    generate_node,
    plan_node,
    reason_node,
    reflect_node,
    retrieve_node,
    should_continue,
    tool_call_node,
    _default_plan,
    _format_context_for_reasoning,
    _format_context_for_generation,
    _heuristic_reasoning,
    _no_llm_summary,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _state_with_chunks(query: str = "test query") -> AgentState:
    """Create a state with some retrieved chunks for testing."""
    state = initial_state(query)
    state["retrieved_chunks"] = [
        {
            "chunk_id": "c1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "source": "doc1",
            "score": 0.9,
        },
        {
            "chunk_id": "c2",
            "content": "Deep learning uses neural networks with many layers.",
            "source": "doc1",
            "score": 0.8,
        },
    ]
    return state


# ── Graph Construction Tests ───────────────────────────────────────────────


class TestGraphConstruction:
    """Tests for the StateGraph construction."""

    def test_build_graph_returns_state_graph(self):
        graph = build_agent_graph()
        assert graph is not None

    def test_compiled_graph(self):
        from src.agent.workflow import get_agent_graph
        graph = get_agent_graph()
        assert graph is not None
        # Second call returns cached instance
        graph2 = get_agent_graph()
        assert graph is graph2


# ── Node Tests ─────────────────────────────────────────────────────────────


class TestPlanNode:
    """Tests for the plan_node."""

    @pytest.mark.asyncio
    async def test_plan_node_default_plan(self):
        state = initial_state("What is RAG?")
        result = await plan_node(state)
        assert len(result["plan"]) == 3
        assert "RAG" in result["plan"][0]
        assert result["next_action"] == AgentAction.RETRIEVE.value
        assert result["status"] == AgentStatus.RUNNING.value
        assert result["current_step_index"] == 0
        assert len(result["steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_node_increments_iteration(self):
        state = initial_state("query", max_iterations=5)
        state["iteration_count"] = 2
        result = await plan_node(state)
        assert result["iteration_count"] == 3


class TestRetrieveNode:
    """Tests for the retrieve_node (without real retriever)."""

    @pytest.mark.asyncio
    async def test_retrieve_node_no_retriever(self):
        state = initial_state("test query")
        result = await retrieve_node(state)
        # Without a real retriever, no chunks are added
        assert result["next_action"] == AgentAction.REASON.value
        assert len(result["steps"]) == 1

    @pytest.mark.asyncio
    async def test_retrieve_node_preserves_existing_chunks(self):
        state = _state_with_chunks()
        result = await retrieve_node(state)
        # Existing chunks are preserved
        assert len(result["retrieved_chunks"]) >= 2

    @pytest.mark.asyncio
    async def test_retrieve_node_increments_iteration(self):
        state = initial_state("query")
        result = await retrieve_node(state)
        assert result["iteration_count"] == 1


class TestReasonNode:
    """Tests for the reason_node."""

    @pytest.mark.asyncio
    async def test_reason_node_heuristic(self):
        state = _state_with_chunks()
        result = await reason_node(state)
        assert result["next_action"] == AgentAction.GENERATE.value
        assert len(result["reasoning_chain"]) == 1
        reasoning = result["reasoning_chain"][0]
        assert "2 chunks" in reasoning

    @pytest.mark.asyncio
    async def test_reason_node_no_chunks(self):
        state = initial_state("query")
        result = await reason_node(state)
        assert "No information retrieved" in result["reasoning_chain"][0]


class TestGenerateNode:
    """Tests for the generate_node."""

    @pytest.mark.asyncio
    async def test_generate_node_no_llm(self):
        state = _state_with_chunks("What is ML?")
        result = await generate_node(state)
        assert len(result["answer"]) > 0
        assert result["status"] == AgentStatus.COMPLETED.value
        assert result["next_action"] == AgentAction.FINISH.value
        assert len(result["citations"]) > 0

    @pytest.mark.asyncio
    async def test_generate_node_empty_chunks(self):
        state = initial_state("query")
        result = await generate_node(state)
        assert "couldn't find" in result["answer"].lower()
        assert result["status"] == AgentStatus.COMPLETED.value


class TestReflectNode:
    """Tests for the reflect_node."""

    @pytest.mark.asyncio
    async def test_reflect_node_with_good_answer(self):
        state = initial_state("query")
        state["answer"] = "This is a comprehensive answer that fully addresses the query."
        result = await reflect_node(state)
        assert result["status"] == AgentStatus.COMPLETED.value
        assert result["next_action"] == AgentAction.FINISH.value

    @pytest.mark.asyncio
    async def test_reflect_node_max_iterations(self):
        state = initial_state("query", max_iterations=5)
        state["iteration_count"] = 5
        state["answer"] = ""
        result = await reflect_node(state)
        assert result["status"] == AgentStatus.COMPLETED.value
        assert len(result.get("answer", "")) > 0  # Fallback answer provided

    @pytest.mark.asyncio
    async def test_reflect_node_loops_back(self):
        state = initial_state("query")
        state["answer"] = "short"  # Too short
        result = await reflect_node(state)
        assert result["next_action"] == AgentAction.RETRIEVE.value


class TestToolCallNode:
    """Tests for the tool_call_node."""

    @pytest.mark.asyncio
    async def test_tool_call_node_no_pending_tool(self):
        state = initial_state("query")
        result = await tool_call_node(state)
        assert result["next_action"] == AgentAction.GENERATE.value


# ── Conditional Edge Tests ─────────────────────────────────────────────────


class TestShouldContinue:
    """Tests for the should_continue routing function."""

    def test_fatal_error_ends(self):
        state = initial_state("query")
        state["fatal_error"] = "fatal"
        assert should_continue(state) == "__end__"

    def test_completed_ends(self):
        state = initial_state("query")
        state["status"] = AgentStatus.COMPLETED.value
        assert should_continue(state) == "__end__"

    def test_failed_ends(self):
        state = initial_state("query")
        state["status"] = AgentStatus.FAILED.value
        assert should_continue(state) == "__end__"

    def test_next_action_plan(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.PLAN.value
        assert should_continue(state) == "plan"

    def test_next_action_retrieve(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.RETRIEVE.value
        assert should_continue(state) == "retrieve"

    def test_next_action_reason(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.REASON.value
        assert should_continue(state) == "reason"

    def test_next_action_generate(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.GENERATE.value
        assert should_continue(state) == "generate"

    def test_next_action_reflect(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.REFLECT.value
        assert should_continue(state) == "reflect"

    def test_next_action_finish(self):
        state = initial_state("query")
        state["next_action"] = AgentAction.FINISH.value
        assert should_continue(state) == "__end__"

    def test_fallback_routing_with_answer(self):
        state = initial_state("query")
        state["answer"] = "some answer"
        state["next_action"] = ""  # No explicit action set
        assert should_continue(state) == "reflect"

    def test_fallback_routing_no_chunks(self):
        state = initial_state("query")
        state["next_action"] = ""
        assert should_continue(state) == "retrieve"


# ── Helper Function Tests ──────────────────────────────────────────────────


class TestHelpers:
    """Tests for workflow helper functions."""

    def test_default_plan(self):
        plan = _default_plan("What is RAG?")
        assert len(plan) == 3
        assert any("RAG" in step for step in plan)

    def test_format_context_for_reasoning(self):
        state = _state_with_chunks()
        text = _format_context_for_reasoning(state)
        assert "machine learning" in text.lower()
        assert "Source:" in text

    def test_format_context_for_reasoning_empty(self):
        state = initial_state("query")
        text = _format_context_for_reasoning(state)
        assert text == ""

    def test_format_context_for_generation(self):
        state = _state_with_chunks()
        state["reasoning_chain"] = ["Analysis step 1"]
        text = _format_context_for_generation(state)
        assert "### Analysis" in text
        assert "### Source" in text

    def test_format_context_for_generation_empty(self):
        state = initial_state("query")
        text = _format_context_for_generation(state)
        assert "No context available" in text

    def test_heuristic_reasoning(self):
        chunks = [{"content": "test content"}]
        result = _heuristic_reasoning("query", chunks)
        assert "1 chunks" in result

    def test_heuristic_reasoning_empty(self):
        result = _heuristic_reasoning("query", [])
        assert "No information" in result

    def test_no_llm_summary(self):
        chunks = [
            {"content": "First chunk content here.", "score": 0.9},
            {"content": "Second chunk content.", "score": 0.7},
        ]
        result = _no_llm_summary("query", chunks)
        assert "First chunk" in result

    def test_no_llm_summary_empty(self):
        result = _no_llm_summary("query", [])
        assert "couldn't find" in result.lower()
