"""
Tests for the new Agentic RAG components — AgentRunState, AgentRouter, AgentWorkflow.
"""

import pytest

from src.agent.state import AgentRunState
from src.agent.router import AgentRouter, QueryCategory, RouterResult
from src.agent.tools import (
    AnswerVerifyTool,
    DocumentSummaryTool,
    GraphSearchTool,
    VectorSearchTool,
)
from src.agent.workflow import AgentWorkflow
from src.models.rag import RAGContext, RAGChunk, RAGSource


# ═══════════════════════════════════════════════════════════════════════════
# AgentRunState Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRunState:
    """Tests for the new finite-state AgentRunState."""

    def test_initial_state(self):
        s = AgentRunState(query="What is RAG?")
        assert s.query == "What is RAG?"
        assert s.selected_tools == []
        assert s.tool_results == {}
        assert s.steps == []
        assert s.final_answer is None
        assert s.max_steps == 4
        assert s.current_step == 0
        assert s.status == "running"
        assert s.errors == []

    def test_is_finished(self):
        s = AgentRunState()
        assert not s.is_finished
        s.status = "completed"
        assert s.is_finished
        s.status = "failed"
        assert s.is_finished

    def test_is_max_steps_exceeded(self):
        s = AgentRunState(max_steps=3)
        assert not s.is_max_steps_exceeded
        s.current_step = 4  # > max_steps (3)
        assert s.is_max_steps_exceeded

    def test_record_step(self):
        s = AgentRunState()
        s.record_step("intent_detection")
        assert s.steps == ["intent_detection"]
        assert s.current_step == 1

    def test_set_answer(self):
        s = AgentRunState()
        s.set_answer("Paris is the capital.")
        assert s.final_answer == "Paris is the capital."
        assert s.status == "completed"

    def test_set_failed(self):
        s = AgentRunState()
        s.set_failed("timeout")
        assert s.status == "failed"
        assert "timeout" in s.errors

    def test_to_dict(self):
        s = AgentRunState(query="test")
        s.record_step("intent_detection")
        s.set_answer("42")
        d = s.to_dict()
        assert d["query"] == "test"
        assert d["steps"] == ["intent_detection"]
        assert d["final_answer"] == "42"
        assert d["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════
# AgentRouter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRouter:
    """Tests for the deterministic AgentRouter tool selection."""

    def setup_method(self):
        self.router = AgentRouter()

    # ── Fact questions → vector_search ─────────────────────────────────

    def test_fact_what_is(self):
        r = self.router.route("What is machine learning?")
        assert r.tools == ["vector_search"]
        assert r.category == QueryCategory.FACT

    def test_fact_who_is(self):
        r = self.router.route("Who is Alan Turing?")
        assert r.tools == ["vector_search"]

    def test_fact_define(self):
        r = self.router.route("Define entropy in information theory")
        assert r.tools == ["vector_search"]

    # ── Entity / relationship → graph_search ───────────────────────────

    def test_entity_relationship(self):
        r = self.router.route("What is the relationship between mitochondria and ATP?")
        assert r.tools == ["graph_search"]
        assert r.category == QueryCategory.ENTITY

    def test_entity_hierarchy(self):
        r = self.router.route("Explain the hierarchy of biological taxonomy")
        assert r.tools == ["graph_search"]

    def test_entity_network(self):
        r = self.router.route("How are neurons connected in a neural network?")
        assert r.tools == ["graph_search"]

    # ── Document summary → document_summary ────────────────────────────

    def test_summary_summarize(self):
        r = self.router.route("Summarize the key points of the Paris Agreement")
        assert "document_summary" in r.tools
        assert r.category == QueryCategory.DOCUMENT_SUMMARY

    def test_summary_overview(self):
        r = self.router.route("Give me an overview of quantum computing")
        assert "document_summary" in r.tools

    # ── Complex → vector_search + graph_search ─────────────────────────

    def test_complex_compare(self):
        r = self.router.route("Compare and contrast transformers vs RNNs")
        assert "vector_search" in r.tools
        assert "graph_search" in r.tools
        assert r.category == QueryCategory.COMPLEX

    def test_complex_analyze(self):
        r = self.router.route("Analyze the pros and cons of microservices architecture")
        assert r.tools == ["vector_search", "graph_search"]

    def test_complex_difference(self):
        r = self.router.route("What is the difference between DNA and RNA?")
        assert "vector_search" in r.tools
        assert "graph_search" in r.tools

    # ── Default ────────────────────────────────────────────────────────

    def test_default_no_keywords(self):
        r = self.router.route("Hello")
        assert r.tools == ["vector_search"]

    def test_returns_router_result(self):
        r = self.router.route("test query")
        assert isinstance(r, RouterResult)
        assert isinstance(r.category, QueryCategory)
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.tools, list)
        assert len(r.tools) >= 1

    # ── Priority: summary > entity > complex > fact ────────────────────

    def test_summary_priority(self):
        """Even if 'relationship' is in query, 'summarize' takes priority."""
        r = self.router.route("Summarize the relationship between A and B")
        assert r.tools == ["document_summary"]


# ═══════════════════════════════════════════════════════════════════════════
# AgentWorkflow Tests
# ═══════════════════════════════════════════════════════════════════════════


class _MockLLM:
    async def generate(self, prompt: str, context: str, **kw):
        return f"[LLM] Answer to query based on context ({len(context)} chars)"


class _MockRetriever:
    def retrieve(self, query: str, top_k: int = 5, **kw):
        return RAGContext(
            query=query,
            chunks=[RAGChunk(content=f"Chunk {i}: info about {query}", source=RAGSource(document_id=f"d{i}")) for i in range(3)],
        )
    @property
    def retriever_name(self): return "mock"


def _make_context_with_tools():
    """Build a RAGContext with mocked retriever, LLM, and tools."""
    tool_list = [
        VectorSearchTool(_MockRetriever()),
        DocumentSummaryTool(_MockLLM()),
        AnswerVerifyTool(),
        GraphSearchTool(),  # No graph — will return error gracefully
    ]

    return RAGContext(
        query="test",
        llm=_MockLLM(),
        retriever=_MockRetriever(),
        tools=tool_list,
    )


class TestAgentWorkflow:
    """Tests for the finite-state AgentWorkflow."""

    def setup_method(self):
        self.workflow = AgentWorkflow(max_steps=4)

    def test_default_max_steps(self):
        assert AgentWorkflow().max_steps == 4

    def test_custom_max_steps(self):
        wf = AgentWorkflow(max_steps=6)
        assert wf.max_steps == 6

    def test_max_steps_validation(self):
        with pytest.raises(ValueError, match="at least 2"):
            AgentWorkflow(max_steps=1)

    @pytest.mark.asyncio
    async def test_run_fact_query(self):
        """Full workflow for a simple fact query."""
        state = AgentRunState(query="What is machine learning?")
        ctx = _make_context_with_tools()

        result = await self.workflow.run(state, ctx)

        assert result.status == "completed"
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        # Steps must include: intent_detection, tool_selection, tool_execution:vector_search, generate, verify
        assert "intent_detection" in result.steps
        assert "tool_selection" in result.steps
        assert any("tool_execution" in s for s in result.steps)
        assert "generate" in result.steps
        assert "verify" in result.steps

    @pytest.mark.asyncio
    async def test_run_entity_query(self):
        """Entity query routes to graph_search."""
        state = AgentRunState(query="How are entities connected in the knowledge graph?")
        ctx = _make_context_with_tools()

        result = await self.workflow.run(state, ctx)

        assert result.status == "completed"
        assert "graph_search" in result.selected_tools

    @pytest.mark.asyncio
    async def test_tool_results_populated(self):
        """Tool results are stored in state after execution."""
        state = AgentRunState(query="What is RAG?")
        ctx = _make_context_with_tools()

        result = await self.workflow.run(state, ctx)

        # vector_search should have produced results
        assert "vector_search" in result.tool_results
        vs_result = result.tool_results["vector_search"]
        assert vs_result is not None

    @pytest.mark.asyncio
    async def test_max_steps_enforced(self):
        """Workflow terminates when max_steps is reached."""
        wf = AgentWorkflow(max_steps=2)
        # Force selected_tools to use many tools to exhaust budget
        state = AgentRunState(query="test", max_steps=2)

        # Override router: force 3 tools selected (should stop after 2 steps worth)
        class _FakeRouter:
            def route(self, q): return RouterResult(category=QueryCategory.FACT, tools=["vector_search", "graph_search", "document_summary"], confidence=0.5)
        wf._router = _FakeRouter()

        ctx = _make_context_with_tools()
        result = await wf.run(state, ctx)

        # Should have completed (generate + verify still work)
        assert result.status in ("completed", "failed")
        # Should not have executed all 3 tools
        tool_exec_count = sum(1 for s in result.steps if "tool_execution" in s)
        assert tool_exec_count <= 2

    @pytest.mark.asyncio
    async def test_no_llm_fallback(self):
        """Workflow works even without an LLM."""
        state = AgentRunState(query="test")
        ctx = _make_context_with_tools()
        ctx.llm = None  # Remove LLM

        result = await self.workflow.run(state, ctx)
        assert result.status in ("completed", "failed")
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_workflow_order(self):
        """Steps are recorded in the correct order."""
        state = AgentRunState(query="What is Python?")
        ctx = _make_context_with_tools()

        result = await self.workflow.run(state, ctx)

        # Check order
        step_names = result.steps
        idx_intent = step_names.index("intent_detection")
        idx_select = step_names.index("tool_selection")
        idx_generate = step_names.index("generate")
        idx_verify = step_names.index("verify")

        assert idx_intent < idx_select < idx_generate < idx_verify

    @pytest.mark.asyncio
    async def test_verify_checks_answer(self):
        """The verify step runs answer verification."""
        state = AgentRunState(query="test")
        ctx = _make_context_with_tools()

        result = await self.workflow.run(state, ctx)

        # verify step should have been recorded
        assert "verify" in result.steps
        # answer_verify results should be in tool_results
        assert "answer_verify" in result.tool_results


class TestAgentWorkflowNoTools:
    """Workflow behaviour when no tools are available."""

    @pytest.mark.asyncio
    async def test_no_executor_graceful(self):
        wf = AgentWorkflow()
        state = AgentRunState(query="test")
        ctx = RAGContext(query="test")  # No tools configured

        result = await wf.run(state, ctx)

        # Should complete (generate + verify run)
        assert result.status in ("completed", "failed")
        # Should note the lack of tools
        assert "No ToolExecutor" in result.errors or result.final_answer is not None
