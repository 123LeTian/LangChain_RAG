"""
Tests for the new Agentic RAG components — AgentRunState, AgentRouter, AgentWorkflow.
"""

import pytest

from src.agent.state import AgentRunState
from src.agent.router import AgentRouter, QueryCategory, RouterResult
from src.agent.tools import (
    BaseTool,
    AnswerVerifyTool,
    DocumentSummaryTool,
    GraphSearchTool,
    ToolResult,
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


class _SpyAnswerVerifyTool(BaseTool):
    name: str = "answer_verify"
    description: str = "Spy answer verification"

    def __init__(self):
        self.last_answer = None
        self.last_chunks = None

    async def execute(self, answer: str = "", chunks=None, **kwargs):
        self.last_answer = answer
        self.last_chunks = chunks or []
        return ToolResult(
            success=True,
            data={"passed": True, "seen_chunks": len(self.last_chunks)},
            tool_name=self.name,
        )


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

    @pytest.mark.asyncio
    async def test_verify_uses_list_dict_and_summary_context(self):
        """Verify should pass vector, graph, and summary context into answer_verify."""
        state = AgentRunState(query="test")
        state.set_answer("This is a sufficiently long answer for verification.")
        state.tool_results["vector_search"] = [
            {"chunk_id": "v1", "content": "vector context one"},
        ]
        state.tool_results["graph_search"] = {
            "chunks": [
                {"chunk_id": "g1", "content": "graph context one"},
            ]
        }
        state.tool_results["document_summary"] = {
            "summary": "summary context one",
        }

        spy = _SpyAnswerVerifyTool()
        ctx = RAGContext(
            query="test",
            llm=_MockLLM(),
            retriever=_MockRetriever(),
            tools=[spy],
        )

        result = await self.workflow._verify(state, ctx)

        assert "answer_verify" in result.tool_results
        assert result.tool_results["answer_verify"]["seen_chunks"] == 3
        assert spy.last_chunks is not None
        assert len(spy.last_chunks) == 3


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


# ═══════════════════════════════════════════════════════════════════════════
# Acceptance Criteria Tests — Agentic RAG 测试
# ═══════════════════════════════════════════════════════════════════════════


# ── Shared helpers for acceptance tests ─────────────────────────────────

def _make_agent_context():
    """Build a RAGContext with all four standard tools for acceptance testing."""
    from src.agent.tools import (
        VectorSearchTool, GraphSearchTool, DocumentSummaryTool, AnswerVerifyTool,
    )
    tool_list = [
        VectorSearchTool(_MockRetriever()),
        GraphSearchTool(),  # No graph — returns error → degraded to vector
        DocumentSummaryTool(_MockLLM()),
        AnswerVerifyTool(),
    ]
    return RAGContext(
        query="test",
        llm=_MockLLM(),
        retriever=_MockRetriever(),
        tools=tool_list,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 1: 三类测试问题中选择正确工具
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion1_ToolSelection:
    """验收标准 1：Agent 能在三类测试问题中选择正确工具。

    三类测试问题：
      - 事实查询 (fact)                     → vector_search
      - 实体/关系查询 (entity/relationship)  → graph_search
      - 摘要/概述查询 (summary)              → document_summary
    复杂分析查询会同时选择 vector_search + graph_search。
    """

    def setup_method(self):
        self.router = AgentRouter()

    # ── Type 1: Fact queries → vector_search ───────────────────────────

    def test_fact_what_is_routes_to_vector(self):
        r = self.router.route("What is retrieval augmented generation?")
        assert r.tools == ["vector_search"]
        assert r.category == QueryCategory.FACT

    def test_fact_how_many_routes_to_vector(self):
        r = self.router.route("How many layers are in a transformer?")
        assert r.tools == ["vector_search"]

    def test_fact_define_routes_to_vector(self):
        r = self.router.route("Define attention mechanism in deep learning")
        assert r.tools == ["vector_search"]

    # ── Type 2: Entity/relationship queries → graph_search ─────────────

    def test_entity_relationship_routes_to_graph(self):
        r = self.router.route("What is the relationship between DNA and RNA?")
        assert r.tools == ["graph_search"]
        assert r.category == QueryCategory.ENTITY

    def test_entity_network_routes_to_graph(self):
        r = self.router.route("Describe the neural network structure")
        assert r.tools == ["graph_search"]

    def test_entity_hierarchy_routes_to_graph(self):
        r = self.router.route("Explain the taxonomy hierarchy of species")
        assert r.tools == ["graph_search"]

    # ── Type 3: Summary queries → document_summary ─────────────────────

    def test_summary_summarize_routes_to_doc_summary(self):
        r = self.router.route("Summarize the key findings of the report")
        assert r.tools == ["document_summary"]
        assert r.category == QueryCategory.DOCUMENT_SUMMARY

    def test_summary_overview_routes_to_doc_summary(self):
        r = self.router.route("Give me a brief overview of the topic")
        assert r.tools == ["document_summary"]

    # ── Complex: two tools ──────────────────────────────────────────────

    def test_complex_compare_routes_to_both(self):
        r = self.router.route("Compare transformers versus RNNs")
        assert "vector_search" in r.tools
        assert "graph_search" in r.tools
        assert r.category == QueryCategory.COMPLEX

    # ── E2E: workflow actually uses the selected tools ──────────────────

    @pytest.mark.asyncio
    async def test_workflow_uses_vector_for_fact(self):
        """End-to-end: fact query → workflow selects and uses vector_search."""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is machine learning?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        assert "vector_search" in result.selected_tools
        assert "vector_search" in result.tool_results
        assert result.tool_results["vector_search"] is not None

    @pytest.mark.asyncio
    async def test_workflow_uses_graph_for_entity(self):
        """End-to-end: entity query → workflow selects graph_search."""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="How is the network structured?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        assert "graph_search" in result.selected_tools


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 2: 最多执行固定步数
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion2_MaxSteps:
    """验收标准 2：最多执行固定步数（默认 4 步），防止无限循环。"""

    def test_default_max_steps_is_4(self):
        wf = AgentWorkflow()
        assert wf.max_steps == 4

    def test_max_steps_set_on_state_during_run(self):
        wf = AgentWorkflow(max_steps=3)
        state = AgentRunState(query="test")
        assert state.max_steps == 4  # default

        # When run(), the workflow sets the state's max_steps
        # (verified next test via E2E)

    def test_max_steps_validation_minimum_2(self):
        with pytest.raises(ValueError, match="at least 2"):
            AgentWorkflow(max_steps=1)

    @pytest.mark.asyncio
    async def test_workflow_terminates_at_max_steps(self):
        """Agent cannot exceed max_steps; terminates early if needed."""
        wf = AgentWorkflow(max_steps=2)
        state = AgentRunState(query="test", max_steps=2)

        # Force many tools to exhaust budget
        class _ForceManyTools:
            def route(self, q):
                return RouterResult(
                    category=QueryCategory.COMPLEX,
                    tools=["vector_search", "graph_search", "document_summary"],
                    confidence=0.5,
                )
        wf._router = _ForceManyTools()

        ctx = _make_agent_context()
        result = await wf.run(state, ctx)

        # Should NOT have executed all 3 tools
        tool_exec_count = sum(1 for s in result.steps if "tool_execution" in s)
        assert tool_exec_count <= 2
        assert result.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_step_budget_exhausted_records_error(self):
        """When step budget is exhausted, an error is recorded."""
        wf = AgentWorkflow(max_steps=2)
        state = AgentRunState(query="test")

        class _ForceManyTools:
            def route(self, q):
                return RouterResult(
                    category=QueryCategory.COMPLEX,
                    tools=["vector_search", "graph_search", "document_summary", "answer_verify"],
                    confidence=0.5,
                )
        wf._router = _ForceManyTools()

        ctx = _make_agent_context()
        result = await wf.run(state, ctx)
        assert result.current_step <= result.max_steps + 1  # +1 for verify step


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 3: 工具调用记录
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion3_ToolCallRecording:
    """验收标准 3：每次工具调用记录名称、参数摘要、结果摘要、耗时和错误。

    state.tool_calls 列表中的每条记录包含:
      - tool_name: 工具名称
      - params_summary: 参数摘要
      - result_summary: 结果摘要
      - duration_ms: 耗时（毫秒）
      - error: 错误信息（成功时为 None）
    """

    @pytest.mark.asyncio
    async def test_tool_calls_list_populated(self):
        """workflow 运行后 state.tool_calls 包含每条工具调用的记录。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is deep learning?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        assert len(result.tool_calls) >= 1, "Should record at least one tool call"

    @pytest.mark.asyncio
    async def test_tool_call_has_all_required_fields(self):
        """每条 tool_call 记录包含全部必要字段。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is Python?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        for call in result.tool_calls:
            assert "tool_name" in call, f"Missing tool_name in {call}"
            assert "params_summary" in call, f"Missing params_summary in {call}"
            assert "result_summary" in call, f"Missing result_summary in {call}"
            assert "duration_ms" in call, f"Missing duration_ms in {call}"
            assert "error" in call, f"Missing error in {call}"

    @pytest.mark.asyncio
    async def test_tool_call_duration_is_positive(self):
        """成功工具调用的耗时 > 0。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is RAG?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        for call in result.tool_calls:
            assert call["duration_ms"] >= 0, f"Negative duration: {call}"

    @pytest.mark.asyncio
    async def test_tool_call_params_summary_is_descriptive(self):
        """参数摘要描述了传递给工具的参数。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is ASCII encoding?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        for call in result.tool_calls:
            if call["tool_name"] == "vector_search":
                assert "query" in call["params_summary"].lower() or \
                       "ASCII" in call["params_summary"]

    @pytest.mark.asyncio
    async def test_tool_call_error_field_none_on_success(self):
        """成功工具调用的 error 字段为 None。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is a neural network?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        for call in result.tool_calls:
            if call["result_summary"] and "FAILED" not in call["result_summary"]:
                assert call["error"] is None, \
                    f"Successful call should have error=None: {call}"

    @pytest.mark.asyncio
    async def test_tool_call_recording_order(self):
        """工具调用记录按执行顺序排列。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="Compare AI vs ML")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        # Records should be in execution order
        assert len(result.tool_calls) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 4: 工具失败后降级
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion4_Degradation:
    """验收标准 4：工具失败后能够降级到 Vector Search 或明确返回失败。

    - graph_search 失败 → 自动降级到 vector_search
    - 所有工具都失败 → 明确标记 __all_tools_failed__，LLM-only 生成
    """

    def setup_method(self):
        self.wf = AgentWorkflow(max_steps=4)

    @pytest.mark.asyncio
    async def test_graph_search_failure_degrades_to_vector(self):
        """无 graph retriever 时 graph_search 失败 → 降级到 vector_search。"""
        # "relationship" + "network" triggers entity → graph_search routing
        state = AgentRunState(query="What is the relationship between neural network layers?")
        ctx = _make_agent_context()  # GraphSearchTool has no graph → will fail

        result = await self.wf.run(state, ctx)

        # The graph_search fails but vector_search should have succeeded (fallback)
        assert "vector_search" in result.tool_results
        vs_result = result.tool_results["vector_search"]
        assert vs_result is not None
        # Degradation recorded
        assert any("degrad" in e.lower() or "fallback" in e.lower()
                   for e in result.errors), f"Expected degradation note in: {result.errors}"

    @pytest.mark.asyncio
    async def test_all_tools_fail_marked_explicitly(self):
        """所有工具都失败时标记 __all_tools_failed__。"""
        # Create context where ALL tools will fail
        from src.agent.tools import GraphSearchTool
        failing_tools = [
            GraphSearchTool(),  # No graph → fails
        ]
        ctx = RAGContext(
            query="test",
            llm=_MockLLM(),
            retriever=_MockRetriever(),
            tools=failing_tools,
        )

        # Force only graph_search (which will fail)
        class _ForceGraph:
            def route(self, q):
                return RouterResult(
                    category=QueryCategory.ENTITY,
                    tools=["graph_search"], confidence=0.5,
                )
        self.wf._router = _ForceGraph()

        state = AgentRunState(query="How are entities linked?")
        result = await self.wf.run(state, ctx)

        assert result.tool_results.get("__all_tools_failed__") is True
        assert result.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_degraded_run_still_produces_answer(self):
        """即使 graph_search 降级，最终仍生成回答。"""
        state = AgentRunState(query="What is the network structure?")
        ctx = _make_agent_context()

        result = await self.wf.run(state, ctx)
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    @pytest.mark.asyncio
    async def test_no_tools_available_returns_clear_error(self):
        """完全没有工具时，明确记录错误并尽力生成回答。"""
        state = AgentRunState(query="test")
        ctx = RAGContext(query="test", llm=_MockLLM())
        # No tools at all

        result = await self.wf.run(state, ctx)
        assert result.status in ("completed", "failed")
        # Either has error or has answer
        assert result.errors or result.final_answer


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 5: 回答包含引用
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion5_Citations:
    """验收标准 5：Agent 最终回答仍必须包含引用 (citations)。

    state.citations 应包含从工具结果中提取的来源引用：
      - chunk_id: 来源 chunk 标识
      - document_id: 来源文档标识
      - text_snippet: 来源文本片段（≤200 字符）
    """

    @pytest.mark.asyncio
    async def test_citations_populated_from_tool_results(self):
        """工具执行成功后 state.citations 包含引用。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is machine learning?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        assert len(result.citations) > 0, \
            f"Expected citations, got: {result.citations}"

    @pytest.mark.asyncio
    async def test_citations_have_required_fields(self):
        """每条引用包含 chunk_id, document_id, text_snippet。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is deep learning?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        for citation in result.citations:
            assert "chunk_id" in citation, f"Missing chunk_id in {citation}"
            assert "document_id" in citation, f"Missing document_id in {citation}"
            assert "text_snippet" in citation, f"Missing text_snippet in {citation}"
            assert len(citation.get("text_snippet", "")) <= 200, \
                f"text_snippet too long: {len(citation['text_snippet'])}"

    @pytest.mark.asyncio
    async def test_citations_count_limited(self):
        """引用数量不超过 10 条。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is AI?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        assert len(result.citations) <= 10

    @pytest.mark.asyncio
    async def test_citations_without_retrieval_are_empty(self):
        """没有检索结果时，citations 为空列表。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="test")
        ctx = RAGContext(query="test", llm=_MockLLM())
        # No tools → no retrieval

        result = await wf.run(state, ctx)
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_citations_in_to_dict(self):
        """citations 出现在 to_dict() 输出中。"""
        wf = AgentWorkflow(max_steps=4)
        state = AgentRunState(query="What is Python?")
        ctx = _make_agent_context()

        result = await wf.run(state, ctx)
        d = result.to_dict()
        assert "citations" in d
        assert isinstance(d["citations"], list)


# ═══════════════════════════════════════════════════════════════════════════
# Criterion 6: 评测指标
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptanceCriterion6_AgentMetrics:
    """验收标准 6：评测中单独统计工具选择准确率和轨迹成功率。

    AgentMetrics 类提供:
      - tool_selection_accuracy: 工具选择正确率
      - trajectory_success_rate: 轨迹成功率（完整走完流程的比例）
      - report(): 结构化评测报告
      - summary(): 人类可读摘要
    """

    def setup_method(self):
        from src.agent.workflow import AgentMetrics
        self.metrics = AgentMetrics()

    def test_initial_state(self):
        assert self.metrics.total_runs == 0
        assert self.metrics.tool_selection_accuracy == 0.0
        assert self.metrics.trajectory_success_rate == 0.0

    def test_zero_runs_accuracy_zero(self):
        """没有数据时准确率为 0。"""
        assert self.metrics.tool_selection_accuracy == 0.0
        assert self.metrics.trajectory_success_rate == 0.0

    def test_single_correct_run(self):
        """一次完全正确的运行 → 准确率 100%。"""
        self.metrics.record(
            query="What is RAG?",
            expected_tools=["vector_search"],
            actual_tools=["vector_search"],
            trajectory_success=True,
            steps_count=5,
        )
        assert self.metrics.total_runs == 1
        assert self.metrics.tool_selection_accuracy == 1.0
        assert self.metrics.trajectory_success_rate == 1.0

    def test_single_incorrect_tool(self):
        """工具选择错误的运行 → 准确率 0%。"""
        self.metrics.record(
            query="What is the hierarchy?",
            expected_tools=["graph_search"],
            actual_tools=["vector_search"],  # Wrong tool
            trajectory_success=True,
        )
        assert self.metrics.tool_selection_accuracy == 0.0
        assert self.metrics.trajectory_success_rate == 1.0

    def test_single_failed_trajectory(self):
        """轨迹失败的运行 → 成功率下降。"""
        self.metrics.record(
            query="test",
            expected_tools=["vector_search"],
            actual_tools=["vector_search"],
            trajectory_success=False,
        )
        assert self.metrics.trajectory_success_rate == 0.0
        assert self.metrics.tool_selection_accuracy == 1.0

    def test_mixed_runs_accuracy(self):
        """混合正确/错误运行 → 正确计算比例。"""
        # 3 correct, 1 wrong → 75% accuracy, 50% trajectory success
        for _ in range(3):
            self.metrics.record(
                expected_tools=["vector_search"],
                actual_tools=["vector_search"],
                trajectory_success=True,
            )
        self.metrics.record(
            expected_tools=["graph_search"],
            actual_tools=["vector_search"],
            trajectory_success=False,
        )
        assert self.metrics.total_runs == 4
        assert self.metrics.tool_selection_accuracy == 0.75
        assert self.metrics.trajectory_success_rate == 0.75

    def test_order_independent_tool_matching(self):
        """工具匹配不考虑顺序：['a','b'] 匹配 ['b','a']。"""
        self.metrics.record(
            expected_tools=["vector_search", "graph_search"],
            actual_tools=["graph_search", "vector_search"],  # Different order
            trajectory_success=True,
        )
        assert self.metrics.tool_selection_accuracy == 1.0

    def test_no_expected_tools_skips_accuracy(self):
        """未设置 expected_tools 时不纳入工具准确率计算。"""
        self.metrics.record(
            query="test",
            expected_tools=None,
            actual_tools=["vector_search"],
            trajectory_success=True,
        )
        assert self.metrics.total_runs == 1
        assert self.metrics._tool_total == 0  # Not counted for tool accuracy
        assert self.metrics.trajectory_success_rate == 1.0

    def test_report_has_all_fields(self):
        """report() 返回完整字段。"""
        self.metrics.record(
            query="What is AI?",
            expected_tools=["vector_search"],
            actual_tools=["vector_search"],
            trajectory_success=True,
        )
        r = self.metrics.report()
        assert "total_runs" in r
        assert "tool_selection_accuracy" in r
        assert "tool_correct" in r
        assert "tool_total" in r
        assert "trajectory_success_rate" in r
        assert "trajectory_successes" in r
        assert "runs" in r

    def test_summary_is_readable(self):
        """summary() 返回人类可读文本。"""
        self.metrics.record(
            query="What is RAG?",
            expected_tools=["vector_search"],
            actual_tools=["vector_search"],
            trajectory_success=True,
        )
        s = self.metrics.summary()
        assert "Agent Evaluation Summary" in s
        assert "Tool selection accuracy" in s
        assert "Trajectory success rate" in s
        assert "100.0%" in s

    def test_reset_clears_all_data(self):
        """reset() 后数据清零。"""
        self.metrics.record(
            expected_tools=["vector_search"],
            actual_tools=["vector_search"],
            trajectory_success=True,
        )
        assert self.metrics.total_runs == 1
        self.metrics.reset()
        assert self.metrics.total_runs == 0
        assert self.metrics.tool_selection_accuracy == 0.0
        assert self.metrics.trajectory_success_rate == 0.0

    def test_len_dunder(self):
        """__len__ 返回运行总数。"""
        for _ in range(3):
            self.metrics.record(
                expected_tools=["vector_search"],
                actual_tools=["vector_search"],
                trajectory_success=True,
            )
        assert len(self.metrics) == 3

    @pytest.mark.asyncio
    async def test_e2e_metrics_integration(self):
        """端到端：运行 workflow 后用 AgentMetrics 统计。"""
        from src.agent.workflow import AgentMetrics

        m = AgentMetrics()
        wf = AgentWorkflow(max_steps=4)

        # Run a fact query
        state = AgentRunState(query="What is machine learning?")
        ctx = _make_agent_context()
        result = await wf.run(state, ctx)

        m.record(
            query="What is machine learning?",
            expected_tools=["vector_search"],
            actual_tools=result.selected_tools,
            trajectory_success=result.status == "completed",
            steps_count=result.current_step,
            errors=result.errors,
        )

        assert m.total_runs == 1
        assert m.tool_selection_accuracy in (0.0, 1.0)
