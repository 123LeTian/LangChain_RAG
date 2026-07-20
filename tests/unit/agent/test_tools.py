"""
Tests for src/agent/tools.py — BaseTool, ToolRegistry, ToolExecutor, concrete tools.
"""

import asyncio
import pytest

from src.agent.tools import (
    BaseTool,
    CalculatorTool,
    DirectAnswerTool,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


# ── Fake tool for testing ──────────────────────────────────────────────────


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the input."

    async def execute(self, message: str = "", **kwargs):
        return ToolResult(success=True, data={"echo": message}, tool_name=self.name)

    def _parameter_schema(self):
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }


class _FailingTool(BaseTool):
    name = "failing"
    description = "Always fails."

    async def execute(self, **kwargs):
        raise RuntimeError("deliberate failure")


class _SlowTool(BaseTool):
    name = "slow"
    description = "Always times out."

    async def execute(self, **kwargs):
        await asyncio.sleep(10)
        return ToolResult(success=True)


# ── Tests ──────────────────────────────────────────────────────────────────


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_get(self):
        tool = _EchoTool()
        self.registry.register(tool)
        assert "echo" in self.registry
        assert self.registry.get("echo") is tool

    def test_get_missing_raises(self):
        with pytest.raises(KeyError) as exc:
            self.registry.get("nonexistent")
        assert "nonexistent" in str(exc.value)

    def test_unregister(self):
        tool = _EchoTool()
        self.registry.register(tool)
        self.registry.unregister("echo")
        assert "echo" not in self.registry

    def test_list_names(self):
        self.registry.register(_EchoTool())
        self.registry.register(CalculatorTool())
        names = self.registry.list_names()
        assert "echo" in names
        assert "calculator" in names

    def test_list_tools(self):
        self.registry.register(_EchoTool())
        tools = self.registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    def test_len(self):
        assert len(self.registry) == 0
        self.registry.register(_EchoTool())
        assert len(self.registry) == 1

    def test_to_openai_functions(self):
        self.registry.register(_EchoTool())
        funcs = self.registry.to_openai_functions()
        assert len(funcs) == 1
        assert funcs[0]["type"] == "function"
        assert funcs[0]["function"]["name"] == "echo"
        assert "message" in funcs[0]["function"]["parameters"]["required"]

    def test_to_anthropic_tools(self):
        self.registry.register(_EchoTool())
        tools = self.registry.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        assert "input_schema" in tools[0]


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(_EchoTool())
        self.registry.register(_FailingTool())

    @pytest.mark.asyncio
    async def test_execute_success(self):
        executor = ToolExecutor(self.registry)
        result = await executor.execute("echo", message="hello")
        assert result.success
        assert result.data["echo"] == "hello"
        assert result.tool_name == "echo"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_success_no_retry_on_success(self):
        executor = ToolExecutor(self.registry, max_retries=3)
        result = await executor.execute("echo", message="world")
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        executor = ToolExecutor(self.registry, max_retries=0)
        result = await executor.execute("failing")
        assert not result.success
        assert "deliberate failure" in result.error
        assert result.tool_name == "failing"

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self):
        executor = ToolExecutor(self.registry, max_retries=2)
        result = await executor.execute("failing")
        assert not result.success
        # Should have tried 3 times total (1 initial + 2 retries)

    @pytest.mark.asyncio
    async def test_execute_missing_tool_raises(self):
        executor = ToolExecutor(self.registry)
        with pytest.raises(KeyError):
            await executor.execute("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        self.registry.register(CalculatorTool())
        executor = ToolExecutor(self.registry)
        results = await executor.execute_batch([
            ("echo", {"message": "a"}),
            ("echo", {"message": "b"}),
        ])
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].data["echo"] == "a"
        assert results[1].data["echo"] == "b"


class TestConcreteTools:
    """Tests for each built-in tool."""

    @pytest.mark.asyncio
    async def test_calculator_basic(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3")
        assert result.success
        assert result.data["result"] == 5

    @pytest.mark.asyncio
    async def test_calculator_sqrt(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="sqrt(16)")
        assert result.success
        assert result.data["result"] == 4.0

    @pytest.mark.asyncio
    async def test_calculator_invalid(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="__import__('os')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_direct_answer_tool(self):
        tool = DirectAnswerTool()
        result = await tool.execute(question="What is Python?")
        assert result.success
        assert result.data["question"] == "What is Python?"
        assert result.data["action"] == "generate_directly"

    def test_calculator_openai_function(self):
        tool = CalculatorTool()
        func = tool.to_openai_function()
        assert func["function"]["name"] == "calculator"
        assert "expression" in func["function"]["parameters"]["required"]

    def test_direct_answer_openai_function(self):
        tool = DirectAnswerTool()
        func = tool.to_openai_function()
        assert func["function"]["name"] == "direct_answer"
