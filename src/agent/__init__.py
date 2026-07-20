# Agent — Owner: C
#
# Agentic RAG subsystem: state management, tool execution, query routing,
# error degradation, and LangGraph workflow.
#
# Exports:
#   - AgentState and state helpers
#   - Tool definitions, registry, and executor
#   - QueryRouter, CircuitBreaker, FallbackChain
#   - LangGraph workflow nodes and graph builder

from src.agent.state import (
    AgentAction,
    AgentState,
    AgentStatus,
    AgentStep,
    initial_state,
    is_finished,
    set_answer,
    set_error,
    state_snapshot,
)
from src.agent.tools import (
    BaseTool,
    CalculatorTool,
    DirectAnswerTool,
    GraphSearchTool,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    VectorSearchTool,
    create_default_tools,
)
from src.agent.router import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackChain,
    FallbackEntry,
    QueryRouter,
    RouteDecision,
    create_router_and_fallback,
    default_fallback_chain,
)
from src.agent.workflow import (
    build_agent_graph,
    generate_node,
    get_agent_graph,
    plan_node,
    reason_node,
    reflect_node,
    retrieve_node,
    should_continue,
    tool_call_node,
)

__all__ = [
    # State
    "AgentState",
    "AgentStatus",
    "AgentAction",
    "AgentStep",
    "initial_state",
    "state_snapshot",
    "is_finished",
    "set_answer",
    "set_error",
    # Tools
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    "VectorSearchTool",
    "GraphSearchTool",
    "CalculatorTool",
    "DirectAnswerTool",
    "create_default_tools",
    # Router
    "QueryRouter",
    "RouteDecision",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "FallbackChain",
    "FallbackEntry",
    "default_fallback_chain",
    "create_router_and_fallback",
    # Workflow
    "build_agent_graph",
    "get_agent_graph",
    "plan_node",
    "retrieve_node",
    "reason_node",
    "tool_call_node",
    "generate_node",
    "reflect_node",
    "should_continue",
]
