# Agent — Owner: C
#
# Agentic RAG subsystem: state management, tool interfaces, query routing,
# error degradation, and finite-state workflow.
#
# Exports:
#   - AgentRunState + legacy AgentState
#   - Four tool interfaces + ToolRegistry + ToolExecutor
#   - AgentRouter (deterministic) + legacy QueryRouter + CircuitBreaker
#   - AgentWorkflow (finite-state) + legacy LangGraph nodes

from src.agent.state import (
    AgentAction,
    AgentRunState,
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
from src.agent.tools import (
    AnswerVerifyTool,
    BaseTool,
    DocumentSummaryTool,
    GraphSearchTool,
    ToolExecutor,
    ToolProtocol,
    ToolRegistry,
    ToolResult,
    VectorSearchTool,
    create_default_tools,
)
from src.agent.router import (
    AgentRouter,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackChain,
    FallbackEntry,
    QueryCategory,
    QueryRouter,
    RouteDecision,
    RouterResult,
    create_router_and_fallback,
    default_fallback_chain,
)
from src.agent.workflow import (
    AgentWorkflow,
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
    "AgentRunState",
    "AgentState",
    "AgentStatus",
    "AgentAction",
    "AgentStep",
    "initial_state",
    "state_snapshot",
    "is_finished",
    "set_answer",
    "set_error",
    "append_step",
    # Tools
    "ToolProtocol",
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    "VectorSearchTool",
    "GraphSearchTool",
    "DocumentSummaryTool",
    "AnswerVerifyTool",
    "create_default_tools",
    # Router
    "AgentRouter",
    "QueryCategory",
    "RouterResult",
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
    "AgentWorkflow",
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
