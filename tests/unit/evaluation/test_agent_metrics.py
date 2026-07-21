from src.evaluation.agent_metrics import (
    infer_expected_tools,
    summarize_agent_metrics,
    tool_selection_accuracy,
)


def test_infer_expected_tools_from_answer_points():
    tools = infer_expected_tools(
        ["The expected tool is vector_search.", "Use graph search if needed."]
    )
    assert "vector_search" in tools
    assert "graph_search" in tools


def test_tool_selection_accuracy():
    results = [
        {
            "id": "a1",
            "agent": {
                "tools_selected": ["vector_search"],
                "expected_tools": ["vector_search"],
            },
        },
        {
            "id": "a2",
            "agent": {
                "tools_selected": ["route_agent"],
                "expected_tools": ["vector_search"],
            },
        },
    ]
    assert tool_selection_accuracy(results) == 0.5


def test_summarize_agent_metrics_allows_null_accuracy():
    summary = summarize_agent_metrics([{"id": "x", "agent": {}}])
    assert summary["tool_selection_accuracy"] is None
