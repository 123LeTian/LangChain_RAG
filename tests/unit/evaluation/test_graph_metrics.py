from src.evaluation.graph_metrics import (
    community_report_usage_rate,
    graph_path_presence_rate,
    graph_source_trace_rate,
    summarize_graph_metrics,
)


def test_graph_metrics_detect_trace_path_and_community_usage():
    results = [
        {
            "hits": [
                {
                    "retriever": "graph_local",
                    "graph_path": ["entity:a", "relationship:links"],
                    "metadata": {"graph_source_trace": True},
                }
            ]
        },
        {
            "hits": [
                {
                    "retriever": "graph_global",
                    "metadata": {
                        "graph_scope": "global",
                        "community_report_used": True,
                    },
                }
            ]
        },
    ]

    assert graph_source_trace_rate(results) == 1.0
    assert graph_path_presence_rate(results) == 0.5
    assert community_report_usage_rate(results) == 0.5
    assert summarize_graph_metrics(results) == {
        "graph_source_trace_rate": 1.0,
        "graph_path_presence_rate": 0.5,
        "community_report_usage_rate": 0.5,
    }
