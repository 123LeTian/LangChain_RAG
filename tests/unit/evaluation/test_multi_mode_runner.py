import asyncio
import json

import pytest

from src.evaluation.dataset import EvaluationSample, ExpectedSource
from src.evaluation.multi_mode_runner import (
    ALL_EVAL_MODES,
    EvaluationModeRunner,
    MockModeRunner,
    MultiModeEvaluationRunner,
    create_runner,
)


def sample():
    return EvaluationSample(
        id="eval_test",
        question="What is RAG?",
        answer_points=["Retrieval plus generation."],
        expected_sources=[
            ExpectedSource(document_id="doc-1", chunk_id="chunk-1", filename="rag.md")
        ],
        tags=["fact", "agent_route"],
        difficulty="easy",
    )


def write_dataset(path):
    path.write_text(json.dumps(sample().to_dict()) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mode", ALL_EVAL_MODES)
def test_mock_mode_runner_runs(mode):
    result = asyncio.run(MockModeRunner(mode).run_sample(sample()))
    assert result["hit_at_1"] == 1.0
    assert "answer_point_coverage" in result
    if mode == "graph":
        assert result["hits"][0].get("graph_path")
    if mode == "agentic":
        assert "agent" in result


def test_partial_modes(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    write_dataset(dataset_path)

    report = asyncio.run(
        MultiModeEvaluationRunner(
            dataset_path=str(dataset_path),
            modes=["naive", "graph"],
            runner_factory=lambda mode: create_runner(mode, use_mock=True),
        ).run()
    )
    assert len(report["per_mode_results"]) == 2


def test_failed_mode_factory_does_not_break_report(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    write_dataset(dataset_path)

    def factory(mode: str):
        if mode == "graph":
            raise RuntimeError("graph unavailable")
        return create_runner(mode, use_mock=True)

    report = asyncio.run(
        MultiModeEvaluationRunner(
            dataset_path=str(dataset_path),
            modes=["naive", "graph"],
            runner_factory=factory,
        ).run()
    )
    modes = {item["mode"]: item for item in report["per_mode_results"]}
    assert modes["naive"]["status"] == "ok"
    assert modes["graph"]["status"] == "unavailable"


def test_evaluation_mode_runner_aggregates_system_metrics():
    runner = MockModeRunner("advanced")
    summary = asyncio.run(
        EvaluationModeRunner(runner).run([sample()])
    )
    for field in (
        "p50_latency_ms",
        "p95_latency_ms",
        "average_token_count",
        "citation_presence_rate",
    ):
        assert field in summary
