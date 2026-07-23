import asyncio
import json
from types import SimpleNamespace

import pytest

from src.api.api_models import RAGMode
from src.evaluation.dataset import EvaluationSample, ExpectedSource
from src.evaluation.multi_mode_runner import (
    ALL_EVAL_MODES,
    EvaluationModeRunner,
    MockModeRunner,
    MultiModeEvaluationRunner,
    build_chunk_evaluation_samples,
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


def test_multi_mode_runner_limits_samples(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    rows = []
    for index in range(4):
        item = EvaluationSample(
            id=f"eval_test_{index}",
            question="What is RAG?",
            answer_points=["Retrieval plus generation."],
            expected_sources=[
                ExpectedSource(document_id="doc-1", chunk_id="chunk-1", filename="rag.md")
            ],
            tags=["fact"],
            difficulty="easy",
        )
        rows.append(json.dumps(item.to_dict()))
    dataset_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = asyncio.run(
        MultiModeEvaluationRunner(
            dataset_path=str(dataset_path),
            modes=["naive"],
            sample_limit=2,
            runner_factory=lambda mode: create_runner(mode, use_mock=True),
        ).run()
    )

    assert report["sample_limit"] == 2
    assert report["sample_count"] == 2
    assert report["per_mode_results"][0]["sample_count"] == 2


def test_build_chunk_evaluation_samples_uses_real_chunk_sources():
    chunk = SimpleNamespace(
        id="doc-1:chunk-0",
        document_id="doc-1",
        kb_id="kb-demo",
        text="Alpha beta gamma. This chunk contains the source truth.",
        metadata={"filename": "demo.md"},
    )

    samples = build_chunk_evaluation_samples([chunk], sample_limit=1)

    assert len(samples) == 1
    assert samples[0].metadata["kb_id"] == "kb-demo"
    assert samples[0].expected_sources[0].chunk_id == "doc-1:chunk-0"
    assert samples[0].expected_sources[0].filename == "demo.md"


def test_multi_mode_runner_runs_modes_concurrently(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    write_dataset(dataset_path)
    state = {"active": 0, "max_active": 0}

    class SlowModeRunner:
        def __init__(self, mode: str):
            self.mode = mode
            self.runner_type = f"slow_{mode}"

        async def run_sample(self, sample):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return {
                "id": sample.id,
                "question": sample.question,
                "answer": "Retrieval plus generation.",
                "hits": [],
                "citations": [],
                "latency_ms": 10.0,
                "token_usage": {"total": 1},
                "warnings": [],
                "hit_at_1": 1.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "reciprocal_rank": 1.0,
            }

    report = asyncio.run(
        MultiModeEvaluationRunner(
            dataset_path=str(dataset_path),
            modes=["naive", "advanced"],
            runner_factory=lambda mode: SlowModeRunner(mode),
        ).run()
    )

    assert [item["mode"] for item in report["per_mode_results"]] == ["naive", "advanced"]
    assert state["max_active"] == 2


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


def test_real_mode_runner_passes_shared_model_config_to_rag_service():
    class CapturingService:
        def __init__(self):
            self.request = None

        async def query(self, request):
            self.request = request
            return SimpleNamespace(
                answer="Retrieval plus generation.",
                hits=[],
                citations=[],
                usage={"total": 7},
                warnings=[],
            )

    service = CapturingService()
    model_config = {"id": "model-abc", "provider": "deepseek", "model_name": "deepseek-v4-pro"}

    result = asyncio.run(
        create_runner(
            "naive",
            use_mock=False,
            service=service,
            kb_id="kb_001",
            model_config=model_config,
        ).run_sample(sample())
    )

    assert service.request.mode == RAGMode.NAIVE
    assert service.request.options["model"] == model_config
    assert service.request.options["top_k"] == 5
    assert result["token_usage"]["total"] == 7
