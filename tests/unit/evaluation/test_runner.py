import asyncio
import json

from src.evaluation.dataset import EvaluationSample, ExpectedSource
from src.evaluation.runner import (
    MockRAGRunner,
    RAGServiceNaiveRunner,
    run_evaluation,
)
from src.models.rag import RAGChunk, RAGResult, RAGSource


def sample():
    return EvaluationSample(
        id="eval_test",
        question="What is RAG?",
        answer_points=["Retrieval plus generation."],
        expected_sources=[
            ExpectedSource(
                document_id="doc-1",
                chunk_id="chunk-1",
                filename="rag.md",
                quote="RAG combines retrieval and generation.",
            )
        ],
        tags=["fact", "global"],
        difficulty="easy",
    )


def write_dataset(path):
    payload = sample().to_dict()
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_mock_runner_can_build_per_sample_result():
    result = asyncio.run(MockRAGRunner().run_sample(sample()))

    assert result["id"] == "eval_test"
    assert result["hit_at_1"] == 1.0
    assert result["hit_at_3"] == 1.0
    assert result["hit_at_5"] == 1.0
    assert result["reciprocal_rank"] == 1.0
    assert result["error"] is None


def test_mock_runner_generates_report_with_comparison_fields(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    report_path = tmp_path / "report.json"
    write_dataset(dataset_path)

    report = asyncio.run(
        run_evaluation(
            dataset_path=dataset_path,
            runner=MockRAGRunner(),
            report_path=report_path,
        )
    )

    assert report_path.exists()
    assert report["dataset_path"] == str(dataset_path)
    assert report["runner_type"] == "mock_naive"
    assert report["sample_count"] == 1
    assert report["tag_distribution"] == {"fact": 1, "global": 1}
    for field in (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "average_latency_ms",
        "failed_samples",
        "per_sample_results",
        "created_at",
    ):
        assert field in report


def test_real_service_runner_calls_rag_service_with_naive_mode():
    class FakeService:
        def __init__(self):
            self.requests = []

        async def run_safe(self, request, timeout=None):
            self.requests.append((request, timeout))
            return RAGResult(
                answer="answer",
                hits=[
                    RAGChunk(
                        chunk_id="chunk-1",
                        content="text",
                        source=RAGSource(document_id="doc-1"),
                    )
                ],
                citations=[],
            )

    service = FakeService()
    result = asyncio.run(
        RAGServiceNaiveRunner(service, kb_id="kb-1", timeout=3).run_sample(sample())
    )

    request, timeout = service.requests[0]
    assert request.mode.value == "naive"
    assert request.kb_id == "kb-1"
    assert timeout == 3
    assert result["hit_at_1"] == 1.0
