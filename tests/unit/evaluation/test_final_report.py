import asyncio
import json

from src.evaluation.multi_mode_runner import MultiModeEvaluationRunner, create_runner
from src.evaluation.report import build_final_evaluation_report, write_markdown_report


def test_final_report_fields(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "eval_test",
                "question": "Q",
                "answer_points": ["A"],
                "expected_sources": [{"document_id": "d1", "chunk_id": "c1"}],
                "tags": ["fact"],
                "difficulty": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = asyncio.run(
        MultiModeEvaluationRunner(
            dataset_path=str(dataset_path),
            modes=["naive", "advanced"],
            runner_factory=lambda mode: create_runner(mode, use_mock=True),
        ).run()
    )
    for field in (
        "dataset_path",
        "created_at",
        "modes",
        "runner_summary",
        "mode_summaries",
        "comparisons",
        "tag_breakdown",
        "regression_checks",
        "risks",
        "per_mode_results",
    ):
        assert field in report

    md_path = tmp_path / "report.md"
    write_markdown_report(report, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "# RAG Final Evaluation Report" in text
    assert "mock" in text.lower()


def test_build_final_evaluation_report_marks_mock_runners():
    report = build_final_evaluation_report(
        dataset_path="ds.jsonl",
        modes=["naive"],
        samples=[],
        per_mode_results=[
            {
                "mode": "naive",
                "runner_type": "mock_naive",
                "status": "ok",
                "sample_count": 0,
                "per_sample_results": [],
            }
        ],
    )
    assert report["runner_summary"]["naive"]["is_mock"] is True
