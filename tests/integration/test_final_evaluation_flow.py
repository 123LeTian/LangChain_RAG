import json

from src.evaluation.runner import main


def test_final_evaluation_cli_generates_json_markdown_and_csv(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "eval_final",
                "question": "What is RAG?",
                "answer_points": ["RAG retrieves knowledge before generation."],
                "expected_sources": [
                    {
                        "document_id": "doc-final",
                        "chunk_id": "chunk-final",
                        "filename": "final.md",
                    }
                ],
                "tags": ["fact", "relation", "agent_route"],
                "difficulty": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "final_evaluation.json"
    markdown = tmp_path / "evaluation-report.md"
    csv_path = tmp_path / "final_evaluation.csv"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--modes",
            "naive",
            "advanced",
            "modular",
            "graph",
            "agentic",
            "--mock",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["modes"] == [
        "naive",
        "advanced",
        "modular",
        "graph",
        "agentic",
    ]
    assert payload["runner_summary"]["agentic"]["is_mock"] is True
    assert payload["regression_checks"]["passed"] is True
    assert "# RAG Final Evaluation Report" in markdown.read_text(encoding="utf-8")
    assert "agentic" in csv_path.read_text(encoding="utf-8")
