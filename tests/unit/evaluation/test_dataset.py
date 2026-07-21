import json

import pytest

from src.evaluation.dataset import EvaluationDatasetError, load_jsonl_dataset


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def valid_sample(**overrides):
    sample = {
        "id": "eval_test",
        "question": "What is RAG?",
        "answer_points": ["Retrieval plus generation."],
        "expected_sources": [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "filename": "rag.md",
                "quote": "RAG combines retrieval and generation.",
            }
        ],
        "tags": ["fact"],
        "difficulty": "easy",
    }
    sample.update(overrides)
    return sample


def test_jsonl_loads_valid_samples(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [valid_sample()])

    samples = load_jsonl_dataset(path)

    assert len(samples) == 1
    assert samples[0].id == "eval_test"
    assert samples[0].expected_sources[0].document_id == "doc-1"


def test_missing_required_field_reports_line_and_sample_id(tmp_path):
    path = tmp_path / "eval.jsonl"
    sample = valid_sample()
    sample.pop("question")
    write_jsonl(path, [sample])

    with pytest.raises(EvaluationDatasetError, match="line 1, sample eval_test"):
        load_jsonl_dataset(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tags", "fact", "tags must be a list"),
        ("expected_sources", "doc-1", "expected_sources must be a list"),
        ("answer_points", [], "answer_points must contain"),
    ],
)
def test_invalid_field_types_raise_clear_errors(tmp_path, field, value, message):
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [valid_sample(**{field: value})])

    with pytest.raises(EvaluationDatasetError, match=message):
        load_jsonl_dataset(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"id": "broken"\n', encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match="line 1: invalid JSON"):
        load_jsonl_dataset(path)
