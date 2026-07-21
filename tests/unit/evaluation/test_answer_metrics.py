from src.evaluation.answer_metrics import (
    answer_point_coverage,
    citation_presence_rate,
    unanswerable_refusal_rate,
)
from src.evaluation.dataset import EvaluationSample, ExpectedSource


def test_answer_point_coverage_matches_keywords():
    coverage = answer_point_coverage(
        "RAG retrieves external knowledge before generation.",
        [
            "RAG retrieves relevant external knowledge before generation.",
            "Unrelated point about databases.",
        ],
    )
    assert coverage == 0.5


def test_citation_presence_rate():
    rate = citation_presence_rate(
        [
            {"citations": [{"chunk_id": "a"}]},
            {"citations": []},
        ]
    )
    assert rate == 0.5


def test_unanswerable_refusal_rate():
    samples = [
        EvaluationSample(
            id="u1",
            question="q",
            answer_points=["It should avoid fabricating an answer."],
            expected_sources=[ExpectedSource(document_id="d1")],
            tags=["unanswerable"],
            difficulty="easy",
        )
    ]
    results = [
        {
            "id": "u1",
            "answer": "It should avoid fabricating an answer.",
        }
    ]
    assert unanswerable_refusal_rate(samples, results) == 1.0
