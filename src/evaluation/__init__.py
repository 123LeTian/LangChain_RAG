"""Evaluation utilities for RAG baselines."""

from src.evaluation.dataset import (
    EvaluationDatasetError,
    EvaluationSample,
    ExpectedSource,
    load_jsonl_dataset,
)
from src.evaluation.report import build_evaluation_report, write_report
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    mrr,
    reciprocal_rank,
    source_matches,
)
__all__ = [
    "EvaluationDatasetError",
    "EvaluationSample",
    "ExpectedSource",
    "MockRAGRunner",
    "RAGServiceNaiveRunner",
    "build_evaluation_report",
    "hit_at_k",
    "load_jsonl_dataset",
    "mrr",
    "reciprocal_rank",
    "run_evaluation",
    "source_matches",
    "write_report",
]


def __getattr__(name: str):
    if name in {"MockRAGRunner", "RAGServiceNaiveRunner", "run_evaluation"}:
        from importlib import import_module

        module = import_module("src.evaluation.runner")
        return getattr(module, name)
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
