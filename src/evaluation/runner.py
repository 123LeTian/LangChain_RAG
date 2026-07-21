"""Naive RAG baseline evaluation runner."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.evaluation.dataset import EvaluationSample, ExpectedSource, load_jsonl_dataset
from src.evaluation.report import build_evaluation_report, write_report
from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank
from src.models.rag import RAGMode, RAGRequest


DEFAULT_DATASET_PATH = "datasets/evaluation/rag_eval_v1.jsonl"
DEFAULT_REPORT_PATH = "datasets/evaluation/reports/naive_baseline.json"
DEFAULT_FINAL_REPORT_PATH = "datasets/evaluation/reports/final_evaluation.json"
DEFAULT_FINAL_MARKDOWN_PATH = "docs/evaluation-report.md"


class MockRAGRunner:
    """Offline runner with the same report contract as the real runner."""

    runner_type = "mock_naive"

    def __init__(self, *, top_k: int = 5) -> None:
        self.top_k = top_k

    async def run_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        started = time.perf_counter()
        hits = [
            _mock_hit(source, rank=index + 1)
            for index, source in enumerate(sample.expected_sources[: self.top_k])
        ]
        citations = [_mock_citation(source) for source in sample.expected_sources]
        answer = " ".join(sample.answer_points)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return build_sample_result(
            sample=sample,
            answer=answer,
            hits=hits,
            citations=citations,
            latency_ms=latency_ms,
            error=None,
        )


class RAGServiceNaiveRunner:
    """Runner that calls the real RAGService using RAGMode.NAIVE."""

    runner_type = "rag_service_naive"

    def __init__(
        self,
        service: Any,
        *,
        kb_id: str = "default",
        top_k: int = 5,
        timeout: float | None = None,
    ) -> None:
        self.service = service
        self.kb_id = kb_id
        self.top_k = top_k
        self.timeout = timeout

    async def run_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        started = time.perf_counter()
        request = RAGRequest(
            query=sample.question,
            kb_id=str(sample.metadata.get("kb_id") or self.kb_id),
            mode=RAGMode.NAIVE,
            options={"top_k": self.top_k},
        )
        result = await self.service.run_safe(request, timeout=self.timeout)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        warnings = getattr(result, "warnings", []) or []
        return build_sample_result(
            sample=sample,
            answer=getattr(result, "answer", ""),
            hits=list(getattr(result, "hits", []) or []),
            citations=list(getattr(result, "citations", []) or []),
            latency_ms=latency_ms,
            error="; ".join(warnings) if warnings and not getattr(result, "answer", "") else None,
        )


async def run_evaluation(
    *,
    dataset_path: str | Path,
    runner: Any,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """Load a dataset, run all samples, build a report, and optionally write it."""

    samples = load_jsonl_dataset(dataset_path)
    per_sample_results = [
        await runner.run_sample(sample)
        for sample in samples
    ]
    report = build_evaluation_report(
        dataset_path=str(dataset_path),
        runner_type=runner.runner_type,
        samples=samples,
        per_sample_results=per_sample_results,
    )
    if report_path is not None:
        write_report(report, report_path)
    return report


def build_sample_result(
    *,
    sample: EvaluationSample,
    answer: str,
    hits: Sequence[Any],
    citations: Sequence[Any],
    latency_ms: float,
    error: str | None,
) -> Dict[str, Any]:
    """Build the required per-sample result payload with retrieval metrics."""

    expected = sample.expected_sources
    return {
        "id": sample.id,
        "question": sample.question,
        "answer": answer,
        "hits": [_serialize_source_like(hit) for hit in hits],
        "citations": [_serialize_source_like(citation) for citation in citations],
        "latency_ms": latency_ms,
        "hit_at_1": hit_at_k(hits, expected, k=1),
        "hit_at_3": hit_at_k(hits, expected, k=3),
        "hit_at_5": hit_at_k(hits, expected, k=5),
        "reciprocal_rank": reciprocal_rank(hits, expected),
        "error": error,
    }


def _mock_hit(source: ExpectedSource, *, rank: int) -> Dict[str, Any]:
    return {
        "document_id": source.document_id,
        "chunk_id": source.chunk_id,
        "filename": source.filename,
        "quote": source.quote,
        "score": round(max(0.0, 1.0 - (rank - 1) * 0.05), 3),
        "rank": rank,
        "retriever": "mock_vector",
    }


def _mock_citation(source: ExpectedSource) -> Dict[str, Any]:
    return {
        "document_id": source.document_id,
        "chunk_id": source.chunk_id,
        "filename": source.filename,
        "quote": source.quote,
    }


def _serialize_source_like(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    payload: Dict[str, Any] = {}
    for name in (
        "document_id",
        "chunk_id",
        "filename",
        "text_snippet",
        "score",
        "rank",
        "retriever",
    ):
        if hasattr(value, name):
            payload[name] = getattr(value, name)
    source = getattr(value, "source", None)
    if source is not None:
        payload["source"] = _serialize_source_like(source)
    return payload


async def run_final_evaluation(
    *,
    dataset_path: str | Path,
    modes: Sequence[str],
    use_mock: bool,
    output_path: str | Path,
    markdown_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    kb_id: str = "default",
    top_k: int = 5,
    service: Any | None = None,
) -> Dict[str, Any]:
    """Run the D6 multi-mode evaluation and write final reports."""

    from src.evaluation.multi_mode_runner import MultiModeEvaluationRunner, create_runner
    from src.evaluation.report import (
        write_csv_summary,
        write_markdown_report,
        write_report,
    )

    def runner_factory(mode: str) -> Any:
        return create_runner(
            mode,
            use_mock=use_mock,
            service=service,
            kb_id=kb_id,
            top_k=top_k,
        )

    report = await MultiModeEvaluationRunner(
        dataset_path=str(dataset_path),
        modes=modes,
        runner_factory=runner_factory,
    ).run()
    write_report(report, output_path)
    if markdown_path is not None:
        write_markdown_report(report, markdown_path)
    if csv_path is not None:
        write_csv_summary(report, csv_path)
    return report


async def _main_async(args: argparse.Namespace) -> int:
    if args.modes:
        if not args.mock and args.service is not None:
            service = args.service
        elif not args.mock:
            raise SystemExit(
                "Multi-mode CLI requires --mock unless a configured RAGService "
                "is injected by application code."
            )
        else:
            service = None
        await run_final_evaluation(
            dataset_path=args.dataset,
            modes=args.modes,
            use_mock=args.mock,
            output_path=args.output,
            markdown_path=args.markdown,
            csv_path=args.csv,
            kb_id=args.kb_id,
            top_k=args.top_k,
            service=service,
        )
        return 0

    if not args.mock:
        raise SystemExit(
            "The CLI needs --mock unless a configured RAGService is supplied by "
            "application code."
        )
    runner = MockRAGRunner(top_k=args.top_k)
    await run_evaluation(
        dataset_path=args.dataset,
        runner=runner,
        report_path=args.output,
    )
    return 0


def main(argv: List[str] | None = None) -> int:
    from src.evaluation.multi_mode_runner import ALL_EVAL_MODES

    parser = argparse.ArgumentParser(
        description=(
            "Run RAG evaluation. Default: D2 naive baseline report. "
            "Use --modes for D6 multi-mode final evaluation."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--mode", default="naive", choices=["naive"])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(ALL_EVAL_MODES),
        help="Run final multi-mode evaluation for the listed modes.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default=None,
        help="Report JSON path. Defaults to naive_baseline.json or final_evaluation.json.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional Markdown report path (final evaluation only).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV summary path (final evaluation only).",
    )
    parser.add_argument("--kb-id", default="default")
    args = parser.parse_args(argv)

    if args.modes:
        if args.output is None:
            args.output = DEFAULT_FINAL_REPORT_PATH
        if args.markdown is None:
            args.markdown = DEFAULT_FINAL_MARKDOWN_PATH
    else:
        if args.output is None:
            args.output = DEFAULT_REPORT_PATH
    args.service = None
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_FINAL_MARKDOWN_PATH",
    "DEFAULT_FINAL_REPORT_PATH",
    "DEFAULT_REPORT_PATH",
    "MockRAGRunner",
    "RAGServiceNaiveRunner",
    "build_sample_result",
    "main",
    "run_evaluation",
    "run_final_evaluation",
]
