"""Multi-mode RAG evaluation runners (D6)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from src.evaluation.agent_metrics import infer_expected_tools, summarize_agent_metrics
from src.evaluation.answer_metrics import (
    citation_presence_rate,
    enrich_sample_answer_metrics,
    unanswerable_refusal_rate,
)
from src.evaluation.dataset import EvaluationSample, load_jsonl_dataset
from src.evaluation.graph_metrics import summarize_graph_metrics
from src.evaluation.runner import (
    MockRAGRunner,
    build_sample_result,
    _mock_citation,
    _mock_hit,
)
from src.models.rag import RAGMode, RAGRequest

ALL_EVAL_MODES = ("naive", "advanced", "modular", "graph", "agentic")


class BaseModeRunner(ABC):
    """Unified interface for a single evaluation mode."""

    mode: str
    runner_type: str

    @abstractmethod
    async def run_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        raise NotImplementedError


class MockModeRunner(BaseModeRunner):
    """Offline mock runner for any RAG mode; does not call external services."""

    def __init__(self, mode: str, *, top_k: int = 5) -> None:
        if mode not in ALL_EVAL_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        self.mode = mode
        self.runner_type = f"mock_{mode}"
        self.top_k = top_k
        self._naive = MockRAGRunner(top_k=top_k)

    async def run_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        started = time.perf_counter()
        warnings: List[str] = []
        try:
            if self.mode == "naive":
                base = await self._naive.run_sample(sample)
            else:
                base = self._build_mode_sample(sample)
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            base["latency_ms"] = latency_ms
            base["warnings"] = warnings
            base["token_usage"] = _estimate_token_usage(base.get("answer", ""))
            return enrich_sample_answer_metrics(sample, base)
        except Exception as exc:  # noqa: BLE001 — evaluation must not crash
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return enrich_sample_answer_metrics(
                sample,
                {
                    "id": sample.id,
                    "question": sample.question,
                    "answer": "",
                    "hits": [],
                    "citations": [],
                    "latency_ms": latency_ms,
                    "token_usage": {},
                    "warnings": [str(exc)],
                    "hit_at_1": 0.0,
                    "hit_at_3": 0.0,
                    "hit_at_5": 0.0,
                    "reciprocal_rank": 0.0,
                    "error": str(exc),
                },
            )

    def _build_mode_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        hits = [
            self._mode_hit(source, rank=index + 1)
            for index, source in enumerate(sample.expected_sources[: self.top_k])
        ]
        citations = [_mock_citation(source) for source in sample.expected_sources]
        answer = " ".join(sample.answer_points)
        payload = build_sample_result(
            sample=sample,
            answer=answer,
            hits=hits,
            citations=citations,
            latency_ms=0.0,
            error=None,
        )
        if self.mode == "agentic":
            expected_tools = infer_expected_tools(sample.answer_points)
            selected = list(expected_tools)
            if not selected and "agent_route" in sample.tags:
                selected = ["route_agent"]
            payload["agent"] = {
                "tools_selected": selected,
                "expected_tools": expected_tools or selected,
                "trajectory_success": not payload.get("error"),
                "tool_steps": max(1, len(selected)),
            }
        return payload

    def _mode_hit(self, source: Any, *, rank: int) -> Dict[str, Any]:
        hit = _mock_hit(source, rank=rank)
        if self.mode == "advanced":
            hit["retriever"] = "mock_hybrid"
        elif self.mode == "modular":
            hit["retriever"] = "mock_modular"
        elif self.mode == "graph":
            hit["retriever"] = "graph_local"
            hit["graph_source_trace"] = True
            hit["graph_path"] = ["entity:start", "relation:links", "entity:target"]
            hit["metadata"] = {
                "graph_scope": "local" if rank == 1 else "global",
                "graph_source_trace": True,
                "graph_path": hit["graph_path"],
                "community_report_used": rank > 1,
            }
            if rank > 1:
                hit["graph_scope"] = "global"
                hit["community_report_used"] = True
        elif self.mode == "agentic":
            hit["retriever"] = "mock_agent_vector"
        return hit


class RAGServiceModeRunner(BaseModeRunner):
    """Calls RAGService.run_safe for a specific RAGMode."""

    _MODE_MAP = {
        "naive": RAGMode.NAIVE,
        "advanced": RAGMode.ADVANCED,
        "modular": RAGMode.MODULAR,
        "graph": RAGMode.GRAPH,
        "agentic": RAGMode.AGENTIC,
    }

    def __init__(
        self,
        service: Any,
        mode: str,
        *,
        kb_id: str = "default",
        top_k: int = 5,
        timeout: float | None = None,
    ) -> None:
        if mode not in ALL_EVAL_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        self.mode = mode
        self.runner_type = f"rag_service_{mode}"
        self.service = service
        self.kb_id = kb_id
        self.top_k = top_k
        self.timeout = timeout

    async def run_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        started = time.perf_counter()
        request = RAGRequest(
            query=sample.question,
            kb_id=str(sample.metadata.get("kb_id") or self.kb_id),
            mode=self._MODE_MAP[self.mode],
            options={"top_k": self.top_k},
        )
        try:
            result = await self.service.run_safe(request, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return enrich_sample_answer_metrics(
                sample,
                {
                    "id": sample.id,
                    "question": sample.question,
                    "answer": "",
                    "hits": [],
                    "citations": [],
                    "latency_ms": latency_ms,
                    "token_usage": {},
                    "warnings": [str(exc)],
                    "hit_at_1": 0.0,
                    "hit_at_3": 0.0,
                    "hit_at_5": 0.0,
                    "reciprocal_rank": 0.0,
                    "error": str(exc),
                },
            )

        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        warnings = list(getattr(result, "warnings", []) or [])
        usage = dict(getattr(result, "usage", {}) or {})
        answer = getattr(result, "answer", "") or ""
        error = None
        if warnings and not answer:
            error = "; ".join(warnings)
        payload = build_sample_result(
            sample=sample,
            answer=answer,
            hits=list(getattr(result, "hits", []) or []),
            citations=list(getattr(result, "citations", []) or []),
            latency_ms=latency_ms,
            error=error,
        )
        payload["warnings"] = warnings
        payload["token_usage"] = usage
        return enrich_sample_answer_metrics(sample, payload)


class EvaluationModeRunner:
    """Runs evaluation for one mode and aggregates metrics."""

    def __init__(self, runner: BaseModeRunner) -> None:
        self.runner = runner

    async def run(
        self, samples: Sequence[EvaluationSample]
    ) -> Dict[str, Any]:
        per_sample: List[Dict[str, Any]] = []
        mode_error: str | None = None
        status = "ok"
        for sample in samples:
            try:
                per_sample.append(await self.runner.run_sample(sample))
            except Exception as exc:  # noqa: BLE001
                mode_error = str(exc)
                per_sample.append(
                    enrich_sample_answer_metrics(
                        sample,
                        {
                            "id": sample.id,
                            "question": sample.question,
                            "answer": "",
                            "hits": [],
                            "citations": [],
                            "latency_ms": 0.0,
                            "token_usage": {},
                            "warnings": [str(exc)],
                            "hit_at_1": 0.0,
                            "hit_at_3": 0.0,
                            "hit_at_5": 0.0,
                            "reciprocal_rank": 0.0,
                            "error": str(exc),
                        },
                    )
                )

        if mode_error and all(result.get("error") for result in per_sample):
            status = "failed"

        return summarize_mode_result(
            mode=self.runner.mode,
            runner_type=self.runner.runner_type,
            samples=samples,
            per_sample_results=per_sample,
            status=status,
            error=mode_error,
        )


class MultiModeEvaluationRunner:
    """Runs evaluation across multiple RAG modes."""

    def __init__(
        self,
        *,
        dataset_path: str,
        modes: Sequence[str],
        runner_factory: Any,
    ) -> None:
        self.dataset_path = dataset_path
        self.modes = list(modes)
        self.runner_factory = runner_factory

    async def run(self) -> Dict[str, Any]:
        from src.evaluation.report import build_final_evaluation_report

        samples = load_jsonl_dataset(self.dataset_path)
        per_mode_results: List[Dict[str, Any]] = []
        for mode in self.modes:
            try:
                runner = self.runner_factory(mode)
            except Exception as exc:  # noqa: BLE001
                per_mode_results.append(
                    {
                        "mode": mode,
                        "runner_type": "unavailable",
                        "status": "unavailable",
                        "error": str(exc),
                        "sample_count": len(samples),
                        "per_sample_results": [],
                    }
                )
                continue
            try:
                mode_result = await EvaluationModeRunner(runner).run(samples)
            except Exception as exc:  # noqa: BLE001
                mode_result = {
                    "mode": mode,
                    "runner_type": getattr(runner, "runner_type", "unknown"),
                    "status": "failed",
                    "error": str(exc),
                    "sample_count": len(samples),
                    "per_sample_results": [],
                }
            per_mode_results.append(mode_result)

        return build_final_evaluation_report(
            dataset_path=self.dataset_path,
            modes=self.modes,
            samples=samples,
            per_mode_results=per_mode_results,
        )


def summarize_mode_result(
    *,
    mode: str,
    runner_type: str,
    samples: Sequence[EvaluationSample],
    per_sample_results: Sequence[Dict[str, Any]],
    status: str = "ok",
    error: str | None = None,
) -> Dict[str, Any]:
    latencies = [
        float(result["latency_ms"])
        for result in per_sample_results
        if isinstance(result.get("latency_ms"), (int, float))
    ]
    token_counts = [
        _token_total(result.get("token_usage"))
        for result in per_sample_results
    ]
    failures = [
        result["id"] for result in per_sample_results if result.get("error")
    ]
    warning_count = sum(len(result.get("warnings") or []) for result in per_sample_results)

    graph_metrics = None
    if mode == "graph" and per_sample_results:
        graph_metrics = summarize_graph_metrics(per_sample_results)

    agent_metrics = None
    if mode == "agentic" and per_sample_results:
        samples_by_id = {sample.id: sample for sample in samples}
        agent_metrics = summarize_agent_metrics(
            per_sample_results, samples_by_id=samples_by_id
        )

    answer_cov_values = [
        float(result["answer_point_coverage"])
        for result in per_sample_results
        if isinstance(result.get("answer_point_coverage"), (int, float))
    ]

    payload: Dict[str, Any] = {
        "mode": mode,
        "runner_type": runner_type,
        "status": status,
        "error": error,
        "sample_count": len(samples),
        "hit_at_1": _average_field(per_sample_results, "hit_at_1"),
        "hit_at_3": _average_field(per_sample_results, "hit_at_3"),
        "hit_at_5": _average_field(per_sample_results, "hit_at_5"),
        "mrr": _average_field(per_sample_results, "reciprocal_rank"),
        "average_latency_ms": _average_list(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "average_token_count": _average_list(token_counts),
        "failure_count": len(failures),
        "failed_samples": failures,
        "warning_count": warning_count,
        "citation_presence_rate": citation_presence_rate(per_sample_results),
        "answer_point_coverage": _average_list(answer_cov_values),
        "unanswerable_refusal_rate": unanswerable_refusal_rate(
            samples, per_sample_results
        ),
        "graph_metrics": graph_metrics,
        "agent_metrics": agent_metrics,
        "per_sample_results": list(per_sample_results),
    }
    if graph_metrics:
        payload.update(graph_metrics)
    if agent_metrics:
        payload.update(
            {
                key: value
                for key, value in agent_metrics.items()
                if value is not None
            }
        )
    return payload


def _average_field(results: Sequence[Dict[str, Any]], field: str) -> float:
    values = [
        float(result[field])
        for result in results
        if isinstance(result.get(field), (int, float))
    ]
    return _average_list(values)


def _average_list(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    index = max(0, min(index, len(ordered) - 1))
    return round(ordered[index], 6)


def _token_total(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    total = usage.get("total")
    if isinstance(total, (int, float)):
        return float(total)
    prompt = usage.get("prompt") or 0
    completion = usage.get("completion") or 0
    if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
        return float(prompt or 0) + float(completion or 0)
    return 0.0


def _estimate_token_usage(answer: str) -> Dict[str, int]:
    approx = max(1, len((answer or "").split()))
    total = approx * 12
    return {"prompt": total // 2, "completion": total - total // 2, "total": total}


def create_runner(
    mode: str,
    *,
    use_mock: bool,
    service: Any | None = None,
    kb_id: str = "default",
    top_k: int = 5,
    timeout: float | None = None,
) -> BaseModeRunner:
    if use_mock:
        return MockModeRunner(mode, top_k=top_k)
    if service is None:
        raise RuntimeError(
            f"Real runner for mode {mode!r} requires a configured RAGService"
        )
    return RAGServiceModeRunner(
        service, mode, kb_id=kb_id, top_k=top_k, timeout=timeout
    )


__all__ = [
    "ALL_EVAL_MODES",
    "BaseModeRunner",
    "EvaluationModeRunner",
    "MockModeRunner",
    "MultiModeEvaluationRunner",
    "RAGServiceModeRunner",
    "create_runner",
    "summarize_mode_result",
]
