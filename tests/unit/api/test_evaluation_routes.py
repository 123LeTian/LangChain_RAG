import pytest

from src.api.routes.evaluation import EvaluationRunRequest, get_evaluation_result, run_evaluation


class FakeEvaluationRunner:
    def __init__(self):
        self.request = None

    async def run_evaluation(self, request):
        self.request = request
        return {
            "run_id": "run_real",
            "status": "completed",
            "kb_id": request.kb_id,
            "modes": list(request.modes),
            "model_id": request.model_id,
            "sample_limit": request.sample_limit,
            "message": "done",
        }

    async def get_result(self, run_id):
        assert run_id == "run_real"
        return {
            "run_id": run_id,
            "model_id": "model-abc",
            "per_mode_results": [
                {
                    "mode": "naive",
                    "sample_count": 2,
                    "hit_at_1": 0.5,
                    "hit_at_3": 1.0,
                    "hit_at_5": 1.0,
                    "mrr": 0.75,
                    "average_latency_ms": 100.0,
                    "p50_latency_ms": 90.0,
                    "average_token_count": 12.5,
                    "per_sample_results": [
                        {"token_usage": {"total": 10}},
                        {"token_usage": {"prompt": 6, "completion": 9}},
                    ],
                },
                {
                    "mode": "graph",
                    "sample_count": 2,
                    "hit_at_1": 1.0,
                    "hit_at_3": 1.0,
                    "hit_at_5": 1.0,
                    "mrr": 1.0,
                    "average_latency_ms": 150.0,
                    "graph_metrics": {
                        "relation_hit_rate": 0.8,
                        "local_search_success": 0.7,
                        "global_search_success": 0.6,
                    },
                    "per_sample_results": [],
                },
            ],
        }


@pytest.mark.asyncio
async def test_evaluation_run_accepts_shared_model_and_filters_modes():
    request = EvaluationRunRequest(
        kb_id="kb_001",
        modes=["naive", "graph"],
        model_id="model-abc",
        sample_limit=3,
    )

    response = await run_evaluation(request)
    assert response["kb_id"] == "kb_001"
    assert response["model_id"] == "model-abc"
    assert response["modes"] == ["naive", "graph"]
    assert response["sample_limit"] == 3

    results = await get_evaluation_result(response["run_id"])
    assert [item["mode"] for item in results] == ["naive", "graph"]
    assert results[0]["sample_count"] == 3


@pytest.mark.asyncio
async def test_evaluation_routes_use_injected_runner_and_map_report():
    fake = FakeEvaluationRunner()
    request = EvaluationRunRequest(
        kb_id="kb_maotai",
        modes=["naive", "graph"],
        model_id="model-abc",
        sample_limit=2,
    )

    response = await run_evaluation(request, eval_runner=fake)

    assert fake.request is request
    assert response["status"] == "completed"
    assert response["model_id"] == "model-abc"
    assert response["sample_limit"] == 2

    results = await get_evaluation_result("run_real", eval_runner=fake)
    assert [item["mode"] for item in results] == ["naive", "graph"]
    assert results[0]["run_id"] == "run_real"
    assert results[0]["model_id"] == "model-abc"
    assert results[0]["metrics"]["retrieval"]["hit_at_3"] == 1.0
    assert results[0]["metrics"]["system"]["total_tokens"] == 25
    assert results[1]["metrics"]["graph"]["relation_hit_rate"] == 0.8
