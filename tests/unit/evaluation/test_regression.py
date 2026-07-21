from src.evaluation.regression import RegressionThresholds, run_regression_checks


def _mode_result(**overrides):
    base = {
        "mode": "naive",
        "runner_type": "mock_naive",
        "status": "ok",
        "failure_count": 0,
        "citation_presence_rate": 1.0,
        "hit_at_1": 1.0,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mrr": 1.0,
        "average_latency_ms": 1.0,
        "per_sample_results": [],
    }
    base.update(overrides)
    return base


def test_regression_passes_for_mock_summary():
    report = {"per_mode_results": [_mode_result()]}
    checks = run_regression_checks(
        report, thresholds=RegressionThresholds(mock_relaxed=True)
    )
    assert all(check["passed"] for check in checks)


def test_regression_fails_on_high_failure_count():
    report = {
        "per_mode_results": [
            _mode_result(failure_count=3),
        ]
    }
    checks = run_regression_checks(report, thresholds=RegressionThresholds())
    failed = [check for check in checks if not check["passed"]]
    assert failed
