from __future__ import annotations

import json
import sys

from scripts.build_check import BuildStep, REDACTED, TAIL_LIMIT, run_build_check


def test_build_check_generates_json_log(tmp_path):
    exit_code, output_path, payload = run_build_check(
        [
            BuildStep(
                name="ok",
                command=[sys.executable, "-c", "print('hello')"],
                cwd=tmp_path,
            )
        ],
        output_dir=tmp_path / "logs",
    )

    assert exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["exit_code"] == 0
    assert data["steps"][0]["exit_code"] == 0
    assert data["steps"][0]["stdout_tail"].strip() == "hello"
    assert payload["steps"][0]["name"] == "ok"


def test_build_check_failure_returns_nonzero(tmp_path):
    exit_code, output_path, payload = run_build_check(
        [
            BuildStep(
                name="fail",
                command=[sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"],
                cwd=tmp_path,
            )
        ],
        output_dir=tmp_path / "logs",
    )

    assert exit_code == 3
    assert payload["exit_code"] == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["steps"][0]["exit_code"] == 3


def test_build_check_tail_is_limited(tmp_path):
    long_text = "x" * (TAIL_LIMIT + 100)
    exit_code, _output_path, payload = run_build_check(
        [
            BuildStep(
                name="tail",
                command=[sys.executable, "-c", f"print('{long_text}')"],
                cwd=tmp_path,
            )
        ],
        output_dir=tmp_path / "logs",
    )

    assert exit_code == 0
    assert len(payload["steps"][0]["stdout_tail"]) <= TAIL_LIMIT


def test_build_check_redacts_sensitive_output(tmp_path):
    exit_code, _output_path, payload = run_build_check(
        [
            BuildStep(
                name="redact",
                command=[
                    sys.executable,
                    "-c",
                    "print('api_key=abc DEEPSEEK_API_KEY .env')",
                ],
                cwd=tmp_path,
            )
        ],
        output_dir=tmp_path / "logs",
    )

    text = json.dumps(payload, ensure_ascii=False)
    assert exit_code == 0
    assert "api_key" not in text
    assert "DEEPSEEK_API_KEY" not in text
    assert ".env" not in text
    assert REDACTED in text
