"""Run a lightweight build check and write a sanitized JSON build log."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_config import get_runtime_config

TAIL_LIMIT = 4000
NPM_EXECUTABLE = "npm.cmd" if os.name == "nt" else "npm"
SENSITIVE_MARKERS = [
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "set-cookie",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    ".env",
]
REDACTED = "***REDACTED***"


@dataclass
class BuildStep:
    name: str
    command: list[str]
    cwd: Path = PROJECT_ROOT


DEFAULT_STEPS = [
    BuildStep(
        name="chat_backend_tests",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_chat_sessions.py",
            "tests/test_chat_stream.py",
            "tests/test_chat_models.py",
            "tests/test_chat_presets.py",
            "tests/test_chat_search.py",
            "tests/test_chat_stats_export.py",
        ],
    ),
    BuildStep(
        name="frontend_build",
        command=[NPM_EXECUTABLE, "run", "build"],
        cwd=PROJECT_ROOT / "frontend",
    ),
]


def get_build_log_dir() -> Path:
    return get_runtime_config().build_log_dir


def run_build_check(
    steps: Iterable[BuildStep] = DEFAULT_STEPS,
    *,
    output_dir: Optional[Path] = None,
) -> tuple[int, Path, dict]:
    output_dir = output_dir or get_build_log_dir()
    started = datetime.now(timezone.utc)
    results = []
    final_exit_code = 0

    for step in steps:
        result = run_step(step)
        results.append(result)
        if result["exit_code"] != 0 and final_exit_code == 0:
            final_exit_code = result["exit_code"]

    ended = datetime.now(timezone.utc)
    payload = {
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_ms": int((ended - started).total_seconds() * 1000),
        "exit_code": final_exit_code,
        "steps": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"build-check-{ended.strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_exit_code, output_path, payload


def run_step(step: BuildStep) -> dict:
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            step.command,
            cwd=step.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_safe_env(),
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except Exception as exc:
        exit_code = 1
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
    ended = datetime.now(timezone.utc)
    return {
        "name": step.name,
        "command": _sanitize(" ".join(step.command)),
        "cwd": str(step.cwd),
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "duration_ms": int((ended - started).total_seconds() * 1000),
        "exit_code": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _safe_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if _is_sensitive(key):
            env[key] = REDACTED
    return env


def _tail(value: str, limit: int = TAIL_LIMIT) -> str:
    return _sanitize(value)[-limit:]


def _sanitize(value: str) -> str:
    sanitized = value or ""
    for marker in SENSITIVE_MARKERS:
        sanitized = sanitized.replace(marker, REDACTED)
    return sanitized


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(marker.lower().replace("-", "_") in lowered for marker in SENSITIVE_MARKERS)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run backend/frontend build checks.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else get_build_log_dir()
    exit_code, output_path, _payload = run_build_check(output_dir=output_dir)
    print(f"Build log: {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
