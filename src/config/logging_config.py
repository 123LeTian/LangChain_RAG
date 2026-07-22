"""Logging paths for local structured logs."""

from __future__ import annotations

from pathlib import Path

from src.config.runtime_config import get_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_app_log_path() -> Path:
    return get_runtime_config().app_log_path


def get_error_log_path() -> Path:
    return get_runtime_config().error_log_path


def get_build_log_dir() -> Path:
    return get_runtime_config().build_log_dir


def get_log_dir() -> Path:
    return get_runtime_config().app_log_path.parent


APP_LOG_PATH = get_app_log_path()
ERROR_LOG_PATH = get_error_log_path()
BUILD_LOG_DIR = get_build_log_dir()
LOG_DIR = get_log_dir()


__all__ = [
    "APP_LOG_PATH",
    "BUILD_LOG_DIR",
    "ERROR_LOG_PATH",
    "LOG_DIR",
    "PROJECT_ROOT",
    "get_app_log_path",
    "get_build_log_dir",
    "get_error_log_path",
    "get_log_dir",
]
