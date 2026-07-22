"""Safe runtime configuration routes."""

from __future__ import annotations

from fastapi import APIRouter

from src.config.runtime_config import get_runtime_config


router = APIRouter(prefix="/api/config", tags=["Runtime Config"])


@router.get("/runtime")
async def runtime_config() -> dict:
    """Return non-sensitive runtime configuration for diagnostics."""
    return get_runtime_config().public_dict()

