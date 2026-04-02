"""app/routers/health.py — GET /api/health"""
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.research.worker import _health_state, _research_state

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
    uptime_human: str
    last_check: float
    checks_ok: int
    checks_fail: int
    endpoints: dict[str, Any]
    research_data_points: int
    research_last_analysis: float


@router.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Liefert Health- und Security-Status aller Endpunkte."""
    uptime = time.time() - _health_state.get("uptime_start", time.time())
    return HealthResponse(
        status=_health_state.get("status", "unknown"),
        uptime_seconds=int(uptime),
        uptime_human=f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        last_check=_health_state.get("last_check", 0),
        checks_ok=_health_state.get("checks_ok", 0),
        checks_fail=_health_state.get("checks_fail", 0),
        endpoints=_health_state.get("endpoints", {}),
        research_data_points=len(_research_state.get("push_data", [])),
        research_last_analysis=_research_state.get("last_analysis", 0),
    )
