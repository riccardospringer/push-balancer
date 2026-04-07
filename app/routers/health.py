"""app/routers/health.py — GET /api/health"""
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.research.worker import _health_state, _research_state

router = APIRouter()


@router.get("/api/health")
def get_health() -> JSONResponse:
    """Liefert Health- und Security-Status aller Endpunkte."""
    uptime = time.time() - _health_state.get("uptime_start", time.time())
    return JSONResponse(content={
        "status": _health_state.get("status", "unknown"),
        "uptimeSeconds": int(uptime),
        "uptimeHuman": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "lastCheck": _health_state.get("last_check", 0),
        "checksOk": _health_state.get("checks_ok", 0),
        "checksFail": _health_state.get("checks_fail", 0),
        "endpoints": _health_state.get("endpoints", {}),
        "researchDataPoints": len(_research_state.get("push_data", [])),
        "researchLastAnalysis": _research_state.get("last_analysis", 0),
    })
