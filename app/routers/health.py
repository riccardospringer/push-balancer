"""app/routers/health.py — GET /api/health, GET /api/memory-stats"""
import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.research.worker import _health_state, _research_state

router = APIRouter()


def _process_rss_mb() -> float:
    """Liest den aktuellen RSS-Speicherverbrauch des Prozesses in MB."""
    try:
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: ru_maxrss in Bytes; Linux: in KB
        if os.uname().sysname == "Darwin":
            return round(rss_bytes / 1024 / 1024, 1)
        return round(rss_bytes / 1024, 1)
    except Exception:
        return -1.0


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


@router.get("/api/memory-stats")
def get_memory_stats() -> JSONResponse:
    """Zeigt aktuellen RAM-Verbrauch und Puffer-Größen des Prozesses.

    Nützlich um zu prüfen ob der Memory-Cleanup-Worker funktioniert:
    - done_runs_pending_cleanup sollte nach 5 Minuten = 0 sein
    - event_log_runs zeigt die Gesamtgröße der Analyse-Historie
    """
    from app.research.worker import _cleanup_stats, _cleanup_stats_lock, _BUFFER_LIMITS

    s = _research_state
    buffers = {key: len(s.get(key, [])) for key in _BUFFER_LIMITS}

    with _cleanup_stats_lock:
        cs = dict(_cleanup_stats)

    now = time.time()
    last_cleanup_ago = int(now - cs.get("last_cleanup_ts", 0)) if cs.get("last_cleanup_ts") else -1

    return JSONResponse(content={
        "process_rss_mb": _process_rss_mb(),
        "buffers": buffers,
        "buffer_limits": _BUFFER_LIMITS,
        "done_runs_pending_cleanup": cs.get("done_runs_pending_cleanup", 0),
        "event_log_runs": len(s.get("accuracy_history", [])),
        "analysis_generation": s.get("analysis_generation", 0),
        "cleanup_runs": cs.get("cleanup_runs", 0),
        "items_freed_total": cs.get("items_freed_total", 0),
        "items_freed_last": cs.get("items_freed_last", 0),
        "last_cleanup_ago_s": last_cleanup_ago,
        "push_data_count": len(s.get("push_data", [])),
    })
