"""app/routers/health.py — GET /api/health, GET /api/memory-stats"""
import os
import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.auth import require_admin_key
from app.config import (
    ADOBE_TRAFFIC_ENABLED,
    ARTICLE_PREDICTION_ENRICHMENT_ENABLED,
    BACKGROUND_AUTOMATIONS_ENABLED,
    ECONOMY_MODE,
    HEALTH_ACTIVE_CHECKS_ENABLED,
    LIVE_FEED_FALLBACK_ENABLED,
    OPENAI_BACKFILL_ENABLED,
    OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY,
    OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR,
    OPENAI_PREDICTION_SCORING_ENABLED,
    OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY,
    OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR,
    OPENAI_TITLE_GENERATION_ENABLED,
    PAID_EXTERNAL_APIS_ENABLED,
    PUSH_LIVE_FETCH_ENABLED,
    RESEARCH_EXTERNAL_CONTEXT_ENABLED,
    TAGESPLAN_ON_DEMAND_BUILD_ENABLED,
)
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
    raw_status = _health_state.get("status", "starting")
    if not HEALTH_ACTIVE_CHECKS_ENABLED and raw_status in {"starting", "", None}:
        raw_status = "ok"
    status = "healthy" if raw_status == "ok" else raw_status
    if status not in {"healthy", "degraded", "unhealthy"}:
        status = "unhealthy"
    endpoints = _health_state.get("endpoints", {})
    checks = {
        key: {
            "ok": bool(value.get("ok")),
            **({"error": value.get("error")} if value.get("error") else {}),
        }
        for key, value in endpoints.items()
    }
    cost_controls = {
        "paidExternalApisEnabled": PAID_EXTERNAL_APIS_ENABLED,
        "openaiTitleGenerationEnabled": bool(
            PAID_EXTERNAL_APIS_ENABLED
            and OPENAI_TITLE_GENERATION_ENABLED
            and OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR > 0
            and OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY > 0
        ),
        "openaiPredictionScoringEnabled": bool(
            PAID_EXTERNAL_APIS_ENABLED
            and OPENAI_PREDICTION_SCORING_ENABLED
            and OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR > 0
            and OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY > 0
        ),
        "openaiBackfillEnabled": bool(
            PAID_EXTERNAL_APIS_ENABLED and OPENAI_BACKFILL_ENABLED
        ),
        "adobeTrafficEnabled": bool(
            PAID_EXTERNAL_APIS_ENABLED and ADOBE_TRAFFIC_ENABLED
        ),
        "backgroundAutomationsEnabled": BACKGROUND_AUTOMATIONS_ENABLED,
        "healthActiveChecksEnabled": HEALTH_ACTIVE_CHECKS_ENABLED,
        "economyMode": ECONOMY_MODE,
        "pushLiveFetchEnabled": PUSH_LIVE_FETCH_ENABLED,
        "researchExternalContextEnabled": RESEARCH_EXTERNAL_CONTEXT_ENABLED,
        "liveFeedFallbackEnabled": LIVE_FEED_FALLBACK_ENABLED,
        "articlePredictionEnrichmentEnabled": ARTICLE_PREDICTION_ENRICHMENT_ENABLED,
        "tagesplanOnDemandBuildEnabled": TAGESPLAN_ON_DEMAND_BUILD_ENABLED,
    }
    return JSONResponse(content={
        "status": status,
        "uptime": int(uptime),
        "checks": checks,
        "costControls": cost_controls,
        "research": {
            "version": len(_research_state.get("push_data", [])),
            "lastUpdate": (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(_research_state.get("last_analysis", 0)),
                )
                if _research_state.get("last_analysis")
                else ""
            ),
        },
        "uptimeSeconds": int(uptime),
        "uptimeHuman": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "lastCheck": _health_state.get("last_check", 0),
        "checksOk": _health_state.get("checks_ok", 0),
        "checksFail": _health_state.get("checks_fail", 0),
        "endpoints": endpoints,
        "researchDataPoints": len(_research_state.get("push_data", [])),
        "researchLastAnalysis": _research_state.get("last_analysis", 0),
    })


@router.get("/api/memory-stats", dependencies=[Depends(require_admin_key)])
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
