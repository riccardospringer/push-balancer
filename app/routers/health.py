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
    teams_channel: dict = {}
    try:
        from app.notifications.teams import TeamsAlertConfig, channel_health

        teams_config = TeamsAlertConfig()
        teams_channel = channel_health(teams_config)
        # Ein stehender oder dauerhaft scheiternder Teams-Kanal darf nicht als
        # "healthy" durchgehen - sonst bleibt Stille unbemerkt.
        if teams_config.enabled and not teams_channel.get("healthy"):
            raw_status = "degraded"
    except Exception as exc:  # pragma: no cover - reine Diagnose
        teams_channel = {"status": "unknown", "error": type(exc).__name__}

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
        "teamsChannel": {
            "status": teams_channel.get("status", "unknown"),
            "healthy": bool(teams_channel.get("healthy", True)),
            "reason": teams_channel.get("reason") or "",
            "cycleAgeSeconds": teams_channel.get("cycleAgeSeconds"),
            "cycleCount": teams_channel.get("cycleCount", 0),
            "workerRestarts": teams_channel.get("workerRestarts", 0),
            "consecutiveTransportFailures": teams_channel.get(
                "consecutiveTransportFailures", 0
            ),
            "lastSendTs": teams_channel.get("lastSendTs", 0),
        },
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


@router.get("/api/teams-readiness", include_in_schema=False)
def get_teams_readiness() -> JSONResponse:
    """Live-Nachweis: sind alle Voraussetzungen des Teams-Kanals erfuellt?

    Prueft die reale Kette (Kandidatenfeld inkl. Score-API, Push-Historie,
    Slot-Planung, Ruhezeit, Webhook-Konfiguration) gegen die laufende Instanz.
    """
    import datetime as dt
    import time as _time
    from zoneinfo import ZoneInfo

    from app.notifications.teams import (
        TeamsAlertConfig,
        _daily_runtime_opportunities,
        _quiet_hours_reason,
        build_teams_alert_context,
    )

    now_ts = int(_time.time())
    config = TeamsAlertConfig()
    berlin_now = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin"))

    score_api: dict = {"enabled": bool(config.require_internal_score_api)}
    candidates: list = []
    try:
        from app.routers.feed import build_articles_payload

        payload = build_articles_payload(
            offset=0,
            limit=12,
            include_teams_decisions=False,
            use_internal_score_api=config.require_internal_score_api,
        )
        candidates = payload.get("articles") or []
        sources: dict[str, int] = {}
        for item in candidates:
            source = str(item.get("scoreSource") or "unknown")
            sources[source] = sources.get(source, 0) + 1
        score_api["checkedCandidates"] = len(candidates)
        score_api["sources"] = dict(sorted(sources.items()))
        score_api["freshCanonicalScores"] = sources.get("internal_score_api", 0)
        score_api["outageBuffered"] = sum(
            1 for item in candidates if item.get("scoreServedFromOutageBuffer")
        )
        score_api["ok"] = bool(
            not config.require_internal_score_api
            or sources.get("internal_score_api", 0) > 0
        )
    except Exception as exc:  # pragma: no cover - reine Diagnose
        score_api["ok"] = False
        score_api["error"] = type(exc).__name__

    history_info: dict = {}
    try:
        context = build_teams_alert_context(candidates, now_ts=now_ts, config=config)
        last_push_ts = int(context.get("lastPushTs") or 0)
        history_info = {
            "ok": last_push_ts > 0,
            "lastPushAgeMinutes": (
                round((now_ts - last_push_ts) / 60) if last_push_ts > 0 else None
            ),
            "pushesToday": context.get("pushesToday"),
            "sportPushesToday": context.get("sportPushesToday"),
            "historyAuthoritative": bool(context.get("historyAuthoritative")),
        }
    except Exception as exc:  # pragma: no cover - reine Diagnose
        history_info = {"ok": False, "error": type(exc).__name__}

    slots_info: dict = {}
    try:
        slots = _daily_runtime_opportunities(berlin_now.date(), config)
        upcoming = [slot for slot in slots if int(slot.get("ts") or 0) > now_ts]
        next_slot = min(upcoming, key=lambda s: int(s.get("ts") or 0)) if upcoming else None
        slots_info = {
            "ok": 11 <= len(slots) <= 17,
            "plannedToday": len(slots),
            "labels": [str(slot.get("label")) for slot in slots],
            "nextSlot": (
                {
                    "label": str(next_slot.get("label")),
                    "role": str(next_slot.get("slotRole") or ""),
                }
                if next_slot
                else None
            ),
        }
    except Exception as exc:  # pragma: no cover - reine Diagnose
        slots_info = {"ok": False, "error": type(exc).__name__}

    quiet_reason = _quiet_hours_reason(now_ts, config)
    from app.notifications.teams import channel_configuration_problems, channel_health

    runtime = channel_health(config, now_ts=now_ts)
    config_problems = channel_configuration_problems(config)
    ready = bool(
        config.enabled
        and bool(config.webhook_url)
        and score_api.get("ok")
        and history_info.get("ok")
        and slots_info.get("ok")
        and runtime.get("healthy", True)
        and not config_problems
    )
    return JSONResponse(
        content={
            "ready": ready,
            "berlinTime": berlin_now.strftime("%Y-%m-%d %H:%M"),
            "teamsAlertsEnabled": bool(config.enabled),
            "webhookConfigured": bool(config.webhook_url),
            "quietHoursActive": bool(quiet_reason),
            "quietHoursReason": quiet_reason or None,
            "volume": {
                "min": int(config.min_alerts_per_day),
                "max": int(config.max_alerts_per_day),
                "sportMin": int(config.sport_min_per_day),
                "sportMax": int(config.sport_max_per_day),
            },
            "scoreApi": score_api,
            "pushHistory": history_info,
            "slots": slots_info,
            "runtime": {
                "status": runtime.get("status"),
                "reason": runtime.get("reason") or "",
                "cycleAgeSeconds": runtime.get("cycleAgeSeconds"),
                "cycleCount": runtime.get("cycleCount"),
                "consecutiveCycleErrors": runtime.get("consecutiveCycleErrors"),
                "consecutiveTransportFailures": runtime.get(
                    "consecutiveTransportFailures"
                ),
                "workerRestarts": runtime.get("workerRestarts"),
                "lastSendTs": runtime.get("lastSendTs"),
            },
            "configurationProblems": config_problems,
        }
    )


@router.get("/api/ready", include_in_schema=False)
def get_ready() -> JSONResponse:
    """Leichte Readiness-Probe fuer die volle App (K8s/Flux-Deployment)."""
    return JSONResponse(content={"ready": True})


@router.get("/api/teams-effectiveness", include_in_schema=False)
def get_teams_effectiveness(days: int = 14) -> JSONResponse:
    """Wirkungsnachweis: Annahmequote und Opening-Rate-Vergleich des Kanals."""
    from app.notifications.teams_effectiveness import build_effectiveness_report

    safe_days = max(1, min(int(days or 14), 90))
    return JSONResponse(content=build_effectiveness_report(safe_days))
