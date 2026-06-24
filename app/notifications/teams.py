"""Microsoft Teams recommendation alerts for truly push-worthy candidates.

The module separates editorial decisioning, message construction, persistence,
and transport so UI refreshes can explain decisions without sending messages.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any
from zoneinfo import ZoneInfo

from app.config import (
    PUSH_TEAMS_ACTIVE_HOURS_END,
    PUSH_TEAMS_ACTIVE_HOURS_START,
    PUSH_TEAMS_ALERT_COOLDOWN_MINUTES,
    PUSH_TEAMS_ALERTS_ENABLED,
    PUSH_TEAMS_ALLOWED_SECTIONS,
    PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_BREAKING_MIN_OR,
    PUSH_TEAMS_BREAKING_MIN_SCORE,
    PUSH_TEAMS_BREAKING_OVERRIDE,
    PUSH_TEAMS_CANDIDATE_LIMIT,
    PUSH_TEAMS_CONSTANT_FORECAST_MIN_FIELD,
    PUSH_TEAMS_DASHBOARD_TOP_LIMIT,
    PUSH_TEAMS_DEFAULT_REACH,
    PUSH_TEAMS_DAILY_PLAN_MAX_ITEMS,
    PUSH_TEAMS_DAILY_PLAN_MIN_ITEMS,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE,
    PUSH_TEAMS_EDITORIAL_GATE_ENABLED,
    PUSH_TEAMS_EDITORIAL_TOP_LIMIT,
    PUSH_TEAMS_EVENT_GATE_ENABLED,
    PUSH_TEAMS_EXCLUDED_SECTIONS,
    PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES,
    PUSH_TEAMS_KNOWN_DEFAULT_FORECASTS,
    PUSH_TEAMS_KNOWN_DEFAULT_MIN_FIELD,
    PUSH_TEAMS_LLM_TITLE_ENABLED,
    PUSH_TEAMS_MAX_ALERTS_PER_DAY,
    PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS,
    PUSH_TEAMS_MAX_PUSHES_LAST_6H,
    PUSH_TEAMS_MIN_ALERTS_PER_DAY,
    PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE,
    PUSH_TEAMS_MIN_EDITORIAL_SCORE,
    PUSH_TEAMS_MIN_ALERT_SCORE,
    PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE,
    PUSH_TEAMS_MIN_OR,
    PUSH_TEAMS_MIN_SCORE,
    PUSH_TEAMS_MIN_SELECTION_MARGIN,
    PUSH_TEAMS_MIN_TIME_FIT_SCORE,
    PUSH_TEAMS_QUIET_HOURS_END,
    PUSH_TEAMS_QUIET_HOURS_START,
    PUSH_TEAMS_REALERT_OR_DELTA,
    PUSH_TEAMS_REALERT_SCORE_DELTA,
    PUSH_TEAMS_REQUIRE_ARTICLE_FORECAST,
    PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS,
    PUSH_TEAMS_REQUIRE_VALID_PREDICTION,
    PUSH_TEAMS_FEED_OVERTAKEN_ENABLED,
    PUSH_TEAMS_SCORE_ONLY_MODE,
    PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER,
    PUSH_TEAMS_SPECULATIVE_GUARD_ENABLED,
    PUSH_TEAMS_SPECULATIVE_MAX_AGE_HOURS,
    PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS,
    PUSH_TEAMS_TARGET_PUSHES_PER_DAY,
    PUSH_TEAMS_TOPIC_DEDUP_HOURS,
    PUSH_TEAMS_TOPIC_DEDUP_SIMILARITY,
    PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED,
    PUSH_TEAMS_VISIT_SELECTION_WEIGHT,
    PUSH_TEAMS_WEBHOOK_URL,
)
from app.database import (
    push_db_load_all,
    teams_alert_last_sent_ts,
    teams_alert_list_recent,
    teams_alert_load_for_keys,
    teams_alert_record,
    teams_alert_sent_count_since,
    teams_alert_try_claim_send,
)

log = logging.getLogger("push-balancer")

_RECENT_SEND_LOCK = threading.Lock()
_RECENT_SEND_MEMORY: dict[str, dict[str, Any]] = {}

_TOKEN_RE = re.compile(r"[a-z0-9äöüßaeoeue]{4,}", re.IGNORECASE)
_STOP_WORDS = {
    "aber", "alle", "auch", "auf", "aus", "bei", "das", "dass", "dem", "den",
    "der", "des", "die", "dies", "diese", "dieser", "doch", "eine", "einem",
    "einen", "einer", "fuer", "für", "hat", "haben", "hier", "ist", "jetzt",
    "kann", "mit", "nach", "nicht", "noch", "oder", "sich", "sind", "ueber",
    "über", "und", "von", "vor", "war", "was", "weil", "wenn", "wie", "wird",
    "wurde", "zum", "zur",
}


@dataclass(frozen=True)
class TeamsAlertConfig:
    enabled: bool = PUSH_TEAMS_ALERTS_ENABLED
    webhook_url: str = PUSH_TEAMS_WEBHOOK_URL
    min_score: float = PUSH_TEAMS_MIN_SCORE
    min_alert_score: float = PUSH_TEAMS_MIN_ALERT_SCORE
    score_only_mode: bool = PUSH_TEAMS_SCORE_ONLY_MODE
    dashboard_top_limit: int = PUSH_TEAMS_DASHBOARD_TOP_LIMIT
    no_forecast_min_alert_score: float = PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE
    editorial_gate_enabled: bool = PUSH_TEAMS_EDITORIAL_GATE_ENABLED
    editorial_top_limit: int = PUSH_TEAMS_EDITORIAL_TOP_LIMIT
    event_gate_enabled: bool = PUSH_TEAMS_EVENT_GATE_ENABLED
    llm_title_enabled: bool = PUSH_TEAMS_LLM_TITLE_ENABLED
    min_editorial_score: float = PUSH_TEAMS_MIN_EDITORIAL_SCORE
    min_editorial_news_value: float = PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE
    min_time_fit_score: float = PUSH_TEAMS_MIN_TIME_FIT_SCORE
    quiet_hours_start: str = PUSH_TEAMS_QUIET_HOURS_START
    quiet_hours_end: str = PUSH_TEAMS_QUIET_HOURS_END
    min_or: float = PUSH_TEAMS_MIN_OR
    min_minutes_since_last_push: int = PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH
    realert_score_delta: float = PUSH_TEAMS_REALERT_SCORE_DELTA
    realert_or_delta: float = PUSH_TEAMS_REALERT_OR_DELTA
    alert_cooldown_minutes: int = PUSH_TEAMS_ALERT_COOLDOWN_MINUTES
    repeat_suppression_hours: int = PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS
    global_cooldown_minutes: int = PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES
    allowed_sections: tuple[str, ...] = tuple(PUSH_TEAMS_ALLOWED_SECTIONS)
    excluded_sections: tuple[str, ...] = tuple(PUSH_TEAMS_EXCLUDED_SECTIONS)
    breaking_override: bool = PUSH_TEAMS_BREAKING_OVERRIDE
    breaking_min_score: float = PUSH_TEAMS_BREAKING_MIN_SCORE
    breaking_min_or: float = PUSH_TEAMS_BREAKING_MIN_OR
    breaking_min_minutes_since_last_push: int = PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH
    max_article_age_hours: int = PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS
    max_pushes_last_6h: int = PUSH_TEAMS_MAX_PUSHES_LAST_6H
    require_valid_prediction: bool = PUSH_TEAMS_REQUIRE_VALID_PREDICTION
    require_article_forecast: bool = PUSH_TEAMS_REQUIRE_ARTICLE_FORECAST
    known_default_forecasts: tuple[float, ...] = tuple(PUSH_TEAMS_KNOWN_DEFAULT_FORECASTS)
    constant_forecast_min_field: int = PUSH_TEAMS_CONSTANT_FORECAST_MIN_FIELD
    known_default_min_field: int = PUSH_TEAMS_KNOWN_DEFAULT_MIN_FIELD
    target_pushes_per_day: int = PUSH_TEAMS_TARGET_PUSHES_PER_DAY
    min_alerts_per_day: int = PUSH_TEAMS_MIN_ALERTS_PER_DAY
    max_alerts_per_day: int = PUSH_TEAMS_MAX_ALERTS_PER_DAY
    daily_plan_min_items: int = PUSH_TEAMS_DAILY_PLAN_MIN_ITEMS
    daily_plan_max_items: int = PUSH_TEAMS_DAILY_PLAN_MAX_ITEMS
    dynamic_threshold_enabled: bool = PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED
    dynamic_threshold_max_drop: float = PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP
    dynamic_threshold_max_rise: float = PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE
    active_hours_start: int = PUSH_TEAMS_ACTIVE_HOURS_START
    active_hours_end: int = PUSH_TEAMS_ACTIVE_HOURS_END
    min_selection_margin: float = PUSH_TEAMS_MIN_SELECTION_MARGIN
    selection_clear_editorial_buffer: float = PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER
    visit_optimization_enabled: bool = PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED
    visit_selection_weight: float = PUSH_TEAMS_VISIT_SELECTION_WEIGHT
    default_reach: int = PUSH_TEAMS_DEFAULT_REACH
    speculative_guard_enabled: bool = PUSH_TEAMS_SPECULATIVE_GUARD_ENABLED
    speculative_max_age_hours: float = PUSH_TEAMS_SPECULATIVE_MAX_AGE_HOURS
    feed_overtaken_enabled: bool = PUSH_TEAMS_FEED_OVERTAKEN_ENABLED
    topic_dedup_hours: float = PUSH_TEAMS_TOPIC_DEDUP_HOURS
    topic_dedup_similarity: float = PUSH_TEAMS_TOPIC_DEDUP_SIMILARITY
    pushed_topic_window_hours: float = PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS


def candidate_key(candidate: dict[str, Any]) -> str:
    url = _normalize_url(_url(candidate))
    if url:
        return url
    raw_id = str(candidate.get("id") or candidate.get("articleId") or "").strip()
    if raw_id.startswith(("http://", "https://")):
        return _normalize_url(raw_id)
    return raw_id or _title(candidate)


def title_hash(candidate: dict[str, Any]) -> str:
    value = _title(candidate).strip().lower()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _effective_global_cooldown_minutes(config: TeamsAlertConfig) -> int:
    configured = int(config.global_cooldown_minutes or 0)
    if configured <= 0:
        return 0
    return max(configured, int(config.min_minutes_since_last_push or 0), 45)


def _memory_send_blocker_or_reserve(
    *,
    article_key: str,
    title: str,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Process-local last line of defence against duplicate Teams sends.

    The SQLite claim is authoritative across workers. This in-memory reservation
    additionally protects one running process from rapid repeated cycles before
    the external Teams webhook result is fully recorded.
    """
    cooldown_seconds = _effective_global_cooldown_minutes(config) * 60
    article_seconds = max(int(config.alert_cooldown_minutes or 0) * 60, cooldown_seconds)
    topic_seconds = max(int(float(config.topic_dedup_hours or 0.0) * 3600), article_seconds)
    keep_seconds = max(topic_seconds, 3600)
    title_tokens = _tokens(title)

    with _RECENT_SEND_LOCK:
        stale_before = now_ts - keep_seconds
        for key, entry in list(_RECENT_SEND_MEMORY.items()):
            if _safe_int(entry.get("ts")) < stale_before:
                _RECENT_SEND_MEMORY.pop(key, None)

        for key, entry in _RECENT_SEND_MEMORY.items():
            entry_ts = _safe_int(entry.get("ts"))
            if not entry_ts:
                continue
            age = now_ts - entry_ts
            if cooldown_seconds > 0 and age < cooldown_seconds:
                return {
                    "blocked": True,
                    "reason": "memory_global_alert_cooldown",
                    "ageSeconds": age,
                    "otherKey": key,
                }
            if key == article_key and age < article_seconds:
                return {
                    "blocked": True,
                    "reason": "memory_article_alert_cooldown",
                    "ageSeconds": age,
                    "otherKey": key,
                }
            other_tokens = set(entry.get("tokens") or set())
            threshold = min(float(config.topic_dedup_similarity or 0.5), 0.45)
            if title_tokens and _same_topic(title_tokens, other_tokens, threshold) and age < topic_seconds:
                return {
                    "blocked": True,
                    "reason": "memory_topic_duplicate",
                    "ageSeconds": age,
                    "otherKey": key,
                    "otherTitle": str(entry.get("title") or ""),
                }

        _RECENT_SEND_MEMORY[article_key] = {
            "ts": now_ts,
            "title": title,
            "tokens": title_tokens,
        }
    return {"blocked": False}


def _memory_record_send_result(article_key: str, *, ok: bool, now_ts: int) -> None:
    with _RECENT_SEND_LOCK:
        entry = _RECENT_SEND_MEMORY.get(article_key)
        if entry is not None:
            entry["status"] = "sent" if ok else "failed"
            entry["ts"] = now_ts


def build_teams_alert_context(
    candidates: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
    alert_state: dict[str, dict[str, Any]] | None = None,
    last_teams_alert_ts: int | None = None,
    teams_alerts_today: int | None = None,
    recent_alerts: list[dict[str, Any]] | None = None,
    now_ts: int | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    if history is None:
        try:
            history = push_db_load_all(max_days=7, max_rows=500)
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load push history: %s", exc)
            history = []
    if alert_state is None:
        try:
            alert_state = teams_alert_load_for_keys([candidate_key(item) for item in candidates])
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load alert state: %s", exc)
            alert_state = {}
    if last_teams_alert_ts is None:
        try:
            last_teams_alert_ts = teams_alert_last_sent_ts()
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load global alert cooldown state: %s", exc)
            last_teams_alert_ts = 0

    last_push_ts = 0
    for item in history:
        try:
            ts = int(item.get("ts_num", item.get("ts", 0)) or 0)
        except (TypeError, ValueError):
            continue
        last_push_ts = max(last_push_ts, ts)

    recent_6h_count = sum(
        1
        for item in history
        if _safe_int(item.get("ts_num", item.get("ts", 0))) >= now - 6 * 3600
    )

    day_start = _local_day_start_ts(now)
    pushes_today = sum(
        1
        for item in history
        if _safe_int(item.get("ts_num", item.get("ts", 0))) >= day_start
    )

    alerts_today = teams_alerts_today
    if alerts_today is None:
        try:
            alerts_today = teams_alert_sent_count_since(day_start)
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load alert-of-day count: %s", exc)
            alerts_today = 0

    if recent_alerts is None:
        window_start = now - int(max(0.0, config.topic_dedup_hours) * 3600)
        recent_alerts = []
        try:
            for row in teams_alert_list_recent(limit=40):
                if str(row.get("status") or "") != "sent":
                    continue
                if _safe_int(row.get("last_alert_ts")) < window_start:
                    continue
                recent_alerts.append(
                    {"key": str(row.get("article_key") or ""), "title": str(row.get("article_title") or "")}
                )
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load recent alert titles: %s", exc)
            recent_alerts = []

    return {
        "nowTs": now,
        "history": history,
        "alertState": alert_state,
        "lastPushTs": last_push_ts,
        "lastTeamsAlertTs": int(last_teams_alert_ts or 0),
        "recentPushCount6h": recent_6h_count,
        "pushesToday": pushes_today,
        "teamsAlertsToday": int(alerts_today or 0),
        "recentTeamsAlerts": recent_alerts,
        "suspectForecastValues": _detect_suspect_forecast_values(candidates, config),
        "reachStats": _reach_baselines(history, now, config),
    }


def _detect_suspect_forecast_values(
    candidates: list[dict[str, Any]],
    config: TeamsAlertConfig,
) -> list[float]:
    """Find OR forecast values that look like a constant default, not a real prediction.

    A value that repeats across the candidate field (e.g. a global-average default
    such as 4.77 %) is treated as not belastbar. Known defaults trip at a lower
    repetition count than unknown ones.
    """
    counts: dict[float, int] = {}
    for candidate in candidates or []:
        if bool(candidate.get("predictedORIsFallback")):
            continue
        basis = str(candidate.get("predictedORBasis") or "").strip().lower()
        if basis in {"global_avg", "error_fallback"}:
            continue
        value = normalize_predicted_or(
            candidate.get("predictedOR", candidate.get("predictedOpenRate"))
        )
        if value is None:
            continue
        rounded = round(value, 2)
        counts[rounded] = counts.get(rounded, 0) + 1

    known = {round(float(v), 2) for v in config.known_default_forecasts}
    min_field = max(2, int(config.constant_forecast_min_field or 3))
    known_min = max(2, int(config.known_default_min_field or 2))
    suspects: list[float] = []
    for value, count in counts.items():
        if count >= min_field or (value in known and count >= known_min):
            suspects.append(value)
    return sorted(suspects)


def should_notify_teams(
    candidate: dict[str, Any],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Return a structured notify/skip decision for one candidate."""
    context = context or {}
    config = config or TeamsAlertConfig()
    now_ts = int(context.get("nowTs") or time.time())
    key = candidate_key(candidate)
    url = _url(candidate)
    title = _title(candidate)
    section = _section(candidate)
    score = _score(candidate)
    suspect_values = {float(v) for v in (context.get("suspectForecastValues") or [])}
    forecast = _candidate_forecast(candidate, now_ts, suspect_values)
    predicted_or = forecast["value"]
    raw_model_or = normalize_predicted_or(
        candidate.get("predictedOR", candidate.get("predictedOpenRate"))
    )
    forecast_suspected_default = bool(
        raw_model_or is not None
        and forecast.get("source") != "article_model"
        and any(abs(raw_model_or - suspect) <= 0.01 for suspect in suspect_values)
    )
    breaking = _is_breaking(candidate)
    min_score, min_or, min_pause = _effective_thresholds(config, breaking)
    last_push_ts = _safe_int(context.get("lastPushTs"))
    last_teams_alert_ts = _safe_int(context.get("lastTeamsAlertTs"))
    minutes_since_last_push = (
        round((now_ts - last_push_ts) / 60, 1) if last_push_ts > 0 and now_ts >= last_push_ts else None
    )
    minutes_since_last_teams_alert = (
        round((now_ts - last_teams_alert_ts) / 60, 1)
        if last_teams_alert_ts > 0 and now_ts >= last_teams_alert_ts
        else None
    )
    alert_state = (context.get("alertState") or {}).get(key)
    dashboard_rank = _safe_int(context.get("dashboardRank"))
    dashboard_top_limit = max(1, int(config.dashboard_top_limit or PUSH_TEAMS_CANDIDATE_LIMIT))

    positive: list[str] = []
    blockers: list[str] = []
    status = "skip"
    candidate_score_reason = _score_reason(candidate)
    candidate_drivers = _editorial_list(candidate, "performanceDrivers")
    candidate_risks = _editorial_list(candidate, "risks")
    candidate_breakdown = candidate.get("scoreBreakdown") if isinstance(candidate.get("scoreBreakdown"), dict) else {}

    if not config.enabled:
        blockers.append("Teams Alerts deaktiviert")

    quiet_reason = _quiet_hours_reason(now_ts, config)
    if quiet_reason:
        blockers.append(quiet_reason)
        status = "observe"

    if not title:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Headline")
    if not url:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Artikel-Link")

    if dashboard_rank > 0:
        if dashboard_rank <= dashboard_top_limit:
            positive.append(f"Top-Kandidat im Push Balancer: Rang {dashboard_rank}")
        else:
            blockers.append(
                f"Nicht im oberen Push-Balancer-Feld: Rang {dashboard_rank} > {dashboard_top_limit}"
            )

    excluded = {item.lower() for item in config.excluded_sections if item.strip()}
    if section.lower() in excluded:
        blockers.append(f"Ressort {section} ist fuer Teams Alerts ausgeschlossen")
    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if allowed and section.lower() not in allowed:
        blockers.append(f"Ressort {section} nicht fuer Teams Alerts freigegeben")

    pushes_today = context.get("pushesToday")
    pushes_today = _safe_int(pushes_today) if pushes_today is not None else None
    teams_alerts_today = _safe_int(context.get("teamsAlertsToday"))
    push_pacing = _push_pacing_review(pushes_today, now_ts, config)
    minimum_pressure = _minimum_pressure_review(push_pacing, teams_alerts_today, now_ts, config)
    minimum_active = bool(minimum_pressure.get("active"))
    minimum_pressure_value = float(minimum_pressure.get("pressure") or 0.0)
    effective_min_score = min_score
    if minimum_active and not breaking:
        effective_min_score = max(65.0 if minimum_pressure_value >= 3.0 else 68.0, min_score - 7.0)
        positive.append(
            f"Teams-Mindest-Erfuellung: Score-Schwelle kontrolliert {min_score:.1f} -> "
            f"{effective_min_score:.1f}"
        )

    if score >= effective_min_score:
        positive.append(f"Push Score {score:.1f} liegt ueber Schwelle {effective_min_score:.1f}")
        if candidate_score_reason:
            positive.append(f"Push-Score-Begruendung: {candidate_score_reason}")
        positive.extend(candidate_drivers[:3])
    else:
        blockers.append(f"Score zu niedrig: {score:.1f} < {effective_min_score:.1f}")

    alert_model = _teams_alert_score(
        candidate,
        score=score,
        predicted_or=predicted_or,
        freshness_hours=_freshness_hours(candidate, now_ts),
        minutes_since_last_push=minutes_since_last_push,
        recent_push_count_6h=int(context.get("recentPushCount6h") or 0),
        pushes_today=pushes_today,
        now_ts=now_ts,
        config=config,
    )
    alert_score = float(alert_model["score"])
    positive.append(f"Teams Alert Score {alert_score:.1f}/100")
    positive.extend(list(alert_model["reasons"])[:4])
    effective_min_alert_score, push_budget_reason = _dynamic_alert_threshold(
        config.min_alert_score,
        pushes_today,
        now_ts,
        breaking,
        config,
    )
    if push_budget_reason:
        positive.append(push_budget_reason)
    if minimum_pressure["active"]:
        floor_drop = float(minimum_pressure.get("thresholdDrop") or 0.0)
        effective_min_alert_score = max(66.0, effective_min_alert_score - floor_drop)
        positive.append(str(minimum_pressure["label"]))
    if alert_score < effective_min_alert_score:
        blockers.append(
            f"Teams Alert Score zu niedrig: {alert_score:.1f} < {effective_min_alert_score:.1f}"
        )
    effective_no_forecast_min_alert_score = config.no_forecast_min_alert_score
    if minimum_active and not breaking:
        effective_no_forecast_min_alert_score = max(68.0, config.no_forecast_min_alert_score - 8.0)
    if predicted_or is None and alert_score < effective_no_forecast_min_alert_score:
        blockers.append(
            "Keine belastbare Prognose und Teams Alert Score nicht hoch genug: "
            f"{alert_score:.1f} < {effective_no_forecast_min_alert_score:.1f}"
        )
    forecast_is_reliable = forecast.get("source") == "article_model" and predicted_or is not None
    effective_min_or = min_or
    if minimum_active and forecast_is_reliable and not breaking:
        effective_min_or = max(4.3, min_or - 0.7)
        positive.append(
            f"Teams-Mindest-Erfuellung: OR-Schwelle kontrolliert {min_or:.2f}% -> "
            f"{effective_min_or:.2f}%"
        )
    low_forecast_blocker = (
        predicted_or is not None
        and predicted_or < effective_min_or
        and (forecast_is_reliable or not config.score_only_mode)
        and not (breaking and config.breaking_override and predicted_or >= config.breaking_min_or)
    )
    if low_forecast_blocker:
        blockers.append(f"Prognose zu niedrig: {predicted_or:.2f}% OR < {effective_min_or:.2f}%")
    if config.require_valid_prediction and not forecast_is_reliable:
        blockers.append("Belastbare OR-Prognose erforderlich, aktuell nur Fallback verfuegbar")
    forecast_quality = _forecast_quality_review(
        candidate, forecast, alert_score, breaking, config, minimum_pressure
    )
    positive.extend(forecast_quality["reasons"])
    blockers.extend(forecast_quality["blockers"])

    freshness_hours = _freshness_hours(candidate, now_ts)
    editorial_review = _editorial_cvd_review(
        candidate,
        score=score,
        predicted_or=predicted_or,
        freshness_hours=freshness_hours,
        minutes_since_last_push=minutes_since_last_push,
        dashboard_rank=dashboard_rank,
        alert_score=alert_score,
        pushes_today=pushes_today,
        now_ts=now_ts,
        config=config,
        minimum_pressure=minimum_pressure,
    )
    positive.extend(editorial_review["reasons"])
    blockers.extend(editorial_review["blockers"])
    strategy_review = _daily_strategy_review(
        candidate,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        news_value=float(editorial_review["newsValue"]),
        predicted_or=predicted_or,
        push_pacing=push_pacing,
        breaking=breaking,
        config=config,
        minimum_pressure=minimum_pressure,
    )
    positive.extend(strategy_review["reasons"])
    blockers.extend(strategy_review["blockers"])
    visit_potential = _visit_potential(
        candidate,
        predicted_or=predicted_or,
        editorial_score=float(editorial_review["score"]),
        alert_score=alert_score,
        score=score,
        breaking=breaking,
        now_ts=now_ts,
        reach_stats=context.get("reachStats") if isinstance(context.get("reachStats"), dict) else {},
        config=config,
    )
    if visit_potential.get("reason"):
        positive.append(str(visit_potential["reason"]))
    selection_score = _recommendation_selection_score(
        score=score,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        predicted_or=predicted_or,
        dashboard_rank=dashboard_rank,
        breaking=breaking,
        visit_score=float(visit_potential.get("score") or 0.0),
        config=config,
    )

    if config.score_only_mode:
        positive.append("Score-Modus aktiv: Teams Alert Score entscheidet final")
        if minutes_since_last_push is None:
            blockers.append("Letzter Push-Zeitpunkt nicht verfuegbar")
            status = "observe"
        elif minutes_since_last_push >= min_pause:
            positive.append(
                f"Letzter Push vor {minutes_since_last_push:.0f} Minuten, Mindestpause erfuellt"
            )
        else:
            blockers.append(
                f"Pause seit letztem Push zu kurz: {minutes_since_last_push:.0f} < {min_pause} Minuten"
            )
            status = "observe"
    else:
        if predicted_or is None:
            blockers.append("Prognose fehlt")
        elif predicted_or >= effective_min_or:
            positive.append(
                f"Prognose {predicted_or:.2f}% OR liegt ueber Mindestwert {effective_min_or:.2f}%"
            )

        if minutes_since_last_push is None:
            blockers.append("Letzter Push-Zeitpunkt nicht verfuegbar")
            status = "observe"
        elif minutes_since_last_push >= min_pause:
            positive.append(
                f"Letzter Push vor {minutes_since_last_push:.0f} Minuten, Mindestpause erfuellt"
            )
        else:
            blockers.append(
                f"Pause seit letztem Push zu kurz: {minutes_since_last_push:.0f} < {min_pause} Minuten"
            )
            status = "observe"

    if freshness_hours is not None:
        if freshness_hours <= config.max_article_age_hours:
            positive.append(f"Aktuell: Artikel vor {freshness_hours:.1f} Stunden veroeffentlicht")
        elif not config.score_only_mode:
            blockers.append(
                f"Artikel nicht frisch genug: {freshness_hours:.1f}h > {config.max_article_age_hours}h"
            )

    # Spekulative/erwartete Lagen koennen von der Realitaet ueberholt sein
    # (z. B. "bereitet wohl Ruecktritt vor", obwohl bereits zurueckgetreten).
    # Ohne externe Faktenpruefung ist das nur eine Heuristik: aelter als die
    # Schwelle -> nicht mehr pushen; frisch -> als Risiko markieren.
    is_speculative = bool(config.speculative_guard_enabled and _is_speculative(title))
    speculative_caution = ""
    overtaken_reason = _overtaken_by_feed(title, config) if is_speculative else ""
    if overtaken_reason:
        # Hartes Signal: eine frischere Quelle meldet die Lage bereits als vollzogen.
        blockers.append(overtaken_reason)
    if is_speculative and not breaking:
        if freshness_hours is not None and freshness_hours > config.speculative_max_age_hours:
            blockers.append(
                "Spekulative/erwartete Lage ('wohl', 'bereitet vor', 'soll zuruecktreten') "
                f"und nicht mehr frisch ({freshness_hours:.1f}h) - wahrscheinlich ueberholt, nicht pushen"
            )
        else:
            speculative_caution = (
                "Spekulative/erwartete Lage - vor Push gegen die aktuelle Meldungslage pruefen "
                "(koennte bereits ueberholt sein)"
            )
            candidate_risks = [speculative_caution, *candidate_risks]

    if (
        not config.score_only_mode
        and int(context.get("recentPushCount6h") or 0) > config.max_pushes_last_6h
        and not breaking
    ):
        blockers.append("Push-Dichte in den letzten 6 Stunden zu hoch")

    duplicate_reason = _already_pushed_reason(candidate, context.get("history") or [], now_ts, config)
    if duplicate_reason:
        blockers.append(duplicate_reason)

    topic_dup_reason = _topic_already_alerted_reason(
        candidate, key, context.get("recentTeamsAlerts") or [], config
    )
    if topic_dup_reason:
        blockers.append(topic_dup_reason)

    realert_reason = _realert_blocker_or_reason(
        candidate,
        alert_state,
        score,
        predicted_or,
        breaking,
        now_ts,
        config,
    )
    if realert_reason.get("blocker"):
        blockers.append(str(realert_reason["blocker"]))
    elif realert_reason.get("positive"):
        positive.append(str(realert_reason["positive"]))

    effective_global_cooldown = _effective_global_cooldown_minutes(config)
    if (
        effective_global_cooldown > 0
        and minutes_since_last_teams_alert is not None
        and minutes_since_last_teams_alert < effective_global_cooldown
    ):
        blockers.append(
            "Teams-Cooldown aktiv: letzter Hinweis vor "
            f"{minutes_since_last_teams_alert:.0f} < {effective_global_cooldown} Minuten"
        )
        status = "observe"

    if (
        config.max_alerts_per_day > 0
        and teams_alerts_today >= config.max_alerts_per_day
        and not (breaking and config.breaking_override)
    ):
        blockers.append(
            f"Tageslimit fuer Teams-Hinweise erreicht: {teams_alerts_today} von "
            f"{config.max_alerts_per_day}"
        )

    stronger = context.get("strongerCandidate")
    if stronger:
        stronger_title = _title(stronger)
        blockers.append(
            "Staerkerer Kandidat vorhanden"
            + (f": {stronger_title}" if stronger_title else "")
        )

    if not blockers:
        status = "notify"
        positive.append("Kein staerkerer Kandidat aktuell verfuegbar")
    elif alert_state and alert_state.get("status") == "sent":
        status = "sent"

    summary = positive[0] if status == "notify" and positive else (blockers[0] if blockers else "")
    if status == "notify" and candidate_score_reason:
        summary = candidate_score_reason

    return {
        "candidateId": key,
        "articleId": str(candidate.get("id") or key),
        "articleUrl": url,
        "headline": title,
        "recommendedAction": "Jetzt pushen" if status == "notify" else "",
        "shouldNotify": status == "notify",
        "status": status,
        "summary": summary,
        "reasons": positive,
        "blockingReasons": blockers,
        "scoreReason": candidate_score_reason,
        "performanceDrivers": candidate_drivers,
        "risks": candidate_risks,
        "isSpeculative": is_speculative,
        "speculativeCaution": speculative_caution,
        "overtakenByFeed": overtaken_reason,
        "scoreBreakdown": candidate_breakdown,
        "score": score,
        "teamsAlertScore": alert_score,
        "teamsAlertScoreThreshold": round(effective_min_alert_score, 1),
        "teamsAlertScoreBaseThreshold": config.min_alert_score,
        "teamsAlertScoreBreakdown": alert_model["breakdown"],
        "pushesToday": pushes_today,
        "teamsAlertsToday": teams_alerts_today,
        "pushBudgetReason": push_budget_reason,
        "pushPacing": push_pacing,
        "minimumPressure": minimum_pressure,
        "minAlertsPerDay": config.min_alerts_per_day,
        "pushBudgetTarget": config.target_pushes_per_day,
        "maxAlertsPerDay": config.max_alerts_per_day,
        "editorialReview": editorial_review,
        "editorialScore": editorial_review["score"],
        "selectionScore": selection_score,
        "visitPotential": visit_potential,
        "expectedVisits": int(round(float(visit_potential.get("expectedVisits") or 0.0))),
        "estimatedReach": int(round(float(visit_potential.get("estimatedReach") or 0.0))),
        "visitPotentialScore": round(float(visit_potential.get("score") or 0.0), 1),
        "predictedOR": predicted_or,
        "forecast": forecast,
        "forecastSuspectedDefault": forecast_suspected_default,
        "forecastSuspectValue": round(raw_model_or, 2) if forecast_suspected_default else None,
        "predictedORSource": forecast["source"],
        "predictedORBasis": forecast["basis"],
        "predictedORConfidence": forecast["confidence"],
        "minScore": min_score,
        "minOR": min_or,
        "minMinutesSinceLastPush": min_pause,
        "minutesSinceLastPush": minutes_since_last_push,
        "dashboardRank": dashboard_rank or None,
        "dashboardTopLimit": dashboard_top_limit,
        "lastPushAt": _iso_from_ts(last_push_ts) if last_push_ts else None,
        "lastGlobalTeamsAlertAt": _iso_from_ts(last_teams_alert_ts) if last_teams_alert_ts else None,
        "minutesSinceLastGlobalTeamsAlert": minutes_since_last_teams_alert,
        "lastTeamsAlertAt": _iso_from_ts(_safe_int(alert_state.get("last_alert_ts"))) if alert_state else None,
        "alertCount": int(alert_state.get("alert_count") or 0) if alert_state else 0,
        "isBreaking": breaking,
        "scoreOnlyMode": config.score_only_mode,
        "section": section,
        "evaluatedAt": _iso_from_ts(now_ts),
    }


def evaluate_teams_alert_candidates(
    candidates: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a batch and mark only the strongest eligible candidate notify."""
    config = config or TeamsAlertConfig()
    context = context or build_teams_alert_context(candidates, config=config)
    base_context = dict(context)
    base_context.pop("strongerCandidate", None)

    top_limit = max(1, int(config.dashboard_top_limit or PUSH_TEAMS_CANDIDATE_LIMIT))
    base_decisions = []
    for index, candidate in enumerate(candidates, start=1):
        decision_context = dict(base_context)
        decision_context["dashboardRank"] = index
        decision_context["dashboardTopLimit"] = top_limit
        base_decisions.append((candidate, should_notify_teams(candidate, decision_context, config)))
    eligible = [
        (candidate, decision)
        for candidate, decision in base_decisions
        if decision.get("shouldNotify")
    ]
    selected_key: str | None = None
    selected_candidate: dict[str, Any] | None = None
    selected_decision: dict[str, Any] | None = None
    if eligible:
        def _selection_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, ...]:
            candidate, decision = item
            if config.visit_optimization_enabled:
                return (
                    float((decision.get("visitPotential") or {}).get("qualityAdjustedVisits") or 0.0),
                    float(decision.get("expectedVisits") or 0.0),
                    float(decision.get("visitPotentialScore") or 0.0),
                    float(decision.get("selectionScore") or 0.0),
                    float(decision.get("editorialScore") or 0.0),
                    float(decision.get("teamsAlertScore") or 0.0),
                    *_candidate_rank(candidate),
                )
            return (
                float(decision.get("selectionScore") or 0.0),
                float(decision.get("editorialScore") or 0.0),
                float(decision.get("teamsAlertScore") or 0.0),
                *_candidate_rank(candidate),
            )

        selected_candidate, selected_decision = max(
            eligible,
            key=_selection_key,
        )
        selected_key = candidate_key(selected_candidate)

    # "Klarer Gewinner"-Regel: wenn der Top-Kandidat nur knapp vor dem Verfolger
    # liegt und selbst nicht eindeutig stark ist, ist das Feld unsicher -> kein Alert.
    uncertainty_reason = ""
    if (
        selected_decision is not None
        and len(eligible) >= 2
        and config.min_selection_margin > 0
        and not bool(selected_decision.get("isBreaking"))
        and not bool((selected_decision.get("minimumPressure") or {}).get("active"))
    ):
        runner_up = max(
            (
                decision
                for candidate, decision in eligible
                if candidate_key(candidate) != selected_key
            ),
            key=lambda decision: (
                float(decision.get("expectedVisits") or 0.0)
                if config.visit_optimization_enabled
                else float(decision.get("selectionScore") or 0.0)
            ),
            default=None,
        )
        if runner_up is not None:
            if config.visit_optimization_enabled:
                winner_value = float(
                    (selected_decision.get("visitPotential") or {}).get("qualityAdjustedVisits") or 0.0
                )
                runner_value = float(
                    (runner_up.get("visitPotential") or {}).get("qualityAdjustedVisits") or 0.0
                )
                # Keep the existing point-based margin semantics by comparing
                # quality-adjusted visits in thousands.
                winner_value /= 1000.0
                runner_value /= 1000.0
            else:
                winner_value = float(selected_decision.get("selectionScore") or 0.0)
                runner_value = float(runner_up.get("selectionScore") or 0.0)
            margin = winner_value - runner_value
            winner_editorial = float(selected_decision.get("editorialScore") or 0.0)
            clear_level = config.min_editorial_score + config.selection_clear_editorial_buffer
            if winner_editorial < clear_level and margin < config.min_selection_margin:
                uncertainty_reason = (
                    f"Feld unsicher: kein klarer Gewinner (Abstand {margin:.1f} < "
                    f"{config.min_selection_margin:.1f} Punkte)"
                )
                selected_key = None
                selected_candidate = None
                selected_decision = None

    final: list[dict[str, Any]] = []
    for index, (candidate, base_decision) in enumerate(base_decisions, start=1):
        key = candidate_key(candidate)
        if uncertainty_reason and base_decision.get("shouldNotify"):
            decision = dict(base_decision)
            decision["shouldNotify"] = False
            decision["status"] = "observe"
            decision["recommendedAction"] = ""
            decision["blockingReasons"] = [
                *decision.get("blockingReasons", []),
                uncertainty_reason,
            ]
            final.append({"candidate": candidate, "decision": decision})
            continue
        if selected_key and key != selected_key and base_decision.get("shouldNotify"):
            decision_context = dict(base_context)
            decision_context["strongerCandidate"] = selected_candidate
            decision_context["dashboardRank"] = index
            decision_context["dashboardTopLimit"] = top_limit
            decision = should_notify_teams(candidate, decision_context, config)
        else:
            decision = dict(base_decision)
            if selected_key and key == selected_key:
                competitors = max(0, len(eligible) - 1)
                decision["competition"] = {
                    "eligibleCompetitors": competitors,
                    "summary": (
                        f"{competitors} weitere pushwuerdige Kandidaten geprueft"
                        if competitors
                        else "Kein staerkerer Kandidat aktuell verfuegbar"
                    ),
                }
                if competitors:
                    selection_reason = (
                        "Hoechstes erwartetes Visit-Potenzial unter den freigegebenen Kandidaten"
                        if config.visit_optimization_enabled
                        else "Beste CvD-Eignung aus Nachrichtenwert, Timing und Nutzerbelastung im Kandidatenfeld"
                    )
                    decision["reasons"] = [
                        *decision.get("reasons", []),
                        selection_reason,
                    ]
        final.append({"candidate": candidate, "decision": decision})

    return {
        "selectedCandidateId": selected_key,
        "selectedCandidate": selected_candidate,
        "decisions": final,
        "fieldUncertain": bool(uncertainty_reason),
        "uncertaintyReason": uncertainty_reason,
        "evaluatedAt": _iso_from_ts(int(context.get("nowTs") or time.time())),
    }


def annotate_candidates_with_teams_decisions(
    candidates: list[dict[str, Any]],
    *,
    config: TeamsAlertConfig | None = None,
    now_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Attach transparent Teams decision metadata without sending anything."""
    if not candidates:
        return candidates
    config = config or TeamsAlertConfig()
    context = build_teams_alert_context(candidates, now_ts=now_ts, config=config)
    evaluations = evaluate_teams_alert_candidates(candidates, context, config)
    by_key = {
        item["decision"]["candidateId"]: item["decision"]
        for item in evaluations["decisions"]
    }
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["teamsAlert"] = by_key.get(candidate_key(candidate))
        annotated.append(item)
    return annotated


def build_teams_push_recommendation(
    candidate: dict[str, Any],
    context: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Build a Teams-compatible message payload for a selected candidate."""
    context = context or {}
    config = config or TeamsAlertConfig()
    decision = decision or should_notify_teams(candidate, context, config)

    title = _title(candidate)
    url = _url(candidate)
    section = _section(candidate)
    section_label = _format_section(section)
    score = _score(candidate)
    now_ts = int(context.get("nowTs") or time.time())
    forecast = decision.get("forecast") if isinstance(decision.get("forecast"), dict) else None
    if not forecast:
        suspect_values = {float(v) for v in (context.get("suspectForecastValues") or [])}
        forecast = _candidate_forecast(candidate, now_ts, suspect_values)
    predicted_or = forecast["value"]
    minutes = decision.get("minutesSinceLastPush")
    minutes_known = isinstance(minutes, (int, float))
    score_only_mode = bool(decision.get("scoreOnlyMode") or config.score_only_mode)
    score_threshold = float(decision.get("minScore", config.min_score) or config.min_score)
    alert_score = float(decision.get("teamsAlertScore") or 0.0)
    alert_threshold = float(decision.get("teamsAlertScoreThreshold") or config.min_alert_score)
    editorial_review = decision.get("editorialReview") or {}
    editorial_score = float(editorial_review.get("score") or 0.0)
    editorial_reasons = list(editorial_review.get("reasons") or [])
    editorial_breakdown = editorial_review.get("breakdown") or {}
    time_fit_score = float(editorial_breakdown.get("timeFit") or 0.0)
    time_fit_label = str(editorial_breakdown.get("timeFitLabel") or "").strip()
    pacing = decision.get("pushPacing") if isinstance(decision.get("pushPacing"), dict) else {}
    pacing_label = str(pacing.get("label") or "").strip()
    selection_score = float(decision.get("selectionScore") or 0.0)
    visit_potential = decision.get("visitPotential") if isinstance(decision.get("visitPotential"), dict) else {}
    visit_reason = str(visit_potential.get("reason") or "").strip()
    push_text, push_title_source = _teams_push_title_recommendation(candidate, title, section, url, config)
    push_text_matches_title = _same_editorial_text(push_text, title)
    competition_meta = decision.get("competition") or {}
    competitors = int(competition_meta.get("eligibleCompetitors") or 0)
    competition = (
        f"Im aktuellen Kandidatenfeld ist das der stärkste Vorschlag ({competitors + 1} geprüft)."
        if competitors
        else "Im aktuellen Kandidatenfeld gibt es keinen stärkeren Push-Vorschlag."
    )
    candidate_score_reason = _score_reason(candidate)
    candidate_drivers = _editorial_list(candidate, "performanceDrivers")
    candidate_risks = _editorial_list(candidate, "risks")
    candidate_breakdown_lines = _score_breakdown_lines(candidate)

    threshold_reason = (
        f"Das Teams-Alert-Modell bewertet den Artikel mit {_format_number(alert_score)} "
        f"von 100 Punkten (Schwelle: {_format_number(alert_threshold, 0)})."
    )
    score_reason = (
        f"Der redaktionelle Push-Score liegt bei {_format_number(score)} und damit über "
        f"dem Mindestwert von {_format_number(score_threshold, 0)}."
    )
    forecast_reason = _forecast_sentence(forecast)
    duplicate_reason = "Der Artikel wurde nicht bereits gepusht und nicht erneut per Teams gemeldet."
    if isinstance(minutes, (int, float)):
        if score_only_mode:
            timing_reason = (
                f"Der letzte Push liegt {float(minutes):.0f} Minuten zurück; die Nutzerbelastung "
                "ist im Teams-Alert-Score berücksichtigt."
            )
        else:
            timing_reason = f"Der letzte Push liegt {float(minutes):.0f} Minuten zurück."
    else:
        timing_reason = "Ein letzter Push-Zeitpunkt ist aktuell nicht bekannt."
    editorial_reason = (
        f"CvD-Einordnung: {_format_number(editorial_score)} von 100 Punkten; "
        f"Auswahlwert {_format_number(selection_score)}."
        if editorial_review.get("approved", True)
        else "CvD-Einordnung: keine redaktionelle Freigabe."
    )
    time_fit_reason = f"Zeitfenster: {time_fit_label}." if time_fit_label else ""
    why_now = _dedupe(
        [
            editorial_reason,
            visit_reason,
            *( [f"Push-Balancer-Score: {candidate_score_reason}"] if candidate_score_reason else [] ),
            time_fit_reason,
            forecast_reason,
            threshold_reason,
            score_reason,
            timing_reason,
            competition,
            duplicate_reason,
        ]
    )[:6]
    why_pushworthy = _dedupe([
        *candidate_drivers[:5],
        editorial_reason,
        score_reason,
        forecast_reason,
        timing_reason,
        competition,
    ])[:7]
    speculative_caution = str(decision.get("speculativeCaution") or "").strip()
    if speculative_caution and speculative_caution not in candidate_risks:
        candidate_risks = [speculative_caution, *candidate_risks]
    what_speaks_against = candidate_risks[:5] or ["Keine harten Gegenargumente im Push-Balancer-Score."]
    # "Warum jetzt?" fuehrt mit der inhaltlichen Substanz (was die Story stark macht),
    # dann Zeitfenster und Prognose. Kein Modell-Jargon ("X von 100").
    compact_reasons = _dedupe(
        [
            *candidate_drivers[:1],
            visit_reason,
            time_fit_reason,
            pacing_label,
            forecast_reason,
            timing_reason,
        ]
    )[:5]
    subject = f"🚨 Jetzt pushen: {_compact_text(push_text or title, 120)}"

    text_lines = [subject, "", "Alternativer Push-Titel:", push_text]
    if not push_text_matches_title:
        text_lines.extend(["", "Artikel:", title])
    if url:
        text_lines.append(url)
    text_lines.extend(
        [
            "",
            (
                f"{section_label} | Score {_format_number(score)} | "
                f"Prognose {_format_or(predicted_or)} | "
                f"Visit-Potenzial ca. {_format_int(visit_potential.get('expectedVisits') or 0)} | "
                f"letzter Push {_format_minutes(minutes)}"
            ),
            "",
            "Warum jetzt?",
            *[f"- {reason}" for reason in compact_reasons],
            "",
            f"Empfehlung: Jetzt pushen. (Stand {_format_time(now_ts)} Uhr)",
        ]
    )
    text = "\n".join(text_lines)
    message_html = _build_power_automate_message_html(
        title=title,
        url=url,
        section=section_label,
        score=score,
        predicted_or=predicted_or,
        forecast=forecast,
        recommended_text=push_text,
        now_ts=now_ts,
        minutes_since_last_push=minutes,
        score_threshold=score_threshold,
        alert_score=alert_score,
        alert_threshold=alert_threshold,
        editorial_score=editorial_score,
        why_now=compact_reasons,
        subject=subject,
        push_text_matches_title=push_text_matches_title,
        score_reason=candidate_score_reason,
        performance_drivers=candidate_drivers,
        risks=what_speaks_against,
        score_breakdown_lines=candidate_breakdown_lines,
    )
    return {
        "text": text,
        "payload": {
            "type": "push_recommendation",
            "subject": subject,
            "recommendedAction": "Jetzt pushen",
            "articleTitle": title,
            "articleUrl": url,
            "category": section,
            "pushScore": score,
            "teamsAlertScore": alert_score,
            "teamsAlertScoreThreshold": alert_threshold,
            "teamsAlertScoreBreakdown": decision.get("teamsAlertScoreBreakdown") or {},
            "editorialReview": editorial_review,
            "editorialScore": editorial_score,
            "editorialReasons": editorial_reasons,
            "scoreReason": candidate_score_reason,
            "performanceDrivers": candidate_drivers,
            "risks": candidate_risks,
            "scoreBreakdown": candidate.get("scoreBreakdown") if isinstance(candidate.get("scoreBreakdown"), dict) else {},
            "scoreBreakdownLabel": "; ".join(candidate_breakdown_lines),
            "timeFitScore": time_fit_score,
            "timeFitLabel": time_fit_label,
            "selectionScore": selection_score,
            "visitPotential": visit_potential,
            "expectedVisits": int(visit_potential.get("expectedVisits") or 0),
            "estimatedReach": int(visit_potential.get("estimatedReach") or 0),
            "visitPotentialScore": float(visit_potential.get("score") or 0.0),
            "predictedOR": round(float(predicted_or), 4) if predicted_or is not None else 0.0,
            "predictedORAvailable": predicted_or is not None,
            "predictedORLabel": _format_or(predicted_or),
            "predictedORSource": forecast["source"],
            "predictedORBasis": forecast["basis"],
            "predictedORConfidence": forecast["confidence"],
            "predictedORExplanation": forecast["explanation"],
            "recommendedPushText": push_text,
            "alternativePushTitle": push_text,
            "pushTitleSource": push_title_source,
            "recommendedAt": _format_dt(now_ts),
            "minutesSinceLastPush": round(float(minutes), 1) if minutes_known else 0.0,
            "lastPushKnown": minutes_known,
            "timeSinceLastPushLabel": _format_minutes(minutes),
            "whyNow": why_now,
            "compactWhyNow": compact_reasons,
            "whyPushworthy": why_pushworthy,
            "competition": competition,
            "messageText": text,
            "messageHtml": message_html,
            "text": text,
        },
        "summary": subject,
    }


def send_teams_notification(
    message: dict[str, Any],
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Send one Teams webhook message and return a non-throwing result."""
    config = config or TeamsAlertConfig()
    if not config.enabled:
        return {"ok": False, "error": "Teams alerts disabled"}
    if not config.webhook_url:
        return {"ok": False, "error": "Teams webhook URL missing"}

    payload = message.get("payload") or {"text": str(message.get("text") or "")}
    try:
        req = urllib.request.Request(
            config.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
            status = getattr(response, "status", 200)
        return {"ok": True, "status": status}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        error = f"HTTP {exc.code}: {body or exc.reason}"
        log.warning("[TeamsAlert] Teams webhook send failed: %s", error)
        return {"ok": False, "error": error}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)
        log.warning("[TeamsAlert] Teams webhook send failed: %s", error)
        return {"ok": False, "error": error}


def evaluate_and_send_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    config: TeamsAlertConfig | None = None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate batch, send the best recommendation, and persist state."""
    config = config or TeamsAlertConfig()
    dashboard_limit = max(1, min(int(config.dashboard_top_limit or 20), PUSH_TEAMS_CANDIDATE_LIMIT))
    limited = candidates[:dashboard_limit]
    context = build_teams_alert_context(limited, now_ts=now_ts, config=config)
    evaluation = evaluate_teams_alert_candidates(limited, context, config)
    selected = evaluation.get("selectedCandidate")
    selected_decision = None

    for item in evaluation["decisions"]:
        candidate = item["candidate"]
        decision = item["decision"]
        _log_decision(candidate, decision)
        if selected and candidate_key(candidate) == candidate_key(selected):
            selected_decision = decision

    if not selected or not selected_decision or not selected_decision.get("shouldNotify"):
        return {"ok": True, "sent": False, "reason": "no_candidate", "evaluation": evaluation}

    article_key = candidate_key(selected)
    article_id = str(selected.get("id") or article_key)
    article_url = _url(selected)
    decision_ts = int(context.get("nowTs") or time.time())
    memory_claim = _memory_send_blocker_or_reserve(
        article_key=article_key,
        title=_title(selected),
        now_ts=decision_ts,
        config=config,
    )
    if memory_claim.get("blocked"):
        log.info(
            "[TeamsAlert] send skipped by memory guard candidateId=%s articleId=%s url=%s reason=%s",
            article_key,
            article_id,
            article_url,
            memory_claim.get("reason"),
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "send_memory_blocked",
            "claim": memory_claim,
            "candidateId": article_key,
            "evaluation": evaluation,
        }
    selected_forecast = _candidate_forecast(
        selected,
        decision_ts,
        {float(v) for v in (context.get("suspectForecastValues") or [])},
    )
    claim = teams_alert_try_claim_send(
        article_key=article_key,
        article_id=article_id,
        article_url=article_url,
        title_hash=title_hash(selected),
        article_title=_title(selected),
        score=_score(selected),
        predicted_or=selected_forecast["value"] or 0.0,
        candidate_updated_at=_candidate_updated_ts(selected),
        is_breaking=_is_breaking(selected),
        reason=selected_decision.get("summary") or "Push empfohlen",
        decision_ts=decision_ts,
        alert_cooldown_minutes=config.alert_cooldown_minutes,
        global_cooldown_minutes=_effective_global_cooldown_minutes(config),
        failed_cooldown_minutes=max(
            config.alert_cooldown_minutes,
            config.repeat_suppression_hours * 60,
        ),
    )
    if not claim.get("claimed"):
        log.info(
            "[TeamsAlert] send skipped by claim candidateId=%s articleId=%s url=%s reason=%s",
            article_key,
            article_id,
            article_url,
            claim.get("reason"),
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "send_claim_blocked",
            "claim": claim,
            "candidateId": article_key,
            "evaluation": evaluation,
        }

    message = build_teams_push_recommendation(selected, context, selected_decision, config)
    send_result = send_teams_notification(message, config)
    _memory_record_send_result(article_key, ok=bool(send_result.get("ok")), now_ts=decision_ts)
    status = "sent" if send_result.get("ok") else "failed"
    reason = selected_decision.get("summary") or "Push empfohlen"
    teams_alert_record(
        article_key=article_key,
        article_id=article_id,
        article_url=article_url,
        title_hash=title_hash(selected),
        article_title=_title(selected),
        score=_score(selected),
        predicted_or=selected_forecast["value"] or 0.0,
        candidate_updated_at=_candidate_updated_ts(selected),
        is_breaking=_is_breaking(selected),
        reason=reason,
        status=status,
        error=str(send_result.get("error") or ""),
        decision_ts=int(context.get("nowTs") or time.time()),
    )

    log.info(
        "[TeamsAlert] send_result candidateId=%s articleId=%s url=%s status=%s ok=%s",
        article_key,
        article_id,
        article_url,
        status,
        bool(send_result.get("ok")),
    )
    return {
        "ok": True,
        "sent": bool(send_result.get("ok")),
        "sendResult": send_result,
        "candidateId": article_key,
        "evaluation": evaluation,
    }


def send_teams_test_notification(
    config: TeamsAlertConfig | None = None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Build and send a clearly marked test message to the configured Teams channel.

    Uses the server-side webhook configuration, so no secret needs to be handled
    by the caller. The message uses the real recommendation format with a TEST
    banner so nobody mistakes it for an actual push recommendation.
    """
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    sample = {
        "id": "teams-test",
        "url": "https://www.bild.de/",
        "title": "TEST: Integrationstest der Push-Empfehlung",
        "category": "news",
        "pubDate": _iso_from_ts(now - 15 * 60),
        "score": 84.0,
        "predictedOR": 0.061,
        "performanceDrivers": [
            "Dies ist eine Testnachricht aus dem Push Balancer",
            "Format-Check: Substanz zuerst, kompakt, echte Umlaute",
        ],
        "risks": ["Kein echter Artikel – bitte nicht pushen"],
        "recommendedText": "TEST: Push-Empfehlung (bitte ignorieren)",
        "isBreaking": False,
    }
    context = build_teams_alert_context([sample], now_ts=now, config=config)
    decision = should_notify_teams(sample, context, config)
    message = build_teams_push_recommendation(sample, context, decision, config)

    banner = "TESTNACHRICHT – keine echte Push-Empfehlung (Integrationstest Push Balancer)"
    text = f"{banner}\n\n{message['text']}"
    message["text"] = text
    message["summary"] = banner
    payload = message["payload"]
    payload["text"] = text
    payload["messageText"] = text
    payload["subject"] = "TEST: Push-Empfehlung"
    payload["isTest"] = True
    payload["messageHtml"] = (
        f"<p><strong>{html.escape(banner)}</strong></p>" + str(payload.get("messageHtml") or "")
    )

    result = send_teams_notification(message, config)
    log.info("[TeamsAlert] test message send ok=%s", bool(result.get("ok")))
    return result


def run_teams_alert_cycle() -> dict[str, Any]:
    """Fetch current article candidates and run one Teams alert cycle."""
    try:
        _refresh_push_history_for_timing()
        from app.routers.feed import build_articles_payload

        config = TeamsAlertConfig()
        dashboard_limit = max(1, min(int(config.dashboard_top_limit or 20), PUSH_TEAMS_CANDIDATE_LIMIT))
        payload = build_articles_payload(offset=0, limit=dashboard_limit)
        candidates = payload.get("articles") or []
        return evaluate_and_send_best_candidate(candidates, config=config)
    except Exception as exc:
        log.exception("[TeamsAlert] Cycle failed")
        return {"ok": False, "sent": False, "error": str(exc)}


def _refresh_push_history_for_timing() -> None:
    """Best-effort refresh so last-push timing decisions use fresh history."""
    try:
        from app.routers.push import _build_refresh_response

        result = _build_refresh_response()
        log.info(
            "[TeamsAlert] push history refresh source=%s synced=%s db_written=%s",
            result.get("source"),
            result.get("synced"),
            result.get("db_written"),
        )
    except Exception as exc:
        log.warning("[TeamsAlert] push history refresh skipped: %s", exc)


def select_teams_push_recommendation(
    candidates: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Pick the single best CvD-fit candidate and build its recommendation.

    Returns the selected candidate, its decision and a ready-to-send message
    (or None when the field has no truly push-worthy candidate right now).
    """
    config = config or TeamsAlertConfig()
    context = context or build_teams_alert_context(candidates, config=config)
    evaluation = evaluate_teams_alert_candidates(candidates, context, config)
    selected = evaluation.get("selectedCandidate")
    if not selected:
        return {
            "selected": None,
            "decision": None,
            "recommendation": None,
            "evaluation": evaluation,
        }
    selected_key = candidate_key(selected)
    decision = next(
        (
            item["decision"]
            for item in evaluation["decisions"]
            if item["decision"].get("candidateId") == selected_key
        ),
        None,
    )
    recommendation = (
        build_teams_push_recommendation(selected, context, decision, config)
        if decision and decision.get("shouldNotify")
        else None
    )
    return {
        "selected": selected,
        "decision": decision,
        "recommendation": recommendation,
        "evaluation": evaluation,
    }


def build_teams_daily_push_plan(
    candidates: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
    *,
    target_date: str | dt.date | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Build a Teams-ready CvD day plan from the current Push-Balancer field.

    Unlike the immediate alert path, this is a planning surface: hard spam and
    duplicate blockers still exclude candidates, while softer quality blockers
    downgrade confidence/status instead of hiding the next-best topics.
    """
    config = config or TeamsAlertConfig()
    now = int(now_ts or (context or {}).get("nowTs") or time.time())
    min_count = max(1, int(min_items or config.daily_plan_min_items or 15))
    max_count = max(min_count, int(max_items or config.daily_plan_max_items or min_count))
    plan_config = replace(
        config,
        enabled=True,
        quiet_hours_start="00:00",
        quiet_hours_end="00:00",
        global_cooldown_minutes=0,
        max_alerts_per_day=0,
        llm_title_enabled=False,
        target_pushes_per_day=max(int(config.target_pushes_per_day or 0), min_count),
        min_alerts_per_day=max(int(config.min_alerts_per_day or 0), min_count),
    )
    target = _parse_daily_plan_date(target_date, now)
    context = context or build_teams_alert_context(candidates, now_ts=now, config=plan_config)
    context = dict(context)
    context["nowTs"] = now

    raw_entries = _daily_plan_candidate_entries(candidates, context, plan_config)
    ranked_entries, not_recommended = _daily_plan_dedupe_and_rank(raw_entries, min_count)
    plan_count = min(max_count, max(min_count, len(ranked_entries)))
    selected_entries = ranked_entries[:plan_count]
    slots = _daily_plan_slots(target, max(len(selected_entries), min_count), plan_config)
    assigned = _daily_plan_assign_slots(selected_entries, slots, context, plan_config)
    items = sorted(assigned, key=lambda item: (item["slotTs"], -float(item["planScore"])))
    for index, item in enumerate(items, start=1):
        item["number"] = index

    top_items = sorted(items, key=lambda item: float(item["planScore"]), reverse=True)[:5]
    traffic_slots = _daily_plan_traffic_slots(target, plan_config)
    watch_topics = _daily_plan_watch_topics(ranked_entries[len(selected_entries):], raw_entries)
    selected_ids = {str(item.get("candidateId") or "") for item in items}
    not_recommended = _daily_plan_not_recommended(
        not_recommended,
        raw_entries,
        selected_ids=selected_ids,
        limit=8,
    )

    plan = {
        "type": "teams_daily_push_plan",
        "date": target.isoformat(),
        "weekday": _weekday_name_de(target.weekday()),
        "generatedAt": _format_dt(now),
        "generatedAtIso": _iso_from_ts(now),
        "minimumItems": min_count,
        "maximumItems": max_count,
        "count": len(items),
        "meetsMinimum": len(items) >= min_count,
        "items": items,
        "top5": [_daily_plan_item_summary(item) for item in top_items],
        "trafficSlots": traffic_slots,
        "watchTopics": watch_topics,
        "notRecommended": not_recommended,
        "pacing": _push_pacing_review(context.get("pushesToday"), now, plan_config),
        "assumptions": [
            "Der Tagesplan ist eine CvD-Planung, kein automatischer Versand an Nutzer.",
            "Fixe Vorschläge sind echte Push-Kandidaten; optionale Slots bleiben lageabhängig.",
            "Sport, Dubletten, bereits gepushte und bereits per Teams gemeldete Artikel sind ausgeschlossen.",
        ],
    }
    message = build_teams_daily_push_plan_message(plan)
    plan["messageText"] = message["text"]
    plan["messageHtml"] = message["html"]
    plan["payload"] = {
        "type": "push_daily_plan",
        "subject": message["subject"],
        "messageText": message["text"],
        "messageHtml": message["html"],
        "date": plan["date"],
        "weekday": plan["weekday"],
        "count": plan["count"],
        "items": plan["items"],
        "top5": plan["top5"],
        "trafficSlots": traffic_slots,
        "watchTopics": watch_topics,
        "notRecommended": not_recommended,
    }
    return plan


def build_teams_daily_push_plan_message(plan: dict[str, Any]) -> dict[str, str]:
    """Render a compact Teams-readable message for a daily push plan."""
    date_label = f"{plan.get('date')}, {plan.get('weekday')}"
    count = int(plan.get("count") or 0)
    minimum = int(plan.get("minimumItems") or 0)
    subject = f"Tagesplan Pushes für {date_label}: {count} Vorschläge"
    header = [
        f"Tagesplan Pushes für {date_label}",
        f"Ziel: mindestens {minimum} Pushes. Schwächere Vorschläge sind transparent markiert.",
        "",
    ]
    lines: list[str] = [*header]
    for item in plan.get("items") or []:
        lines.extend(
            [
                f"{int(item.get('number') or 0)}. {item.get('time')} – {item.get('pushText')}",
                f"Ressort: {item.get('sectionLabel')}",
                f"Priorität: {item.get('priority')} | Status: {item.get('status')}",
                (
                    f"Visit-Potenzial: {item.get('visitPotential')}/10 "
                    f"(ca. {_format_int(item.get('expectedVisits') or 0)} Visits)"
                ),
                (
                    f"Dringlichkeit: {item.get('urgency')}/10 | "
                    f"Timing-Fit: {item.get('timingFit')}/10 | "
                    f"Push-Müdigkeit: {item.get('fatigueRisk')}"
                ),
                f"Confidence: {item.get('confidence')}",
                f"Warum dieser Push: {item.get('why')}",
                f"Alternative: {item.get('alternativeTime')}",
                "",
            ]
        )

    lines.extend(["Top 5 Pushes des Tages:"])
    for item in plan.get("top5") or []:
        lines.append(f"- {item.get('time')} {item.get('pushText')} ({item.get('priority')})")

    lines.extend(["", "Slots mit besonders hohem Traffic-Potenzial:"])
    for slot in plan.get("trafficSlots") or []:
        label = slot.get("label")
        forecast = _format_or(slot.get("avgOR")) if slot.get("avgOR") else "keine OR-Basis"
        lines.append(f"- {label}: {forecast}, {slot.get('reason')}")

    lines.extend(["", "Themen beobachten:"])
    watch = plan.get("watchTopics") or []
    if watch:
        for item in watch:
            lines.append(f"- {item.get('title')}: {item.get('reason')}")
    else:
        lines.append("- Aktuell keine zusätzlichen Beobachtungsthemen im Kandidatenfeld.")

    lines.extend(["", "Bewusst nicht pushen:"])
    skipped = plan.get("notRecommended") or []
    if skipped:
        for item in skipped:
            lines.append(f"- {item.get('title')}: {item.get('reason')}")
    else:
        lines.append("- Keine harten Ausschlüsse im geprüften Kandidatenfeld.")

    text = "\n".join(lines).strip()
    html_message = _build_teams_daily_plan_html(subject, plan)
    return {"subject": subject, "text": text, "html": html_message}


def _daily_plan_candidate_entries(
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    config: TeamsAlertConfig,
) -> list[dict[str, Any]]:
    top_limit = max(1, int(config.dashboard_top_limit or PUSH_TEAMS_CANDIDATE_LIMIT))
    entries: list[dict[str, Any]] = []
    base_context = dict(context)
    base_context.pop("strongerCandidate", None)
    for index, candidate in enumerate(candidates or [], start=1):
        decision_context = dict(base_context)
        decision_context["dashboardRank"] = index
        decision_context["dashboardTopLimit"] = top_limit
        decision = should_notify_teams(candidate, decision_context, config)
        entries.append(_daily_plan_entry(candidate, decision, index, config))
    return entries


def _daily_plan_entry(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    dashboard_rank: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    blockers = list(decision.get("blockingReasons") or [])
    hard_blockers = _daily_plan_hard_blockers(candidate, decision, config)
    editorial = decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    breakdown = editorial.get("breakdown") if isinstance(editorial.get("breakdown"), dict) else {}
    visit_score = float(decision.get("visitPotentialScore") or 0.0)
    editorial_score = float(decision.get("editorialScore") or editorial.get("score") or 0.0)
    alert_score = float(decision.get("teamsAlertScore") or 0.0)
    raw_score = float(decision.get("score") or _score(candidate))
    urgency = float(breakdown.get("urgency") or 0.0)
    time_fit = float(breakdown.get("timeFit") or 0.0)
    plan_score = _daily_plan_rank_score(
        decision=decision,
        visit_score=visit_score,
        editorial_score=editorial_score,
        alert_score=alert_score,
        raw_score=raw_score,
        urgency=urgency,
        time_fit=time_fit,
    )
    priority = _daily_plan_priority(plan_score, blockers, hard_blockers, decision)
    fatigue_risk = _daily_plan_fatigue_risk(decision, config)
    confidence = _daily_plan_confidence(decision, priority, hard_blockers)
    status = _daily_plan_status(priority, confidence, blockers, hard_blockers, fatigue_risk, decision)
    title = _title(candidate)
    push_text, push_source = _teams_push_title_recommendation(
        candidate,
        title,
        _section(candidate),
        _url(candidate),
        config,
    )
    return {
        "candidate": candidate,
        "decision": decision,
        "dashboardRank": dashboard_rank,
        "candidateId": decision.get("candidateId") or candidate_key(candidate),
        "title": title,
        "url": _url(candidate),
        "section": _section(candidate),
        "score": raw_score,
        "pushText": push_text or title,
        "pushTitleSource": push_source,
        "planScore": plan_score,
        "priority": priority,
        "confidence": confidence,
        "status": status,
        "fatigueRisk": fatigue_risk,
        "visitPotential": _score_to_ten(visit_score),
        "urgency": _score_to_ten(urgency, scale=16.0),
        "timingFit": _score_to_ten(time_fit * 10.0),
        "hardBlockers": hard_blockers,
        "softBlockers": [reason for reason in blockers if reason not in hard_blockers],
        "why": _daily_plan_reason(candidate, decision, status),
    }


def _daily_plan_dedupe_and_rank(
    entries: list[dict[str, Any]],
    min_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(entries, key=lambda item: float(item.get("planScore") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in ranked:
        if entry.get("hardBlockers"):
            skipped.append({**entry, "skipReason": str(entry["hardBlockers"][0])})
            continue
        duplicate_of = _daily_plan_duplicate_of(entry, selected)
        if duplicate_of:
            skipped.append(
                {
                    **entry,
                    "skipReason": (
                        "Dublette im Tagesplan; stärkerer Kandidat: "
                        f"{_compact_text(str(duplicate_of.get('title') or ''), 80)}"
                    ),
                }
            )
            continue
        selected.append(entry)

    if len(selected) < min_count:
        # Transparente Mindestmengen-Ergaenzung: nur nicht-harte Kandidaten, die
        # wegen Plan-Dublette herausfielen, bleiben ausgeschlossen.
        supplements = [
            entry
            for entry in ranked
            if entry not in selected
            and not entry.get("hardBlockers")
            and not any(
                skipped_item.get("candidateId") == entry.get("candidateId")
                and "Dublette im Tagesplan" in str(skipped_item.get("skipReason") or "")
                for skipped_item in skipped
            )
        ]
        for entry in supplements:
            if len(selected) >= min_count:
                break
            entry = dict(entry)
            entry["priority"] = "C"
            entry["confidence"] = "niedrig"
            entry["status"] = "nur bei ruhiger Nachrichtenlage"
            entry["why"] = (
                "Nur zur Mindestplanung: kein Top-Push, aber der nächstbeste verfügbare Kandidat."
            )
            selected.append(entry)

    return selected, skipped


def _daily_plan_duplicate_of(
    entry: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidate = entry.get("candidate") or {}
    title_tokens = _tokens(str(entry.get("title") or ""))
    slug_tokens = _url_slug_tokens(_url(candidate))
    for other in selected:
        other_candidate = other.get("candidate") or {}
        if _same_topic(slug_tokens, _url_slug_tokens(_url(other_candidate)), 0.58):
            return other
        if _same_topic(title_tokens, _tokens(str(other.get("title") or "")), 0.58):
            return other
    return None


def _daily_plan_assign_slots(
    entries: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    context: dict[str, Any],
    config: TeamsAlertConfig,
) -> list[dict[str, Any]]:
    priority_slots = sorted(
        slots,
        key=lambda slot: (
            float(slot.get("weight") or 0.0),
            float(slot.get("slotScore") or 0.0),
            float(slot.get("avgOR") or 0.0),
        ),
        reverse=True,
    )
    assigned: list[dict[str, Any]] = []
    for entry, slot in zip(entries, priority_slots, strict=False):
        alternative = _daily_plan_alternative_slot(slot, slots)
        assigned.append(_daily_plan_finalize_item(entry, slot, alternative, context, config))
    return assigned


def _daily_plan_finalize_item(
    entry: dict[str, Any],
    slot: dict[str, Any],
    alternative: dict[str, Any] | None,
    context: dict[str, Any],
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    candidate = entry.get("candidate") or {}
    decision = entry.get("decision") if isinstance(entry.get("decision"), dict) else {}
    editorial = decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    editorial_score = float(decision.get("editorialScore") or editorial.get("score") or 0.0)
    alert_score = float(decision.get("teamsAlertScore") or 0.0)
    score = float(decision.get("score") or _score(candidate))
    forecast = _candidate_forecast(
        candidate,
        int(slot["ts"]),
        {float(value) for value in (context.get("suspectForecastValues") or [])},
    )
    visit = _visit_potential(
        candidate,
        predicted_or=forecast.get("value"),
        editorial_score=editorial_score,
        alert_score=alert_score,
        score=score,
        breaking=_is_breaking(candidate),
        now_ts=int(slot["ts"]),
        reach_stats=context.get("reachStats") if isinstance(context.get("reachStats"), dict) else {},
        config=config,
    )
    time_fit = _time_fit_review(
        now_ts=int(slot["ts"]),
        section=_section(candidate),
        breaking=_is_breaking(candidate),
        config=config,
        pushes_today=context.get("pushesToday"),
    )
    item = {
        key: value
        for key, value in entry.items()
        if key not in {"candidate", "decision", "hardBlockers", "softBlockers"}
    }
    item.update(
        {
            "articleTitle": _title(candidate),
            "articleUrl": _url(candidate),
            "sectionLabel": _format_section(_section(candidate)),
            "time": slot["label"],
            "slotTs": int(slot["ts"]),
            "slotHour": int(slot["hour"]),
            "slotWeight": round(float(slot.get("weight") or 0.0), 2),
            "slotAvgOR": slot.get("avgOR"),
            "slotReason": slot.get("reason"),
            "alternativeTime": (
                f"{alternative['label']} Uhr" if alternative else "Lageabhängig prüfen"
            ),
            "forecast": forecast,
            "predictedOR": forecast.get("value"),
            "predictedORLabel": _format_forecast(forecast),
            "expectedVisits": int(visit.get("expectedVisits") or 0),
            "estimatedReach": int(visit.get("estimatedReach") or 0),
            "visitPotential": _score_to_ten(float(visit.get("score") or 0.0)),
            "timingFit": _score_to_ten(float(time_fit.get("score") or 0.0) * 10.0),
            "timingLabel": time_fit.get("label"),
            "blockingReasons": list(decision.get("blockingReasons") or []),
            "positiveReasons": list(decision.get("reasons") or []),
        }
    )
    item["why"] = _daily_plan_reason(candidate, decision, str(item.get("status") or ""), slot, visit)
    return item


def _daily_plan_hard_blockers(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    config: TeamsAlertConfig,
) -> list[str]:
    section = _section(candidate).lower()
    excluded = {item.lower() for item in config.excluded_sections if item.strip()}
    hard: list[str] = []
    if not _title(candidate):
        hard.append("Keine Headline")
    if not _url(candidate):
        hard.append("Kein Artikel-Link")
    if section in excluded:
        hard.append(f"Ressort {_format_section(section)} ist ausgeschlossen")
    hard_markers = (
        "Bereits live gepusht",
        "Bereits per Teams gemeldet",
        "Thema bereits per Teams gemeldet",
        "Teams-Hinweis wird bereits versendet",
        "Bereits als Teams-Kandidat versucht",
        "Artikel-Link",
        "Headline",
    )
    for reason in decision.get("blockingReasons") or []:
        text = str(reason or "")
        if any(marker in text for marker in hard_markers):
            hard.append(text)
    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if allowed and section not in allowed:
        hard.append(f"Ressort {_format_section(section)} ist nicht freigegeben")
    return _dedupe(hard)


def _daily_plan_rank_score(
    *,
    decision: dict[str, Any],
    visit_score: float,
    editorial_score: float,
    alert_score: float,
    raw_score: float,
    urgency: float,
    time_fit: float,
) -> float:
    urgency_score = _score_to_hundred(urgency, scale=16.0)
    time_score = _score_to_hundred(time_fit, scale=10.0)
    total = (
        visit_score * 0.34
        + editorial_score * 0.25
        + alert_score * 0.16
        + raw_score * 0.11
        + time_score * 0.08
        + urgency_score * 0.06
    )
    blockers = [str(reason) for reason in decision.get("blockingReasons") or []]
    soft_penalty = min(18.0, len(blockers) * 3.0)
    if any("Kurios-/Click-Reiz" in reason for reason in blockers):
        soft_penalty += 10.0
    if any("Service-/Raetsel" in reason or "kein konkretes Nachrichten-Ereignis" in reason for reason in blockers):
        soft_penalty += 8.0
    if decision.get("shouldNotify"):
        total += 4.0
    if decision.get("isBreaking"):
        total += 3.0
    return round(_clamp(total - soft_penalty, 0.0, 100.0), 1)


def _daily_plan_priority(
    plan_score: float,
    blockers: list[str],
    hard_blockers: list[str],
    decision: dict[str, Any],
) -> str:
    if hard_blockers:
        return "nicht pushen"
    if plan_score >= 80.0 and decision.get("shouldNotify"):
        return "A"
    if plan_score >= 74.0 and not any("CvD:" in reason for reason in blockers):
        return "A"
    if plan_score >= 62.0:
        return "B"
    return "C"


def _daily_plan_status(
    priority: str,
    confidence: str,
    blockers: list[str],
    hard_blockers: list[str],
    fatigue_risk: str,
    decision: dict[str, Any],
) -> str:
    if hard_blockers:
        return "bewusst nicht pushen"
    if priority == "A" and confidence != "niedrig" and fatigue_risk != "hoch":
        return "fix"
    if priority in {"A", "B"}:
        return "optional"
    if decision.get("shouldNotify") and not blockers:
        return "optional"
    return "nur bei ruhiger Nachrichtenlage"


def _daily_plan_confidence(
    decision: dict[str, Any],
    priority: str,
    hard_blockers: list[str],
) -> str:
    if hard_blockers:
        return "niedrig"
    forecast = decision.get("forecast") if isinstance(decision.get("forecast"), dict) else {}
    source = str(forecast.get("source") or "")
    forecast_confidence = _safe_float(forecast.get("confidence")) or 0.0
    editorial = decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    approved = bool(editorial.get("approved", False))
    if priority == "A" and approved and source == "article_model":
        return "hoch"
    if priority in {"A", "B"} and (source == "article_model" or forecast_confidence >= 0.45):
        return "mittel"
    return "niedrig"


def _daily_plan_fatigue_risk(decision: dict[str, Any], config: TeamsAlertConfig) -> str:
    minutes = decision.get("minutesSinceLastPush")
    recent_pushes = _safe_int(decision.get("recentPushCount6h"))
    pushes_today = _safe_int(decision.get("pushesToday"))
    if isinstance(minutes, (int, float)) and minutes < max(20, config.min_minutes_since_last_push):
        return "hoch"
    if recent_pushes > config.max_pushes_last_6h or pushes_today >= max(1, config.target_pushes_per_day):
        return "hoch"
    if isinstance(minutes, (int, float)) and minutes < config.min_minutes_since_last_push + 25:
        return "mittel"
    if pushes_today >= max(1, config.target_pushes_per_day - 2):
        return "mittel"
    return "niedrig"


def _daily_plan_reason(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    status: str,
    slot: dict[str, Any] | None = None,
    visit: dict[str, Any] | None = None,
) -> str:
    drivers = _editorial_list(candidate, "performanceDrivers")
    forecast = decision.get("forecast") if isinstance(decision.get("forecast"), dict) else {}
    visit_data = visit or decision.get("visitPotential") or {}
    expected_visits = _safe_int(visit_data.get("expectedVisits"))
    predicted_or = _safe_float(visit_data.get("predictedOR"))
    visit_reason = ""
    if expected_visits > 0:
        visit_reason = f"Visit-Potenzial ca. {_format_int(expected_visits)} Visits"
        if predicted_or:
            visit_reason += f" bei {_format_or(predicted_or)}"
    slot_reason = str((slot or {}).get("reason") or "").strip()
    blockers = [str(reason) for reason in decision.get("blockingReasons") or []]
    forecast_reason = _forecast_sentence(forecast) if forecast and not visit_reason else ""
    parts = _dedupe(
        [
            *(drivers[:1] if drivers else []),
            visit_reason,
            slot_reason,
            forecast_reason,
            *(blockers[:1] if status != "fix" else []),
        ]
    )
    reason = "; ".join(_compact_text(part, 105) for part in parts[:4] if part)
    return reason or "Solider Kandidat im aktuellen Push-Balancer-Feld."


def _daily_plan_slots(
    target_date: dt.date,
    count: int,
    config: TeamsAlertConfig,
) -> list[dict[str, Any]]:
    weekday = target_date.weekday()
    start = int(_clamp(config.active_hours_start, 0, 23))
    end = int(_clamp(config.active_hours_end, start, 23))
    slots: list[dict[str, Any]] = []
    for hour in range(start, end + 1):
        slots.append(_daily_plan_slot(target_date, hour, 0, weekday, config))
    if count > len(slots):
        extra_hours = sorted(
            range(start, end + 1),
            key=lambda hour: _slot_weight(hour, weekday, config),
            reverse=True,
        )
        for hour in extra_hours:
            if len(slots) >= count:
                break
            slots.append(_daily_plan_slot(target_date, hour, 30, weekday, config))
    return slots


def _daily_plan_slot(
    target_date: dt.date,
    hour: int,
    minute: int,
    weekday: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    slot_dt = dt.datetime.combine(
        target_date,
        dt.time(hour=int(hour), minute=int(minute)),
        tzinfo=ZoneInfo("Europe/Berlin"),
    )
    baseline = _slot_baseline(hour, weekday)
    avg_or = _safe_float(baseline.get("avg_or")) if baseline else None
    stars = int(baseline.get("stars") or 0) if baseline else 0
    slot_score = _slot_baseline_score(hour, weekday, baseline, breaking=False)
    reason_parts = []
    if avg_or:
        reason_parts.append(f"historisch {_format_number(avg_or, 2)} % OR")
    if stars >= 2:
        reason_parts.append("starker historischer Slot")
    top_cat = str(baseline.get("top_cat") or "").strip()
    if top_cat:
        reason_parts.append(f"Top-Ressort {_format_section(top_cat)}")
    return {
        "ts": int(slot_dt.timestamp()),
        "label": f"{hour:02d}:{minute:02d}",
        "hour": hour,
        "minute": minute,
        "weekday": weekday,
        "weight": _slot_weight(hour, weekday, config),
        "slotScore": round(slot_score, 1),
        "avgOR": round(avg_or, 2) if avg_or is not None else None,
        "stars": stars,
        "topCategory": top_cat or None,
        "reason": ", ".join(reason_parts) or "brauchbares Standardfenster",
    }


def _daily_plan_alternative_slot(
    slot: dict[str, Any],
    slots: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_ts = int(slot.get("ts") or 0)
    future = [candidate for candidate in slots if int(candidate.get("ts") or 0) > current_ts]
    if future:
        return min(future, key=lambda candidate: int(candidate.get("ts") or 0))
    past = [candidate for candidate in slots if int(candidate.get("ts") or 0) < current_ts]
    if past:
        return max(past, key=lambda candidate: int(candidate.get("ts") or 0))
    return None


def _daily_plan_traffic_slots(
    target_date: dt.date,
    config: TeamsAlertConfig,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    slots = _daily_plan_slots(target_date, int(config.active_hours_end) + 1, config)
    top = sorted(
        slots,
        key=lambda slot: (
            float(slot.get("weight") or 0.0),
            float(slot.get("avgOR") or 0.0),
        ),
        reverse=True,
    )[:limit]
    return [
        {
            "label": f"{slot['label']} Uhr",
            "hour": slot["hour"],
            "avgOR": slot.get("avgOR"),
            "weight": slot.get("weight"),
            "reason": slot.get("reason"),
        }
        for slot in sorted(top, key=lambda item: int(item.get("ts") or 0))
    ]


def _daily_plan_watch_topics(
    remaining: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    pool = remaining or [
        entry for entry in raw_entries if not entry.get("hardBlockers") and entry.get("priority") in {"B", "C"}
    ]
    result: list[dict[str, Any]] = []
    for entry in pool[:limit]:
        result.append(
            {
                "title": _compact_text(str(entry.get("title") or ""), 100),
                "section": _format_section(str(entry.get("section") or "")),
                "score": round(float(entry.get("score") or 0.0), 1),
                "reason": _compact_text(str(entry.get("why") or "Lage beobachten."), 140),
            }
        )
    return result


def _daily_plan_not_recommended(
    skipped: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
    *,
    selected_ids: set[str] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    selected_ids = selected_ids or set()
    pool = list(skipped)
    if len(pool) < limit:
        low_quality = [
            entry
            for entry in raw_entries
            if entry not in pool
            and str(entry.get("candidateId") or "") not in selected_ids
            and (
                entry.get("priority") == "C"
                or any("Kurios-/Click-Reiz" in reason for reason in entry.get("softBlockers") or [])
            )
        ]
        pool.extend(low_quality)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in pool:
        key = str(entry.get("candidateId") or entry.get("url") or entry.get("title") or "")
        if key in selected_ids:
            continue
        if key in seen:
            continue
        seen.add(key)
        reasons = list(entry.get("hardBlockers") or [])
        if not reasons:
            reasons = [str(entry.get("skipReason") or "zu schwach im Tagesvergleich")]
        result.append(
            {
                "title": _compact_text(str(entry.get("title") or ""), 100),
                "section": _format_section(str(entry.get("section") or "")),
                "reason": _compact_text(str(reasons[0]), 160),
            }
        )
        if len(result) >= limit:
            break
    return result


def _daily_plan_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": item.get("time"),
        "pushText": item.get("pushText"),
        "articleTitle": item.get("articleTitle"),
        "articleUrl": item.get("articleUrl"),
        "section": item.get("sectionLabel"),
        "priority": item.get("priority"),
        "status": item.get("status"),
        "visitPotential": item.get("visitPotential"),
        "confidence": item.get("confidence"),
        "expectedVisits": item.get("expectedVisits"),
        "why": item.get("why"),
    }


def _build_teams_daily_plan_html(subject: str, plan: dict[str, Any]) -> str:
    items_html = "".join(
        "<li>"
        f"<strong>{html.escape(str(item.get('time') or ''))} – "
        f"{html.escape(str(item.get('pushText') or ''))}</strong><br>"
        f"Ressort: {html.escape(str(item.get('sectionLabel') or ''))} | "
        f"Priorität: {html.escape(str(item.get('priority') or ''))} | "
        f"Status: {html.escape(str(item.get('status') or ''))}<br>"
        f"Visit-Potenzial: {html.escape(str(item.get('visitPotential') or ''))}/10 | "
        f"Confidence: {html.escape(str(item.get('confidence') or ''))}<br>"
        f"{html.escape(str(item.get('why') or ''))}"
        "</li>"
        for item in plan.get("items") or []
    )
    top_html = "".join(
        f"<li>{html.escape(str(item.get('time') or ''))} "
        f"{html.escape(str(item.get('pushText') or ''))}</li>"
        for item in plan.get("top5") or []
    )
    skipped_html = "".join(
        f"<li>{html.escape(str(item.get('title') or ''))}: "
        f"{html.escape(str(item.get('reason') or ''))}</li>"
        for item in plan.get("notRecommended") or []
    )
    return (
        f"<h2>{html.escape(subject)}</h2>"
        f"<p>Ziel: mindestens {int(plan.get('minimumItems') or 0)} Pushes. "
        "Schwächere Vorschläge sind markiert.</p>"
        f"<ol>{items_html}</ol>"
        "<p><strong>Top 5 Pushes des Tages</strong></p>"
        f"<ul>{top_html}</ul>"
        "<p><strong>Bewusst nicht pushen</strong></p>"
        f"<ul>{skipped_html}</ul>"
    )


def _parse_daily_plan_date(value: str | dt.date | None, now_ts: int) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if value:
        try:
            return dt.date.fromisoformat(str(value))
        except ValueError:
            pass
    return dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin")).date()


def _weekday_name_de(weekday: int) -> str:
    names = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
    if 0 <= int(weekday) < len(names):
        return names[int(weekday)]
    return "Wochentag"


def _score_to_ten(value: float, *, scale: float = 100.0) -> float:
    return round(_clamp((float(value or 0.0) / max(scale, 1.0)) * 10.0, 1.0, 10.0), 1)


def _score_to_hundred(value: float, *, scale: float = 100.0) -> float:
    return _clamp((float(value or 0.0) / max(scale, 1.0)) * 100.0, 0.0, 100.0)


# Compatibility aliases requested in the implementation brief.
def selectTeamsPushRecommendation(
    candidates: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    return select_teams_push_recommendation(candidates, context, config)


def shouldNotifyTeams(
    candidate: dict[str, Any],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    return should_notify_teams(candidate, context, config)


def buildTeamsPushRecommendation(
    candidate: dict[str, Any],
    context: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    return build_teams_push_recommendation(candidate, context, decision, config)


def buildTeamsDailyPushPlan(
    candidates: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    config: TeamsAlertConfig | None = None,
    *,
    target_date: str | dt.date | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    return build_teams_daily_push_plan(
        candidates,
        context,
        config,
        target_date=target_date,
        min_items=min_items,
        max_items=max_items,
        now_ts=now_ts,
    )


def sendTeamsNotification(
    message: dict[str, Any],
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    return send_teams_notification(message, config)


def normalize_predicted_or(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    percent = numeric * 100.0 if numeric <= 1.0 else numeric
    # The prediction models are calibrated in percent and clamped to a practical
    # editorial range. Tiny values such as 0.0004 usually mean a double-scaled
    # ratio, not a real 0.04% opening-rate forecast.
    if percent < 0.5 or percent > 30.0:
        return None
    return percent


def _median(values: list[float]) -> float | None:
    clean = sorted(float(value) for value in values if value and value > 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _section_key(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return "news"
    aliases = {
        "geld": "wirtschaft",
        "business": "wirtschaft",
        "inland": "news",
        "ausland": "news",
        "crime": "news",
        "wetter": "news",
        "ratgeber": "leben-wissen",
        "leben": "leben-wissen",
        "service": "leben-wissen",
    }
    return aliases.get(raw, raw)


def _history_reach(item: dict[str, Any]) -> int:
    for key in ("total_recipients", "recipientCount", "recipients", "received"):
        value = _safe_int(item.get(key))
        if value > 0:
            return value
    return 0


def _history_hour(item: dict[str, Any]) -> int | None:
    raw_hour = item.get("hour")
    hour = _safe_int(raw_hour) if raw_hour not in (None, "") else -1
    if 0 <= hour <= 23:
        return hour
    ts_value = _safe_int(item.get("ts_num", item.get("ts", 0)))
    if ts_value > 0:
        return dt.datetime.fromtimestamp(ts_value, ZoneInfo("Europe/Berlin")).hour
    return None


def _reach_baselines(
    history: list[dict[str, Any]],
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Build robust reach baselines from recent real push recipients."""
    section_values: dict[str, list[float]] = {}
    hour_values: dict[int, list[float]] = {}
    global_values: list[float] = []
    cutoff = int(now_ts) - 30 * 86400
    for item in history or []:
        ts_value = _safe_int(item.get("ts_num", item.get("ts", 0)))
        if ts_value and ts_value < cutoff:
            continue
        reach = _history_reach(item)
        if reach <= 0:
            continue
        global_values.append(float(reach))
        section = _section_key(item.get("cat") or item.get("category"))
        section_values.setdefault(section, []).append(float(reach))
        hour = _history_hour(item)
        if hour is not None:
            hour_values.setdefault(hour, []).append(float(reach))

    default_reach = max(1, int(config.default_reach or 250000))
    global_median = _median(global_values) or float(default_reach)
    return {
        "globalMedian": float(global_median),
        "bySection": {
            section: float(_median(values) or global_median)
            for section, values in section_values.items()
        },
        "byHour": {
            str(hour): float(_median(values) or global_median)
            for hour, values in hour_values.items()
        },
        "sectionCounts": {section: len(values) for section, values in section_values.items()},
        "hourCounts": {str(hour): len(values) for hour, values in hour_values.items()},
        "sampleSize": len(global_values),
        "defaultReach": default_reach,
    }


def _candidate_explicit_reach(candidate: dict[str, Any]) -> int:
    for key in ("estimatedReach", "expectedReach", "estimatedRecipients", "recipients", "recipientCount"):
        value = _safe_int(candidate.get(key))
        if value > 0:
            return value
    return 0


def _audience_breadth_adjustment(candidate: dict[str, Any], *, breaking: bool) -> dict[str, Any]:
    """Estimate how broad the reachable push audience is for this story.

    OR alone rewards clicky narrow stories. This factor keeps the visit objective
    tied to actual CvD judgment: broad public-impact stories get more reach
    potential, curiosity/soft stories get less.
    """
    title = _title(candidate)
    section = _section_key(_section(candidate))
    factor = 1.0
    labels: list[str] = []

    if breaking:
        factor *= 1.08
        labels.append("Breaking-Breitenfaktor")
    if _has_broad_public_impact(title, section):
        factor *= 1.12
        labels.append("breite öffentliche Relevanz")
    if _is_low_civic_impact_story(title, section):
        factor *= 0.62
        labels.append("enger Kurios-/Click-Reiz")

    section_factor = {
        "politik": 1.04,
        "news": 1.0,
        "wirtschaft": 0.97,
        "regional": 0.88,
        "digital": 0.82,
        "unterhaltung": 0.72,
        "leben-wissen": 0.74,
        "service": 0.68,
    }.get(section, 0.90)
    if section in {"politik", "news", "wirtschaft"}:
        labels.append("breites Ressort")
    elif section in {"unterhaltung", "digital", "leben-wissen", "service"}:
        labels.append("engeres Ressort")
    elif section == "regional":
        labels.append("regional begrenzte Reichweite")
    factor *= section_factor

    return {
        "factor": _clamp(factor, 0.45, 1.30),
        "label": ", ".join(_dedupe(labels[:3])),
    }


def _estimated_reach(
    candidate: dict[str, Any],
    *,
    now_ts: int,
    reach_stats: dict[str, Any],
    breaking: bool,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    explicit = _candidate_explicit_reach(candidate)
    if explicit > 0:
        return {"value": float(explicit), "source": "candidate", "confidence": 1.0}

    global_median = _safe_float(reach_stats.get("globalMedian")) or float(config.default_reach or 250000)
    by_section = reach_stats.get("bySection") if isinstance(reach_stats.get("bySection"), dict) else {}
    by_hour = reach_stats.get("byHour") if isinstance(reach_stats.get("byHour"), dict) else {}
    section_counts = (
        reach_stats.get("sectionCounts") if isinstance(reach_stats.get("sectionCounts"), dict) else {}
    )
    hour_counts = reach_stats.get("hourCounts") if isinstance(reach_stats.get("hourCounts"), dict) else {}

    section = _section_key(_section(candidate))
    section_reach = _safe_float(by_section.get(section)) or global_median
    section_count = _safe_int(section_counts.get(section))
    hour = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin")).hour
    hour_reach = _safe_float(by_hour.get(str(hour)))
    hour_count = _safe_int(hour_counts.get(str(hour)))
    hour_factor = 1.0
    if hour_reach and global_median > 0:
        hour_factor = _clamp(hour_reach / global_median, 0.70, 1.35)
    reach = section_reach * hour_factor
    if breaking:
        reach *= 1.08
    audience = _audience_breadth_adjustment(candidate, breaking=breaking)
    reach *= float(audience["factor"])
    confidence = _clamp(
        (min(section_count, 12) / 12.0) * 0.70 + (min(hour_count, 12) / 12.0) * 0.30,
        0.20,
        0.95,
    )
    source = "historische Reichweite"
    if section_count > 0:
        source = f"historische {section}-Reichweite"
    if hour_count > 0:
        source += f" + Slot {hour:02d} Uhr"
    if audience["label"]:
        source += f" + {audience['label']}"
    return {
        "value": round(max(1.0, reach), 1),
        "source": source,
        "confidence": round(confidence, 2),
        "audienceFactor": round(float(audience["factor"]), 2),
        "audienceLabel": audience["label"],
    }


def _visit_potential(
    candidate: dict[str, Any],
    *,
    predicted_or: float | None,
    editorial_score: float,
    alert_score: float,
    score: float,
    breaking: bool,
    now_ts: int,
    reach_stats: dict[str, Any],
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    reach = _estimated_reach(
        candidate,
        now_ts=now_ts,
        reach_stats=reach_stats,
        breaking=breaking,
        config=config,
    )
    or_value = max(0.0, float(predicted_or or 0.0))
    expected_visits = float(reach["value"]) * or_value / 100.0
    quality_factor = _clamp(
        0.55
        + max(0.0, editorial_score - 60.0) / 100.0
        + max(0.0, alert_score - 60.0) / 180.0
        + max(0.0, score - 70.0) / 300.0,
        0.55,
        1.20,
    )
    quality_adjusted_visits = expected_visits * quality_factor
    global_reach = _safe_float(reach_stats.get("globalMedian")) or float(config.default_reach or 250000)
    benchmark = max(2500.0, global_reach * max(float(config.min_or or 5.0), 4.5) / 100.0)
    ratio = quality_adjusted_visits / benchmark if benchmark > 0 else 0.0
    visit_score = _clamp(45.0 + ratio * 35.0, 0.0, 100.0)
    reason = ""
    if or_value > 0:
        audience_label = str(reach.get("audienceLabel") or "").strip()
        audience_suffix = f"; {audience_label}" if audience_label else ""
        reason = (
            "Visit-Potenzial: ca. "
            f"{_format_int(expected_visits)} erwartete Push-Visits "
            f"({_format_number(or_value, 2)} % OR x ca. {_format_int(float(reach['value']))} Reichweite"
            f"{audience_suffix})"
        )
    return {
        "expectedVisits": int(round(expected_visits)),
        "qualityAdjustedVisits": int(round(quality_adjusted_visits)),
        "estimatedReach": int(round(float(reach["value"]))),
        "predictedOR": round(or_value, 2) if or_value else None,
        "score": round(visit_score, 1),
        "reachSource": reach["source"],
        "reachConfidence": reach["confidence"],
        "audienceFactor": reach.get("audienceFactor"),
        "audienceLabel": reach.get("audienceLabel"),
        "reason": reason,
    }


def _recommendation_selection_score(
    *,
    score: float,
    alert_score: float,
    editorial_score: float,
    predicted_or: float | None,
    dashboard_rank: int,
    breaking: bool,
    visit_score: float = 0.0,
    config: TeamsAlertConfig | None = None,
) -> float:
    """Rank eligible candidates by expected visit value and editorial suitability."""
    config = config or TeamsAlertConfig()
    forecast_bonus = min(float(predicted_or or 0.0), 10.0) * 0.4
    rank_bonus = max(0.0, 4.0 - max(0, dashboard_rank - 1) * 0.4) if dashboard_rank > 0 else 0.0
    breaking_bonus = 2.0 if breaking else 0.0
    editorial_total = (
        editorial_score * 0.55
        + alert_score * 0.30
        + score * 0.10
        + forecast_bonus
        + rank_bonus
        + breaking_bonus
    )
    if config.visit_optimization_enabled:
        weight = _clamp(float(config.visit_selection_weight or 0.0), 0.0, 0.85)
        total = editorial_total * (1.0 - weight) + float(visit_score or 0.0) * weight
    else:
        total = editorial_total
    return round(_clamp(total, 0.0, 100.0), 1)


def _forecast_quality_review(
    candidate: dict[str, Any],
    forecast: dict[str, Any],
    alert_score: float,
    breaking: bool,
    config: TeamsAlertConfig,
    minimum_pressure: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    if not config.require_article_forecast:
        return {"reasons": [], "blockers": []}

    source = str(forecast.get("source") or "")
    if source == "article_model":
        return {"reasons": ["Belastbare Artikel-Prognose vorhanden"], "blockers": []}
    if breaking and config.breaking_override:
        return {"reasons": ["Breaking-Override: Slot-Prognose nur Timing-Kontext"], "blockers": []}
    if _has_hard_public_need(_title(candidate), _section(candidate)) and alert_score >= config.no_forecast_min_alert_score:
        return {
            "reasons": ["Öffentliche Warn-/Nutzwertlage: auch ohne Artikelmodell prüfbar"],
            "blockers": [],
        }
    minimum_pressure = minimum_pressure or {}
    if (
        minimum_pressure.get("active")
        and _has_news_event(_title(candidate))
        and alert_score >= 68.0
        and not _is_soft_service_or_quiz(_title(candidate))
        and not _is_nonessential_curiosity(_title(candidate))
        and not _is_abstract_explainer_without_update(_title(candidate))
        and not _is_scheduled_process_without_update(_title(candidate))
    ):
        return {
            "reasons": ["Mindest-Pacing: historische Slot-Prognose als Timing-Kontext akzeptiert"],
            "blockers": [],
        }
    return {
        "reasons": [],
        "blockers": [
            "Belastbare Artikel-Prognose fehlt; historische Slot-Prognose reicht für normale Teams-Empfehlung nicht"
        ],
    }


def _daily_strategy_review(
    candidate: dict[str, Any],
    *,
    alert_score: float,
    editorial_score: float,
    news_value: float,
    predicted_or: float | None,
    push_pacing: dict[str, Any],
    breaking: bool,
    config: TeamsAlertConfig,
    minimum_pressure: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    if breaking:
        return {"reasons": ["Tagesstrategie: Breaking darf Push-Bestand übersteuern"], "blockers": []}
    if not push_pacing.get("known"):
        return {"reasons": [], "blockers": []}

    surplus = float(push_pacing.get("surplus") or 0.0)
    deficit = float(push_pacing.get("deficit") or 0.0)
    minimum_pressure = minimum_pressure or {}
    blockers: list[str] = []
    reasons: list[str] = []
    strong_enough_when_ahead = (
        alert_score >= config.min_alert_score + 8.0
        and editorial_score >= config.min_editorial_score + 8.0
        and news_value >= config.min_editorial_news_value + 8.0
        and (predicted_or is None or predicted_or >= config.min_or + 0.5)
    )
    if surplus >= 2.0 and not strong_enough_when_ahead:
        if minimum_pressure.get("active"):
            reasons.append(
                "Teams-Mindest-Pacing: Rueckstand im Teams-Kanal, Push-Vorsprung blockiert die Empfehlung nicht"
            )
        else:
            blockers.append(
                "Tagesstrategie: Push-Bestand liegt vorn; normale Lage nicht stark genug für zusätzliche Nutzerbelastung"
            )
    elif surplus >= 2.0:
        reasons.append("Tagesstrategie: trotz Push-Vorsprung stark genug")
    elif deficit >= 1.5:
        reasons.append("Tagesstrategie: Push-Rückstand, aber Qualitäts-Gates bleiben aktiv")
    return {"reasons": reasons, "blockers": blockers}


def _minimum_pressure_review(
    push_pacing: dict[str, Any],
    teams_alerts_today: int,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    minimum = max(0, int(config.min_alerts_per_day or 0))
    if minimum <= 0:
        return {"active": False, "label": "Mindest-Pacing deaktiviert", "thresholdDrop": 0.0}
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    hour = local_dt.hour + local_dt.minute / 60.0
    expected = float(push_pacing.get("expectedByNow") or 0.0)
    actual_pushes_today = push_pacing.get("pushesToday")
    actual_known = actual_pushes_today is not None
    current = _safe_int(actual_pushes_today) if actual_known else _safe_int(teams_alerts_today)
    basis = "realer Push-Bestand" if actual_known else "Teams-Hinweise"
    deficit = max(0.0, expected - current)
    late_day_floor = 0.0
    if hour >= 18:
        late_day_floor = max(0.0, minimum * 0.72 - current)
    if hour >= 21:
        late_day_floor = max(late_day_floor, minimum * 0.9 - current)
    pressure = max(deficit, late_day_floor)
    active = pressure >= 1.0
    threshold_drop = min(10.0, 3.0 + pressure * 2.0) if active else 0.0
    if not active:
        label = (
            f"Teams-Mindest-Pacing: {basis} im Plan fuer mindestens {minimum} Push-Empfehlungen"
        )
    else:
        label = (
            f"Teams-Mindest-Pacing aktiv ({basis}): Rueckstand {pressure:.1f} auf mindestens "
            f"{minimum} Empfehlungen, Schwellen werden kontrolliert gelockert"
        )
    return {
        "active": active,
        "minimum": minimum,
        "current": current,
        "basis": "actual_pushes" if actual_known else "teams_alerts",
        "actualPushesToday": _safe_int(actual_pushes_today) if actual_pushes_today is not None else None,
        "teamsAlertsToday": _safe_int(teams_alerts_today),
        "expectedByNow": round(expected, 2),
        "pressure": round(pressure, 2),
        "thresholdDrop": round(threshold_drop, 1),
        "label": label,
    }


def _editorial_cvd_review(
    candidate: dict[str, Any],
    *,
    score: float,
    predicted_or: float | None,
    freshness_hours: float | None,
    minutes_since_last_push: float | None,
    dashboard_rank: int,
    alert_score: float,
    pushes_today: int | None,
    now_ts: int,
    config: TeamsAlertConfig,
    minimum_pressure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hard editorial gate before Teams can become an action recommendation."""
    if not config.editorial_gate_enabled:
        return {
            "enabled": False,
            "approved": True,
            "score": 100.0,
            "newsValue": 100.0,
            "blockers": [],
            "reasons": ["CvD-Gate deaktiviert"],
            "breakdown": {},
        }

    title = _title(candidate)
    title_l = title.lower()
    section = _section(candidate).lower()
    breaking = _is_breaking(candidate)
    rank_limit = max(1, min(int(config.editorial_top_limit or 10), int(config.dashboard_top_limit or 20)))

    high_impact_terms = (
        "krieg", "terror", "anschlag", "iran", "israel", "ukraine", "russland",
        "putin", "trump", "merz", "kanzler", "regierung", "bundestag", "polizei",
        "tote", "tot", "vermisst", "verletzte", "gefahr", "warnung", "evakuierung",
        "ruecktritt", "rücktritt", "feuerpause", "atom", "nato", "gericht",
        "urteil", "streik", "insolvenz", "festnahme",
    )
    public_need_terms = (
        "warnung", "gefahr", "polizei", "streik", "ausfall", "sperrung", "rueckruf",
        "rückruf", "steuer", "rente", "krankenkasse", "geld", "preis", "verbraucher",
        "gericht", "urteil", "regierung", "bundestag", "nato", "krieg", "feuerpause",
    )
    soft_terms = (
        "quiz", "horoskop", "shopping", "rabatt", "sommertrend", "fans", "star",
        "stars", "app", "promi", "liebe", "beauty", "mode", "urlaub", "reise",
        "peinlich", "witzig",
    )
    vague_terms = ("diese", "dieser", "darum", "so ", "jetzt wissen", "experte erklaert")

    section_points = {
        "politik": 30.0,
        "news": 28.0,
        "wirtschaft": 23.0,
        "regional": 22.0,
        "digital": 16.0,
        "unterhaltung": 10.0,
        "leben": 10.0,
        "leben-wissen": 12.0,
        "service": 7.0,
    }.get(section, 17.0)

    impact_bonus = 0.0
    if breaking:
        impact_bonus += 8.0
    impact_matches = [term for term in high_impact_terms if term in title_l]
    if impact_matches:
        impact_bonus += min(10.0, 4.0 + 2.0 * (len(impact_matches) - 1))
    live_ticker = _is_live_ticker_title(title)
    live_ticker_has_update = _has_live_ticker_push_update(title)
    scheduled_process = _is_scheduled_process_without_update(title)
    abstract_explainer = _is_abstract_explainer_without_update(title)
    nonessential_curiosity = _is_nonessential_curiosity(title)
    low_civic_impact = _is_low_civic_impact_story(title, section)
    if live_ticker and live_ticker_has_update:
        impact_bonus += 2.0
    elif live_ticker:
        impact_bonus -= 6.0
    if scheduled_process and not breaking:
        impact_bonus -= 8.0
    if abstract_explainer and not breaking:
        impact_bonus -= 10.0
    if nonessential_curiosity and not breaking:
        impact_bonus -= 8.0
    if low_civic_impact and not breaking:
        impact_bonus -= 12.0
    soft_matches = [term for term in soft_terms if term in title_l]
    if soft_matches and not impact_matches and not breaking:
        impact_bonus -= 8.0
    is_soft_service = _is_soft_service_or_quiz(title)
    urgent_public_service = _is_urgent_public_service_title(title)
    if is_soft_service and not breaking and not urgent_public_service:
        impact_bonus -= 18.0

    news_value = _clamp(section_points + impact_bonus, 0.0, 40.0)

    if freshness_hours is None:
        urgency = 8.0
    elif freshness_hours <= 0.5:
        urgency = 16.0
    elif freshness_hours <= 1.5:
        urgency = 14.0
    elif freshness_hours <= 3.0:
        urgency = 11.0
    elif freshness_hours <= 6.0:
        urgency = 7.0
    else:
        urgency = 2.0
    if breaking:
        urgency = max(urgency, 14.0)

    user_need = 6.0
    need_matches = [term for term in public_need_terms if term in title_l]
    if need_matches:
        user_need += min(9.0, 4.0 + 1.5 * (len(need_matches) - 1))
    if section in {"politik", "news", "wirtschaft", "regional"}:
        user_need += 2.0
    if soft_matches and not need_matches and not impact_matches:
        user_need -= 4.0
    if low_civic_impact and not breaking:
        user_need -= 3.0
    user_need = _clamp(user_need, 0.0, 15.0)

    if predicted_or is None:
        timing = 4.0
    elif predicted_or >= config.min_or + 1.0:
        timing = 10.0
    elif predicted_or >= config.min_or:
        timing = 8.0
    elif predicted_or >= max(4.0, config.min_or - 0.5):
        timing = 5.0
    else:
        timing = 2.0

    time_fit = _time_fit_review(
        now_ts=now_ts,
        section=section,
        breaking=breaking,
        config=config,
        pushes_today=pushes_today,
    )
    pacing = _push_pacing_review(pushes_today, now_ts, config)

    clarity = 8.0 if len(title) >= 35 else 5.0
    if any(term in title_l for term in vague_terms):
        clarity -= 2.0
    if _url(candidate):
        clarity += 1.0
    clarity = _clamp(clarity, 0.0, 10.0)

    load = 4.0
    if minutes_since_last_push is None:
        load = 0.0
    elif minutes_since_last_push >= config.min_minutes_since_last_push + 20:
        load = 5.0
    elif minutes_since_last_push >= config.min_minutes_since_last_push:
        load = 3.0
    elif breaking:
        load = 2.0
    else:
        load = 0.0

    total = _clamp(
        news_value
        + urgency
        + user_need
        + timing
        + clarity
        + load
        + time_fit["score"]
        + pacing["editorialAdjustment"],
        0.0,
        100.0,
    )
    blockers: list[str] = []
    reasons: list[str] = [
        f"CvD-Score {total:.1f}/100",
        f"CvD-Nachrichtenwert {news_value:.1f}/40",
        f"CvD-Zeitfenster {time_fit['score']:.1f}/10: {time_fit['label']}",
        str(pacing["label"]),
    ]

    minimum_pressure = minimum_pressure or {}
    minimum_active = bool(minimum_pressure.get("active"))
    min_editorial_score = max(
        66.0,
        config.min_editorial_score - (6.0 if minimum_active and not breaking else 0.0),
    )

    if dashboard_rank > rank_limit and not breaking:
        blockers.append(f"CvD: nicht in den Top {rank_limit} des Dashboard-Felds")
    if news_value < config.min_editorial_news_value:
        blockers.append(
            f"CvD: Nachrichtenwert zu niedrig ({news_value:.1f} < {config.min_editorial_news_value:.1f})"
        )
    if total < min_editorial_score:
        blockers.append(f"CvD: redaktionelle Gesamtfreigabe zu schwach ({total:.1f} < {min_editorial_score:.1f})")
    if soft_matches and not breaking and news_value < config.min_editorial_news_value + 6.0:
        blockers.append("CvD: weiches Thema ohne ausreichenden aktuellen Nachrichtenwert")
    if is_soft_service and not breaking and not urgent_public_service:
        blockers.append("CvD: Service-/Raetsel-/Ratgeber-Format, nicht pushwuerdig")
    if live_ticker and not breaking and not live_ticker_has_update:
        blockers.append("CvD: Live-Ticker ohne neue pushwürdige Lage")
    if scheduled_process and not breaking:
        blockers.append("CvD: Termin-/Prozesslage ohne neue Entwicklung")
    if abstract_explainer and not breaking:
        blockers.append("CvD: Erklär-/Debattenstück ohne neue aktuelle Lage")
    if nonessential_curiosity and not breaking:
        blockers.append("CvD: Kurios-/Click-Reiz ohne ausreichenden öffentlichen Nachrichtenwert")
    if low_civic_impact and not breaking:
        blockers.append("CvD: enger Kurios-/Click-Reiz ohne ausreichend breite öffentliche Relevanz")
    missing_event_signal = config.event_gate_enabled and not breaking and not _has_news_event(title)
    if (
        missing_event_signal
        and minimum_active
        and urgent_public_service
        and news_value >= config.min_editorial_news_value + 8.0
        and total >= min_editorial_score + 8.0
        and (predicted_or is None or predicted_or >= max(4.3, config.min_or - 0.7))
    ):
        reasons.append("Teams-Mindest-Pacing: starkes CvD-Signal uebersteuert enges Ereignis-Gate")
    elif missing_event_signal:
        blockers.append("CvD: kein konkretes Nachrichten-Ereignis erkennbar (Service/Teaser)")
    if predicted_or is None and not breaking and alert_score < config.no_forecast_min_alert_score:
        blockers.append("CvD: ohne belastbare OR-Prognose nur bei absoluter Top-Lage")
    if not breaking and time_fit["score"] < config.min_time_fit_score:
        blockers.append(
            f"CvD: unguenstiges Zeitfenster ({time_fit['score']:.1f} < {config.min_time_fit_score:.1f})"
        )
    if (
        not breaking
        and time_fit.get("waitRecommended")
        and float(pacing.get("deficit") or 0.0) < 1.5
        and not minimum_active
    ):
        blockers.append(str(time_fit["waitReason"]))
    elif (
        not breaking
        and time_fit.get("waitRecommended")
        and minimum_active
    ):
        reasons.append("Teams-Mindest-Pacing: besseres Zeitfenster wird nicht abgewartet")

    if not blockers:
        reasons.append("CvD-Freigabe: klare Push-Lage mit aktueller Relevanz")

    return {
        "enabled": True,
        "approved": not blockers,
        "score": round(total, 1),
        "newsValue": round(news_value, 1),
        "blockers": blockers,
        "reasons": reasons,
        "breakdown": {
            "newsValue": round(news_value, 1),
            "urgency": round(urgency, 1),
            "userNeed": round(user_need, 1),
            "timing": round(timing, 1),
            "timeFit": round(float(time_fit["score"]), 1),
            "timeFitLabel": time_fit["label"],
            "slotAvgOR": time_fit.get("slotAvgOR"),
            "slotStars": time_fit.get("slotStars"),
            "slotTopCategory": time_fit.get("slotTopCategory"),
            "nextBetterSlot": time_fit.get("nextBetterSlot"),
            "pushPacing": pacing,
            "localHour": time_fit["localHour"],
            "weekday": time_fit["weekday"],
            "clarity": round(clarity, 1),
            "load": round(load, 1),
            "minEditorialScore": round(min_editorial_score, 1),
            "minimumPressure": minimum_pressure,
        },
    }


def _time_fit_review(
    *,
    now_ts: int,
    section: str,
    breaking: bool,
    config: TeamsAlertConfig,
    pushes_today: int | None,
) -> dict[str, Any]:
    local_dt = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin"))
    hour = local_dt.hour
    weekday = local_dt.weekday()
    is_weekend = weekday >= 5

    if 0 <= hour < 5:
        score, label = (5.0, "Nachtfenster nur für Breaking") if breaking else (1.0, "Nachtfenster")
    elif 5 <= hour < 7:
        score, label = (6.0, "frühes Morgenfenster") if breaking else (3.0, "frühes Morgenfenster")
    elif 7 <= hour < 10:
        score, label = 10.0, "starkes Morgenfenster"
    elif 10 <= hour < 12:
        score, label = 8.0, "gutes Vormittagsfenster"
    elif 12 <= hour < 14:
        score, label = 8.0, "Mittagsfenster"
    elif 14 <= hour < 17:
        score, label = 7.0, "Nachmittagsfenster"
    elif 17 <= hour < 20:
        score, label = 10.0, "starkes Feierabendfenster"
    elif 20 <= hour < 22:
        score, label = 7.0, "Abendfenster"
    elif 22 <= hour < 24:
        score, label = (6.0, "spätes Breaking-Fenster") if breaking else (3.0, "spätes Abendfenster")
    else:
        score, label = 4.0, "unbekanntes Zeitfenster"

    manual_score = score
    section_l = section.lower()
    if is_weekend:
        if section_l in {"wirtschaft", "digital", "service"} and not breaking:
            manual_score -= 1.0
        elif section_l in {"news", "regional", "unterhaltung"}:
            manual_score += 0.5
        label += " am Wochenende"
    else:
        if section_l in {"politik", "wirtschaft"} and 7 <= hour < 18:
            manual_score += 0.5
        if section_l == "unterhaltung" and 18 <= hour < 22:
            manual_score += 1.0
        label += " an einem Werktag"

    slot = _slot_baseline(hour, weekday)
    slot_avg = float(slot.get("avg_or") or 0.0) if slot else None
    slot_stars = int(slot.get("stars") or 0) if slot else 0
    slot_top_cat = str(slot.get("top_cat") or "").strip().lower() if slot else ""
    slot_score = _slot_baseline_score(hour, weekday, slot, breaking)
    section_fit_delta = _slot_section_fit_delta(section_l, slot_top_cat)

    score = manual_score * 0.35 + slot_score * 0.65 + section_fit_delta
    label_parts = [label]
    if slot_avg is not None:
        label_parts.append(f"historisch {slot_avg:.2f}% OR")
    if slot_stars >= 2:
        label_parts.append("Top-Slot")
    if slot_top_cat and _sections_match(section_l, slot_top_cat):
        label_parts.append(f"Ressort passt zum Slot ({_format_section(slot_top_cat)})")

    try:
        from app.push_schedule.weekly_baseline import PDF_KPI

        mandatory_hours = set(PDF_KPI.get("mandatory_hours") or [])
        avoid_hours = set(PDF_KPI.get("avoid_hours") or [])
    except Exception:
        mandatory_hours = {20, 21}
        avoid_hours = {10, 11}

    is_mandatory = hour in mandatory_hours
    is_avoid = hour in avoid_hours
    if is_mandatory:
        score = max(score, 8.5 if not breaking else 9.0)
        label_parts.append("Pflicht-/Goldfenster")
    if is_avoid and not breaking:
        score = min(score, 4.0)
        label_parts.append("historische Totzone")

    next_better = _next_better_slot(
        now_ts,
        score,
        config,
        current_avg_or=slot_avg,
        max_lookahead_hours=6 if is_avoid else 4,
    )
    pacing = _push_pacing_review(pushes_today, now_ts, config)
    meaningful_wait = bool(
        next_better
        and (
            is_avoid
            or (
                score < 7.0
                and float(next_better.get("orGain") or 0.0) >= 0.7
            )
            or float(next_better.get("score") or 0.0) >= score + 1.5
        )
    )
    wait_recommended = bool(
        not breaking
        and meaningful_wait
        and float(pacing.get("deficit") or 0.0) < 1.5
    )
    wait_reason = ""
    if wait_recommended:
        weak_slot_label = "eine historische Totzone" if is_avoid else "historisch schwächer"
        wait_reason = (
            f"CvD: aktuelles Push-Fenster ist {weak_slot_label}; "
            f"besseres Fenster um {next_better['hour']:02d}:00 Uhr abwarten"
        )

    return {
        "score": round(_clamp(score, 0.0, 10.0), 1),
        "label": "; ".join(label_parts),
        "localHour": hour,
        "weekday": weekday,
        "isWeekend": is_weekend,
        "slotAvgOR": round(slot_avg, 2) if slot_avg is not None else None,
        "slotStars": slot_stars,
        "slotTopCategory": slot_top_cat or None,
        "isMandatorySlot": is_mandatory,
        "isAvoidSlot": is_avoid,
        "waitRecommended": wait_recommended,
        "waitReason": wait_reason,
        "nextBetterSlot": next_better,
    }


def _quiet_hours_reason(now_ts: int, config: TeamsAlertConfig) -> str:
    start = _parse_hhmm_to_minutes(config.quiet_hours_start)
    end = _parse_hhmm_to_minutes(config.quiet_hours_end)
    if start is None or end is None or start == end:
        return ""
    local_dt = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin"))
    current = local_dt.hour * 60 + local_dt.minute
    if start < end:
        in_quiet_hours = start <= current < end
    else:
        in_quiet_hours = current >= start or current < end
    if not in_quiet_hours:
        return ""
    return f"Teams-Ruhezeit aktiv: {_format_hhmm(start)} bis {_format_hhmm(end)} Uhr"


def _parse_hhmm_to_minutes(value: Any) -> int | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _format_hhmm(minutes: int) -> str:
    hour, minute = divmod(int(minutes), 60)
    return f"{hour:02d}:{minute:02d}"


def _teams_alert_score(
    candidate: dict[str, Any],
    *,
    score: float,
    predicted_or: float | None,
    freshness_hours: float | None,
    minutes_since_last_push: float | None,
    recent_push_count_6h: int,
    pushes_today: int | None,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    title = _title(candidate)
    title_l = title.lower()
    section = _section(candidate).lower()
    breaking = _is_breaking(candidate)

    push_score_points = _clamp(score - 60.0, 0.0, 25.0)

    section_points = {
        "politik": 24.0,
        "news": 22.0,
        "regional": 18.0,
        "wirtschaft": 18.0,
        "sport": 16.0,
        "digital": 12.0,
        "unterhaltung": 9.0,
        "leben": 8.0,
        "leben-wissen": 8.0,
        "service": 3.0,
        "horoskop": 1.0,
    }.get(section, 14.0)
    high_impact_terms = (
        "krieg", "terror", "anschlag", "iran", "israel", "ukraine", "putin",
        "trump", "merz", "regierung", "bundestag", "polizei", "tote",
        "tot", "warnung", "gefahr", "ruecktritt", "rücktritt", "feuerpause",
    )
    soft_terms = ("horoskop", "quiz", "abo", "shopping", "rabatt", "sommertrend")
    impact_bonus = 0.0
    if breaking:
        impact_bonus += 6.0
    if any(term in title_l for term in high_impact_terms):
        impact_bonus += 4.0
    if "live-ticker" in title_l or "liveticker" in title_l:
        impact_bonus += 2.0
    if any(term in title_l for term in soft_terms):
        impact_bonus -= 5.0
    news_value_points = _clamp(section_points + impact_bonus, 0.0, 30.0)

    if freshness_hours is None:
        urgency_points = 6.0
    elif freshness_hours <= 0.5:
        urgency_points = 15.0
    elif freshness_hours <= 1.5:
        urgency_points = 13.0
    elif freshness_hours <= 3.0:
        urgency_points = 10.0
    elif freshness_hours <= 6.0:
        urgency_points = 7.0
    elif freshness_hours <= 12.0:
        urgency_points = 3.0
    else:
        urgency_points = 0.0
    if breaking:
        urgency_points = max(urgency_points, 12.0)

    competition_points = 12.0

    if predicted_or is None:
        timing_points = 4.0
    elif predicted_or >= config.min_or:
        timing_points = 10.0
    elif predicted_or >= max(4.0, config.min_or - 0.5):
        timing_points = 7.0
    else:
        timing_points = 3.0

    time_fit = _time_fit_review(
        now_ts=now_ts,
        section=section,
        breaking=breaking,
        config=config,
        pushes_today=pushes_today,
    )
    slot_adjustment = _clamp((float(time_fit["score"]) - 5.0) * 0.9, -4.0, 4.0)
    timing_points = _clamp(timing_points + slot_adjustment, 0.0, 12.0)

    load_penalty = 0.0
    if minutes_since_last_push is not None and not breaking:
        if minutes_since_last_push < 20:
            load_penalty -= 10.0
        elif minutes_since_last_push < config.min_minutes_since_last_push:
            load_penalty -= 5.0
    if recent_push_count_6h > config.max_pushes_last_6h + 4 and not breaking:
        load_penalty -= 10.0
    elif recent_push_count_6h > config.max_pushes_last_6h and not breaking:
        load_penalty -= 5.0

    total = _clamp(
        push_score_points
        + news_value_points
        + urgency_points
        + competition_points
        + timing_points
        + load_penalty,
        0.0,
        100.0,
    )
    reasons = [
        f"Nachrichtenwert {news_value_points:.0f}/30",
        f"Aktualitaet {urgency_points:.0f}/15",
        f"Timing/OR {timing_points:.0f}/10",
        f"Zeitfenster {time_fit['score']:.1f}/10",
    ]
    if load_penalty < 0:
        reasons.append(f"Nutzerbelastung {load_penalty:.0f} Punkte")
    if predicted_or is None:
        reasons.append("OR nicht belastbar, deshalb nur kleiner Timing-Bonus")

    return {
        "score": round(total, 1),
        "breakdown": {
            "pushScore": round(push_score_points, 1),
            "newsValue": round(news_value_points, 1),
            "urgency": round(urgency_points, 1),
            "competition": round(competition_points, 1),
            "timing": round(timing_points, 1),
            "timeFit": round(float(time_fit["score"]), 1),
            "slotAdjustment": round(slot_adjustment, 1),
            "loadPenalty": round(load_penalty, 1),
        },
        "reasons": reasons,
    }


def _candidate_predicted_or(candidate: dict[str, Any]) -> float | None:
    return _candidate_model_forecast(candidate)


def _candidate_forecast(
    candidate: dict[str, Any],
    now_ts: int | None = None,
    suspect_values: set[float] | None = None,
) -> dict[str, Any]:
    model_value = _candidate_model_forecast(candidate, suspect_values)
    basis = str(candidate.get("predictedORBasis") or "").strip()
    confidence = _safe_float(candidate.get("predictedORConfidence"))
    if model_value is not None:
        explanation = "Artikelmodell"
        if basis:
            explanation += f": {basis}"
        if confidence is not None:
            explanation += f", Konfidenz {_format_number(confidence * 100, 0)} %"
        return {
            "value": model_value,
            "source": "article_model",
            "basis": basis or "model",
            "confidence": confidence,
            "explanation": explanation,
        }

    return _historical_slot_forecast(candidate, now_ts)


def _candidate_model_forecast(
    candidate: dict[str, Any],
    suspect_values: set[float] | None = None,
) -> float | None:
    if bool(candidate.get("predictedORIsFallback")):
        return None
    basis = str(candidate.get("predictedORBasis") or "").strip().lower()
    if basis in {"global_avg", "error_fallback"}:
        return None
    try:
        confidence = float(candidate.get("predictedORConfidence"))
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and confidence <= 0.1:
        return None
    value = normalize_predicted_or(candidate.get("predictedOR", candidate.get("predictedOpenRate")))
    if value is not None and suspect_values:
        if any(abs(value - suspect) <= 0.01 for suspect in suspect_values):
            return None
    return value


def _historical_slot_forecast(candidate: dict[str, Any], now_ts: int | None = None) -> dict[str, Any]:
    local_dt = dt.datetime.fromtimestamp(int(now_ts or time.time()), ZoneInfo("Europe/Berlin"))
    hour = local_dt.hour
    weekday = local_dt.weekday()
    value: float | None = None
    basis = "historical_slot_baseline"
    confidence: float | None = None
    explanation = ""

    try:
        from app.push_schedule.weekly_baseline import (
            PDF_HOUR_AVG,
            PDF_OVERALL_AVG,
            baseline_for,
        )

        slot = baseline_for(hour, weekday)
        if isinstance(slot, dict) and slot.get("avg_or"):
            value = float(slot["avg_or"])
            count = int(slot.get("count") or 0)
            confidence = round(_clamp(count / 250.0, 0.25, 0.75), 3)
            explanation = (
                f"historische Slot-Prognose: {_weekday_label(weekday)} {hour:02d}:00, "
                f"n={count}"
            )
        else:
            value = float(PDF_HOUR_AVG.get(hour, PDF_OVERALL_AVG))
            confidence = 0.25
            basis = "historical_hour_baseline"
            explanation = f"historische Stunden-Prognose: {hour:02d}:00"
    except Exception as exc:
        log.warning("[TeamsAlert] historical forecast unavailable: %s", exc)

    normalized = normalize_predicted_or(value)
    return {
        "value": normalized,
        "source": "historical_slot_baseline" if basis == "historical_slot_baseline" else basis,
        "basis": basis,
        "confidence": confidence,
        "explanation": explanation or "historische OR-Prognose",
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _local_day_start_ts(now_ts: int) -> int:
    """Unix timestamp of the current local (Europe/Berlin) day start."""
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    midnight = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp())


def _slot_baseline(hour: int, weekday: int) -> dict[str, Any]:
    try:
        from app.push_schedule.weekly_baseline import (
            PDF_HOUR_AVG,
            PDF_OVERALL_AVG,
            baseline_for,
        )

        slot = baseline_for(hour, weekday)
        if slot:
            return dict(slot)
        return {
            "avg_or": float(PDF_HOUR_AVG.get(hour, PDF_OVERALL_AVG)),
            "count": 0,
            "top_cat": None,
            "stars": 0,
            "source": "hour_avg",
        }
    except Exception as exc:
        log.warning("[TeamsAlert] slot baseline unavailable: %s", exc)
        return {}


def _slot_baseline_score(
    hour: int,
    weekday: int,
    slot: dict[str, Any] | None,
    breaking: bool,
) -> float:
    slot = slot or _slot_baseline(hour, weekday)
    avg_or = _safe_float(slot.get("avg_or")) if slot else None
    stars = int(slot.get("stars") or 0) if slot else 0
    if avg_or is None:
        score = 4.5
    else:
        score = 2.0 + ((avg_or - 4.2) / (7.3 - 4.2)) * 7.0
    score += stars * 0.25
    try:
        from app.push_schedule.weekly_baseline import PDF_KPI

        mandatory_hours = set(PDF_KPI.get("mandatory_hours") or [])
        avoid_hours = set(PDF_KPI.get("avoid_hours") or [])
    except Exception:
        mandatory_hours = {20, 21}
        avoid_hours = {10, 11}
    if hour in mandatory_hours:
        score = max(score, 8.5)
    if hour in avoid_hours and not breaking:
        score = min(score, 3.5)
    return _clamp(score, 0.0, 10.0)


def _sections_match(section: str, top_cat: str) -> bool:
    section_l = (section or "").lower()
    top_l = (top_cat or "").lower()
    aliases = {
        "geld": "wirtschaft",
        "wirtschaft": "wirtschaft",
        "politik": "politik",
        "news": "news",
        "regional": "regional",
        "unterhaltung": "unterhaltung",
        "digital": "digital",
    }
    return aliases.get(section_l, section_l) == aliases.get(top_l, top_l)


def _slot_section_fit_delta(section: str, top_cat: str) -> float:
    if not top_cat:
        return 0.0
    if _sections_match(section, top_cat):
        return 0.8
    if section in {"politik", "news"} and top_cat in {"politik", "news"}:
        return 0.2
    if top_cat in {"unterhaltung", "regional"} and section in {"politik", "wirtschaft"}:
        return -0.4
    return -0.2


def _slot_weight(hour: int, weekday: int, config: TeamsAlertConfig) -> float:
    if hour < int(config.active_hours_start) or hour > int(config.active_hours_end):
        return 0.0
    slot = _slot_baseline(hour, weekday)
    avg_or = _safe_float(slot.get("avg_or")) if slot else None
    try:
        from app.push_schedule.weekly_baseline import PDF_KPI, PDF_OVERALL_AVG

        overall = float(PDF_OVERALL_AVG)
        mandatory_hours = set(PDF_KPI.get("mandatory_hours") or [])
        avoid_hours = set(PDF_KPI.get("avoid_hours") or [])
    except Exception:
        overall = 5.44
        mandatory_hours = {20, 21}
        avoid_hours = {10, 11}
    weight = (avg_or or overall) / overall
    if hour in mandatory_hours:
        weight *= 1.45
    if hour in avoid_hours:
        weight *= 0.35
    return _clamp(weight, 0.15, 1.9)


def _next_better_slot(
    now_ts: int,
    current_score: float,
    config: TeamsAlertConfig,
    *,
    current_avg_or: float | None = None,
    max_lookahead_hours: int = 4,
) -> dict[str, Any] | None:
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    weekday = local_dt.weekday()
    current_hour = local_dt.hour
    best: dict[str, Any] | None = None
    horizon = min(23, int(config.active_hours_end), current_hour + max(1, int(max_lookahead_hours)))
    for hour in range(current_hour + 1, horizon + 1):
        slot = _slot_baseline(hour, weekday)
        slot_score = _slot_baseline_score(hour, weekday, slot, breaking=False)
        avg_or = _safe_float(slot.get("avg_or")) if slot else None
        or_gain = (
            round(avg_or - current_avg_or, 2)
            if avg_or is not None and current_avg_or is not None
            else None
        )
        score_gain = slot_score - current_score
        if score_gain < 1.0 and (or_gain is None or or_gain < 0.7):
            continue
        candidate = {
            "hour": hour,
            "score": round(slot_score, 1),
            "avgOR": round(avg_or, 2) if avg_or is not None else None,
            "orGain": or_gain,
            "scoreGain": round(score_gain, 1),
            "stars": int(slot.get("stars") or 0) if slot else 0,
        }
        if best is None or (
            float(candidate.get("orGain") or 0.0),
            float(candidate.get("score") or 0.0),
        ) > (
            float(best.get("orGain") or 0.0),
            float(best.get("score") or 0.0),
        ):
            best = candidate
    return best


def _expected_pushes_by_now(now_ts: int, config: TeamsAlertConfig) -> float:
    """Estimate how many pushes a CvD should have sent by this time of day.

    The curve is not linear: it follows the historical slot baseline, so weak
    10/11 Uhr windows carry little target pressure while 20/21 Uhr carry more.
    """
    target = max(0, int(config.target_pushes_per_day or 0))
    if target <= 0:
        return 0.0
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    weekday = local_dt.weekday()
    hour = local_dt.hour + local_dt.minute / 60.0
    start = int(_clamp(config.active_hours_start, 0.0, 23.0))
    end = int(_clamp(config.active_hours_end, start + 0.1, 23.0))
    if hour <= start:
        return 0.0
    if hour >= end:
        return float(target)
    weights = {h: _slot_weight(h, weekday, config) for h in range(start, end + 1)}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    elapsed_weight = 0.0
    for slot_hour, weight in weights.items():
        if hour >= slot_hour + 1:
            elapsed_weight += weight
        elif slot_hour <= hour < slot_hour + 1:
            elapsed_weight += weight * (hour - slot_hour)
    fraction = _clamp(elapsed_weight / total_weight, 0.0, 1.0)
    return round(target * fraction, 2)


def _push_pacing_review(
    pushes_today: int | None,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    expected = _expected_pushes_by_now(now_ts, config)
    if pushes_today is None:
        return {
            "known": False,
            "pushesToday": None,
            "expectedByNow": expected,
            "deficit": 0.0,
            "surplus": 0.0,
            "editorialAdjustment": 0.0,
            "label": "Tagespacing: Push-Bestand heute nicht bekannt",
        }
    deficit = max(0.0, expected - pushes_today)
    surplus = max(0.0, pushes_today - expected)
    adjustment = 0.0
    if deficit >= 1.5:
        adjustment = min(4.0, deficit * 0.8)
    elif surplus >= 2.0:
        adjustment = -min(7.0, surplus * 1.1)
    if deficit >= 1.5:
        label = f"Tagespacing: Rueckstand ({pushes_today} statt {expected:.1f} Pushes)"
    elif surplus >= 2.0:
        label = f"Tagespacing: Vorsprung ({pushes_today} statt {expected:.1f} Pushes)"
    else:
        label = f"Tagespacing: im Plan ({pushes_today} statt {expected:.1f} Pushes)"
    return {
        "known": True,
        "pushesToday": pushes_today,
        "expectedByNow": expected,
        "deficit": round(deficit, 2),
        "surplus": round(surplus, 2),
        "editorialAdjustment": round(adjustment, 1),
        "label": label,
    }


def _dynamic_alert_threshold(
    base_threshold: float,
    pushes_today: int | None,
    now_ts: int,
    breaking: bool,
    config: TeamsAlertConfig,
) -> tuple[float, str]:
    """Adjust the Teams-readiness threshold based on the daily push budget.

    Too few pushes so far → threshold may drop slightly so good chances are
    not missed. Already many pushes today → threshold rises to protect users.
    Breaking news never raises the threshold.
    """
    if not config.dynamic_threshold_enabled or pushes_today is None:
        return base_threshold, ""

    target = max(0, int(config.target_pushes_per_day or 0))
    expected = _expected_pushes_by_now(now_ts, config)
    max_drop = max(0.0, float(config.dynamic_threshold_max_drop or 0.0))
    max_rise = max(0.0, float(config.dynamic_threshold_max_rise or 0.0))

    if target and pushes_today >= target and not breaking:
        surplus = pushes_today - target
        rise = min(max_rise, 4.0 + surplus * 3.0)
        return base_threshold + rise, (
            f"Tagesbudget erreicht ({pushes_today}/{target} Pushes): Schwelle +{rise:.0f}"
        )

    ahead = pushes_today - expected
    if ahead >= 2.0 and not breaking:
        rise = min(max_rise, ahead * 1.5)
        return base_threshold + rise, (
            f"Push-Vorsprung heute ({pushes_today} statt {expected:.0f}): Schwelle +{rise:.0f}"
        )

    deficit = expected - pushes_today
    if deficit >= 1.5:
        drop = min(max_drop, deficit * 1.5)
        return base_threshold - drop, (
            f"Push-Rueckstand heute ({pushes_today} statt {expected:.0f}): Schwelle -{drop:.0f}"
        )

    return base_threshold, ""


def _effective_thresholds(config: TeamsAlertConfig, breaking: bool) -> tuple[float, float, int]:
    if config.score_only_mode:
        return config.min_score, config.min_or, config.min_minutes_since_last_push
    if breaking and config.breaking_override:
        return (
            min(config.min_score, config.breaking_min_score),
            min(config.min_or, config.breaking_min_or),
            min(config.min_minutes_since_last_push, config.breaking_min_minutes_since_last_push),
        )
    return config.min_score, config.min_or, config.min_minutes_since_last_push


def _realert_blocker_or_reason(
    candidate: dict[str, Any],
    alert_state: dict[str, Any] | None,
    score: float,
    predicted_or: float | None,
    breaking: bool,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, str]:
    if not alert_state:
        return {}

    alert_status = str(alert_state.get("status") or "")
    last_decision_ts = _safe_int(alert_state.get("last_decision_ts"))
    if alert_status == "sending" and now_ts - last_decision_ts < 15 * 60:
        return {"blocker": "Teams-Hinweis wird bereits versendet"}
    if (
        alert_status == "failed"
        and last_decision_ts
        and now_ts - last_decision_ts < max(
            config.alert_cooldown_minutes * 60,
            config.repeat_suppression_hours * 3600,
        )
    ):
        return {"blocker": f"Bereits als Teams-Kandidat versucht: Sperre {config.repeat_suppression_hours}h"}
    if alert_status != "sent":
        return {}

    last_alert_ts = _safe_int(alert_state.get("last_alert_ts"))
    cooldown_ok = now_ts - last_alert_ts >= config.alert_cooldown_minutes * 60
    improvements: list[str] = []

    old_score = float(alert_state.get("last_score") or 0.0)
    old_or = float(alert_state.get("last_predicted_or") or 0.0)
    old_breaking = bool(alert_state.get("last_is_breaking"))
    old_title = str(alert_state.get("article_title") or "")
    cur_title = _title(candidate)

    if score - old_score >= config.realert_score_delta:
        improvements.append(f"Score deutlich gestiegen (+{score - old_score:.1f})")
    if predicted_or is not None and predicted_or - old_or >= config.realert_or_delta:
        improvements.append(f"Prognose deutlich besser (+{predicted_or - old_or:.2f}pp)")
    # Ein bloßer modDate-Bump ist KEIN Re-Alert-Grund: BILD setzt Artikel
    # (gerade Evergreens/Ticker) staendig neu, das wuerde dieselbe Empfehlung
    # taeglich wiederholen. Nur eine inhaltlich deutlich geaenderte Schlagzeile
    # zaehlt als echte neue Lage.
    if old_title and cur_title:
        similarity = _token_similarity(_tokens(cur_title), _tokens(old_title))
        if similarity < 0.5:
            improvements.append("Schlagzeile inhaltlich deutlich geaendert")
    if breaking and not old_breaking:
        improvements.append("Breaking-News-Status neu")

    if not improvements:
        return {"blocker": "Bereits per Teams gemeldet"}
    if not cooldown_ok:
        return {"blocker": "Re-Alert-Cooldown noch nicht erfuellt"}
    return {"positive": "Re-Alert wegen relevanter Veraenderung: " + ", ".join(improvements[:2])}


def _already_pushed_reason(
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    now_ts: int | None = None,
    config: TeamsAlertConfig | None = None,
) -> str:
    """Block candidates that we already pushed live (exact article OR same story).

    Exakte Artikel-URL zaehlt im gesamten Verlauf; die unscharfe Themen-Erkennung
    (URL-Slug-/Titel-Aehnlichkeit) nur innerhalb des konfigurierten Fensters, damit
    sich entwickelnde Stories ueber Tage nicht dauerhaft gesperrt werden.
    """
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    window_start = now - int(max(0.0, config.pushed_topic_window_hours) * 3600)

    url = _normalize_url(_url(candidate))
    title = _title(candidate)
    title_tokens = _tokens(title)
    slug_tokens = _url_slug_tokens(_url(candidate))

    for item in history:
        item_url = _normalize_url(_url(item))
        if url and item_url and url == item_url:
            return "Bereits live gepusht (gleiche Artikel-URL)"
        item_title = _title(item)
        if title and item_title and _normalize_title(title) == _normalize_title(item_title):
            return "Bereits live gepusht (identischer Titel)"

        # Unscharfe Themen-Erkennung nur fuer kuerzlich Gepushtes.
        item_ts = _safe_int(item.get("ts_num", item.get("ts", 0)))
        if item_ts and item_ts < window_start:
            continue
        if _same_topic(slug_tokens, _url_slug_tokens(_url(item)), 0.6):
            return "Bereits live gepusht (gleiche Story, URL-Slug)"
        if _same_topic(title_tokens, _tokens(item_title), 0.6):
            return "Bereits live gepusht (sehr aehnliche Meldung)"
    return ""


def _url_slug_tokens(url: str) -> set[str]:
    """Bedeutungstragende Tokens aus dem URL-Pfad/Slug (ohne Ressort-Segmente)."""
    from urllib.parse import urlsplit

    path = urlsplit(str(url or "")).path.lower()
    drop = {
        "politik", "inland", "ausland", "sport", "fussball", "fußball", "news",
        "regional", "unterhaltung", "stars", "leute", "leben", "wissen", "auto",
        "geld", "wirtschaft", "digital", "video", "videos", "ratgeber", "reise",
        "spiele", "lifestyle", "mobil", "bild", "html", "amp", "www",
    }
    raw = re.split(r"[/\-_.]+", path)
    return {
        token for token in raw
        if len(token) >= 4 and token not in drop and not token.isdigit()
    } - _STOP_WORDS


def _same_topic(a: set[str], b: set[str], threshold: float) -> bool:
    """Overlap-Koeffizient + mind. ein markantes gemeinsames Token (robust gegen
    unterschiedlich lange Slugs/Titel zur selben Story)."""
    if not a or not b:
        return False
    shared = a & b
    if len(shared) < 2 or not any(len(token) >= 5 for token in shared):
        return False
    return len(shared) / max(1, min(len(a), len(b))) >= threshold


def _topic_already_alerted_reason(
    candidate: dict[str, Any],
    candidate_key_value: str,
    recent_alerts: list[dict[str, Any]],
    config: TeamsAlertConfig,
) -> str:
    """Block a second alert about the same event already reported via Teams.

    Catches near-duplicate topics from different articles (e.g. two headlines
    about the same explosion). The candidate's own key is handled by the re-alert
    logic and is skipped here.
    """
    title = _title(candidate)
    title_tokens = _tokens(title)
    if not title_tokens:
        return ""
    threshold = float(config.topic_dedup_similarity or 0.5)
    for entry in recent_alerts:
        other_key = str(entry.get("key") or "")
        if other_key and other_key == candidate_key_value:
            continue
        other_tokens = _tokens(str(entry.get("title") or ""))
        if not other_tokens:
            continue
        shared = title_tokens & other_tokens
        # Mindestens ein markantes (Eigennamen-/langes) gemeinsames Token verlangen,
        # damit nur echte Themen-Dubletten greifen, nicht zufaellige Wortueberschneidung.
        if not any(len(token) >= 5 for token in shared):
            continue
        # Overlap-Koeffizient: robust gegen unterschiedlich lange Schlagzeilen
        # zur selben Lage ("13 Tote in Hafen" vs "13 Tote bei Explosion in Katar").
        overlap = len(shared) / max(1, min(len(title_tokens), len(other_tokens)))
        if overlap >= threshold:
            return (
                "Thema bereits per Teams gemeldet (Dublette): "
                f"\"{_compact_text(str(entry.get('title') or ''), 80)}\""
            )
    return ""


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    score = _score(candidate)
    predicted = _candidate_predicted_or(candidate) or 0.0
    breaking = 1.0 if _is_breaking(candidate) else 0.0
    freshness = _candidate_updated_ts(candidate) or 0
    return (score, predicted, breaking, float(freshness))


def _log_decision(candidate: dict[str, Any], decision: dict[str, Any]) -> None:
    blocking_reasons = list(decision.get("blockingReasons") or [])
    positive_reasons = list(decision.get("reasons") or [])
    reason_text = "; ".join([*blocking_reasons, *positive_reasons])
    log.info(
        "[TeamsAlert] decision candidateId=%s articleId=%s url=%s score=%.1f predicted_or=%s "
        "expected_visits=%s estimated_reach=%s teams_alert_score=%s editorial_score=%s "
        "selection_score=%s visit_score=%s last_push=%s "
        "decision=%s reasons=%s evaluated_at=%s",
        decision.get("candidateId"),
        decision.get("articleId"),
        decision.get("articleUrl"),
        float(decision.get("score") or 0.0),
        decision.get("predictedOR"),
        decision.get("expectedVisits"),
        decision.get("estimatedReach"),
        decision.get("teamsAlertScore"),
        decision.get("editorialScore"),
        decision.get("selectionScore"),
        decision.get("visitPotentialScore"),
        decision.get("lastPushAt"),
        "notify" if decision.get("shouldNotify") else "skip",
        reason_text,
        decision.get("evaluatedAt"),
    )


def _title(candidate: dict[str, Any]) -> str:
    return str(candidate.get("title") or candidate.get("headline") or "").strip()


def _url(candidate: dict[str, Any]) -> str:
    return str(candidate.get("url") or candidate.get("link") or "").strip()


def _section(candidate: dict[str, Any]) -> str:
    return str(candidate.get("category") or candidate.get("cat") or "news").strip() or "news"


def _score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_breaking(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("isBreaking")
        or candidate.get("isEilmeldung")
        or candidate.get("is_eilmeldung")
    )


def _candidate_updated_ts(candidate: dict[str, Any]) -> int:
    for key in ("modDate", "updatedAt", "pubDate", "publishedAt"):
        parsed = _parse_ts(candidate.get(key))
        if parsed:
            return parsed
    return 0


def _freshness_hours(candidate: dict[str, Any], now_ts: int) -> float | None:
    published = _parse_ts(candidate.get("pubDate") or candidate.get("publishedAt"))
    if not published:
        return None
    return max(0.0, (now_ts - published) / 3600.0)


def _parse_ts(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str) or not value.strip():
        return 0
    try:
        return int(float(value))
    except ValueError:
        pass
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp())
    except ValueError:
        return 0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weekday_label(weekday: int) -> str:
    labels = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
    if 0 <= int(weekday) < len(labels):
        return labels[int(weekday)]
    return "Wochentag"


def _iso_from_ts(ts_value: int) -> str | None:
    if not ts_value:
        return None
    return dt.datetime.fromtimestamp(int(ts_value)).isoformat()


def _format_dt(ts_value: int) -> str:
    return dt.datetime.fromtimestamp(ts_value).strftime("%d.%m.%Y %H:%M")


def _format_time(ts_value: int) -> str:
    return dt.datetime.fromtimestamp(ts_value).strftime("%H:%M")


def _format_number(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def _format_int(value: float | int) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", ".")


def _format_or(value: float | None) -> str:
    return f"{_format_number(float(value), 2)} % OR" if value is not None else "keine belastbare Prognose"


def _format_forecast(forecast: dict[str, Any] | None) -> str:
    if not forecast or forecast.get("value") is None:
        return "keine belastbare Prognose"
    label = _format_or(float(forecast["value"]))
    explanation = str(forecast.get("explanation") or "").strip()
    return f"{label} ({explanation})" if explanation else label


def _forecast_sentence(forecast: dict[str, Any] | None) -> str:
    if not forecast or forecast.get("value") is None:
        return "Es gibt aktuell keine belastbare OR-Prognose."
    source = str(forecast.get("source") or "")
    if source == "article_model":
        return f"Die Artikel-Prognose liegt aktuell bei {_format_or(float(forecast['value']))}."
    return f"Die Zeitfenster-Prognose liegt aktuell bei {_format_or(float(forecast['value']))}."


def _format_section(value: str) -> str:
    label = str(value or "").strip()
    if not label:
        return "News"
    known = {
        "politik": "Politik",
        "sport": "Sport",
        "unterhaltung": "Unterhaltung",
        "wirtschaft": "Wirtschaft",
        "regional": "Regional",
        "digital": "Digital",
        "news": "News",
    }
    return known.get(label.lower(), label[:1].upper() + label[1:])


def _same_editorial_text(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    return bool(normalize(left)) and normalize(left) == normalize(right)


_GENERIC_PUSH_PHRASES = (
    "darum geht es jetzt",
    "das ist jetzt wichtig",
    "was jetzt wichtig ist",
    "was jetzt passiert",
    "das bedeutet das für sie",
    "das bedeutet das fuer sie",
    "das musst du wissen",
    "das musst du jetzt wissen",
    "hier alle infos",
    "alle infos",
    "das steckt dahinter",
    "so reagiert das netz",
)


def _is_generic_push_title(text: str) -> bool:
    """True if a push title is just a generic, value-free filler phrase."""
    lowered = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    if not lowered:
        return True
    return any(phrase in lowered for phrase in _GENERIC_PUSH_PHRASES)


def _sanitize_push_title(text: str, *, breaking: bool) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not breaking:
        value = re.sub(r"(?i)^\s*(?:eil|breaking)\s*(?::|!|\s-\s)\s*", "", value).strip()
    return value


_LIVE_TICKER_UPDATE_TERMS = (
    "urteil", "verurteilt", "freigesprochen", "entscheidung", "festnahme",
    "festgenommen", "verhaftet", "razzia", "gesteht", "geständnis", "gestaendnis",
    "tot", "tote", "verletzte", "explosion", "brand", "anschlag", "angriff",
    "warnung", "evakuierung", "evakuiert", "rücktritt", "ruecktritt",
    "tritt zurück", "tritt zurueck", "zurückgetreten", "zurueckgetreten",
    "feuerpause", "waffenruhe", "einigung", "eskaliert", "bestätigt", "bestaetigt",
    "stoppt", "beginnt", "beendet", "abgesagt", "geschlossen", "gesperrt",
)
_LIVE_TICKER_SCHEDULE_PATTERNS = (
    r"\bsagen heute aus\b",
    r"\bheute (?:als )?zeugen\b",
    r"\bim zeugenstand\b",
    r"\bvor dem zeugenstand\b",
    r"\bwie verhielt sich\b",
    r"\bantwortet auf fragen\b",
    r"\bheute im prozess\b",
)
_LIVE_TICKER_SCHEDULE_RE = re.compile("|".join(_LIVE_TICKER_SCHEDULE_PATTERNS), re.IGNORECASE)


def _is_live_ticker_title(title: str) -> bool:
    text = str(title or "").casefold()
    return "live-ticker" in text or "liveticker" in text or "liveblog" in text


def _has_live_ticker_push_update(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if _LIVE_TICKER_SCHEDULE_RE.search(text):
        return False
    return any(term in text for term in _LIVE_TICKER_UPDATE_TERMS)


_HARD_PUBLIC_NEED_TERMS = (
    "warnung", "gefahr", "evakuierung", "evakuiert", "sperrung", "ausfall",
    "rückruf", "rueckruf", "streik", "hochwasser", "unwetter", "hitzewarnung",
    "polizei", "terror", "anschlag", "angriff", "krieg", "feuerpause",
    "waffenruhe", "tote", "tot", "verletzte", "vermisst", "festnahme",
    "festgenommen", "urteil", "verurteilt", "rücktritt", "ruecktritt",
    "zurückgetreten", "zurueckgetreten", "insolvenz", "pleite",
)


def _has_hard_public_need(title: str, section: str = "") -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if any(_contains_editorial_term(text, term) for term in _HARD_PUBLIC_NEED_TERMS):
        return True
    if re.search(r"\b\d+\s+(?:tote|verletzte|opfer|festnahmen)\b", text):
        return True
    return str(section or "").strip().lower() == "politik" and any(
        term in text for term in ("regierung beschliesst", "regierung beschließt", "bundestag beschliesst", "bundestag beschließt")
    )


_BROAD_PUBLIC_IMPACT_TERMS = (
    "deutschland", "bundesweit", "regierung", "bundestag", "kanzler", "krieg",
    "iran", "israel", "ukraine", "russland", "nato", "bahn", "deutsche bahn",
    "streik", "ausfall", "sperrung", "warnung", "rückruf", "rueckruf", "steuer",
    "rente", "krankenkasse", "preis", "geld", "polizei", "razzia", "gasversorgung",
    "terror", "anschlag", "angriff", "explosion", "brand", "tote", "verletzte",
    "vermisst", "evakuierung", "unwetter", "hochwasser", "hitzewarnung",
)


def _has_broad_public_impact(title: str, section: str = "") -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if _has_hard_public_need(text, section):
        return True
    if any(_contains_editorial_term(text, term) for term in _BROAD_PUBLIC_IMPACT_TERMS):
        return True
    if re.search(r"\b\d+\s+(?:tote|verletzte|opfer|festnahmen|festgenommen)\b", text):
        return True
    return str(section or "").strip().lower() in {"politik", "wirtschaft"} and _has_news_event(text)


_LOW_CIVIC_IMPACT_PATTERNS = (
    r"\bbienen?\b",
    r"\bbienenstich\b",
    r"\bshoweinlage\b",
    r"\bstar am seil\b",
    r"\bplank\b",
    r"\bfitness-test\b",
    r"\bgta\s*6\b",
    r"\bvorbestellungen?\b",
    r"\bfans\b",
    r"\bpromi\b",
    r"\burlaub\b",
    r"\breise\b",
    r"\bbungee\b",
)
_LOW_CIVIC_IMPACT_RE = re.compile("|".join(_LOW_CIVIC_IMPACT_PATTERNS), re.IGNORECASE)
_ACCIDENT_PUBLIC_IMPACT_RE = re.compile(
    r"\b(?:tote|toter|tot|verletzte|schwerverletzt|opfer|vermisst|"
    r"gesperrt|sperrung|stau|evakuierung|evakuiert|warnung|gefahr|"
    r"brand|explosion|anschlag|terror)\b",
    re.IGNORECASE,
)


def _is_low_civic_impact_story(title: str, section: str = "") -> bool:
    """Click-/Kurios-Lage ohne breite öffentliche Relevanz.

    This is intentionally conservative: a real warning, disruption, fatality or
    political/public-service event is not downgraded here.
    """
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if _has_broad_public_impact(text, section):
        return False
    if _is_nonessential_curiosity(text) or _LOW_CIVIC_IMPACT_RE.search(text):
        return True
    if _is_soft_service_or_quiz(text) or _is_abstract_explainer_without_update(text):
        return True
    if _contains_editorial_term(text, "unfall") and not _ACCIDENT_PUBLIC_IMPACT_RE.search(text):
        return True
    return False


def _contains_editorial_term(text: str, term: str) -> bool:
    term = str(term or "").strip().casefold()
    if not term:
        return False
    if " " in term:
        return term in text
    return bool(re.search(rf"(?<![a-z0-9äöüß]){re.escape(term)}(?![a-z0-9äöüß])", text))


_SCHEDULED_PROCESS_PATTERNS = (
    r"\bsagen heute aus\b",
    r"\bheute (?:als )?zeugen\b",
    r"\bim zeugenstand\b",
    r"\bvor gericht\b.*\bheute\b",
    r"\bprozessauftakt\b",
    r"\bheute im prozess\b",
)
_SCHEDULED_PROCESS_RE = re.compile("|".join(_SCHEDULED_PROCESS_PATTERNS), re.IGNORECASE)


def _is_scheduled_process_without_update(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if not _SCHEDULED_PROCESS_RE.search(text):
        return False
    return not _has_live_ticker_push_update(text)


_ABSTRACT_EXPLAINER_PATTERNS = (
    r"\bauf dem prüfstand\b",
    r"\bauf dem pruefstand\b",
    r"\bvorurteil",
    r"\bgefährlicher als\b",
    r"\bgefaehrlicher als\b",
    r"\bhäufiger als\b",
    r"\bhaeufiger als\b",
    r"\bstimmt das\b",
    r"\bwas .* bedeutet\b",
    r"\bdas bedeutet\b",
)
_ABSTRACT_EXPLAINER_RE = re.compile("|".join(_ABSTRACT_EXPLAINER_PATTERNS), re.IGNORECASE)


def _is_abstract_explainer_without_update(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if "?" not in text and not _ABSTRACT_EXPLAINER_RE.search(text):
        return False
    return not _has_hard_public_need(text)


_CURIOSITY_PATTERNS = (
    r"\bschock auf dem highway\b",
    r"\bkurios\b",
    r"\bskurril\b",
    r"\bunglaublich\b",
    r"\bunfassbar\b",
)
_CURIOSITY_RE = re.compile("|".join(_CURIOSITY_PATTERNS), re.IGNORECASE)


def _is_nonessential_curiosity(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text or not _CURIOSITY_RE.search(text):
        return False
    return not _has_hard_public_need(text)


_SPECULATIVE_MARKERS = (
    "wohl", "offenbar", "angeblich", "vermutlich", "moeglicherweise", "möglicherweise",
    "koennte", "könnte", "duerfte", "dürfte",
)
_SPECULATIVE_PATTERNS = (
    r"bereitet\s+.*\bvor\b",
    r"steht\s+vor\s+dem\b",
    r"vor\s+dem\s+(?:r[uü]cktritt|aus)\b",
    r"soll\s+.*zur[uü]ck",
    r"plant\s+.*r[uü]cktritt",
    r"droht\s+der\s+r[uü]cktritt",
)


_SOFT_CONTENT_PATTERNS = (
    # Raetsel / Quiz / Mitmach-Formate
    r"\berkennen sie\b",
    r"\braten sie\b",
    r"\bkennen sie\b",
    r"\bsch[aä]tzen sie\b",
    r"\btesten sie\b",
    r"\bdas gesuchte\b",
    r"\b(bilder)?r[aä]tsel\b",
    r"\bquiz\b",
    r"\bwer ist das\b",
    # Gewinnspiel / Promo / Lotto
    r"\bgewinnspiel\b",
    r"\blotto\b",
    r"\bjackpot\b",
    r"\bverlosung\b",
    r"\bzu gewinnen\b",
    r"\bgewinnen sie\b",
    r"\bwer holt sich\b",
    # Service / Ratgeber / Lifestyle-Listicle
    r"\blohnt sich\b",
    r"\bkalorien spar",
    r"\bdiese (drinks|tricks|hausmittel|lebensmittel|fehler|tipps|rechte)\b",
    r"machen es m[oö]glich",
    r"\bso (gelingt|sparen|klappt|funktioniert|sch[uü]tzen|erkennen|vermeiden|reagieren) sie\b",
    r"\bdarum sollten sie\b",
    r"\bdas m[uü]ssen sie (wissen|beachten)\b",
    r"\bdie besten \w+ tipps\b",
    r"\babnehmen\b",
    # Verbraucher-/Finanz-Ratgeber (Frage-/Hinweis-Formate)
    r"\bkommt man\b",
    r"\bbietet \w+ nur\b",
    r"\bscheinsicherheit\b",
    r"\bk[aä]uferschutz\b",
    r"\bfestgeld",
    r"\bwertsachen verstecken\b",
    r"\bgeld verstecken\b",
    r"\bwo \w+ suchen\b",
    r"\bworauf (sie achten|es ankommt)\b",
    r"\bdas hilft (gegen|bei)\b",
)
_SOFT_CONTENT_RE = re.compile("|".join(_SOFT_CONTENT_PATTERNS), re.IGNORECASE)


def _is_soft_service_or_quiz(title: str) -> bool:
    """True for Raetsel/Quiz/Service/Ratgeber-Formate - kein CvD-Push-Stoff."""
    return bool(_SOFT_CONTENT_RE.search(str(title or "")))


_URGENT_PUBLIC_SERVICE_RE = re.compile(
    r"\b(?:bahn|deutsche bahn|streik|blackout|totalausfall|funkst[oö]rung|"
    r"st[oö]rung|ausfall|sperrung|warnung|r[uü]ckruf|geld zur[uü]ck|"
    r"entsch[aä]digung|reisende|verkehr|flughafen|hitze|unwetter)\b",
    re.IGNORECASE,
)


def _is_urgent_public_service_title(title: str) -> bool:
    """Allow acute utility pieces when they explain a current disruption."""
    return bool(_URGENT_PUBLIC_SERVICE_RE.search(str(title or "")))


# Positives Nachrichten-Ereignis-Signal: etwas ist passiert / wird gemeldet.
# Bewusst breit gehalten, damit echte News nicht faelschlich blockiert werden.
_NEWS_EVENT_TERMS = (
    # Tod / Gewalt / Unglueck
    "tot", "tote", "toter", "getötet", "getoetet", "stirbt", "gestorben", "leiche",
    "opfer", "verletzt", "verletzte", "schwerverletzt", "attacke", "angriff",
    "anschlag", "terror", "schuss", "schüsse", "schuesse", "messer", "explosion",
    "explodiert", "brand", "brennt", "feuer", "unfall", "crash", "absturz",
    "ertrunken", "vermisst", "entführt", "entfuehrt", "überfall", "ueberfall",
    "amok", "drama", "tragödie", "tragoedie", "katastrophe",
    # Kriminalitaet / Justiz
    "festnahme", "festgenommen", "verhaftet", "razzia", "urteil", "verurteilt",
    "gericht", "prozess", "anklage", "ermittl", "gesteht", "gestanden", "betrug",
    # Politik / Entscheidungen
    "beschließt", "beschliesst", "beschlossen", "stimmt", "abstimmung", "wahl",
    "gewählt", "gewaehlt", "rücktritt", "ruecktritt", "zurückgetreten",
    "zurueckgetreten", "tritt zurück", "tritt zurueck", "entlassen", "ernannt",
    "einigt", "einigen", "einigung", "gesetz", "verbietet", "verbot", "verhängt",
    "verhaengt", "sanktion", "kündigt an", "kuendigt an", "erklärt", "erklaert",
    "krieg", "waffenruhe", "feuerpause", "eskaliert", "droht", "warnt", "warnung",
    "regierungsbefragung",
    # Wirtschaft
    "insolvenz", "pleite", "entlassungen", "streik", "rekord", "rückruf",
    "rueckruf", "kollaps", "erhöht", "erhoeht", "senkt", "blackout",
    "totalausfall", "funkstörung", "funkstoerung",
    # Sport-Ereignisse
    "gewinnt", "gewonnen", "verliert", "verloren", "siegt", "niederlage",
    "wechselt", "wechsel", "verpflichtet", "gefeuert", "transfer", "ausfall",
    "ausgeschieden", "meister", "rekord",
    # Wetter / Natur
    "unwetter", "sturm", "hochwasser", "überflutung", "ueberflutung", "erdbeben",
    "hitzewarnung", "evakuiert", "evakuierung",
    # Meldung / Ankuendigung allgemein
    "meldet", "bestätigt", "bestaetigt", "ankündigung", "ankuendigung",
    "gibt bekannt", "stoppt", "räumt ein", "raeumt ein", "enthüllt", "enthuellt",
    "ist tot", "gestürzt", "gestuerzt",
)
_NEWS_EVENT_RE = re.compile(
    r"(?:\b(?:" + "|".join(re.escape(t) for t in _NEWS_EVENT_TERMS) + r")\b)"
    r"|\b\d+\s+(?:tote|tote[nr]?|verletzte|opfer|festnahmen|festgenommen)\b",
    re.IGNORECASE,
)


def _has_news_event(title: str) -> bool:
    """True if the headline carries a concrete news-event signal (something happened)."""
    text = str(title or "")
    if "++" in text or re.search(r"(?i)\beil(meldung)?\b", text):
        return True
    return bool(_NEWS_EVENT_RE.search(text))


def _is_speculative(title: str) -> bool:
    """Heuristic: anticipatory/uncertain framing that reality may have overtaken.

    No external ground truth - this only flags speculative wording so a stale
    "soll wohl zuruecktreten" is not pushed after the fact.
    """
    lowered = " " + re.sub(r"\s+", " ", str(title or "")).strip().lower() + " "
    if any(f" {marker} " in lowered for marker in _SPECULATIVE_MARKERS):
        return True
    return any(re.search(pattern, lowered) for pattern in _SPECULATIVE_PATTERNS)


# Lagen, deren Vollzug sich gegen die Konkurrenz-Feeds pruefen laesst, plus die
# Cues, die einen bereits vollzogenen Vollzug signalisieren (DE + EN).
_RESIGNATION_TOPIC_CUES = ("rücktritt", "ruecktritt", "zurücktreten", "zurueck", "abdank", "amt nieder")
_RESIGNATION_DONE_CUES = (
    "zurückgetreten", "zurueckgetreten", "tritt zurück", "tritt zurueck",
    "ist zurückgetreten", "rücktritt erklärt", "ruecktritt erklaert", "tritt ab",
    "ist abgetreten", "nachfolger", "nachfolge steht",
    "resigns", "resigned", "steps down", "stepped down", "quits", "has quit",
)


def _recent_feed_headlines() -> list[str]:
    """Cached competitor/international headlines (no live fetch in the alert cycle)."""
    headlines: list[str] = []
    try:
        from app.research.worker import get_cached_feeds
    except Exception:
        return headlines
    for feed_type in ("competitors", "international"):
        try:
            data = get_cached_feeds(feed_type)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for items in data.values():
            if not isinstance(items, list):
                continue
            for item in items:
                text = str((item or {}).get("t") or "").strip()
                if text:
                    headlines.append(text)
    return headlines


def _overtaken_by_feed(title: str, config: TeamsAlertConfig) -> str:
    """Return a reason if a fresher feed source already reports the speculated
    event as done (e.g. candidate 'bereitet wohl Ruecktritt vor' while BBC/Welt
    already report 'Starmer resigns'). Currently covers resignation-type events.
    """
    if not config.feed_overtaken_enabled:
        return ""
    title_l = title.lower()
    if not (_is_speculative(title) and any(cue in title_l for cue in _RESIGNATION_TOPIC_CUES)):
        return ""
    # Markante Entitaets-Tokens (Eigennamen/lange Begriffe) aus dem Kandidaten.
    entity_tokens = {token for token in _tokens(title) if len(token) >= 5}
    if not entity_tokens:
        return ""
    for headline in _recent_feed_headlines():
        hl = headline.lower()
        if not any(cue in hl for cue in _RESIGNATION_DONE_CUES):
            continue
        if _is_speculative(headline):
            continue
        shared = entity_tokens & _tokens(headline)
        if shared:
            return (
                "Bereits als vollzogen gemeldet (Feed-Abgleich: "
                f"{', '.join(sorted(shared))}): \"{_compact_text(headline, 80)}\""
            )
    return ""


def _llm_push_title(title: str, section: str, url: str, config: TeamsAlertConfig) -> str:
    """KI-generierter Push-Titel (push_title_agent), nur wenn der LLM verfuegbar ist.

    Gibt "" zurueck, wenn LLM deaktiviert/ohne Key/Budget, der LLM nicht wirklich
    lief, oder das Ergebnis generisch/leer ist - dann greift der Fallback.
    """
    if not config.llm_title_enabled:
        return ""
    try:
        from push_title_agent import _llm_unavailable_reason, generate_push_title

        if _llm_unavailable_reason():
            return ""
        from app.push_titles import infer_content_type

        result = generate_push_title(
            article_title=title,
            category=section or "news",
            article_type=infer_content_type(url, title),
            force_llm=True,
        )
        if not (result.get("meta") or {}).get("llm_call_started"):
            return ""
        llm_title = str(result.get("title") or "").strip()
        if llm_title and not _is_generic_push_title(llm_title):
            return _compact_text(llm_title, 100)
    except Exception as exc:
        log.warning("[TeamsAlert] LLM push title unavailable: %s", exc)
    return ""


def _teams_push_title_recommendation(
    candidate: dict[str, Any],
    title: str,
    section: str,
    url: str,
    config: TeamsAlertConfig | None = None,
) -> tuple[str, str]:
    """Return (push_title, source) where source is 'llm' | 'editorial' | 'headline'."""
    config = config or TeamsAlertConfig()
    breaking = _is_breaking(candidate)

    # 1) KI-generierter Titel hat Vorrang (wenn LLM verfuegbar).
    llm_title = _llm_push_title(title, section, url, config) if config.llm_title_enabled else ""
    if llm_title:
        clean = _sanitize_push_title(llm_title, breaking=breaking)
        if clean and not _is_generic_push_title(clean):
            return _compact_text(clean, 100), "llm"

    explicit_candidates: list[str] = []
    for key in (
        "alternativePushTitle",
        "pushTitleAlternative",
        "recommendedPushTitle",
        "recommendedPushText",
        "recommendedText",
    ):
        value = str(candidate.get(key) or "").strip()
        if value:
            explicit_candidates.append(value)
    for key in ("alternativeTitles", "pushTitleAlternatives"):
        values = candidate.get(key)
        if isinstance(values, list):
            explicit_candidates.extend(str(item or "").strip() for item in values if str(item or "").strip())

    # Eine nicht-generische Empfehlung, die sich von der Schlagzeile unterscheidet,
    # gewinnt. Generische Floskeln werden grundsaetzlich uebersprungen.
    for value in _dedupe(explicit_candidates):
        clean = _sanitize_push_title(value, breaking=breaking)
        if clean and not _same_editorial_text(clean, title) and not _is_generic_push_title(clean):
            return _compact_text(clean, 100), "editorial"

    try:
        from app.push_titles import build_push_title_suggestions

        result = build_push_title_suggestions(title, category=section, url=url)
        generated = [
            str(result.get("title") or "").strip(),
            *[str(item or "").strip() for item in result.get("alternativeTitles", [])],
            str((result.get("alternative") or {}).get("titel") or "").strip(),
        ]
        for value in _dedupe([item for item in generated if item]):
            clean = _sanitize_push_title(value, breaking=breaking)
            if clean and not _same_editorial_text(clean, title) and not _is_generic_push_title(clean):
                return _compact_text(clean, 100), "editorial"
    except Exception as exc:
        log.warning("[TeamsAlert] could not build alternative push title: %s", exc)

    # Lieber die echte Schlagzeile als eine generische Floskel.
    for value in explicit_candidates:
        clean = _sanitize_push_title(value, breaking=breaking)
        if clean and not _is_generic_push_title(clean):
            return _compact_text(clean, 100), "editorial"
    fallback = _sanitize_push_title(title or (explicit_candidates[0] if explicit_candidates else ""), breaking=breaking)
    return _compact_text(fallback, 100), "headline"


def _score_reason(candidate: dict[str, Any]) -> str:
    return str(candidate.get("scoreReason") or "").strip()


def _editorial_list(candidate: dict[str, Any], key: str) -> list[str]:
    raw = candidate.get(key)
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return _dedupe(result)


def _score_breakdown_lines(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("scoreBreakdown")
    if not isinstance(raw, dict):
        return []
    labels = [
        ("freshness", "Freshness"),
        ("bildReiz", "BILD-Reiz"),
        ("headlineStrength", "Headline"),
        ("openingRatePotential", "OR-Potenzial"),
        ("mixBalance", "Mix"),
        ("politicsContext", "Politik-Kontext"),
        ("videoFit", "Video-Fit"),
        ("editorialFeedback", "Redaktionsfeedback"),
        ("riskAndFatigue", "Risiko/Fatigue"),
    ]
    lines: list[str] = []
    for key, label in labels:
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            continue
        lines.append(f"{label}: {value:.1f}/100")
    return lines


def _format_minutes(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"vor {float(value):.0f} Minuten"
    return "kein letzter Push bekannt"


def _compact_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max(1, max_len - 1)].rstrip() + "…"


def _build_power_automate_message_html(
    *,
    title: str,
    url: str,
    section: str,
    score: float,
    predicted_or: float | None,
    forecast: dict[str, Any] | None,
    recommended_text: str,
    now_ts: int,
    minutes_since_last_push: Any,
    score_threshold: float,
    alert_score: float,
    alert_threshold: float,
    editorial_score: float,
    why_now: list[str],
    subject: str,
    push_text_matches_title: bool = False,
    score_reason: str = "",
    performance_drivers: list[str] | None = None,
    risks: list[str] | None = None,
    score_breakdown_lines: list[str] | None = None,
) -> str:
    article_html = (
        f'<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
        if url
        else html.escape(title)
    )
    why_now_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in why_now)
    lead_html = (
        "<p><strong>Alternativer Push-Titel:</strong><br>"
        f"{html.escape(recommended_text)}</p>"
        "<p><strong>Artikel:</strong><br>"
        f"{article_html}</p>"
    )
    return (
        f"<h2>{html.escape(subject)}</h2>"
        f"{lead_html}"
        "<p>"
        f"<strong>Ressort:</strong> {html.escape(section)}<br>"
        f"<strong>Push-Score:</strong> {html.escape(_format_number(score))} "
        f"(Mindestwert {html.escape(_format_number(score_threshold, 0))})<br>"
        f"<strong>Prognose:</strong> {html.escape(_format_forecast(forecast))}<br>"
        f"<strong>Letzter Push:</strong> "
        f"{html.escape(_format_minutes(minutes_since_last_push))}<br>"
        f"<strong>Stand:</strong> {html.escape(_format_time(now_ts))} Uhr"
        "</p>"
        "<p><strong>Warum jetzt?</strong></p>"
        f"<ul>{why_now_html}</ul>"
        "<p><strong>Empfehlung:</strong> Jetzt pushen.</p>"
    )


def _normalize_url(url: str) -> str:
    return url.strip().split("?", 1)[0].rstrip("/").lower()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _tokens(title: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(title or "")} - _STOP_WORDS


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
