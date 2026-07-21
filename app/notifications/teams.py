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
    PUSH_BALANCER_SCORE_API_ENABLED,
    PUSH_TEAMS_ACTIVE_HOURS_END,
    PUSH_TEAMS_ACTIVE_HOURS_START,
    PUSH_TEAMS_AGENT_REVIEW_ENABLED,
    PUSH_TEAMS_AGENT_REVIEW_MAX_LATENCY_MS,
    PUSH_TEAMS_AGENT_REVIEW_MIN_EVIDENCE_APPROVALS,
    PUSH_TEAMS_AGENT_REVIEW_MIN_CONSENSUS_SCORE,
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
    PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED,
    PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME,
    PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE,
    PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE,
    PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP,
    PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE,
    PUSH_TEAMS_EDITORIAL_GATE_ENABLED,
    PUSH_TEAMS_EDITORIAL_TOP_LIMIT,
    PUSH_TEAMS_EARLY_EXCEPTIONAL_ALERT_SCORE,
    PUSH_TEAMS_EARLY_EXCEPTIONAL_EDITORIAL_SCORE,
    PUSH_TEAMS_EARLY_EXCEPTIONAL_SCORE,
    PUSH_TEAMS_EVENT_GATE_ENABLED,
    PUSH_TEAMS_EXCLUDED_SECTIONS,
    PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES,
    PUSH_TEAMS_HIGH_SCORE_ALWAYS_THRESHOLD,
    PUSH_TEAMS_INDEPENDENT_PACING_ENABLED,
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
    PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY,
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
    PUSH_TEAMS_PEAK_SLOT_MIN_OR,
    PUSH_TEAMS_POST_SEND_DECAY_MINUTES,
    PUSH_TEAMS_POST_SEND_PEAK_SCORE,
    PUSH_TEAMS_POST_SEND_THRESHOLD_ENABLED,
    PUSH_TEAMS_TARGET_PUSHES_PER_DAY,
    PUSH_TEAMS_TOPIC_DEDUP_HOURS,
    PUSH_TEAMS_TOPIC_DEDUP_SIMILARITY,
    PUSH_TEAMS_SLOT_DEADLINE_MINUTE,
    PUSH_TEAMS_SLOT_GATE_ENABLED,
    PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED,
    PUSH_TEAMS_VISIT_SELECTION_WEIGHT,
    PUSH_TEAMS_WEBHOOK_URL,
)
from app.database import (
    push_db_load_all,
    teams_daily_schedule_record,
    teams_daily_schedule_try_claim,
    teams_alert_last_sent_ts,
    teams_alert_list_recent,
    teams_alert_load_for_keys,
    teams_alert_record,
    teams_alert_sent_count_since,
    teams_alert_try_claim_send,
    teams_recommendation_record,
)
from app.notifications.teams_review import add_agent_review_veto, run_agent_review_network
from app.scoring.editorial import (
    assess_germany_relevance,
    is_german_public_figure_parenthood_story,
)

log = logging.getLogger("push-balancer")

_TEAMS_RECOMMENDATION_POLICY_VERSION = "internal-score-adaptive-threshold-v6"
_MANDATORY_QUIET_HOURS_START_MINUTE = 0
_MANDATORY_QUIET_HOURS_END_MINUTE = 5 * 60 + 30
_HARD_NORMAL_PUSH_SCORE_FLOOR = 75.0
_HARD_BREAKING_PUSH_SCORE_FLOOR = 72.0
_PUSH_SCORE_SELECTION_BAND = 3.0

_HARD_TEAMS_BLOCKER_MARKERS = (
    "alerts deaktiviert",
    "kein gueltiger interner push-balancer-score",
    "ruhezeit aktiv",
    "ohne headline",
    "ohne artikel-link",
    "kein artikel:",
    "aktualitaet:",
    "zeitstempel",
    "veroeffentlichungs- oder aktualisierungszeit fehlt",
    "artikel ist zu alt",
    "artikel nicht frisch genug",
    "ressort ",
    "letzter push-zeitpunkt nicht verfuegbar",
    "pause seit letztem push zu kurz",
    "push-dichte in den letzten",
    "bereits per teams gemeldet",
    "thema bereits per teams gemeldet",
    "re-alert-cooldown",
    "teams-cooldown aktiv",
    "tageslimit fuer teams-hinweise",
    "staerkerer kandidat vorhanden",
    "wahrscheinlich ueberholt",
    "bereits als vollzogen gemeldet",
    "bereits als teams-kandidat versucht",
    "teams-hinweis wird bereits versendet",
    "sport ohne ",
    "live-ticker ohne neue pushwuerdige lage",
    "live-ticker ohne neue pushwürdige lage",
    "kein konkretes nachrichten-ereignis",
    "service-/raetsel-/ratgeber-format",
    "kurios-/click-reiz",
    "enger kurios-/click-reiz",
    "termin-/prozesslage ohne neue entwicklung",
    "erklär-/debattenstück ohne neue aktuelle lage",
    "erklaer-/debattenstueck ohne neue aktuelle lage",
    "morgenfit:",
    "feld unsicher",
    "deutschland-relevanz",
    "tagesplan:",
    "tagesplan im soll",
)

_RECENT_SEND_LOCK = threading.Lock()
_RECENT_SEND_MEMORY: dict[str, dict[str, Any]] = {}

_TOKEN_RE = re.compile(r"[a-z0-9äöüßaeoeue]{4,}", re.IGNORECASE)
_STOP_WORDS = {
    "aber",
    "alle",
    "auch",
    "auf",
    "aus",
    "bei",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "dieser",
    "doch",
    "eine",
    "einem",
    "einen",
    "einer",
    "fuer",
    "für",
    "hat",
    "haben",
    "hier",
    "ist",
    "jetzt",
    "kann",
    "mit",
    "nach",
    "nicht",
    "noch",
    "oder",
    "sich",
    "sind",
    "ueber",
    "über",
    "und",
    "von",
    "vor",
    "war",
    "was",
    "weil",
    "wenn",
    "wie",
    "wird",
    "wurde",
    "zum",
    "zur",
}


@dataclass(frozen=True)
class TeamsAlertConfig:
    enabled: bool = PUSH_TEAMS_ALERTS_ENABLED
    webhook_url: str = PUSH_TEAMS_WEBHOOK_URL
    require_internal_score_api: bool = PUSH_BALANCER_SCORE_API_ENABLED
    min_score: float = PUSH_TEAMS_MIN_SCORE
    min_alert_score: float = PUSH_TEAMS_MIN_ALERT_SCORE
    score_only_mode: bool = PUSH_TEAMS_SCORE_ONLY_MODE
    dashboard_top_limit: int = PUSH_TEAMS_DASHBOARD_TOP_LIMIT
    candidate_limit: int = PUSH_TEAMS_CANDIDATE_LIMIT
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
    post_send_threshold_enabled: bool = PUSH_TEAMS_POST_SEND_THRESHOLD_ENABLED
    post_send_peak_score: float = PUSH_TEAMS_POST_SEND_PEAK_SCORE
    post_send_decay_minutes: int = PUSH_TEAMS_POST_SEND_DECAY_MINUTES
    high_score_always_threshold: float = PUSH_TEAMS_HIGH_SCORE_ALWAYS_THRESHOLD
    independent_pacing_enabled: bool = PUSH_TEAMS_INDEPENDENT_PACING_ENABLED
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
    slot_gate_enabled: bool = PUSH_TEAMS_SLOT_GATE_ENABLED
    slot_deadline_minute: int = PUSH_TEAMS_SLOT_DEADLINE_MINUTE
    peak_slot_min_or: float = PUSH_TEAMS_PEAK_SLOT_MIN_OR
    early_exceptional_score: float = PUSH_TEAMS_EARLY_EXCEPTIONAL_SCORE
    early_exceptional_alert_score: float = PUSH_TEAMS_EARLY_EXCEPTIONAL_ALERT_SCORE
    early_exceptional_editorial_score: float = PUSH_TEAMS_EARLY_EXCEPTIONAL_EDITORIAL_SCORE
    deadline_fallback_min_score: float = PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE
    deadline_fallback_min_alert_score: float = PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE
    deadline_fallback_min_editorial_score: float = PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE
    daily_schedule_send_enabled: bool = PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED
    daily_schedule_send_time: str = PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME
    agent_review_enabled: bool = PUSH_TEAMS_AGENT_REVIEW_ENABLED
    agent_review_min_evidence_approvals: int = PUSH_TEAMS_AGENT_REVIEW_MIN_EVIDENCE_APPROVALS
    agent_review_min_consensus_score: float = PUSH_TEAMS_AGENT_REVIEW_MIN_CONSENSUS_SCORE
    min_recommendation_quality: float = PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY
    agent_review_max_latency_ms: int = PUSH_TEAMS_AGENT_REVIEW_MAX_LATENCY_MS
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
    if config.independent_pacing_enabled:
        return max(configured, 30)
    return max(configured, int(config.min_minutes_since_last_push or 0), 30)


def _post_send_score_threshold(
    base_threshold: float,
    *,
    minutes_since_last_teams_alert: float | None,
    breaking: bool,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Raise the raw score floor after a Teams send, then decay it predictably."""
    always_threshold = _clamp(float(config.high_score_always_threshold or 80.0), 0.0, 100.0)
    base = _clamp(min(float(base_threshold or 0.0), always_threshold), 0.0, 100.0)
    configured_peak = _clamp(float(config.post_send_peak_score or 80.0), 0.0, 100.0)
    peak = max(base, min(configured_peak, always_threshold))
    cooldown = max(0, _effective_global_cooldown_minutes(config))
    decay_end = max(cooldown, int(config.post_send_decay_minutes or 0))
    review = {
        "enabled": bool(config.post_send_threshold_enabled),
        "active": False,
        "baseThreshold": round(base, 2),
        "currentThreshold": round(base, 2),
        "peakThreshold": round(peak, 2),
        "cooldownMinutes": cooldown,
        "decayEndMinutes": decay_end,
        "minutesSinceLastTeamsAlert": minutes_since_last_teams_alert,
        "phase": "baseline",
        "reason": "Keine vorherige Teams-Empfehlung; Basis-Schwelle gilt.",
    }
    if not config.post_send_threshold_enabled:
        review["phase"] = "disabled"
        review["reason"] = "Adaptive Schwelle ist deaktiviert; Basis-Schwelle gilt."
        return review
    if breaking:
        review["phase"] = "breaking"
        review["reason"] = "Verifiziertes Breaking nutzt seine eigene Score-Schwelle."
        return review
    if minutes_since_last_teams_alert is None:
        return review

    elapsed = max(0.0, float(minutes_since_last_teams_alert))
    if elapsed <= cooldown:
        current = peak
        phase = "peak"
    elif elapsed < decay_end and decay_end > cooldown:
        progress = (elapsed - cooldown) / float(decay_end - cooldown)
        current = peak - (peak - base) * _clamp(progress, 0.0, 1.0)
        phase = "decay"
    else:
        current = base
        phase = "baseline"

    review["active"] = current > base
    review["currentThreshold"] = round(current, 2)
    review["phase"] = phase
    if phase == "peak":
        review["reason"] = (
            f"Nach dem letzten Teams-Hinweis gilt die erhoehte Push-Score-Schwelle "
            f"{current:.1f}."
        )
    elif phase == "decay":
        review["reason"] = (
            f"Die Schwelle faellt kontrolliert von {peak:.1f} auf {base:.1f}; "
            f"aktuell sind {current:.1f} erforderlich."
        )
    else:
        review["reason"] = (
            f"Das Schutzfenster ist abgelaufen; die Basis-Schwelle {base:.1f} gilt wieder."
        )
    return review


def _is_hard_teams_blocker(blocker: str) -> bool:
    normalized = str(blocker or "").casefold()
    return any(marker in normalized for marker in _HARD_TEAMS_BLOCKER_MARKERS)


def _memory_send_blocker_locked(
    *,
    article_key: str,
    title: str,
    now_ts: int,
    config: TeamsAlertConfig,
    bypass_global_cooldown: bool = False,
) -> dict[str, Any]:
    cooldown_seconds = (
        0 if bypass_global_cooldown else _effective_global_cooldown_minutes(config) * 60
    )
    article_seconds = max(int(config.alert_cooldown_minutes or 0) * 60, cooldown_seconds)
    topic_seconds = max(int(float(config.topic_dedup_hours or 0.0) * 3600), article_seconds)
    keep_seconds = max(topic_seconds, 3600)
    title_tokens = _tokens(title)

    stale_before = now_ts - keep_seconds
    for key, entry in list(_RECENT_SEND_MEMORY.items()):
        if entry.get("status") == "failed" or _safe_int(entry.get("ts")) < stale_before:
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
        if (
            title_tokens
            and _same_topic(title_tokens, other_tokens, threshold)
            and age < topic_seconds
        ):
            return {
                "blocked": True,
                "reason": "memory_topic_duplicate",
                "ageSeconds": age,
                "otherKey": key,
                "otherTitle": str(entry.get("title") or ""),
            }
    return {"blocked": False}


def _memory_send_blocker(
    *,
    article_key: str,
    title: str,
    now_ts: int,
    config: TeamsAlertConfig,
    bypass_global_cooldown: bool = False,
) -> dict[str, Any]:
    """Check process-local duplicate state without reserving a send."""
    with _RECENT_SEND_LOCK:
        return _memory_send_blocker_locked(
            article_key=article_key,
            title=title,
            now_ts=now_ts,
            config=config,
            bypass_global_cooldown=bypass_global_cooldown,
        )


def _memory_send_blocker_or_reserve(
    *,
    article_key: str,
    title: str,
    now_ts: int,
    config: TeamsAlertConfig,
    bypass_global_cooldown: bool = False,
) -> dict[str, Any]:
    """Atomically reserve a process-local send after duplicate checks."""
    with _RECENT_SEND_LOCK:
        blocker = _memory_send_blocker_locked(
            article_key=article_key,
            title=title,
            now_ts=now_ts,
            config=config,
            bypass_global_cooldown=bypass_global_cooldown,
        )
        if blocker.get("blocked"):
            return blocker
        _RECENT_SEND_MEMORY[article_key] = {
            "ts": now_ts,
            "title": title,
            "tokens": _tokens(title),
            "status": "reserved",
        }
    return {"blocked": False, "reserved": True}


def _memory_record_send_result(article_key: str, *, ok: bool, now_ts: int) -> None:
    with _RECENT_SEND_LOCK:
        entry = _RECENT_SEND_MEMORY.get(article_key)
        if entry is None:
            return
        if ok:
            entry["status"] = "sent"
            entry["ts"] = now_ts
        else:
            _RECENT_SEND_MEMORY.pop(article_key, None)


def _memory_release_reservation(article_key: str) -> None:
    with _RECENT_SEND_LOCK:
        entry = _RECENT_SEND_MEMORY.get(article_key)
        if entry is not None and entry.get("status") == "reserved":
            _RECENT_SEND_MEMORY.pop(article_key, None)


def _memory_eligible_candidates(
    candidates: list[dict[str, Any]],
    *,
    now_ts: int,
    config: TeamsAlertConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove process-local duplicates before ranking the candidate field."""
    eligible: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        blocker = _memory_send_blocker(
            article_key=key,
            title=_title(candidate),
            now_ts=now_ts,
            config=config,
            bypass_global_cooldown=bool(_is_breaking(candidate) and config.breaking_override),
        )
        if not blocker.get("blocked"):
            eligible.append(candidate)
            continue
        reason = str(blocker.get("reason") or "memory_guard")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        article_ref = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        log.info(
            "[TeamsAlert] candidate skipped before ranking article_ref=%s reason=%s",
            article_ref,
            reason,
        )
    return eligible, {
        "skippedCandidates": sum(reason_counts.values()),
        "reasons": dict(sorted(reason_counts.items())),
    }


def build_teams_alert_context(
    candidates: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
    history_authoritative: bool | None = None,
    alert_state: dict[str, dict[str, Any]] | None = None,
    last_teams_alert_ts: int | None = None,
    teams_alerts_today: int | None = None,
    recent_alerts: list[dict[str, Any]] | None = None,
    now_ts: int | None = None,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    history_was_supplied = history is not None
    context_available = {
        "history": history is not None,
        "alertState": alert_state is not None,
        "globalCooldown": last_teams_alert_ts is not None,
        "dailyAlertCount": teams_alerts_today is not None,
        "recentTeamsAlerts": recent_alerts is not None,
    }
    if history is None:
        try:
            # Die echte Push-Historie ist reiner Vergleichskontext. Ob ein Artikel
            # bereits per Teams empfohlen wurde, entscheidet nur die Teams-Historie.
            history = push_db_load_all(max_days=90, max_rows=3000)
            context_available["history"] = True
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load push history: %s", exc)
            history = []
    if alert_state is None:
        try:
            alert_state = teams_alert_load_for_keys([candidate_key(item) for item in candidates])
            context_available["alertState"] = True
        except Exception as exc:
            log.warning("[TeamsAlert] Could not load alert state: %s", exc)
            alert_state = {}
    if last_teams_alert_ts is None:
        try:
            last_teams_alert_ts = teams_alert_last_sent_ts()
            context_available["globalCooldown"] = True
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
        1 for item in history if _safe_int(item.get("ts_num", item.get("ts", 0))) >= now - 6 * 3600
    )

    day_start = _local_day_start_ts(now)
    pushes_today = sum(
        1 for item in history if _safe_int(item.get("ts_num", item.get("ts", 0))) >= day_start
    )

    alerts_today = teams_alerts_today
    if alerts_today is None:
        try:
            alerts_today = teams_alert_sent_count_since(day_start)
            context_available["dailyAlertCount"] = True
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
                    {
                        "key": str(row.get("article_key") or ""),
                        "title": str(row.get("article_title") or ""),
                    }
                )
            context_available["recentTeamsAlerts"] = True
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
        "historyReviewIndex": _push_history_review_index(history, now, config),
        "historyAuthoritative": (
            bool(history_authoritative)
            if history_authoritative is not None
            else history_was_supplied
        ),
        "contextAvailable": context_available,
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
    score_source = str(candidate.get("scoreSource") or "server_editorial_fallback")
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
    publication_review = _publication_time_review(candidate, now_ts, config)
    freshness_hours = (
        float(publication_review["ageHours"])
        if publication_review.get("status") == "valid"
        else None
    )
    min_score, min_or, min_pause = _effective_thresholds(config, breaking)
    last_push_ts = _safe_int(context.get("lastPushTs"))
    last_teams_alert_ts = _safe_int(context.get("lastTeamsAlertTs"))
    minutes_since_last_push = (
        round((now_ts - last_push_ts) / 60, 1)
        if last_push_ts > 0 and now_ts >= last_push_ts
        else None
    )
    minutes_since_last_teams_alert = (
        round((now_ts - last_teams_alert_ts) / 60, 1)
        if last_teams_alert_ts > 0 and now_ts >= last_teams_alert_ts
        else None
    )
    alert_state = (context.get("alertState") or {}).get(key)
    dashboard_rank = _safe_int(context.get("dashboardRank"))
    dashboard_top_limit = max(1, int(config.dashboard_top_limit or PUSH_TEAMS_CANDIDATE_LIMIT))
    dashboard_rank_blocker = ""

    positive: list[str] = []
    blockers: list[str] = []
    status = "skip"
    candidate_score_reason = _score_reason(candidate)
    candidate_drivers = _editorial_list(candidate, "performanceDrivers")
    candidate_risks = _editorial_list(candidate, "risks")
    candidate_breakdown = (
        candidate.get("scoreBreakdown") if isinstance(candidate.get("scoreBreakdown"), dict) else {}
    )

    if not config.enabled:
        blockers.append("Teams Alerts deaktiviert")

    if config.require_internal_score_api:
        if score_source == "internal_score_api":
            positive.append("Kanonischer Score: frische interne Push-Balancer-API")
        else:
            blockers.append(
                "Kein gueltiger interner Push-Balancer-Score; lokaler Fallback ist gesperrt"
            )

    quiet_reason = _quiet_hours_reason(now_ts, config)
    if quiet_reason:
        blockers.append(quiet_reason)
        status = "observe"

    if not title:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Headline")
    if not url:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Artikel-Link")
    if publication_review.get("status") != "valid":
        publication_reason = str(
            publication_review.get("reason")
            or "Veroeffentlichungszeit ist nicht belastbar"
        )
        blockers.append(f"Aktualitaet: {publication_reason}")
    non_article_reason = _daily_plan_non_article_reason(candidate)
    if non_article_reason:
        blockers.append(non_article_reason)

    if dashboard_rank > 0:
        if dashboard_rank <= dashboard_top_limit:
            positive.append(f"Top-Kandidat im Push Balancer: Rang {dashboard_rank}")
        else:
            dashboard_rank_blocker = (
                f"Nicht im oberen Push-Balancer-Feld: Rang {dashboard_rank} > {dashboard_top_limit}"
            )

    if score_source == "captured_push_balancer":
        positive.append("Hauptnote: frisches Rating aus der Push-Balancer-Kandidatenansicht")

    section_key = _section_key(section)
    excluded = {item.lower() for item in config.excluded_sections if item.strip()}
    if section.lower() in excluded or section_key in excluded:
        blockers.append(f"Ressort {section} ist fuer Teams Alerts ausgeschlossen")
    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if allowed and section.lower() not in allowed and section_key not in allowed:
        blockers.append(f"Ressort {section} nicht fuer Teams Alerts freigegeben")

    relevance_candidate = dict(candidate)
    for flag in ("isBreaking", "isEilmeldung", "is_eilmeldung"):
        relevance_candidate[flag] = breaking
    germany_relevance = assess_germany_relevance(relevance_candidate)
    relevance_reason = str(germany_relevance.get("reason") or "").strip()
    relevance_min_score = float(germany_relevance.get("minimumScore") or min_score)
    if germany_relevance.get("hardBlock") and not breaking:
        blockers.append(relevance_reason)
    elif (
        str(germany_relevance.get("level") or "").startswith("international")
        and not breaking
        and score < relevance_min_score
    ):
        blockers.append(f"{relevance_reason}: {score:.1f} < {relevance_min_score:.1f}")
    elif relevance_reason:
        positive.append(relevance_reason)
    if float(germany_relevance.get("adjustment") or 0.0) < 0 and relevance_reason:
        candidate_risks = [relevance_reason, *candidate_risks]

    pushes_today = context.get("pushesToday")
    pushes_today = _safe_int(pushes_today) if pushes_today is not None else None
    teams_alerts_today = _safe_int(context.get("teamsAlertsToday"))
    independent_pacing = bool(config.independent_pacing_enabled)
    pacing_count_today = teams_alerts_today if independent_pacing else pushes_today
    pacing_basis = "teams_alerts" if independent_pacing else "actual_pushes"
    push_pacing = _push_pacing_review(
        pacing_count_today,
        now_ts,
        config,
        basis=pacing_basis,
        actual_pushes_today=pushes_today,
    )
    minimum_pressure = _minimum_pressure_review(push_pacing, teams_alerts_today, now_ts, config)
    minimum_active = bool(minimum_pressure.get("active"))
    post_send_threshold = _post_send_score_threshold(
        min_score,
        minutes_since_last_teams_alert=minutes_since_last_teams_alert,
        breaking=breaking,
        config=config,
    )
    effective_min_score = float(post_send_threshold["currentThreshold"])
    if minimum_active and not breaking:
        positive.append(
            "Teams-Tagesziel aktiv: zusätzliche gute Slots sind erlaubt; "
            "die Score-Schwelle bleibt unverändert"
        )
    if post_send_threshold.get("active"):
        positive.append(f"Adaptive Score-Schwelle: {post_send_threshold['reason']}")

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
        freshness_hours=freshness_hours,
        minutes_since_last_push=(
            minutes_since_last_teams_alert if independent_pacing else minutes_since_last_push
        ),
        recent_push_count_6h=(
            0 if independent_pacing else int(context.get("recentPushCount6h") or 0)
        ),
        pushes_today=pacing_count_today,
        now_ts=now_ts,
        config=config,
    )
    alert_score = float(alert_model["score"])
    positive.append(f"Teams Alert Score {alert_score:.1f}/100")
    positive.extend(list(alert_model["reasons"])[:4])
    threshold_count_today = _safe_int(minimum_pressure.get("current"))
    effective_min_alert_score, push_budget_reason = _dynamic_alert_threshold(
        config.min_alert_score,
        threshold_count_today,
        now_ts,
        breaking,
        config,
    )
    if minimum_pressure.get("basis") == "teams_alerts" and push_budget_reason:
        push_budget_reason = (
            push_budget_reason.replace("Push-Rueckstand heute", "Teams-Rueckstand heute")
            .replace("Push-Vorsprung heute", "Teams-Vorsprung heute")
            .replace("Tagesbudget erreicht", "Teams-Tagesbudget erreicht")
            .replace("Pushes", "Hinweise")
        )
    if push_budget_reason:
        positive.append(push_budget_reason)
    if minimum_pressure["active"]:
        positive.append(str(minimum_pressure["label"]))
    if alert_score < effective_min_alert_score:
        blockers.append(
            f"Teams Alert Score zu niedrig: {alert_score:.1f} < {effective_min_alert_score:.1f}"
        )
    effective_no_forecast_min_alert_score = config.no_forecast_min_alert_score
    if predicted_or is None and alert_score < effective_no_forecast_min_alert_score:
        blockers.append(
            "Keine belastbare Prognose und Teams Alert Score nicht hoch genug: "
            f"{alert_score:.1f} < {effective_no_forecast_min_alert_score:.1f}"
        )
    forecast_is_reliable = forecast.get("source") == "article_model" and predicted_or is not None
    effective_min_or = min_or
    people_parenthood_near_miss = bool(
        predicted_or is not None
        and _german_people_parenthood_or_near_miss(
            title=title,
            section=section,
            predicted_or=predicted_or,
            min_or=effective_min_or,
            alert_score=alert_score,
            min_alert_score=effective_min_alert_score,
        )
    )
    low_forecast_blocker = (
        predicted_or is not None
        and predicted_or < effective_min_or
        and (forecast_is_reliable or not config.score_only_mode)
        and not _public_money_fraud_or_near_miss(
            title=title,
            predicted_or=predicted_or,
            min_or=effective_min_or,
            alert_score=alert_score,
            min_alert_score=effective_min_alert_score,
        )
        and not _celebrity_conflict_or_near_miss(
            title=title,
            section=section,
            predicted_or=predicted_or,
            min_or=effective_min_or,
            alert_score=alert_score,
            min_alert_score=effective_min_alert_score,
        )
        and not people_parenthood_near_miss
        and not (breaking and config.breaking_override and predicted_or >= config.breaking_min_or)
    )
    if low_forecast_blocker:
        blockers.append(f"Prognose zu niedrig: {predicted_or:.2f}% OR < {effective_min_or:.2f}%")
    elif (
        predicted_or is not None
        and predicted_or < effective_min_or
        and _public_money_fraud_or_near_miss(
            title=title,
            predicted_or=predicted_or,
            min_or=effective_min_or,
            alert_score=alert_score,
            min_alert_score=effective_min_alert_score,
        )
    ):
        positive.append(
            "Polizei-/Betrugs-Lage mit starkem Public-Need: OR knapp unter Schwelle wird akzeptiert"
        )
    elif (
        predicted_or is not None
        and predicted_or < effective_min_or
        and _celebrity_conflict_or_near_miss(
            title=title,
            section=section,
            predicted_or=predicted_or,
            min_or=effective_min_or,
            alert_score=alert_score,
            min_alert_score=effective_min_alert_score,
        )
    ):
        positive.append(
            "Promi-/Beziehungs-/Geldkonflikt im Abendfenster: OR knapp unter Schwelle wird akzeptiert"
        )
    elif people_parenthood_near_miss:
        positive.append(
            "Deutschland-People-Ereignis: OR knapp unter Schwelle wird bei starkem Push Score akzeptiert"
        )
    if config.require_valid_prediction and not forecast_is_reliable:
        blockers.append("Belastbare OR-Prognose erforderlich, aktuell nur Fallback verfuegbar")
    forecast_quality = _forecast_quality_review(candidate, forecast, alert_score, breaking, config)
    positive.extend(forecast_quality["reasons"])
    blockers.extend(forecast_quality["blockers"])

    editorial_review = _editorial_cvd_review(
        candidate,
        score=score,
        predicted_or=predicted_or,
        freshness_hours=freshness_hours,
        minutes_since_last_push=(
            minutes_since_last_teams_alert if independent_pacing else minutes_since_last_push
        ),
        dashboard_rank=dashboard_rank,
        alert_score=alert_score,
        pushes_today=pacing_count_today,
        now_ts=now_ts,
        config=config,
        minimum_pressure=minimum_pressure,
    )
    positive.extend(editorial_review["reasons"])
    blockers.extend(editorial_review["blockers"])
    morning_review = _morning_reader_value_review(candidate, now_ts)
    positive.extend(morning_review["reasons"])
    blockers.extend(morning_review["blockers"])
    slot_gate = _daily_slot_gate_review(
        candidate,
        score=score,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        predicted_or=predicted_or,
        pushes_today=pushes_today,
        teams_alerts_today=teams_alerts_today,
        breaking=breaking,
        now_ts=now_ts,
        config=config,
    )
    positive.extend(slot_gate["reasons"])
    blockers.extend(slot_gate["blockers"])
    strategy_review = _daily_strategy_review(
        candidate,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        news_value=float(editorial_review["newsValue"]),
        predicted_or=predicted_or,
        push_pacing=push_pacing,
        breaking=breaking,
        config=config,
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
        reach_stats=context.get("reachStats")
        if isinstance(context.get("reachStats"), dict)
        else {},
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
        germany_selection_adjustment=float(germany_relevance.get("selectionAdjustment") or 0.0),
        config=config,
    )
    expanded_field = _expanded_field_candidate_review(
        candidate,
        score=score,
        alert_score=alert_score,
        min_alert_score=effective_min_alert_score,
        predicted_or=predicted_or,
        dashboard_rank=dashboard_rank,
        dashboard_top_limit=dashboard_top_limit,
        config=config,
    )
    if dashboard_rank_blocker:
        if expanded_field["allowed"]:
            positive.append(str(expanded_field["reason"]))
        else:
            blockers.append(dashboard_rank_blocker)

    allow_unknown_last_push_for_minimum = independent_pacing
    if config.score_only_mode:
        positive.append("Score-Modus aktiv: Teams Alert Score entscheidet final")
    else:
        if predicted_or is None:
            blockers.append("Prognose fehlt")
        elif predicted_or >= effective_min_or:
            positive.append(
                f"Prognose {predicted_or:.2f}% OR liegt ueber Mindestwert {effective_min_or:.2f}%"
            )

    if independent_pacing:
        positive.append(
            "Eigenstaendiger Teams-Takt: Live-Pushes beeinflussen weder Tagespacing "
            "noch Empfehlungs-Cooldown"
        )
    else:
        if minutes_since_last_push is None:
            if allow_unknown_last_push_for_minimum:
                positive.append(
                    "Teams-Mindest-Pacing: letzter Push-Zeitpunkt fehlt, "
                    "Teams-Cooldown uebernimmt den Lastschutz"
                )
            else:
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
        and not independent_pacing
        and int(context.get("recentPushCount6h") or 0) > config.max_pushes_last_6h
        and not breaking
    ):
        blockers.append("Push-Dichte in den letzten 6 Stunden zu hoch")

    live_push_match_reason = _live_push_comparison_reason(
        candidate,
        context.get("history") or [],
        now_ts,
        config,
        history_index=(
            context.get("historyReviewIndex")
            if isinstance(context.get("historyReviewIndex"), dict)
            else None
        ),
    )
    live_comparison_available = bool(
        (context.get("contextAvailable") or {}).get("history")
        and context.get("historyAuthoritative")
    )
    live_push_comparison = {
        "available": live_comparison_available,
        "matched": bool(live_push_match_reason) if live_comparison_available else False,
        "matchType": (
            _live_push_match_type(live_push_match_reason) if live_comparison_available else ""
        ),
        "reason": live_push_match_reason if live_comparison_available else "",
    }
    if live_comparison_available and live_push_match_reason:
        positive.append(
            "Live-Vergleich: diese unabhängige Teams-Empfehlung entspricht einem echten Push"
        )
    elif live_comparison_available:
        positive.append("Live-Vergleich: zum Prüfzeitpunkt noch kein entsprechender echter Push")
    else:
        positive.append("Live-Vergleich: echte Push-Historie aktuell nicht belastbar verfügbar")

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
        and not (breaking and config.breaking_override)
    ):
        blockers.append(
            "Teams-Cooldown aktiv: letzter Hinweis vor "
            f"{minutes_since_last_teams_alert:.0f} < {effective_global_cooldown} Minuten"
        )
        status = "observe"
    elif (
        breaking
        and config.breaking_override
        and minutes_since_last_teams_alert is not None
        and minutes_since_last_teams_alert < effective_global_cooldown
    ):
        positive.append(
            "Breaking-News: eigener Teams-Cooldown wird für die Sofortprüfung übergangen"
        )

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
            "Staerkerer Kandidat vorhanden" + (f": {stronger_title}" if stronger_title else "")
        )

    high_score_override = _high_score_override_review(
        blockers,
        score=score,
        score_source=score_source,
        config=config,
    )
    if high_score_override["active"]:
        blockers = list(high_score_override["remainingBlockers"])
        if high_score_override["approved"]:
            positive.extend(high_score_override["reasons"])

    deadline_fallback = _deadline_fallback_review(
        candidate,
        slot_gate=slot_gate,
        blockers=blockers,
        score=score,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        config=config,
    )
    if deadline_fallback["approved"]:
        blockers = list(deadline_fallback["remainingBlockers"])
        positive.extend(deadline_fallback["reasons"])

    agent_review = _final_agent_review(
        candidate,
        context=context,
        config=config,
        min_pause=min_pause,
        min_or=effective_min_or,
        minutes_since_last_push=(None if independent_pacing else minutes_since_last_push),
        minutes_since_last_teams_alert=minutes_since_last_teams_alert,
        allow_unknown_last_push=allow_unknown_last_push_for_minimum,
        freshness_hours=freshness_hours,
        publication_review=publication_review,
        is_speculative=is_speculative,
        overtaken_reason=overtaken_reason,
        non_article_reason=non_article_reason,
        live_push_match_reason=live_push_match_reason,
        topic_duplicate_reason=topic_dup_reason,
        realert_reason=realert_reason,
        editorial_review=editorial_review,
        forecast=forecast,
        predicted_or=predicted_or,
        forecast_near_miss_accepted=bool(
            predicted_or is not None
            and predicted_or < effective_min_or
            and not low_forecast_blocker
        ),
        slot_gate=slot_gate,
        deadline_fallback=deadline_fallback,
        high_score_override=high_score_override,
        visit_potential=visit_potential,
        push_pacing=push_pacing,
        minimum_pressure=minimum_pressure,
        teams_alerts_today=teams_alerts_today,
        candidate_risks=candidate_risks,
        remaining_blockers=blockers,
    )
    if agent_review.get("enabled") and agent_review.get("approved"):
        positive.append(f"Agenten-Check: {agent_review['summary']}")
    elif agent_review.get("enabled"):
        review_blocker = str(agent_review.get("blockingReason") or "Agenten-Konsens verweigert")
        if review_blocker not in blockers:
            blockers.append(review_blocker)
        if high_score_override.get("approved"):
            high_score_override["approved"] = False
            high_score_override["hardBlockers"] = [
                *list(high_score_override.get("hardBlockers") or []),
                review_blocker,
            ]
            high_score_override["remainingBlockers"] = list(blockers)

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
        "publicationReview": publication_review,
        "scoreBreakdown": candidate_breakdown,
        "score": score,
        "scoreSource": score_source,
        "teamsAlertScore": alert_score,
        "teamsAlertScoreThreshold": round(effective_min_alert_score, 1),
        "teamsAlertScoreBaseThreshold": config.min_alert_score,
        "teamsAlertScoreBreakdown": alert_model["breakdown"],
        "postSendScoreThreshold": post_send_threshold,
        "highScoreOverride": high_score_override,
        "pushesToday": pushes_today,
        "teamsAlertsToday": teams_alerts_today,
        "pacingBasis": pacing_basis,
        "recommendationsIndependentFromLivePushes": independent_pacing,
        "livePushComparison": live_push_comparison,
        "pushBudgetReason": push_budget_reason,
        "pushPacing": push_pacing,
        "slotGate": slot_gate,
        "deadlineFallback": deadline_fallback,
        "agentReview": agent_review,
        "minimumPressure": minimum_pressure,
        "minAlertsPerDay": config.min_alerts_per_day,
        "pushBudgetTarget": config.target_pushes_per_day,
        "maxAlertsPerDay": config.max_alerts_per_day,
        "editorialReview": editorial_review,
        "morningReview": morning_review,
        "editorialScore": editorial_review["score"],
        "selectionScore": selection_score,
        "visitPotential": visit_potential,
        "expectedOpens": int(round(float(visit_potential.get("expectedOpens") or 0.0))),
        # Legacy alias for the existing persistence/API schema. The metric is
        # explicitly labelled below and must not be presented as a visit count.
        "expectedVisits": int(round(float(visit_potential.get("expectedOpens") or 0.0))),
        "responseMetric": "expected_opens",
        "estimatedReach": int(round(float(visit_potential.get("estimatedReach") or 0.0))),
        "visitPotentialScore": round(float(visit_potential.get("score") or 0.0), 1),
        "predictedOR": predicted_or,
        "forecast": forecast,
        "forecastSuspectedDefault": forecast_suspected_default,
        "forecastSuspectValue": round(raw_model_or, 2) if forecast_suspected_default else None,
        "predictedORSource": forecast["source"],
        "predictedORBasis": forecast["basis"],
        "predictedORConfidence": forecast["confidence"],
        "minScore": round(effective_min_score, 2),
        "baseMinScore": round(float(min_score), 2),
        "minOR": min_or,
        "minMinutesSinceLastPush": min_pause,
        "minutesSinceLastPush": minutes_since_last_push,
        "teamsCooldownMinutes": _effective_global_cooldown_minutes(config),
        "dashboardRank": dashboard_rank or None,
        "dashboardTopLimit": dashboard_top_limit,
        "candidateLimit": max(1, int(config.candidate_limit or PUSH_TEAMS_CANDIDATE_LIMIT)),
        "expandedFieldCandidate": bool(expanded_field.get("allowed")),
        "expandedFieldReason": str(expanded_field.get("reason") or ""),
        "lastPushAt": _iso_from_ts(last_push_ts) if last_push_ts else None,
        "lastGlobalTeamsAlertAt": _iso_from_ts(last_teams_alert_ts)
        if last_teams_alert_ts
        else None,
        "minutesSinceLastGlobalTeamsAlert": minutes_since_last_teams_alert,
        "lastTeamsAlertAt": _iso_from_ts(_safe_int(alert_state.get("last_alert_ts")))
        if alert_state
        else None,
        "alertCount": int(alert_state.get("alert_count") or 0) if alert_state else 0,
        "isBreaking": breaking,
        "scoreOnlyMode": config.score_only_mode,
        "section": section,
        "germanyRelevance": germany_relevance,
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
    breaking_eligible = [item for item in eligible if bool(item[1].get("isBreaking"))]
    ranking_pool = breaking_eligible if breaking_eligible else eligible
    canonical_api_selection = bool(config.require_internal_score_api and ranking_pool)
    if canonical_api_selection:
        top_raw_score = max(float(item[1].get("score") or 0.0) for item in ranking_pool)
        # The API score is the ranking contract. Secondary models may only break
        # an exact score tie and can never let a lower API score overtake Top 1.
        selection_pool = [
            item
            for item in ranking_pool
            if abs(float(item[1].get("score") or 0.0) - top_raw_score) < 0.001
        ]
    elif breaking_eligible:
        selection_pool = breaking_eligible
    elif eligible:
        top_raw_score = max(float(item[1].get("score") or 0.0) for item in eligible)
        selection_pool = [
            item
            for item in eligible
            if float(item[1].get("score") or 0.0) >= top_raw_score - _PUSH_SCORE_SELECTION_BAND
        ]
    else:
        selection_pool = []

    def _selection_value(decision: dict[str, Any]) -> float:
        return float(decision.get("selectionScore") or 0.0)

    selected_key: str | None = None
    selected_candidate: dict[str, Any] | None = None
    selected_decision: dict[str, Any] | None = None
    if selection_pool:

        def _selection_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
            candidate, decision = item
            if canonical_api_selection:
                return (
                    float(decision.get("score") or 0.0),
                    _selection_value(decision),
                    float(decision.get("editorialScore") or 0.0),
                    float(decision.get("teamsAlertScore") or 0.0),
                    float(decision.get("expectedOpens") or 0.0),
                    *_candidate_rank(candidate),
                    candidate_key(candidate),
                )
            return (
                _selection_value(decision),
                float(decision.get("score") or 0.0),
                float(decision.get("editorialScore") or 0.0),
                float(decision.get("teamsAlertScore") or 0.0),
                float(decision.get("expectedOpens") or 0.0),
                *_candidate_rank(candidate),
                candidate_key(candidate),
            )

        selected_candidate, selected_decision = max(
            selection_pool,
            key=_selection_key,
        )
        selected_key = candidate_key(selected_candidate)

    runner_up_decision: dict[str, Any] | None = None
    selection_margin: float | None = None
    selection_margin_percent: float | None = None
    selection_confidence = "hoch"
    runner_up_pool = ranking_pool if canonical_api_selection else selection_pool
    if selected_decision is not None and len(runner_up_pool) >= 2:
        runner_up_pair = max(
            (item for item in runner_up_pool if candidate_key(item[0]) != selected_key),
            key=_selection_key,
            default=None,
        )
        if runner_up_pair is not None:
            runner_up_decision = runner_up_pair[1]
            winner_value = (
                float(selected_decision.get("score") or 0.0)
                if canonical_api_selection
                else _selection_value(selected_decision)
            )
            runner_value = (
                float(runner_up_decision.get("score") or 0.0)
                if canonical_api_selection
                else _selection_value(runner_up_decision)
            )
            selection_margin = winner_value - runner_value
            selection_margin_percent = 100.0 * selection_margin / max(abs(winner_value), 0.001)
            if canonical_api_selection and selection_margin >= 5.0:
                selection_confidence = "hoch"
            elif canonical_api_selection and selection_margin >= 2.0:
                selection_confidence = "mittel"
            elif selection_margin_percent >= 15.0:
                selection_confidence = "hoch"
            elif selection_margin_percent >= 5.0:
                selection_confidence = "mittel"
            else:
                selection_confidence = "niedrig"

    # "Klarer Gewinner"-Regel: wenn der Top-Kandidat nur knapp vor dem Verfolger
    # liegt und selbst nicht eindeutig stark ist, ist das Feld unsicher -> kein Alert.
    uncertainty_reason = ""
    if (
        selected_decision is not None
        and len(selection_pool) >= 2
        and config.min_selection_margin > 0
        and not canonical_api_selection
        and not bool(selected_decision.get("isBreaking"))
        and not bool((selected_decision.get("minimumPressure") or {}).get("active"))
    ):
        if runner_up_decision is not None and selection_margin is not None:
            margin = selection_margin
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
            decision["agentReview"] = add_agent_review_veto(
                decision.get("agentReview"),
                agent="Kandidatenfeld",
                reason=uncertainty_reason,
            )
            final.append({"candidate": candidate, "decision": decision})
            continue
        if selected_key and key != selected_key and base_decision.get("shouldNotify"):
            decision = dict(base_decision)
            stronger_reason = "Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"
            decision["shouldNotify"] = False
            decision["status"] = "observe"
            decision["recommendedAction"] = ""
            decision["blockingReasons"] = [
                *decision.get("blockingReasons", []),
                stronger_reason,
            ]
            decision["agentReview"] = add_agent_review_veto(
                decision.get("agentReview"),
                agent="Kandidatenfeld",
                reason=stronger_reason,
            )
        else:
            decision = dict(base_decision)
            if selected_key and key == selected_key:
                competitors = max(0, len(ranking_pool) - 1)
                decision["competition"] = {
                    "eligibleCompetitors": competitors,
                    "selectionMargin": (
                        round(selection_margin, 2) if selection_margin is not None else None
                    ),
                    "selectionMarginPercent": (
                        round(selection_margin_percent, 1)
                        if selection_margin_percent is not None
                        else None
                    ),
                    "selectionMetric": (
                        "internal_push_balancer_score"
                        if canonical_api_selection
                        else (
                            "push_score_weighted_selection"
                            if config.visit_optimization_enabled
                            else "selection_score"
                        )
                    ),
                    "winnerScore": float(decision.get("score") or 0.0),
                    "runnerUpScore": (
                        float(runner_up_decision.get("score") or 0.0)
                        if runner_up_decision is not None
                        else None
                    ),
                    "scoreDelta": (
                        round(selection_margin, 2)
                        if canonical_api_selection and selection_margin is not None
                        else None
                    ),
                    "selectionConfidence": selection_confidence,
                    "summary": (
                        f"{competitors} weitere pushwuerdige Kandidaten geprueft"
                        if competitors
                        else "Kein staerkerer Kandidat aktuell verfuegbar"
                    ),
                }
                if competitors:
                    selection_reason = (
                        "Top 1 nach kanonischem API-Push-Score; Sekundaermodelle duerfen nur exakte Gleichstaende entscheiden"
                        if canonical_api_selection
                        else (
                            "Staerkster Push-Score-gewichteter Kandidat; Response-Potenzial entscheidet nur mit"
                            if config.visit_optimization_enabled
                            else "Beste CvD-Eignung aus Nachrichtenwert, Timing und Nutzerbelastung im Kandidatenfeld"
                        )
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
        "canonicalApiTop1": canonical_api_selection,
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
        item["decision"]["candidateId"]: item["decision"] for item in evaluations["decisions"]
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
    score_source = str(
        decision.get("scoreSource")
        or candidate.get("scoreSource")
        or "server_editorial_fallback"
    )
    score_source_label = {
        "internal_score_api": "kanonischer Score der internen Push-Balancer-API",
        "captured_push_balancer": "frisches Push-Balancer-Rating",
    }.get(score_source, "serverseitiger Editorial-Fallback")
    score_scored_at = str(candidate.get("pushBalancerScoreScoredAt") or "").strip()
    now_ts = int(context.get("nowTs") or time.time())
    forecast = decision.get("forecast") if isinstance(decision.get("forecast"), dict) else None
    if not forecast:
        suspect_values = {float(v) for v in (context.get("suspectForecastValues") or [])}
        forecast = _candidate_forecast(candidate, now_ts, suspect_values)
    predicted_or = forecast["value"]
    minutes = decision.get("minutesSinceLastPush")
    minutes_known = isinstance(minutes, (int, float))
    teams_minutes = decision.get("minutesSinceLastGlobalTeamsAlert")
    teams_minutes_known = isinstance(teams_minutes, (int, float))
    independent_pacing = bool(
        decision.get("recommendationsIndependentFromLivePushes")
        or config.independent_pacing_enabled
    )
    score_only_mode = bool(decision.get("scoreOnlyMode") or config.score_only_mode)
    score_threshold = float(decision.get("minScore", config.min_score) or config.min_score)
    is_breaking = bool(decision.get("isBreaking"))
    mandatory_score_threshold = max(
        score_threshold,
        _HARD_BREAKING_PUSH_SCORE_FLOOR if is_breaking else _HARD_NORMAL_PUSH_SCORE_FLOOR,
    )
    alert_score = float(decision.get("teamsAlertScore") or 0.0)
    alert_threshold = float(decision.get("teamsAlertScoreThreshold") or config.min_alert_score)
    editorial_review = decision.get("editorialReview") or {}
    editorial_score = float(editorial_review.get("score") or 0.0)
    editorial_reasons = list(editorial_review.get("reasons") or [])
    editorial_breakdown = editorial_review.get("breakdown") or {}
    time_fit_score = float(editorial_breakdown.get("timeFit") or 0.0)
    time_fit_label = str(editorial_breakdown.get("timeFitLabel") or "").strip()
    selection_score = float(decision.get("selectionScore") or 0.0)
    visit_potential = (
        decision.get("visitPotential") if isinstance(decision.get("visitPotential"), dict) else {}
    )
    opening_estimate = _format_expected_openings(visit_potential)
    visit_reason = f"Öffnungspotenzial: {opening_estimate}." if opening_estimate else ""
    germany_relevance = (
        decision.get("germanyRelevance")
        if isinstance(decision.get("germanyRelevance"), dict)
        else {}
    )
    germany_relevance_reason = {
        "germany_broad": (
            "Deutschland-Relevanz: Das Thema betrifft viele Menschen in Deutschland direkt."
        ),
        "germany_domestic": (
            "Deutschland-Relevanz: Es ist eine starke Inlandsmeldung ohne künstlichen Reichweitenbonus."
        ),
        "germany_people": (
            "Deutschland-People: Eine benannte deutsche öffentliche Person und ein bestätigtes "
            "Lebensereignis erzeugen breiten Gesprächswert."
        ),
        "international_breaking": (
            "Internationales Breaking: Die weltweite Tragweite rechtfertigt die Sofortprüfung."
        ),
        "international": (
            "Internationaler Ausnahmefall: Der Kandidat erfüllt die erhöhte Push-Score-Hürde."
        ),
    }.get(str(germany_relevance.get("level") or ""), "")
    agent_review = (
        decision.get("agentReview") if isinstance(decision.get("agentReview"), dict) else {}
    )
    agent_summary = _agent_review_display_summary(agent_review)
    agent_counterargument = (
        str(agent_review.get("mainCounterargument") or "").strip()
        if agent_review.get("enabled")
        else ""
    )
    push_text, push_title_source, push_title_review = _teams_push_title_selection(
        candidate,
        title,
        section,
        url,
        config,
    )
    push_text_matches_title = _same_editorial_text(push_text, title)
    push_title_click_reason = str(push_title_review.get("clickReason") or "").strip()
    push_title_display_reason = (
        push_title_click_reason[:1].upper() + push_title_click_reason[1:]
        if push_title_click_reason
        else ""
    )
    push_title_review_line = (
        f"Titelqualität: {_format_number(float(push_title_review.get('score') or 0.0), 0)}/100"
        + (f": {push_title_display_reason}" if push_title_display_reason else "")
    )
    timing_brief = _recommendation_timing_brief(
        candidate,
        decision,
        now_ts=now_ts,
        config=config,
    )
    recommendation_review = _recommendation_quality_review(
        candidate,
        decision,
        push_title_review,
        timing_brief,
        config=config,
    )
    recommendation_score = float(recommendation_review.get("score") or 0.0)
    recommendation_confidence = str(recommendation_review.get("confidence") or "niedrig")
    decision_basis = _recommendation_decision_basis(timing_brief)
    agent_approved = bool(not config.agent_review_enabled or agent_review.get("approved"))
    context_available = context.get("contextAvailable") or {}
    teams_dedup_approved = bool(
        context_available.get("alertState") and context_available.get("recentTeamsAlerts")
    )
    dispatch_approved = bool(
        decision.get("shouldNotify")
        and push_title_review.get("approved")
        and recommendation_review.get("approved")
        and agent_approved
        and teams_dedup_approved
    )
    dispatch_blockers = _dedupe(
        [
            *list(decision.get("blockingReasons") or []),
            *list(recommendation_review.get("blockers") or []),
            *(
                ["Die eigene Teams-Dublettenhistorie ist nicht vollständig verfügbar."]
                if not teams_dedup_approved
                else []
            ),
            *(
                [str(agent_review.get("blockingReason") or "Agentenfreigabe fehlt")]
                if not agent_approved
                else []
            ),
            *(
                list(push_title_review.get("risks") or [])
                if not push_title_review.get("approved")
                else []
            ),
        ]
    )
    dispatch_blocking_reason = (
        dispatch_blockers[0]
        if dispatch_blockers
        else "Die vollständige lokale Versandfreigabe fehlt."
    )
    competition_meta = decision.get("competition") or {}
    competitors = int(competition_meta.get("eligibleCompetitors") or 0)
    competition = (
        (
            f"Top 1 nach internem Push-Score ({competitors + 1} freigegebene Kandidaten geprüft)."
            if score_source == "internal_score_api"
            else f"Im aktuellen Kandidatenfeld ist das der stärkste Vorschlag ({competitors + 1} geprüft)."
        )
        if competitors
        else (
            "Top 1 nach internem Push-Score; kein weiterer freigegebener Kandidat."
            if score_source == "internal_score_api"
            else "Im aktuellen Kandidatenfeld gibt es keinen stärkeren Push-Vorschlag."
        )
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
    live_push_comparison = (
        decision.get("livePushComparison")
        if isinstance(decision.get("livePushComparison"), dict)
        else {}
    )
    if not live_push_comparison.get("available"):
        live_comparison_label = "Live-Vergleich: aktuell nicht belastbar verfügbar"
    elif live_push_comparison.get("matched"):
        live_comparison_label = "Live-Vergleich: entspricht einem echten Live-Push"
    else:
        live_comparison_label = "Live-Vergleich: zum Prüfzeitpunkt kein entsprechender Live-Push"
    if independent_pacing:
        timing_reason = (
            "Der Teams-Kanal arbeitet unabhängig von echten Live-Pushes mit eigenem "
            "Tagespacing und Cooldown; die Live-Historie ist nur ein Vergleichssignal."
        )
    elif isinstance(minutes, (int, float)):
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
    post_send_threshold = (
        decision.get("postSendScoreThreshold")
        if isinstance(decision.get("postSendScoreThreshold"), dict)
        else {}
    )
    high_score_override = (
        decision.get("highScoreOverride")
        if isinstance(decision.get("highScoreOverride"), dict)
        else {}
    )
    adaptive_threshold_reason = (
        f"Adaptive Schwelle: {post_send_threshold.get('reason')}"
        if post_send_threshold.get("active")
        else ""
    )
    high_score_reason = (
        "High-Score-Regel: Der kanonische Push Score liegt über 80; weiche "
        "Qualitätsgates treten zurück, alle harten Sperren sind erfüllt."
        if high_score_override.get("approved")
        else ""
    )
    why_now = _dedupe(
        [
            editorial_reason,
            visit_reason,
            *([f"Push-Balancer-Score: {candidate_score_reason}"] if candidate_score_reason else []),
            time_fit_reason,
            forecast_reason,
            threshold_reason,
            score_reason,
            adaptive_threshold_reason,
            high_score_reason,
            timing_reason,
            competition,
            live_comparison_label,
        ]
    )[:6]
    speculative_caution = str(decision.get("speculativeCaution") or "").strip()
    if speculative_caution and speculative_caution not in candidate_risks:
        candidate_risks = [speculative_caution, *candidate_risks]
    what_speaks_against = _dedupe(
        [
            str(recommendation_review.get("strongestRisk") or "").strip(),
            agent_counterargument,
            *candidate_risks,
        ]
    )[:5] or ["Kein harter Einwand; die OR-Prognose bleibt eine Schätzung."]
    why_article = _dedupe(
        [
            germany_relevance_reason,
            *candidate_drivers[:2],
            push_title_display_reason,
            forecast_reason,
            (
                f"Öffnungspotenzial: {opening_estimate}."
                if predicted_or is not None and opening_estimate
                else ""
            ),
            competition,
        ]
    )[:5]
    compact_reasons = _dedupe(
        [
            *list(timing_brief.get("reasons") or []),
            adaptive_threshold_reason,
            high_score_reason,
            timing_reason,
        ]
    )[:5]
    subject_prefix = "🚨 Jetzt pushen" if dispatch_approved else "Nicht senden"
    subject = f"{subject_prefix}: {_compact_text(push_text or title, 120)}"

    text_lines = [subject, "", "Empfohlener Push-Titel:", push_text]
    if not push_text_matches_title:
        text_lines.extend(["", "Artikel:", title])
    if url:
        text_lines.append(url)
    text_lines.extend(
        [
            "",
            f"Versandfenster: {timing_brief['windowLabel']}",
            (
                f"Push-Score: {_format_number(score, 1)}/100 "
                f"(harte Schwelle {_format_number(score_threshold, 0)}) | "
                f"Quelle: {score_source_label}"
            ),
            *([f"Score-Stand: {score_scored_at}"] if score_scored_at else []),
            (
                f"Qualitätsurteil: {_format_number(recommendation_score, 0)}/100 | "
                f"Empfehlungsstärke {recommendation_confidence}"
            ),
            f"Entscheidungsbasis: {decision_basis}",
            live_comparison_label,
            "",
            (
                f"{section_label} | OR-Prognose {_format_or(predicted_or)} | "
                f"{opening_estimate or 'Öffnungspotenzial nicht belastbar'} | "
                + (
                    f"letzter Teams-Hinweis {_format_teams_alert_minutes(teams_minutes)}"
                    if independent_pacing
                    else f"letzter Push {_format_minutes(minutes)}"
                )
            ),
            *([f"Prüfstatus: {agent_summary}"] if agent_summary else []),
            *([push_title_review_line] if push_title_review_line else []),
            "",
            "Warum dieser Push?",
            *[f"- {reason}" for reason in why_article],
            "",
            "Warum jetzt?",
            *[f"- {reason}" for reason in compact_reasons],
            "",
            "Gegencheck:",
            f"- {what_speaks_against[0]}",
            "",
            (
                f"Empfehlung: Jetzt pushen. (Stand {_format_time(now_ts)} Uhr)"
                if dispatch_approved
                else f"Empfehlung: Nicht senden. {dispatch_blocking_reason}"
            ),
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
        minutes_since_last_teams_alert=teams_minutes,
        independent_pacing=independent_pacing,
        score_threshold=score_threshold,
        score_source_label=score_source_label,
        alert_score=alert_score,
        alert_threshold=alert_threshold,
        editorial_score=editorial_score,
        why_now=compact_reasons,
        why_pushworthy=why_article,
        subject=subject,
        push_text_matches_title=push_text_matches_title,
        score_reason=candidate_score_reason,
        performance_drivers=candidate_drivers,
        risks=what_speaks_against,
        score_breakdown_lines=candidate_breakdown_lines,
        agent_review=agent_review,
        push_title_review=push_title_review,
        recommendation_review=recommendation_review,
        live_push_comparison=live_push_comparison,
        dispatch_approved=dispatch_approved,
        decision_basis=decision_basis,
        dispatch_blocking_reason=dispatch_blocking_reason,
    )
    return {
        "text": text,
        "payload": {
            "type": ("push_recommendation" if dispatch_approved else "push_recommendation_preview"),
            "subject": subject,
            "recommendedAction": "Jetzt pushen" if dispatch_approved else "",
            "dispatchApproved": dispatch_approved,
            "recommendationPolicyVersion": _TEAMS_RECOMMENDATION_POLICY_VERSION,
            "minimumPushScore": mandatory_score_threshold,
            "isBreaking": is_breaking,
            "decisionBasis": decision_basis,
            "dispatchBlockingReason": ("" if dispatch_approved else dispatch_blocking_reason),
            "articleTitle": title,
            "articleUrl": url,
            "category": section,
            "pushScore": score,
            "pushScoreSource": score_source,
            "pushScoreSourceLabel": score_source_label,
            "pushScoreScoredAt": score_scored_at,
            "teamsAlertScore": alert_score,
            "teamsAlertScoreThreshold": alert_threshold,
            "teamsAlertScoreBreakdown": decision.get("teamsAlertScoreBreakdown") or {},
            "postSendScoreThreshold": post_send_threshold,
            "highScoreOverride": {
                "active": bool(high_score_override.get("active")),
                "approved": bool(high_score_override.get("approved")),
                "threshold": high_score_override.get("threshold"),
                "waivedGateCount": len(high_score_override.get("waivedBlockers") or []),
                "hardBlockerCount": len(high_score_override.get("hardBlockers") or []),
            },
            "editorialReview": editorial_review,
            "editorialScore": editorial_score,
            "editorialReasons": editorial_reasons,
            "scoreReason": candidate_score_reason,
            "performanceDrivers": candidate_drivers,
            "risks": candidate_risks,
            "scoreBreakdown": candidate.get("scoreBreakdown")
            if isinstance(candidate.get("scoreBreakdown"), dict)
            else {},
            "scoreBreakdownLabel": "; ".join(candidate_breakdown_lines),
            "timeFitScore": time_fit_score,
            "timeFitLabel": time_fit_label,
            "selectionScore": selection_score,
            "visitPotential": visit_potential,
            "expectedOpens": int(visit_potential.get("expectedOpens") or 0),
            "expectedVisits": int(visit_potential.get("expectedOpens") or 0),
            "responseMetric": "expected_opens",
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
            "pushTitleReview": _public_push_title_review(push_title_review),
            "recommendationQuality": _public_recommendation_review(recommendation_review),
            "recommendedSendWindow": timing_brief["windowLabel"],
            "recommendedSendBy": timing_brief["sendBy"],
            "recommendedAt": _format_dt(now_ts),
            "minutesSinceLastPush": round(float(minutes), 1) if minutes_known else 0.0,
            "lastPushKnown": minutes_known,
            "timeSinceLastPushLabel": _format_minutes(minutes),
            "recommendationsIndependentFromLivePushes": independent_pacing,
            "livePushComparison": {
                "available": bool(live_push_comparison.get("available")),
                "matched": bool(live_push_comparison.get("matched")),
                "matchType": str(live_push_comparison.get("matchType") or ""),
            },
            "minutesSinceLastTeamsAlert": (
                round(float(teams_minutes), 1) if teams_minutes_known else 0.0
            ),
            "lastTeamsAlertKnown": teams_minutes_known,
            "timeSinceLastTeamsAlertLabel": _format_teams_alert_minutes(teams_minutes),
            "whyNow": why_now,
            "compactWhyNow": compact_reasons,
            "whyPushworthy": why_article,
            "countercheck": what_speaks_against[0],
            "competition": competition,
            "messageText": text,
            "messageHtml": message_html,
            "text": text,
        },
        "_agentReview": agent_review,
        "_pushTitleReview": push_title_review,
        "_recommendationReview": recommendation_review,
        "_dispatchApproved": dispatch_approved,
        "_teamsDedupApproved": teams_dedup_approved,
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
    quiet_reason = _quiet_hours_reason(int(time.time()), config)
    if quiet_reason:
        log.info("[TeamsAlert] send blocked by quiet hours")
        return {
            "ok": False,
            "blocked": True,
            "reason": "quiet_hours",
            "error": quiet_reason,
        }

    payload = message.get("payload") or {"text": str(message.get("text") or "")}
    payload_type = str(payload.get("type") or "")
    if payload_type == "push_recommendation_preview":
        log.warning("[TeamsAlert] push dispatch blocked: preview is not sendable")
        return {
            "ok": False,
            "blocked": True,
            "error": "Push recommendation is not fully approved",
        }
    if payload_type == "push_recommendation":
        if not message.get("_dispatchApproved") or not payload.get("dispatchApproved"):
            log.warning("[TeamsAlert] push dispatch blocked: dispatch approval missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Dispatch approval missing",
            }
        if payload.get("recommendationPolicyVersion") != _TEAMS_RECOMMENDATION_POLICY_VERSION:
            log.warning("[TeamsAlert] push dispatch blocked: recommendation policy mismatch")
            return {
                "ok": False,
                "blocked": True,
                "error": "Current recommendation policy approval missing",
            }
        if payload.get("recommendationsIndependentFromLivePushes") is not True:
            log.warning("[TeamsAlert] push dispatch blocked: live-push-independent pacing missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Independent Teams pacing approval missing",
            }
        if (
            config.require_internal_score_api
            and payload.get("pushScoreSource") != "internal_score_api"
        ):
            log.warning("[TeamsAlert] push dispatch blocked: canonical score missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Canonical internal Push Balancer score is missing",
            }
        push_score = _safe_float(payload.get("pushScore"))
        payload_floor = _safe_float(payload.get("minimumPushScore"))
        is_breaking = bool(payload.get("isBreaking"))
        high_score_override = payload.get("highScoreOverride")
        high_score_override = (
            high_score_override if isinstance(high_score_override, dict) else {}
        )
        high_score_override_active = bool(
            high_score_override.get("approved")
            and push_score is not None
            and push_score > float(config.high_score_always_threshold or 80.0)
        )
        configured_floor = (
            float(config.high_score_always_threshold or 80.0)
            if high_score_override_active
            else float(config.breaking_min_score)
            if is_breaking
            else float(config.min_score)
        )
        policy_floor = (
            _HARD_BREAKING_PUSH_SCORE_FLOOR if is_breaking else _HARD_NORMAL_PUSH_SCORE_FLOOR
        )
        required_score = max(policy_floor, configured_floor, float(payload_floor or 0.0))
        if push_score is None or push_score < required_score:
            log.warning(
                "[TeamsAlert] push dispatch blocked: raw score %.1f below %.1f",
                float(push_score or 0.0),
                required_score,
            )
            return {
                "ok": False,
                "blocked": True,
                "error": "Raw Push Score is below the mandatory dispatch floor",
            }
        if not message.get("_teamsDedupApproved"):
            log.warning("[TeamsAlert] push dispatch blocked: Teams duplicate context missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Teams duplicate protection approval missing",
            }
        title_review = message.get("_pushTitleReview")
        if not isinstance(title_review, dict) or not title_review.get("approved"):
            log.warning("[TeamsAlert] push dispatch blocked: grounded title approval missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Grounded title approval missing",
            }
        recommendation_review = message.get("_recommendationReview")
        if (
            not isinstance(recommendation_review, dict)
            or not recommendation_review.get("enforced")
            or not recommendation_review.get("approved")
        ):
            log.warning("[TeamsAlert] push dispatch blocked: final recommendation quality missing")
            return {
                "ok": False,
                "blocked": True,
                "error": "Final recommendation quality approval missing",
            }
        if config.agent_review_enabled:
            review = message.get("_agentReview")
            if not isinstance(review, dict) or not review.get("approved"):
                log.warning("[TeamsAlert] push dispatch blocked: local agent approval missing")
                return {
                    "ok": False,
                    "blocked": True,
                    "error": "Local agent approval missing",
                }
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
    history_authoritative: bool | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate batch, send the best recommendation, and persist state."""
    config = config or TeamsAlertConfig()
    candidate_limit = max(
        1,
        min(int(config.candidate_limit or PUSH_TEAMS_CANDIDATE_LIMIT), PUSH_TEAMS_CANDIDATE_LIMIT),
    )
    decision_ts = int(now_ts or time.time())
    memory_eligible, memory_guard = _memory_eligible_candidates(
        candidates,
        now_ts=decision_ts,
        config=config,
    )
    limited = memory_eligible[:candidate_limit]
    context = build_teams_alert_context(
        limited,
        history_authoritative=history_authoritative,
        now_ts=decision_ts,
        config=config,
    )
    evaluation = evaluate_teams_alert_candidates(limited, context, config)
    evaluation["memoryGuard"] = memory_guard
    selected = evaluation.get("selectedCandidate")
    selected_decision = None

    for item in evaluation["decisions"]:
        candidate = item["candidate"]
        decision = item["decision"]
        _log_decision(candidate, decision)
        if selected and candidate_key(candidate) == candidate_key(selected):
            selected_decision = decision

    if not selected or not selected_decision or not selected_decision.get("shouldNotify"):
        diagnostics = _no_candidate_diagnostics(evaluation, context, config)
        log.info(
            "[TeamsAlert] no_candidate evaluated=%s score_eligible=%s teams_today=%s "
            "due=%s target=%s shortfall=%s memory_skipped=%s blocker_categories=%s",
            diagnostics["evaluatedCandidates"],
            diagnostics["scoreEligibleCandidates"],
            diagnostics["teamsAlertsToday"],
            diagnostics["dueOpportunityCount"],
            diagnostics["targetCount"],
            diagnostics["projectedShortfall"],
            diagnostics["memorySkippedCandidates"],
            diagnostics["blockerCategories"],
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "no_candidate",
            "diagnostics": diagnostics,
            "evaluation": evaluation,
        }

    selected_review = selected_decision.get("agentReview")
    if config.agent_review_enabled and (
        not isinstance(selected_review, dict) or not selected_review.get("approved")
    ):
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="agent_review_blocked",
            send_status="blocked",
            send_error="Local agent approval missing",
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "agent_review_blocked",
            "evaluation": evaluation,
        }

    article_key = candidate_key(selected)
    article_id = str(selected.get("id") or article_key)
    article_url = _url(selected)
    article_ref = hashlib.sha256(article_key.encode("utf-8")).hexdigest()[:12]
    decision_ts = int(context.get("nowTs") or decision_ts)
    dispatch_comparison = _dispatch_live_push_comparison(
        selected,
        now_ts=decision_ts,
        config=config,
        comparison_authoritative=bool(context.get("historyAuthoritative")),
    )
    selected_decision = dict(selected_decision)
    selected_decision["livePushComparison"] = dict(
        dispatch_comparison.get("livePushComparison") or {}
    )
    message = build_teams_push_recommendation(selected, context, selected_decision, config)
    selected_title_review = message.get("_pushTitleReview")
    if not isinstance(selected_title_review, dict) or not selected_title_review.get("approved"):
        selected_decision = dict(selected_decision)
        selected_decision["pushTitleReview"] = selected_title_review or {}
        log.info(
            "[TeamsAlert] title jury blocked article_ref=%s score=%.1f",
            article_ref,
            float((selected_title_review or {}).get("score") or 0.0),
        )
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="title_review_blocked",
            send_status="blocked",
            send_error="Grounded title approval missing",
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "title_review_blocked",
            "titleReview": selected_title_review or {},
            "candidateId": article_key,
            "evaluation": evaluation,
        }
    selected_recommendation_review = message.get("_recommendationReview")
    public_recommendation_review = _public_recommendation_review(
        selected_recommendation_review if isinstance(selected_recommendation_review, dict) else None
    )
    selected_decision = dict(selected_decision)
    selected_decision["recommendationQuality"] = public_recommendation_review
    recommendation_enforced = True
    if recommendation_enforced and (
        not isinstance(selected_recommendation_review, dict)
        or not selected_recommendation_review.get("enforced")
        or not selected_recommendation_review.get("approved")
    ):
        recommendation_score = (
            float(selected_recommendation_review.get("score") or 0.0)
            if isinstance(selected_recommendation_review, dict)
            else 0.0
        )
        log.info(
            "[TeamsAlert] recommendation jury blocked article_ref=%s score=%.1f",
            article_ref,
            recommendation_score,
        )
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="recommendation_quality_blocked",
            send_status="blocked",
            send_error="Final recommendation quality approval missing",
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "recommendation_quality_blocked",
            "recommendationQuality": public_recommendation_review,
            "candidateId": article_key,
            "evaluation": evaluation,
        }
    if not message.get("_dispatchApproved"):
        log.info(
            "[TeamsAlert] dispatch approval blocked before send claim article_ref=%s",
            article_ref,
        )
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="dispatch_approval_blocked",
            send_status="blocked",
            send_error="Teams duplicate protection approval missing",
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "dispatch_approval_blocked",
            "candidateId": article_key,
            "evaluation": evaluation,
        }
    memory_claim = _memory_send_blocker_or_reserve(
        article_key=article_key,
        title=_title(selected),
        now_ts=decision_ts,
        config=config,
        bypass_global_cooldown=bool(_is_breaking(selected) and config.breaking_override),
    )
    if memory_claim.get("blocked"):
        log.info(
            "[TeamsAlert] send skipped by memory guard article_ref=%s reason=%s",
            article_ref,
            memory_claim.get("reason"),
        )
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="send_memory_blocked",
            send_status="blocked",
            send_error=str(memory_claim.get("reason") or ""),
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
        global_cooldown_minutes=(
            0
            if _is_breaking(selected) and config.breaking_override
            else _effective_global_cooldown_minutes(config)
        ),
        failed_cooldown_minutes=max(
            config.alert_cooldown_minutes,
            config.repeat_suppression_hours * 60,
        ),
    )
    if not claim.get("claimed"):
        _memory_release_reservation(article_key)
        log.info(
            "[TeamsAlert] send skipped by claim article_ref=%s reason=%s",
            article_ref,
            claim.get("reason"),
        )
        _persist_teams_recommendation(
            selected,
            selected_decision,
            context,
            config,
            status="send_claim_blocked",
            send_status="blocked",
            send_error=str(claim.get("reason") or ""),
        )
        return {
            "ok": True,
            "sent": False,
            "reason": "send_claim_blocked",
            "claim": claim,
            "candidateId": article_key,
            "evaluation": evaluation,
        }

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
    _persist_teams_recommendation(
        selected,
        selected_decision,
        context,
        config,
        status=status,
        send_status=status,
        send_error=str(send_result.get("error") or ""),
        sent_at_ts=decision_ts if send_result.get("ok") else 0,
    )

    log.info(
        "[TeamsAlert] send_result article_ref=%s status=%s ok=%s",
        article_ref,
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


def _no_candidate_diagnostics(
    evaluation: dict[str, Any],
    context: dict[str, Any],
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Build aggregate, title-free diagnostics for an empty worker cycle."""
    now_ts = int(context.get("nowTs") or time.time())
    local_date = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin")).date()
    opportunities = _daily_runtime_opportunities(local_date, config)
    due = [slot for slot in opportunities if int(slot.get("ts") or 0) <= now_ts]
    future = [slot for slot in opportunities if int(slot.get("ts") or 0) > now_ts]
    teams_today = _safe_int(context.get("teamsAlertsToday"))
    target = max(1, int(config.target_pushes_per_day or 15), len(opportunities))
    decisions = list(evaluation.get("decisions") or [])
    memory_guard = evaluation.get("memoryGuard")
    memory_guard = memory_guard if isinstance(memory_guard, dict) else {}
    memory_skipped = _safe_int(memory_guard.get("skippedCandidates"))
    score_eligible = 0
    category_counts: dict[str, int] = {}
    for item in decisions:
        candidate = item.get("candidate") if isinstance(item, dict) else {}
        decision = item.get("decision") if isinstance(item, dict) else {}
        candidate = candidate if isinstance(candidate, dict) else {}
        decision = decision if isinstance(decision, dict) else {}
        floor = (
            _HARD_BREAKING_PUSH_SCORE_FLOOR
            if _is_breaking(candidate)
            else _HARD_NORMAL_PUSH_SCORE_FLOOR
        )
        if _score(candidate) >= floor:
            score_eligible += 1
        categories = {
            _blocking_reason_category(str(reason or ""))
            for reason in decision.get("blockingReasons") or []
        }
        for category in categories or {"none"}:
            category_counts[category] = category_counts.get(category, 0) + 1

    for reason, count in (memory_guard.get("reasons") or {}).items():
        reason_text = str(reason or "")
        if reason_text == "memory_global_alert_cooldown":
            category = "teams_cooldown"
        elif reason_text in {"memory_article_alert_cooldown", "memory_topic_duplicate"}:
            category = "teams_duplicate"
        else:
            category = "other"
        category_counts[category] = category_counts.get(category, 0) + _safe_int(count)

    projected_maximum = min(target, teams_today + len(future))
    next_slot = min(future, key=lambda slot: int(slot.get("ts") or 0)) if future else None
    return {
        "evaluatedCandidates": len(decisions),
        "inputCandidates": len(decisions) + memory_skipped,
        "memorySkippedCandidates": memory_skipped,
        "scoreEligibleCandidates": score_eligible,
        "teamsAlertsToday": teams_today,
        "dueOpportunityCount": len(due),
        "remainingOpportunityCount": len(future),
        "plannedOpportunityCount": len(opportunities),
        "targetCount": target,
        "projectedMaximumFromPlan": projected_maximum,
        "projectedShortfall": max(0, target - projected_maximum),
        "nextOpportunity": str((next_slot or {}).get("label") or ""),
        "blockerCategories": dict(
            sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _blocking_reason_category(reason: str) -> str:
    normalized = str(reason or "").casefold()
    markers = (
        ("score", ("score zu niedrig", "push score", "push-score")),
        ("slot_wait", ("tagesplan:", "slot-logik", "fenster")),
        ("teams_cooldown", ("teams-cooldown", "tageslimit")),
        ("teams_duplicate", ("bereits per teams", "thema bereits per teams")),
        ("freshness", ("artikel nicht frisch", "zeitstempel", "aktualitaet")),
        ("forecast", ("prognose", "forecast", "or-erwartung")),
        ("editorial", ("cvd", "nachrichtenwert", "ressort")),
        ("morning_fit", ("morgenfit",)),
        ("sport", ("sport ohne", "sport-ereignis", "live-ticker")),
        ("quality", ("qualitaet", "qualität", "titel")),
        ("agent_review", ("agenten", "pruefer", "prüfer")),
    )
    for category, needles in markers:
        if any(needle in normalized for needle in needles):
            return category
    return "other"


def _dispatch_live_push_comparison(
    candidate: dict[str, Any],
    *,
    now_ts: int,
    config: TeamsAlertConfig,
    comparison_authoritative: bool = True,
) -> dict[str, Any]:
    """Refresh the non-blocking real-push comparison immediately before delivery."""
    if not comparison_authoritative:
        return {
            "blocked": False,
            "code": "comparison_not_authoritative",
            "livePushComparison": {
                "available": False,
                "matched": False,
                "matchType": "",
            },
        }
    try:
        history = push_db_load_all(max_days=90, max_rows=3000)
    except Exception as exc:
        return {
            "blocked": False,
            "code": "comparison_unavailable",
            "livePushComparison": {
                "available": False,
                "matched": False,
                "matchType": "",
            },
            "errorType": type(exc).__name__,
        }
    if not history:
        return {
            "blocked": False,
            "code": "comparison_empty",
            "livePushComparison": {
                "available": False,
                "matched": False,
                "matchType": "",
            },
        }

    history_index = _push_history_review_index(history, now_ts, config)
    live_push_match_reason = _live_push_comparison_reason(
        candidate,
        history,
        now_ts,
        config,
        history_index=history_index,
    )
    return {
        "blocked": False,
        "code": "live_push_match" if live_push_match_reason else "no_live_push_match",
        "historyRows": len(history),
        "livePushCadenceIgnored": True,
        "livePushComparison": {
            "available": True,
            "matched": bool(live_push_match_reason),
            "matchType": _live_push_match_type(live_push_match_reason),
            "reason": live_push_match_reason,
        },
    }


def _persist_teams_recommendation(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    context: dict[str, Any],
    config: TeamsAlertConfig,
    *,
    recommendation_type: str = "teams_alert",
    status: str = "",
    send_status: str = "",
    send_error: str = "",
    scheduled_for_ts: int = 0,
    sent_at_ts: int = 0,
    record_id: str = "",
) -> None:
    """Best-effort durable recommendation audit log."""
    try:
        now_ts = int(context.get("nowTs") or time.time())
        forecast_ts = int(scheduled_for_ts or now_ts)
        forecast = _candidate_forecast(
            candidate,
            forecast_ts,
            {float(value) for value in (context.get("suspectForecastValues") or [])},
        )
        visit = (
            decision.get("visitPotential")
            if isinstance(decision.get("visitPotential"), dict)
            else {}
        )
        recommendation_id = record_id
        if recommendation_id:
            recommendation_id = hashlib.sha256(recommendation_id.encode("utf-8")).hexdigest()
        teams_recommendation_record(
            article_key=candidate_key(candidate),
            article_id=str(candidate.get("id") or candidate_key(candidate)),
            article_url=_url(candidate),
            article_title=_title(candidate),
            section=_section(candidate),
            recommendation_type=recommendation_type,
            status=status or str(decision.get("status") or ""),
            should_notify=bool(decision.get("shouldNotify")),
            score=float(decision.get("score") or _score(candidate)),
            teams_alert_score=float(decision.get("teamsAlertScore") or 0.0),
            teams_alert_threshold=float(
                decision.get("teamsAlertScoreThreshold") or config.min_alert_score
            ),
            editorial_score=float(decision.get("editorialScore") or 0.0),
            predicted_or=float(forecast.get("value") or 0.0),
            predicted_or_label=_format_forecast(forecast),
            expected_visits=int(decision.get("expectedOpens") or visit.get("expectedOpens") or 0),
            dashboard_rank=int(decision.get("dashboardRank") or 0),
            scheduled_for_ts=int(scheduled_for_ts or 0),
            decided_at_ts=now_ts,
            sent_at_ts=int(sent_at_ts or 0),
            send_status=send_status,
            send_error=send_error,
            summary=str(decision.get("summary") or ""),
            reasons=list(decision.get("reasons") or []),
            blocking_reasons=list(decision.get("blockingReasons") or []),
            decision=_teams_recommendation_decision_snapshot(decision),
            record_id=recommendation_id,
        )
    except Exception as exc:
        log.warning("[TeamsAlert] recommendation persistence failed: %s", exc)


def _teams_recommendation_decision_snapshot(decision: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidateId",
        "articleId",
        "articleUrl",
        "headline",
        "status",
        "recommendedAction",
        "shouldNotify",
        "summary",
        "score",
        "minScore",
        "baseMinScore",
        "postSendScoreThreshold",
        "highScoreOverride",
        "teamsAlertScore",
        "teamsAlertScoreThreshold",
        "editorialScore",
        "selectionScore",
        "visitPotentialScore",
        "expectedOpens",
        "expectedVisits",
        "responseMetric",
        "estimatedReach",
        "predictedOR",
        "predictedORSource",
        "predictedORBasis",
        "predictedORConfidence",
        "minOR",
        "minutesSinceLastPush",
        "dashboardRank",
        "dashboardTopLimit",
        "expandedFieldCandidate",
        "expandedFieldReason",
        "pushesToday",
        "teamsAlertsToday",
        "minAlertsPerDay",
        "maxAlertsPerDay",
        "pushBudgetTarget",
        "minimumPressure",
        "pushPacing",
        "section",
        "evaluatedAt",
        "reasons",
        "blockingReasons",
        "agentReview",
        "recommendationQuality",
        "publicationReview",
    )
    return {key: decision.get(key) for key in keys if key in decision}


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
    quiet_reason = _quiet_hours_reason(now, config)
    if quiet_reason:
        return {
            "ok": True,
            "sent": False,
            "reason": "quiet_hours",
            "blockingReason": quiet_reason,
        }
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
    payload["type"] = "push_recommendation_test"
    payload["isTest"] = True
    payload["messageHtml"] = f"<p><strong>{html.escape(banner)}</strong></p>" + str(
        payload.get("messageHtml") or ""
    )

    result = send_teams_notification(message, config)
    log.info("[TeamsAlert] test message send ok=%s", bool(result.get("ok")))
    return result


def send_teams_daily_schedule_if_due(
    config: TeamsAlertConfig | None = None,
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Send one compact schedule per Berlin calendar day, restart-safe."""
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    if not config.enabled or not config.daily_schedule_send_enabled:
        return {"ok": True, "sent": False, "reason": "disabled"}
    quiet_reason = _quiet_hours_reason(now, config)
    if quiet_reason:
        return {
            "ok": True,
            "sent": False,
            "reason": "quiet_hours",
            "blockingReason": quiet_reason,
        }
    local_dt = dt.datetime.fromtimestamp(now, ZoneInfo("Europe/Berlin"))
    send_minute = _parse_hhmm_to_minutes(config.daily_schedule_send_time)
    if send_minute is None:
        return {"ok": False, "sent": False, "reason": "invalid_send_time"}
    current_minute = local_dt.hour * 60 + local_dt.minute
    if current_minute < send_minute:
        return {"ok": True, "sent": False, "reason": "not_due"}

    date_iso = local_dt.date().isoformat()
    try:
        claim = teams_daily_schedule_try_claim(date_iso, now_ts=now)
    except Exception as exc:
        log.warning("[TeamsAlert] daily schedule claim failed: %s", exc)
        return {"ok": False, "sent": False, "reason": "claim_failed", "error": str(exc)}
    if not claim.get("claimed"):
        return {"ok": True, "sent": False, "reason": claim.get("reason") or "not_claimed"}

    schedule = build_teams_daily_schedule(local_dt.date(), config, now_ts=now)
    result = send_teams_notification({"payload": schedule["payload"]}, config)
    status = "sent" if result.get("ok") else "failed"
    teams_daily_schedule_record(
        date_iso,
        status=status,
        item_count=int(schedule.get("count") or 0),
        error=str(result.get("error") or ""),
        now_ts=now,
    )
    log.info("[TeamsAlert] daily schedule date=%s status=%s", date_iso, status)
    return {
        "ok": bool(result.get("ok")),
        "sent": bool(result.get("ok")),
        "reason": status,
        "date": date_iso,
        "count": int(schedule.get("count") or 0),
        "sendResult": result,
    }


def run_teams_alert_cycle() -> dict[str, Any]:
    """Fetch current article candidates and run one Teams alert cycle."""
    try:
        refresh_result = _refresh_push_history_for_timing()
        from app.routers.feed import build_articles_payload

        config = TeamsAlertConfig()
        schedule_result = send_teams_daily_schedule_if_due(config)
        candidate_limit = max(
            1,
            min(
                int(config.candidate_limit or PUSH_TEAMS_CANDIDATE_LIMIT),
                PUSH_TEAMS_CANDIDATE_LIMIT,
            ),
        )
        payload = build_articles_payload(
            offset=0,
            limit=candidate_limit,
            include_teams_decisions=False,
            use_internal_score_api=config.require_internal_score_api,
        )
        candidates = payload.get("articles") or []
        result = evaluate_and_send_best_candidate(
            candidates,
            config=config,
            history_authoritative=bool(refresh_result.get("history_authoritative")),
        )
        result["dailySchedule"] = schedule_result
        return result
    except Exception as exc:
        log.exception("[TeamsAlert] Cycle failed")
        return {"ok": False, "sent": False, "error": str(exc)}


def _refresh_push_history_for_timing() -> dict[str, Any]:
    """Best-effort refresh for live comparison and aggregate reach baselines."""
    try:
        from app.routers.push import _build_refresh_response

        result = _build_refresh_response()
        log.info(
            "[TeamsAlert] push history refresh source=%s synced=%s db_written=%s authoritative=%s age=%s",
            result.get("source"),
            result.get("synced"),
            result.get("db_written"),
            result.get("history_authoritative"),
            result.get("snapshot_age_seconds"),
        )
        return result
    except Exception as exc:
        log.warning("[TeamsAlert] push history refresh skipped: %s", exc)
        return {
            "ok": False,
            "source": "refresh-error",
            "history_authoritative": False,
            "snapshot_age_seconds": None,
            "error": type(exc).__name__,
        }


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
    persist: bool = False,
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
        slot_gate_enabled=False,
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
    double_opportunities = _daily_plan_double_opportunities(
        target,
        slots,
        plan_config,
        limit=24,
    )
    assignment_slots = _daily_plan_assignment_slots(
        slots,
        double_opportunities,
        limit=min(max_count, max(min_count, len(selected_entries))),
    )
    assigned = _daily_plan_assign_slots(
        selected_entries,
        assignment_slots,
        context,
        plan_config,
    )
    items = sorted(assigned, key=lambda item: (item["slotTs"], -float(item["planScore"])))
    for index, item in enumerate(items, start=1):
        item["number"] = index

    top_items = sorted(items, key=lambda item: float(item["planScore"]), reverse=True)[:5]
    quality_summary = _daily_plan_quality_summary(items)
    traffic_slots = _daily_plan_traffic_slots(target, plan_config)
    watch_topics = _daily_plan_watch_topics(ranked_entries[len(selected_entries) :], raw_entries)
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
        "requiredSlotCount": len(slots),
        "qualityOpportunityCount": len(double_opportunities),
        "items": items,
        "top5": [_daily_plan_item_summary(item) for item in top_items],
        "qualitySummary": quality_summary,
        "trafficSlots": traffic_slots,
        "doubleOpportunities": double_opportunities,
        "watchTopics": watch_topics,
        "notRecommended": not_recommended,
        "pacing": _push_pacing_review(
            context.get("teamsAlertsToday"),
            now,
            plan_config,
            basis="teams_alerts",
            actual_pushes_today=context.get("pushesToday"),
        ),
        "assumptions": [
            "Der Tagesplan ist eine CvD-Planung, kein automatischer Versand an Nutzer.",
            "Historische Totzonen werden nicht verwendet, nur um das Tagesziel rechnerisch zu fuellen.",
            "Reguläre Fenster entscheiden um :45; notwendige Doppelchancen zusätzlich um :00.",
            "Im fälligen Mindestfenster dominiert der harte Push-Score 75; Fakten-, Aktualitäts-, Titel-, Ruhezeit- und Dublettengates bleiben unverändert.",
            "Sport wird nur bei bestaetigter Ereignislage und passendem Tages-/Zeitfenster eingeplant.",
            "Bereits per Teams gemeldete Artikel und Teams-Themendubletten sind ausgeschlossen; Live-Pushes dienen nur dem Vergleich.",
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
        "requiredSlotCount": plan["requiredSlotCount"],
        "qualityOpportunityCount": plan["qualityOpportunityCount"],
        "items": plan["items"],
        "top5": plan["top5"],
        "qualitySummary": quality_summary,
        "trafficSlots": traffic_slots,
        "doubleOpportunities": double_opportunities,
        "watchTopics": watch_topics,
        "notRecommended": not_recommended,
    }
    if persist:
        _persist_daily_plan_recommendations(plan, decided_at_ts=now)
    return plan


def _persist_daily_plan_recommendations(plan: dict[str, Any], *, decided_at_ts: int) -> None:
    """Store generated day-plan suggestions for dashboard and audit history."""
    try:
        plan_date = str(plan.get("date") or "")
        for item in plan.get("items") or []:
            article_key = str(item.get("candidateId") or item.get("articleUrl") or "")
            if not article_key:
                continue
            record_seed = "|".join(
                [
                    "daily_plan",
                    plan_date,
                    str(int(item.get("slotTs") or 0)),
                    article_key,
                ]
            )
            reasons = [str(item.get("why") or "")]
            reasons.extend(str(reason) for reason in item.get("positiveReasons") or [])
            teams_recommendation_record(
                article_key=article_key,
                article_id=article_key,
                article_url=str(item.get("articleUrl") or ""),
                article_title=str(item.get("articleTitle") or item.get("title") or ""),
                section=str(item.get("section") or item.get("sectionLabel") or ""),
                recommendation_type="daily_plan",
                status=str(item.get("status") or ""),
                should_notify=str(item.get("status") or "") == "fix",
                score=float(item.get("score") or 0.0),
                teams_alert_score=float(item.get("teamsAlertScore") or 0.0),
                teams_alert_threshold=0.0,
                editorial_score=float(item.get("editorialScore") or 0.0),
                predicted_or=float(item.get("predictedOR") or 0.0),
                predicted_or_label=str(item.get("predictedORLabel") or ""),
                expected_visits=int(item.get("expectedOpens") or item.get("expectedVisits") or 0),
                dashboard_rank=int(item.get("dashboardRank") or 0),
                scheduled_for_ts=int(item.get("slotTs") or 0),
                decided_at_ts=int(decided_at_ts or time.time()),
                send_status="planned",
                summary=str(item.get("why") or ""),
                reasons=reasons,
                blocking_reasons=list(item.get("blockingReasons") or []),
                decision={
                    "planDate": plan_date,
                    "number": item.get("number"),
                    "time": item.get("time"),
                    "alternativeTime": item.get("alternativeTime"),
                    "priority": item.get("priority"),
                    "confidence": item.get("confidence"),
                    "fatigueRisk": item.get("fatigueRisk"),
                    "visitPotential": item.get("visitPotential"),
                    "urgency": item.get("urgency"),
                    "timingFit": item.get("timingFit"),
                    "planScore": item.get("planScore"),
                    "slotReason": item.get("slotReason"),
                    "pushText": item.get("pushText"),
                },
                record_id=hashlib.sha256(record_seed.encode("utf-8")).hexdigest(),
            )
    except Exception as exc:
        log.warning("[TeamsAlert] daily-plan persistence failed: %s", exc)


def build_teams_daily_push_plan_message(plan: dict[str, Any]) -> dict[str, str]:
    """Render a compact Teams-readable message for a daily push plan."""
    date_label = f"{plan.get('date')}, {plan.get('weekday')}"
    count = int(plan.get("count") or 0)
    minimum = int(plan.get("minimumItems") or 0)
    subject = f"Tagesplan Pushes für {date_label}: {count} Vorschläge"
    header = [
        f"Tagesplan Pushes für {date_label}",
        f"Ziel: {minimum} hochwertige Push-Chancen. Qualität wird nicht für die Menge abgesenkt.",
        _daily_plan_quality_sentence(plan.get("qualitySummary") or {}),
        "",
    ]
    lines: list[str] = [*header]
    for item in plan.get("items") or []:
        window_label = (
            "optionale Qualitätschance" if item.get("qualityOnly") else "regulaere :45-Entscheidung"
        )
        lines.extend(
            [
                f"{int(item.get('number') or 0)}. {item.get('time')} – {item.get('pushText')}",
                f"Ressort: {item.get('sectionLabel')} | Fenster: {window_label}",
                f"Priorität: {item.get('priority')} | Status: {item.get('status')}",
                (
                    f"Response-Potenzial: {item.get('visitPotential')}/10 "
                    f"(ca. {_format_int(item.get('expectedOpens') or 0)} Oeffnungen)"
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

    lines.extend(["", "Optionale Doppel-Chancen:"])
    doubles = plan.get("doubleOpportunities") or []
    if doubles:
        for slot in doubles:
            lines.append(
                f"- {slot.get('label')}: frueher Zusatz-Push nur bei Top-Kandidat; "
                f"bei mindestens zwei fehlenden Pushes auch als Aufholchance; "
                f"sonst bis {slot.get('deadline')} sammeln"
            )
    else:
        lines.append("- Heute keine belastbare Doppel-Chance ausserhalb von Breaking News.")

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


def build_teams_daily_schedule(
    target_date: str | dt.date | None = None,
    config: TeamsAlertConfig | None = None,
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Build the compact daily timing/section plan sent to Teams once a day."""
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    target = _parse_daily_plan_date(target_date, now)
    target_count = max(1, int(config.target_pushes_per_day or 15))
    slots = _daily_plan_slots(target, target_count, config)
    doubles = _daily_plan_double_opportunities(
        target,
        slots,
        config,
        limit=24,
    )
    minimum_double_count = len(doubles)
    optional_double_count = 0
    minimum_recovery_count = sum(1 for slot in slots if bool(slot.get("minimumRecovery")))
    runtime_opportunities = _daily_runtime_opportunities(target, config)
    coverage_count = len(runtime_opportunities)
    all_slots = _daily_plan_slot_candidates(target, config)
    selected_timestamps = {int(slot.get("ts") or 0) for slot in slots if slot.get("required")}
    omitted = [
        {
            "time": slot["label"],
            "avgOR": slot.get("avgOR"),
            "reason": "historisch schwaecher; nur Breaking oder aussergewoehnlicher Kandidat",
        }
        for slot in all_slots
        if int(slot.get("ts") or 0) not in selected_timestamps
    ]
    weekday = _weekday_name_de(target.weekday())
    subject = (
        f"Push-Fahrplan {weekday}, {target.isoformat()}: "
        f"{coverage_count} verbindliche Top-1-Entscheidungen"
    )
    lines = [
        subject,
        (
            f"Plan: {len(slots)} verbindliche Entscheidungsfenster, davon "
            f"{minimum_double_count} Doppelstunden und {minimum_recovery_count} "
            f"Recovery-Fenster; Mindestabdeckung {coverage_count}/{target_count}."
        ),
        "Regel: Um :15 gewinnt Top 1 nach frischem internem Push-Score. Um :45 "
        "werden Scores neu geladen, bereits empfohlene Artikel/Themen entfernt und "
        "Top 1 erneut bestimmt.",
        "Shortfall-Recovery: Reichen die verbleibenden Pflichtfenster rechnerisch "
        "nicht mehr für 15, wird am nächsten 30-Minuten-Cool-down-Rand geprüft.",
        "Der höchste gültige API-Score ab 75 gewinnt nach Fakten-, Aktualitäts-, "
        "Dubletten-, Titel- und Ruhezeitprüfung. Kein lokaler Fake-Score; Breaking darf sofort.",
        "",
        "Verbindliche Entscheidungsfenster:",
    ]
    for slot in slots:
        ressort = _format_section(str(slot.get("topCategory") or "News"))
        sport = f" | Sport: {slot['sportContext']}" if slot.get("sportContext") else ""
        lines.append(
            f"- {slot['label']} ({slot['tier']}): {ressort}, "
            f"historisch {_format_or(slot.get('avgOR'))}{sport}"
        )
    if doubles:
        lines.extend(["", "Verbindliche Doppelstunden:"])
        for slot in doubles:
            lines.append(f"- {slot['label']}: {slot['condition']}")
    if omitted:
        lines.extend(["", "Heute bewusst nachrangig:"])
        for slot in omitted:
            lines.append(f"- {slot['time']}: {slot['reason']}")
    text = "\n".join(lines)

    html_lines = [
        f"<p><strong>{html.escape(subject)}</strong></p>",
        (
            f"<p><strong>Plan:</strong> {len(slots)} verbindliche Entscheidungsfenster, "
            f"davon {minimum_double_count} Doppelstunden und "
            f"{minimum_recovery_count} Recovery-Fenster; Mindestabdeckung "
            f"{coverage_count}/{target_count}.</p>"
        ),
        (
            "<p><strong>Regel:</strong> Um :15 gewinnt Top 1 nach frischem internem "
            "Push-Score. Um :45 werden Scores neu geladen, Teams-Dubletten entfernt "
            "und Top 1 erneut bestimmt. Der höchste gültige API-Score ab 75 gewinnt "
            "nach allen harten Schutzprüfungen; ein lokaler Score-Fallback ist "
            "gesperrt. Breaking darf sofort.</p>"
        ),
        "<p><strong>Verbindliche Entscheidungsfenster</strong></p><ul>",
    ]
    for slot in slots:
        ressort = _format_section(str(slot.get("topCategory") or "News"))
        sport = f"; Sport: {slot['sportContext']}" if slot.get("sportContext") else ""
        html_lines.append(
            "<li><strong>"
            + html.escape(str(slot["label"]))
            + "</strong> ("
            + html.escape(str(slot["tier"]))
            + "): "
            + html.escape(ressort)
            + ", historisch "
            + html.escape(_format_or(slot.get("avgOR")))
            + html.escape(sport)
            + "</li>"
        )
    html_lines.append("</ul>")
    if doubles:
        html_lines.append("<p><strong>Verbindliche Doppelstunden</strong></p><ul>")
        for slot in doubles:
            html_lines.append(
                f"<li>{html.escape(str(slot['label']))}: "
                f"{html.escape(str(slot['condition']))}</li>"
            )
        html_lines.append("</ul>")
    if omitted:
        html_lines.append("<p><strong>Heute bewusst nachrangig</strong></p><ul>")
        for slot in omitted:
            html_lines.append(
                f"<li>{html.escape(str(slot['time']))}: " f"{html.escape(str(slot['reason']))}</li>"
            )
        html_lines.append("</ul>")
    message_html = "".join(html_lines)
    return {
        "type": "teams_daily_schedule",
        "date": target.isoformat(),
        "weekday": weekday,
        "targetPushes": target_count,
        "count": coverage_count,
        "requiredCount": len(slots),
        "qualityOpportunityCount": len(doubles),
        "minimumDoubleCount": minimum_double_count,
        "minimumRecoveryCount": minimum_recovery_count,
        "optionalDoubleCount": optional_double_count,
        "runtimeOpportunityCount": coverage_count,
        "meetsTargetCoverage": coverage_count >= target_count,
        "slots": slots,
        "doubleOpportunities": doubles,
        "deprioritizedSlots": omitted,
        "subject": subject,
        "messageText": text,
        "messageHtml": message_html,
        "payload": {
            "type": "push_daily_schedule",
            "subject": subject,
            "messageText": text,
            "messageHtml": message_html,
            "date": target.isoformat(),
            "weekday": weekday,
            "targetPushes": target_count,
            "count": coverage_count,
            "requiredCount": len(slots),
            "qualityOpportunityCount": len(doubles),
            "minimumDoubleCount": minimum_double_count,
            "minimumRecoveryCount": minimum_recovery_count,
            "optionalDoubleCount": optional_double_count,
            "runtimeOpportunityCount": coverage_count,
            "meetsTargetCoverage": coverage_count >= target_count,
            "slots": slots,
            "doubleOpportunities": doubles,
        },
    }


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
    editorial = (
        decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    )
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
    status = _daily_plan_status(
        priority, confidence, blockers, hard_blockers, fatigue_risk, decision
    )
    title = _title(candidate)
    push_text, push_source, push_title_review = _teams_push_title_selection(
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
        "teamsAlertScore": alert_score,
        "editorialScore": editorial_score,
        "pushText": push_text or title,
        "pushTitleSource": push_source,
        **(
            {"pushTitleReview": _public_push_title_review(push_title_review)}
            if config.agent_review_enabled
            else {}
        ),
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
    remaining = list(entries)
    assigned: list[dict[str, Any]] = []
    for slot in priority_slots:
        if not remaining:
            break
        entry = max(
            remaining,
            key=lambda candidate: _daily_plan_pair_score(candidate, slot),
        )
        remaining.remove(entry)
        alternative = _daily_plan_alternative_slot(slot, slots)
        assigned.append(_daily_plan_finalize_item(entry, slot, alternative, context, config))
    return assigned


def _daily_plan_pair_score(entry: dict[str, Any], slot: dict[str, Any]) -> float:
    """Rank an article-slot pairing, not only the article in isolation."""
    candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
    section = _section(candidate).lower()
    top_cat = str(slot.get("topCategory") or "").lower()
    section_fit = _slot_section_fit_delta(section, top_cat) * 9.0
    sport_fit = 0.0
    if section == "sport":
        sport = _sport_candidate_review(
            _title(candidate),
            int(slot.get("ts") or 0),
            candidate,
        )
        sport_fit = float(sport.get("timingDelta") or 0.0) * 6.0
        sport_fit += 10.0 if sport.get("eventful") else -18.0
    return (
        float(entry.get("planScore") or 0.0)
        + section_fit
        + sport_fit
        + float(slot.get("weight") or 0.0) * 2.0
    )


def _daily_plan_finalize_item(
    entry: dict[str, Any],
    slot: dict[str, Any],
    alternative: dict[str, Any] | None,
    context: dict[str, Any],
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    candidate = entry.get("candidate") or {}
    decision = entry.get("decision") if isinstance(entry.get("decision"), dict) else {}
    editorial = (
        decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    )
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
        reach_stats=context.get("reachStats")
        if isinstance(context.get("reachStats"), dict)
        else {},
        config=config,
    )
    time_fit = _time_fit_review(
        now_ts=int(slot["ts"]),
        section=_section(candidate),
        title=_title(candidate),
        candidate=candidate,
        breaking=_is_breaking(candidate),
        config=config,
        pushes_today=(
            context.get("teamsAlertsToday")
            if config.independent_pacing_enabled
            else context.get("pushesToday")
        ),
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
            "requiredSlot": bool(slot.get("required")),
            "qualityOnly": bool(slot.get("qualityOnly")),
            "doubleOpportunity": bool(slot.get("doubleOpportunity")),
            "alternativeTime": (
                f"{alternative['label']} Uhr" if alternative else "Lageabhängig prüfen"
            ),
            "forecast": forecast,
            "predictedOR": forecast.get("value"),
            "predictedORLabel": _format_forecast(forecast),
            "expectedOpens": int(visit.get("expectedOpens") or 0),
            "expectedVisits": int(visit.get("expectedOpens") or 0),
            "responseMetric": "expected_opens",
            "estimatedReach": int(visit.get("estimatedReach") or 0),
            "visitPotential": _score_to_ten(float(visit.get("score") or 0.0)),
            "timingFit": _score_to_ten(float(time_fit.get("score") or 0.0) * 10.0),
            "timingLabel": time_fit.get("label"),
            "blockingReasons": list(decision.get("blockingReasons") or []),
            "positiveReasons": list(decision.get("reasons") or []),
            "livePushComparison": dict(decision.get("livePushComparison") or {}),
        }
    )
    item["why"] = _daily_plan_reason(
        candidate, decision, str(item.get("status") or ""), slot, visit
    )
    return item


def _daily_plan_hard_blockers(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    config: TeamsAlertConfig,
) -> list[str]:
    section = _section_key(_section(candidate))
    excluded = {item.lower() for item in config.excluded_sections if item.strip()}
    hard: list[str] = []
    review = decision.get("agentReview") if isinstance(decision.get("agentReview"), dict) else {}
    if review.get("enabled") and int(review.get("hardVetoCount") or 0) > 0:
        hard.append(str(review.get("blockingReason") or "Agenten-Veto"))
    non_article_reason = _daily_plan_non_article_reason(candidate)
    if non_article_reason:
        hard.append(non_article_reason)
    if not _title(candidate):
        hard.append("Keine Headline")
    if not _url(candidate):
        hard.append("Kein Artikel-Link")
    if section in excluded:
        hard.append(f"Ressort {_format_section(section)} ist ausgeschlossen")
    hard_markers = (
        "Bereits per Teams gemeldet",
        "Thema bereits per Teams gemeldet",
        "Teams-Hinweis wird bereits versendet",
        "Bereits als Teams-Kandidat versucht",
        "Artikel-Link",
        "Headline",
        "Agenten-Veto",
        "Zeitstempel",
        "Veroeffentlichungs- oder Aktualisierungszeit fehlt",
        "Artikel ist zu alt",
        "kein konkretes Nachrichten-Ereignis",
        "Sport ohne ",
        "Faktenrisiko",
        "Deutschland-Relevanz",
    )
    for reason in decision.get("blockingReasons") or []:
        text = str(reason or "")
        if any(marker in text for marker in hard_markers):
            hard.append(text)
    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if allowed and section not in allowed:
        hard.append(f"Ressort {_format_section(section)} ist nicht freigegeben")
    return _dedupe(hard)


def _daily_plan_non_article_reason(candidate: dict[str, Any]) -> str:
    """Exclude meta/profile pages from the editorial day plan.

    The recommendations feed can contain author/profile pages. Their reach can
    look attractive to the response model, but they are not actionable articles and
    must never occupy one of the minimum daily push slots.
    """
    from urllib.parse import urlsplit

    url = _url(candidate)
    if not url:
        return ""
    path = urlsplit(url).path.lower().strip("/")
    if not path:
        return "Kein Artikel: Start-/Übersichtsseite"
    segments = [segment for segment in path.split("/") if segment]
    first_segment = segments[0] if segments else ""
    meta_roots = {
        "autor",
        "autoren",
        "suche",
        "newsletter",
        "impressum",
        "datenschutz",
        "agb",
        "kontakt",
    }
    if first_segment in meta_roots:
        return "Kein Artikel: Autor-/Meta-Seite"
    if "/autor/" in f"/{path}/" or "/autoren/" in f"/{path}/":
        return "Kein Artikel: Autor-/Meta-Seite"
    if first_segment in {"thema", "themen", "tag", "tags"}:
        return "Kein Artikel: Themen-/Tag-Seite"
    return ""


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
        raw_score * 0.35
        + editorial_score * 0.20
        + alert_score * 0.15
        + visit_score * 0.20
        + time_score * 0.06
        + urgency_score * 0.04
    )
    blockers = [str(reason) for reason in decision.get("blockingReasons") or []]
    soft_penalty = min(18.0, len(blockers) * 3.0)
    if any("Kurios-/Click-Reiz" in reason for reason in blockers):
        soft_penalty += 10.0
    if any(
        "Service-/Raetsel" in reason or "kein konkretes Nachrichten-Ereignis" in reason
        for reason in blockers
    ):
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
    editorial = (
        decision.get("editorialReview") if isinstance(decision.get("editorialReview"), dict) else {}
    )
    approved = bool(editorial.get("approved", False))
    if priority == "A" and approved and source == "article_model":
        return "hoch"
    if priority in {"A", "B"} and (source == "article_model" or forecast_confidence >= 0.45):
        return "mittel"
    return "niedrig"


def _daily_plan_fatigue_risk(decision: dict[str, Any], config: TeamsAlertConfig) -> str:
    independent = bool(decision.get("recommendationsIndependentFromLivePushes"))
    minutes = (
        decision.get("minutesSinceLastGlobalTeamsAlert")
        if independent
        else decision.get("minutesSinceLastPush")
    )
    recent_pushes = 0 if independent else _safe_int(decision.get("recentPushCount6h"))
    pushes_today = _safe_int(
        decision.get("teamsAlertsToday") if independent else decision.get("pushesToday")
    )
    if isinstance(minutes, (int, float)) and minutes < max(20, config.min_minutes_since_last_push):
        return "hoch"
    if recent_pushes > config.max_pushes_last_6h or pushes_today >= max(
        1, config.target_pushes_per_day
    ):
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
    expected_opens = _safe_int(visit_data.get("expectedOpens") or visit_data.get("expectedVisits"))
    predicted_or = _safe_float(visit_data.get("predictedOR"))
    visit_reason = ""
    if expected_opens > 0:
        visit_reason = f"Response-Potenzial ca. {_format_int(expected_opens)} Oeffnungen"
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
    """Build 15-18 binding slots from the weekday matrix.

    06:15/06:45 are always present. Every red/yellow hour contributes both a
    :15 Top-1 decision and a :45 re-ranking. Strong :45 reserve slots fill the
    day to at least 15; the 10/11 o'clock dead zone is used only as a last resort.
    """
    requested = max(1, int(count or config.target_pushes_per_day or 15))
    maximum = max(
        requested,
        int(config.max_alerts_per_day or config.daily_plan_max_items or 18),
    )
    candidates = _daily_plan_slot_candidates(target_date, config)
    peak = [slot for slot in candidates if slot.get("mustUse")]
    reserve = [
        slot
        for slot in candidates
        if not slot.get("mustUse") and float(slot.get("avgOR") or 0.0) >= 5.0
    ]
    recovery = [
        slot
        for slot in candidates
        if not slot.get("mustUse") and slot not in reserve and bool(slot.get("hasWeekdayData"))
    ]
    unmeasured = [
        slot
        for slot in candidates
        if not slot.get("mustUse") and slot not in reserve and not bool(slot.get("hasWeekdayData"))
    ]
    rank_key = lambda slot: (
        float(slot.get("weight") or 0.0),
        float(slot.get("slotScore") or 0.0),
        float(slot.get("avgOR") or 0.0),
        -int(slot.get("hour") or 0),
    )
    desired = min(maximum, max(requested, len(peak)))
    selected = sorted(peak, key=rank_key, reverse=True)[:maximum]
    recovery_timestamps: set[int] = set()
    avoid_hours = {10, 11}
    preferred_reserve = [slot for slot in reserve if int(slot.get("hour") or -1) not in avoid_hours]
    preferred_recovery = [
        slot for slot in recovery if int(slot.get("hour") or -1) not in avoid_hours
    ]
    preferred_unmeasured = [
        slot for slot in unmeasured if int(slot.get("hour") or -1) not in avoid_hours
    ]
    last_resort = [
        slot
        for slot in (*reserve, *recovery, *unmeasured)
        if int(slot.get("hour") or -1) in avoid_hours
    ]
    for pool, minimum_recovery in (
        (preferred_reserve, False),
        (preferred_recovery, True),
        (preferred_unmeasured, True),
        (last_resort, True),
    ):
        for slot in sorted(pool, key=rank_key, reverse=True):
            if len(selected) >= desired:
                break
            selected.append(slot)
            if minimum_recovery:
                recovery_timestamps.add(int(slot.get("ts") or 0))
        if len(selected) >= desired:
            break

    selected = sorted(selected, key=lambda slot: int(slot.get("ts") or 0))
    for index, slot in enumerate(selected, start=1):
        slot["planOrder"] = index
        slot.setdefault("required", True)
        slot.setdefault("qualityOnly", False)
        slot["minimumRequired"] = True
        slot["minimumRecovery"] = int(slot.get("ts") or 0) in recovery_timestamps
        if slot["minimumRecovery"]:
            slot["reason"] = (
                f"{slot.get('reason') or 'historisch nachrangiges Fenster'}, "
                "Recovery-Fenster fuer das Tagesminimum; nur bester freigegebener "
                "Kandidat ab Push-Score 75"
            )
    return selected


def _daily_plan_slot_candidates(
    target_date: dt.date,
    config: TeamsAlertConfig,
) -> list[dict[str, Any]]:
    weekday = target_date.weekday()
    start = int(_clamp(config.active_hours_start, 0, 23))
    end = int(_clamp(config.active_hours_end, start, 23))
    deadline = int(_clamp(config.slot_deadline_minute, 0, 59))
    candidates: list[dict[str, Any]] = []
    for hour in range(start, end + 1):
        deadline_slot = _daily_plan_slot(target_date, hour, deadline, weekday, config)
        is_golden = bool(deadline_slot.get("mustUse"))
        is_morning_base = hour == 6
        minutes = sorted({15, deadline}) if (is_golden or is_morning_base) else [deadline]
        for minute in minutes:
            slot = _daily_plan_slot(target_date, hour, minute, weekday, config)
            pair_required = is_golden or is_morning_base
            slot["goldenHour"] = is_golden
            slot["morningBase"] = is_morning_base
            slot["pairRequired"] = pair_required
            slot["pairSequence"] = (
                "first_top_1" if pair_required and minute == 15 else "fresh_rerank"
            )
            if pair_required:
                slot["mustUse"] = True
                if is_morning_base and not is_golden:
                    slot["tier"] = "basis"
                action = "erste Top-1-Entscheidung" if minute == 15 else "frisches Top-1-Re-Ranking"
                slot["reason"] = f"{slot['reason']}, verbindliche {action}"
            candidates.append(slot)
    return candidates


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
    has_weekday_data = int(baseline.get("count") or 0) > 0 if baseline else False
    stars = int(baseline.get("stars") or 0) if baseline else 0
    slot_score = _slot_baseline_score(hour, weekday, baseline, breaking=False)
    reason_parts = []
    if avg_or and has_weekday_data:
        reason_parts.append(f"historisch {_format_number(avg_or, 2)} % OR")
    elif not has_weekday_data:
        reason_parts.append("keine belastbare Wochentagszelle")
    if stars >= 2:
        reason_parts.append("starker historischer Slot")
    top_cat = str(baseline.get("top_cat") or "").strip()
    if top_cat:
        reason_parts.append(f"Top-Ressort {_format_section(top_cat)}")
    peak_min = float(config.peak_slot_min_or or 6.0)
    if has_weekday_data and avg_or is not None and avg_or >= 6.4:
        tier = "rot"
    elif has_weekday_data and avg_or is not None and avg_or >= peak_min:
        tier = "gelb"
    elif has_weekday_data and avg_or is not None and avg_or >= 5.3:
        tier = "reserve"
    else:
        tier = "fallback"
    sport_context = _sport_schedule_context(target_date, hour)
    preferred_sections = [top_cat] if top_cat else []
    if sport_context:
        preferred_sections.append("sport")
    return {
        "ts": int(slot_dt.timestamp()),
        "label": f"{hour:02d}:{minute:02d}",
        "hour": hour,
        "minute": minute,
        "weekday": weekday,
        "weight": _slot_weight(hour, weekday, config),
        "slotScore": round(slot_score, 1),
        "avgOR": round(avg_or, 2) if avg_or is not None and has_weekday_data else None,
        "hasWeekdayData": has_weekday_data,
        "stars": stars,
        "topCategory": top_cat or None,
        "preferredSections": _dedupe(preferred_sections),
        "sportContext": sport_context or None,
        "tier": tier,
        "mustUse": tier in {"rot", "gelb"},
        "required": True,
        "qualityOnly": False,
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
    slots = _daily_plan_slot_candidates(target_date, config)
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


def _daily_plan_double_opportunities(
    target_date: dt.date,
    planned_slots: list[dict[str, Any]],
    config: TeamsAlertConfig,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    by_hour: dict[int, list[dict[str, Any]]] = {}
    for slot in planned_slots:
        if slot.get("pairRequired"):
            by_hour.setdefault(int(slot.get("hour") or 0), []).append(slot)

    result: list[dict[str, Any]] = []
    cooldown = _effective_global_cooldown_minutes(config)
    for hour in sorted(by_hour):
        pair = sorted(by_hour[hour], key=lambda item: int(item.get("minute") or 0))
        labels = {str(item.get("label") or "") for item in pair}
        if f"{hour:02d}:15" not in labels or f"{hour:02d}:45" not in labels:
            continue
        reference = pair[-1]
        result.append(
            {
                "hour": hour,
                "label": f"{hour:02d}:15 + {hour:02d}:45",
                "deadline": f"{hour:02d}:45",
                "avgOR": reference.get("avgOR"),
                "topCategory": reference.get("topCategory"),
                "sportContext": reference.get("sportContext"),
                "requiredForMinimum": True,
                "cooldownCompatible": cooldown <= 30,
                "condition": (
                    "Top 1 um :15; um :45 Scores neu laden, Teams-Dubletten entfernen "
                    "und die dann hoechste gueltige API-Score-Meldung empfehlen"
                ),
            }
        )
    return result[:limit]


def _daily_plan_assignment_slots(
    required_slots: list[dict[str, Any]],
    double_opportunities: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Add clearly marked early quality chances to a planning-only slot list."""
    selected = [dict(slot) for slot in required_slots]
    if len(selected) >= limit:
        return selected[:limit]

    by_hour = {int(slot.get("hour") or -1): slot for slot in required_slots}
    for opportunity in double_opportunities:
        if len(selected) >= limit:
            break
        hour = int(opportunity.get("hour") or -1)
        source = by_hour.get(hour)
        if not source:
            continue
        early_dt = dt.datetime.fromtimestamp(
            int(source.get("ts") or 0), ZoneInfo("Europe/Berlin")
        ).replace(minute=0, second=0, microsecond=0)
        selected.append(
            {
                **source,
                "ts": int(early_dt.timestamp()),
                "label": f"{hour:02d}:00",
                "mustUse": False,
                "required": False,
                "qualityOnly": True,
                "doubleOpportunity": True,
                "reason": (
                    f"optionale fruehe Qualitätschance vor {source['label']}; "
                    "nur mit voller Freigabe und eingehaltenem Cooldown"
                ),
            }
        )

    selected.sort(key=lambda slot: int(slot.get("ts") or 0))
    for index, slot in enumerate(selected, start=1):
        slot["planOrder"] = index
    return selected[:limit]


def _daily_runtime_opportunities(
    target_date: dt.date,
    config: TeamsAlertConfig,
) -> list[dict[str, Any]]:
    """Return all binding :15/:45 decisions for the Berlin calendar day."""
    target = max(1, int(config.target_pushes_per_day or 15))
    selected = _daily_plan_slots(target_date, target, config)
    opportunities: list[dict[str, Any]] = []
    for slot in selected:
        item = dict(slot)
        item["minimumRequired"] = True
        item["minimumDouble"] = bool(item.get("pairRequired"))
        if item.get("morningBase"):
            item["slotKind"] = "morning_base"
        elif item.get("goldenHour"):
            item["slotKind"] = "golden_hour"
        elif item.get("minimumRecovery"):
            item["slotKind"] = "minimum_recovery"
        else:
            item["slotKind"] = "regular"
        item["deadlineMinute"] = int(item.get("minute") or 45)
        opportunities.append(item)
    return sorted(opportunities, key=lambda item: int(item.get("ts") or 0))


def _daily_plan_watch_topics(
    remaining: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    pool = remaining or [
        entry
        for entry in raw_entries
        if not entry.get("hardBlockers") and entry.get("priority") in {"B", "C"}
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


def _daily_plan_quality_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    fixed = sum(1 for item in items if item.get("status") == "fix")
    optional = sum(1 for item in items if item.get("status") == "optional")
    quiet = sum(1 for item in items if item.get("status") == "nur bei ruhiger Nachrichtenlage")
    high_confidence = sum(1 for item in items if item.get("confidence") == "hoch")
    medium_confidence = sum(1 for item in items if item.get("confidence") == "mittel")
    low_confidence = sum(1 for item in items if item.get("confidence") == "niedrig")
    return {
        "fix": fixed,
        "optional": optional,
        "quietOnly": quiet,
        "highConfidence": high_confidence,
        "mediumConfidence": medium_confidence,
        "lowConfidence": low_confidence,
    }


def _daily_plan_quality_sentence(summary: dict[str, Any]) -> str:
    fix = _safe_int(summary.get("fix"))
    optional = _safe_int(summary.get("optional"))
    quiet = _safe_int(summary.get("quietOnly"))
    sentence = f"Qualität: {fix} fix, {optional} optional, {quiet} nur bei ruhiger Nachrichtenlage."
    if fix <= 0:
        sentence += " Achtung: aktuell kein fixer Top-Push im Feld."
    return sentence


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
        "expectedOpens": item.get("expectedOpens"),
        "expectedVisits": item.get("expectedVisits"),
        "responseMetric": "expected_opens",
        "why": item.get("why"),
    }


def _build_teams_daily_plan_html(subject: str, plan: dict[str, Any]) -> str:
    items_html = "".join(
        "<li>"
        f"<strong>{html.escape(str(item.get('time') or ''))} – "
        f"{html.escape(str(item.get('pushText') or ''))}</strong><br>"
        f"Ressort: {html.escape(str(item.get('sectionLabel') or ''))} | "
        f"Fenster: {'optionale Qualitätschance' if item.get('qualityOnly') else 'regulaere :45-Entscheidung'} | "
        f"Priorität: {html.escape(str(item.get('priority') or ''))} | "
        f"Status: {html.escape(str(item.get('status') or ''))}<br>"
        f"Response-Potenzial: {html.escape(str(item.get('visitPotential') or ''))}/10 | "
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
        f"<p>Ziel: {int(plan.get('minimumItems') or 0)} hochwertige Push-Chancen. "
        "Qualität wird nicht für die Menge abgesenkt; frühe Doppelchancen sind optional.<br>"
        f"{html.escape(_daily_plan_quality_sentence(plan.get('qualitySummary') or {}))}</p>"
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
    for key in (
        "estimatedReach",
        "expectedReach",
        "estimatedRecipients",
        "recipients",
        "recipientCount",
    ):
        value = _safe_int(candidate.get(key))
        if value > 0:
            return value
    return 0


def _audience_breadth_adjustment(candidate: dict[str, Any], *, breaking: bool) -> dict[str, Any]:
    """Estimate how broad the reachable push audience is for this story.

    OR alone rewards clicky narrow stories. This factor keeps the response objective
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

    global_median = _safe_float(reach_stats.get("globalMedian")) or float(
        config.default_reach or 250000
    )
    by_section = (
        reach_stats.get("bySection") if isinstance(reach_stats.get("bySection"), dict) else {}
    )
    by_hour = reach_stats.get("byHour") if isinstance(reach_stats.get("byHour"), dict) else {}
    section_counts = (
        reach_stats.get("sectionCounts")
        if isinstance(reach_stats.get("sectionCounts"), dict)
        else {}
    )
    hour_counts = (
        reach_stats.get("hourCounts") if isinstance(reach_stats.get("hourCounts"), dict) else {}
    )

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
    """Estimate expected push opens; legacy name retained for API compatibility."""
    reach = _estimated_reach(
        candidate,
        now_ts=now_ts,
        reach_stats=reach_stats,
        breaking=breaking,
        config=config,
    )
    or_value = max(0.0, float(predicted_or or 0.0))
    expected_opens = float(reach["value"]) * or_value / 100.0
    quality_factor = _clamp(
        0.55
        + max(0.0, editorial_score - 60.0) / 100.0
        + max(0.0, alert_score - 60.0) / 180.0
        + max(0.0, score - 70.0) / 300.0,
        0.55,
        1.20,
    )
    quality_adjusted_opens = expected_opens * quality_factor
    global_reach = _safe_float(reach_stats.get("globalMedian")) or float(
        config.default_reach or 250000
    )
    benchmark = max(2500.0, global_reach * max(float(config.min_or or 5.0), 4.5) / 100.0)
    ratio = quality_adjusted_opens / benchmark if benchmark > 0 else 0.0
    response_score = _clamp(45.0 + ratio * 35.0, 0.0, 100.0)
    reason = ""
    if or_value > 0:
        audience_label = str(reach.get("audienceLabel") or "").strip()
        audience_suffix = f"; {audience_label}" if audience_label else ""
        reason = (
            "Response-Potenzial: ca. "
            f"{_format_int(expected_opens)} erwartete Push-Oeffnungen "
            f"({_format_number(or_value, 2)} % OR x ca. {_format_int(float(reach['value']))} Reichweite"
            f"{audience_suffix})"
        )
    return {
        "metric": "expected_opens",
        "expectedOpens": int(round(expected_opens)),
        "qualityAdjustedOpens": int(round(quality_adjusted_opens)),
        # Keep old keys until consumers migrate; their values are openings.
        "expectedVisits": int(round(expected_opens)),
        "qualityAdjustedVisits": int(round(quality_adjusted_opens)),
        "estimatedReach": int(round(float(reach["value"]))),
        "predictedOR": round(or_value, 2) if or_value else None,
        "score": round(response_score, 1),
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
    germany_selection_adjustment: float = 0.0,
    config: TeamsAlertConfig | None = None,
) -> float:
    """Rank eligible candidates with raw Push Score as the primary signal."""
    config = config or TeamsAlertConfig()
    forecast_score = _clamp(float(predicted_or or 0.0) * 10.0, 0.0, 100.0)
    rank_bonus = max(0.0, 1.5 - max(0, dashboard_rank - 1) * 0.15) if dashboard_rank > 0 else 0.0
    # Verified breaking can narrowly beat a stronger routine candidate, but the
    # bonus is too small to rescue a weak raw Push Score past its hard floor.
    breaking_bonus = 3.0 if breaking else 0.0
    editorial_total = (
        score * 0.82 + editorial_score * 0.07 + alert_score * 0.08 + forecast_score * 0.03
    )
    if config.visit_optimization_enabled:
        response_weight = _clamp(float(config.visit_selection_weight or 0.0), 0.0, 0.10)
        total = (
            editorial_total * (1.0 - response_weight) + float(visit_score or 0.0) * response_weight
        )
    else:
        total = editorial_total
    relevance_adjustment = _clamp(germany_selection_adjustment, -20.0, 7.0)
    return round(
        _clamp(
            total + rank_bonus + breaking_bonus + relevance_adjustment,
            0.0,
            100.0,
        ),
        1,
    )


def _forecast_quality_review(
    candidate: dict[str, Any],
    forecast: dict[str, Any],
    alert_score: float,
    breaking: bool,
    config: TeamsAlertConfig,
) -> dict[str, list[str]]:
    if not config.require_article_forecast:
        return {"reasons": [], "blockers": []}

    source = str(forecast.get("source") or "")
    if source == "article_model":
        return {"reasons": ["Belastbare Artikel-Prognose vorhanden"], "blockers": []}
    if breaking and config.breaking_override:
        return {"reasons": ["Breaking-Override: Slot-Prognose nur Timing-Kontext"], "blockers": []}
    if (
        _has_hard_public_need(_title(candidate), _section(candidate))
        and alert_score >= config.no_forecast_min_alert_score
    ):
        return {
            "reasons": ["Öffentliche Warn-/Nutzwertlage: auch ohne Artikelmodell prüfbar"],
            "blockers": [],
        }
    if (
        is_german_public_figure_parenthood_story(candidate)
        and alert_score >= config.no_forecast_min_alert_score
    ):
        return {
            "reasons": [
                "Bestaetigtes Deutschland-People-Ereignis: hoher Push- und Alert-Score "
                "erlauben die Prüfung ohne Artikelforecast"
            ],
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
) -> dict[str, list[str]]:
    pacing_basis = str(push_pacing.get("basis") or "actual_pushes")
    stock_label = "Teams-Empfehlungsstand" if pacing_basis == "teams_alerts" else "Push-Bestand"
    if breaking:
        return {
            "reasons": [f"Tagesstrategie: Breaking darf {stock_label} übersteuern"],
            "blockers": [],
        }
    if not push_pacing.get("known"):
        return {"reasons": [], "blockers": []}

    surplus = float(push_pacing.get("surplus") or 0.0)
    deficit = float(push_pacing.get("deficit") or 0.0)
    blockers: list[str] = []
    reasons: list[str] = []
    strong_enough_when_ahead = (
        alert_score >= config.min_alert_score + 8.0
        and editorial_score >= config.min_editorial_score + 8.0
        and news_value >= config.min_editorial_news_value + 8.0
        and (predicted_or is None or predicted_or >= config.min_or + 0.5)
    )
    if surplus >= 2.0 and not strong_enough_when_ahead:
        blockers.append(
            f"Tagesstrategie: {stock_label} liegt vorn; normale Lage nicht stark genug "
            "für einen zusätzlichen Teams-Hinweis"
        )
    elif surplus >= 2.0:
        reasons.append("Tagesstrategie: trotz Push-Vorsprung stark genug")
    elif deficit >= 1.5:
        reasons.append(
            f"Tagesstrategie: Rueckstand beim {stock_label}; Qualitaet wird bis zur "
            "faelligen :45-Entscheidung priorisiert"
        )
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
    basis_key = str(push_pacing.get("basis") or "actual_pushes")
    count_today = push_pacing.get("countToday")
    count_known = count_today is not None
    current = _safe_int(count_today) if count_known else _safe_int(teams_alerts_today)
    basis = "Teams-Hinweise" if basis_key == "teams_alerts" else "realer Push-Bestand"
    deficit = max(0.0, expected - current)
    late_day_floor = 0.0
    if hour >= 18:
        late_day_floor = max(0.0, minimum * 0.72 - current)
    if hour >= 21:
        late_day_floor = max(late_day_floor, minimum * 0.9 - current)
    pressure = max(deficit, late_day_floor)
    active = pressure >= 1.0
    threshold_drop = 0.0
    if not active:
        label = f"Teams-Mindest-Pacing: {basis} im Plan fuer mindestens {minimum} Hinweise"
    else:
        label = (
            f"Teams-Mindest-Pacing aktiv ({basis}): Rueckstand {pressure:.1f} auf mindestens "
            f"{minimum} Hinweise; zusaetzliche gute Slots ja, niedrigere Qualitaet nein"
        )
    return {
        "active": active,
        "minimum": minimum,
        "current": current,
        "basis": basis_key if count_known else "teams_alerts",
        "actualPushesToday": push_pacing.get("actualPushesToday"),
        "teamsAlertsToday": _safe_int(teams_alerts_today),
        "expectedByNow": round(expected, 2),
        "pressure": round(pressure, 2),
        "thresholdDrop": round(threshold_drop, 1),
        "label": label,
    }


def _expanded_field_candidate_review(
    candidate: dict[str, Any],
    *,
    score: float,
    alert_score: float,
    min_alert_score: float,
    predicted_or: float | None,
    dashboard_rank: int,
    dashboard_top_limit: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Allow exceptional response candidates beyond the visible dashboard top field."""
    if dashboard_rank <= 0 or dashboard_rank <= dashboard_top_limit:
        return {"allowed": False, "reason": ""}

    candidate_limit = max(
        dashboard_top_limit, int(config.candidate_limit or PUSH_TEAMS_CANDIDATE_LIMIT)
    )
    if dashboard_rank > candidate_limit:
        return {
            "allowed": False,
            "reason": f"Rang {dashboard_rank} liegt ausserhalb des erweiterten Kandidatenfelds",
        }

    title = _title(candidate)
    section = _section_key(_section(candidate))
    if section in {item.lower() for item in config.excluded_sections if item.strip()}:
        return {"allowed": False, "reason": ""}

    if not _has_expanded_field_visit_pattern(title, section):
        return {"allowed": False, "reason": ""}

    min_score = max(float(config.min_score or 0.0), 75.0)
    if score < min_score:
        return {"allowed": False, "reason": ""}

    if alert_score < max(74.0, min_alert_score - 2.0):
        return {"allowed": False, "reason": ""}

    if (
        predicted_or is not None
        and predicted_or < max(4.5, float(config.min_or or 0.0) - 0.5)
        and not _public_money_fraud_or_near_miss(
            title=title,
            predicted_or=predicted_or,
            min_or=float(config.min_or or 0.0),
            alert_score=alert_score,
            min_alert_score=min_alert_score,
        )
        and not _celebrity_conflict_or_near_miss(
            title=title,
            section=section,
            predicted_or=predicted_or,
            min_or=float(config.min_or or 0.0),
            alert_score=alert_score,
            min_alert_score=min_alert_score,
        )
    ):
        return {"allowed": False, "reason": ""}

    if (
        _is_soft_service_or_quiz(title)
        or _is_nonessential_curiosity(title)
        or _is_abstract_explainer_without_update(title)
        or _is_scheduled_process_without_update(title)
        or _is_low_civic_impact_story(title, section)
    ) and not _is_urgent_public_service_title(title):
        return {"allowed": False, "reason": ""}

    return {
        "allowed": True,
        "reason": (
            "Expanded Field: starker Response-/Public-Need-Kandidat ausserhalb der Top "
            f"{dashboard_top_limit} (Rang {dashboard_rank} von {candidate_limit})"
        ),
    }


def _has_expanded_field_visit_pattern(title: str, section: str = "") -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if _is_public_money_fraud_enforcement(text):
        return True
    if _is_celebrity_relationship_money_conflict(text, section):
        return True
    if _is_urgent_public_service_title(text):
        return True
    if is_german_public_figure_parenthood_story({"title": title, "category": section}):
        return True
    hard_visit_terms = (
        "terror",
        "anschlag",
        "explosion",
        "brand",
        "evakuierung",
        "vermisst",
        "tote",
        "verletzte",
        "festnahme",
        "festgenommen",
        "razzia",
        "großrazzia",
        "grossrazzia",
        "leistungsbetrug",
        "sozialbetrug",
        "bürgergeld",
        "buergergeld",
        "betrug",
        "warnung",
        "gefahr",
        "streik",
        "ausfall",
        "sperrung",
        "funkstörung",
        "funkstoerung",
        "totalausfall",
        "blackout",
        "rückruf",
        "rueckruf",
    )
    return any(_contains_editorial_term(text, term) for term in hard_visit_terms)


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
    section = _section_key(_section(candidate))
    breaking = _is_breaking(candidate)
    people_parenthood = is_german_public_figure_parenthood_story(candidate)
    sport_review = _sport_candidate_review(title, now_ts, candidate) if section == "sport" else {}
    rank_limit = max(
        1, min(int(config.editorial_top_limit or 10), int(config.dashboard_top_limit or 20))
    )

    high_impact_terms = (
        "krieg",
        "terror",
        "anschlag",
        "iran",
        "israel",
        "ukraine",
        "russland",
        "putin",
        "trump",
        "merz",
        "kanzler",
        "regierung",
        "bundestag",
        "polizei",
        "tote",
        "tot",
        "vermisst",
        "verletzte",
        "gefahr",
        "warnung",
        "evakuierung",
        "ruecktritt",
        "rücktritt",
        "feuerpause",
        "atom",
        "nato",
        "gericht",
        "urteil",
        "streik",
        "insolvenz",
        "festnahme",
        "razzia",
        "großrazzia",
        "grossrazzia",
        "betrug",
        "betrüger",
        "betrueger",
        "leistungsbetrug",
        "leistungsbetrüger",
        "leistungsbetrueger",
        "sozialbetrug",
        "bürgergeld",
        "buergergeld",
        "polizist",
        "polizisten",
        "scheidung",
        "scheidungszoff",
        "trennung",
        "ehe-aus",
        "unterhalt",
        "vermögen",
        "vermoegen",
        "wm-held",
        "weltmeister",
        "schweini",
        "schweinsteiger",
    )
    public_need_terms = (
        "warnung",
        "gefahr",
        "polizei",
        "streik",
        "ausfall",
        "sperrung",
        "rueckruf",
        "rückruf",
        "steuer",
        "rente",
        "krankenkasse",
        "geld",
        "preis",
        "verbraucher",
        "gericht",
        "urteil",
        "regierung",
        "bundestag",
        "nato",
        "krieg",
        "feuerpause",
        "razzia",
        "großrazzia",
        "grossrazzia",
        "betrug",
        "betrüger",
        "betrueger",
        "leistungsbetrug",
        "leistungsbetrüger",
        "leistungsbetrueger",
        "sozialbetrug",
        "bürgergeld",
        "buergergeld",
        "polizist",
        "polizisten",
    )
    soft_terms = (
        "quiz",
        "horoskop",
        "shopping",
        "rabatt",
        "sommertrend",
        "fans",
        "star",
        "stars",
        "app",
        "promi",
        "liebe",
        "beauty",
        "mode",
        "urlaub",
        "reise",
        "peinlich",
        "witzig",
    )
    vague_terms = ("diese", "dieser", "darum", "so ", "jetzt wissen", "experte erklaert")

    section_points = {
        "politik": 30.0,
        "news": 28.0,
        "wirtschaft": 23.0,
        "regional": 22.0,
        "sport": 24.0,
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
    if _is_celebrity_relationship_money_conflict(title, section):
        impact_bonus += 16.0
    if people_parenthood:
        impact_bonus += 18.0
    if sport_review:
        impact_bonus += 8.0 if sport_review.get("eventful") else -12.0
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
    if low_civic_impact and not breaking and not sport_review.get("eventful"):
        impact_bonus -= 12.0
    soft_matches = [term for term in soft_terms if term in title_l]
    if (
        soft_matches
        and not impact_matches
        and not breaking
        and not _is_celebrity_relationship_money_conflict(title, section)
        and not sport_review.get("eventful")
    ):
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
    if sport_review.get("eventful"):
        user_need += 2.0
    if people_parenthood:
        user_need += 2.0
    if (
        soft_matches
        and not need_matches
        and not impact_matches
        and not _is_celebrity_relationship_money_conflict(title, section)
        and not sport_review.get("eventful")
    ):
        user_need -= 4.0
    if low_civic_impact and not breaking and not sport_review.get("eventful"):
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
        title=title,
        candidate=candidate,
        breaking=breaking,
        config=config,
        pushes_today=pushes_today,
    )
    pacing = _push_pacing_review(
        pushes_today,
        now_ts,
        config,
        basis=("teams_alerts" if config.independent_pacing_enabled else "actual_pushes"),
    )

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
    if people_parenthood:
        reasons.append(
            "CvD-People-Signal: benannte deutsche oeffentliche Person, bestaetigte "
            "Elternschaft, positive Überraschung"
        )

    minimum_pressure = minimum_pressure or {}
    min_editorial_score = float(config.min_editorial_score)
    expanded_field = _expanded_field_candidate_review(
        candidate,
        score=score,
        alert_score=alert_score,
        min_alert_score=config.min_alert_score,
        predicted_or=predicted_or,
        dashboard_rank=dashboard_rank,
        dashboard_top_limit=int(config.dashboard_top_limit or PUSH_TEAMS_CANDIDATE_LIMIT),
        config=config,
    )

    if dashboard_rank > rank_limit and not breaking:
        if expanded_field["allowed"]:
            reasons.append(str(expanded_field["reason"]))
        else:
            blockers.append(f"CvD: nicht in den Top {rank_limit} des Dashboard-Felds")
    min_news_value = float(config.min_editorial_news_value)
    if news_value < min_news_value:
        blockers.append(
            f"CvD: Nachrichtenwert zu niedrig ({news_value:.1f} < {min_news_value:.1f})"
        )
    if total < min_editorial_score:
        blockers.append(
            f"CvD: redaktionelle Gesamtfreigabe zu schwach ({total:.1f} < {min_editorial_score:.1f})"
        )
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
    if low_civic_impact and not breaking and not sport_review.get("eventful"):
        blockers.append(
            "CvD: enger Kurios-/Click-Reiz ohne ausreichend breite öffentliche Relevanz"
        )
    if sport_review and not sport_review.get("eventful"):
        blockers.append(
            "CvD: Sport ohne frische bestaetigte Ergebnis-, Transfer-, Personal- oder Live-Lage"
        )
    elif sport_review.get("eventful"):
        reasons.append(f"CvD-Sport-Gate: {sport_review['label']}")
    missing_event_signal = config.event_gate_enabled and not breaking and not _has_news_event(title)
    if missing_event_signal:
        blockers.append("CvD: kein konkretes Nachrichten-Ereignis erkennbar (Service/Teaser)")
    if predicted_or is None and not breaking and alert_score < config.no_forecast_min_alert_score:
        blockers.append("CvD: ohne belastbare OR-Prognose nur bei absoluter Top-Lage")
    if not breaking and time_fit["score"] < config.min_time_fit_score:
        blockers.append(
            f"CvD: unguenstiges Zeitfenster ({time_fit['score']:.1f} < {config.min_time_fit_score:.1f})"
        )
    if (
        not breaking
        and config.slot_gate_enabled
        and time_fit.get("waitRecommended")
        and float(pacing.get("deficit") or 0.0) < 1.5
    ):
        blockers.append(str(time_fit["waitReason"]))

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
            "sportReview": sport_review or None,
        },
    }


def _sport_schedule_context(target_date: dt.date, hour: int) -> str:
    """Return the day-specific sport window without calling an external API."""
    weekday = target_date.weekday()
    month = target_date.month
    contexts: list[str] = []
    if month in {1, 6, 7, 8} and hour in {7, 8, 12, 13, 18, 19, 20}:
        contexts.append("bestaetigter Transfer")
    in_club_season = month >= 8 or month <= 5
    if in_club_season:
        if weekday in {1, 2} and 18 <= hour <= 23:
            contexts.append("Champions-League-Live/Ergebnis")
        elif weekday == 3 and 18 <= hour <= 23:
            contexts.append("Europa-/Conference-League-Live/Ergebnis")
        elif weekday == 4 and 18 <= hour <= 22:
            contexts.append("Bundesliga-Freitag")
        elif weekday == 5 and 14 <= hour <= 22:
            contexts.append("Bundesliga-Samstag")
        elif weekday == 6 and 14 <= hour <= 22:
            contexts.append("Bundesliga-Sonntag")
        elif weekday == 0 and 18 <= hour <= 22:
            contexts.append("Sport-Nachlauf mit neuer Lage")
    return ", ".join(contexts)


def _sport_candidate_review(
    title: str,
    now_ts: int,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a sport state and require a recent event timestamp."""
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    local_dt = dt.datetime.fromtimestamp(int(now_ts or time.time()), ZoneInfo("Europe/Berlin"))
    context = _sport_schedule_context(local_dt.date(), local_dt.hour)
    candidate = candidate or {}
    transfer_topic = any(
        term in text for term in ("transfer", "wechsel", "wechselt", "verpflicht", "unterschreibt")
    )
    transfer_confirmed = transfer_topic and any(
        term in text
        for term in (
            "wechselt",
            "verpflichtet",
            "unterschreibt",
            "bestaetigt",
            "bestätigt",
            "fix",
            "offiziell",
        )
    )
    prematch = bool(
        any(term in text for term in ("anpfiff", "vorschau", "aufstellung", "startelf"))
        or re.search(r"\b(?:beginnt|startet|spielt)\b.{0,24}\b\d{1,2}[:.]\d{2}\b", text)
        or re.search(r"\b\d{1,2}[:.]\d{2}\s*uhr\b", text)
    )
    score_signal = bool(re.search(r"\b\d{1,2}:\d{1,2}\b(?!\s*uhr)", text))
    final_signal = any(
        term in text
        for term in (
            "gewinnt",
            "gewonnen",
            "siegt",
            "verliert",
            "verloren",
            "niederlage",
            "meister",
            "ausgeschieden",
            "finale erreicht",
            "abpfiff",
            "endstand",
            "schlusspfiff",
        )
    )
    final_result = bool(final_signal or (score_signal and not prematch and "live" not in text))
    availability = any(
        term in text
        for term in (
            "faellt aus",
            "fällt aus",
            "verletzt",
            "sperre",
            "gesperrt",
            "entlassen",
            "gefeuert",
            "tritt zurueck",
            "tritt zurück",
        )
    )
    live_material_terms = (
        "tor",
        "ausgleich",
        "fuehrung",
        "führung",
        "rote karte",
        "platzverweis",
        "elfmeter",
        "halbzeit",
        "spielabbruch",
        "abgebrochen",
        "verletzung",
    )
    live_update = bool(
        ("live" in text or _is_live_ticker_title(text))
        and (
            _has_live_ticker_push_update(text)
            or score_signal
            or any(term in text for term in live_material_terms)
        )
        and not prematch
    )
    speculative = _is_speculative(text)
    if transfer_confirmed:
        state, event_type, max_age_minutes = "TRANSFER", "bestaetigter Transfer", 180.0
    elif final_result:
        state, event_type, max_age_minutes = "FINAL", "Endstand/Ergebnis", 60.0
    elif live_update:
        state, event_type, max_age_minutes = "LIVE_MATERIAL", "materielles Live-Update", 10.0
    elif availability:
        state, event_type, max_age_minutes = (
            "PERSONNEL",
            "Ausfall/Personalentscheidung",
            180.0,
        )
    elif prematch:
        state, event_type, max_age_minutes = "PREMATCH", "Anpfiff/Vorbericht", 0.0
    else:
        state, event_type, max_age_minutes = "ROUTINE", "keine bestaetigte neue Sportlage", 0.0

    event_timestamp = 0
    event_source = ""
    for key in (
        "eventUpdatedAt",
        "sportEventUpdatedAt",
        "updatedAt",
        "modDate",
        "pubDate",
        "publishedAt",
    ):
        parsed = _parse_ts(candidate.get(key))
        if parsed > 0:
            event_timestamp = parsed
            event_source = key
            break
    event_age_minutes = (
        max(0.0, (int(now_ts) - event_timestamp) / 60.0)
        if event_timestamp > 0 and event_timestamp <= int(now_ts) + 5 * 60
        else None
    )
    confirmed_state = state in {"TRANSFER", "FINAL", "LIVE_MATERIAL", "PERSONNEL"}
    fresh_enough = bool(
        event_age_minutes is not None
        and max_age_minutes > 0
        and event_age_minutes <= max_age_minutes
    )
    eventful = bool(
        confirmed_state and fresh_enough and not (speculative and not transfer_confirmed)
    )

    competition_context = any(
        term in context for term in ("League", "Bundesliga", "Sport-Nachlauf")
    )
    transfer_context = "Transfer" in context
    context_matches = bool(
        (state == "TRANSFER" and transfer_context)
        or (state in {"FINAL", "LIVE_MATERIAL"} and competition_context)
    )
    timing_delta = 0.2 if eventful else -1.8
    if eventful and context_matches:
        timing_delta = 1.5
    elif eventful and context:
        timing_delta = 0.4
    if state == "LIVE_MATERIAL" and not competition_context:
        timing_delta = min(timing_delta, 0.2)

    freshness_label = ""
    if confirmed_state and event_age_minutes is None:
        freshness_label = "; Ereigniszeit fehlt"
    elif confirmed_state and not fresh_enough:
        freshness_label = f"; Ereignis {event_age_minutes:.0f} Min. alt"
    return {
        "eventful": eventful,
        "state": state,
        "eventType": event_type,
        "context": context,
        "contextMatches": context_matches,
        "timingDelta": timing_delta,
        "eventUpdatedAt": event_timestamp or None,
        "eventTimestampSource": event_source,
        "eventAgeMinutes": round(event_age_minutes, 1) if event_age_minutes is not None else None,
        "maxEventAgeMinutes": max_age_minutes,
        "freshEnough": fresh_enough,
        "bypassSlotWait": bool(eventful and state in {"LIVE_MATERIAL", "FINAL"}),
        "label": (
            f"Sport: {event_type}{freshness_label}"
            + (f"; passender Kontext {context}" if context_matches else "")
        ),
    }


_MORNING_FATALITY_RE = re.compile(
    r"\b(?:stirbt|starb|gestorben|tot|toter|tote|todesfall|ertrinkt|ertrank|"
    r"ertrunken|leiche|toedlich|tödlich|ums leben)\b",
    re.IGNORECASE,
)
_MORNING_ACTIONABLE_RE = re.compile(
    r"\b(?:warnung|gefahr|evakuierung|sperrung|ausfall|rueckruf|rückruf|"
    r"fahndung|vermisst|streik|unwetter|hochwasser|hitzewarnung)\b",
    re.IGNORECASE,
)
_MORNING_MAJOR_EVENT_RE = re.compile(
    r"\b(?:terror|anschlag|amok|krieg|explosion|grossbrand|großbrand|"
    r"flugzeugabsturz|zugunglueck|zugunglück|katastrophe)\b",
    re.IGNORECASE,
)
_MORNING_PROMINENCE_RE = re.compile(
    r"\b(?:bundespraesident|bundespräsident|praesident|präsident|kanzler|"
    r"papst|weltstar|nationalspieler|weltmeister)\b",
    re.IGNORECASE,
)
_MORNING_MULTIPLE_VICTIMS_RE = re.compile(
    r"(?:\b\d+\s+(?:tote|verletzte|opfer)\b|"
    r"\b(?:mehrere|viele|zahlreiche)\s+(?:tote|verletzte|opfer)\b)",
    re.IGNORECASE,
)


def _morning_reader_value_review(
    candidate: dict[str, Any],
    now_ts: int,
) -> dict[str, Any]:
    """Block isolated tragedy stories in the early daypart unless they carry public need."""
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    if not 5 <= local_dt.hour < 10:
        return {"applies": False, "approved": True, "reasons": [], "blockers": []}
    if _is_breaking(candidate):
        return {
            "applies": True,
            "approved": True,
            "reasons": ["Morgenfit: bestaetigte Breaking-Lage"],
            "blockers": [],
        }

    title = _title(candidate)
    if not _MORNING_FATALITY_RE.search(title):
        return {"applies": True, "approved": True, "reasons": [], "blockers": []}

    wider_need = bool(
        _MORNING_ACTIONABLE_RE.search(title)
        or _MORNING_MAJOR_EVENT_RE.search(title)
        or _MORNING_PROMINENCE_RE.search(title)
        or _MORNING_MULTIPLE_VICTIMS_RE.search(title)
    )
    if wider_need:
        return {
            "applies": True,
            "approved": True,
            "reasons": [
                "Morgenfit: Todes-/Unglueckslage hat eine konkrete uebergeordnete Relevanz"
            ],
            "blockers": [],
        }

    reason = (
        "Morgenfit: isolierte Todes-/Ungluecksgeschichte ohne akute Warn-, "
        "Handlungs- oder uebergeordnete Nachrichtenrelevanz"
    )
    return {
        "applies": True,
        "approved": False,
        "reasons": [],
        "blockers": [reason],
    }


def _time_fit_review(
    *,
    now_ts: int,
    section: str,
    title: str = "",
    candidate: dict[str, Any] | None = None,
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
        score, label = 9.0, "starkes Mittagsfenster"
    elif 14 <= hour < 17:
        score, label = 7.0, "Nachmittagsfenster"
    elif 17 <= hour < 20:
        score, label = 10.0, "starkes Feierabendfenster"
    elif 20 <= hour < 22:
        score, label = 7.0, "Abendfenster"
    elif 22 <= hour < 24:
        score, label = (
            (6.0, "spätes Breaking-Fenster") if breaking else (3.0, "spätes Abendfenster")
        )
    else:
        score, label = 4.0, "unbekanntes Zeitfenster"

    manual_score = score
    section_l = section.lower()
    people_parenthood = is_german_public_figure_parenthood_story(
        candidate or {"title": title, "category": section_l}
    )
    sport_review = _sport_candidate_review(title, now_ts, candidate) if section_l == "sport" else {}
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
    if people_parenthood and 16 <= hour < 22:
        manual_score += 1.0
        label += "; People-Ereignis passt in Feierabend/Abend"
    if sport_review:
        manual_score += float(sport_review.get("timingDelta") or 0.0)
        if not sport_review.get("eventful") and not breaking:
            manual_score = min(manual_score, 3.0)
        label += f"; {sport_review['label']}"

    slot = _slot_baseline(hour, weekday)
    slot_avg = float(slot.get("avg_or") or 0.0) if slot else None
    slot_count = int(slot.get("count") or 0) if slot else 0
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

    # Die konkrete Wochentagszelle entscheidet. So bleibt Montag 09/10 Uhr
    # nutzbar, waehrend dieselben Stunden an schwachen Wochentagen vermieden werden.
    is_mandatory = bool(
        slot_count > 0
        and slot_avg is not None
        and slot_avg >= float(config.peak_slot_min_or or 6.0)
    )
    is_avoid = bool(slot_count <= 0 or (slot_avg is not None and slot_avg < 5.0))
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
    pacing = _push_pacing_review(
        pushes_today,
        now_ts,
        config,
        basis=("teams_alerts" if config.independent_pacing_enabled else "actual_pushes"),
    )
    meaningful_wait = bool(
        next_better
        and (
            is_avoid
            or (score < 7.0 and float(next_better.get("orGain") or 0.0) >= 0.7)
            or float(next_better.get("score") or 0.0) >= score + 1.5
        )
    )
    wait_recommended = bool(
        not breaking and meaningful_wait and float(pacing.get("deficit") or 0.0) < 1.5
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
        "sportReview": sport_review or None,
    }


def _quiet_hours_reason(now_ts: int, config: TeamsAlertConfig) -> str:
    local_dt = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin"))
    current = local_dt.hour * 60 + local_dt.minute
    if _MANDATORY_QUIET_HOURS_START_MINUTE <= current < _MANDATORY_QUIET_HOURS_END_MINUTE:
        return "Teams-Ruhezeit aktiv: 00:00 bis 05:30 Uhr"

    start = _parse_hhmm_to_minutes(config.quiet_hours_start)
    end = _parse_hhmm_to_minutes(config.quiet_hours_end)
    if start is None or end is None or start == end:
        return ""
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
    section = _section_key(_section(candidate))
    breaking = _is_breaking(candidate)
    people_parenthood = is_german_public_figure_parenthood_story(candidate)

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
        "krieg",
        "terror",
        "anschlag",
        "iran",
        "israel",
        "ukraine",
        "putin",
        "trump",
        "merz",
        "regierung",
        "bundestag",
        "warnung",
        "gefahr",
        "ruecktritt",
        "rücktritt",
        "feuerpause",
        "scheidung",
        "scheidungszoff",
        "trennung",
        "ehe-aus",
        "unterhalt",
        "vermögen",
        "vermoegen",
        "wm-held",
        "weltmeister",
        "schweini",
        "schweinsteiger",
    )
    soft_terms = ("horoskop", "quiz", "abo", "shopping", "rabatt", "sommertrend")
    impact_bonus = 0.0
    if breaking:
        impact_bonus += 6.0
    if any(term in title_l for term in high_impact_terms):
        impact_bonus += 4.0
    if _has_hard_public_need(title, section):
        impact_bonus += 4.0
    if _is_urgent_public_service_title(title):
        impact_bonus += 4.0
    if _is_public_money_fraud_enforcement(title):
        impact_bonus += 4.0
    if _is_celebrity_relationship_money_conflict(title, section):
        impact_bonus += 14.0
    if people_parenthood:
        impact_bonus += 14.0
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
        title=_title(candidate),
        candidate=candidate,
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


def _historical_slot_forecast(
    candidate: dict[str, Any], now_ts: int | None = None
) -> dict[str, Any]:
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
                f"historische Slot-Prognose: {_weekday_label(weekday)} {hour:02d}:00, " f"n={count}"
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
    count = int(slot.get("count") or 0) if slot else 0
    stars = int(slot.get("stars") or 0) if slot else 0
    if avg_or is None or count <= 0:
        score = 4.5
    else:
        score = 2.0 + ((avg_or - 4.2) / (7.3 - 4.2)) * 7.0
    score += stars * 0.25
    if avg_or is not None and avg_or >= 6.0:
        score = max(score, 8.5)
    if avg_or is not None and avg_or < 5.0 and not breaking:
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
        "sport": "sport",
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
    count = int(slot.get("count") or 0) if slot else 0
    try:
        from app.push_schedule.weekly_baseline import PDF_OVERALL_AVG

        overall = float(PDF_OVERALL_AVG)
    except Exception:
        overall = 5.44
    if count <= 0:
        return 0.35
    weight = (avg_or or overall) / overall
    if avg_or is not None and avg_or >= 6.0:
        weight *= 1.25
    if avg_or is not None and avg_or < 5.0:
        weight *= 0.55
    return _clamp(weight, 0.15, 1.9)


def _is_lunch_prime_hour(hour: int) -> bool:
    return 12 <= int(hour) < 14


def _is_lunch_prime_ts(now_ts: int) -> bool:
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    return _is_lunch_prime_hour(local_dt.hour)


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

    With the smart slot gate enabled this is the exact number of binding :45
    deadlines already reached, including explicitly marked recovery windows.
    The legacy weighted curve remains available when the gate is disabled.
    """
    target = max(0, int(config.target_pushes_per_day or 0))
    if target <= 0:
        return 0.0
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    if config.slot_gate_enabled:
        slots = _daily_runtime_opportunities(local_dt.date(), config)
        return float(
            sum(
                1
                for slot in slots
                if slot.get("minimumRequired") and int(slot.get("ts") or 0) <= int(now_ts)
            )
        )

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


def _daily_slot_gate_review(
    candidate: dict[str, Any],
    *,
    score: float,
    alert_score: float,
    editorial_score: float,
    predicted_or: float | None,
    pushes_today: int | None,
    teams_alerts_today: int,
    breaking: bool,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Decide whether the live worker should act now or keep collecting to :45."""
    if not config.slot_gate_enabled:
        return {"enabled": False, "approved": True, "reasons": [], "blockers": []}

    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    configured_target = max(1, int(config.target_pushes_per_day or 15))
    regular_slots = _daily_plan_slots(local_dt.date(), configured_target, config)
    planned_slots = _daily_runtime_opportunities(local_dt.date(), config)
    target_count = max(configured_target, len(planned_slots))
    due_slots = [slot for slot in planned_slots if int(slot.get("ts") or 0) <= int(now_ts)]
    due_count = len(due_slots)
    independent = bool(config.independent_pacing_enabled)
    current_count = (
        int(teams_alerts_today or 0)
        if independent
        else (pushes_today if pushes_today is not None else int(teams_alerts_today or 0))
    )
    count_basis = "teams_alerts" if independent or pushes_today is None else "actual_pushes"
    count_label = "Teams-Hinweise" if count_basis == "teams_alerts" else "echte Pushes"
    deficit = max(0, due_count - int(current_count or 0))
    upcoming = [slot for slot in planned_slots if int(slot.get("ts") or 0) > int(now_ts)]
    next_slot = min(upcoming, key=lambda slot: int(slot.get("ts") or 0)) if upcoming else None
    hour_slots = [slot for slot in planned_slots if int(slot.get("hour") or -1) == local_dt.hour]
    overdue_hour_slots = [slot for slot in hour_slots if int(slot.get("ts") or 0) <= int(now_ts)]
    upcoming_hour_slots = [slot for slot in hour_slots if int(slot.get("ts") or 0) > int(now_ts)]
    if deficit > 0 and overdue_hour_slots:
        current_slot = max(overdue_hour_slots, key=lambda slot: int(slot.get("ts") or 0))
    elif upcoming_hour_slots:
        current_slot = min(upcoming_hour_slots, key=lambda slot: int(slot.get("ts") or 0))
    else:
        current_slot = None
    available_overdue_decision = bool(deficit > 0 and overdue_hour_slots)
    projected_maximum_from_plan = min(
        target_count,
        int(current_count or 0) + len(upcoming) + int(available_overdue_decision),
    )
    projected_shortfall = max(0, target_count - projected_maximum_from_plan)

    sport_review = (
        _sport_candidate_review(_title(candidate), now_ts, candidate)
        if _section_key(_section(candidate)) == "sport"
        else {}
    )
    if sport_review.get("bypassSlotWait"):
        return {
            "enabled": True,
            "approved": True,
            "mode": "sport_event_override",
            "dueCount": due_count,
            "currentCount": current_count,
            "targetCount": target_count,
            "plannedOpportunityCount": len(planned_slots),
            "projectedMaximumFromPlan": projected_maximum_from_plan,
            "projectedShortfall": projected_shortfall,
            "countBasis": count_basis,
            "deficit": deficit,
            "slot": current_slot,
            "nextSlot": next_slot,
            "sportState": sport_review.get("state"),
            "reasons": [
                "Slot-Logik: frischer materieller Sportzustand darf sofort geprueft werden"
            ],
            "blockers": [],
        }

    if breaking:
        return {
            "enabled": True,
            "approved": True,
            "mode": "breaking_override",
            "dueCount": due_count,
            "currentCount": current_count,
            "targetCount": target_count,
            "plannedOpportunityCount": len(planned_slots),
            "projectedMaximumFromPlan": projected_maximum_from_plan,
            "projectedShortfall": projected_shortfall,
            "countBasis": count_basis,
            "deficit": deficit,
            "slot": current_slot,
            "nextSlot": next_slot,
            "reasons": ["Slot-Logik: Breaking darf sofort"],
            "blockers": [],
        }

    slot_avg = float((current_slot or {}).get("avgOR") or 0.0)
    peak = bool(current_slot and slot_avg >= float(config.peak_slot_min_or or 6.0))
    advertised_double_hours = {
        int(item.get("hour") or -1)
        for item in _daily_plan_double_opportunities(
            local_dt.date(),
            regular_slots,
            config,
            limit=max(
                3,
                target_count - len(regular_slots),
            ),
        )
    }
    projected_shortfall_catchup = bool(
        current_slot and int(current_slot.get("ts") or 0) > int(now_ts) and projected_shortfall > 0
    )
    double_slot = bool(
        current_slot
        and (current_slot.get("minimumDouble") or local_dt.hour in advertised_double_hours)
    )
    exceptional = bool(
        peak
        and double_slot
        and score >= float(config.early_exceptional_score or 88.0)
        and alert_score >= float(config.early_exceptional_alert_score or 86.0)
        and editorial_score >= float(config.early_exceptional_editorial_score or 80.0)
        and (predicted_or is None or predicted_or >= max(5.0, float(config.min_or or 0.0)))
        and _has_news_event(_title(candidate))
    )

    result = {
        "enabled": True,
        "approved": False,
        "mode": "wait",
        "dueCount": due_count,
        "currentCount": current_count,
        "targetCount": target_count,
        "plannedOpportunityCount": len(planned_slots),
        "projectedMaximumFromPlan": projected_maximum_from_plan,
        "projectedShortfall": projected_shortfall,
        "countBasis": count_basis,
        "deficit": deficit,
        "slot": current_slot,
        "nextSlot": next_slot,
        "reasons": [],
        "blockers": [],
        "exceptional": exceptional,
        "doubleOpportunity": double_slot,
        "minimumDouble": bool((current_slot or {}).get("minimumDouble")),
        "minimumCommitment": False,
    }
    if current_slot is None:
        if projected_shortfall > 0:
            result["approved"] = True
            result["mode"] = "projected_shortfall_catchup"
            result["minimumCommitment"] = True
            result["reasons"].append(
                f"Shortfall-Recovery: Der verbindliche Restplan erreicht nur "
                f"{projected_maximum_from_plan}/{target_count} {count_label}; "
                "am freien Cool-down-Rand jetzt den besten Kandidaten ab Push-Score 75 waehlen"
            )
            return result
        next_label = str((next_slot or {}).get("label") or "naechsten starken Slot")
        result["blockers"].append(
            f"Tagesplan: aktuelles Stundenfenster bewusst nachrangig; bis {next_label} sammeln"
        )
        return result

    current_deadline_ts = int(current_slot.get("ts") or 0)
    if int(now_ts) < current_deadline_ts:
        catchup_double = bool(double_slot and deficit >= 2 and local_dt.minute <= 5)
        result["catchupDouble"] = catchup_double
        if projected_shortfall_catchup:
            result["approved"] = True
            result["mode"] = "projected_shortfall_catchup"
            result["minimumCommitment"] = True
            result["reasons"].append(
                f"Shortfall-Recovery: Mit den verbleibenden Planfenstern sind nur "
                f"{projected_maximum_from_plan}/{target_count} {count_label} erreichbar; "
                "nach Cool-down jetzt den besten Kandidaten ab Push-Score 75 waehlen"
            )
        elif catchup_double:
            result["approved"] = True
            result["mode"] = "peak_catchup_first"
            result["reasons"].append(
                f"Roter/gelber Slot {current_slot['hour']:02d} Uhr: fruehe Aufholchance bei "
                f"{deficit} fehlenden {count_label}; normale Qualitaetsgates bleiben aktiv"
            )
        elif exceptional:
            result["approved"] = True
            result["mode"] = "peak_early_exception"
            result["reasons"].append(
                f"Roter/gelber Slot {current_slot['hour']:02d} Uhr: aussergewoehnlicher Kandidat darf vor :45 raus"
            )
        else:
            result["blockers"].append(
                f"Tagesplan: bis {current_slot['label']} weitere Kandidaten sammeln; danach besten verfuegbaren waehlen"
            )
        return result

    if deficit > 0:
        result["approved"] = True
        result["mode"] = "deadline_fallback"
        result["minimumCommitment"] = True
        result["reasons"].append(
            f"Plan-Entscheidung {current_slot['label']} faellig: {current_count} "
            f"{count_label} bei {due_count} faelligen Mindestfenstern; besten "
            "Kandidaten ab Push-Score 75 waehlen"
        )
        return result
    if exceptional:
        result["approved"] = True
        result["mode"] = "peak_double_opportunity"
        result["reasons"].append(
            "Roter/gelber Slot: zusaetzliche Qualitätschance trotz erreichtem Tagespacing"
        )
        return result

    result["blockers"].append(
        f"Tagesplan im Soll ({current_count}/{due_count}); bis zum naechsten geplanten Fenster sammeln"
    )
    return result


def _high_score_override_review(
    blockers: list[str],
    *,
    score: float,
    score_source: str,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Let canonical scores above 80 waive soft gates, never safety invariants."""
    threshold = _clamp(float(config.high_score_always_threshold or 80.0), 0.0, 100.0)
    canonical = bool(
        score_source == "internal_score_api" or not config.require_internal_score_api
    )
    active = bool(canonical and float(score) > threshold)
    review = {
        "active": active,
        "approved": False,
        "canonicalScore": canonical,
        "threshold": round(threshold, 1),
        "score": round(float(score), 1),
        "waivedBlockers": [],
        "hardBlockers": [],
        "remainingBlockers": list(blockers),
        "reasons": [],
    }
    if not active:
        return review

    hard = [blocker for blocker in blockers if _is_hard_teams_blocker(blocker)]
    soft = [blocker for blocker in blockers if not _is_hard_teams_blocker(blocker)]
    review["waivedBlockers"] = soft
    review["hardBlockers"] = hard
    review["remainingBlockers"] = hard
    if hard:
        return review

    review["approved"] = True
    review["reasons"] = [
        f"Push Score {score:.1f} liegt strikt ueber {threshold:.1f}: "
        "weiche Qualitaets- und Ermuedungsgates werden ueberstimmt",
        "Ruhezeit, Zeitpunkt, Cooldown, Fakten, Aktualitaet, Ressort und "
        "Teams-Dubletten wurden weiterhin hart geprueft",
    ]
    if soft:
        review["reasons"].append(
            f"High-Score-Regel lockert {len(soft)} weiche Gate(s); "
            "der kanonische Push Score entscheidet"
        )
    return review


def _deadline_fallback_review(
    candidate: dict[str, Any],
    *,
    slot_gate: dict[str, Any],
    blockers: list[str],
    score: float,
    alert_score: float,
    editorial_score: float,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """For a due or mathematically required recovery, trust raw Push Score."""
    fallback_modes = {"deadline_fallback", "projected_shortfall_catchup"}
    review = {
        "active": bool(slot_gate.get("mode") in fallback_modes),
        "approved": False,
        "waivedBlockers": [],
        "remainingBlockers": list(blockers),
        "reasons": [],
        "secondaryCautions": [],
        "minimumCommitment": bool(slot_gate.get("minimumCommitment")),
    }
    if not review["active"]:
        return review

    floor_failures: list[str] = []
    breaking = bool(_is_breaking(candidate) and config.breaking_override)
    score_floor = (
        float(config.breaking_min_score or 0.0)
        if breaking
        else max(
            float(config.deadline_fallback_min_score or 0.0),
            float(config.min_score or 0.0),
        )
    )
    alert_reference = max(
        float(config.deadline_fallback_min_alert_score or 0.0),
        float(config.min_alert_score or 0.0) - 5.0,
    )
    editorial_reference = max(
        float(config.deadline_fallback_min_editorial_score or 0.0),
        float(config.min_editorial_score or 0.0) - 5.0,
    )
    if score < score_floor:
        floor_failures.append(
            f"Deadline-Fallback: Push Score {score:.1f} < absolute Untergrenze "
            f"{score_floor:.1f}"
        )
    if alert_score < alert_reference:
        review["secondaryCautions"].append(
            f"Alert Score {alert_score:.1f} unter Referenz {alert_reference:.1f}; "
            "der harte Push-Score entscheidet im Mindestfenster"
        )
    if editorial_score < editorial_reference:
        review["secondaryCautions"].append(
            f"CvD-Score {editorial_score:.1f} unter Referenz {editorial_reference:.1f}; "
            "der harte Push-Score entscheidet im Mindestfenster"
        )
    if floor_failures:
        review["remainingBlockers"] = [*blockers, *floor_failures]
        return review

    hard: list[str] = []
    soft: list[str] = []
    for blocker in blockers:
        if _is_hard_teams_blocker(blocker):
            hard.append(blocker)
        else:
            soft.append(blocker)

    review["waivedBlockers"] = soft
    review["remainingBlockers"] = hard
    if hard:
        return review

    review["approved"] = True
    if slot_gate.get("mode") == "projected_shortfall_catchup":
        review["reasons"] = [
            "Mathematische Shortfall-Recovery: bester verfuegbarer, noch nicht per "
            "Teams empfohlener Kandidat ab Push-Score 75",
        ]
    else:
        review["reasons"] = [
            "Faelliges Mindestfenster im Rueckstand: bester verfuegbarer, noch nicht "
            "per Teams empfohlener Kandidat ab Push-Score 75",
        ]
    if soft:
        review["reasons"].append(
            f"Deadline-Fallback lockert {len(soft)} weiche Qualitaetsgates; harte Sperren bleiben aktiv"
        )
    review["reasons"].extend(review["secondaryCautions"])
    if _section(candidate).lower() == "sport":
        review["reasons"].append("Sport-Fallback nur mit bestaetigter neuer Sportlage")
    return review


def _agent_hard_blockers(blockers: list[str]) -> list[str]:
    """Keep only operational invariants for the adversarial final reviewer."""
    markers = (
        "alerts deaktiviert",
        "kein gueltiger interner push-balancer-score",
        "ruhezeit aktiv",
        "ohne headline",
        "ohne artikel-link",
        "kein artikel:",
        "nicht fuer teams alerts freigegeben",
        "ist fuer teams alerts ausgeschlossen",
        "letzter push-zeitpunkt nicht verfuegbar",
        "pause seit letztem push zu kurz",
        "push-dichte in den letzten",
        "bereits per teams gemeldet",
        "thema bereits per teams gemeldet",
        "re-alert-cooldown",
        "teams-cooldown aktiv",
        "tageslimit fuer teams-hinweise",
        "staerkerer kandidat vorhanden",
        "wahrscheinlich ueberholt",
        "bereits als vollzogen gemeldet",
        "bereits als teams-kandidat versucht",
        "teams-hinweis wird bereits versendet",
        "morgenfit:",
        "feld unsicher",
        "deutschland-relevanz",
    )
    return [
        blocker
        for blocker in blockers
        if any(marker in str(blocker or "").casefold() for marker in markers)
    ]


def _final_agent_review(
    candidate: dict[str, Any],
    *,
    context: dict[str, Any],
    config: TeamsAlertConfig,
    min_pause: int,
    min_or: float,
    minutes_since_last_push: float | None,
    minutes_since_last_teams_alert: float | None,
    allow_unknown_last_push: bool,
    freshness_hours: float | None,
    publication_review: dict[str, Any],
    is_speculative: bool,
    overtaken_reason: str,
    non_article_reason: str,
    live_push_match_reason: str,
    topic_duplicate_reason: str,
    realert_reason: dict[str, Any],
    editorial_review: dict[str, Any],
    forecast: dict[str, Any],
    predicted_or: float | None,
    forecast_near_miss_accepted: bool,
    slot_gate: dict[str, Any],
    deadline_fallback: dict[str, Any],
    high_score_override: dict[str, Any],
    visit_potential: dict[str, Any],
    push_pacing: dict[str, Any],
    minimum_pressure: dict[str, Any],
    teams_alerts_today: int,
    candidate_risks: list[str],
    remaining_blockers: list[str],
) -> dict[str, Any]:
    """Build one ephemeral signal snapshot and fan it out to all local reviewers."""
    title = _title(candidate)
    section = _section_key(_section(candidate))
    breakdown = editorial_review.get("breakdown") or {}
    sport_review = breakdown.get("sportReview") or {}
    slot = slot_gate.get("slot") or {}
    allowed_sections = {
        _section_key(value) for value in config.allowed_sections if str(value).strip()
    }
    excluded_sections = {
        _section_key(value) for value in config.excluded_sections if str(value).strip()
    }
    weak_news_shape = bool(
        (_is_live_ticker_title(title) and not _has_live_ticker_push_update(title))
        or _is_scheduled_process_without_update(title)
        or _is_abstract_explainer_without_update(title)
        or _is_nonessential_curiosity(title)
    )
    history_index = context.get("historyReviewIndex")
    if not isinstance(history_index, dict):
        history_index = _push_history_review_index(
            [item for item in (context.get("history") or []) if isinstance(item, dict)],
            int(context.get("nowTs") or time.time()),
            config,
        )
    snapshot = {
        "contextAvailable": dict(
            context.get("contextAvailable")
            or {
                "history": True,
                "alertState": True,
                "globalCooldown": True,
                "dailyAlertCount": True,
                "recentTeamsAlerts": True,
            }
        ),
        "historyAuthoritative": bool(context.get("historyAuthoritative", False)),
        "title": title,
        "url": _url(candidate),
        "section": section,
        "nonArticleReason": non_article_reason,
        "allowedSections": allowed_sections,
        "excludedSections": excluded_sections,
        "historyExactUrls": history_index.get("exactUrls") or frozenset(),
        "historyExactTitles": history_index.get("exactTitles") or frozenset(),
        "livePushMatchReason": live_push_match_reason,
        "topicDuplicateReason": topic_duplicate_reason,
        "realertBlocker": str(realert_reason.get("blocker") or ""),
        "overtakenReason": overtaken_reason,
        "freshnessHours": freshness_hours,
        "publicationStatus": str(publication_review.get("status") or "missing"),
        "publicationReason": str(publication_review.get("reason") or ""),
        "maxArticleAgeHours": float(config.max_article_age_hours),
        "speculativeMaxAgeHours": float(config.speculative_max_age_hours),
        "isSpeculative": is_speculative,
        "speculationConfirmed": bool(
            candidate.get("factConfirmed") is True
            or candidate.get("sourceConfirmed") is True
            or str(candidate.get("confirmationStatus") or "").strip().casefold() == "confirmed"
        ),
        "breaking": _is_breaking(candidate),
        "independentTeamsPacing": bool(config.independent_pacing_enabled),
        "candidateRisks": list(candidate_risks),
        "minutesSinceLastPush": minutes_since_last_push,
        "minPause": float(max(min_pause, _effective_global_cooldown_minutes(config))),
        "minutesSinceLastTeamsAlert": minutes_since_last_teams_alert,
        "globalCooldownMinutes": _effective_global_cooldown_minutes(config),
        "allowUnknownLastPush": allow_unknown_last_push,
        "recentPushCount6h": (
            0 if config.independent_pacing_enabled else int(context.get("recentPushCount6h") or 0)
        ),
        "maxPushesLast6h": int(config.max_pushes_last_6h or 0),
        "softService": _is_soft_service_or_quiz(title),
        "urgentService": _is_urgent_public_service_title(title),
        "weakNewsShape": weak_news_shape,
        "newsEvent": _has_news_event(title),
        "editorialApproved": bool(editorial_review.get("approved")),
        "editorialNewsValue": float(editorial_review.get("newsValue") or 0.0),
        "forecastSource": str(forecast.get("source") or ""),
        "forecastConfidence": float(forecast.get("confidence") or 0.0),
        "predictedOR": float(predicted_or) if predicted_or is not None else None,
        "minOR": float(min_or),
        "forecastNearMissAccepted": forecast_near_miss_accepted,
        "requireArticleForecast": bool(config.require_article_forecast),
        "hardPublicNeed": _has_hard_public_need(title, section),
        "verifiedPeopleMilestone": is_german_public_figure_parenthood_story(candidate),
        "deadlineApproved": bool(deadline_fallback.get("approved")),
        "highScoreOverrideApproved": bool(high_score_override.get("approved")),
        "expectedOpens": int(visit_potential.get("expectedOpens") or 0),
        "qualityAdjustedOpens": int(visit_potential.get("qualityAdjustedOpens") or 0),
        "reachConfidence": float(visit_potential.get("reachConfidence") or 0.0),
        "slotGateEnabled": bool(slot_gate.get("enabled")),
        "slotApproved": bool(slot_gate.get("approved")),
        "slotMode": str(slot_gate.get("mode") or ""),
        "timeFit": float(breakdown.get("timeFit") or 0.0),
        "minTimeFit": float(config.min_time_fit_score),
        "hasCurrentSlot": bool(slot),
        "slotPreferredSections": {
            _section_key(value) for value in (slot.get("preferredSections") or []) if value
        },
        "slotTopCategory": _section_key(str(slot.get("topCategory") or "")),
        "sportEventful": bool(sport_review.get("eventful")),
        "sportEventType": str(sport_review.get("eventType") or ""),
        "headlineClarity": float(breakdown.get("clarity") or 0.0),
        "genericHeadline": _is_generic_push_title(title),
        "minimumPressureActive": bool(minimum_pressure.get("active")),
        "pushSurplus": float(push_pacing.get("surplus") or 0.0),
        "teamsAlertsToday": int(teams_alerts_today or 0),
        "maxAlertsPerDay": int(config.max_alerts_per_day or 0),
        "remainingBlockers": _agent_hard_blockers(remaining_blockers),
        "waivedBlockers": [
            *list(deadline_fallback.get("waivedBlockers") or []),
            *list(high_score_override.get("waivedBlockers") or []),
        ],
    }
    review = run_agent_review_network(
        snapshot,
        enabled=config.agent_review_enabled,
        min_evidence_approvals=config.agent_review_min_evidence_approvals,
        min_consensus_score=config.agent_review_min_consensus_score,
        max_latency_ms=config.agent_review_max_latency_ms,
    )
    if (
        high_score_override.get("approved")
        and review.get("enabled")
        and not review.get("approved")
        and int(review.get("hardVetoCount") or 0) == 0
    ):
        review = dict(review)
        review["approved"] = True
        review["highScoreOverrideApplied"] = True
        review["blockingReason"] = ""
        review["summary"] = (
            f"{review.get('summary')}; kanonischer Push Score ueber 80 "
            "ueberstimmt nur den weichen Evidenz-Konsens"
        )
    return review


def _push_pacing_review(
    pushes_today: int | None,
    now_ts: int,
    config: TeamsAlertConfig,
    *,
    basis: str = "actual_pushes",
    actual_pushes_today: int | None = None,
) -> dict[str, Any]:
    expected = _expected_pushes_by_now(now_ts, config)
    basis_key = "teams_alerts" if basis == "teams_alerts" else "actual_pushes"
    basis_label = "Teams-Hinweise" if basis_key == "teams_alerts" else "echte Pushes"
    if pushes_today is None:
        return {
            "known": False,
            "pushesToday": None,
            "countToday": None,
            "basis": basis_key,
            "actualPushesToday": actual_pushes_today,
            "expectedByNow": expected,
            "deficit": 0.0,
            "surplus": 0.0,
            "editorialAdjustment": 0.0,
            "label": f"Tagespacing: {basis_label} heute nicht bekannt",
        }
    deficit = max(0.0, expected - pushes_today)
    surplus = max(0.0, pushes_today - expected)
    adjustment = 0.0
    if surplus >= 2.0:
        adjustment = -min(7.0, surplus * 1.1)
    if deficit >= 1.5:
        label = f"Tagespacing {basis_label}: Rueckstand ({pushes_today} statt {expected:.1f})"
    elif surplus >= 2.0:
        label = f"Tagespacing {basis_label}: Vorsprung ({pushes_today} statt {expected:.1f})"
    else:
        label = f"Tagespacing {basis_label}: im Plan ({pushes_today} statt {expected:.1f})"
    return {
        "known": True,
        "pushesToday": pushes_today,
        "countToday": pushes_today,
        "basis": basis_key,
        "actualPushesToday": actual_pushes_today,
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

    Too few pushes open additional good timing opportunities but never lower
    the quality threshold. Already many pushes today raise the threshold to
    protect users. Breaking news never raises the threshold.
    """
    if not config.dynamic_threshold_enabled or pushes_today is None:
        return base_threshold, ""

    target = max(0, int(config.target_pushes_per_day or 0))
    expected = _expected_pushes_by_now(now_ts, config)
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
        return base_threshold, (
            f"Push-Rueckstand heute ({pushes_today} statt {expected:.0f}): "
            "Qualitaetsschwelle bleibt unveraendert"
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
        and now_ts - last_decision_ts
        < max(
            config.alert_cooldown_minutes * 60,
            config.repeat_suppression_hours * 3600,
        )
    ):
        return {
            "blocker": f"Bereits als Teams-Kandidat versucht: Sperre {config.repeat_suppression_hours}h"
        }
    if alert_status != "sent":
        return {}
    return {
        "blocker": "Bereits per Teams gemeldet; derselbe Artikel wird nicht erneut vorgeschlagen"
    }


def _live_push_comparison_reason(
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    now_ts: int | None = None,
    config: TeamsAlertConfig | None = None,
    *,
    history_index: dict[str, Any] | None = None,
) -> str:
    """Describe whether the independent Teams choice matches a real live push.

    The result is comparison metadata only. Exact URLs are compared across the
    retained history; fuzzy story matching uses the configured recent window.
    """
    config = config or TeamsAlertConfig()
    now = int(now_ts or time.time())
    window_start = now - int(max(0.0, config.pushed_topic_window_hours) * 3600)

    url = _article_identity_url(_url(candidate))
    title = _title(candidate)
    title_tokens = _tokens(title)
    slug_tokens = _url_slug_tokens(_url(candidate))

    if history_index is not None:
        if url and url in (history_index.get("exactUrls") or ()):
            return "Bereits live gepusht (gleiche Artikel-URL)"
        normalized_title = _normalize_title(title)
        if normalized_title and normalized_title in (history_index.get("exactTitles") or ()):
            return "Bereits live gepusht (identischer Titel)"
        for item_slug_tokens, item_title_tokens in history_index.get("recentTopics") or ():
            if _same_topic(slug_tokens, set(item_slug_tokens), 0.6):
                return "Bereits live gepusht (gleiche Story, URL-Slug)"
            if _same_topic(title_tokens, set(item_title_tokens), 0.6):
                return "Bereits live gepusht (sehr aehnliche Meldung)"
        return ""

    for item in history:
        item_url = _article_identity_url(_url(item))
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


def _live_push_match_type(reason: str) -> str:
    normalized = str(reason or "").casefold()
    if "artikel-url" in normalized:
        return "exact_article"
    if "identischer titel" in normalized:
        return "exact_title"
    if "url-slug" in normalized:
        return "same_story_slug"
    if "aehnliche meldung" in normalized or "ähnliche meldung" in normalized:
        return "same_story_title"
    return ""


def _push_history_review_index(
    history: list[dict[str, Any]],
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Build immutable exact and recent-topic indices once per candidate batch."""
    window_start = int(now_ts) - int(max(0.0, config.pushed_topic_window_hours) * 3600)
    exact_urls: set[str] = set()
    exact_titles: set[str] = set()
    recent_topics: list[tuple[frozenset[str], frozenset[str]]] = []
    for item in history:
        item_url = _article_identity_url(_url(item))
        item_title = _title(item)
        normalized_title = _normalize_title(item_title)
        if item_url:
            exact_urls.add(item_url)
        if normalized_title:
            exact_titles.add(normalized_title)
        item_ts = _safe_int(item.get("ts_num", item.get("ts", 0)))
        if item_ts and item_ts < window_start:
            continue
        recent_topics.append(
            (
                frozenset(_url_slug_tokens(_url(item))),
                frozenset(_tokens(item_title)),
            )
        )
    return {
        "exactUrls": frozenset(exact_urls),
        "exactTitles": frozenset(exact_titles),
        "recentTopics": tuple(recent_topics),
    }


def _url_slug_tokens(url: str) -> set[str]:
    """Bedeutungstragende Tokens aus dem URL-Pfad/Slug (ohne Ressort-Segmente)."""
    from urllib.parse import urlsplit

    path = urlsplit(str(url or "")).path.lower()
    drop = {
        "politik",
        "inland",
        "ausland",
        "sport",
        "fussball",
        "fußball",
        "news",
        "regional",
        "unterhaltung",
        "stars",
        "leute",
        "leben",
        "wissen",
        "auto",
        "geld",
        "wirtschaft",
        "digital",
        "video",
        "videos",
        "ratgeber",
        "reise",
        "spiele",
        "lifestyle",
        "mobil",
        "bild",
        "html",
        "amp",
        "www",
    }
    raw = re.split(r"[/\-_.]+", path)
    return {
        token for token in raw if len(token) >= 4 and token not in drop and not token.isdigit()
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
    article_ref = hashlib.sha256(
        str(decision.get("candidateId") or candidate_key(candidate)).encode("utf-8")
    ).hexdigest()[:12]
    review = decision.get("agentReview") if isinstance(decision.get("agentReview"), dict) else {}
    log.info(
        "[TeamsAlert] decision article_ref=%s score=%.1f predicted_or=%s "
        "expected_opens=%s estimated_reach=%s teams_alert_score=%s editorial_score=%s "
        "selection_score=%s response_score=%s last_push=%s "
        "decision=%s blocker_count=%s reason_count=%s agent_consensus=%s evaluated_at=%s",
        article_ref,
        float(decision.get("score") or 0.0),
        decision.get("predictedOR"),
        decision.get("expectedOpens"),
        decision.get("estimatedReach"),
        decision.get("teamsAlertScore"),
        decision.get("editorialScore"),
        decision.get("selectionScore"),
        decision.get("visitPotentialScore"),
        decision.get("lastPushAt"),
        "notify" if decision.get("shouldNotify") else "skip",
        len(decision.get("blockingReasons") or []),
        len(decision.get("reasons") or []),
        review.get("summary"),
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
    flagged = bool(
        candidate.get("isBreaking")
        or candidate.get("isEilmeldung")
        or candidate.get("is_eilmeldung")
    )
    if not flagged:
        return False
    title = _title(candidate)
    explicit_signal = bool(
        "++" in title or re.search(r"(?i)\b(?:eilmeldung|eil|breaking)\b", title)
    )
    provenance = str(candidate.get("breakingProvenance") or "").strip().casefold()
    trusted_provenance = provenance in {
        "cms_verified",
        "editorial_verified",
        "manual_verified",
        "wire_verified",
    }
    return bool(explicit_signal or trusted_provenance)


def _candidate_updated_ts(candidate: dict[str, Any]) -> int:
    for key in ("modDate", "updatedAt", "pubDate", "publishedAt"):
        parsed = _parse_ts(candidate.get(key))
        if parsed:
            return parsed
    return 0


def _freshness_hours(candidate: dict[str, Any], now_ts: int) -> float | None:
    review = _publication_time_review(candidate, now_ts)
    if review["status"] != "valid":
        return None
    return float(review["ageHours"])


def _publication_time_review(
    candidate: dict[str, Any],
    now_ts: int,
    config: TeamsAlertConfig | None = None,
) -> dict[str, Any]:
    """Validate article time and use the freshest legitimate publication/update."""
    config = config or TeamsAlertConfig()
    supplied: list[tuple[str, Any]] = []
    parsed: list[tuple[str, int]] = []
    for key in ("pubDate", "publishedAt", "modDate", "updatedAt"):
        raw = candidate.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        supplied.append((key, raw))
        timestamp = _parse_ts(raw)
        if timestamp <= 0:
            return {
                "status": "invalid",
                "ageHours": None,
                "timestamp": 0,
                "source": key,
                "reason": f"Zeitstempel {key} ist ungueltig",
            }
        parsed.append((key, timestamp))

    if not supplied:
        return {
            "status": "missing",
            "ageHours": None,
            "timestamp": 0,
            "source": "",
            "reason": "Veroeffentlichungs- oder Aktualisierungszeit fehlt",
        }

    future_tolerance_seconds = 5 * 60
    future = [(key, value) for key, value in parsed if value > now_ts + future_tolerance_seconds]
    if future:
        key, value = max(future, key=lambda item: item[1])
        minutes = (value - now_ts) / 60.0
        return {
            "status": "future",
            "ageHours": None,
            "timestamp": value,
            "source": key,
            "reason": f"Zeitstempel {key} liegt {minutes:.0f} Minuten in der Zukunft",
        }

    source, freshest = max(parsed, key=lambda item: item[1])
    age_hours = max(0.0, (now_ts - freshest) / 3600.0)
    if age_hours > float(config.max_article_age_hours):
        return {
            "status": "stale",
            "ageHours": round(age_hours, 3),
            "timestamp": freshest,
            "source": source,
            "reason": (
                f"Artikel ist zu alt: {age_hours:.1f}h > "
                f"{float(config.max_article_age_hours):.1f}h"
            ),
        }
    return {
        "status": "valid",
        "ageHours": round(age_hours, 3),
        "timestamp": freshest,
        "source": source,
        "reason": f"Belastbarer Zeitstempel aus {source}",
    }


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
    return dt.datetime.fromtimestamp(int(ts_value), ZoneInfo("Europe/Berlin")).isoformat()


def _format_dt(ts_value: int) -> str:
    return dt.datetime.fromtimestamp(ts_value, ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M")


def _format_time(ts_value: int) -> str:
    return dt.datetime.fromtimestamp(ts_value, ZoneInfo("Europe/Berlin")).strftime("%H:%M")


def _format_number(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def _format_int(value: float | int) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", ".")


def _format_expected_openings(visit_potential: dict[str, Any]) -> str:
    expected = _safe_int(visit_potential.get("expectedOpens"))
    if expected <= 0:
        return ""
    confidence = _safe_float(visit_potential.get("reachConfidence"))
    if confidence is None or confidence < 0.4:
        step = 1000 if expected >= 5000 else 500
        rounded = max(step, int(round(expected / step) * step))
        return (
            f"grob ca. {_format_int(rounded)} erwartete Öffnungen "
            "(modelliert; Reichweitenbasis unsicher)"
        )
    if confidence < 0.7:
        step = 500 if expected >= 2500 else 100
        rounded = max(step, int(round(expected / step) * step))
        return (
            f"ca. {_format_int(rounded)} erwartete Öffnungen "
            "(modellierte mittlere Reichweitensicherheit)"
        )
    return f"ca. {_format_int(expected)} erwartete Öffnungen (modelliert)"


def _format_or(value: float | None) -> str:
    return (
        f"{_format_number(float(value), 2)} % OR"
        if value is not None
        else "keine belastbare Prognose"
    )


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
        confidence = _safe_float(forecast.get("confidence"))
        confidence_label = " Modellkonfidenz: nicht ausgewiesen."
        if confidence is not None:
            confidence_percent = confidence * 100.0 if confidence <= 1.0 else confidence
            confidence_label = f" Modellkonfidenz: {_format_number(confidence_percent, 0)} %."
        return (
            f"Die Artikel-Prognose liegt aktuell bei {_format_or(float(forecast['value']))}."
            f"{confidence_label} Sie ist eine Schätzung, keine Garantie."
        )
    return (
        f"Die Zeitfenster-Prognose liegt aktuell bei {_format_or(float(forecast['value']))}. "
        "Sie stammt aus der historischen Slot-Baseline, nicht aus einem "
        "artikelindividuellen Modell."
    )


def _format_section(value: str) -> str:
    label = str(value or "").strip()
    if not label:
        return "News"
    known = {
        "politik": "Politik",
        "sport": "Sport",
        "unterhaltung": "Unterhaltung",
        "wirtschaft": "Wirtschaft",
        "geld": "Geld",
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
    "im fokus",
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
    "urteil",
    "verurteilt",
    "freigesprochen",
    "entscheidung",
    "festnahme",
    "festgenommen",
    "verhaftet",
    "razzia",
    "gesteht",
    "geständnis",
    "gestaendnis",
    "tot",
    "tote",
    "verletzte",
    "explosion",
    "brand",
    "anschlag",
    "angriff",
    "warnung",
    "evakuierung",
    "evakuiert",
    "rücktritt",
    "ruecktritt",
    "tritt zurück",
    "tritt zurueck",
    "zurückgetreten",
    "zurueckgetreten",
    "feuerpause",
    "waffenruhe",
    "einigung",
    "eskaliert",
    "bestätigt",
    "bestaetigt",
    "stoppt",
    "beendet",
    "abgesagt",
    "geschlossen",
    "gesperrt",
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
    "warnung",
    "gefahr",
    "evakuierung",
    "evakuiert",
    "sperrung",
    "ausfall",
    "rückruf",
    "rueckruf",
    "streik",
    "hochwasser",
    "unwetter",
    "hitzewarnung",
    "polizei",
    "terror",
    "anschlag",
    "angriff",
    "krieg",
    "feuerpause",
    "waffenruhe",
    "tote",
    "tot",
    "verletzte",
    "vermisst",
    "festnahme",
    "festgenommen",
    "razzia",
    "großrazzia",
    "grossrazzia",
    "leistungsbetrug",
    "leistungsbetrüger",
    "leistungsbetrueger",
    "sozialbetrug",
    "betrug",
    "betrüger",
    "betrueger",
    "urteil",
    "verurteilt",
    "rücktritt",
    "ruecktritt",
    "zurückgetreten",
    "zurueckgetreten",
    "insolvenz",
    "pleite",
)


def _is_public_money_fraud_enforcement(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    enforcement = (
        "razzia",
        "großrazzia",
        "grossrazzia",
        "polizei",
        "polizist",
        "polizisten",
        "beamte",
        "ermittler",
        "festnahme",
        "festgenommen",
    )
    fraud = (
        "leistungsbetrug",
        "leistungsbetrüger",
        "leistungsbetrueger",
        "sozialbetrug",
        "betrug",
        "betrüger",
        "betrueger",
        "bürgergeld",
        "buergergeld",
        "sozialleistung",
        "sozialleistungen",
    )
    return any(term in text for term in enforcement) and any(term in text for term in fraud)


def _public_money_fraud_or_near_miss(
    *,
    title: str,
    predicted_or: float,
    min_or: float,
    alert_score: float,
    min_alert_score: float,
) -> bool:
    return (
        _is_public_money_fraud_enforcement(title)
        and predicted_or >= max(4.75, min_or - 0.25)
        and alert_score >= min_alert_score - 1.5
    )


def _is_celebrity_relationship_money_conflict(title: str, section: str = "") -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    if not text:
        return False
    if str(section or "").strip().lower() == "sport":
        return False
    prominence = (
        "wm-held",
        "weltmeister",
        "star",
        "promi",
        "bastian",
        "schweini",
        "schweinsteiger",
        "becker",
        "bohlen",
        "helene",
        "gottschalk",
    )
    relationship_conflict = (
        "scheidung",
        "scheidungszoff",
        "trennung",
        "ehe-aus",
        "liebes-aus",
        "paar",
        "paare",
        "beziehung",
        "unterhalt",
    )
    money_conflict = (
        "geld",
        "vermögen",
        "vermoegen",
        "millionen",
        "unterhalt",
        "zoff",
        "streit",
        "scheidungszoff",
    )
    return (
        any(term in text for term in prominence)
        and any(term in text for term in relationship_conflict)
        and any(term in text for term in money_conflict)
    )


def _celebrity_conflict_or_near_miss(
    *,
    title: str,
    section: str,
    predicted_or: float,
    min_or: float,
    alert_score: float,
    min_alert_score: float,
) -> bool:
    return (
        _is_celebrity_relationship_money_conflict(title, section)
        and predicted_or >= max(4.75, min_or - 0.25)
        and alert_score >= min_alert_score - 1.5
    )


def _german_people_parenthood_or_near_miss(
    *,
    title: str,
    section: str,
    predicted_or: float,
    min_or: float,
    alert_score: float,
    min_alert_score: float,
) -> bool:
    """Allow only a narrow OR near-miss for a confirmed German People event."""
    return (
        is_german_public_figure_parenthood_story({"title": title, "category": section})
        and predicted_or >= max(4.75, min_or - 0.25)
        and alert_score >= min_alert_score
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
        term in text
        for term in (
            "regierung beschliesst",
            "regierung beschließt",
            "bundestag beschliesst",
            "bundestag beschließt",
        )
    )


_BROAD_PUBLIC_IMPACT_TERMS = (
    "deutschland",
    "bundesweit",
    "regierung",
    "bundestag",
    "kanzler",
    "krieg",
    "iran",
    "israel",
    "ukraine",
    "russland",
    "nato",
    "bahn",
    "deutsche bahn",
    "streik",
    "ausfall",
    "sperrung",
    "warnung",
    "rückruf",
    "rueckruf",
    "steuer",
    "rente",
    "krankenkasse",
    "preis",
    "geld",
    "polizei",
    "razzia",
    "großrazzia",
    "grossrazzia",
    "leistungsbetrug",
    "leistungsbetrüger",
    "leistungsbetrueger",
    "sozialbetrug",
    "betrug",
    "betrüger",
    "betrueger",
    "bürgergeld",
    "buergergeld",
    "gasversorgung",
    "terror",
    "anschlag",
    "angriff",
    "explosion",
    "brand",
    "tote",
    "verletzte",
    "vermisst",
    "evakuierung",
    "unwetter",
    "hochwasser",
    "hitzewarnung",
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
    "wohl",
    "offenbar",
    "angeblich",
    "vermutlich",
    "moeglicherweise",
    "möglicherweise",
    "koennte",
    "könnte",
    "duerfte",
    "dürfte",
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
    "tot",
    "tote",
    "toter",
    "getötet",
    "getoetet",
    "stirbt",
    "gestorben",
    "leiche",
    "opfer",
    "verletzt",
    "verletzte",
    "schwerverletzt",
    "attacke",
    "angriff",
    "anschlag",
    "terror",
    "schuss",
    "schüsse",
    "schuesse",
    "messer",
    "explosion",
    "explodiert",
    "brand",
    "brennt",
    "feuer",
    "unfall",
    "crash",
    "absturz",
    "ertrunken",
    "vermisst",
    "entführt",
    "entfuehrt",
    "überfall",
    "ueberfall",
    "amok",
    "drama",
    "tragödie",
    "tragoedie",
    "katastrophe",
    # Kriminalitaet / Justiz
    "festnahme",
    "festgenommen",
    "verhaftet",
    "razzia",
    "großrazzia",
    "grossrazzia",
    "urteil",
    "verurteilt",
    "gericht",
    "prozess",
    "anklage",
    "ermittl",
    "gesteht",
    "gestanden",
    "betrug",
    "betrüger",
    "betrueger",
    "leistungsbetrug",
    "leistungsbetrüger",
    "leistungsbetrueger",
    "sozialbetrug",
    # Promi / Beziehung
    "scheidung",
    "scheidungszoff",
    "trennung",
    "ehe-aus",
    "liebes-aus",
    "unterhalt",
    "mama geworden",
    "mamas geworden",
    "papa geworden",
    "papas geworden",
    "eltern geworden",
    "baby ist da",
    # Politik / Entscheidungen
    "beschließt",
    "beschliesst",
    "beschlossen",
    "stimmt",
    "abstimmung",
    "wahl",
    "gewählt",
    "gewaehlt",
    "rücktritt",
    "ruecktritt",
    "zurückgetreten",
    "zurueckgetreten",
    "tritt zurück",
    "tritt zurueck",
    "entlassen",
    "ernannt",
    "einigt",
    "einigen",
    "einigung",
    "gesetz",
    "verbietet",
    "verbot",
    "verhängt",
    "verhaengt",
    "sanktion",
    "kündigt an",
    "kuendigt an",
    "erklärt",
    "erklaert",
    "krieg",
    "waffenruhe",
    "feuerpause",
    "eskaliert",
    "droht",
    "warnt",
    "warnung",
    "regierungsbefragung",
    # Wirtschaft
    "insolvenz",
    "pleite",
    "entlassungen",
    "streik",
    "rekord",
    "rückruf",
    "rueckruf",
    "kollaps",
    "erhöht",
    "erhoeht",
    "erhöhen",
    "erhoehen",
    "steigt",
    "senkt",
    "kündigt",
    "kuendigt",
    "findet",
    "blackout",
    "totalausfall",
    "funkstörung",
    "funkstoerung",
    # Sport-Ereignisse
    "gewinnt",
    "gewonnen",
    "verliert",
    "verloren",
    "siegt",
    "niederlage",
    "wechselt",
    "wechsel",
    "verpflichtet",
    "gefeuert",
    "transfer",
    "ausfall",
    "ausgeschieden",
    "meister",
    "rekord",
    # Wetter / Natur
    "unwetter",
    "sturm",
    "hochwasser",
    "überflutung",
    "ueberflutung",
    "erdbeben",
    "hitzewarnung",
    "evakuiert",
    "evakuierung",
    # Meldung / Ankuendigung allgemein
    "meldet",
    "bestätigt",
    "bestaetigt",
    "ankündigung",
    "ankuendigung",
    "gibt bekannt",
    "stoppt",
    "räumt ein",
    "raeumt ein",
    "enthüllt",
    "enthuellt",
    "ist tot",
    "gestürzt",
    "gestuerzt",
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
    if _NEWS_EVENT_RE.search(text):
        return True
    lowered = text.casefold()
    compound_markers = (
        "streik",
        "ausfall",
        "warnung",
        "festnahme",
        "razzia",
        "angriff",
        "anschlag",
        "explosion",
        "evakuierung",
        "rueckruf",
        "rückruf",
        "verbot",
    )
    return any(marker in lowered for marker in compound_markers)


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
_RESIGNATION_TOPIC_CUES = (
    "rücktritt",
    "ruecktritt",
    "zurücktreten",
    "zurueck",
    "abdank",
    "amt nieder",
)
_RESIGNATION_DONE_CUES = (
    "zurückgetreten",
    "zurueckgetreten",
    "tritt zurück",
    "tritt zurueck",
    "ist zurückgetreten",
    "rücktritt erklärt",
    "ruecktritt erklaert",
    "tritt ab",
    "ist abgetreten",
    "nachfolger",
    "nachfolge steht",
    "resigns",
    "resigned",
    "steps down",
    "stepped down",
    "quits",
    "has quit",
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


def _public_push_title_review(review: dict[str, Any] | None) -> dict[str, Any]:
    """Minimized article-level title verdict safe for the Teams payload."""
    source = review or {}
    return {
        "approved": bool(source.get("approved")),
        "score": round(float(source.get("score") or 0.0), 1),
        "clickReason": str(source.get("clickReason") or "").strip(),
    }


def _recommendation_timing_brief(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    *,
    now_ts: int,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Turn slot evidence into one exact, human-readable dispatch window."""
    local_dt = dt.datetime.fromtimestamp(int(now_ts), ZoneInfo("Europe/Berlin"))
    editorial = decision.get("editorialReview")
    editorial = editorial if isinstance(editorial, dict) else {}
    breakdown = editorial.get("breakdown")
    breakdown = breakdown if isinstance(breakdown, dict) else {}
    slot_gate = decision.get("slotGate")
    slot_gate = slot_gate if isinstance(slot_gate, dict) else {}
    slot = slot_gate.get("slot")
    slot = slot if isinstance(slot, dict) else {}
    mode = str(slot_gate.get("mode") or "").strip()
    breaking = bool(decision.get("isBreaking") or _is_breaking(candidate))
    fast_window = breaking or mode in {"breaking_override", "sport_event_override"}
    window_minutes = 3 if fast_window else 5
    send_by = local_dt + dt.timedelta(minutes=window_minutes)

    time_fit_score = float(breakdown.get("timeFit") or 0.0)
    slot_avg_or = _safe_float(slot.get("avgOR"))
    if slot_avg_or is None:
        slot_avg_or = _safe_float(breakdown.get("slotAvgOR"))
    slot_top_category = str(
        slot.get("topCategory") or breakdown.get("slotTopCategory") or ""
    ).strip()
    slot_top_category_key = _section_key(slot_top_category) if slot_top_category else ""
    section = _section(candidate)
    section_key = _section_key(section)
    slot_hour = int(slot.get("hour")) if slot.get("hour") is not None else local_dt.hour
    weekday_names = (
        "Montag",
        "Dienstag",
        "Mittwoch",
        "Donnerstag",
        "Freitag",
        "Samstag",
        "Sonntag",
    )
    weekday = weekday_names[local_dt.weekday()]

    timing_score = time_fit_score * 10.0
    if slot_avg_or is not None:
        if slot_avg_or >= float(config.peak_slot_min_or or 6.0):
            timing_score += 4.0
        elif slot_avg_or < 5.0:
            timing_score -= 8.0
    if slot_gate.get("enabled") and not slot_gate.get("approved"):
        timing_score = min(timing_score, 35.0)
    mode_floors = {
        "breaking_override": 92.0,
        "sport_event_override": 90.0,
        "peak_early_exception": 88.0,
        "peak_double_opportunity": 84.0,
        "peak_catchup_first": 82.0,
        "projected_shortfall_catchup": 80.0,
        "deadline_fallback": 76.0,
    }
    if slot_gate.get("approved") and mode in mode_floors:
        timing_score = max(timing_score, mode_floors[mode])
    timing_score = round(_clamp(timing_score, 0.0, 100.0), 1)

    reasons: list[str] = []
    mode_reason = {
        "breaking_override": "Breaking-Lage: Die Nachricht verliert bei weiterem Warten an Wert.",
        "sport_event_override": "Frische Sport-Lage: Das Ereignis ist jetzt materiell und pushwürdig.",
        "peak_early_exception": "Außergewöhnlich starker Kandidat im aktuellen Goldfenster.",
        "peak_double_opportunity": "Zusätzliche Qualitätschance in einem starken Doppel-Slot.",
        "peak_catchup_first": "Fälliges Tagesfenster wird mit dem stärksten verfügbaren Kandidaten besetzt.",
        "projected_shortfall_catchup": (
            "Shortfall-Recovery: Ohne eine zusätzliche Entscheidung nach Cool-down "
            "ist das Tagesminimum nicht mehr erreichbar."
        ),
        "deadline_fallback": "Das geplante Mindestfenster ist fällig; der stärkste nicht gesperrte Kandidat gewinnt.",
    }.get(mode)
    if mode_reason:
        reasons.append(mode_reason)
    if slot_avg_or is not None:
        reasons.append(
            f"{weekday} um {slot_hour:02d} Uhr erreicht historisch "
            f"{_format_number(slot_avg_or, 2)} % OR."
        )
    if slot_top_category_key and _sections_match(section_key, slot_top_category_key):
        reasons.append(
            f"{_format_section(section)} ist das historisch passende Ressort für diesen Slot."
        )
    elif slot_top_category_key:
        reasons.append(
            f"Slot-Schwerpunkt ist {_format_section(slot_top_category)}; "
            f"{_format_section(section)} muss deshalb über die Story-Stärke überzeugen."
        )
    publication = decision.get("publicationReview")
    publication = publication if isinstance(publication, dict) else {}
    age_hours = _safe_float(publication.get("ageHours"))
    if age_hours is not None:
        age_minutes = max(0, int(round(age_hours * 60.0)))
        if age_minutes < 60:
            reasons.append(f"Der Artikel ist erst {age_minutes} Minuten alt und noch unverbraucht.")
        elif age_hours <= 3.0:
            reasons.append(
                f"Der Artikel ist seit {_format_number(age_hours, 1)} Stunden live und noch aktuell."
            )
    if not reasons:
        label = str(breakdown.get("timeFitLabel") or "aktuelles Zeitfenster").strip()
        reasons.append(f"Timing-Prüfung: {label}.")

    action = "Sofort senden" if fast_window else "Jetzt senden"
    return {
        "approved": bool(not slot_gate.get("enabled") or slot_gate.get("approved") or breaking),
        "score": timing_score,
        "mode": mode or ("breaking" if breaking else "ungated"),
        "action": action,
        "windowMinutes": window_minutes,
        "windowLabel": f"{action}, ideal bis {send_by.strftime('%H:%M')} Uhr",
        "sendBy": send_by.isoformat(),
        "sendByLabel": send_by.strftime("%H:%M"),
        "slotAvgOR": round(slot_avg_or, 2) if slot_avg_or is not None else None,
        "slotTopCategory": slot_top_category or None,
        "slotHour": slot_hour,
        "weekday": local_dt.weekday(),
        "hour": local_dt.hour,
        "reasons": _dedupe(reasons)[:4],
    }


def _recommendation_decision_basis(timing: dict[str, Any]) -> str:
    """Explain why this recommendation may be presented as an active send."""
    mode = str(timing.get("mode") or "").strip()
    return {
        "breaking_override": (
            "Breaking-Vollfreigabe: Aktualität rechtfertigt den sofortigen Versand; "
            "Dubletten- und Cooldown-Gates bleiben aktiv."
        ),
        "sport_event_override": (
            "Frische materielle Sportlage: Ereignis- und Aktualitätsgate sind erfüllt."
        ),
        "peak_early_exception": (
            "Frühe Peak-Freigabe: außergewöhnlicher Kandidat mit vollständiger Qualitätsfreigabe."
        ),
        "peak_double_opportunity": (
            "Optionale Doppelchance: nur wegen überdurchschnittlicher Qualität im starken Slot."
        ),
        "peak_catchup_first": (
            "Frühe Aufholchance im ausgewiesenen Doppel-Slot; normale Qualitätsgates gelten unverändert."
        ),
        "projected_shortfall_catchup": (
            "Shortfall-Recovery: Die verbleibenden Pflichtfenster reichen nicht mehr; "
            "bester freigegebener Kandidat ab Push-Score 75 nach Cool-down."
        ),
        "deadline_fallback": (
            "Mindestfenster-Auswahl: bester freigegebener Kandidat ab Push-Score 75; "
            "harte Fakten-, Aktualitäts-, Titel-, Ruhezeit- und Dublettengates sind erfüllt."
        ),
    }.get(
        mode,
        "Reguläre Vollfreigabe: Artikel, Titel, Zeitpunkt und Nutzerbelastung sind geprüft.",
    )


def _recommendation_quality_review(
    candidate: dict[str, Any],
    decision: dict[str, Any],
    title_review: dict[str, Any],
    timing: dict[str, Any],
    *,
    config: TeamsAlertConfig,
) -> dict[str, Any]:
    """Issue the final CvD verdict after every specialist check has completed."""
    editorial = decision.get("editorialReview")
    editorial = editorial if isinstance(editorial, dict) else {}
    forecast = decision.get("forecast")
    forecast = forecast if isinstance(forecast, dict) else {}
    agent = decision.get("agentReview")
    agent = agent if isinstance(agent, dict) else {}
    competition = decision.get("competition")
    competition = competition if isinstance(competition, dict) else {}
    slot_gate = decision.get("slotGate")
    slot_gate = slot_gate if isinstance(slot_gate, dict) else {}
    germany_relevance = decision.get("germanyRelevance")
    germany_relevance = germany_relevance if isinstance(germany_relevance, dict) else {}
    minimum_commitment = bool(
        slot_gate.get("minimumCommitment")
        and (decision.get("deadlineFallback") or {}).get("approved")
    )
    high_score_override = decision.get("highScoreOverride")
    high_score_override = (
        high_score_override if isinstance(high_score_override, dict) else {}
    )
    high_score_override_approved = bool(high_score_override.get("approved"))

    push_score = _clamp(
        float(decision.get("score") or _score(candidate)),
        0.0,
        100.0,
    )
    push_score_threshold = float(
        decision.get("minScore")
        or (
            config.breaking_min_score
            if bool(decision.get("isBreaking") or _is_breaking(candidate))
            else config.min_score
        )
    )
    editorial_score = _clamp(float(editorial.get("score") or 0.0), 0.0, 100.0)
    predicted_or = _safe_float(forecast.get("value"))
    if predicted_or is None:
        predicted_or = _safe_float(decision.get("predictedOR"))
    forecast_confidence = _safe_float(forecast.get("confidence"))
    if forecast_confidence is None:
        forecast_confidence = 0.75 if forecast.get("source") == "article_model" else 0.45
    forecast_confidence = _clamp(forecast_confidence, 0.0, 1.0)
    if predicted_or is None:
        forecast_score = 28.0
    else:
        or_strength = _clamp(
            65.0 + (predicted_or - float(config.min_or or 0.0)) * 18.0,
            30.0,
            100.0,
        )
        forecast_score = 0.68 * or_strength + 0.32 * forecast_confidence * 100.0

    title_score = _clamp(float(title_review.get("score") or 0.0), 0.0, 100.0)
    agent_enabled = bool(agent.get("enabled"))
    agent_score = (
        _clamp(float(agent.get("consensusScore") or 0.0), 0.0, 100.0) if agent_enabled else None
    )

    competitors = int(competition.get("eligibleCompetitors") or 0)
    margin_percent = _safe_float(competition.get("selectionMarginPercent"))
    field_confidence = str(competition.get("selectionConfidence") or "").strip()
    if competitors <= 0:
        competition_score = 86.0
        field_confidence = field_confidence or "hoch"
    elif margin_percent is not None and margin_percent >= 15.0:
        competition_score = 96.0
    elif margin_percent is not None and margin_percent >= 5.0:
        competition_score = 82.0
    else:
        competition_score = 58.0
        field_confidence = field_confidence or "niedrig"

    dimensions = {
        "pushScore": round(push_score, 1),
        "articleStrength": round(editorial_score, 1),
        "orForecast": round(_clamp(forecast_score, 0.0, 100.0), 1),
        "timing": round(float(timing.get("score") or 0.0), 1),
        "title": round(title_score, 1),
        "agentConsensus": round(agent_score, 1) if agent_score is not None else None,
        "candidateField": round(competition_score, 1),
        "germanyRelevance": {
            "germany_broad": 100.0,
            "germany_domestic": 82.0,
            "germany_people": 90.0,
            "neutral": 65.0,
            "international_breaking": 72.0,
            "international": 40.0,
            "usa_domestic": 0.0,
        }.get(str(germany_relevance.get("level") or "neutral"), 65.0),
    }
    if minimum_commitment:
        weighted_dimensions = [
            (dimensions["pushScore"], 0.55),
            (dimensions["articleStrength"], 0.08),
            (dimensions["orForecast"], 0.06),
            (dimensions["timing"], 0.08),
            (dimensions["title"], 0.05),
            (dimensions["candidateField"], 0.02),
            (dimensions["germanyRelevance"], 0.06),
        ]
    else:
        weighted_dimensions = [
            (dimensions["pushScore"], 0.50),
            (dimensions["articleStrength"], 0.10),
            (dimensions["orForecast"], 0.08),
            (dimensions["timing"], 0.09),
            (dimensions["title"], 0.06),
            (dimensions["candidateField"], 0.02),
            (dimensions["germanyRelevance"], 0.05),
        ]
    if agent_score is not None:
        weighted_dimensions.append((dimensions["agentConsensus"], 0.10))
    active_weight = sum(weight for _, weight in weighted_dimensions)
    quality_score = round(
        sum(float(score) * weight for score, weight in weighted_dimensions) / active_weight,
        1,
    )
    threshold = float(config.min_recommendation_quality or 72.0)
    breaking = bool(decision.get("isBreaking") or _is_breaking(candidate))
    hard_public_need = _has_hard_public_need(
        _title(candidate),
        _section_key(_section(candidate)),
    )
    blockers: list[str] = []
    if push_score < push_score_threshold:
        blockers.append(
            f"Push-Score {push_score:.1f} liegt unter der harten Freigabeschwelle "
            f"{push_score_threshold:.1f}."
        )
    if not title_review.get("approved"):
        blockers.append("Der Push-Titel erzeugt noch keinen ehrlichen, konkreten Klickgrund.")
    if agent.get("enabled") and not agent.get("approved"):
        blockers.append(str(agent.get("blockingReason") or "Der Agenten-Konsens fehlt."))
    if not timing.get("approved") and not breaking:
        blockers.append("Das aktuelle Versandfenster ist nicht freigegeben.")
    if dimensions["timing"] < 55.0 and not breaking:
        blockers.append("Der Zeitpunkt ist für eine maximale OR zu schwach.")
    if dimensions["orForecast"] < 45.0 and not (
        breaking or hard_public_need or minimum_commitment or high_score_override_approved
    ):
        blockers.append("Die OR-Erwartung ist für eine klare Empfehlung nicht belastbar genug.")
    if quality_score < threshold and not high_score_override_approved:
        blockers.append(
            f"Gesamtqualität {quality_score:.1f} liegt unter der Freigabeschwelle {threshold:.1f}."
        )

    risk_candidates = [
        str(agent.get("mainCounterargument") or "").strip(),
        *_editorial_list(candidate, "risks"),
    ]
    if field_confidence == "niedrig":
        risk_candidates.append("Der Vorsprung zum zweitbesten Kandidaten ist knapp.")
    if forecast.get("source") != "article_model":
        risk_candidates.append(
            "Die OR-Prognose basiert auf dem historischen Slot und nicht auf einem Artikelsignal."
        )
    strongest_risk = next((item for item in _dedupe(risk_candidates) if item), "")
    if not strongest_risk:
        strongest_risk = "Kein harter Einwand; die OR-Prognose bleibt eine Schätzung."

    if quality_score >= 84.0 and field_confidence != "niedrig":
        confidence = "hoch"
    elif quality_score >= threshold or high_score_override_approved:
        confidence = "mittel"
    else:
        confidence = "niedrig"
    enforced = True
    return {
        "approved": not blockers,
        "enforced": enforced,
        "score": quality_score,
        "threshold": round(threshold, 1),
        "minimumCommitment": minimum_commitment,
        "highScoreOverrideApplied": high_score_override_approved,
        "confidence": confidence,
        "dimensions": dimensions,
        "blockers": _dedupe(blockers),
        "strongestRisk": strongest_risk,
        "timing": timing,
        "summary": (
            f"{quality_score:.0f}/100, Empfehlungsstärke {confidence}; "
            f"{str(timing.get('windowLabel') or 'jetzt senden')}"
        ),
    }


def _public_recommendation_review(review: dict[str, Any] | None) -> dict[str, Any]:
    """Minimize the final verdict before it crosses the Teams boundary."""
    source = review or {}
    timing = source.get("timing")
    timing = timing if isinstance(timing, dict) else {}
    return {
        "approved": bool(source.get("approved")),
        "score": round(float(source.get("score") or 0.0), 1),
        "confidence": str(source.get("confidence") or "niedrig"),
        "window": {
            "label": str(timing.get("windowLabel") or "").strip(),
            "sendBy": str(timing.get("sendBy") or "").strip(),
            "slotAvgOR": timing.get("slotAvgOR"),
            "slotTopCategory": timing.get("slotTopCategory"),
        },
    }


def _teams_push_title_recommendation(
    candidate: dict[str, Any],
    title: str,
    section: str,
    url: str,
    config: TeamsAlertConfig | None = None,
) -> tuple[str, str]:
    """Return the strongest grounded title after the local title jury."""
    push_title, source, _review = _teams_push_title_selection(
        candidate,
        title,
        section,
        url,
        config,
    )
    return push_title, source


def _teams_push_title_selection(
    candidate: dict[str, Any],
    title: str,
    section: str,
    url: str,
    config: TeamsAlertConfig | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Compare every title source for relevance, honest curiosity and grounding."""
    config = config or TeamsAlertConfig()
    breaking = _is_breaking(candidate)
    title_pool: list[tuple[str, str]] = []

    # A model suggestion is one candidate, never an automatic winner.
    llm_title = _llm_push_title(title, section, url, config) if config.llm_title_enabled else ""
    if llm_title:
        title_pool.append((llm_title, "llm"))

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
            explicit_candidates.extend(
                str(item or "").strip() for item in values if str(item or "").strip()
            )

    for value in _dedupe(explicit_candidates):
        title_pool.append((value, "editorial"))

    try:
        from app.push_titles import build_push_title_suggestions

        result = build_push_title_suggestions(title, category=section, url=url)
        generated = [
            str(result.get("title") or "").strip(),
            *[str(item or "").strip() for item in result.get("alternativeTitles", [])],
            str((result.get("alternative") or {}).get("titel") or "").strip(),
        ]
        for value in _dedupe([item for item in generated if item]):
            title_pool.append((value, "editorial"))
    except Exception as exc:
        log.warning("[TeamsAlert] could not build alternative push title: %s", exc)

    title_pool.append((title, "headline"))
    reviewed: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        from app.push_titles import review_push_title

        for value, source in title_pool:
            clean = _sanitize_push_title(value, breaking=breaking)
            if not clean or _is_generic_push_title(clean):
                continue
            clean = _compact_text(clean, 100)
            key = _normalize_title(clean)
            if not key or key in seen:
                continue
            seen.add(key)
            review = review_push_title(
                clean,
                original_title=title,
                category=section or "news",
                url=url,
            )
            reviewed.append(
                {
                    "title": clean,
                    "source": source,
                    "review": review,
                    "distinct": not _same_editorial_text(clean, title),
                }
            )
    except Exception as exc:
        log.warning("[TeamsAlert] title jury unavailable: %s", exc)

    if reviewed:
        source_priority = {"llm": 2, "editorial": 1, "headline": 0}
        reviewed.sort(
            key=lambda item: (
                bool(item["review"].get("approved")),
                float(item["review"].get("score") or 0.0),
                bool(item["distinct"]),
                source_priority.get(str(item["source"]), 0),
            ),
            reverse=True,
        )
        winner = reviewed[0]
        return winner["title"], winner["source"], winner["review"]

    # Defensive legacy fallback if the local jury itself is unavailable.
    for value, source in title_pool:
        clean = _sanitize_push_title(value, breaking=breaking)
        if clean and not _is_generic_push_title(clean):
            return (
                _compact_text(clean, 100),
                source,
                {
                    "approved": False,
                    "score": 0.0,
                    "clickReason": "",
                    "risks": ["Titelprüfung war technisch nicht verfügbar"],
                },
            )
    fallback = _sanitize_push_title(
        title or (explicit_candidates[0] if explicit_candidates else ""), breaking=breaking
    )
    return (
        _compact_text(fallback, 100),
        "headline",
        {
            "approved": False,
            "score": 0.0,
            "clickReason": "",
            "risks": ["Kein belastbarer Push-Titel vorhanden"],
        },
    )


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


def _agent_review_display_summary(review: dict[str, Any] | None) -> str:
    source = review or {}
    if not source.get("enabled"):
        return ""
    agent_count = int(source.get("agentCount") or 0)
    evidence_approved = int(source.get("evidenceApprovalCount") or 0)
    evidence_count = int(source.get("evidenceReviewerCount") or 0)
    hard_vetoes = int(source.get("hardVetoCount") or 0)
    veto_label = "keine harten Vetos" if hard_vetoes == 0 else f"{hard_vetoes} harte Vetos"
    if agent_count <= 0:
        return str(source.get("summary") or "").strip()
    return (
        f"{agent_count} lokale Checks | Evidenz "
        f"{evidence_approved}/{evidence_count} | {veto_label}"
    )


def _format_minutes(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"vor {float(value):.0f} Minuten"
    return "kein letzter Push bekannt"


def _format_teams_alert_minutes(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"vor {float(value):.0f} Minuten"
    return "noch keiner bekannt"


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
    minutes_since_last_teams_alert: Any,
    independent_pacing: bool,
    score_threshold: float,
    score_source_label: str,
    alert_score: float,
    alert_threshold: float,
    editorial_score: float,
    why_now: list[str],
    why_pushworthy: list[str],
    subject: str,
    push_text_matches_title: bool = False,
    score_reason: str = "",
    performance_drivers: list[str] | None = None,
    risks: list[str] | None = None,
    score_breakdown_lines: list[str] | None = None,
    agent_review: dict[str, Any] | None = None,
    push_title_review: dict[str, Any] | None = None,
    recommendation_review: dict[str, Any] | None = None,
    live_push_comparison: dict[str, Any] | None = None,
    dispatch_approved: bool = False,
    decision_basis: str = "",
    dispatch_blocking_reason: str = "",
) -> str:
    article_html = (
        f'<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
        if url
        else html.escape(title)
    )
    why_now_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in why_now)
    why_pushworthy_html = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in (why_pushworthy or [])
    )
    lead_html = (
        "<p><strong>Empfohlener Push-Titel:</strong><br>"
        f"{html.escape(recommended_text)}</p>"
        "<p><strong>Artikel:</strong><br>"
        f"{article_html}</p>"
    )
    review = agent_review or {}
    review_summary = _agent_review_display_summary(review)
    review_html = ""
    if review_summary:
        review_html = "<p><strong>Prüfstatus:</strong> " f"{html.escape(review_summary)}" + "</p>"
    title_review = push_title_review or {}
    title_review_reason = str(title_review.get("clickReason") or "").strip()
    if title_review_reason:
        title_review_reason = title_review_reason[:1].upper() + title_review_reason[1:]
    title_review_html = ""
    if title_review:
        title_review_html = (
            "<p><strong>Titelqualität:</strong> "
            f"{html.escape(_format_number(float(title_review.get('score') or 0.0), 0))}/100"
            + (f"<br>{html.escape(title_review_reason)}" if title_review_reason else "")
            + "</p>"
        )
    quality = recommendation_review or {}
    quality_timing = quality.get("timing")
    quality_timing = quality_timing if isinstance(quality_timing, dict) else {}
    quality_score = _format_number(float(quality.get("score") or 0.0), 0)
    quality_confidence = str(quality.get("confidence") or "niedrig")
    window_label = str(quality_timing.get("windowLabel") or "Jetzt senden").strip()
    strongest_risk = str(quality.get("strongestRisk") or "").strip()
    live_comparison = live_push_comparison or {}
    if not live_comparison.get("available"):
        live_comparison_label = "aktuell nicht belastbar verfügbar"
    elif live_comparison.get("matched"):
        live_comparison_label = "entspricht einem echten Live-Push"
    else:
        live_comparison_label = "zum Prüfzeitpunkt kein entsprechender Live-Push"
    recommendation_html = (
        "<p><strong>Empfehlung:</strong> Jetzt pushen.</p>"
        if dispatch_approved
        else (
            "<p><strong>Empfehlung:</strong> Nicht senden.<br>"
            f"{html.escape(dispatch_blocking_reason)}</p>"
        )
    )
    cadence_html = (
        "<strong>Letzter Teams-Hinweis:</strong> "
        f"{html.escape(_format_teams_alert_minutes(minutes_since_last_teams_alert))}<br>"
        "<strong>Takt:</strong> unabhängig von echten Live-Pushes"
        if independent_pacing
        else (
            "<strong>Letzter Push:</strong> "
            f"{html.escape(_format_minutes(minutes_since_last_push))}"
        )
    )
    return (
        f"<h2>{html.escape(subject)}</h2>"
        f"{lead_html}"
        "<p>"
        f"<strong>Versandfenster:</strong> {html.escape(window_label)}<br>"
        f"<strong>Push-Score:</strong> {html.escape(_format_number(score, 1))}/100 "
        f"(harte Schwelle {html.escape(_format_number(score_threshold, 0))})<br>"
        f"<strong>Score-Quelle:</strong> {html.escape(score_source_label)}<br>"
        f"<strong>Qualitätsurteil:</strong> {html.escape(quality_score)}/100, "
        f"Empfehlungsstärke {html.escape(quality_confidence)}<br>"
        f"<strong>Entscheidungsbasis:</strong> {html.escape(decision_basis)}"
        "</p>"
        f"{review_html}"
        f"{title_review_html}"
        "<p>"
        f"<strong>Ressort:</strong> {html.escape(section)}<br>"
        f"<strong>Prognose:</strong> {html.escape(_format_forecast(forecast))}<br>"
        f"<strong>Live-Vergleich:</strong> {html.escape(live_comparison_label)}<br>"
        f"{cadence_html}<br>"
        f"<strong>Stand:</strong> {html.escape(_format_time(now_ts))} Uhr"
        "</p>"
        "<p><strong>Warum dieser Push?</strong></p>"
        f"<ul>{why_pushworthy_html}</ul>"
        "<p><strong>Warum jetzt?</strong></p>"
        f"<ul>{why_now_html}</ul>"
        "<p><strong>Gegencheck:</strong><br>"
        f"{html.escape(strongest_risk)}</p>"
        f"{recommendation_html}"
    )


def _normalize_url(url: str) -> str:
    return url.strip().split("?", 1)[0].rstrip("/").lower()


def _article_identity_url(url: str) -> str:
    """Canonical URL identity for real-push deduplication only."""
    from urllib.parse import urlsplit

    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").casefold()
    if host == "bild.de" or host.endswith(".bild.de"):
        host = "bild.de"
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/").casefold()
    path = re.sub(r"/(?:amp|amphtml)$", "", path).rstrip("/")
    return f"{host}{path}" if host else path


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
