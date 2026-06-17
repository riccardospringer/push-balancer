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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from app.config import (
    PUSH_TEAMS_ALERT_COOLDOWN_MINUTES,
    PUSH_TEAMS_ALERTS_ENABLED,
    PUSH_TEAMS_ALLOWED_SECTIONS,
    PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_BREAKING_MIN_OR,
    PUSH_TEAMS_BREAKING_MIN_SCORE,
    PUSH_TEAMS_BREAKING_OVERRIDE,
    PUSH_TEAMS_CANDIDATE_LIMIT,
    PUSH_TEAMS_DASHBOARD_TOP_LIMIT,
    PUSH_TEAMS_EDITORIAL_GATE_ENABLED,
    PUSH_TEAMS_EDITORIAL_TOP_LIMIT,
    PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES,
    PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS,
    PUSH_TEAMS_MAX_PUSHES_LAST_6H,
    PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE,
    PUSH_TEAMS_MIN_EDITORIAL_SCORE,
    PUSH_TEAMS_MIN_ALERT_SCORE,
    PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE,
    PUSH_TEAMS_MIN_OR,
    PUSH_TEAMS_MIN_SCORE,
    PUSH_TEAMS_MIN_TIME_FIT_SCORE,
    PUSH_TEAMS_REALERT_OR_DELTA,
    PUSH_TEAMS_REALERT_SCORE_DELTA,
    PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS,
    PUSH_TEAMS_SCORE_ONLY_MODE,
    PUSH_TEAMS_WEBHOOK_URL,
)
from app.database import (
    push_db_load_all,
    teams_alert_last_sent_ts,
    teams_alert_load_for_keys,
    teams_alert_record,
    teams_alert_try_claim_send,
)

log = logging.getLogger("push-balancer")

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
    min_editorial_score: float = PUSH_TEAMS_MIN_EDITORIAL_SCORE
    min_editorial_news_value: float = PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE
    min_time_fit_score: float = PUSH_TEAMS_MIN_TIME_FIT_SCORE
    min_or: float = PUSH_TEAMS_MIN_OR
    min_minutes_since_last_push: int = PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH
    realert_score_delta: float = PUSH_TEAMS_REALERT_SCORE_DELTA
    realert_or_delta: float = PUSH_TEAMS_REALERT_OR_DELTA
    alert_cooldown_minutes: int = PUSH_TEAMS_ALERT_COOLDOWN_MINUTES
    repeat_suppression_hours: int = PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS
    global_cooldown_minutes: int = PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES
    allowed_sections: tuple[str, ...] = tuple(PUSH_TEAMS_ALLOWED_SECTIONS)
    breaking_override: bool = PUSH_TEAMS_BREAKING_OVERRIDE
    breaking_min_score: float = PUSH_TEAMS_BREAKING_MIN_SCORE
    breaking_min_or: float = PUSH_TEAMS_BREAKING_MIN_OR
    breaking_min_minutes_since_last_push: int = PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH
    max_article_age_hours: int = PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS
    max_pushes_last_6h: int = PUSH_TEAMS_MAX_PUSHES_LAST_6H


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


def build_teams_alert_context(
    candidates: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
    alert_state: dict[str, dict[str, Any]] | None = None,
    last_teams_alert_ts: int | None = None,
    now_ts: int | None = None,
) -> dict[str, Any]:
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

    return {
        "nowTs": now,
        "history": history,
        "alertState": alert_state,
        "lastPushTs": last_push_ts,
        "lastTeamsAlertTs": int(last_teams_alert_ts or 0),
        "recentPushCount6h": recent_6h_count,
    }


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
    forecast = _candidate_forecast(candidate, now_ts)
    predicted_or = forecast["value"]
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

    if not config.enabled:
        blockers.append("Teams Alerts deaktiviert")

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

    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if allowed and section.lower() not in allowed:
        blockers.append(f"Ressort {section} nicht fuer Teams Alerts freigegeben")

    if score >= min_score:
        positive.append(f"Push Score {score:.1f} liegt ueber Schwelle {min_score:.1f}")
    else:
        blockers.append(f"Score zu niedrig: {score:.1f} < {min_score:.1f}")

    alert_model = _teams_alert_score(
        candidate,
        score=score,
        predicted_or=predicted_or,
        freshness_hours=_freshness_hours(candidate, now_ts),
        minutes_since_last_push=minutes_since_last_push,
        recent_push_count_6h=int(context.get("recentPushCount6h") or 0),
        config=config,
    )
    alert_score = float(alert_model["score"])
    positive.append(f"Teams Alert Score {alert_score:.1f}/100")
    positive.extend(list(alert_model["reasons"])[:4])
    if alert_score < config.min_alert_score:
        blockers.append(
            f"Teams Alert Score zu niedrig: {alert_score:.1f} < {config.min_alert_score:.1f}"
        )
    if predicted_or is None and alert_score < config.no_forecast_min_alert_score:
        blockers.append(
            "Keine belastbare Prognose und Teams Alert Score nicht hoch genug: "
            f"{alert_score:.1f} < {config.no_forecast_min_alert_score:.1f}"
        )

    freshness_hours = _freshness_hours(candidate, now_ts)
    editorial_review = _editorial_cvd_review(
        candidate,
        score=score,
        predicted_or=predicted_or,
        freshness_hours=freshness_hours,
        minutes_since_last_push=minutes_since_last_push,
        dashboard_rank=dashboard_rank,
        alert_score=alert_score,
        now_ts=now_ts,
        config=config,
    )
    positive.extend(editorial_review["reasons"])
    blockers.extend(editorial_review["blockers"])
    selection_score = _recommendation_selection_score(
        score=score,
        alert_score=alert_score,
        editorial_score=float(editorial_review["score"]),
        predicted_or=predicted_or,
        dashboard_rank=dashboard_rank,
        breaking=breaking,
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
        elif predicted_or >= min_or:
            positive.append(f"Prognose {predicted_or:.2f}% OR liegt ueber Mindestwert {min_or:.2f}%")
        else:
            blockers.append(f"Prognose zu niedrig: {predicted_or:.2f}% OR < {min_or:.2f}%")

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

    if (
        not config.score_only_mode
        and int(context.get("recentPushCount6h") or 0) > config.max_pushes_last_6h
        and not breaking
    ):
        blockers.append("Push-Dichte in den letzten 6 Stunden zu hoch")

    duplicate_reason = _already_pushed_reason(candidate, context.get("history") or [])
    if duplicate_reason:
        blockers.append(duplicate_reason)

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

    if (
        config.global_cooldown_minutes > 0
        and minutes_since_last_teams_alert is not None
        and minutes_since_last_teams_alert < config.global_cooldown_minutes
    ):
        blockers.append(
            "Teams-Cooldown aktiv: letzter Hinweis vor "
            f"{minutes_since_last_teams_alert:.0f} < {config.global_cooldown_minutes} Minuten"
        )
        status = "observe"

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
        "score": score,
        "teamsAlertScore": alert_score,
        "teamsAlertScoreThreshold": config.min_alert_score,
        "teamsAlertScoreBreakdown": alert_model["breakdown"],
        "editorialReview": editorial_review,
        "editorialScore": editorial_review["score"],
        "selectionScore": selection_score,
        "predictedOR": predicted_or,
        "forecast": forecast,
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
    context = context or build_teams_alert_context(candidates)
    config = config or TeamsAlertConfig()
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
    if eligible:
        selected_candidate, _selected_decision = max(
            eligible,
            key=lambda item: (
                float(item[1].get("selectionScore") or 0.0),
                float(item[1].get("editorialScore") or 0.0),
                float(item[1].get("teamsAlertScore") or 0.0),
                *_candidate_rank(item[0]),
            ),
        )
        selected_key = candidate_key(selected_candidate)

    final: list[dict[str, Any]] = []
    for index, (candidate, base_decision) in enumerate(base_decisions, start=1):
        key = candidate_key(candidate)
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
                    decision["reasons"] = [
                        *decision.get("reasons", []),
                        "Beste CvD-Eignung aus Nachrichtenwert, Timing und Nutzerbelastung im Kandidatenfeld",
                    ]
        final.append({"candidate": candidate, "decision": decision})

    return {
        "selectedCandidateId": selected_key,
        "selectedCandidate": selected_candidate,
        "decisions": final,
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
    context = build_teams_alert_context(candidates, now_ts=now_ts)
    evaluations = evaluate_teams_alert_candidates(candidates, context, config or TeamsAlertConfig())
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
        forecast = _candidate_forecast(candidate, now_ts)
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
    selection_score = float(decision.get("selectionScore") or 0.0)
    push_text = str(candidate.get("recommendedText") or title)
    push_text_matches_title = _same_editorial_text(push_text, title)
    competition_meta = decision.get("competition") or {}
    competitors = int(competition_meta.get("eligibleCompetitors") or 0)
    competition = (
        f"Im aktuellen Kandidatenfeld ist das der stärkste Vorschlag ({competitors + 1} geprüft)."
        if competitors
        else "Im aktuellen Kandidatenfeld gibt es keinen stärkeren Push-Vorschlag."
    )

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
    time_fit_reason = (
        f"Zeitfenster: {time_fit_label} ({_format_number(time_fit_score)} von 10)."
        if time_fit_label
        else ""
    )
    why_now = _dedupe(
        [
            editorial_reason,
            time_fit_reason,
            forecast_reason,
            threshold_reason,
            score_reason,
            timing_reason,
            competition,
            duplicate_reason,
        ]
    )[:6]
    why_pushworthy = _dedupe([editorial_reason, score_reason, forecast_reason, timing_reason, competition])[:4]

    text_lines = ["🚨 Push-Empfehlung: Jetzt pushen", ""]
    if push_text_matches_title:
        text_lines.extend(["Artikel und empfohlener Push:", f"{title}\n{url}" if url else title])
    else:
        text_lines.extend(
            [
                "Empfohlener Push-Text:",
                push_text,
                "",
                "Artikel:",
                f"{title}\n{url}" if url else title,
            ]
        )
    text_lines.extend(
        [
            "",
            "Warum jetzt?",
            *[f"- {reason}" for reason in why_now],
            "",
            "Einordnung:",
            f"- Ressort: {section_label}",
            f"- Push-Score: {_format_number(score)} (Mindestwert: {_format_number(score_threshold, 0)})",
            f"- Teams-Alert-Score: {_format_number(alert_score)} (Schwelle: {_format_number(alert_threshold, 0)})",
            f"- CvD-Score: {_format_number(editorial_score)}",
            f"- Auswahlwert: {_format_number(selection_score)}",
            f"- Zeitfenster: {time_fit_label or 'nicht bewertet'}",
            f"- Prognose: {_format_forecast(forecast)}",
            f"- Letzter Push: {_format_minutes(minutes)}",
            f"- Empfehlung um: {_format_dt(now_ts)} Uhr",
            "",
            "Empfehlung:",
            "Jetzt pushen.",
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
        why_now=why_now,
        push_text_matches_title=push_text_matches_title,
    )
    return {
        "text": text,
        "payload": {
            "type": "push_recommendation",
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
            "timeFitScore": time_fit_score,
            "timeFitLabel": time_fit_label,
            "selectionScore": selection_score,
            "predictedOR": round(float(predicted_or), 4) if predicted_or is not None else 0.0,
            "predictedORAvailable": predicted_or is not None,
            "predictedORLabel": _format_or(predicted_or),
            "predictedORSource": forecast["source"],
            "predictedORBasis": forecast["basis"],
            "predictedORConfidence": forecast["confidence"],
            "predictedORExplanation": forecast["explanation"],
            "recommendedPushText": push_text,
            "recommendedAt": _format_dt(now_ts),
            "minutesSinceLastPush": round(float(minutes), 1) if minutes_known else 0.0,
            "lastPushKnown": minutes_known,
            "timeSinceLastPushLabel": _format_minutes(minutes),
            "whyNow": why_now,
            "whyPushworthy": why_pushworthy,
            "competition": competition,
            "messageText": text,
            "messageHtml": message_html,
            "text": text,
        },
        "summary": f"Handlungsempfehlung: Jetzt pushen - {title}",
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
    context = build_teams_alert_context(limited, now_ts=now_ts)
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
    selected_forecast = _candidate_forecast(selected, int(context.get("nowTs") or time.time()))
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
        decision_ts=int(context.get("nowTs") or time.time()),
        alert_cooldown_minutes=config.alert_cooldown_minutes,
        global_cooldown_minutes=config.global_cooldown_minutes,
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


# Compatibility aliases requested in the implementation brief.
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


def _recommendation_selection_score(
    *,
    score: float,
    alert_score: float,
    editorial_score: float,
    predicted_or: float | None,
    dashboard_rank: int,
    breaking: bool,
) -> float:
    """Rank eligible candidates by editorial suitability, not dashboard order."""
    forecast_bonus = min(float(predicted_or or 0.0), 10.0) * 0.4
    rank_bonus = max(0.0, 4.0 - max(0, dashboard_rank - 1) * 0.4) if dashboard_rank > 0 else 0.0
    breaking_bonus = 2.0 if breaking else 0.0
    total = (
        editorial_score * 0.55
        + alert_score * 0.30
        + score * 0.10
        + forecast_bonus
        + rank_bonus
        + breaking_bonus
    )
    return round(_clamp(total, 0.0, 100.0), 1)


def _editorial_cvd_review(
    candidate: dict[str, Any],
    *,
    score: float,
    predicted_or: float | None,
    freshness_hours: float | None,
    minutes_since_last_push: float | None,
    dashboard_rank: int,
    alert_score: float,
    now_ts: int,
    config: TeamsAlertConfig,
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
    if "live-ticker" in title_l or "liveticker" in title_l or "liveblog" in title_l:
        impact_bonus += 3.0
    soft_matches = [term for term in soft_terms if term in title_l]
    if soft_matches and not impact_matches and not breaking:
        impact_bonus -= 8.0

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

    time_fit = _time_fit_review(now_ts=now_ts, section=section, breaking=breaking)

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

    total = _clamp(news_value + urgency + user_need + timing + clarity + load + time_fit["score"], 0.0, 100.0)
    blockers: list[str] = []
    reasons: list[str] = [
        f"CvD-Score {total:.1f}/100",
        f"CvD-Nachrichtenwert {news_value:.1f}/40",
        f"CvD-Zeitfenster {time_fit['score']:.1f}/10: {time_fit['label']}",
    ]

    if dashboard_rank > rank_limit and not breaking:
        blockers.append(f"CvD: nicht in den Top {rank_limit} des Dashboard-Felds")
    if news_value < config.min_editorial_news_value:
        blockers.append(
            f"CvD: Nachrichtenwert zu niedrig ({news_value:.1f} < {config.min_editorial_news_value:.1f})"
        )
    if total < config.min_editorial_score:
        blockers.append(f"CvD: redaktionelle Gesamtfreigabe zu schwach ({total:.1f} < {config.min_editorial_score:.1f})")
    if soft_matches and not breaking and news_value < config.min_editorial_news_value + 6.0:
        blockers.append("CvD: weiches Thema ohne ausreichenden aktuellen Nachrichtenwert")
    if predicted_or is None and not breaking and alert_score < config.no_forecast_min_alert_score:
        blockers.append("CvD: ohne belastbare OR-Prognose nur bei absoluter Top-Lage")
    if not breaking and time_fit["score"] < config.min_time_fit_score:
        blockers.append(
            f"CvD: unguenstiges Zeitfenster ({time_fit['score']:.1f} < {config.min_time_fit_score:.1f})"
        )

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
            "localHour": time_fit["localHour"],
            "weekday": time_fit["weekday"],
            "clarity": round(clarity, 1),
            "load": round(load, 1),
        },
    }


def _time_fit_review(*, now_ts: int, section: str, breaking: bool) -> dict[str, Any]:
    local_dt = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin"))
    hour = local_dt.hour
    weekday = local_dt.weekday()
    is_weekend = weekday >= 5

    if 0 <= hour < 5:
        score, label = (5.0, "Nachtfenster nur fuer Breaking") if breaking else (1.0, "Nachtfenster")
    elif 5 <= hour < 7:
        score, label = (6.0, "fruehes Morgenfenster") if breaking else (3.0, "fruehes Morgenfenster")
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
        score, label = (6.0, "spaetes Breaking-Fenster") if breaking else (3.0, "spaetes Abendfenster")
    else:
        score, label = 4.0, "unbekanntes Zeitfenster"

    section_l = section.lower()
    if is_weekend:
        if section_l in {"wirtschaft", "digital", "service"} and not breaking:
            score -= 1.0
        elif section_l in {"news", "regional", "unterhaltung"}:
            score += 0.5
        label += " am Wochenende"
    else:
        if section_l in {"politik", "wirtschaft"} and 7 <= hour < 18:
            score += 0.5
        if section_l == "unterhaltung" and 18 <= hour < 22:
            score += 1.0
        label += " an einem Werktag"

    return {
        "score": round(_clamp(score, 0.0, 10.0), 1),
        "label": label,
        "localHour": hour,
        "weekday": weekday,
        "isWeekend": is_weekend,
    }


def _teams_alert_score(
    candidate: dict[str, Any],
    *,
    score: float,
    predicted_or: float | None,
    freshness_hours: float | None,
    minutes_since_last_push: float | None,
    recent_push_count_6h: int,
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
            "loadPenalty": round(load_penalty, 1),
        },
        "reasons": reasons,
    }


def _candidate_predicted_or(candidate: dict[str, Any]) -> float | None:
    return _candidate_model_forecast(candidate)


def _candidate_forecast(candidate: dict[str, Any], now_ts: int | None = None) -> dict[str, Any]:
    model_value = _candidate_model_forecast(candidate)
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


def _candidate_model_forecast(candidate: dict[str, Any]) -> float | None:
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
    return normalize_predicted_or(candidate.get("predictedOR", candidate.get("predictedOpenRate")))


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
    old_updated = _safe_int(alert_state.get("last_candidate_updated_at"))
    old_breaking = bool(alert_state.get("last_is_breaking"))
    current_updated = _candidate_updated_ts(candidate)

    if score - old_score >= config.realert_score_delta:
        improvements.append(f"Score deutlich gestiegen (+{score - old_score:.1f})")
    if predicted_or is not None and predicted_or - old_or >= config.realert_or_delta:
        improvements.append(f"Prognose deutlich besser (+{predicted_or - old_or:.2f}pp)")
    if current_updated and current_updated > old_updated:
        improvements.append("Artikel wurde aktualisiert")
    if breaking and not old_breaking:
        improvements.append("Breaking-News-Status neu")

    if not improvements:
        return {"blocker": "Bereits per Teams gemeldet"}
    if not cooldown_ok:
        return {"blocker": "Re-Alert-Cooldown noch nicht erfuellt"}
    return {"positive": "Re-Alert wegen relevanter Veraenderung: " + ", ".join(improvements[:2])}


def _already_pushed_reason(candidate: dict[str, Any], history: list[dict[str, Any]]) -> str:
    url = _normalize_url(_url(candidate))
    title = _title(candidate)
    title_tokens = _tokens(title)
    for item in history:
        item_url = _normalize_url(_url(item))
        if url and item_url and url == item_url:
            return "Bereits gepushter Artikel"
        item_title = _title(item)
        if title and item_title and _normalize_title(title) == _normalize_title(item_title):
            return "Duplicate: sehr aehnlicher Titel wurde bereits gepusht"
        if title_tokens:
            similarity = _token_similarity(title_tokens, _tokens(item_title))
            if similarity >= 0.88:
                return "Duplicate: fast identische Meldung wurde bereits gepusht"
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
        "teams_alert_score=%s editorial_score=%s selection_score=%s last_push=%s "
        "decision=%s reasons=%s evaluated_at=%s",
        decision.get("candidateId"),
        decision.get("articleId"),
        decision.get("articleUrl"),
        float(decision.get("score") or 0.0),
        decision.get("predictedOR"),
        decision.get("teamsAlertScore"),
        decision.get("editorialScore"),
        decision.get("selectionScore"),
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


def _format_number(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


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


def _format_minutes(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"vor {float(value):.0f} Minuten"
    return "kein letzter Push bekannt"


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
    push_text_matches_title: bool = False,
) -> str:
    article_html = (
        f'<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
        if url
        else html.escape(title)
    )
    why_now_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in why_now)
    if push_text_matches_title:
        lead_html = (
            "<p><strong>Artikel und empfohlener Push:</strong><br>"
            f"{article_html}</p>"
        )
    else:
        lead_html = (
            "<p><strong>Empfohlener Push-Text:</strong><br>"
            f"{html.escape(recommended_text)}</p>"
            "<p><strong>Artikel:</strong><br>"
            f"{article_html}</p>"
        )
    return (
        "<h2>🚨 Push-Empfehlung: Jetzt pushen</h2>"
        f"{lead_html}"
        "<p><strong>Warum jetzt?</strong></p>"
        f"<ul>{why_now_html}</ul>"
        "<p>"
        f"<strong>Ressort:</strong> {html.escape(section)}<br>"
        f"<strong>Push-Score:</strong> {html.escape(_format_number(score))} "
        f"(Mindestwert {html.escape(_format_number(score_threshold, 0))})<br>"
        f"<strong>Teams-Alert-Score:</strong> {html.escape(_format_number(alert_score))} "
        f"(Schwelle {html.escape(_format_number(alert_threshold, 0))})<br>"
        f"<strong>CvD-Score:</strong> {html.escape(_format_number(editorial_score))}<br>"
        f"<strong>Prognose:</strong> {html.escape(_format_forecast(forecast))}<br>"
        f"<strong>Letzter Push:</strong> "
        f"{html.escape(_format_minutes(minutes_since_last_push))}<br>"
        f"<strong>Empfehlung um:</strong> {html.escape(_format_dt(now_ts))} Uhr"
        "</p>"
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
