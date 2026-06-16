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

from app.config import (
    PUSH_TEAMS_ALERT_COOLDOWN_MINUTES,
    PUSH_TEAMS_ALERTS_ENABLED,
    PUSH_TEAMS_ALLOWED_SECTIONS,
    PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_BREAKING_MIN_OR,
    PUSH_TEAMS_BREAKING_MIN_SCORE,
    PUSH_TEAMS_BREAKING_OVERRIDE,
    PUSH_TEAMS_CANDIDATE_LIMIT,
    PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS,
    PUSH_TEAMS_MAX_PUSHES_LAST_6H,
    PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH,
    PUSH_TEAMS_MIN_OR,
    PUSH_TEAMS_MIN_SCORE,
    PUSH_TEAMS_REALERT_OR_DELTA,
    PUSH_TEAMS_REALERT_SCORE_DELTA,
    PUSH_TEAMS_SCORE_ONLY_MODE,
    PUSH_TEAMS_WEBHOOK_URL,
)
from app.database import (
    push_db_load_all,
    teams_alert_load_for_keys,
    teams_alert_record,
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
    score_only_mode: bool = PUSH_TEAMS_SCORE_ONLY_MODE
    min_or: float = PUSH_TEAMS_MIN_OR
    min_minutes_since_last_push: int = PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH
    realert_score_delta: float = PUSH_TEAMS_REALERT_SCORE_DELTA
    realert_or_delta: float = PUSH_TEAMS_REALERT_OR_DELTA
    alert_cooldown_minutes: int = PUSH_TEAMS_ALERT_COOLDOWN_MINUTES
    allowed_sections: tuple[str, ...] = tuple(PUSH_TEAMS_ALLOWED_SECTIONS)
    breaking_override: bool = PUSH_TEAMS_BREAKING_OVERRIDE
    breaking_min_score: float = PUSH_TEAMS_BREAKING_MIN_SCORE
    breaking_min_or: float = PUSH_TEAMS_BREAKING_MIN_OR
    breaking_min_minutes_since_last_push: int = PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH
    max_article_age_hours: int = PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS
    max_pushes_last_6h: int = PUSH_TEAMS_MAX_PUSHES_LAST_6H


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("id")
        or candidate.get("articleId")
        or candidate.get("url")
        or candidate.get("link")
        or _title(candidate)
    ).strip()


def title_hash(candidate: dict[str, Any]) -> str:
    value = _title(candidate).strip().lower()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def build_teams_alert_context(
    candidates: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
    alert_state: dict[str, dict[str, Any]] | None = None,
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
    predicted_or = normalize_predicted_or(candidate.get("predictedOR", candidate.get("predictedOpenRate")))
    breaking = _is_breaking(candidate)
    min_score, min_or, min_pause = _effective_thresholds(config, breaking)
    last_push_ts = _safe_int(context.get("lastPushTs"))
    minutes_since_last_push = (
        round((now_ts - last_push_ts) / 60, 1) if last_push_ts > 0 and now_ts >= last_push_ts else None
    )
    alert_state = (context.get("alertState") or {}).get(key)

    positive: list[str] = []
    blockers: list[str] = []
    status = "skip"

    if not config.enabled:
        blockers.append("Teams Alerts deaktiviert")

    if not title:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Headline")
    if not url:
        blockers.append("Keine Teams-Handlungsempfehlung ohne Artikel-Link")

    allowed = {item.lower() for item in config.allowed_sections if item.strip()}
    if not config.score_only_mode and allowed and section.lower() not in allowed:
        blockers.append(f"Ressort {section} nicht fuer Teams Alerts freigegeben")

    if score >= min_score:
        positive.append(f"Push Score {score:.1f} liegt ueber Schwelle {min_score:.1f}")
    else:
        blockers.append(f"Score zu niedrig: {score:.1f} < {min_score:.1f}")

    if config.score_only_mode:
        positive.append("Score-Modus aktiv: Score ueber Schwelle reicht fuer Teams-Hinweis")
        if predicted_or is not None:
            positive.append(f"Prognose als Kontext: {predicted_or:.2f}% OR")
        if minutes_since_last_push is not None:
            positive.append(f"Letzter Push vor {minutes_since_last_push:.0f} Minuten")
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

    freshness_hours = _freshness_hours(candidate, now_ts)
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
        "predictedOR": predicted_or,
        "minScore": min_score,
        "minOR": min_or,
        "minMinutesSinceLastPush": min_pause,
        "minutesSinceLastPush": minutes_since_last_push,
        "lastPushAt": _iso_from_ts(last_push_ts) if last_push_ts else None,
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

    base_decisions = [
        (candidate, should_notify_teams(candidate, base_context, config))
        for candidate in candidates
    ]
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
            key=lambda item: _candidate_rank(item[0]),
        )
        selected_key = candidate_key(selected_candidate)

    final: list[dict[str, Any]] = []
    for candidate, base_decision in base_decisions:
        key = candidate_key(candidate)
        if selected_key and key != selected_key and base_decision.get("shouldNotify"):
            decision_context = dict(base_context)
            decision_context["strongerCandidate"] = selected_candidate
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
                        "Beste Kombination aus Score, Prognose und Dringlichkeit im Kandidatenfeld",
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
    score = _score(candidate)
    predicted_or = normalize_predicted_or(candidate.get("predictedOR", candidate.get("predictedOpenRate")))
    now_ts = int(context.get("nowTs") or time.time())
    minutes = decision.get("minutesSinceLastPush")
    score_only_mode = bool(decision.get("scoreOnlyMode") or config.score_only_mode)
    score_threshold = float(decision.get("minScore", config.min_score) or config.min_score)
    push_text = str(candidate.get("recommendedText") or title)
    competition = (decision.get("competition") or {}).get("summary") or "Kein staerkerer Kandidat aktuell verfuegbar"

    bullets = _dedupe(
        [
            f"Push Score: {score:.1f} und damit ueber Schwelle {decision.get('minScore', config.min_score):.1f}",
            (
                f"Prognose: {_format_or(predicted_or)}, aktuell guter Zeitpunkt"
                if not score_only_mode
                else f"Score-Modus aktiv: Teams-Hinweis wird ab Score {score_threshold:.1f} ausgeloest"
            ),
            (
                f"Letzter Push vor {float(minutes):.0f} Minuten, Mindestpause erfuellt"
                if isinstance(minutes, (int, float))
                else (
                    "Pause seit letztem Push blockiert im Score-Modus nicht"
                    if score_only_mode
                    else "Keine aktuelle Push-Pause blockiert die Empfehlung"
                )
            ),
            *list(decision.get("reasons") or [])[:4],
        ]
    )[:6]

    if score_only_mode:
        why_now = [
            f"Der Push Score liegt bei {score:.1f} und damit ueber der Teams-Schwelle.",
            f"Der aktuelle Teams-Modus ist score-basiert: ab {score_threshold:.1f} wird ein Hinweis ausgeloest.",
            "Anti-Spam-Regeln verhindern Wiederholungen fuer denselben Artikel.",
        ]
        if predicted_or is not None:
            why_now.append(f"Prognose als Kontext: {_format_or(predicted_or)}.")
    else:
        why_now = [
            f"Prognose liegt bei {_format_or(predicted_or)} und damit ueber dem Mindestwert.",
            (
                f"Seit dem letzten Push sind {float(minutes):.0f} Minuten vergangen; die Mindestpause ist erfuellt."
                if isinstance(minutes, (int, float))
                else "In der geladenen Historie blockiert kein letzter Push die Empfehlung."
            ),
            "Der Kandidat ist aktuell der staerkste pushwuerdige Artikel im Feld.",
        ]
    why_pushworthy = [*bullets, f"Konkurrenzlage: {competition}"]

    text_lines = [
        "🚨 Handlungsempfehlung: Jetzt pushen",
        "",
        "Was soll ich pushen?",
        push_text,
        "",
        "Welcher Artikel ist gemeint?",
        f"{title}\n{url}" if url else title,
        "",
        f"Ressort / Kategorie: {section}",
        f"Push Score: {score:.1f}",
        f"Prognose: {_format_or(predicted_or)}",
        f"Zeitpunkt der Empfehlung: {_format_dt(now_ts)}",
        f"Zeit seit letztem Push: {_format_minutes(minutes)}",
        "",
        "Warum genau jetzt?",
        *[f"- {reason}" for reason in why_now],
        "",
        "Warum ist diese Nachricht pushwürdig?",
        *[f"- {reason}" for reason in why_pushworthy],
        "",
        "Empfehlung:",
        "Jetzt pushen.",
    ]
    text = "\n".join(text_lines)
    message_html = _build_power_automate_message_html(
        title=title,
        url=url,
        section=section,
        score=score,
        predicted_or=predicted_or,
        recommended_text=push_text,
        now_ts=now_ts,
        minutes_since_last_push=minutes,
        why_now=why_now,
        why_pushworthy=why_pushworthy,
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
            "predictedOR": predicted_or,
            "predictedORLabel": _format_or(predicted_or),
            "recommendedPushText": push_text,
            "recommendedAt": _format_dt(now_ts),
            "minutesSinceLastPush": minutes,
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
    limited = candidates[: max(1, PUSH_TEAMS_CANDIDATE_LIMIT)]
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

    message = build_teams_push_recommendation(selected, context, selected_decision, config)
    send_result = send_teams_notification(message, config)
    status = "sent" if send_result.get("ok") else "failed"
    reason = selected_decision.get("summary") or "Push empfohlen"
    teams_alert_record(
        article_key=candidate_key(selected),
        article_id=str(selected.get("id") or candidate_key(selected)),
        article_url=_url(selected),
        title_hash=title_hash(selected),
        score=_score(selected),
        predicted_or=normalize_predicted_or(selected.get("predictedOR")) or 0.0,
        candidate_updated_at=_candidate_updated_ts(selected),
        is_breaking=_is_breaking(selected),
        reason=reason,
        status=status,
        error=str(send_result.get("error") or ""),
        decision_ts=int(context.get("nowTs") or time.time()),
    )

    log.info(
        "[TeamsAlert] send_result candidateId=%s articleId=%s url=%s status=%s ok=%s",
        candidate_key(selected),
        selected.get("id") or "",
        _url(selected),
        status,
        bool(send_result.get("ok")),
    )
    return {
        "ok": True,
        "sent": bool(send_result.get("ok")),
        "sendResult": send_result,
        "candidateId": candidate_key(selected),
        "evaluation": evaluation,
    }


def run_teams_alert_cycle() -> dict[str, Any]:
    """Fetch current article candidates and run one Teams alert cycle."""
    try:
        _refresh_push_history_for_timing()
        from app.routers.feed import build_articles_payload

        payload = build_articles_payload(offset=0, limit=PUSH_TEAMS_CANDIDATE_LIMIT)
        candidates = payload.get("articles") or []
        return evaluate_and_send_best_candidate(candidates)
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
    return numeric * 100.0 if numeric <= 1.0 else numeric


def _effective_thresholds(config: TeamsAlertConfig, breaking: bool) -> tuple[float, float, int]:
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
    if not alert_state or alert_state.get("status") != "sent":
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
    predicted = normalize_predicted_or(candidate.get("predictedOR")) or 0.0
    breaking = 1.0 if _is_breaking(candidate) else 0.0
    freshness = _candidate_updated_ts(candidate) or 0
    return (score, predicted, breaking, float(freshness))


def _log_decision(candidate: dict[str, Any], decision: dict[str, Any]) -> None:
    blocking_reasons = list(decision.get("blockingReasons") or [])
    positive_reasons = list(decision.get("reasons") or [])
    reason_text = "; ".join([*blocking_reasons, *positive_reasons])
    log.info(
        "[TeamsAlert] decision candidateId=%s articleId=%s url=%s score=%.1f predicted_or=%s "
        "last_push=%s decision=%s reasons=%s evaluated_at=%s",
        decision.get("candidateId"),
        decision.get("articleId"),
        decision.get("articleUrl"),
        float(decision.get("score") or 0.0),
        decision.get("predictedOR"),
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


def _iso_from_ts(ts_value: int) -> str | None:
    if not ts_value:
        return None
    return dt.datetime.fromtimestamp(int(ts_value)).isoformat()


def _format_dt(ts_value: int) -> str:
    return dt.datetime.fromtimestamp(ts_value).strftime("%d.%m.%Y %H:%M")


def _format_or(value: float | None) -> str:
    return f"{value:.2f} % OR" if value is not None else "keine Prognose"


def _format_minutes(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0f} Minuten"
    return "kein letzter Push bekannt"


def _build_power_automate_message_html(
    *,
    title: str,
    url: str,
    section: str,
    score: float,
    predicted_or: float | None,
    recommended_text: str,
    now_ts: int,
    minutes_since_last_push: Any,
    why_now: list[str],
    why_pushworthy: list[str],
) -> str:
    article_html = (
        f'<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
        if url
        else html.escape(title)
    )
    why_now_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in why_now)
    why_pushworthy_html = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in why_pushworthy
    )
    return (
        "<h2>🚨 Handlungsempfehlung: Jetzt pushen</h2>"
        "<p><strong>Was soll ich pushen?</strong><br>"
        f"{html.escape(recommended_text)}</p>"
        "<p><strong>Welcher Artikel ist gemeint?</strong><br>"
        f"{article_html}</p>"
        "<p>"
        f"<strong>Ressort / Kategorie:</strong> {html.escape(section)}<br>"
        f"<strong>Push Score:</strong> {score:.1f}<br>"
        f"<strong>Prognose:</strong> {html.escape(_format_or(predicted_or))}<br>"
        f"<strong>Zeitpunkt der Empfehlung:</strong> {html.escape(_format_dt(now_ts))}<br>"
        f"<strong>Zeit seit letztem Push:</strong> "
        f"{html.escape(_format_minutes(minutes_since_last_push))}"
        "</p>"
        "<p><strong>Warum genau jetzt?</strong></p>"
        f"<ul>{why_now_html}</ul>"
        "<p><strong>Warum ist diese Nachricht pushwuerdig?</strong></p>"
        f"<ul>{why_pushworthy_html}</ul>"
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
