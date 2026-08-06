"""Least-privilege hand-off from Push Balancer to a scheduled Power Automate flow."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import time
from dataclasses import replace
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth import require_power_automate_key
from app.cms.article_context import fetch_public_article_context
from app.cms.url_api import (
    UrlApiNotConfigured,
    UrlApiUnavailable,
    get_canonical_article_url,
)
from app.config import POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY
from app.database import teams_alert_get, teams_alert_try_claim_send
from app.notifications.teams import (
    TeamsAlertConfig,
    _BINDING_SLOT_DISPATCH_GRACE_SECONDS,
    _MANDATORY_TOP1_CANDIDATE_LIMIT,
    _candidate_updated_ts,
    _dispatch_live_push_comparison,
    _is_breaking,
    _is_sport_item,
    _memory_eligible_candidates,
    _refresh_push_history_for_dedup,
    _score,
    _title,
    _url,
    build_teams_alert_context,
    build_teams_push_recommendation,
    candidate_key,
    evaluate_teams_alert_candidates,
    title_hash,
)
from app.routers.feed import build_articles_payload
from app.teams_slot_claims import (
    teams_alert_get_by_ref,
    teams_recommendation_slot_get,
    teams_recommendation_slot_fail_if_owned,
    teams_recommendation_slot_record_receipt,
    teams_recommendation_slot_replay,
    teams_recommendation_slot_try_claim,
)

router = APIRouter()

_BERLIN = ZoneInfo("Europe/Berlin")
POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS = (
    "06:00",
    "06:36",
    "07:12",
    "07:47",
    "08:23",
    "08:59",
    "12:30",
    "17:30",
    "18:49",
    "20:08",
    "21:26",
    "22:45",
)
POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS = (
    "08:00",
    "08:36",
    "09:12",
    "09:47",
    "10:23",
    "10:59",
    "12:30",
    "17:30",
    "18:49",
    "20:08",
    "21:26",
    "22:45",
)
# Backwards-compatible weekday alias for existing imports.
POWER_AUTOMATE_TEAMS_SLOT_LABELS = POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS
_POWER_AUTOMATE_MIN_DELIVERY_BUDGET_SECONDS = 30
_PUSH_BALANCER_CANDIDATES_URL = (
    "https://editorial.one/push-balancer/bild/kandidaten"
)
_SLOT_ID_RE = re.compile(r"^teams-recommendation-(?P<timestamp>[0-9]{9,11})$")
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "Vary": "X-Power-Automate-Key",
}


def _no_op(reason: str) -> JSONResponse:
    """Return an expected no-send outcome without failing the scheduled flow."""
    return JSONResponse(
        content={"ready": False, "reason": str(reason or "not_ready")},
        headers=_NO_STORE_HEADERS,
    )


class PowerAutomateClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requestId: str = Field(min_length=1, max_length=200)

    @field_validator("requestId")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("requestId must not be blank")
        return normalized


class PowerAutomateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    slotId: str
    requestId: str = Field(min_length=1, max_length=200)
    status: Literal["sent", "failed", "delivery_uncertain"]

    @field_validator("requestId")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("requestId must not be blank")
        return normalized


class PowerAutomateHeadlineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    articleId: str = Field(min_length=24, max_length=24)

    @field_validator("articleId")
    @classmethod
    def validate_article_id(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{24}", normalized):
            raise ValueError("articleId must be a 24-character CMS ID")
        return normalized


def _headline_article_context(article_id: str) -> dict[str, str] | None:
    try:
        from app.score_api_client import resolve_cms_id

        payload = build_articles_payload(
            offset=0,
            limit=200,
            include_teams_decisions=False,
            use_internal_score_api=False,
        )
        for item in payload.get("articles") or []:
            if not isinstance(item, dict) or resolve_cms_id(item) != article_id:
                continue
            return {
                "url": str(item.get("url") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "text": str(item.get("description") or "").strip(),
                "category": str(item.get("category") or "news").strip() or "news",
            }
    except Exception:
        pass

    try:
        url = get_canonical_article_url(article_id)
    except (UrlApiNotConfigured, UrlApiUnavailable):
        return None
    return fetch_public_article_context(url) if url else None


def _headline_candidates(result: dict[str, Any]) -> list[dict[str, str]]:
    winner = result.get("gewinner") if isinstance(result.get("gewinner"), dict) else {}
    winner_title = str(winner.get("titel") or "").strip()
    raw: list[dict[str, Any]] = []
    for group in (result.get("alle_kandidaten") or {}).values():
        if isinstance(group, list):
            raw.extend(item for item in group if isinstance(item, dict))
    raw.sort(key=lambda item: str(item.get("titel") or "") != winner_title)

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        headline = str(item.get("titel") or "").strip()
        line_two = str(item.get("zeile2") or "").strip()
        if not headline or not line_two or headline.casefold() in seen:
            continue
        seen.add(headline.casefold())
        candidates.append(
            {
                "type": str(item.get("ansatz") or "FAKT").strip(),
                "headline": headline,
                "line2": line_two,
            }
        )
        if len(candidates) == 3:
            break
    return candidates


def _headline_message_html(
    *,
    article_id: str,
    article_url: str,
    result: dict[str, Any],
    candidates: list[dict[str, str]],
) -> str:
    level = int(result.get("stufe") or 2)
    level_reason = html.escape(str(result.get("stufe_begruendung") or "").strip())
    parts = [
        "<h2>Headline-Vorschläge</h2>",
        f"<p><strong>Artikel:</strong> <a href=\"{html.escape(article_url, quote=True)}\">{html.escape(article_id)}</a></p>",
        f"<p><strong>Stufe {level}</strong>{' · ' + level_reason if level_reason else ''}</p>",
    ]
    for index, item in enumerate(candidates):
        label = chr(ord("A") + index)
        parts.append(
            f"<p><strong>{label} — {html.escape(item['type'])}</strong><br>"
            f"{html.escape(item['headline'])} ({len(item['headline'])})<br>"
            f"{html.escape(item['line2'])} ({len(item['line2'])})</p>"
        )
    reason = str(result.get("reasoning") or "").strip()
    warning = str(result.get("warnhinweis") or "").strip()
    if reason:
        parts.append(f"<p><strong>Empfehlung:</strong> A · {html.escape(reason)}</p>")
    if warning:
        parts.append(f"<p><strong>Prüfpunkt:</strong> {html.escape(warning)}</p>")
    parts.append("<p><em>Redaktioneller Vorschlag – bitte vor Versand prüfen.</em></p>")
    return "".join(parts)


def _slot_id(slot_ts: int) -> str:
    return f"teams-recommendation-{int(slot_ts)}"


def _slot_ts(slot_id: str) -> int:
    match = _SLOT_ID_RE.fullmatch(str(slot_id or "").strip())
    if not match:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid slotId.",
        )
    return int(match.group("timestamp"))


def _iso_at(slot_ts: int) -> str:
    return dt.datetime.fromtimestamp(int(slot_ts), _BERLIN).isoformat()


def power_automate_slot_labels_for_date(local_date: dt.date) -> tuple[str, ...]:
    """Use a two-hour-later morning block on Saturday and Sunday."""
    return (
        POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS
        if local_date.weekday() >= 5
        else POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS
    )


def _power_automate_binding_slot(now_ts: int) -> dict[str, Any] | None:
    """Resolve only the fixed Power Automate schedule, independent of legacy tuning."""
    now = int(now_ts)
    berlin_now = dt.datetime.fromtimestamp(now, _BERLIN)
    for label in power_automate_slot_labels_for_date(berlin_now.date()):
        hour, minute = (int(part) for part in label.split(":"))
        slot_dt = berlin_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        slot_ts = int(slot_dt.timestamp())
        if slot_ts <= now < slot_ts + _BINDING_SLOT_DISPATCH_GRACE_SECONDS:
            return {
                "ts": slot_ts,
                "label": label,
                "slotRole": "power_automate_fixed",
            }
    return None


def _power_automate_slot_open(now_ts: int, *, expected_slot_ts: int) -> bool:
    slot = _power_automate_binding_slot(now_ts)
    return bool(slot and int(slot.get("ts") or 0) == int(expected_slot_ts))


def _selected_decision(evaluation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = evaluation.get("selectedCandidate")
    if not isinstance(selected, dict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No recommendation is available for this slot.",
        )
    selected_key = candidate_key(selected)
    decision = next(
        (
            item.get("decision")
            for item in evaluation.get("decisions") or []
            if isinstance(item, dict)
            and isinstance(item.get("decision"), dict)
            and str(item["decision"].get("candidateId") or "") == selected_key
        ),
        None,
    )
    if not isinstance(decision, dict) or not decision.get("shouldNotify"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No recommendation is available for this slot.",
        )
    return selected, decision


def _article_payload(
    *,
    title: str,
    url: str,
    category: str,
    push_score: float,
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "category": category,
        "pushScore": round(float(push_score or 0.0), 1),
        "isSport": _is_sport_item({"category": category, "url": url}),
    }


def _scheduled_recommendations(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return up to five technically valid candidates ordered by Push Score."""
    recommendations: list[dict[str, Any]] = []
    for item in evaluation.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate")
        decision = item.get("decision")
        if not isinstance(candidate, dict) or not isinstance(decision, dict):
            continue
        blockers = [
            str(reason)
            for reason in (decision.get("blockingReasons") or [])
            if not str(reason).startswith("Staerkerer Kandidat vorhanden:")
        ]
        if blockers:
            continue
        title = _title(candidate).strip()
        url = _url(candidate).strip()
        if not title or not url:
            continue
        recommendations.append(
            {
                "title": title,
                "url": url,
                "pushScore": round(float(decision.get("score") or _score(candidate)), 1),
            }
        )
    recommendations.sort(key=lambda item: float(item["pushScore"]), reverse=True)
    return recommendations[:5]


def _scheduled_message_html(recommendations: list[dict[str, Any]]) -> str:
    parts = [
        "<h2>🔵 JETZT MÜSSEN (!) WIR PUSHEN</h2>",
        "<p>Das sind meine 5 Empfehlungen. Nichts dabei? Dann schau in den "
        f'<a href="{html.escape(_PUSH_BALANCER_CANDIDATES_URL, quote=True)}">'
        "Push Balancer</a>.</p>",
    ]
    for rank, recommendation in enumerate(recommendations, start=1):
        title = html.escape(str(recommendation["title"]))
        url = html.escape(str(recommendation["url"]), quote=True)
        score = f"{float(recommendation['pushScore']):.1f}".replace(".", ",")
        parts.append(
            f'<p><strong>Top {rank}:</strong> <a href="{url}">{title}</a><br>'
            f"<strong>Score:</strong> {score}/100</p>"
        )
    return "".join(parts)


def _claim_response_payload(
    *,
    slot_ts: int,
    selected: dict[str, Any],
    message: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    alternative = (
        payload.get("alternativeRecommendation")
        if isinstance(payload.get("alternativeRecommendation"), dict)
        else {}
    )
    top = _article_payload(
        title=str(payload.get("articleTitle") or _title(selected)).strip(),
        url=str(payload.get("articleUrl") or _url(selected)).strip(),
        category=str(payload.get("category") or selected.get("category") or "news").strip()
        or "news",
        push_score=float(payload.get("pushScore") or _score(selected)),
    )
    alternative_payload = None
    if str(alternative.get("articleTitle") or "").strip() and str(
        alternative.get("articleUrl") or ""
    ).strip():
        proposed_alternative = _article_payload(
            title=str(alternative.get("articleTitle") or "").strip(),
            url=str(alternative.get("articleUrl") or "").strip(),
            category=str(alternative.get("category") or "news").strip() or "news",
            push_score=float(alternative.get("pushScore") or 0.0),
        )
        if proposed_alternative["isSport"] != top["isSport"]:
            alternative_payload = proposed_alternative
    return {
        "ready": True,
        "slotId": _slot_id(slot_ts),
        "scheduledAt": _iso_at(slot_ts),
        "expiresAt": _iso_at(slot_ts + _BINDING_SLOT_DISPATCH_GRACE_SECONDS),
        "top": top,
        "alternative": alternative_payload,
        "messageHtml": _scheduled_message_html(recommendations),
    }


def _find_alert_by_ref(article_ref: str) -> dict[str, Any] | None:
    return teams_alert_get_by_ref(article_ref)


def _alert_from_slot_state(slot_state: dict[str, Any]) -> dict[str, Any] | None:
    """Recover minimal alert metadata if a process died after the durable slot claim."""
    article_ref = str(slot_state.get("article_ref") or "")
    existing = _find_alert_by_ref(article_ref)
    if existing is not None:
        return existing
    try:
        claim_payload = json.loads(str(slot_state.get("claim_payload_json") or ""))
    except (TypeError, ValueError):
        return None
    top = claim_payload.get("top") if isinstance(claim_payload, dict) else None
    if not isinstance(top, dict):
        return None
    article_url = str(top.get("url") or "").strip()
    article_key = candidate_key({"url": article_url})
    if not article_key or hashlib.sha256(article_key.encode("utf-8")).hexdigest() != article_ref:
        return None
    article_title = str(top.get("title") or "").strip()
    return {
        "article_key": article_key,
        "article_id": article_key,
        "article_url": article_url,
        "article_title": article_title,
        "title_hash": hashlib.sha256(article_title.casefold().encode("utf-8")).hexdigest(),
        "last_score": float(top.get("pushScore") or 0.0),
        "last_predicted_or": 0.0,
        "last_candidate_updated_at": 0,
        "last_is_breaking": 0,
        "last_reason": "Push empfohlen",
        "status": "sending",
    }


def _article_claim_owned_by_slot(article_key: str, *, claimed_at: int) -> bool:
    alert = teams_alert_get(article_key)
    return bool(
        isinstance(alert, dict)
        and str(alert.get("status") or "") == "sending"
        and int(alert.get("last_decision_ts") or 0) == int(claimed_at or 0)
    )


def _ensure_replay_article_claim(
    slot_state: dict[str, Any],
    replay_payload: dict[str, Any],
) -> bool:
    """Repair the narrow crash gap between the durable slot and article claims."""
    top = replay_payload.get("top")
    if not isinstance(top, dict):
        return False
    article_url = str(top.get("url") or "").strip()
    article_key = candidate_key({"url": article_url})
    claimed_at = int(slot_state.get("claimed_at") or 0)
    expected_ref = hashlib.sha256(article_key.encode("utf-8")).hexdigest()
    if (
        not article_key
        or claimed_at <= 0
        or expected_ref != str(slot_state.get("article_ref") or "")
    ):
        return False
    existing = teams_alert_get(article_key)
    if isinstance(existing, dict) and str(existing.get("status") or "") in {
        "sent",
        "delivery_uncertain",
    }:
        return False
    if _article_claim_owned_by_slot(article_key, claimed_at=claimed_at):
        return True

    article_title = str(top.get("title") or "").strip()
    claim = teams_alert_try_claim_send(
        article_key=article_key,
        article_id=article_key,
        article_url=article_url,
        article_title=article_title,
        title_hash=hashlib.sha256(article_title.casefold().encode("utf-8")).hexdigest(),
        score=float(top.get("pushScore") or 0.0),
        predicted_or=0.0,
        candidate_updated_at=0,
        is_breaking=False,
        reason="Push empfohlen",
        decision_ts=claimed_at,
        alert_cooldown_minutes=0,
        global_cooldown_minutes=0,
        in_progress_cooldown_minutes=5,
        failed_cooldown_minutes=0,
        transport_failure_cooldown_minutes=0,
    )
    return bool(
        claim.get("claimed")
        or _article_claim_owned_by_slot(article_key, claimed_at=claimed_at)
    )


def _safe_replay_response(
    *,
    slot_ts: int,
    request_id: str,
    now_ts: int,
) -> JSONResponse:
    """Replay only a still-owned sending claim with a matching article claim."""
    owned_payload = teams_recommendation_slot_replay(
        slot_ts,
        request_id=request_id,
        now_ts=now_ts,
    )
    if owned_payload is None:
        return _no_op("slot_already_claimed")
    slot_state = teams_recommendation_slot_get(slot_ts)
    request_ref = hashlib.sha256(request_id.strip().encode("utf-8")).hexdigest()
    if (
        not isinstance(slot_state, dict)
        or str(slot_state.get("status") or "") != "sending"
        or str(slot_state.get("request_ref") or "") != request_ref
    ):
        return _no_op("slot_already_claimed")
    if _ensure_replay_article_claim(slot_state, owned_payload):
        confirmed_payload = teams_recommendation_slot_replay(
            slot_ts,
            request_id=request_id,
            now_ts=now_ts,
        )
        if confirmed_payload is not None:
            return JSONResponse(content=confirmed_payload, headers=_NO_STORE_HEADERS)
        return _no_op("slot_already_claimed")
    teams_recommendation_slot_fail_if_owned(
        slot_ts,
        request_id=request_id,
        error="replay_article_claim_unavailable",
        now_ts=now_ts,
    )
    return _no_op("article_claim_unavailable")


@router.post(
    "/api/v1/power-automate/teams/claim",
    dependencies=[Depends(require_power_automate_key)],
    include_in_schema=False,
)
def claim_power_automate_teams_recommendation(
    claim_request: PowerAutomateClaimRequest,
) -> JSONResponse:
    """Reserve and return exactly one current mandatory-slot recommendation."""
    config = TeamsAlertConfig()
    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Teams recommendation claims are disabled.",
        )

    started_at = int(time.time())
    binding_slot = _power_automate_binding_slot(started_at)
    if binding_slot is None:
        return _no_op("outside_window")
    binding_slot_ts = int(binding_slot.get("ts") or 0)
    replay_payload = teams_recommendation_slot_replay(
        binding_slot_ts,
        request_id=claim_request.requestId,
        now_ts=started_at,
    )
    if replay_payload is not None:
        return _safe_replay_response(
            slot_ts=binding_slot_ts,
            request_id=claim_request.requestId,
            now_ts=started_at,
        )

    existing_slot = teams_recommendation_slot_get(binding_slot_ts)
    if isinstance(existing_slot, dict) and str(existing_slot.get("status") or "") in {
        "sending",
        "sent",
        "delivery_uncertain",
    }:
        return _no_op("slot_already_claimed")
    dispatch_config = replace(
        config,
        require_internal_score_api=True,
        mandatory_sport_quota_enabled=False,
        slot_delay_date="",
        slot_delay_from="",
        slot_delay_minutes=0,
    )

    article_payload = build_articles_payload(
        offset=0,
        limit=_MANDATORY_TOP1_CANDIDATE_LIMIT,
        include_teams_decisions=False,
        use_internal_score_api=True,
    )
    candidates = [
        item for item in (article_payload.get("articles") or []) if isinstance(item, dict)
    ]
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The current recommendation field is unavailable.",
        )

    refresh = _refresh_push_history_for_dedup()
    history = refresh.get("history")
    history_authoritative = bool(
        refresh.get("history_authoritative") and isinstance(history, list)
    )
    if not isinstance(history, list):
        history = []
    if POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY and not history_authoritative:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Live-push duplicate protection is temporarily unavailable.",
        )

    decision_now = int(time.time())
    if not _power_automate_slot_open(
        decision_now,
        expected_slot_ts=binding_slot_ts,
    ):
        return _no_op("slot_closed")

    candidates, _ = _memory_eligible_candidates(
        candidates,
        now_ts=decision_now,
        config=dispatch_config,
        bypass_global_cooldown=True,
        allow_related_topic=True,
    )
    candidates = candidates[:_MANDATORY_TOP1_CANDIDATE_LIMIT]
    context = build_teams_alert_context(
        candidates,
        history=history,
        history_authoritative=(
            history_authoritative or not POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY
        ),
        now_ts=decision_now,
        config=dispatch_config,
    )
    evaluation = evaluate_teams_alert_candidates(candidates, context, dispatch_config)
    try:
        selected, decision = _selected_decision(evaluation)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_409_CONFLICT:
            return _no_op("no_candidate")
        raise
    message = build_teams_push_recommendation(selected, context, decision, dispatch_config)
    message_payload = (
        message.get("payload") if isinstance(message.get("payload"), dict) else {}
    )
    if (
        not message.get("_dispatchApproved")
        or not message.get("_slotGateApproved")
        or not str(message_payload.get("messageHtml") or "").strip()
    ):
        return _no_op("candidate_not_approved")

    preclaim_now = int(time.time())
    final_dedup = (
        _dispatch_live_push_comparison(
            selected,
            now_ts=preclaim_now,
            config=dispatch_config,
            comparison_authoritative=True,
            history=history,
            refresh_live_history=False,
        )
        if history_authoritative
        else {"blocked": False, "mode": "durable_claim_fallback"}
    )
    if final_dedup.get("blocked"):
        duplicate = str(final_dedup.get("code") or "") == "live_push_exact_article_duplicate"
        if duplicate:
            return _no_op("already_live_pushed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Live-push duplicate protection is temporarily unavailable.",
        )
    if not _power_automate_slot_open(
        preclaim_now,
        expected_slot_ts=binding_slot_ts,
    ):
        return _no_op("slot_closed")
    if (
        preclaim_now + _POWER_AUTOMATE_MIN_DELIVERY_BUDGET_SECONDS
        > binding_slot_ts + _BINDING_SLOT_DISPATCH_GRACE_SECONDS
    ):
        return _no_op("slot_closed")

    response_payload = _claim_response_payload(
        slot_ts=binding_slot_ts,
        selected=selected,
        message=message,
        recommendations=_scheduled_recommendations(evaluation),
    )
    article_key = candidate_key(selected)
    slot_claim = teams_recommendation_slot_try_claim(
        binding_slot_ts,
        article_key=article_key,
        request_id=claim_request.requestId,
        claim_payload=response_payload,
        now_ts=preclaim_now,
        lease_seconds=_BINDING_SLOT_DISPATCH_GRACE_SECONDS,
    )
    if slot_claim.get("reason") == "replayed" and isinstance(
        slot_claim.get("replayPayload"), dict
    ):
        return _safe_replay_response(
            slot_ts=binding_slot_ts,
            request_id=claim_request.requestId,
            now_ts=preclaim_now,
        )
    if not slot_claim.get("claimed"):
        return _no_op("slot_already_claimed")

    article_claim = teams_alert_try_claim_send(
        article_key=article_key,
        article_id=str(selected.get("id") or selected.get("cmsId") or article_key),
        article_url=_url(selected),
        article_title=_title(selected),
        title_hash=title_hash(selected),
        score=_score(selected),
        predicted_or=float(selected.get("predictedOR") or 0.0),
        candidate_updated_at=_candidate_updated_ts(selected),
        is_breaking=_is_breaking(selected),
        reason=str(decision.get("summary") or "Push empfohlen"),
        decision_ts=preclaim_now,
        alert_cooldown_minutes=0,
        global_cooldown_minutes=0,
        in_progress_cooldown_minutes=5,
        failed_cooldown_minutes=0,
        transport_failure_cooldown_minutes=0,
    )
    if not article_claim.get("claimed"):
        slot_state = teams_recommendation_slot_get(binding_slot_ts) or {}
        if not _article_claim_owned_by_slot(
            article_key,
            claimed_at=int(slot_state.get("claimed_at") or 0),
        ):
            teams_recommendation_slot_fail_if_owned(
                binding_slot_ts,
                request_id=claim_request.requestId,
                error=str(article_claim.get("reason") or "article_claim_blocked"),
                now_ts=preclaim_now,
            )
            return _no_op("article_already_claimed")

    return JSONResponse(
        content=response_payload,
        headers=_NO_STORE_HEADERS,
    )


@router.post(
    "/api/v1/power-automate/teams/headline",
    dependencies=[Depends(require_power_automate_key)],
    include_in_schema=False,
)
def generate_power_automate_teams_headlines(
    request: PowerAutomateHeadlineRequest,
) -> JSONResponse:
    """Generate three v1.4 headline pairs for a Teams slash command."""
    context = _headline_article_context(request.articleId)
    if not context or not context.get("title") or not context.get("url"):
        return JSONResponse(
            content={
                "ready": False,
                "reason": "article_not_found",
                "messageHtml": (
                    "<p><strong>Artikel nicht gefunden.</strong><br>"
                    "Bitte die 24-stellige Artikel-ID prüfen.</p>"
                ),
            },
            headers=_NO_STORE_HEADERS,
        )

    from app.routers.misc import PushTitleGenerateRequest, _build_push_title_response

    result = _build_push_title_response(
        PushTitleGenerateRequest(
            url=context["url"],
            title=context["title"],
            text=context.get("text", ""),
            headline=context["title"],
            category=context.get("category", "news"),
            force_llm=True,
        )
    )
    candidates = _headline_candidates(result)
    if len(candidates) < 3:
        return JSONResponse(
            content={
                "ready": False,
                "reason": "headline_generator_unavailable",
                "messageHtml": (
                    "<p><strong>Headline-Generator gerade nicht verfügbar.</strong><br>"
                    "Bitte den Befehl später erneut senden.</p>"
                ),
            },
            headers=_NO_STORE_HEADERS,
        )

    response = {
        "ready": True,
        "articleId": request.articleId,
        "articleUrl": context["url"],
        "suggestions": candidates,
        "messageHtml": _headline_message_html(
            article_id=request.articleId,
            article_url=context["url"],
            result=result,
            candidates=candidates,
        ),
    }
    return JSONResponse(content=response, headers=_NO_STORE_HEADERS)


@router.post(
    "/api/v1/power-automate/teams/receipt",
    dependencies=[Depends(require_power_automate_key)],
    include_in_schema=False,
)
def record_power_automate_teams_receipt(receipt: PowerAutomateReceipt) -> JSONResponse:
    """Finalize the claimed slot after the Teams action succeeded or failed."""
    binding_slot_ts = _slot_ts(receipt.slotId)
    slot_state = teams_recommendation_slot_get(binding_slot_ts)
    if slot_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown slotId.")

    alert = _alert_from_slot_state(slot_state)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The claimed recommendation metadata is unavailable.",
        )

    recorded_at = int(time.time())
    slot_receipt = teams_recommendation_slot_record_receipt(
        binding_slot_ts,
        status=receipt.status,
        request_id=receipt.requestId,
        article_key=str(alert.get("article_key") or ""),
        now_ts=recorded_at,
    )
    if not slot_receipt.get("recorded"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The receipt conflicts with the final slot state.",
        )

    return JSONResponse(
        content={
            "slotId": receipt.slotId,
            "status": receipt.status,
            "recordedAt": _iso_at(recorded_at),
        },
        headers=_NO_STORE_HEADERS,
    )
