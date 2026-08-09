"""Least-privilege hand-off from Push Balancer to a scheduled Power Automate flow."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import re
import time
from dataclasses import replace
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import config as app_config
from app.article_identity import canonical_article_url_identity
from app.auth import require_power_automate_key
from app.cms.article_context import fetch_public_article_context
from app.cms.url_api import (
    UrlApiNotConfigured,
    UrlApiUnavailable,
    get_canonical_article_url,
)
from app.notifications.teams import (
    MANDATORY_BLOCKER_MISSING_CANONICAL_SCORE,
    TeamsAlertConfig,
    _BINDING_SLOT_DISPATCH_GRACE_SECONDS,
    _MANDATORY_TOP1_CANDIDATE_LIMIT,
    _candidate_updated_ts,
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
from app.power_automate_schedule import (
    POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS,
    power_automate_slot_labels_for_date,
)
from app.routers.feed import build_articles_payload
from app.score_api_client import resolve_cms_id
from app.teams_slot_claims import (
    teams_alert_get_by_ref,
    teams_recommendation_article_identity_block_reasons,
    teams_recommendation_slot_group_owned,
    teams_recommendation_slot_get,
    teams_recommendation_slot_fail_if_owned,
    teams_recommendation_slot_record_receipt,
    teams_recommendation_slot_replay,
    teams_recommendation_slot_try_claim_group,
)

router = APIRouter()

_BERLIN = ZoneInfo("Europe/Berlin")
_HEADLINE_COMMAND_ARTICLE_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{24})(?![0-9a-f])",
    re.IGNORECASE,
)
# Backwards-compatible weekday alias for existing imports.
POWER_AUTOMATE_TEAMS_SLOT_LABELS = POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS
_POWER_AUTOMATE_MIN_DELIVERY_BUDGET_SECONDS = 30
_SCHEDULED_RECOMMENDATION_COUNT = 5
_SCHEDULED_CONTRACT_VERSION = 2
_SCHEDULED_FALLBACK_SCORE_SOURCES = frozenset(
    {"captured_push_balancer", "server_editorial_fallback"}
)
_PUSH_BALANCER_CANDIDATES_URL = "https://editorial.one/push-balancer/bild/kandidaten"
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

    articleId: str = Field(min_length=1, max_length=4096)

    @field_validator("articleId")
    @classmethod
    def validate_article_id(cls, value: str) -> str:
        # Teams message details can wrap the slash command in HTML. Accept the
        # existing articleId field in that transport form, but retain exactly
        # one CMS ID and reject ambiguous payloads.
        matches = {
            match.group(1).casefold()
            for match in _HEADLINE_COMMAND_ARTICLE_ID_RE.finditer(html.unescape(value))
        }
        if len(matches) != 1:
            raise ValueError("articleId must contain exactly one 24-character CMS ID")
        return matches.pop()


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
        url = None
    if url:
        context = fetch_public_article_context(url)
        if context:
            return context

    # The command is an exact lookup, so it must not inherit the candidate
    # view's result cap. Reuse the standalone resolver's full-sitemap fallback.
    try:
        from app.routers.headline import resolve_headline_article

        article = resolve_headline_article(article_id)
    except HTTPException:
        return None
    return {
        "url": article["url"],
        "title": article["title"],
        "text": "",
        "category": article["category"],
    }


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


def _utc_iso_at(slot_ts: int) -> str:
    return (
        dt.datetime.fromtimestamp(int(slot_ts), dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _power_automate_binding_slot(now_ts: int) -> dict[str, Any] | None:
    """Resolve only the fixed Power Automate schedule, independent of legacy tuning."""
    now = int(now_ts)
    berlin_now = dt.datetime.fromtimestamp(now, _BERLIN)
    for label in power_automate_slot_labels_for_date(berlin_now.date()):
        hour, minute = (int(part) for part in label.split(":"))
        slot_dt = berlin_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        slot_ts = int(slot_dt.timestamp())
        # A recommendation is generated only once its actual delivery slot is
        # open.  Do not claim or prepare messages in advance: that couples
        # delivery to a second, unnecessary wait step in Power Automate.
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
    """Return five unique display candidates without weakening the canonical Top-1."""
    recommendations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_cms_ids: set[str] = set()
    seen_url_identities: set[str] = set()
    selected = evaluation.get("selectedCandidate")
    selected_key = candidate_key(selected) if isinstance(selected, dict) else ""
    selected_decision = next(
        (
            item.get("decision")
            for item in evaluation.get("decisions") or []
            if isinstance(item, dict)
            and isinstance(item.get("candidate"), dict)
            and candidate_key(item["candidate"]) == selected_key
            and isinstance(item.get("decision"), dict)
        ),
        None,
    )
    selected_score_source = (
        str(selected_decision.get("scoreSource") or selected.get("scoreSource") or "")
        if isinstance(selected_decision, dict) and isinstance(selected, dict)
        else ""
    )
    if (
        not selected_key
        or not isinstance(selected_decision, dict)
        or not selected_decision.get("shouldNotify")
        or selected_score_source != "internal_score_api"
    ):
        return []

    decision_items = [item for item in evaluation.get("decisions") or [] if isinstance(item, dict)]
    decision_items.sort(
        key=lambda item: (
            isinstance(item.get("candidate"), dict)
            and candidate_key(item["candidate"]) == selected_key
        ),
        reverse=True,
    )
    for item in decision_items:
        candidate = item.get("candidate")
        decision = item.get("decision")
        if not isinstance(candidate, dict) or not isinstance(decision, dict):
            continue
        score_source = str(decision.get("scoreSource") or candidate.get("scoreSource") or "")
        canonical_score = score_source == "internal_score_api"
        raw_blockers = [str(reason) for reason in decision.get("blockingReasons") or []]
        canonical_blockers = [
            reason
            for reason in raw_blockers
            if not reason.startswith("Staerkerer Kandidat vorhanden:")
        ]
        fallback_source = str(candidate.get("scoreSourceBeforeInternalApi") or "")
        try:
            fallback_rank = float(candidate.get("scoreBeforeInternalApi"))
        except (TypeError, ValueError):
            fallback_rank = float("nan")
        fallback_only = bool(
            not canonical_score
            and decision.get("mandatorySlotTop1Candidate") is True
            and decision.get("mandatoryTechnicalBlockerCodes")
            == [MANDATORY_BLOCKER_MISSING_CANONICAL_SCORE]
            and fallback_source in _SCHEDULED_FALLBACK_SCORE_SOURCES
            and math.isfinite(fallback_rank)
            and 0.0 < fallback_rank <= 100.0
        )
        if canonical_score:
            if canonical_blockers:
                continue
        elif not fallback_only:
            continue
        title = _title(candidate).strip()
        url = _url(candidate).strip()
        key = candidate_key(candidate)
        cms_id = str(resolve_cms_id(candidate) or "")
        url_identity = canonical_article_url_identity(url)
        if (
            not title
            or not url
            or not key
            or not url_identity
            or key in seen_keys
            or (cms_id and cms_id in seen_cms_ids)
            or url_identity in seen_url_identities
        ):
            continue
        seen_keys.add(key)
        if cms_id:
            seen_cms_ids.add(cms_id)
        seen_url_identities.add(url_identity)
        push_score = (
            round(float(decision.get("score") or _score(candidate)), 1) if canonical_score else None
        )
        recommendations.append(
            {
                "title": title,
                "url": url,
                "pushScore": push_score,
                "publicationTs": _candidate_updated_ts(candidate),
                "_selected": key == selected_key,
                "_canonical": canonical_score,
                "_rank": float(push_score if push_score is not None else fallback_rank),
            }
        )
    recommendations.sort(
        key=lambda item: (
            bool(item["_selected"]),
            bool(item["_canonical"]),
            float(item["_rank"]),
            int(item["publicationTs"] or 0),
        ),
        reverse=True,
    )
    if (
        not recommendations
        or recommendations[0]["_selected"] is not True
        or recommendations[0]["_canonical"] is not True
    ):
        return []
    return [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in recommendations[:_SCHEDULED_RECOMMENDATION_COUNT]
    ]


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
        push_score = recommendation.get("pushScore")
        score_html = (
            "<strong>Score:</strong> " + f"{float(push_score):.1f}".replace(".", ",") + "/100"
            if isinstance(push_score, (int, float)) and not isinstance(push_score, bool)
            else "<em>Kanonischer Push Score steht noch aus.</em>"
        )
        publication_ts = int(recommendation.get("publicationTs") or 0)
        publication_label = (
            dt.datetime.fromtimestamp(publication_ts, _BERLIN).strftime("%d.%m.%Y, %H:%M Uhr")
            if publication_ts > 0
            else "Publikationsdatum unbekannt"
        )
        parts.append(
            f'<p><strong>Top {rank}:</strong> <a href="{url}">{title}</a> '
            f"({publication_label})<br>"
            f"{score_html}</p>"
        )
    # Teams removes paragraph margins. Two explicit breaks preserve one visible
    # blank line between headline, introduction, and every recommendation.
    return "<br><br>".join(parts)


def _scheduled_claim_articles(
    evaluation: dict[str, Any],
    recommendations: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind the exact five rendered identities without expanding the API payload."""
    if len(recommendations) != _SCHEDULED_RECOMMENDATION_COUNT:
        return []
    decisions_by_key: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for item in evaluation.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate")
        decision = item.get("decision")
        if not isinstance(candidate, dict) or not isinstance(decision, dict):
            continue
        key = candidate_key(candidate)
        if key and key not in decisions_by_key:
            decisions_by_key[key] = (candidate, decision)

    claims: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for rank, recommendation in enumerate(recommendations, start=1):
        article_key = candidate_key(recommendation)
        pair = decisions_by_key.get(article_key)
        if not article_key or article_key in seen_keys or pair is None:
            return []
        seen_keys.add(article_key)
        candidate, decision = pair
        claims.append(
            {
                "article_key": article_key,
                "article_id": str(
                    resolve_cms_id(candidate)
                    or candidate.get("cmsId")
                    or candidate.get("id")
                    or article_key
                ),
                "article_url": _url(candidate),
                "article_title": _title(candidate),
                "title_hash": title_hash(candidate),
                "score": float(recommendation.get("pushScore") or 0.0),
                "predicted_or": float(candidate.get("predictedOR") or 0.0),
                "candidate_updated_at": _candidate_updated_ts(candidate),
                "is_breaking": _is_breaking(candidate),
                "reason": str(
                    decision.get("summary")
                    or f"Teams Top {rank} Anzeigeempfehlung"
                ),
            }
        )
    if not claims or claims[0]["article_key"] != candidate_key(selected):
        return []
    return claims


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
    rendered_url_identities = {
        identity
        for recommendation in recommendations
        if (identity := canonical_article_url_identity(recommendation.get("url")))
    }
    if str(alternative.get("articleTitle") or "").strip() and str(
        alternative.get("articleUrl") or ""
    ).strip():
        proposed_alternative = _article_payload(
            title=str(alternative.get("articleTitle") or "").strip(),
            url=str(alternative.get("articleUrl") or "").strip(),
            category=str(alternative.get("category") or "news").strip() or "news",
            push_score=float(alternative.get("pushScore") or 0.0),
        )
        if (
            proposed_alternative["isSport"] != top["isSport"]
            and canonical_article_url_identity(proposed_alternative["url"])
            in rendered_url_identities
        ):
            alternative_payload = proposed_alternative
    return {
        # Power Automate compares this explicit text marker. This avoids the
        # designer coercing a JSON boolean into a mismatched token type.
        "ready": "yes",
        "contractVersion": _SCHEDULED_CONTRACT_VERSION,
        "slotId": _slot_id(slot_ts),
        "scheduledAt": _iso_at(slot_ts),
        "scheduledAtUtc": _utc_iso_at(slot_ts),
        "expiresAt": _iso_at(slot_ts + _BINDING_SLOT_DISPATCH_GRACE_SECONDS),
        "top": top,
        "alternative": alternative_payload,
        "recommendationCount": len(recommendations),
        "messageHtml": _scheduled_message_html(recommendations),
    }


def _valid_scheduled_replay_payload(
    payload: dict[str, Any],
    *,
    slot_ts: int,
) -> bool:
    message_html = str(payload.get("messageHtml") or "")
    contract_version = payload.get("contractVersion")
    recommendation_count = payload.get("recommendationCount")
    if (
        not isinstance(contract_version, int)
        or isinstance(contract_version, bool)
        or not isinstance(recommendation_count, int)
        or isinstance(recommendation_count, bool)
    ):
        return False
    return bool(
        payload.get("ready") == "yes"
        and contract_version == _SCHEDULED_CONTRACT_VERSION
        and str(payload.get("slotId") or "") == _slot_id(slot_ts)
        and recommendation_count == _SCHEDULED_RECOMMENDATION_COUNT
        and message_html.count("<strong>Top ") == _SCHEDULED_RECOMMENDATION_COUNT
    )


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
    if not _valid_scheduled_replay_payload(owned_payload, slot_ts=slot_ts):
        teams_recommendation_slot_fail_if_owned(
            slot_ts,
            request_id=request_id,
            error="stale_scheduled_claim_contract",
            now_ts=now_ts,
        )
        return _no_op("claim_contract_stale")
    slot_state = teams_recommendation_slot_get(slot_ts)
    request_ref = hashlib.sha256(request_id.strip().encode("utf-8")).hexdigest()
    if (
        not isinstance(slot_state, dict)
        or str(slot_state.get("status") or "") != "sending"
        or str(slot_state.get("request_ref") or "") != request_ref
    ):
        return _no_op("slot_already_claimed")
    group_state = teams_recommendation_slot_group_owned(
        slot_ts,
        request_id=request_id,
        now_ts=now_ts,
    )
    if group_state.get("owned") and int(group_state.get("itemCount") or 0) == 5:
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
        error=str(group_state.get("reason") or "replay_article_claim_unavailable"),
        now_ts=now_ts,
    )
    return _no_op("article_claim_unavailable")


def _scheduled_dispatch_config(config: TeamsAlertConfig) -> TeamsAlertConfig:
    """Return the one production policy used by claims and readiness probes."""
    return replace(
        config,
        require_internal_score_api=True,
        allow_durable_live_history_fallback=(
            not app_config.POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY
        ),
        mandatory_sport_quota_enabled=False,
        slot_delay_date="",
        slot_delay_from="",
        slot_delay_minutes=0,
    )


def _prepare_scheduled_recommendation_field(
    candidates: list[dict[str, Any]],
    *,
    binding_slot: dict[str, Any],
    decision_now: int,
    config: TeamsAlertConfig,
    dedup_history: list[dict[str, Any]],
    dedup_history_authoritative: bool,
    readiness_probe: bool = False,
) -> dict[str, Any]:
    """Run the exact read-only candidate path shared by claim and readiness."""
    dispatch_config = _scheduled_dispatch_config(config)
    identity_inputs: list[dict[str, str]] = []
    for candidate in candidates:
        key = candidate_key(candidate)
        if not key:
            continue
        identity_inputs.append(
            {
                "lookup_key": key,
                "article_key": key,
                "article_id": str(
                    resolve_cms_id(candidate)
                    or candidate.get("cmsId")
                    or candidate.get("id")
                    or ""
                ),
                "article_url": _url(candidate),
            }
        )
    identity_blockers = teams_recommendation_article_identity_block_reasons(
        identity_inputs,
        now_ts=decision_now,
        in_progress_seconds=_BINDING_SLOT_DISPATCH_GRACE_SECONDS,
    )
    identity_eligible = [
        candidate
        for candidate in candidates
        if candidate_key(candidate) not in identity_blockers
    ]
    eligible, memory_review = _memory_eligible_candidates(
        identity_eligible,
        now_ts=decision_now,
        config=dispatch_config,
        bypass_global_cooldown=True,
        allow_related_topic=True,
    )
    eligible = eligible[:_MANDATORY_TOP1_CANDIDATE_LIMIT]
    if not eligible:
        return {
            "ready": False,
            "reason": "no_candidate",
            "recommendations": [],
            "memoryReview": memory_review,
            "identityBlockers": identity_blockers,
        }
    context = build_teams_alert_context(
        eligible,
        history=dedup_history,
        history_authoritative=dedup_history_authoritative,
        now_ts=decision_now,
        config=dispatch_config,
    )
    context["_mandatorySlotOverride"] = dict(binding_slot)
    if readiness_probe:
        context["_scheduledReadinessProbe"] = True
    evaluation = evaluate_teams_alert_candidates(eligible, context, dispatch_config)
    try:
        selected, decision = _selected_decision(evaluation)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_409_CONFLICT:
            return {
                "ready": False,
                "reason": "no_candidate",
                "recommendations": [],
                "context": context,
                "evaluation": evaluation,
                "dispatchConfig": dispatch_config,
                "memoryReview": memory_review,
                "identityBlockers": identity_blockers,
            }
        raise
    recommendations = _scheduled_recommendations(evaluation)
    return {
        "ready": len(recommendations) == _SCHEDULED_RECOMMENDATION_COUNT,
        "reason": (
            "ready"
            if len(recommendations) == _SCHEDULED_RECOMMENDATION_COUNT
            else "insufficient_recommendations"
        ),
        "recommendations": recommendations,
        "selected": selected,
        "decision": decision,
        "context": context,
        "evaluation": evaluation,
        "dispatchConfig": dispatch_config,
        "memoryReview": memory_review,
        "identityBlockers": identity_blockers,
    }


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
    if not app_config.durable_db_storage_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Durable recommendation storage is unavailable.",
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

    actual_decision_now = int(time.time())
    if not _power_automate_slot_open(
        actual_decision_now,
        expected_slot_ts=binding_slot_ts,
    ):
        return _no_op("slot_closed")
    # Delivery starts when the fixed slot opens. Do not prepare or hold a
    # message in Power Automate: that added a second failure mode without
    # improving the recommendation.
    decision_now = actual_decision_now

    dedup_history: list[dict[str, Any]] = []
    dedup_history_authoritative = False
    if app_config.POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY:
        history_refresh = _refresh_push_history_for_dedup()
        refreshed_history = history_refresh.get("history")
        dedup_history = (
            refreshed_history
            if isinstance(refreshed_history, list)
            and all(isinstance(item, dict) for item in refreshed_history)
            else []
        )
        dedup_history_authoritative = bool(
            history_refresh.get("history_authoritative")
            and isinstance(refreshed_history, list)
            and len(dedup_history) == len(refreshed_history)
        )
        if not dedup_history_authoritative:
            return _no_op("live_history_unavailable")

    prepared = _prepare_scheduled_recommendation_field(
        candidates,
        binding_slot=binding_slot,
        decision_now=decision_now,
        config=config,
        dedup_history=dedup_history,
        dedup_history_authoritative=dedup_history_authoritative,
    )
    if not prepared.get("ready"):
        return _no_op(str(prepared.get("reason") or "no_candidate"))
    selected = prepared["selected"]
    decision = prepared["decision"]
    context = prepared["context"]
    evaluation = prepared["evaluation"]
    dispatch_config = prepared["dispatchConfig"]
    recommendations = prepared["recommendations"]

    message = build_teams_push_recommendation(selected, context, decision, dispatch_config)
    message_payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    if (
        not message.get("_dispatchApproved")
        or not message.get("_slotGateApproved")
        or int(message.get("_bindingSlotTs") or 0) != binding_slot_ts
        or str(message_payload.get("slotId") or "") != _slot_id(binding_slot_ts)
        or not str(message_payload.get("messageHtml") or "").strip()
    ):
        return _no_op("candidate_not_approved")

    preclaim_now = int(time.time())
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
        recommendations=recommendations,
    )
    claim_articles = _scheduled_claim_articles(
        evaluation,
        recommendations,
        selected=selected,
    )
    if len(claim_articles) != _SCHEDULED_RECOMMENDATION_COUNT:
        return _no_op("candidate_not_approved")
    slot_claim = teams_recommendation_slot_try_claim_group(
        binding_slot_ts,
        articles=claim_articles,
        request_id=claim_request.requestId,
        claim_payload=response_payload,
        now_ts=preclaim_now,
        lease_seconds=_BINDING_SLOT_DISPATCH_GRACE_SECONDS,
    )
    if slot_claim.get("reason") == "replayed" and isinstance(slot_claim.get("replayPayload"), dict):
        return _safe_replay_response(
            slot_ts=binding_slot_ts,
            request_id=claim_request.requestId,
            now_ts=preclaim_now,
        )
    if not slot_claim.get("claimed"):
        claim_reason = str(slot_claim.get("reason") or "")
        if claim_reason.startswith("slot_"):
            return _no_op("slot_already_claimed")
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
