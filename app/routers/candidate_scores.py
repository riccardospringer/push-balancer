"""Candidate-only adapter for canonical Editorial One scores.

This route is intentionally separate from ``/api/articles`` so activating the
Render candidate UI cannot change Power Automate or Teams ranking/transport.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import (
    EDITORIAL_ONE_SCORE_API_BASE_URL,
    EDITORIAL_ONE_SCORE_API_ENABLED,
    EDITORIAL_ONE_SCORE_API_KEY,
    EDITORIAL_ONE_SCORE_API_MAX_AGE_SECONDS,
    PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS,
    PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS,
    PUSH_BALANCER_SCORE_API_MAX_RETRIES,
    PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS,
)
from app.routers.feed import _apply_internal_score_api_scores, build_articles_payload


log = logging.getLogger("push-balancer")
router = APIRouter()

_score_client = None
_score_client_signature: tuple[Any, ...] | None = None


def _get_score_client():
    """Build one isolated client without exposing its credential to the browser."""
    from app.score_api_client import ScoreApiClient

    global _score_client, _score_client_signature
    signature = (
        EDITORIAL_ONE_SCORE_API_BASE_URL,
        bool(EDITORIAL_ONE_SCORE_API_KEY),
        PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS,
        PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS,
        PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS,
        PUSH_BALANCER_SCORE_API_MAX_RETRIES,
    )
    if _score_client is None or _score_client_signature != signature:
        _score_client = ScoreApiClient(
            EDITORIAL_ONE_SCORE_API_BASE_URL,
            EDITORIAL_ONE_SCORE_API_KEY,
            timeout_seconds=PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS,
            batch_timeout_seconds=PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS,
            cache_ttl_seconds=PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS,
            max_retries=PUSH_BALANCER_SCORE_API_MAX_RETRIES,
        )
        _score_client_signature = signature
    return _score_client


def _score_sync_metadata(articles: list[dict], *, required: bool) -> dict:
    synced = sum(
        1
        for article in articles
        if article.get("scoreSource") == "internal_score_api"
        or (
            article.get("scoreSource") == "consumer_recommendations_api"
            and article.get("scoreApiStatus") == "ok"
        )
    )
    return {
        "required": required,
        "source": "editorial_one_score_api" if required else "render",
        "status": "ok" if not required or synced == len(articles) else "partial",
        "syncedCount": synced if required else 0,
        "totalCount": len(articles),
    }


def _candidate_from_consumer_article(
    article: dict[str, Any],
    *,
    snapshot_at: str,
) -> dict[str, Any]:
    """Map the shared versioned-consumer projection to the candidate contract."""
    flags = article.get("flags") or {}
    explanation = article.get("explanation") or {}
    score = article.get("score")
    return {
        "id": article.get("id") or article.get("url") or "",
        "url": article.get("url") or article.get("id") or "",
        "title": article.get("title") or "",
        "category": article.get("category") or "news",
        "pubDate": article.get("publishedAt") or "",
        "score": float(score) if score is not None else 0.0,
        "scoreReason": explanation.get("reason")
        or "Kanonischer Score aus der Recommendations-API",
        "scoreSource": "consumer_recommendations_api",
        "scoreApiStatus": "ok" if score is not None else "missing",
        "pushBalancerScore": float(score) if score is not None else None,
        "pushBalancerScoreScoredAt": snapshot_at,
        "predictedOR": article.get("predictedOpenRate"),
        "predictedORBasis": article.get("predictedOpenRateBasis"),
        "predictedORConfidence": article.get("predictedOpenRateConfidence"),
        "predictedORIsFallback": bool(article.get("predictedOpenRateIsFallback")),
        "performanceDrivers": list(explanation.get("drivers") or []),
        "risks": list(explanation.get("risks") or []),
        "recommendedText": article.get("recommendedText") or article.get("title") or "",
        "mixPriority": article.get("priority") or "",
        "scoreBreakdown": explanation.get("breakdown") or {},
        "isBreaking": bool(flags.get("breaking")),
        "isEilmeldung": bool(flags.get("eilmeldung")),
        "isSport": bool(flags.get("sport")),
        "isVideo": bool(flags.get("video")),
        "isPlusArticle": bool(flags.get("plusArticle")),
        "isLivePush": False,
        "alreadySent": False,
    }


def _get_shared_consumer_candidates(offset: int, limit: int) -> dict[str, Any]:
    """Use the same immutable score snapshot as ``/api/v1/recommendations``."""
    from app.notifications.teams import annotate_candidates_with_teams_decisions
    from app.routers.consumer import _load_consumer_articles

    consumer_payload = _load_consumer_articles(
        offset=offset,
        limit=limit,
        category=None,
        min_score=None,
        include_explanations=True,
    )
    snapshot_at = str(consumer_payload.get("fetchedAt") or "")
    articles = [
        _candidate_from_consumer_article(article, snapshot_at=snapshot_at)
        for article in consumer_payload.get("articles") or []
    ]
    try:
        articles = annotate_candidates_with_teams_decisions(articles)
    except Exception as exc:
        log.warning(
            "[candidate-scores] Teams annotation unavailable: %s",
            type(exc).__name__,
        )

    return {
        "articles": articles,
        "total": int(consumer_payload.get("total") or len(articles)),
        "count": len(articles),
        "offset": offset,
        "limit": limit,
        "fetchedAt": snapshot_at,
        "livePushes": consumer_payload.get("livePushes") or [],
        "livePushCount": int(consumer_payload.get("livePushCount") or 0),
        "scoreSync": {
            **_score_sync_metadata(articles, required=True),
            "source": "consumer_recommendations_api",
            "snapshotAt": snapshot_at,
        },
    }


@router.get("/api/editorial-one/articles")
def get_editorial_one_articles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=60, ge=1, le=200),
) -> JSONResponse:
    """Return candidates with a fail-closed Editorial One score projection."""
    if not EDITORIAL_ONE_SCORE_API_ENABLED:
        return JSONResponse(content=_get_shared_consumer_candidates(offset, limit))

    source_limit = min(200, max(120, offset + limit))
    payload = build_articles_payload(
        offset=0,
        limit=source_limit,
        include_teams_decisions=False,
    )
    try:
        scored = _apply_internal_score_api_scores(
            payload.get("articles") or [],
            client=_get_score_client(),
            max_age_seconds=EDITORIAL_ONE_SCORE_API_MAX_AGE_SECONDS,
        )
    except Exception as exc:
        from app.score_api_client import ScoreApiError

        if isinstance(exc, ScoreApiError):
            log.warning(
                "[candidate-scores] Editorial One sync unavailable: %s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="Editorial One score synchronization is unavailable.",
            ) from exc
        raise

    selected = scored[offset : offset + limit]
    try:
        from app.notifications.teams import annotate_candidates_with_teams_decisions

        selected = annotate_candidates_with_teams_decisions(selected)
    except Exception as exc:
        log.warning("[candidate-scores] Teams annotation unavailable: %s", type(exc).__name__)

    payload.update(
        {
            "articles": selected,
            "total": len(scored),
            "count": len(scored),
            "offset": offset,
            "limit": limit,
            "scoreSync": _score_sync_metadata(selected, required=True),
        }
    )
    return JSONResponse(content=payload)
