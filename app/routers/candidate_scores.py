"""Candidate-only adapter for the versioned Recommendations API.

This route is intentionally separate from ``/api/articles`` so activating the
Render candidate UI cannot change Power Automate or Teams ranking/transport.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

log = logging.getLogger("push-balancer")
router = APIRouter()

_RECOMMENDATIONS_LIMIT = 10
_RECOMMENDATIONS_MIN_SCORE = 70.0


def _score_sync_metadata(articles: list[dict]) -> dict:
    synced = sum(
        1
        for article in articles
        if article.get("scoreSource") == "consumer_recommendations_api"
        and article.get("scoreApiStatus") == "ok"
    )
    return {
        "required": True,
        "source": "consumer_recommendations_api",
        "status": "ok" if synced == len(articles) else "partial",
        "syncedCount": synced,
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


def _get_recommendations_api_candidates(offset: int, limit: int) -> dict[str, Any]:
    """Project the exact authoritative Recommendations API query for the UI."""
    from app.notifications.teams import annotate_candidates_with_teams_decisions
    from app.routers.consumer import _load_consumer_recommendations

    page_limit = min(limit, _RECOMMENDATIONS_LIMIT)
    # This is the same service function used behind the authenticated endpoint.
    # Keeping the call in-process avoids a self-HTTP request and keeps the
    # credential out of the browser while preserving the exact API contract.
    consumer_payload = _load_consumer_recommendations(
        offset=offset,
        limit=page_limit,
        category=None,
        min_score=_RECOMMENDATIONS_MIN_SCORE,
        include_explanations=False,
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
        "limit": page_limit,
        "fetchedAt": snapshot_at,
        "livePushes": consumer_payload.get("livePushes") or [],
        "livePushCount": int(consumer_payload.get("livePushCount") or 0),
        "scoreSync": {
            **_score_sync_metadata(articles),
            "snapshotAt": snapshot_at,
        },
    }


@router.get("/api/editorial-one/articles")
def get_editorial_one_articles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=60, ge=1, le=200),
) -> JSONResponse:
    """Return the exact Recommendations API projection used by the Render UI."""
    return JSONResponse(content=_get_recommendations_api_candidates(offset, limit))
