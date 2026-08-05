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
        1 for article in articles if article.get("scoreSource") == "internal_score_api"
    )
    return {
        "required": required,
        "source": "editorial_one_score_api" if required else "render",
        "status": "ok" if not required or synced == len(articles) else "partial",
        "syncedCount": synced if required else 0,
        "totalCount": len(articles),
    }


@router.get("/api/editorial-one/articles")
def get_editorial_one_articles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=60, ge=1, le=200),
) -> JSONResponse:
    """Return candidates with a fail-closed Editorial One score projection."""
    if not EDITORIAL_ONE_SCORE_API_ENABLED:
        payload = build_articles_payload(offset=offset, limit=limit)
        payload["scoreSync"] = _score_sync_metadata(
            payload.get("articles") or [],
            required=False,
        )
        return JSONResponse(content=payload)

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
