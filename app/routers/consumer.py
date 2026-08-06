"""Versioned read-only API for downstream app consumers."""
from __future__ import annotations

import copy
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.auth import require_consumer_key
from app.routers.feed import build_articles_payload
from app.routers.push import _build_pushes_response, get_push_sync_status

router = APIRouter()

_LIVE_PUSH_LOOKBACK_HOURS = 24
_LIVE_PUSH_LIMIT = 100
_RECOMMENDATIONS_RESPONSE_TTL_SECONDS = 30.0
_RECOMMENDATIONS_CACHE_MAX_ENTRIES = 32
_recommendations_cache_lock = threading.Lock()
_recommendations_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


def _clear_recommendations_cache() -> None:
    """Clear the process-local authoritative response cache."""
    with _recommendations_cache_lock:
        _recommendations_cache.clear()


def _consumer_status_payload() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "status": "ok",
        "advisoryOnly": True,
        "actionAllowed": False,
        "authentication": {
            "bearer": True,
            "consumerKeyHeader": True,
        },
        "endpoints": {
            "recommendations": "/api/v1/recommendations",
            "articles": "/api/v1/articles",
            "scores": "/api/v1/scores",
        },
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_articles(
    articles: list[dict[str, Any]],
    category: str | None,
    min_score: float | None,
) -> list[dict[str, Any]]:
    filtered = articles
    if category:
        category_key = category.strip().lower()
        filtered = [
            article
            for article in filtered
            if str(article.get("category") or "").lower() == category_key
        ]
    if min_score is not None:
        filtered = [
            article
            for article in filtered
            if (_as_float(article.get("score")) or 0.0) >= min_score
        ]
    return filtered


def _consumer_article(article: dict[str, Any], *, include_explanations: bool) -> dict[str, Any]:
    article_id = str(article.get("id") or article.get("url") or "")
    score = _as_float(article.get("score"))
    predicted_open_rate = _as_float(article.get("predictedOR"))

    payload: dict[str, Any] = {
        "id": article_id,
        "url": article.get("url") or article_id,
        "title": article.get("title") or "",
        "category": article.get("category") or "news",
        "publishedAt": article.get("pubDate") or "",
        "score": round(score, 1) if score is not None else None,
        "predictedOpenRate": round(predicted_open_rate, 4)
        if predicted_open_rate is not None
        else None,
        "predictedOpenRateBasis": article.get("predictedORBasis"),
        "predictedOpenRateConfidence": article.get("predictedORConfidence"),
        "predictedOpenRateIsFallback": bool(article.get("predictedORIsFallback")),
        "priority": article.get("mixPriority") or "",
        "recommendedText": article.get("recommendedText") or article.get("title") or "",
        "isLivePush": False,
        "alreadySent": False,
        "flags": {
            "livePush": False,
            "alreadySent": False,
            "breaking": bool(article.get("isBreaking")),
            "eilmeldung": bool(article.get("isEilmeldung")),
            "sport": bool(article.get("isSport")),
            "video": bool(article.get("isVideo")),
            "plusArticle": bool(article.get("isPlusArticle")),
        },
    }
    if include_explanations:
        payload["explanation"] = {
            "reason": article.get("scoreReason") or "",
            "drivers": list(article.get("performanceDrivers") or []),
            "risks": list(article.get("risks") or []),
            "breakdown": article.get("scoreBreakdown") or {},
        }
    return payload


def _consumer_live_push(push: dict[str, Any]) -> dict[str, Any]:
    score = _as_float(push.get("pushScore"))
    predicted_open_rate = _as_float(push.get("predictedOR"))
    category = str(push.get("category") or "news")

    return {
        "id": str(push.get("id") or ""),
        "url": push.get("url") or "",
        "title": push.get("title") or "",
        "category": category,
        "sentAt": push.get("sentAt") or "",
        "channel": push.get("channel") or "",
        "score": round(score, 1) if score is not None and score > 0 else None,
        "predictedOpenRate": (
            round(predicted_open_rate, 4) if predicted_open_rate is not None else None
        ),
        "isLivePush": True,
        "alreadySent": True,
        "flags": {
            "livePush": True,
            "alreadySent": True,
            "sport": category.strip().lower() == "sport",
        },
    }


def _load_consumer_live_pushes(category: str | None) -> list[dict[str, Any]]:
    payload = _build_pushes_response(
        limit=_LIVE_PUSH_LIMIT,
        days=max(1, _LIVE_PUSH_LOOKBACK_HOURS // 24),
        sort="sentAt",
        category=category or "",
    )
    return [_consumer_live_push(push) for push in payload.get("pushes", [])]


def _with_current_live_pushes(
    payload: dict[str, Any],
    category: str | None,
) -> dict[str, Any]:
    """Attach live-push state without caching it with article scores."""
    live_pushes = _load_consumer_live_pushes(category)
    payload["livePushes"] = live_pushes
    payload["livePushCount"] = len(live_pushes)
    payload["livePushLookbackHours"] = _LIVE_PUSH_LOOKBACK_HOURS
    payload["livePushStatus"] = get_push_sync_status()
    return payload


def _load_consumer_articles(
    offset: int,
    limit: int,
    category: str | None,
    min_score: float | None,
    include_explanations: bool,
) -> dict[str, Any]:
    source_limit = max(offset + limit, 120)
    source_payload = build_articles_payload(offset=0, limit=source_limit)
    filtered = _filter_articles(source_payload["articles"], category, min_score)
    selected = filtered[offset : offset + limit]
    payload = {
        "apiVersion": "v1",
        "advisoryOnly": True,
        "actionAllowed": False,
        "articles": [
            _consumer_article(article, include_explanations=include_explanations)
            for article in selected
        ],
        "total": len(filtered),
        "count": len(selected),
        "offset": offset,
        "limit": limit,
        "fetchedAt": source_payload["fetchedAt"],
    }
    return _with_current_live_pushes(payload, category)


def _load_consumer_recommendations(
    *,
    offset: int,
    limit: int,
    category: str | None,
    min_score: float | None,
    include_explanations: bool,
) -> dict[str, Any]:
    """Return one short-lived, immutable Recommendations API response.

    The authenticated endpoint and the Render candidate adapter use this same
    function and cache key. Separate requests therefore receive the identical
    score response instead of recalculating freshness a few seconds apart.
    """
    key = (offset, limit, category, min_score, include_explanations)
    now = time.monotonic()
    with _recommendations_cache_lock:
        cached = _recommendations_cache.get(key)
        if cached is not None:
            age = now - float(cached["createdMonotonic"])
            if age < _RECOMMENDATIONS_RESPONSE_TTL_SECONDS:
                payload = copy.deepcopy(cached["payload"])
                return _with_current_live_pushes(payload, category)

        payload = _load_consumer_articles(
            offset=offset,
            limit=limit,
            category=category,
            min_score=min_score,
            include_explanations=include_explanations,
        )
        payload["kind"] = "recommendations"
        expired_keys = [
            cache_key
            for cache_key, entry in _recommendations_cache.items()
            if now - float(entry["createdMonotonic"])
            >= _RECOMMENDATIONS_RESPONSE_TTL_SECONDS
        ]
        for cache_key in expired_keys:
            _recommendations_cache.pop(cache_key, None)
        if len(_recommendations_cache) >= _RECOMMENDATIONS_CACHE_MAX_ENTRIES:
            oldest_key = min(
                _recommendations_cache,
                key=lambda cache_key: float(
                    _recommendations_cache[cache_key]["createdMonotonic"]
                ),
            )
            _recommendations_cache.pop(oldest_key, None)
        _recommendations_cache[key] = {
            "createdMonotonic": now,
            "payload": copy.deepcopy(payload),
        }
        return copy.deepcopy(payload)


@router.get("/api/v1/status", dependencies=[Depends(require_consumer_key)])
def get_consumer_status() -> JSONResponse:
    """Return consumer API readiness and integration metadata."""
    return JSONResponse(content=_consumer_status_payload())


@router.get("/api/v1/recommendations", dependencies=[Depends(require_consumer_key)])
def get_consumer_recommendations(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    category: str | None = Query(default=None),
    min_score: float | None = Query(default=60, ge=0, le=100, alias="minScore"),
    include_explanations: bool = Query(default=False, alias="includeExplanations"),
) -> JSONResponse:
    """Return the simplest drop-in list of ranked article recommendations."""
    payload = _load_consumer_recommendations(
        offset=offset,
        limit=limit,
        category=category,
        min_score=min_score,
        include_explanations=include_explanations,
    )
    return JSONResponse(content=payload)


@router.get("/api/v1/articles", dependencies=[Depends(require_consumer_key)])
def get_consumer_articles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    category: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100, alias="minScore"),
    include_explanations: bool = Query(default=True, alias="includeExplanations"),
) -> JSONResponse:
    """Return ranked article candidates for downstream app consumers."""
    payload = _load_consumer_articles(
        offset=offset,
        limit=limit,
        category=category,
        min_score=min_score,
        include_explanations=include_explanations,
    )
    return JSONResponse(content=payload)


@router.get("/api/v1/scores", dependencies=[Depends(require_consumer_key)])
def get_consumer_scores(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    category: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100, alias="minScore"),
) -> JSONResponse:
    """Return compact article score projections for downstream app consumers."""
    payload = _load_consumer_articles(
        offset=offset,
        limit=limit,
        category=category,
        min_score=min_score,
        include_explanations=False,
    )
    scores = [
        {
            "articleId": article["id"],
            "url": article["url"],
            "title": article["title"],
            "category": article["category"],
            "score": article["score"],
            "predictedOpenRate": article["predictedOpenRate"],
            "priority": article["priority"],
            "isLivePush": False,
            "alreadySent": False,
            "updatedAt": payload["fetchedAt"],
        }
        for article in payload["articles"]
    ]
    return JSONResponse(
        content={
            "apiVersion": "v1",
            "advisoryOnly": True,
            "actionAllowed": False,
            "scores": scores,
            "livePushes": payload["livePushes"],
            "livePushCount": payload["livePushCount"],
            "livePushLookbackHours": payload["livePushLookbackHours"],
            "livePushStatus": payload["livePushStatus"],
            "total": payload["total"],
            "count": len(scores),
            "offset": offset,
            "limit": limit,
            "fetchedAt": payload["fetchedAt"],
        }
    )
