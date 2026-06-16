"""Versioned read-only API for downstream app consumers."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.auth import require_consumer_key
from app.routers.feed import build_articles_payload

router = APIRouter()


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
        "priority": article.get("mixPriority") or "",
        "recommendedText": article.get("recommendedText") or article.get("title") or "",
        "flags": {
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

    return {
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
    payload = _load_consumer_articles(
        offset=offset,
        limit=limit,
        category=category,
        min_score=min_score,
        include_explanations=include_explanations,
    )
    payload["kind"] = "recommendations"
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
            "total": payload["total"],
            "count": len(scores),
            "offset": offset,
            "limit": limit,
            "fetchedAt": payload["fetchedAt"],
        }
    )
