"""app/routers/feed.py — Feed- und Competitor-Endpunkte.

GET /api/feed                    — BILD Sitemap (XML-Proxy)
GET /api/competitor/{name}       — Einzelner Competitor-Feed
GET /api/sport-europa            — Sport-Europa-Feeds
GET /api/sport-global            — Sport-Global-Feeds
GET /api/international           — Internationale Feeds (gecacht)
GET /api/international/{name}    — Einzelner internationaler Feed
POST /api/competitors/xor        — Batch-XOR via Wort-Performance-Scoring
"""
import logging
import datetime
import json
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, JSONResponse

from app.auth import require_admin_key
from pydantic import BaseModel

from app.config import (
    ARTICLE_PREDICTION_ENRICHMENT_ENABLED,
    BILD_SITEMAP,
    CACHE_TTL,
    COMPETITOR_FEEDS,
    INTERNATIONAL_FEEDS,
    LIVE_FEED_FALLBACK_ENABLED,
    SPORT_COMPETITOR_FEEDS,
    SPORT_EUROPA_FEEDS,
    SPORT_GLOBAL_FEEDS,
)
from app.research.worker import _xor_perf_cache, _xor_perf_lock, get_cached_feeds

# ── Outlet-Farben für Konkurrenz-Tab ──────────────────────────────────────
_OUTLET_COLORS: dict[str, str] = {
    "welt":          "#003F5C",
    "spiegel":       "#E2001A",
    "focus":         "#F7A600",
    "ntv":           "#002A6E",
    "tagesschau":    "#003366",
    "faz":           "#003F7D",
    "sz":            "#B90000",
    "stern":         "#E31C23",
    "t-online":      "#D9002C",
    "zeit":          "#D6121F",
    "kicker":        "#008F5D",
    "sportschau":    "#004F9F",
    "transfermarkt": "#1D8E3E",
    "sport_de":      "#FF6600",
    "spiegel_sport": "#E2001A",
    "faz_sport":     "#003F7D",
    "rp_sport":      "#C80000",
    "tz_sport":      "#002A6E",
    "11freunde":     "#FF6B00",
}

log = logging.getLogger("push-balancer")
router = APIRouter()

_ARTICLE_CATEGORY_SCORES: dict[str, float] = {
    "politik": 70.0,
    "news": 72.0,
    "crime": 74.0,
    "verbraucher": 72.0,
    "sport": 72.0,
    "wirtschaft": 68.0,
    "unterhaltung": 68.0,
    "wetter": 70.0,
    "regional": 64.0,
    "digital": 66.0,
}
_ARTICLE_BREAKING_KEYWORDS = (
    "EIL",
    "EILMELDUNG",
    "BREAKING",
    "LIVE",
    "EXKLUSIV",
    "SCHOCK",
    "WARNUNG",
)
_ARTICLE_BREAKING_RE = re.compile(
    r"\b(?:eilmeldung|eil|breaking|live|exklusiv|schock|warnung)\b",
    re.IGNORECASE,
)
_ARTICLE_EILMELDUNG_RE = re.compile(
    r"\b(?:eilmeldung|eil)\b",
    re.IGNORECASE,
)

_VIDEO_URL_MARKERS = (
    "/video/",
    "/videos/",
    "-video-",
)
_VIDEO_TITLE_MARKERS = (
    "video",
    "im video",
    "hier sehen sie",
    "hier seht ihr",
    "aufnahmen",
    "clip",
)

# ── In-Memory URL-Cache ────────────────────────────────────────────────────
_url_cache: dict[str, tuple[float, bytes]] = {}
_URL_CACHE_MAX = 80  # max. Einträge — verhindert unbegrenztes Wachstum
_article_prediction_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_ARTICLE_PREDICTION_CACHE_MAX = 256
_ARTICLE_PREDICTION_CACHE_TTL = max(CACHE_TTL, 900)
_LOW_CONFIDENCE_PREDICTION_METHODS = {"global_avg", "error_fallback"}

try:
    import certifi as _certifi
    _SSL_CTX = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _fetch_url(url: str) -> bytes | None:
    """Holt URL-Inhalt mit In-Memory-Cache (TTL: CACHE_TTL Sekunden)."""
    now = time.time()
    cached = _url_cache.get(url)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)",
        })
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = resp.read()
        # LRU-Eviction: ältesten Eintrag entfernen wenn Limit erreicht
        if len(_url_cache) >= _URL_CACHE_MAX:
            oldest = min(_url_cache, key=lambda k: _url_cache[k][0])
            del _url_cache[oldest]
        _url_cache[url] = (now, data)
        return data
    except Exception as e:
        log.warning("[feed] _fetch_url Fehler für %s: %s", url[:60], e)
        return None


def _article_prediction_cache_get(cache_key: str) -> dict[str, Any] | None:
    now = time.time()
    cached = _article_prediction_cache.get(cache_key)
    if cached and now - cached[0] < _ARTICLE_PREDICTION_CACHE_TTL:
        payload = cached[1]
        if isinstance(payload, dict):
            return dict(payload)
        return {"predicted_or": payload}
    if cached:
        _article_prediction_cache.pop(cache_key, None)
    return None


def _article_prediction_cache_set(cache_key: str, result: dict[str, Any]) -> None:
    if len(_article_prediction_cache) >= _ARTICLE_PREDICTION_CACHE_MAX:
        oldest = min(_article_prediction_cache, key=lambda k: _article_prediction_cache[k][0])
        del _article_prediction_cache[oldest]
    _article_prediction_cache[cache_key] = (time.time(), dict(result))


def _apply_prediction_result(article: dict[str, Any], result: dict[str, Any] | None) -> None:
    if not isinstance(result, dict):
        return
    predicted_or = result.get("predicted_or")
    try:
        predicted_value = float(predicted_or)
    except (TypeError, ValueError):
        return

    basis_method = str(result.get("basis_method") or "").strip()
    confidence_raw = result.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None

    is_fallback = basis_method in _LOW_CONFIDENCE_PREDICTION_METHODS or (
        confidence is not None and confidence <= 0.1
    )
    article["predictedORBasis"] = basis_method or None
    article["predictedORConfidence"] = round(confidence, 3) if confidence is not None else None
    article["predictedORIsFallback"] = is_fallback
    article["predictedOR"] = None if is_fallback else round(predicted_value / 100, 4)


class CompetitorXorRequest(BaseModel):
    titles: list[Any]


def _parse_rss_items(xml_bytes: bytes, max_items: int = 30) -> list[dict]:
    """Parst RSS/Atom XML zu kompakten Dicts. Portiert aus dem frueheren Monolithen."""
    import xml.etree.ElementTree as ET
    xml_str = xml_bytes.decode("utf-8", errors="replace")
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_str)
        # RSS 2.0
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            desc = (item.findtext("description") or "").strip()[:200]
            cats = [c.text.strip() for c in item.findall("category") if c.text]
            items.append({"t": title, "l": link, "p": pub, "d": desc, "c": cats})
            if len(items) >= max_items:
                break
        if not items:
            # Atom fallback
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//a:entry", ns):
                title = (entry.findtext("a:title", "", ns) or "").strip()
                link_el = entry.find("a:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                pub = (entry.findtext("a:published", "", ns) or entry.findtext("a:updated", "", ns) or "").strip()
                desc = (entry.findtext("a:summary", "", ns) or "").strip()[:200]
                items.append({"t": title, "l": link, "p": pub, "d": desc, "c": []})
                if len(items) >= max_items:
                    break
    except ET.ParseError:
        pass
    # Live-Ticker markieren
    for it in items:
        tl = (it.get("t") or "").lower()
        ll = (it.get("l") or "").lower()
        it["lt"] = bool(
            re.search(r'/(liveticker|newsticker|news-ticker|alle-news|news-blog)(/|$|\?)', ll)
            or re.search(r'/live[\d/]', ll)
            or re.search(r'/ticker(/|$|\?)', ll)
            or re.search(r'live[\s-]?ticker|news[\s-]?ticker|newsticker', tl)
            or re.search(r'alle\s+(news|infos|entwicklungen|meldungen)\s+(zu|zum|zur|im|aus|über)', tl)
        )
    return items


def _infer_article_category(url: str, title: str) -> str:
    url_lower = url.lower()
    title_lower = title.lower()
    if "/sport/" in url_lower:
        return "sport"
    if "/politik/" in url_lower:
        return "politik"
    if "/unterhaltung/" in url_lower or "/leute/" in url_lower:
        return "unterhaltung"
    if "/geld/" in url_lower or "/wirtschaft/" in url_lower:
        return "wirtschaft"
    if "/ratgeber/" in url_lower or any(
        word in title_lower
        for word in ("preise", "kosten", "gebühren", "gebuehren", "abzocke", "rückruf", "rueckruf", "kunden")
    ):
        return "verbraucher"
    if "/digital/" in url_lower or "ki" in title_lower:
        return "digital"
    if "/regional/" in url_lower:
        return "regional"
    if any(
        word in title_lower
        for word in ("mord", "messer", "polizei", "razzia", "festnahme", "gericht", "prozess", "leiche", "täter", "taeter")
    ):
        return "crime"
    if any(word in title_lower for word in ("wetter", "unwetter", "sturm", "gewitter", "hitze", "schnee", "warnung")):
        return "wetter"
    return "news"


def _infer_article_type(url: str, title: str) -> str:
    text = f"{url} {title}".lower()
    if any(marker in url.lower() for marker in _VIDEO_URL_MARKERS):
        return "video"
    if any(marker in text for marker in _VIDEO_TITLE_MARKERS):
        return "video"
    return "editorial"


def _parse_pub_timestamp(pub_date: str) -> float:
    import datetime
    from email.utils import parsedate_to_datetime

    if not pub_date:
        return 0.0

    try:
        normalized = pub_date.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(normalized).timestamp()
    except Exception:
        pass

    try:
        return parsedate_to_datetime(pub_date).timestamp()
    except Exception:
        return 0.0


def _has_article_breaking_signal(title: str) -> bool:
    return bool(_ARTICLE_BREAKING_RE.search(title or ""))


def _has_article_eilmeldung_signal(title: str) -> bool:
    return bool(_ARTICLE_EILMELDUNG_RE.search(title or ""))


def _build_article_score(category: str, title: str, pub_date: str, article_type: str) -> tuple[float, str]:
    now_ts = time.time()
    pub_ts = _parse_pub_timestamp(pub_date)
    age_hours = max(0.0, (now_ts - pub_ts) / 3600) if pub_ts > 0 else 6.0
    title_lower = title.lower()

    base = {
        "politik": 58.0,
        "sport": 55.0,
        "news": 57.0,
        "wirtschaft": 55.0,
        "unterhaltung": 51.0,
        "regional": 52.0,
        "digital": 52.0,
    }.get(category, 50.0)

    score = base
    reasons: list[str] = []

    if age_hours <= 1:
        score += 18.0
        reasons.append("sehr frisch")
    elif age_hours <= 3:
        score += 12.0
        reasons.append("frisch")
    elif age_hours <= 8:
        score += 6.0
    elif age_hours >= 24:
        score -= 8.0
        reasons.append("älter")

    if _has_article_breaking_signal(title):
        score += 8.0
        reasons.append("breaking")

    if "liveticker" in title_lower or "live" in title_lower:
        score += 4.0
        reasons.append("live")

    # kein genereller Sport-Bonus — verhindert Sport-Monopol in der Liste

    if article_type == "video":
        score -= 9.0
        reasons.append("video-abschlag")

    return max(18.0, min(score, 100.0)), ", ".join(reasons[:3])


def _extract_sitemap_articles(xml_bytes: bytes, max_items: int = 200) -> list[dict[str, Any]]:
    ns_sitemap = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    ns_news = {"news": "http://www.google.com/schemas/sitemap-news/0.9"}
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    articles: list[dict[str, Any]] = []
    for url_el in root.findall("sm:url", ns_sitemap):
        loc = (url_el.findtext("sm:loc", "", ns_sitemap) or "").strip()
        news_el = url_el.find("news:news", ns_news)
        if news_el is None or not loc:
            continue

        title = (news_el.findtext("news:title", "", ns_news) or "").strip()
        pub_date = (news_el.findtext("news:publication_date", "", ns_news) or "").strip()
        if not title:
            continue

        category = _infer_article_category(loc, title)
        article_type = _infer_article_type(loc, title)
        score, score_reason = _build_article_score(category, title, pub_date, article_type)
        is_breaking = _has_article_breaking_signal(title)
        is_eilmeldung = _has_article_eilmeldung_signal(title)

        articles.append(
            {
                "id": loc,
                "url": loc,
                "title": title,
                "category": category,
                "pubDate": pub_date,
                "score": min(score, 100.0),
                "scoreReason": score_reason,
                "predictedOR": None,
                "predictedORBasis": None,
                "predictedORConfidence": None,
                "predictedORIsFallback": False,
                "isBreaking": is_breaking,
                "isEilmeldung": is_eilmeldung,
                "isSport": category == "sport",
                "isVideo": article_type == "video",
                "isPlusArticle": False,
                "type": article_type,
            }
        )
        if len(articles) >= max_items:
            break

    return articles


@router.get("/api/feed")
def get_feed() -> Response:
    """Proxy zur BILD News-Sitemap (XML)."""
    data = _fetch_url(BILD_SITEMAP)
    if data is None:
        raise HTTPException(status_code=502, detail="BILD Sitemap nicht erreichbar")
    return Response(content=data, media_type="application/xml; charset=utf-8")


def build_articles_payload(offset: int = 0, limit: int = 60) -> dict[str, Any]:
    """Return article candidates from the BILD sitemap as a JSON-ready payload."""
    data = _fetch_url(BILD_SITEMAP)
    if data is None:
        raise HTTPException(status_code=502, detail="BILD sitemap is not reachable")

    try:
        articles = _extract_sitemap_articles(data, max_items=max(offset + limit, 120))
    except ET.ParseError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid sitemap XML: {exc}") from exc

    now = datetime.datetime.now()
    now_ts = int(now.timestamp())
    history: list[dict[str, Any]] = []
    research_state: dict[str, Any] = {}

    if ARTICLE_PREDICTION_ENRICHMENT_ENABLED:
        try:
            from app.ml.predict import predict_or
            from app.research.worker import _research_state

            research_state = _research_state
            history = _research_state.get("push_data") or []
            for article in articles:
                cache_key = "|".join(
                    [
                        article["url"],
                        article["title"],
                        article["pubDate"],
                        article["category"],
                        "1" if article["isEilmeldung"] else "0",
                    ]
                )
                cached_prediction = _article_prediction_cache_get(cache_key)
                if cached_prediction is not None:
                    _apply_prediction_result(article, cached_prediction)
                    continue
                result = predict_or(
                    {
                        "title": article["title"],
                        "headline": article["title"],
                        "cat": article["category"],
                        "hour": now.hour,
                        "ts_num": now_ts,
                        "is_eilmeldung": article["isEilmeldung"],
                        "link": article["url"],
                        "channels": [],
                        "pubDate": article["pubDate"],
                    },
                    _research_state,
                )
                if result and (result or {}).get("predicted_or") is not None:
                    _apply_prediction_result(article, result)
                    if not article.get("predictedORIsFallback"):
                        _article_prediction_cache_set(cache_key, result)
        except Exception as exc:
            log.warning("[articles] prediction enrichment failed: %s", exc)

    try:
        if not history:
            from app.research.worker import _research_state

            research_state = _research_state
            history = _research_state.get("push_data") or []

        from app.scoring.editorial import rebalance_push_mix, score_push_candidate

        for article in articles:
            editorial_score = score_push_candidate(
                {
                    "title": article["title"],
                    "cat": article["category"],
                    "hour": now.hour,
                    "ts_num": now_ts,
                    "is_eilmeldung": article["isEilmeldung"],
                    "isVideo": article.get("isVideo"),
                    "video": article.get("isVideo"),
                    "pubDate": article["pubDate"],
                    "link": article["url"],
                },
                history=history,
                state=research_state,
                predicted_or=article.get("predictedOR"),
            )
            article.update(editorial_score)
        articles = rebalance_push_mix(articles, history=history, target_ts=now_ts)
    except Exception as exc:
        log.warning("[articles] editorial scoring enrichment failed: %s", exc)
        articles.sort(key=lambda article: (article["score"], article["pubDate"]), reverse=True)

    selected = articles[offset : offset + limit]

    try:
        from app.notifications.teams import annotate_candidates_with_teams_decisions

        selected = annotate_candidates_with_teams_decisions(selected)
    except Exception as exc:
        log.warning("[articles] Teams decision annotation failed: %s", exc)

    return {
        "articles": selected,
        "total": len(articles),
        "count": len(articles),
        "offset": offset,
        "limit": limit,
        "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@router.get("/api/articles")
def get_articles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=60, ge=1, le=200),
) -> JSONResponse:
    """Return article candidates from the BILD sitemap as a typed JSON collection."""
    return JSONResponse(content=build_articles_payload(offset=offset, limit=limit))


def _iso_from_unix_ts(ts: int | float | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromtimestamp(float(ts)).isoformat()
    except (TypeError, ValueError, OSError):
        return None


@router.get("/api/teams-alerts")
def get_teams_alerts(limit: int = Query(default=20, ge=1, le=100)) -> JSONResponse:
    """Return recent Teams recommendation decisions for dashboard transparency."""
    try:
        from app.database import teams_alert_list_recent

        rows = teams_alert_list_recent(limit)
    except Exception as exc:
        log.warning("[teams-alerts] could not load Teams alert history: %s", exc)
        rows = []

    items = []
    for row in rows:
        last_alert_ts = int(row.get("last_alert_ts") or 0)
        last_decision_ts = int(row.get("last_decision_ts") or 0)
        items.append(
            {
                "articleKey": row.get("article_key") or "",
                "articleId": row.get("article_id") or "",
                "articleUrl": row.get("article_url") or "",
                "articleTitle": row.get("article_title") or "",
                "status": row.get("status") or "",
                "score": float(row.get("last_score") or 0.0),
                "predictedOR": float(row.get("last_predicted_or") or 0.0),
                "reason": row.get("last_reason") or "",
                "lastError": row.get("last_error") or "",
                "alertCount": int(row.get("alert_count") or 0),
                "lastAlertAt": _iso_from_unix_ts(last_alert_ts),
                "lastDecisionAt": _iso_from_unix_ts(last_decision_ts),
                "isBreaking": bool(row.get("last_is_breaking") or 0),
            }
        )

    return JSONResponse(
        content={
            "items": items,
            "total": len(items),
            "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )


def _parse_db_json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError):
        return fallback


@router.get("/api/teams-recommendations")
def get_teams_recommendations(limit: int = Query(default=50, ge=1, le=200)) -> JSONResponse:
    """Return persisted Teams push suggestions and day-plan entries."""
    try:
        from app.database import teams_recommendation_list_recent

        rows = teams_recommendation_list_recent(limit)
    except Exception as exc:
        log.warning("[teams-recommendations] could not load recommendation history: %s", exc)
        rows = []

    items = []
    for row in rows:
        decided_ts = int(row.get("decided_at_ts") or 0)
        scheduled_ts = int(row.get("scheduled_for_ts") or 0)
        sent_ts = int(row.get("sent_at_ts") or 0)
        items.append(
            {
                "id": row.get("id") or "",
                "articleKey": row.get("article_key") or "",
                "articleId": row.get("article_id") or "",
                "articleUrl": row.get("article_url") or "",
                "articleTitle": row.get("article_title") or "",
                "section": row.get("section") or "",
                "type": row.get("recommendation_type") or "",
                "status": row.get("status") or "",
                "shouldNotify": bool(row.get("should_notify") or 0),
                "score": float(row.get("score") or 0.0),
                "teamsAlertScore": float(row.get("teams_alert_score") or 0.0),
                "teamsAlertThreshold": float(row.get("teams_alert_threshold") or 0.0),
                "editorialScore": float(row.get("editorial_score") or 0.0),
                "predictedOR": float(row.get("predicted_or") or 0.0),
                "predictedORLabel": row.get("predicted_or_label") or "",
                "expectedVisits": int(row.get("expected_visits") or 0),
                "dashboardRank": int(row.get("dashboard_rank") or 0),
                "summary": row.get("summary") or "",
                "reasons": _parse_db_json(row.get("reasons_json"), []),
                "blockingReasons": _parse_db_json(row.get("blocking_reasons_json"), []),
                "decision": _parse_db_json(row.get("decision_json"), {}),
                "sendStatus": row.get("send_status") or "",
                "sendError": row.get("send_error") or "",
                "decidedAt": _iso_from_unix_ts(decided_ts),
                "scheduledFor": _iso_from_unix_ts(scheduled_ts),
                "sentAt": _iso_from_unix_ts(sent_ts),
            }
        )

    return JSONResponse(
        content={
            "items": items,
            "total": len(items),
            "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )


@router.get("/api/teams-daily-plan")
def get_teams_daily_plan(
    date: str | None = Query(default=None, description="YYYY-MM-DD. Optional, default heute."),
    limit: int = Query(default=120, ge=15, le=200),
    min_items: int | None = Query(default=None, ge=1, le=30),
    max_items: int | None = Query(default=None, ge=1, le=30),
) -> JSONResponse:
    """Return a Teams-ready CvD daily push plan from the current article field."""
    from app.notifications.teams import TeamsAlertConfig, build_teams_daily_push_plan

    config = TeamsAlertConfig()
    requested_min = min_items or config.daily_plan_min_items
    source_limit = max(limit, min(200, int(requested_min or 15) * 4))
    payload = build_articles_payload(offset=0, limit=source_limit)
    candidates = payload.get("articles") or []
    plan = build_teams_daily_push_plan(
        candidates,
        config=config,
        target_date=date,
        min_items=min_items,
        max_items=max_items,
        persist=True,
    )
    plan["source"] = {
        "articlesFetched": len(candidates),
        "articleLimit": source_limit,
        "fetchedAt": payload.get("fetchedAt"),
    }
    return JSONResponse(content=plan)


@router.post("/api/teams-alerts/test", dependencies=[Depends(require_admin_key)])
def post_teams_alert_test() -> JSONResponse:
    """Send a clearly marked test message to the configured Teams channel.

    Uses the server-side webhook secret; the URL is never returned or logged.
    """
    from app.notifications.teams import TeamsAlertConfig, send_teams_test_notification

    config = TeamsAlertConfig()
    if not config.enabled:
        return JSONResponse(
            status_code=409,
            content={"ok": False, "error": "Teams alerts disabled (PUSH_TEAMS_ALERTS_ENABLED=false)"},
        )
    if not config.webhook_url:
        return JSONResponse(
            status_code=409,
            content={"ok": False, "error": "Teams webhook URL not configured"},
        )

    result = send_teams_test_notification(config)
    ok = bool(result.get("ok"))
    return JSONResponse(
        status_code=200 if ok else 502,
        content={
            "ok": ok,
            "sent": ok,
            "error": str(result.get("error") or ""),
            "sentAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


def _paginate_feed_dict(
    feeds: dict[str, list],
    offset: int,
    limit: int,
) -> dict:
    """Konvertiert ein Feed-Dict in einen paginierten Pagination-Envelope.

    Jedes Item enthält `name` und `articles`.
    """
    all_items = [{"name": name, "articles": articles} for name, articles in feeds.items()]
    total = len(all_items)
    items = all_items[offset: offset + limit]
    return {"items": items, "total": total, "offset": offset, "limit": limit}


def _fetch_feeds_live(feeds: dict[str, str]) -> dict:
    """Fetcht alle Feeds parallel via ThreadPool als Live-Fallback."""
    if not LIVE_FEED_FALLBACK_ENABLED:
        return {}
    import concurrent.futures
    result: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_url, url): name for name, url in feeds.items()}
        for fut, name in futures.items():
            try:
                xml_bytes = fut.result(timeout=10)
                result[name] = _parse_rss_items(xml_bytes) if xml_bytes else []
            except Exception:
                result[name] = []
    return result


@router.get("/api/competitors")
def get_competitors() -> JSONResponse:
    """Liefert alle Competitor-Feeds aus Background-Cache.

    Format: { "welt": [{t, l, p, d, c, lt}, ...], "spiegel": [...], ... }
    (flaches Dict fuer Kompatibilitaet mit dem frueheren HTML-Client)
    """
    try:
        parsed = get_cached_feeds("competitors")
        if not parsed:
            parsed = _fetch_feeds_live(COMPETITOR_FEEDS)
        return JSONResponse(content=parsed or {})
    except Exception as e:
        log.exception("[feed] Fehler in get_competitors")
        raise HTTPException(status_code=502, detail=f"Competitor feeds error: {e}")


@router.get("/api/competitor/{name}")
def get_competitor(name: str) -> Response:
    """Proxy für einen einzelnen Competitor-Feed (XML)."""
    url = COMPETITOR_FEEDS.get(name)
    if not url:
        raise HTTPException(status_code=404, detail=f"Unknown competitor: {name}")
    data = _fetch_url(url)
    if data is None:
        raise HTTPException(status_code=502, detail=f"Feed {name} nicht erreichbar")
    return Response(content=data, media_type="application/xml; charset=utf-8")


@router.get("/api/sport-competitors")
def get_sport_competitors() -> JSONResponse:
    """Liefert Sport-Competitor-Feeds aus Background-Cache.

    Format: { "kicker": [{t, l, p, d, c, lt}, ...], ... }
    (flaches Dict fuer Kompatibilitaet mit dem frueheren HTML-Client)
    """
    try:
        parsed = get_cached_feeds("sport_competitors")
        if not parsed:
            parsed = _fetch_feeds_live(SPORT_COMPETITOR_FEEDS)
        return JSONResponse(content=parsed or {})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport competitor feeds error: {e}")


@router.get("/api/sport-europa")
def get_sport_europa(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    """Liefert Sport-Europa-Feeds aus Background-Cache (paginiert)."""
    try:
        parsed = get_cached_feeds("sport_europa")
        if not parsed:
            parsed = _fetch_feeds_live(SPORT_EUROPA_FEEDS)
        return JSONResponse(content=_paginate_feed_dict(parsed or {}, offset, limit))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport Europa feeds error: {e}")


@router.get("/api/sport-global")
def get_sport_global(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    """Liefert Sport-Global-Feeds aus Background-Cache (paginiert)."""
    try:
        parsed = get_cached_feeds("sport_global")
        if not parsed:
            parsed = _fetch_feeds_live(SPORT_GLOBAL_FEEDS)
        return JSONResponse(content=_paginate_feed_dict(parsed or {}, offset, limit))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport Global feeds error: {e}")


@router.get("/api/international")
def get_international(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    """Liefert alle internationalen Feeds aus Background-Cache (paginiert)."""
    try:
        parsed = get_cached_feeds("international")
        if not parsed:
            parsed = _fetch_feeds_live(INTERNATIONAL_FEEDS)
        return JSONResponse(content=_paginate_feed_dict(parsed or {}, offset, limit))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"International feeds error: {e}")


@router.get("/api/international/{name}")
def get_international_feed(name: str) -> Response:
    """Proxy für einen einzelnen internationalen Feed (XML)."""
    url = INTERNATIONAL_FEEDS.get(name)
    if not url:
        raise HTTPException(status_code=404, detail=f"Unknown international feed: {name}")
    data = _fetch_url(url)
    if data is None:
        raise HTTPException(status_code=502, detail=f"Feed {name} nicht erreichbar")
    return Response(content=data, media_type="application/xml; charset=utf-8")


def _normalize_words(text: str) -> set[str]:
    """Normalisierte Wortmenge für Jaccard-Ähnlichkeitsvergleich."""
    STOP = {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einem",
        "einen", "eines", "und", "oder", "aber", "nicht", "mit", "von",
        "zu", "in", "an", "auf", "bei", "nach", "für", "ist", "sind",
        "wird", "wurde", "werden", "hat", "haben", "war", "waren", "wie",
        "auch", "sich", "aus", "am", "im", "zum", "zur", "als", "mehr",
        "noch", "nur", "so", "schon", "jetzt", "the", "a", "of", "to",
        "and", "is", "for", "on", "at", "that", "by", "this", "it",
    }
    return {
        w.strip(".,;:!?\"'()[]{}«»–—-").lower()
        for w in text.split()
        if len(w.strip(".,;:!?\"'()[]{}«»–—-")) > 3
    } - STOP


def _build_competitor_response(feeds: dict[str, list[dict]], outlet_colors: dict[str, str]) -> dict:
    """Konvertiert rohe Feed-Dicts zu CompetitorResponse (mit Hot/Exklusiv/Gap-Markierung)."""
    import datetime
    from email.utils import parsedate_to_datetime

    # Alle Items flattenen (Liveticker überspringen)
    flat: list[dict] = []
    for outlet, items in feeds.items():
        color = outlet_colors.get(outlet, "#666666")
        for item in items:
            title = (item.get("t") or "").strip()
            if not title or item.get("lt"):
                continue
            flat.append({
                "title": title,
                "url": item.get("l", ""),
                "pubDate": item.get("p", ""),
                "outlet": outlet,
                "outletColor": color,
                "_words": _normalize_words(title),
            })

    # Jaccard-Clustering: ähnliche Artikel verschiedener Outlets zusammenführen
    n = len(flat)
    merged_into: list[int] = list(range(n))
    for i in range(n):
        if merged_into[i] != i:
            continue
        wi = flat[i]["_words"]
        if not wi:
            continue
        for j in range(i + 1, n):
            if merged_into[j] != j or flat[j]["outlet"] == flat[i]["outlet"]:
                continue
            wj = flat[j]["_words"]
            if wj and len(wi & wj) / len(wi | wj) >= 0.2:
                merged_into[j] = i

    # Cluster-Dicts aufbauen
    clusters: dict[int, dict] = {}
    for i, rep in enumerate(merged_into):
        item = flat[i]
        if rep not in clusters:
            clusters[rep] = {
                "title": flat[rep]["title"],
                "url": flat[rep]["url"],
                "pubDate": flat[rep]["pubDate"],
                "outlet": flat[rep]["outlet"],
                "outletColor": flat[rep]["outletColor"],
                "outlets": set(),
            }
        clusters[rep]["outlets"].add(item["outlet"])

    # Sortierhilfe: pubDate → timestamp
    def _ts(d: str) -> float:
        try:
            return parsedate_to_datetime(d).timestamp()
        except Exception:
            pass
        try:
            return datetime.datetime.fromisoformat(d).timestamp()
        except Exception:
            return 0.0

    result: list[dict] = []
    for c in clusters.values():
        outlets = sorted(c["outlets"])
        n_out = len(outlets)
        result.append({
            "title": c["title"],
            "url": c["url"],
            "pubDate": c["pubDate"],
            "outlet": c["outlet"],
            "outletColor": c["outletColor"],
            "isGap": n_out >= 2,        # 2+ Outlets = Lücke (BILD fehlt)
            "isExklusiv": n_out == 1,   # Nur 1 Outlet = Exklusiv-Winkel
            "isHot": n_out >= 3,        # 3+ Outlets = Hot Topic
            "outlets": outlets,
        })

    result.sort(key=lambda x: _ts(x["pubDate"]), reverse=True)

    total = len(result)
    return {
        "items": result,
        "summary": {
            "total": total,
            "gaps": sum(1 for x in result if x["isGap"]),
            "exklusiv": sum(1 for x in result if x["isExklusiv"]),
            "hot": sum(1 for x in result if x["isHot"]),
        },
        "fetchedAt": datetime.datetime.now().isoformat(),
    }


@router.get("/api/feeds/competitor")
def get_feeds_competitor() -> JSONResponse:
    """Aufbereitete Redaktion-Competitor-Feeds im CompetitorResponse-Format."""
    try:
        parsed = get_cached_feeds("competitors")
        if not parsed:
            parsed = _fetch_feeds_live(COMPETITOR_FEEDS)
        return JSONResponse(content=_build_competitor_response(parsed or {}, _OUTLET_COLORS))
    except Exception as e:
        log.exception("[feed] Fehler in get_feeds_competitor")
        raise HTTPException(status_code=502, detail=f"Competitor feeds error: {e}")


@router.get("/api/feeds/competitor/sport")
def get_feeds_competitor_sport() -> JSONResponse:
    """Aufbereitete Sport-Competitor-Feeds im CompetitorResponse-Format."""
    try:
        parsed: dict[str, list[dict]] = {}
        for feed_type, configured_feeds in (
            ("sport_competitors", SPORT_COMPETITOR_FEEDS),
            ("sport_europa", SPORT_EUROPA_FEEDS),
            ("sport_global", SPORT_GLOBAL_FEEDS),
        ):
            cached = get_cached_feeds(feed_type)
            live = cached if cached else _fetch_feeds_live(configured_feeds)
            if isinstance(live, dict):
                parsed.update(live)
        return JSONResponse(content=_build_competitor_response(parsed or {}, _OUTLET_COLORS))
    except Exception as e:
        log.exception("[feed] Fehler in get_feeds_competitor_sport")
        raise HTTPException(status_code=502, detail=f"Sport competitor feeds error: {e}")


@router.post("/api/competitors/xor")
def post_competitor_xor(body: CompetitorXorRequest) -> JSONResponse:
    """Batch-XOR via Wort-Performance-Scoring.

    Nutzt vorberechneten _xor_perf_cache für O(W)-Lookup pro Titel
    statt O(N*M)-Jaccard über historische Pushes.
    Ergebnis: <50ms, bessere Differenzierung.

    IMPLEMENTIERUNGSHINWEIS:
        Vollstaendige Handler-Logik aus dem frueheren Monolithen:
        _serve_competitor_xor() (Zeile 14319) hierher migrieren.
    """
    titles = body.titles
    if not isinstance(titles, list) or len(titles) > 200:
        raise HTTPException(status_code=400, detail="titles must be a list with max 200 entries")

    t0 = time.monotonic()

    try:
        from app.research.worker import build_xor_perf_cache
        with _xor_perf_lock:
            cache_age = time.time() - _xor_perf_cache["built_at"]
        if cache_age > 1800 or not _xor_perf_cache["word_perf"]:
            build_xor_perf_cache()

        with _xor_perf_lock:
            wp = dict(_xor_perf_cache["word_perf"])
            global_avg = _xor_perf_cache["global_avg"]
    except Exception as e:
        log.warning("[competitor-xor] Cache-Fehler: %s", e)
        wp = {}
        global_avg = 4.77

    simplified: dict = {}

    for item in titles:
        if isinstance(item, str):
            title, cat = item, None
        else:
            title = item.get("title", "")
            cat = item.get("cat")
        if not title:
            continue

        tl = title.lower()
        is_lt = bool(
            re.search(r'live[\s-]?ticker|news[\s-]?ticker|newsticker', tl)
            or re.search(r'alle\s+(news|infos|entwicklungen|meldungen)\s+(zu|zum|zur|im|aus|über)', tl)
        )

        # Kategorie ableiten wenn nicht gegeben
        if not cat:
            if any(w in tl for w in ("bundesliga", "champions", "transfer", "pokal", "fc ", "bvb", "bayern", "dortmund", "formel")):
                cat = "Sport"
            elif any(w in tl for w in ("trump", "biden", "merz", "scholz", "bundestag", "minister", "ukraine", "putin", "israel")):
                cat = "Politik"
            elif any(w in tl for w in ("mord", "messer", "razzia", "polizei", "festnahme")):
                cat = "Regional"
            elif any(w in tl for w in ("helene", "gottschalk", "bohlen", "bachelor", "dschungel", "gntm", "promi")):
                cat = "Unterhaltung"
            else:
                cat = "News"

        is_eil = "+++" in title or "eilmeldung" in tl or "breaking" in tl

        # Wort-Performance-Scoring
        words = set(w.strip(".,;:!?\"'()[]{}") for w in tl.split())
        word_scores = [wp[w]["avg"] for w in words if w in wp and wp[w].get("n", 0) >= 3]
        word_score_avg = sum(word_scores) / len(word_scores) if word_scores else global_avg

        # Eilmeldungs-Bonus
        if is_eil:
            word_score_avg = word_score_avg * 1.15 + 1.0

        predicted_or = round(max(0.5, min(20.0, word_score_avg)), 2)

        simplified[title] = {
            "predictedOr": predicted_or,
            "basis": "word_perf" if word_scores else "global_avg",
            "cat": cat,
            "isLiveticker": is_lt,
            "isEilmeldung": is_eil,
            "wordMatches": len(word_scores),
        }

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    return JSONResponse(content={
        "results": simplified,
        "count": len(simplified),
        "elapsedMs": elapsed_ms,
    })
