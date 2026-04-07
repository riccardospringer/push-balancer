"""app/routers/feed.py — Feed- und Competitor-Endpunkte.

GET /api/feed                    — BILD Sitemap (XML-Proxy)
GET /api/competitors             — Alle Competitor-Feeds (gecacht)
GET /api/competitor/{name}       — Einzelner Competitor-Feed
GET /api/sport-competitors       — Sport-Competitor-Feeds
GET /api/sport-europa            — Sport-Europa-Feeds
GET /api/sport-global            — Sport-Global-Feeds
GET /api/international           — Internationale Feeds (gecacht)
GET /api/international/{name}    — Einzelner internationaler Feed
POST /api/competitor-xor         — Batch-XOR via Wort-Performance-Scoring
"""
import json
import logging
import re
import ssl
import time
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from app.config import (
    BILD_SITEMAP,
    CACHE_TTL,
    COMPETITOR_FEEDS,
    INTERNATIONAL_FEEDS,
    SPORT_COMPETITOR_FEEDS,
    SPORT_EUROPA_FEEDS,
    SPORT_GLOBAL_FEEDS,
)
from app.research.worker import _xor_perf_cache, _xor_perf_lock, get_cached_feeds

log = logging.getLogger("push-balancer")
router = APIRouter()

# ── In-Memory URL-Cache ────────────────────────────────────────────────────
_url_cache: dict[str, tuple[float, bytes]] = {}

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
        _url_cache[url] = (now, data)
        return data
    except Exception as e:
        log.warning("[feed] _fetch_url Fehler für %s: %s", url[:60], e)
        return None


class CompetitorXorRequest(BaseModel):
    titles: list[Any]


def _parse_rss_items(xml_bytes: bytes, max_items: int = 30) -> list[dict]:
    """Parst RSS/Atom XML zu kompakten Dicts. Portiert aus push-balancer-server.py."""
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


@router.get("/api/feed")
def get_feed() -> Response:
    """Proxy zur BILD News-Sitemap (XML)."""
    data = _fetch_url(BILD_SITEMAP)
    if data is None:
        raise HTTPException(status_code=502, detail="BILD Sitemap nicht erreichbar")
    return Response(content=data, media_type="application/xml; charset=utf-8")


def _fetch_feeds_live(feeds: dict[str, str]) -> dict:
    """Fetcht alle Feeds parallel via ThreadPool als Live-Fallback."""
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
    """Liefert alle Competitor-Feeds aus Background-Cache; live-fetcht wenn Cache leer."""
    try:
        parsed = get_cached_feeds("competitors")
        if not parsed:
            parsed = _fetch_feeds_live(COMPETITOR_FEEDS)
        return JSONResponse(content=parsed)
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
    """Liefert Sport-Competitor-Feeds aus Background-Cache."""
    try:
        parsed = get_cached_feeds("sport_competitors")
        return JSONResponse(content=parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport competitor feeds error: {e}")


@router.get("/api/sport-europa")
def get_sport_europa() -> JSONResponse:
    """Liefert Sport-Europa-Feeds aus Background-Cache."""
    try:
        parsed = get_cached_feeds("sport_europa")
        return JSONResponse(content=parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport Europa feeds error: {e}")


@router.get("/api/sport-global")
def get_sport_global() -> JSONResponse:
    """Liefert Sport-Global-Feeds aus Background-Cache."""
    try:
        parsed = get_cached_feeds("sport_global")
        return JSONResponse(content=parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sport Global feeds error: {e}")


@router.get("/api/international")
def get_international() -> JSONResponse:
    """Liefert alle internationalen Feeds aus Background-Cache."""
    try:
        parsed = get_cached_feeds("international")
        return JSONResponse(content=parsed)
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


@router.post("/api/competitor-xor")
def post_competitor_xor(body: CompetitorXorRequest) -> JSONResponse:
    """Batch-XOR via Wort-Performance-Scoring.

    Nutzt vorberechneten _xor_perf_cache für O(W)-Lookup pro Titel
    statt O(N*M)-Jaccard über historische Pushes.
    Ergebnis: <50ms, bessere Differenzierung.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py:
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

    import datetime
    cur_hour = datetime.datetime.now().hour
    simplified: dict = {}

    for item in titles:
        if isinstance(item, str):
            title, cat, hour = item, None, cur_hour
        else:
            title = item.get("title", "")
            cat = item.get("cat")
            hour = item.get("hour", cur_hour)
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
            "predicted_or": predicted_or,
            "basis": "word_perf" if word_scores else "global_avg",
            "cat": cat,
            "is_liveticker": is_lt,
            "is_eilmeldung": is_eil,
            "word_matches": len(word_scores),
        }

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    return JSONResponse(content={
        "results": simplified,
        "count": len(simplified),
        "elapsed_ms": elapsed_ms,
    })
