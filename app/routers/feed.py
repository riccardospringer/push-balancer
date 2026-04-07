"""app/routers/feed.py — Feed- und Competitor-Endpunkte.

GET /api/feed                    — BILD Sitemap (XML-Proxy)
GET /api/competitors             — Alle Competitor-Feeds (gecacht)
GET /api/competitor/{name}       — Einzelner Competitor-Feed
GET /api/sport-competitors       — Sport-Competitor-Feeds
GET /api/sport-europa            — Sport-Europa-Feeds
GET /api/sport-global            — Sport-Global-Feeds
GET /api/international           — Internationale Feeds (gecacht)
GET /api/international/{name}    — Einzelner internationaler Feed
POST /api/competitors/xor        — Batch-XOR via Wort-Performance-Scoring
"""
import json
import logging
import re
import ssl
import time
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException, Query
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

# ── In-Memory URL-Cache ────────────────────────────────────────────────────
_url_cache: dict[str, tuple[float, bytes]] = {}
_URL_CACHE_MAX = 80  # max. Einträge — verhindert unbegrenztes Wachstum

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
    (flaches Dict für push-balancer.html Kompatibilität)
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
    (flaches Dict für push-balancer.html Kompatibilität)
    """
    try:
        parsed = get_cached_feeds("sport_competitors")
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
        parsed = get_cached_feeds("sport_competitors")
        if not parsed:
            parsed = _fetch_feeds_live(SPORT_COMPETITOR_FEEDS)
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
