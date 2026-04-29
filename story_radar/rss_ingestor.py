"""RSS ingestion — fetches German news feeds and builds story clusters."""

from __future__ import annotations

import hashlib
import logging
import re
import ssl
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

if TYPE_CHECKING:
    from .service import StoryRadarService

log = logging.getLogger("story-radar")

FEEDS = [
    ("spiegel",       "https://www.spiegel.de/schlagzeilen/index.rss"),
    ("welt",          "https://www.welt.de/feeds/topnews.rss"),
    ("tagesspiegel",  "https://www.tagesspiegel.de/contentexport/feed/home"),
    ("focus",         "https://rss.focus.de/fol/XML/rss_folnews.xml"),
    ("stern",         "https://www.stern.de/feed/standard/aktuell/"),
    ("zeit",          "https://newsfeed.zeit.de/index"),
    ("sueddeutsche",  "https://rss.sueddeutsche.de/rss/Topthemen"),
    ("faz",           "https://www.faz.net/rss/aktuell/"),
]

# For coverage check: what BILD already has
BILD_FEED = ("bild", "https://www.bild.de/rssfeeds/vw-alles/vw-alles-28748144,view=rss2.xml")

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "crime":    ["polizei", "täter", "angriff", "festnahme", "mord", "einbruch", "schuss", "messer",
                 "razzia", "verbrechen", "straftat", "verdächtiger", "flüchtig", "ermittlung"],
    "breaking": ["eilmeldung", "breaking", "aktuell:", "gerade:", "soeben"],
    "consumer": ["preise", "preiserhöhung", "kosten", "pendler", "bahn", "wetter", "steuer",
                 "krankenkasse", "strom", "gas", "miete", "rente", "renten", "versicherung"],
    "sport":    ["bundesliga", "dfb", "champions", "transfer", "fußball", "fc ", " sc ", " sv ",
                 "bvb", "fcb", "werder", "schalke", "tennis", "formel 1", "olympia"],
    "promi":    ["kate", "prinzessin", "prinz", "william", "harry", "royals", "celebrity",
                 "prominenz", "star ", "sängerin", "schauspieler", "influencer"],
    "politics": ["bundesregierung", "bundestag", "merz", "scholz", "habeck", "lindner", "spd",
                 "cdu", "grüne", "fdp", "afd", "koalition", "minister", "kanzler", "wahl"],
    "energy":   ["energie", "strom", "solar", "windkraft", "atomkraft", "gaspreise", "öl"],
    "health":   ["krankenhaus", "krankheit", "impfung", "medizin", "gesundheit", "corona",
                 "virus", "krebs", "arzt", "klinik"],
}


def _fetch_feed(url: str, timeout: int = 8) -> list[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Story-Radar/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            raw = resp.read(500_000)
        root = ET.fromstring(raw)
    except Exception as exc:
        log.debug("RSS fetch error %s: %s", url, exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)
    articles = []
    for item in items[:40]:
        title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
        summary = (item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or "").strip()
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        pub = item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns) or ""
        link = item.findtext("link") or item.findtext("atom:link", namespaces=ns) or ""
        if isinstance(link, ET.Element):
            link = link.get("href", "")
        if not title:
            continue
        articles.append({"title": title, "summary": summary[:500], "pub": pub, "link": link})
    return articles


def _parse_date(s: str) -> datetime:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s.strip(), fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_entities(text: str) -> list[str]:
    """Simple heuristic: capitalised word pairs."""
    words = text.split()
    entities: list[str] = []
    i = 0
    while i < len(words):
        w = re.sub(r"[^a-zA-ZäöüÄÖÜß\-]", "", words[i])
        if w and w[0].isupper() and len(w) > 2:
            nxt = re.sub(r"[^a-zA-ZäöüÄÖÜß\-]", "", words[i + 1]) if i + 1 < len(words) else ""
            if nxt and nxt[0].isupper() and len(nxt) > 2:
                entities.append(w + " " + nxt)
                i += 2
                continue
            entities.append(w)
        i += 1
    seen: dict[str, int] = {}
    for e in entities:
        seen[e] = seen.get(e, 0) + 1
    return [e for e, c in sorted(seen.items(), key=lambda x: -x[1]) if c >= 1][:8]


def _detect_topics(text: str) -> list[str]:
    tl = text.lower()
    found = [topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in tl for k in kws)]
    return found or ["news"]


def _cluster_id(title: str) -> str:
    return "rss-" + hashlib.md5(title.encode()).hexdigest()[:12]


def build_clusters(articles_by_source: dict[str, list[dict]]) -> list[dict]:
    """Group articles from multiple sources into clusters by similarity."""
    all_articles: list[tuple[str, dict]] = []
    for source, arts in articles_by_source.items():
        for art in arts:
            all_articles.append((source, art))

    groups: list[list[tuple[str, dict]]] = []
    used = set()

    for i, (src_i, art_i) in enumerate(all_articles):
        if i in used:
            continue
        group = [(src_i, art_i)]
        used.add(i)
        tok_i = _tokenize(art_i["title"] + " " + art_i["summary"])
        for j, (src_j, art_j) in enumerate(all_articles):
            if j in used or src_j == src_i:
                continue
            tok_j = _tokenize(art_j["title"] + " " + art_j["summary"])
            if _jaccard(tok_i, tok_j) >= 0.20:
                group.append((src_j, art_j))
                used.add(j)
        groups.append(group)

    now = datetime.now(timezone.utc)
    clusters = []
    for group in groups:
        # Pick the most detailed article as cluster representative
        rep_src, rep = max(group, key=lambda x: len(x[1]["summary"]))
        title = rep["title"]
        summary = rep["summary"] or " ".join(a["summary"] for _, a in group[:2])[:400]
        sources = list({src for src, _ in group})
        entities = _extract_entities(title + " " + summary)
        topics = _detect_topics(title + " " + summary)

        pub_dates = []
        for _, art in group:
            if art["pub"]:
                pub_dates.append(_parse_date(art["pub"]))
        first_seen = min(pub_dates) if pub_dates else now
        last_seen = max(pub_dates) if pub_dates else now

        clusters.append({
            "cluster_id": _cluster_id(title),
            "title": title,
            "summary": summary,
            "entities": entities,
            "topics": topics,
            "countries": ["DE"],
            "source_count": len(sources),
            "document_count": len(group),
            "first_seen_at": first_seen.isoformat(),
            "last_seen_at": last_seen.isoformat(),
            "documents": [
                {
                    "document_id": _cluster_id(a["title"]) + f"-{k}",
                    "source": s,
                    "title": a["title"],
                    "summary": a["summary"][:200],
                    "published_at": a["pub"] or now.isoformat(),
                    "url": a["link"],
                }
                for k, (s, a) in enumerate(group)
            ],
        })

    return clusters


def fetch_and_ingest(service: "StoryRadarService") -> int:
    """Fetch all RSS feeds, build clusters, push into service. Returns cluster count."""
    articles_by_source: dict[str, list[dict]] = {}
    for name, url in FEEDS:
        arts = _fetch_feed(url)
        if arts:
            articles_by_source[name] = arts
            log.info("[RSS] %s: %d articles", name, len(arts))

    if not articles_by_source:
        log.warning("[RSS] No feeds fetched")
        return 0

    clusters = build_clusters(articles_by_source)
    log.info("[RSS] Built %d clusters from %d sources", len(clusters), len(articles_by_source))

    if clusters:
        service.rescore(clusters)

    return len(clusters)


def start_background_ingestor(service: "StoryRadarService", interval: int = 300) -> threading.Thread:
    """Start a background thread that re-ingests RSS feeds every `interval` seconds."""

    def _loop():
        # First run immediately
        try:
            n = fetch_and_ingest(service)
            log.info("[RSS] Initial ingest: %d clusters", n)
        except Exception as exc:
            log.exception("[RSS] Initial ingest failed: %s", exc)

        while True:
            time.sleep(interval)
            try:
                n = fetch_and_ingest(service)
                log.info("[RSS] Refresh: %d clusters", n)
            except Exception as exc:
                log.exception("[RSS] Refresh failed: %s", exc)

    t = threading.Thread(target=_loop, daemon=True, name="rss-ingestor")
    t.start()
    return t
