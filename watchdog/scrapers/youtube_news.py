#!/usr/bin/env python3
"""
Scraper: YouTube Amateur-Videos — Augenzeugenvideos von Einsatzorten.

Sucht gezielt nach Amateur-/Handyvideos zu aktuellen Events.
Nutzt YouTube RSS-Feeds von lokalen Kanälen und Bürger-Reportern,
NICHT von professionellen Nachrichtensendern.

Liefert: embed-fähige YouTube-URLs für Video-Einbettung im Dashboard.
"""

import asyncio
import logging
import re
import ssl
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import certifi

log = logging.getLogger("watchdog.youtube_news")

# ---------------------------------------------------------------------------
# Lokale / Amateur YouTube-Kanäle
# ---------------------------------------------------------------------------

YOUTUBE_CHANNELS = {
    # Blaulicht / Einsatz-Reporter (Amateur!)
    "NEWS5 (Franken)": "UC-Bx1GhdGtV1F3d1Bbbb4Hw",
    "BildTV Regional": "UCSSGkeIvfiUQbnqAiMDT4eQ",
    "Blaulicht Report": "UCVUvf-_8X0k7A3WG0AlOUYg",
    "HEIDELBERG24": "UCcNiTbewwb2hFdpqXQkY5Jg",

    # Regionale Sender (nahe am Geschehen, oft Amateurqualität)
    "TV Mainfranken": "UCQPxiZfpOfYOHYs3ql5eJTg",
    "Hamburg 1": "UCVo84MnVIp4LHgEmLwwnFRg",
    "RheinMain TV": "UCXMiVjH_abCqUfFq1SDoF8g",
    "baden.fm": "UCL0fNK2XQKDJ7FJafsjJe5Q",
    "Franken Fernsehen": "UC2XHXg2XHjzwJGKsL1J9pjg",
    "SachsenFernsehen": "UCcYzLCs3zrQIBVHcR1IKftg",
}


# ---------------------------------------------------------------------------
# Haupt-Funktion: Videos abrufen
# ---------------------------------------------------------------------------

async def fetch_youtube_news() -> list[dict]:
    """
    Lokale/Amateur YouTube-Kanäle abrufen und aktuelle Videos (< 12h alt) zurückgeben.
    Fokus auf Augenzeugen- und Blaulicht-Material, NICHT professionelle Nachrichten.
    """
    videos = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=5)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for name, channel_id in YOUTUBE_CHANNELS.items():
            tasks.append(_fetch_channel_feed(session, name, channel_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                log.debug("YouTube Feed Fehler: %s", result)
                continue
            if result:
                videos.extend(result)

    # Nach Datum sortieren (neueste zuerst)
    videos.sort(key=lambda v: v.get("published", ""), reverse=True)
    log.info("YouTube Amateur: %d aktuelle Videos gefunden", len(videos))
    return videos


async def _fetch_channel_feed(
    session: aiohttp.ClientSession, name: str, channel_id: str
) -> list[dict]:
    """YouTube RSS-Feed eines Kanals abrufen und Videos der letzten 12h extrahieren."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    videos = []

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "NewsWatchdog/1.0"},
        ) as resp:
            if resp.status != 200:
                log.debug("YouTube %s HTTP %d", name, resp.status)
                return videos
            xml = await resp.text()
    except Exception as e:
        log.debug("YouTube %s Fehler: %s", name, e)
        return videos

    # Video-IDs, Titel und Zeitstempel extrahieren
    video_ids = re.findall(r"<yt:videoId>([^<]+)</yt:videoId>", xml)
    titles = re.findall(r"<media:title>([^<]+)</media:title>", xml)
    published_dates = re.findall(r"<published>([^<]+)</published>", xml)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=12)

    for i, (vid_id, title) in enumerate(zip(video_ids, titles)):
        pub_str = published_dates[i + 1] if i + 1 < len(published_dates) else ""
        if pub_str:
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_dt < cutoff:
                    continue
            except Exception:
                continue
        else:
            continue

        title = _decode_html(title)

        # Blaulicht-Keywords filtern — nur relevante Videos
        if not _is_blaulicht_relevant(title):
            continue

        videos.append({
            "video_id": vid_id,
            "title": title,
            "channel": name,
            "published": pub_str,
            "embed_url": f"https://www.youtube.com/embed/{vid_id}",
            "watch_url": f"https://www.youtube.com/watch?v={vid_id}",
            "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
            "keywords": _extract_keywords(title),
        })

    return videos


def _is_blaulicht_relevant(title: str) -> bool:
    """Prüft ob ein Video-Titel Blaulicht/Einsatz-relevant ist."""
    text = title.lower()
    blaulicht_keywords = [
        "polizei", "feuerwehr", "brand", "feuer", "unfall", "rettung",
        "einsatz", "notfall", "explosion", "messer", "schuss", "überfall",
        "demo", "protest", "razzia", "festnahme", "leiche", "tot",
        "hochwasser", "unwetter", "sturm", "evakuierung", "alarm",
        "großeinsatz", "sperrung", "crash", "blaulicht", "sirene",
        "rettungshubschrauber", "notruf", "tatort", "absperrung",
        "durchsuchung", "fahndung", "vermisst", "flucht", "verfolgung",
    ]
    return any(kw in text for kw in blaulicht_keywords)


# ---------------------------------------------------------------------------
# Video-Event-Matching
# ---------------------------------------------------------------------------

def match_videos_to_events(
    videos: list[dict], events: list[dict], threshold: float = 0.4
) -> dict[str, list[str]]:
    """
    Matcht YouTube-Videos mit bestehenden Events.
    Strenge Kriterien: Ortsname UND Themen-Übereinstimmung.
    """
    matches: dict[str, list[str]] = {}

    for event in events:
        event_keywords = _extract_keywords(event.get("titel", ""))
        event_location = _extract_location_keywords(event.get("titel", ""))

        if not event_keywords or len(event_keywords) < 2:
            continue

        best_match = None
        best_score = 0.0

        for video in videos:
            vid_keywords = video.get("keywords", set())

            common = event_keywords & vid_keywords
            if len(common) < 2:
                continue

            score = len(common) / min(len(event_keywords), len(vid_keywords))

            # Location-Match: Ortsname muss übereinstimmen für sicheres Match
            location_match = bool(event_location and event_location & vid_keywords)
            if location_match:
                score += 0.4

            # Ohne Location-Match brauchen wir sehr hohen Keyword-Score
            if not location_match and score < 0.6:
                continue

            if score >= threshold and score > best_score:
                best_score = score
                best_match = video

        if best_match:
            eid = event.get("id", "")
            matches[eid] = [best_match["embed_url"], best_match["thumbnail"]]

    return matches


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

STOPWORDS = {
    "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in", "im",
    "von", "vom", "zu", "zum", "zur", "bei", "mit", "nach", "vor", "auf",
    "an", "für", "aus", "über", "unter", "zwischen", "den", "dem", "des",
    "sich", "ist", "hat", "wird", "sind", "war", "haben", "wurde", "werden",
    "nicht", "noch", "auch", "nur", "als", "wie", "was", "wer", "kann",
    "pol", "ots", "dwd", "amtliche", "warnung", "zeugen", "gesucht",
    "polizei", "sucht",
}


def _extract_keywords(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[A-ZÄÖÜa-zäöüß]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _extract_location_keywords(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"\b([A-ZÄÖÜ][a-zäöüß]{2,})\b", text)
    return {w.lower() for w in words if w.lower() not in STOPWORDS}


def _decode_html(text: str) -> str:
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text
