#!/usr/bin/env python3
"""
Scraper: Telegram öffentliche Kanäle via Web-Preview
Kein API-Key oder Telethon nötig — nutzt die öffentliche Web-Ansicht t.me/s/CHANNEL.

Parsed HTML von Telegram Web-Previews für neue Posts aus Blaulicht-Kanälen.
"""

import asyncio
import hashlib
import logging
import random
import re
import ssl
import time
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Optional

import aiohttp
import certifi

from scrapers.polizei_rss import extract_street_address, geocode_address

log = logging.getLogger("watchdog.telegram")

# ---------------------------------------------------------------------------
# Öffentliche Telegram-Kanäle (Blaulicht / regionale News)
# ---------------------------------------------------------------------------

TELEGRAM_CHANNELS = [
    # Blaulicht-Kanäle — Großstädte
    {"name": "blaulichtberlin", "city": "Berlin", "label": "Blaulicht Berlin"},
    {"name": "blaulichthamburg", "city": "Hamburg", "label": "Blaulicht Hamburg"},
    {"name": "blaulichtmuenchen", "city": "München", "label": "Blaulicht München"},
    {"name": "polizeinews_de", "city": None, "label": "Polizei News DE"},
    # Blaulicht-Kanäle — Regional
    {"name": "blaulichtreport_koeln", "city": "Köln", "label": "Blaulicht Köln"},
    {"name": "blaulichtreport_nrw", "city": None, "label": "Blaulicht NRW"},
    {"name": "blaulicht_frankfurt", "city": "Frankfurt", "label": "Blaulicht Frankfurt"},
    {"name": "blaulicht_stuttgart", "city": "Stuttgart", "label": "Blaulicht Stuttgart"},
    {"name": "blaulicht_sachsen", "city": None, "label": "Blaulicht Sachsen"},
    {"name": "blaulicht_niedersachsen", "city": None, "label": "Blaulicht Niedersachsen"},
    # Erweitert — weitere Großstädte
    {"name": "blaulicht_duesseldorf", "city": "Düsseldorf", "label": "Blaulicht Düsseldorf"},
    {"name": "blaulicht_dortmund", "city": "Dortmund", "label": "Blaulicht Dortmund"},
    {"name": "blaulicht_essen", "city": "Essen", "label": "Blaulicht Essen"},
    {"name": "blaulicht_hannover", "city": "Hannover", "label": "Blaulicht Hannover"},
    {"name": "blaulicht_leipzig", "city": "Leipzig", "label": "Blaulicht Leipzig"},
    {"name": "blaulicht_dresden", "city": "Dresden", "label": "Blaulicht Dresden"},
    {"name": "blaulicht_bremen", "city": "Bremen", "label": "Blaulicht Bremen"},
    {"name": "blaulicht_nuernberg", "city": "Nürnberg", "label": "Blaulicht Nürnberg"},
    {"name": "blaulicht_mannheim", "city": "Mannheim", "label": "Blaulicht Mannheim"},
    {"name": "blaulicht_karlsruhe", "city": "Karlsruhe", "label": "Blaulicht Karlsruhe"},
    {"name": "blaulichtrheinmain", "city": "Frankfurt", "label": "Blaulicht Rhein-Main"},
    {"name": "blaulicht_ruhrgebiet", "city": "Essen", "label": "Blaulicht Ruhrgebiet"},
    {"name": "blaulicht_bayern", "city": None, "label": "Blaulicht Bayern"},
    {"name": "blaulicht_bw", "city": None, "label": "Blaulicht Baden-Württemberg"},
    {"name": "blaulicht_hessen", "city": None, "label": "Blaulicht Hessen"},
    {"name": "blaulicht_sh", "city": None, "label": "Blaulicht Schleswig-Holstein"},
    # Fahndung + Einsatz
    {"name": "fahndung_aktuell", "city": None, "label": "Fahndung Aktuell"},
    {"name": "feuerwehr_de", "city": None, "label": "Feuerwehr DE"},
    {"name": "grosseinsatz_de", "city": None, "label": "Großeinsatz DE"},
    {"name": "blaulichtreporter", "city": None, "label": "Blaulicht Reporter"},
    {"name": "tatort_deutschland", "city": None, "label": "Tatort Deutschland"},
]

# ---------------------------------------------------------------------------
# Bekannte Städte mit Koordinaten
# ---------------------------------------------------------------------------

CITY_COORDS = {
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "münchen": (48.1351, 11.5820),
    "köln": (50.9375, 6.9603),
    "frankfurt": (50.1109, 8.6821),
    "stuttgart": (48.7758, 9.1829),
    "düsseldorf": (51.2277, 6.7735),
    "leipzig": (51.3397, 12.3731),
    "dresden": (51.0504, 13.7373),
    "hannover": (52.3759, 9.7320),
    "dortmund": (51.5136, 7.4653),
    "essen": (51.4556, 7.0116),
    "bremen": (53.0793, 8.8017),
    "nürnberg": (49.4521, 11.0767),
    "bonn": (50.7374, 7.0982),
    "münster": (51.9607, 7.6261),
    "karlsruhe": (49.0069, 8.4037),
    "mannheim": (49.4875, 8.4660),
    "augsburg": (48.3705, 10.8978),
    "mainz": (49.9929, 8.2473),
    "freiburg": (47.9990, 7.8421),
    "kiel": (54.3233, 10.1228),
    "rostock": (54.0924, 12.0991),
    "magdeburg": (52.1205, 11.6276),
    "erfurt": (50.9848, 11.0299),
    "potsdam": (52.3906, 13.0645),
    "chemnitz": (50.8278, 12.9214),
    "saarbrücken": (49.2402, 6.9969),
}

# ---------------------------------------------------------------------------
# Breaking-Keywords (gleich wie Twitter, für Konsistenz)
# ---------------------------------------------------------------------------

BREAKING_KEYWORDS = {
    "tot": 3.0, "tödlich": 3.0, "getötet": 3.0, "leiche": 2.5,
    "mord": 3.0, "mordkommission": 2.5,
    "schüsse": 2.5, "schusswaffe": 2.5, "messer": 2.0, "messerangriff": 2.5,
    "großeinsatz": 2.0, "großlage": 2.5, "sek": 2.0,
    "terror": 3.0, "bombe": 2.5, "sprengstoff": 2.5, "evakuierung": 2.0,
    "geiselnahme": 3.0, "explosion": 2.5,
    "schwerverletzt": 1.5, "lebensgefährlich": 2.0,
    "hubschrauber": 1.5, "vollsperrung": 1.5,
    "großbrand": 2.0, "dachstuhlbrand": 1.0,
    "überfall": 1.5, "fahndung": 1.5,
    "unfall": 1.0, "brand": 1.0, "feuer": 1.0,
    "polizei": 0.5, "feuerwehr": 0.5,
    "demo": 0.5, "vermisst": 1.0,
}

# ---------------------------------------------------------------------------
# Duplikat-Cache
# ---------------------------------------------------------------------------

_recent_posts: list[str] = []
MAX_RECENT = 500


def _is_duplicate(text: str, threshold: float = 0.80) -> bool:
    text_clean = text.lower().strip()[:200]
    for existing in _recent_posts:
        if SequenceMatcher(None, text_clean, existing).ratio() >= threshold:
            return True
    return False


def _register_post(text: str):
    global _recent_posts
    _recent_posts.append(text.lower().strip()[:200])
    if len(_recent_posts) > MAX_RECENT:
        _recent_posts = _recent_posts[-MAX_RECENT:]


# ---------------------------------------------------------------------------
# Telegram Web-Preview HTML parsen
# ---------------------------------------------------------------------------

def _parse_telegram_html(html: str, channel_info: dict) -> list[dict]:
    """
    Telegram Web-Preview HTML parsen und Posts extrahieren.

    t.me/s/CHANNEL liefert HTML mit <div class="tgme_widget_message_wrap">
    Jeder Post hat:
      - .tgme_widget_message_text: Text-Inhalt
      - .tgme_widget_message_date > time[datetime]: Zeitstempel
      - .tgme_widget_message_photo_wrap: Bild-URL (als background-image)
      - data-post="channel/123": Post-ID für den Link
    """
    posts = []

    # Alle Message-Blöcke finden
    message_pattern = re.compile(
        r'<div[^>]+class="[^"]*tgme_widget_message_wrap[^"]*"[^>]*>'
        r'(.*?)</div>\s*</div>\s*</div>\s*</div>',
        re.DOTALL,
    )

    # Alternativ: einfacheres Pattern für data-post Blöcke
    post_blocks = re.findall(
        r'data-post="([^"]+)"(.*?)(?=data-post="|$)',
        html,
        re.DOTALL,
    )

    for post_id, block in post_blocks:
        # Text extrahieren
        text_match = re.search(
            r'<div[^>]+class="[^"]*tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            block,
            re.DOTALL,
        )
        if not text_match:
            continue

        raw_text = text_match.group(1)
        text = _clean_html(raw_text)

        if not text or len(text) < 20:
            continue

        # Zeitstempel extrahieren
        time_match = re.search(r'<time[^>]+datetime="([^"]+)"', block)
        zeitpunkt = ""
        if time_match:
            zeitpunkt = time_match.group(1)
        if not zeitpunkt:
            zeitpunkt = datetime.now(timezone.utc).isoformat()

        # Bilder extrahieren (background-image in style)
        media_urls = []
        img_matches = re.findall(
            r"background-image:url\('([^']+)'\)",
            block,
        )
        media_urls.extend(img_matches)

        # Direkte <img> Tags
        img_tag_matches = re.findall(
            r'<img[^>]+src="([^"]+)"',
            block,
        )
        for img_url in img_tag_matches:
            if "emoji" not in img_url and "avatar" not in img_url:
                media_urls.append(img_url)

        # Video-Thumbnails
        video_matches = re.findall(
            r'<video[^>]+src="([^"]+)"',
            block,
        )
        media_urls.extend(video_matches)

        # Link zum Post
        channel_name = post_id.split("/")[0] if "/" in post_id else channel_info["name"]
        post_number = post_id.split("/")[-1] if "/" in post_id else post_id
        link = f"https://t.me/{channel_name}/{post_number}"

        posts.append({
            "text": text,
            "zeitpunkt": zeitpunkt,
            "media_urls": media_urls[:5],
            "link": link,
            "post_id": post_id,
        })

    return posts


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _geocode_from_text(text: str, channel_city: Optional[str]) -> Optional[tuple[float, float]]:
    """Koordinaten aus Post-Text oder Kanal-Stadt extrahieren."""
    text_lower = text.lower()

    # Stadtnamen im Text suchen
    for city, coords in CITY_COORDS.items():
        if city in text_lower:
            return coords

    # Fallback: Stadt des Kanals
    if channel_city:
        return CITY_COORDS.get(channel_city.lower())

    return None


# ---------------------------------------------------------------------------
# Event-Typ und Scoop-Score
# ---------------------------------------------------------------------------

def _classify_type(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["brand", "feuer", "flammen", "rauch", "großbrand"]):
        return "feuer"
    if any(kw in text_lower for kw in ["unfall", "verkehrsunfall", "kollision", "karambolage"]):
        return "unfall"
    if any(kw in text_lower for kw in ["unwetter", "sturm", "hochwasser", "überschwemmung"]):
        return "unwetter"
    return "polizei"


def _compute_scoop_score(text: str) -> float:
    """Scoop-Score via einheitlichem Scoring-Modul. Wird in insert_event mit BILD-Info nachberechnet."""
    import scoring
    return scoring.compute_scoop_score(
        title=text,
        description="",
        bild_overlap_type="none",
        zeitpunkt="",
        media_urls=[],
    )


# ---------------------------------------------------------------------------
# HTML bereinigen
# ---------------------------------------------------------------------------

def _clean_html(text: str) -> str:
    if not text:
        return ""
    # <br> zu Leerzeichen
    clean = re.sub(r"<br\s*/?>", " ", text)
    # Alle Tags entfernen
    clean = re.sub(r"<[^>]+>", "", clean)
    # HTML-Entities
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"').replace("&#39;", "'")
    # Whitespace normalisieren
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Haupt-Scraper-Funktion
# ---------------------------------------------------------------------------

async def scrape_telegram() -> list[dict]:
    """
    Alle konfigurierten Telegram-Kanäle via Web-Preview abrufen.
    Gibt eine Liste von Event-Dicts im Standard-Format zurück.
    """
    events = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=3)

    async with aiohttp.ClientSession(connector=connector) as session:
        for channel in TELEGRAM_CHANNELS:
            channel_name = channel["name"]
            channel_city = channel.get("city")
            channel_label = channel.get("label", channel_name)

            url = f"https://t.me/s/{channel_name}"

            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36",
                        "Accept-Language": "de-DE,de;q=0.9",
                    },
                ) as resp:
                    if resp.status != 200:
                        log.warning("Telegram %s HTTP %d", channel_name, resp.status)
                        continue
                    html = await resp.text()
            except Exception as e:
                log.warning("Telegram %s Fehler: %s", channel_name, e)
                continue

            # HTML parsen
            posts = _parse_telegram_html(html, channel)
            log.debug("Telegram @%s: %d Posts gefunden", channel_name, len(posts))

            for post in posts:
                text = post["text"]

                # Duplikat-Check
                if _is_duplicate(text):
                    continue
                _register_post(text)

                # Relevanz-Filter
                text_lower = text.lower()
                has_keyword = any(kw in text_lower for kw in BREAKING_KEYWORDS)
                if not has_keyword:
                    continue

                # Zeitpunkt: nur letzte 24h
                zeitpunkt = post["zeitpunkt"]
                try:
                    zp = zeitpunkt.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(zp)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    if age_h > 24:
                        continue
                except Exception:
                    pass

                # Geocoding — mit Straßen-Präzision
                coords = None
                is_fallback = True
                street = extract_street_address(text)
                if street and channel_city:
                    coords = await geocode_address(street, channel_city, session)
                    if coords:
                        is_fallback = False
                if not coords:
                    coords = _geocode_from_text(text, channel_city)
                if not coords:
                    continue

                lat, lon = coords
                if is_fallback:
                    lat += random.uniform(-0.002, 0.002)
                    lon += random.uniform(-0.002, 0.002)

                # Event zusammenbauen
                event = {
                    "titel": text[:200] if len(text) > 200 else text,
                    "typ": _classify_type(text),
                    "lat": lat,
                    "lon": lon,
                    "zeitpunkt": zeitpunkt,
                    "scoop_score": _compute_scoop_score(text),
                    "quellen": [f"Telegram/{channel_label}"],
                    "zusammenfassung": text[:500],
                    "media_urls": post.get("media_urls", []),
                    "link": post.get("link", ""),
                    "quelle_typ": "telegram",
                }
                events.append(event)

            # Pause zwischen Kanälen (Telegram Rate-Limiting vermeiden)
            await asyncio.sleep(1.5)

    log.info("Telegram gesamt: %d Events extrahiert", len(events))
    return events
