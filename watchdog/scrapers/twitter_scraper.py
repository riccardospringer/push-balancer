#!/usr/bin/env python3
"""
Scraper: X/Twitter Monitoring via öffentliche Nitter-Instanzen + RSS-Bridges
Kein API-Key nötig — nutzt öffentliche RSS-Feeds von Nitter-Frontends.

Monitort Blaulicht-Accounts und filtert nach Breaking-News-Keywords.
Fallback-Kette: Nitter RSS → rss.app → Direkt-Scraping
"""

import asyncio
import hashlib
import logging
import random
import re
import ssl
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

import aiohttp
import certifi
import feedparser

from scrapers.polizei_rss import extract_street_address, geocode_address

log = logging.getLogger("watchdog.twitter")

# ---------------------------------------------------------------------------
# Nitter-Instanzen (Fallback-Kette, erste verfügbare wird genutzt)
# ---------------------------------------------------------------------------

NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
    "https://nitter.1d4.us",
    "https://nitter.cz",
    "https://nitter.hostux.net",
    "https://nitter.fdn.fr",
    "https://nitter.net",
    "https://nitter.unixfox.eu",
    "https://nitter.rawbit.ninja",
]

# ---------------------------------------------------------------------------
# Monitored Accounts — Blaulicht-Organisationen
# ---------------------------------------------------------------------------

MONITORED_ACCOUNTS = [
    # Polizei — Großstädte (Top 20)
    "PolizeiBerlin",
    "PolizeiHamburg",
    "PolizeiMuenchen",
    "Polizei_Ffm",
    "polaborka",             # Polizei Köln
    "PolizeiDuworeldorf",    # Polizei Düsseldorf
    "PP_Stuttgart",          # Polizei Stuttgart
    "Polizei_H",             # Polizei Hannover
    "PolizeiBremen",
    "PolizeiNuworernberg",   # Polizei Nürnberg
    "polaboreidortmund",     # Polizei Dortmund
    "Polizei_E",             # Polizei Essen
    "PolizeiLeipzig",
    "PolizeiSachsen_DD",     # Polizei Dresden
    "PolizeiDuisburg",
    "PolizeiBochum",
    "PolizeiWuppertal",
    "PolizeiBielefeld",
    "PolizeiBonn",
    "PolizeiMuenster",
    # Polizei — Landespolizeien (alle 16 Bundesländer)
    "PolizeiSachsen",
    "Polizei_nrw",
    "PolizeiNI",
    "Polizei_BY",            # Polizei Bayern
    "PolizeiBW",             # Polizei Baden-Württemberg
    "PolizeiHessen",
    "PolizeiRP",             # Polizei Rheinland-Pfalz
    "PolizeiSH",             # Polizei Schleswig-Holstein
    "PolizeiBB",             # Polizei Brandenburg
    "PolizeiST",             # Polizei Sachsen-Anhalt
    "PolizeiTH",             # Polizei Thüringen
    "PolizeiMV",             # Polizei Mecklenburg-Vorpommern
    "PolizeiSaarland",
    # Polizei — Weitere Großstädte
    "PolizeiAachen",
    "PolizeiKarlsruhe",
    "PolizeiMannheim",
    "PolizeiGelsenkirchen",
    "PolizeiOberhausen",
    "PolizeiKrefeld",
    "PolizeiHagen",
    "PolizeiMainz",
    "PolizeiKiel",
    "PolizeiRostock",
    "PolizeiMagdeburg",
    "PolizeiErfurt",
    "PolizeiPotsdam",
    "PolizeiChemnitz",
    "PolizeiWiesbaden",
    "PolizeiFreiburg",
    # Feuerwehr — Großstädte
    "BerlinerFeuworehr",     # offizieller Handle
    "FeuerwehrHH",
    "FeuerwehrMuc",          # Feuerwehr München
    "fw_koeln",              # Feuerwehr Köln
    "FeuerwehrS",            # Feuerwehr Stuttgart
    "FeuerwehrFfm",          # Feuerwehr Frankfurt
    "FeuerwehrDO",           # Feuerwehr Dortmund
    "FeuerwehrE",            # Feuerwehr Essen
    "FW_Duesseldorf",
    "FW_Bremen",
    "FW_Hannover",
    "FW_Leipzig",
    "FW_Dresden",
    "FW_Nuernberg",
    # Bundespolizei + BKA + LKA
    "BPol_Nord",
    "BPol_Sued",
    "baborka_de",            # BKA
    "LKA_Bayern",
    "LKA_NRW",
    "LKA_NI",                # LKA Niedersachsen
    "LKA_BW",
    "LKA_Hessen",
    "LKA_Sachsen",
    # Blaulicht-Reporter / Crime-Accounts
    "Abortagent",
    "BlaulichtNews",
    "blaboraulicht_de",
    "BreakingPolizei",
    "EinsatzHamburg",
    "EinsatzBerlin",
    "FahndungDE",
]

# ---------------------------------------------------------------------------
# Breaking-News Keywords mit Score-Boost
# ---------------------------------------------------------------------------

BREAKING_KEYWORDS = {
    # Maximal-Relevanz
    "tot": 3.0, "tödlich": 3.0, "getötet": 3.0, "leiche": 2.5,
    "mord": 3.0, "mordkommission": 2.5,
    "schüsse": 2.5, "schusswaffe": 2.5, "messer": 2.0, "messerangriff": 2.5,
    "großeinsatz": 2.0, "großlage": 2.5, "sek": 2.0,
    "terror": 3.0, "bombe": 2.5, "sprengstoff": 2.5, "evakuierung": 2.0,
    "geiselnahme": 3.0, "amoklauf": 3.0,
    "explosion": 2.5,
    # Hohe Relevanz
    "schwerverletzt": 1.5, "lebensgefährlich": 2.0, "reanimation": 1.5,
    "hubschrauber": 1.5, "rettungshubschrauber": 1.5,
    "vollsperrung": 1.5, "massenkarambolage": 2.0,
    "großbrand": 2.0, "dachstuhlbrand": 1.0,
    "überfall": 1.5, "bankraub": 2.0, "fahndung": 1.5,
    # Mittlere Relevanz
    "unfall": 1.0, "brand": 1.0, "feuer": 1.0,
    "polizei": 0.5, "feuerwehr": 0.5,
    "demo": 0.5, "demonstration": 0.5, "protest": 0.5,
    "vermisst": 1.0,
}

# ---------------------------------------------------------------------------
# Account → Stadt-Mapping für Geocoding-Fallback
# ---------------------------------------------------------------------------

ACCOUNT_CITY = {
    # Polizei Großstädte
    "polizeiberlin": "Berlin",
    "polizeihamburg": "Hamburg",
    "polizeimuenchen": "München",
    "polizei_ffm": "Frankfurt am Main",
    "polaborka": "Köln",
    "polizeiduworeldorf": "Düsseldorf",
    "pp_stuttgart": "Stuttgart",
    "polizei_h": "Hannover",
    "polizeibremen": "Bremen",
    "polizeinuworernberg": "Nürnberg",
    "polaboreidortmund": "Dortmund",
    "polizei_e": "Essen",
    "polizeileipzig": "Leipzig",
    "polizeisachsen_dd": "Dresden",
    # Landespolizeien
    "polizeisachsen": "Dresden",
    "polizei_nrw": "Düsseldorf",
    "polizeini": "Hannover",
    "polizei_by": "München",
    "polizeibw": "Stuttgart",
    "polizeihessen": "Frankfurt am Main",
    "polizeirp": "Mainz",
    "polizeish": "Kiel",
    "polizeibb": "Potsdam",
    "polizeist": "Magdeburg",
    "polizeith": "Erfurt",
    "polizeimv": "Rostock",
    "polizeisaarland": "Saarbrücken",
    # Feuerwehr
    "berlinerfeuworehr": "Berlin",
    "feuerwehrhh": "Hamburg",
    "feuerwehrmuc": "München",
    "fw_koeln": "Köln",
    "feuerwehrs": "Stuttgart",
    "feuerwehrffm": "Frankfurt am Main",
    # Bundespolizei + BKA
    "bpol_nord": "Hamburg",
    "bpol_sued": "München",
    "baborka_de": "Berlin",
    # LKA
    "lka_bayern": "München",
    "lka_nrw": "Düsseldorf",
    "lka_ni": "Hannover",
    # Weitere Großstädte
    "polizeiaachen": "Aachen",
    "polizeikarlsruhe": "Karlsruhe",
    "polizeimannheim": "Mannheim",
    "polizeigelsenkirchen": "Gelsenkirchen",
    "polizeioberhausen": "Oberhausen",
    "polizeikrefeld": "Krefeld",
    "polizeihagen": "Hagen",
    "polizeimainz": "Mainz",
    "polizeikiel": "Kiel",
    "polizeirostock": "Rostock",
    "polizeimagdeburg": "Magdeburg",
    "polizeierfurt": "Erfurt",
    "polizeipotsdam": "Potsdam",
    "polizeichemnitz": "Chemnitz",
    "polizeiwiesbaden": "Wiesbaden",
    "polizeifreiburg": "Freiburg",
    "polizeiduisburg": "Duisburg",
    "polizeibochum": "Bochum",
    "polizeiwuppertal": "Wuppertal",
    "polizeibielefeld": "Bielefeld",
    "polizeibonn": "Bonn",
    "polizeimuenster": "Münster",
    # Erweiterte Feuerwehr
    "feuerwehrdo": "Dortmund",
    "feuerwehre": "Essen",
    "fw_duesseldorf": "Düsseldorf",
    "fw_bremen": "Bremen",
    "fw_hannover": "Hannover",
    "fw_leipzig": "Leipzig",
    "fw_dresden": "Dresden",
    "fw_nuernberg": "Nürnberg",
    # Erweiterte Bundespolizei / LKA
    "lka_bw": "Stuttgart",
    "lka_hessen": "Frankfurt am Main",
    "lka_sachsen": "Dresden",
    # Spezial / Blaulicht-Reporter
    "abortagent": "Berlin",
    "blaulichtnews": "Berlin",
    "blaboraulicht_de": "Berlin",
    "breakingpolizei": "Berlin",
    "einsatzhamburg": "Hamburg",
    "einsatzberlin": "Berlin",
    "fahndungde": "Berlin",
}

# Bekannte Städte mit Koordinaten (identisch mit polizei_rss.py)
CITY_COORDS = {
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "münchen": (48.1351, 11.5820),
    "köln": (50.9375, 6.9603),
    "frankfurt am main": (50.1109, 8.6821),
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
    "duisburg": (51.4344, 6.7623),
    "bochum": (51.4818, 7.2162),
    "wuppertal": (51.2562, 7.1508),
    "bielefeld": (52.0302, 8.5325),
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
# Duplikat-Cache
# ---------------------------------------------------------------------------

_recent_tweets: list[str] = []
MAX_RECENT = 500


def _is_duplicate(text: str, threshold: float = 0.80) -> bool:
    """Prüfe ob ein ähnlicher Tweet kürzlich verarbeitet wurde."""
    text_clean = text.lower().strip()[:200]
    for existing in _recent_tweets:
        if SequenceMatcher(None, text_clean, existing).ratio() >= threshold:
            return True
    return False


def _register_tweet(text: str):
    """Tweet-Text in den Duplikat-Cache aufnehmen."""
    global _recent_tweets
    _recent_tweets.append(text.lower().strip()[:200])
    if len(_recent_tweets) > MAX_RECENT:
        _recent_tweets = _recent_tweets[-MAX_RECENT:]


# ---------------------------------------------------------------------------
# Geocoding aus Tweet-Text
# ---------------------------------------------------------------------------

def _geocode_from_text(text: str, account: str) -> Optional[tuple[float, float]]:
    """
    Versuche Koordinaten aus dem Tweet-Text zu extrahieren.
    1. Bekannte Stadtnamen im Text suchen
    2. Fallback: Stadt des Accounts
    """
    text_lower = text.lower()

    # Schritt 1: Stadtnamen im Text
    for city, coords in CITY_COORDS.items():
        if city in text_lower:
            return coords

    # Schritt 2: Account-Fallback
    account_key = account.lower()
    city = ACCOUNT_CITY.get(account_key, "")
    if city:
        return CITY_COORDS.get(city.lower())

    return None


# ---------------------------------------------------------------------------
# Scoop-Score berechnen
# ---------------------------------------------------------------------------

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
# Event-Typ klassifizieren
# ---------------------------------------------------------------------------

def _classify_type(text: str) -> str:
    """Event-Typ aus Tweet-Text bestimmen."""
    text_lower = text.lower()

    fire_kw = ["brand", "feuer", "flammen", "rauch", "großbrand", "dachstuhl", "feuerwehr"]
    accident_kw = ["unfall", "verkehrsunfall", "kollision", "zusammenstoß", "karambolage"]
    weather_kw = ["unwetter", "sturm", "hochwasser", "überschwemmung", "hagel"]
    crime_kw = ["messer", "schüsse", "überfall", "raub", "mord", "fahndung", "festnahme"]

    if any(kw in text_lower for kw in fire_kw):
        return "feuer"
    if any(kw in text_lower for kw in accident_kw):
        return "unfall"
    if any(kw in text_lower for kw in weather_kw):
        return "unwetter"
    if any(kw in text_lower for kw in crime_kw):
        return "polizei"

    return "polizei"  # Default für Blaulicht-Accounts


# ---------------------------------------------------------------------------
# Media-URLs aus Nitter-HTML extrahieren
# ---------------------------------------------------------------------------

def _extract_media_urls(html_content: str) -> list[str]:
    """Extrahiere Bild- und Video-URLs aus Nitter-Feed-Einträgen."""
    urls = []

    # Bilder: nitter stellt Bilder als /pic/media/... oder absolut bereit
    img_matches = re.findall(r'src="([^"]+\.(?:jpg|jpeg|png|gif|webp))"', html_content, re.IGNORECASE)
    for img in img_matches:
        if "profile" not in img.lower() and "avatar" not in img.lower():
            urls.append(img)

    # Videos: nitter embedded Videos als <video> oder Link zu Twitter-Video
    video_matches = re.findall(r'src="([^"]+\.(?:mp4|m3u8))"', html_content, re.IGNORECASE)
    urls.extend(video_matches)

    return urls[:5]  # Max 5 Medien pro Tweet


# ---------------------------------------------------------------------------
# Nitter RSS Feed für einen Account abrufen
# ---------------------------------------------------------------------------

async def _fetch_nitter_rss(
    account: str,
    session: aiohttp.ClientSession,
    instances: list[str],
) -> Optional[str]:
    """
    RSS-Feed von Nitter-Instanzen holen. Probiert alle Instanzen durch.
    Gibt den RSS-XML-String zurück oder None.
    """
    for instance in instances:
        url = f"{instance}/{account}/rss"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "NewsWatchdog/1.0"},
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    if "<item>" in content or "<entry>" in content:
                        log.debug("Nitter RSS OK: %s/%s", instance, account)
                        return content
                elif resp.status == 429:
                    log.debug("Nitter Rate-Limit: %s", instance)
                    continue
                else:
                    log.debug("Nitter %s/%s HTTP %d", instance, account, resp.status)
        except Exception as e:
            log.debug("Nitter %s/%s Fehler: %s", instance, account, e)
            continue

    return None


# ---------------------------------------------------------------------------
# Haupt-Scraper-Funktion
# ---------------------------------------------------------------------------

async def scrape_twitter() -> list[dict]:
    """
    Alle konfigurierten Twitter/X-Accounts via Nitter-RSS abrufen.
    Gibt eine Liste von Event-Dicts im Standard-Format zurück.
    """
    events = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=5)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Zuerst: verfügbare Nitter-Instanzen prüfen
        available_instances = await _check_nitter_instances(session)
        if not available_instances:
            log.warning("Keine Nitter-Instanz erreichbar. Twitter-Scraping übersprungen.")
            return events

        log.info("Nitter: %d Instanzen verfügbar", len(available_instances))

        # Accounts parallel abrufen (in Batches um Rate-Limits zu vermeiden)
        batch_size = 4
        for i in range(0, len(MONITORED_ACCOUNTS), batch_size):
            batch = MONITORED_ACCOUNTS[i:i + batch_size]
            tasks = [
                _fetch_account_events(account, session, available_instances)
                for account in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    log.debug("Account-Fehler: %s", result)
                    continue
                if isinstance(result, list):
                    events.extend(result)

            # Kurze Pause zwischen Batches
            if i + batch_size < len(MONITORED_ACCOUNTS):
                await asyncio.sleep(2)

    log.info("Twitter gesamt: %d Events extrahiert", len(events))
    return events


async def _check_nitter_instances(session: aiohttp.ClientSession) -> list[str]:
    """Prüfe welche Nitter-Instanzen erreichbar sind."""
    available = []

    for instance in NITTER_INSTANCES:
        try:
            async with session.get(
                instance,
                timeout=aiohttp.ClientTimeout(total=8),
                allow_redirects=True,
            ) as resp:
                if resp.status < 400:
                    available.append(instance)
                    log.debug("Nitter OK: %s", instance)
        except Exception:
            log.debug("Nitter nicht erreichbar: %s", instance)

    return available


async def _fetch_account_events(
    account: str,
    session: aiohttp.ClientSession,
    instances: list[str],
) -> list[dict]:
    """Events für einen einzelnen Account extrahieren."""
    events = []

    rss_content = await _fetch_nitter_rss(account, session, instances)
    if not rss_content:
        log.debug("Kein RSS-Feed für @%s", account)
        return events

    feed = feedparser.parse(rss_content)
    log.debug("@%s: %d Einträge im Feed", account, len(feed.entries))

    for entry in feed.entries[:10]:  # Max 10 pro Account
        title = entry.get("title", "").strip()
        description = entry.get("description", "").strip()
        link = entry.get("link", "")
        published = entry.get("published", "")

        # Tweet-Text: title oder description (Nitter nutzt beides unterschiedlich)
        tweet_text = title or _clean_html(description)
        if not tweet_text or len(tweet_text) < 15:
            continue

        # Duplikat-Check
        if _is_duplicate(tweet_text):
            continue
        _register_tweet(tweet_text)

        # Relevanz-Filter: mindestens ein Breaking-Keyword muss vorkommen
        text_lower = tweet_text.lower()
        has_keyword = any(kw in text_lower for kw in BREAKING_KEYWORDS)
        if not has_keyword:
            continue

        # Zeitpunkt parsen
        zeitpunkt = _parse_time(published)

        # Nur Events der letzten 24 Stunden
        try:
            dt = datetime.fromisoformat(zeitpunkt.replace("Z", "+00:00"))
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
        street = extract_street_address(tweet_text)
        if street:
            # Account-Stadt als Kontext
            account_city = ACCOUNT_CITY.get(account.lower(), "")
            if account_city:
                coords = await geocode_address(street, account_city, session)
                if coords:
                    is_fallback = False
        if not coords:
            coords = _geocode_from_text(tweet_text, account)
        if not coords:
            continue

        lat, lon = coords
        if is_fallback:
            lat += random.uniform(-0.002, 0.002)
            lon += random.uniform(-0.002, 0.002)

        # Media extrahieren
        media_urls = _extract_media_urls(description)

        # Event zusammenbauen
        event = {
            "titel": _truncate(tweet_text, 200),
            "typ": _classify_type(tweet_text),
            "lat": lat,
            "lon": lon,
            "zeitpunkt": zeitpunkt,
            "scoop_score": _compute_scoop_score(tweet_text),
            "quellen": [f"X/@{account}"],
            "zusammenfassung": _truncate(_clean_html(description), 500),
            "media_urls": media_urls,
            "link": link,
            "quelle_typ": "twitter",
        }
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _parse_time(published: str) -> str:
    """RSS-Zeitstempel in ISO-Format umwandeln."""
    if not published:
        return datetime.now(timezone.utc).isoformat()

    for fmt in [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            dt = datetime.strptime(published, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    return datetime.now(timezone.utc).isoformat()


def _clean_html(text: str) -> str:
    """HTML-Tags entfernen."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _truncate(text: str, max_len: int) -> str:
    """Text auf maximale Länge kürzen."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
