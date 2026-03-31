#!/usr/bin/env python3
"""
Scraper: BBK NINA Warnungen
Quelle: https://nina.api.proxy.bund.dev/api31/warnings/mapData.json

NINA (Notfall-Informations- und Nachrichten-App) des Bundesamtes für
Bevölkerungsschutz und Katastrophenhilfe (BBK).

Liefert: Unwetter, Hochwasser, Katastrophenwarnungen — bereits geo-referenziert.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import ssl
import aiohttp
import certifi

log = logging.getLogger("watchdog.nina")

# ---------------------------------------------------------------------------
# NINA API Endpoints
# ---------------------------------------------------------------------------

NINA_MAP_DATA = "https://nina.api.proxy.bund.dev/api31/warnings/mapData.json"
NINA_WARNING_DETAIL = "https://nina.api.proxy.bund.dev/api31/warnings/{warning_id}.json"

# ---------------------------------------------------------------------------
# Typ-Mapping
# ---------------------------------------------------------------------------

NINA_TYPE_MAP = {
    "ALERT": "unwetter",
    "UPDATE": "unwetter",
    "CANCEL": "sonstiges",
    # Severity-basiert
    "Extreme": "unwetter",
    "Severe": "unwetter",
    "Moderate": "unwetter",
    "Minor": "sonstiges",
}

NINA_SEVERITY_SCORE = {
    "Extreme": 9.0,
    "Severe": 7.0,
    "Moderate": 5.0,
    "Minor": 3.0,
}


# ---------------------------------------------------------------------------
# Haupt-Scraper
# ---------------------------------------------------------------------------

async def scrape_nina_warnings() -> list[dict]:
    """
    NINA mapData.json abrufen und Warnungen als Events zurückgeben.
    Die mapData enthält eine Liste von Warnungen mit Geo-Koordinaten.
    """
    events = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
        # Schritt 1: Übersichtsdaten holen
        try:
            async with session.get(
                NINA_MAP_DATA,
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": "NewsWatchdog/1.0"}
            ) as resp:
                if resp.status != 200:
                    log.warning("NINA mapData HTTP %d", resp.status)
                    return events
                data = await resp.json()
        except Exception as e:
            log.error("NINA mapData Fehler: %s", e)
            return events

        if not isinstance(data, list):
            log.warning("NINA mapData unerwartetes Format: %s", type(data))
            return events

        log.info("NINA: %d Warnungen in mapData", len(data))

        # Schritt 2: Jede Warnung verarbeiten
        for warning in data:
            try:
                event = _parse_nina_warning(warning)
                if event:
                    events.append(event)
            except Exception as e:
                log.debug("NINA Warnung Parse-Fehler: %s", e)
                continue

        # Schritt 3: Für die wichtigsten Warnungen Details holen
        severe_events = [e for e in events if e["scoop_score"] >= 6.0]
        for event in severe_events[:10]:  # Max 10 Detail-Requests
            warning_id = event.get("_nina_id")
            if not warning_id:
                continue
            try:
                detail = await _fetch_warning_detail(session, warning_id)
                if detail:
                    event["zusammenfassung"] = detail.get("description", event["zusammenfassung"])
                    if detail.get("instruction"):
                        event["zusammenfassung"] += f" Verhaltenshinweis: {detail['instruction']}"
            except Exception as e:
                log.debug("NINA Detail-Fehler für %s: %s", warning_id, e)

        # _nina_id entfernen (internes Feld)
        for event in events:
            event.pop("_nina_id", None)

    log.info("NINA gesamt: %d Events extrahiert", len(events))
    return events


# ---------------------------------------------------------------------------
# Einzelne Warnung parsen
# ---------------------------------------------------------------------------

def _parse_nina_warning(warning: dict) -> Optional[dict]:
    """
    Eine einzelne NINA-Warnung in ein Event-Dict umwandeln.

    NINA mapData Struktur (vereinfacht):
    {
        "id": "lhp.HOCHWASSERZENTRALEN.DE...",
        "version": 1,
        "startDate": "2026-03-13T10:00:00+01:00",
        "severity": "Severe",
        "type": "ALERT",
        "title": "Hochwasserwarnung Elbe",
        "transKeys": {...},
        "i18nTitle": {"de": "Hochwasserwarnung"},
        "area": {"type": "Point", "coordinates": [lon, lat]}  // oder Polygon
    }
    """
    warning_id = warning.get("id", "")
    if not warning_id:
        return None

    # Titel extrahieren
    titel = ""
    i18n = warning.get("i18nTitle", {})
    if isinstance(i18n, dict):
        titel = i18n.get("de", "") or i18n.get("en", "")
    if not titel:
        titel = warning.get("title", "NINA Warnung")

    # Zeitpunkt
    start_date = warning.get("startDate", "")
    if not start_date:
        start_date = datetime.now(timezone.utc).isoformat()

    # Severity und Score
    severity = warning.get("severity", "Minor")
    scoop_score = NINA_SEVERITY_SCORE.get(severity, 3.0)

    # Geo-Koordinaten aus area extrahieren
    area = warning.get("area", {})
    coords = _extract_coordinates(area)
    if not coords:
        return None

    lat, lon = coords

    # Typ bestimmen
    warning_type = warning.get("type", "ALERT")
    typ = _classify_nina_type(titel, severity, warning_id)

    # Quellen
    quelle = "BBK NINA"
    if "hochwasser" in warning_id.lower() or "hochwasser" in titel.lower():
        quelle = "Hochwasserzentrale"
        typ = "unwetter"
        scoop_score = max(scoop_score, 6.0)
    elif "dwd" in warning_id.lower():
        quelle = "Deutscher Wetterdienst"
        typ = "unwetter"
    elif "polizei" in warning_id.lower() or "bpol" in warning_id.lower():
        quelle = "Polizei (NINA)"
        typ = "polizei"
    elif "lhp" in warning_id.lower():
        quelle = "Hochwasserzentrale"
        typ = "unwetter"

    # Zusammenfassung
    zusammenfassung = f"{titel} — Warnstufe: {severity}"

    return {
        "titel": _clean_title(titel),
        "typ": typ,
        "lat": lat,
        "lon": lon,
        "zeitpunkt": start_date,
        "scoop_score": scoop_score,
        "quellen": [quelle],
        "zusammenfassung": zusammenfassung,
        "media_urls": [],
        "link": f"https://warnung.bund.de/meldung/{warning_id}",
        "quelle_typ": "nina",
        "_nina_id": warning_id,
    }


# ---------------------------------------------------------------------------
# Detail-Endpoint
# ---------------------------------------------------------------------------

async def _fetch_warning_detail(session: aiohttp.ClientSession, warning_id: str) -> Optional[dict]:
    """Detail-Informationen zu einer Warnung holen."""
    url = NINA_WARNING_DETAIL.format(warning_id=warning_id)
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "NewsWatchdog/1.0"}
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

            # Info-Block extrahieren
            info = data.get("info", [{}])
            if isinstance(info, list) and info:
                info_block = info[0]
            elif isinstance(info, dict):
                info_block = info
            else:
                return None

            return {
                "description": _clean_html(info_block.get("description", "")),
                "instruction": _clean_html(info_block.get("instruction", "")),
                "headline": info_block.get("headline", ""),
            }
    except Exception as e:
        log.debug("NINA Detail Fetch Fehler: %s", e)
        return None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _extract_coordinates(area: dict) -> Optional[tuple[float, float]]:
    """
    Geo-Koordinaten aus NINA area-Objekt extrahieren.
    Kann Point oder Polygon sein.
    """
    if not area:
        return None

    geo_type = area.get("type", "")

    if geo_type == "Point":
        coords = area.get("coordinates", [])
        if len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            return (lat, lon)

    elif geo_type in ("Polygon", "MultiPolygon"):
        # Centroid des ersten Polygons berechnen
        coordinates = area.get("coordinates", [])
        if not coordinates:
            return None

        # Bei MultiPolygon: erstes Polygon nehmen
        if geo_type == "MultiPolygon" and coordinates:
            coordinates = coordinates[0]

        if coordinates and coordinates[0]:
            ring = coordinates[0]
            if isinstance(ring[0], (list, tuple)):
                lats = [p[1] for p in ring]
                lons = [p[0] for p in ring]
                return (sum(lats) / len(lats), sum(lons) / len(lons))

    # Fallback: coordinates direkt
    coords = area.get("coordinates", [])
    if isinstance(coords, list) and len(coords) >= 2:
        if isinstance(coords[0], (int, float)):
            return (coords[1], coords[0])

    return None


def _classify_nina_type(titel: str, severity: str, warning_id: str) -> str:
    """NINA-Warnung einem Event-Typ zuordnen."""
    text = f"{titel} {warning_id}".lower()

    if any(kw in text for kw in ["hochwasser", "pegel", "überschwemmung", "flut"]):
        return "unwetter"
    if any(kw in text for kw in ["unwetter", "gewitter", "sturm", "hagel", "orkan", "dwd"]):
        return "unwetter"
    if any(kw in text for kw in ["brand", "feuer", "waldbrand"]):
        return "feuer"
    if any(kw in text for kw in ["bombe", "kampfmittel", "evakuierung", "blindgänger"]):
        return "polizei"
    if any(kw in text for kw in ["polizei", "fahndung", "gefahr"]):
        return "polizei"

    return "unwetter"  # Default für NINA-Warnungen


def _clean_title(title: str) -> str:
    """Titel bereinigen."""
    # Doppelte Leerzeichen entfernen
    title = re.sub(r"\s+", " ", title).strip()
    # Zu lange Titel kürzen
    if len(title) > 200:
        title = title[:197] + "..."
    return title


def _clean_html(text: str) -> str:
    """HTML-Tags und überschüssige Whitespaces entfernen."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:1000]  # Max 1000 Zeichen
