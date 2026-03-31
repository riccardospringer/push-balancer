#!/usr/bin/env python3
"""
Scraper: Deutscher Wetterdienst (DWD) Unwetterwarnungen
Quelle: https://opendata.dwd.de/weather/alerts/cap/ (CAP XML Format)

CAP = Common Alerting Protocol — internationaler Standard für Warnmeldungen.
DWD liefert alle aktiven Unwetterwarnungen als CAP-XML mit Geo-Polygonen.

Alternativ-Endpoint: DWD GeoServer WFS für maschinenlesbare Warnungen.
"""

import asyncio
import hashlib
import logging
import re
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import certifi

log = logging.getLogger("watchdog.dwd")

# ---------------------------------------------------------------------------
# DWD Endpoints
# ---------------------------------------------------------------------------

# CAP-Warnungen als ZIP-Datei (enthält mehrere XML-Dateien)
DWD_CAP_OVERVIEW = "https://opendata.dwd.de/weather/alerts/cap/COMMUNEUNION_DWD_STAT/Z_CAP_C_EDZW_LATEST_PVW_STATUS_PREMIUMDWD_COMMUNEUNION_DE.zip"

# Alternativer JSON-Endpoint (einfacher zu parsen)
DWD_WARNINGS_JSON = "https://www.dwd.de/DWD/warnungen/warnapp/json/warnings.json"

# GeoServer WFS (liefert GeoJSON direkt)
DWD_WFS_URL = (
    "https://maps.dwd.de/geoserver/dwd/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=dwd:Warnungen_Gemeinden_vereinigt"
    "&outputFormat=application/json"
    "&srsName=EPSG:4326"
)

# ---------------------------------------------------------------------------
# Severity-Mapping
# ---------------------------------------------------------------------------

DWD_SEVERITY_MAP = {
    "Minor": {"typ": "unwetter", "score": 3.0, "label": "Wetterwarnung"},
    "Moderate": {"typ": "unwetter", "score": 5.0, "label": "Markante Wetterwarnung"},
    "Severe": {"typ": "unwetter", "score": 7.0, "label": "Unwetterwarnung"},
    "Extreme": {"typ": "unwetter", "score": 9.0, "label": "Extreme Unwetterwarnung"},
}

# Event-Code zu lesbarem Typ
DWD_EVENT_MAP = {
    "THUNDERSTORM": "Gewitter",
    "WIND": "Wind/Sturm",
    "TORNADO": "Tornado",
    "RAIN": "Starkregen",
    "SNOWFALL": "Schneefall",
    "THAW": "Tauwetter",
    "FROST": "Frost",
    "GLAZE": "Glatteis",
    "HEAT": "Hitze",
    "UV": "UV-Warnung",
    "FOG": "Nebel",
    "FLOOD": "Hochwasser",
}

# ---------------------------------------------------------------------------
# Haupt-Scraper: DWD Warnungen via JSON-Endpoint
# ---------------------------------------------------------------------------

async def scrape_dwd_warnings() -> list[dict]:
    """
    DWD-Unwetterwarnungen abrufen und als Events zurückgeben.
    Nutzt primär den JSON-Endpoint (einfacher als CAP-XML).
    Fallback: WFS GeoServer.
    """
    events = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Strategie 1: JSON-Endpoint
        json_events = await _fetch_dwd_json(session)
        if json_events:
            events.extend(json_events)
            log.info("DWD JSON: %d Warnungen", len(json_events))
        else:
            # Fallback: WFS GeoServer
            log.info("DWD JSON nicht verfügbar, versuche WFS...")
            wfs_events = await _fetch_dwd_wfs(session)
            events.extend(wfs_events)
            log.info("DWD WFS: %d Warnungen", len(wfs_events))

    log.info("DWD gesamt: %d Warnungen extrahiert", len(events))
    return events


# ---------------------------------------------------------------------------
# Strategie 1: DWD JSON-Warnungen
# ---------------------------------------------------------------------------

async def _fetch_dwd_json(session: aiohttp.ClientSession) -> list[dict]:
    """
    DWD warnings.json abrufen.
    Format: JavaScript-Callback mit eingebettetem JSON.
    """
    events = []

    try:
        async with session.get(
            DWD_WARNINGS_JSON,
            timeout=aiohttp.ClientTimeout(total=20),
            headers={
                "User-Agent": "NewsWatchdog/1.0",
                "Accept": "application/json, text/javascript, */*",
            },
        ) as resp:
            if resp.status != 200:
                log.warning("DWD JSON HTTP %d", resp.status)
                return events

            raw = await resp.text()
    except Exception as e:
        log.warning("DWD JSON Fehler: %s", e)
        return events

    # Format: "warnWetter.loadWarnings({...});" — JSON extrahieren
    json_match = re.search(r"loadWarnings\((\{.*\})\);?\s*$", raw, re.DOTALL)
    if not json_match:
        # Alternativ: reines JSON
        json_match = re.search(r"(\{.*\})", raw, re.DOTALL)

    if not json_match:
        log.warning("DWD JSON: kein gültiges JSON gefunden")
        return events

    try:
        import json
        data = json.loads(json_match.group(1))
    except Exception as e:
        log.warning("DWD JSON Parse-Fehler: %s", e)
        return events

    # Warnungen durchlaufen
    # Struktur: {"time": ..., "warnings": {"REGION_ID": [warning, ...], ...}}
    warnings = data.get("warnings", {})

    # Aggregiere nach Warnungstyp — EIN Event pro event_type, höchste Severity gewinnt.
    # Das verhindert 30x "Frost", 10x "Glätte", 8x "Gewitter" Events.
    # Key: event_type (z.B. "FROST", "THUNDERSTORM") → Liste von (warning, region_id)
    aggregated: dict[str, list] = {}

    for region_id, warning_list in warnings.items():
        if not isinstance(warning_list, list):
            continue
        for warning in warning_list:
            severity = warning.get("level", 1)
            event_type = warning.get("event", "unknown")
            # Nur Moderate+ (Level 2+)
            if severity < 2:
                continue
            # Aggregations-Key: NUR Warnungstyp (NICHT severity!)
            agg_key = event_type
            if agg_key not in aggregated:
                aggregated[agg_key] = []
            aggregated[agg_key].append((warning, region_id))

    for agg_key, items in aggregated.items():
        # IMMER genau 1 Event pro Warnungstyp — egal ob 1 oder 500 Kreise
        event = _build_aggregated_event(items)
        if event:
            events.append(event)

    return events


def _build_aggregated_event(items: list[tuple[dict, str]]) -> Optional[dict]:
    """Mehrere gleiche DWD-Warnungen zu einem aggregierten Event zusammenfassen.
    Höchste Severity gewinnt, alle Regionen werden zusammengefasst."""
    if not items:
        return None

    # Höchste Severity finden
    max_severity = max(w.get("level", 1) for w, _ in items)
    # Headline von der höchsten Severity nehmen
    best_warning = next((w for w, _ in items if w.get("level", 1) == max_severity), items[0][0])
    headline = best_warning.get("headline", "DWD Warnung")
    event_type = best_warning.get("event", "")
    n = len(items)

    severity_scores = {1: 3.0, 2: 5.0, 3: 7.0, 4: 9.0}
    scoop_score = severity_scores.get(max_severity, 3.0)

    # Unique Regionen sammeln
    regions = []
    seen_regions = set()
    lats, lons = [], []
    for warning, region_id in items:
        rn = warning.get("regionName", "")
        if rn and rn not in seen_regions:
            regions.append(rn)
            seen_regions.add(rn)
        coords = _geocode_region(rn)
        if coords:
            lats.append(coords[0])
            lons.append(coords[1])

    if not lats:
        return None

    lat = sum(lats) / len(lats)
    lon = sum(lons) / len(lons)
    n_regions = len(regions)

    # Zeitpunkt: neuester Start
    start_ms = best_warning.get("start")
    zeitpunkt = datetime.now(timezone.utc).isoformat()
    if start_ms:
        try:
            zeitpunkt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
        except Exception:
            pass

    # Regionen-Zusammenfassung
    if n_regions == 1:
        region_text = regions[0]
    elif n_regions <= 3:
        region_text = ", ".join(regions[:3])
    else:
        region_text = f"{n_regions} Kreise betroffen"

    event_label = DWD_EVENT_MAP.get(event_type.upper(), event_type) if event_type else "Unwetter"

    # Titel: kurz und eindeutig pro Warnungstyp
    kreise_text = f"{n_regions} Kreis" if n_regions == 1 else f"{n_regions} Kreise"
    if n_regions == 1:
        titel = f"DWD: {event_label}-Warnung — {regions[0]}"
    else:
        titel = f"DWD: {event_label}-Warnung — {kreise_text}"

    # Severity-Stufen im Titel wenn Severe+
    if max_severity >= 3:
        stufe = "Unwetterwarnung" if max_severity == 3 else "EXTREME Unwetterwarnung"
        titel = f"DWD: {stufe} ({event_label}) — {kreise_text}"

    return {
        "titel": titel,
        "typ": "unwetter",
        "lat": lat,
        "lon": lon,
        "zeitpunkt": zeitpunkt,
        "scoop_score": scoop_score,
        "quellen": ["Deutscher Wetterdienst"],
        "zusammenfassung": f"{headline} — {region_text}. {event_label}-Warnung Stufe {max_severity} für {n_regions} Kreise.",
        "media_urls": [_dwd_wms_thumbnail(lat, lon, 400)],
        "link": "https://www.dwd.de/DE/wetter/warnungen/warnWetter_node.html",
        "quelle_typ": "dwd",
    }


def _parse_dwd_warning(warning: dict, region_id: str) -> Optional[dict]:
    """Eine einzelne DWD-Warnung in ein Event-Dict umwandeln."""
    headline = warning.get("headline", "")
    description = warning.get("description", "")
    event_type = warning.get("event", "")
    severity = warning.get("level", 1)  # 1-4 im JSON
    region_name = warning.get("regionName", "")

    if not headline:
        return None

    # Severity-Level zu Score mappen (JSON nutzt 1-4 statt Strings)
    severity_scores = {
        1: 3.0,   # Minor
        2: 5.0,   # Moderate
        3: 7.0,   # Severe
        4: 9.0,   # Extreme
    }
    scoop_score = severity_scores.get(severity, 3.0)

    # Nur Moderate+ Warnungen (Level 2+) — Minor-Frost etc. filtert zu viel Rauschen rein
    if severity < 2:
        return None

    # Zeitpunkt: start/end als Unix-Timestamp (ms)
    start_ms = warning.get("start")
    end_ms = warning.get("end")

    zeitpunkt = datetime.now(timezone.utc).isoformat()
    if start_ms:
        try:
            zeitpunkt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
        except Exception:
            pass

    # Geocoding: Region-Name → Koordinaten
    coords = _geocode_region(region_name)
    if not coords:
        return None

    lat, lon = coords

    # Zusammenfassung bauen
    zusammenfassung = headline
    if region_name:
        zusammenfassung = f"{headline} ({region_name})"
    if description:
        clean_desc = _clean_html(description)
        zusammenfassung = f"{zusammenfassung} — {clean_desc}"
    zusammenfassung = zusammenfassung[:500]

    # Event-Typ menschenlesbar
    event_label = DWD_EVENT_MAP.get(event_type.upper(), event_type) if event_type else "Unwetter"

    # Region in Titel aufnehmen damit Duplikat-Detection pro Region unique bleibt
    titel = f"DWD: {headline}"
    if region_name:
        titel = f"DWD: {headline} — {region_name}"
    if len(titel) > 200:
        titel = titel[:197] + "..."

    return {
        "titel": titel,
        "typ": "unwetter",
        "lat": lat,
        "lon": lon,
        "zeitpunkt": zeitpunkt,
        "scoop_score": scoop_score,
        "quellen": ["Deutscher Wetterdienst"],
        "zusammenfassung": zusammenfassung,
        "media_urls": [_dwd_wms_thumbnail(lat, lon)] if lat and lon else [],
        "link": "https://www.dwd.de/DE/wetter/warnungen/warnWetter_node.html",
        "quelle_typ": "dwd",
    }


# ---------------------------------------------------------------------------
# Strategie 2: DWD WFS GeoServer (Fallback)
# ---------------------------------------------------------------------------

async def _fetch_dwd_wfs(session: aiohttp.ClientSession) -> list[dict]:
    """DWD Warnungen via WFS GeoServer als GeoJSON abrufen."""
    events = []

    try:
        async with session.get(
            DWD_WFS_URL,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "NewsWatchdog/1.0"},
        ) as resp:
            if resp.status != 200:
                log.warning("DWD WFS HTTP %d", resp.status)
                return events

            import json
            data = await resp.json()
    except Exception as e:
        log.warning("DWD WFS Fehler: %s", e)
        return events

    features = data.get("features", [])
    log.debug("DWD WFS: %d Features", len(features))

    # Aggregiere nach EVENT-Typ — EIN Event pro Warnungstyp
    aggregated: dict[str, list] = {}

    for feature in features:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        headline = props.get("HEADLINE", "") or props.get("EVENT", "DWD Warnung")
        severity = props.get("SEVERITY", "Minor")
        event_type = props.get("EVENT", "unknown")

        # Nur Moderate+ Warnungen
        if severity == "Minor":
            continue

        coords = _extract_centroid(geometry)
        if not coords:
            continue

        # Key: NUR event_type (höchste Severity gewinnt)
        agg_key = event_type
        if agg_key not in aggregated:
            aggregated[agg_key] = []
        aggregated[agg_key].append((props, coords, headline))

    for agg_key, items in aggregated.items():
        # Höchste Severity finden
        severity_order = {"Minor": 0, "Moderate": 1, "Severe": 2, "Extreme": 3}
        max_sev_str = max((it[0].get("SEVERITY", "Minor") for it in items), key=lambda s: severity_order.get(s, 0))
        sev_info = DWD_SEVERITY_MAP.get(max_sev_str, DWD_SEVERITY_MAP["Minor"])
        n = len(items)
        event_type_label = DWD_EVENT_MAP.get(agg_key.upper(), agg_key)

        lats = [c[0] for _, c, _ in items]
        lons = [c[1] for _, c, _ in items]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        zeitpunkt_str = items[0][0].get("ONSET", "") or datetime.now(timezone.utc).isoformat()

        kreise_text = f"{n} Kreis" if n == 1 else f"{n} Kreise"
        if n == 1:
            titel = f"DWD: {event_type_label}-Warnung"
        else:
            titel = f"DWD: {event_type_label}-Warnung — {kreise_text}"

        if severity_order.get(max_sev_str, 0) >= 2:
            stufe = "Unwetterwarnung" if max_sev_str == "Severe" else "EXTREME Unwetterwarnung"
            titel = f"DWD: {stufe} ({event_type_label}) — {kreise_text}"

        events.append({
            "titel": titel,
            "typ": "unwetter",
            "lat": center_lat, "lon": center_lon,
            "zeitpunkt": zeitpunkt_str,
            "scoop_score": sev_info["score"],
            "quellen": ["Deutscher Wetterdienst"],
            "zusammenfassung": f"{event_type_label}-Warnung — {n} Kreise betroffen. Stufe: {max_sev_str}.",
            "media_urls": [_dwd_wms_thumbnail(center_lat, center_lon, 400)],
            "link": "https://www.dwd.de/DE/wetter/warnungen/warnWetter_node.html",
            "quelle_typ": "dwd",
        })

    return events


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

# Große Städte und Regionen für Geocoding
REGION_COORDS = {
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
    "bremen": (53.0793, 8.8017),
    "nürnberg": (49.4521, 11.0767),
    "bonn": (50.7374, 7.0982),
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
    "dortmund": (51.5136, 7.4653),
    "essen": (51.4556, 7.0116),
    "münster": (51.9607, 7.6261),
    "wuppertal": (51.2562, 7.1508),
    "bielefeld": (52.0302, 8.5325),
    # Bundesländer (Zentren)
    "bayern": (48.7904, 11.4979),
    "niedersachsen": (52.6367, 9.8451),
    "baden-württemberg": (48.6616, 9.3501),
    "nordrhein-westfalen": (51.4332, 7.6616),
    "hessen": (50.6521, 9.1624),
    "sachsen": (51.1045, 13.2017),
    "rheinland-pfalz": (49.9129, 7.4500),
    "schleswig-holstein": (54.2194, 9.6961),
    "thüringen": (50.8614, 11.0516),
    "brandenburg": (52.1310, 13.2152),
    "sachsen-anhalt": (51.9503, 11.6923),
    "mecklenburg-vorpommern": (53.6127, 12.4296),
    "saarland": (49.3964, 7.0230),
    # Regionen
    "nordsee": (54.5, 7.5),
    "ostsee": (54.4, 11.0),
    "alpen": (47.5, 11.0),
    "erzgebirge": (50.5, 13.0),
    "schwarzwald": (48.0, 8.0),
    "harz": (51.8, 10.6),
    "eifel": (50.3, 6.8),
    "bodensee": (47.6, 9.4),
}


def _geocode_region(region_name: str) -> Optional[tuple[float, float]]:
    """Region/Kreis-Name auf Koordinaten mappen."""
    if not region_name:
        return None

    region_lower = region_name.lower()

    # Direkte Treffer
    for key, coords in REGION_COORDS.items():
        if key in region_lower:
            return coords

    # Kreis/Stadt-Suffixe entfernen und nochmal versuchen
    clean = re.sub(r"\b(kreis|stadt|land|region|landkreis)\b", "", region_lower).strip()
    for key, coords in REGION_COORDS.items():
        if key in clean:
            return coords

    # Deutschland-Fallback (Mitte)
    return (51.1657, 10.4515)


def _extract_centroid(geometry: dict) -> Optional[tuple[float, float]]:
    """Centroid aus GeoJSON-Geometry berechnen."""
    if not geometry:
        return None

    geo_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if geo_type == "Point" and len(coords) >= 2:
        return (coords[1], coords[0])

    if geo_type == "Polygon" and coords:
        ring = coords[0]
        if ring and isinstance(ring[0], (list, tuple)):
            lats = [p[1] for p in ring]
            lons = [p[0] for p in ring]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

    if geo_type == "MultiPolygon" and coords:
        # Erstes Polygon nehmen
        first_poly = coords[0]
        if first_poly and first_poly[0]:
            ring = first_poly[0]
            if isinstance(ring[0], (list, tuple)):
                lats = [p[1] for p in ring]
                lons = [p[0] for p in ring]
                return (sum(lats) / len(lats), sum(lons) / len(lons))

    return None


def _dwd_wms_thumbnail(lat: float, lon: float, size: int = 300) -> str:
    """DWD WMS-Karten-Thumbnail für den Bereich um lat/lon generieren."""
    delta = 0.8  # ~80km Ausschnitt
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    return (
        f"https://maps.dwd.de/geoserver/dwd/wms"
        f"?service=WMS&version=1.1.1&request=GetMap"
        f"&layers=dwd:Warnungen_Gemeinden_vereinigt"
        f"&bbox={bbox}&width={size}&height={size}"
        f"&srs=EPSG:4326&format=image/png&transparent=true"
    )


def _clean_html(text: str) -> str:
    """HTML-Tags und überschüssige Whitespaces entfernen."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
