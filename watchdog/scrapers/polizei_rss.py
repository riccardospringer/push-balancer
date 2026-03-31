#!/usr/bin/env python3
"""
Scraper: Polizei-Presseportale via presseportal.de RSS
Quelle: presseportal.de stellt RSS-Feeds pro Blaulicht-Organisation bereit.

Umfasst:
- Alle 16 Landespolizeien (Praesidien + Direktionen)
- Bundespolizei (alle Direktionen)
- Feuerwehren der Top-20-Staedte + weitere
- THW Landesverbaende
- DLRG
- DRK / Rettungsdienste
- Zoll (Fahndungsaemter)

Rate-Limit Nominatim: max 1 Request/Sekunde (wir cachen aggressiv).
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from difflib import SequenceMatcher

import ssl
import aiohttp
import certifi
import feedparser

log = logging.getLogger("watchdog.polizei_rss")

# ---------------------------------------------------------------------------
# Blaulicht-Feeds: presseportal.de Dienststellen-IDs
# Format: https://www.presseportal.de/blaulicht/rss/dienststelle_{ID}.rss2
# ---------------------------------------------------------------------------

POLIZEI_FEEDS: dict[str, int] = {
    # ===================================================================
    # LANDESPOLIZEIEN — Alle 16 Bundesländer
    # ===================================================================

    # --- Baden-Württemberg ---
    "Polizeipräsidium Stuttgart": 110977,
    "Polizeipräsidium Karlsruhe": 110972,
    "Polizeipräsidium Mannheim": 14915,
    "Polizeipräsidium Freiburg": 110970,
    "Polizeipräsidium Konstanz": 110973,
    "Polizeipräsidium Offenburg": 110975,
    "Polizeipräsidium Pforzheim": 137462,  # Neue ID (alt: 110976)
    "Polizeipräsidium Heilbronn": 110971,
    "Polizeipräsidium Ludwigsburg": 110974,
    "Polizeipräsidium Reutlingen": 110978,
    "Polizeipräsidium Ulm": 110979,
    "Polizeipräsidium Aalen": 110969,
    "Polizeipräsidium Ravensburg": 138081,  # Neue ID (alt: 110968)
    "Polizeipräsidium Tuttlingen": 110980,

    # --- Bayern ---
    "Polizeipräsidium München": 6013,
    # OFFLINE: "Polizeipräsidium Oberbayern Nord": 56237,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeipräsidium Oberbayern Süd": 64017,  # Nicht mehr auf presseportal
    "Polizeipräsidium Mittelfranken": 6012,  # Nürnberg
    # OFFLINE: "Polizeipräsidium Unterfranken": 64720,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeipräsidium Oberfranken": 64358,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeipräsidium Oberpfalz": 64656,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeipräsidium Niederbayern": 64492,  # Nicht mehr auf presseportal
    "Polizeipräsidium Schwaben Nord": 169620,  # Augsburg, Neue ID (alt: 56023)
    # OFFLINE: "Polizeipräsidium Schwaben Süd/West": 56302,  # Nicht mehr auf presseportal

    # --- Berlin ---
    # OFFLINE: "Polizei Berlin": 12685,  # Nicht mehr auf presseportal

    # --- Brandenburg ---
    # OFFLINE: "Polizeidirektion Nord (Brandenburg)": 70138,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Ost (Brandenburg)": 70139,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Süd (Brandenburg)": 70140,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion West (Brandenburg)": 70141,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeipräsidium Potsdam": 70067,  # Nicht mehr auf presseportal

    # --- Bremen ---
    "Polizei Bremen": 35235,

    # --- Hamburg ---
    "Polizei Hamburg": 6337,

    # --- Hessen ---
    "Polizeipräsidium Südhessen": 4969,  # Darmstadt
    "Polizeipräsidium Frankfurt am Main": 4970,
    "Polizeipräsidium Mittelhessen": 154582,  # Gießen, Neue ID (alt: 43559)
    "Polizeipräsidium Nordhessen": 44143,  # Kassel
    "Polizeipräsidium Osthessen": 43558,  # Fulda, Neue ID (alt: 56643)
    "Polizeipräsidium Südosthessen": 43561,  # Offenbach
    "Polizeipräsidium Westhessen": 43562,  # Wiesbaden

    # --- Mecklenburg-Vorpommern ---
    "Polizeipräsidium Rostock": 108746,
    "Polizeipräsidium Neubrandenburg": 108747,

    # --- Niedersachsen ---
    "Polizeidirektion Hannover": 66841,
    "Polizeidirektion Braunschweig": 12268,
    "Polizeidirektion Göttingen": 7452,
    # OFFLINE: "Polizeidirektion Oldenburg": 68437,  # Nicht mehr auf presseportal (eigentlich PI Cuxhaven)
    "Polizeidirektion Osnabrück": 104236,
    "Polizeiinspektion Lüneburg/Lüchow-Dannenberg/Uelzen": 59488,
    "Polizeiinspektion Celle": 24220,
    "Polizeiinspektion Hildesheim": 57929,
    "Polizeiinspektion Wilhelmshaven/Friesland": 68463,
    "Polizeiinspektion Cuxhaven": 68416,
    "Polizeiinspektion Nienburg/Schaumburg": 57922,
    "Polizeiinspektion Stade": 59461,
    "Polizeiinspektion Verden/Osterholz": 68441,

    # --- Nordrhein-Westfalen ---
    "Polizei Köln": 12415,
    "Polizei Düsseldorf": 13248,
    "Polizei Dortmund": 4971,
    "Polizei Essen": 11011,
    "Polizei Duisburg": 13387,
    "Polizei Bochum": 11530,
    "Polizei Wuppertal": 11811,
    "Polizei Bielefeld": 12522,
    "Polizei Bonn": 7304,
    "Polizei Münster": 11187,
    "Polizei Gelsenkirchen": 48166,
    "Polizei Mönchengladbach": 30127,
    "Polizei Aachen": 11559,
    "Polizei Krefeld": 50667,
    "Polizei Hagen": 30835,
    "Polizei Recklinghausen": 42900,
    "Polizei Paderborn": 55928,
    "Kreispolizeibehörde Siegen-Wittgenstein": 65854,
    "Kreispolizeibehörde Märkischer Kreis": 65850,
    "Kreispolizeibehörde Rhein-Erft-Kreis": 10374,
    "Kreispolizeibehörde Rhein-Sieg-Kreis": 65853,
    "Kreispolizeibehörde Oberbergischer Kreis": 65843,
    "Kreispolizeibehörde Ennepe-Ruhr-Kreis": 5765,
    "Kreispolizeibehörde Euskirchen": 65841,
    "Kreispolizeibehörde Herford": 65846,
    "Kreispolizeibehörde Kleve": 65847,
    "Kreispolizeibehörde Mettmann": 43777,
    "Kreispolizeibehörde Minden-Lübbecke": 65851,
    "Kreispolizeibehörde Soest": 65855,
    "Kreispolizeibehörde Unna": 65856,
    "Kreispolizeibehörde Viersen": 65857,
    "Kreispolizeibehörde Wesel": 65858,

    # --- Rheinland-Pfalz ---
    "Polizeipräsidium Mainz": 117708,
    # OFFLINE: "Polizeipräsidium Koblenz": 117709,  # Nicht mehr auf presseportal (eigentlich PD Neuwied)
    # OFFLINE: "Polizeipräsidium Trier": 117710,  # Nicht mehr auf presseportal (eigentlich PD Montabaur)
    # OFFLINE: "Polizeipräsidium Westpfalz": 117711,  # Nicht mehr auf presseportal (eigentlich PD Mayen)
    # OFFLINE: "Polizeipräsidium Rheinpfalz": 117712,  # Nicht mehr auf presseportal (eigentlich PD Koblenz)

    # --- Saarland ---
    "Landespolizeipräsidium Saarland": 138225,

    # --- Sachsen ---
    # OFFLINE: "Polizeidirektion Dresden": 14090,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Leipzig": 56528,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Chemnitz": 64560,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Zwickau": 126726,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeidirektion Görlitz": 126725,  # Nicht mehr auf presseportal (eigentlich LPI Suhl)

    # --- Sachsen-Anhalt ---
    # OFFLINE: "Polizeiinspektion Magdeburg": 56520,  # Nicht mehr auf presseportal (eigentlich Wolfsburg)
    # OFFLINE: "Polizeiinspektion Halle": 56518,  # Nicht mehr auf presseportal (eigentlich Goslar)
    # OFFLINE: "Polizeiinspektion Dessau-Roßlau": 126733,  # Nicht mehr auf presseportal
    # OFFLINE: "Polizeiinspektion Stendal": 126735,  # Nicht mehr auf presseportal

    # --- Schleswig-Holstein ---
    "Polizeidirektion Kiel": 14626,
    "Polizeidirektion Lübeck": 43738,  # Neue ID (alt: 14627)
    "Polizeidirektion Flensburg": 6313,  # Neue ID (alt: 14625)
    "Polizeidirektion Neumünster": 14628,
    "Polizeidirektion Itzehoe": 108759,
    "Polizeidirektion Ratzeburg": 108763,

    # --- Thüringen ---
    "Landespolizeiinspektion Erfurt": 126719,  # Fix (alt: 126721)
    "Landespolizeiinspektion Jena": 126722,
    "Landespolizeiinspektion Gera": 126720,
    "Landespolizeiinspektion Gotha": 126721,  # Fix (alt: 126719)
    "Landespolizeiinspektion Saalfeld": 126724,  # Fix (alt: 126723)
    "Landespolizeiinspektion Nordhausen": 126723,  # Fix (alt: 126724)
    "Landespolizeiinspektion Suhl": 126725,  # Fix (alt: 126718)

    # ===================================================================
    # BUNDESPOLIZEI — Direktionen und Inspektionen
    # ===================================================================
    "Bundespolizeidirektion Sankt Augustin": 72277,
    "Bundespolizeiinspektion Frankfurt/Main Flughafen": 62532,
    "Bundespolizeiinspektion Hamburg": 73844,
    "Bundespolizeiinspektion München": 73843,
    "Bundespolizeiinspektion Hannover": 57895,
    "Bundespolizeiinspektion Köln": 70163,
    "Bundespolizeiinspektion Berlin-Ostbahnhof": 73839,
    "Bundespolizeiinspektion Flensburg": 68527,
    "Bundespolizeiinspektion Rostock": 108757,
    "Bundespolizeiinspektion Magdeburg": 73842,
    "Bundespolizeiinspektion Klingenthal": 73841,
    "Bundespolizeiinspektion Bad Bentheim": 68440,
    "Bundespolizeiinspektion Konstanz": 110966,
    "Bundespolizeiinspektion Trier": 117700,
    "Bundespolizeiinspektion Ludwigsdorf": 73840,
}

FEUERWEHR_FEEDS: dict[str, int] = {
    # ===================================================================
    # FEUERWEHREN — Top-20-Städte + weitere
    # ===================================================================
    "Feuerwehr Berlin": 6339,
    "Feuerwehr Hamburg": 82522,  # Neue ID (alt: 6338 war Axel Springer)
    "Feuerwehr München": 131419,  # Neue ID (alt: 6335)
    "Feuerwehr Köln": 24841,
    "Feuerwehr Frankfurt am Main": 31467,
    "Feuerwehr Stuttgart": 116244,
    "Feuerwehr Düsseldorf": 31424,
    "Feuerwehr Leipzig": 60152,
    "Feuerwehr Dresden": 154636,  # Neue ID (alt: 80853)
    "Feuerwehr Hannover": 66842,
    "Feuerwehr Dortmund": 8763,
    "Feuerwehr Essen": 29482,
    "Feuerwehr Bremen": 130368,  # Neue ID (alt: 35238)
    "Feuerwehr Nürnberg": 58743,
    "Feuerwehr Duisburg": 70507,
    "Feuerwehr Bochum": 115868,  # Neue ID (alt: 118794)
    "Feuerwehr Wuppertal": 42285,
    "Feuerwehr Bonn": 29427,
    "Feuerwehr Münster": 21521,
    "Feuerwehr Karlsruhe": 75999,
    # OFFLINE: "Feuerwehr Mannheim": 43735,  # Nicht mehr auf presseportal (eigentlich PD Ratzeburg)
    "Feuerwehr Augsburg": 46651,
    "Feuerwehr Freiburg": 110740,
    "Feuerwehr Kiel": 82765,  # Neue ID (alt: 29384)
    "Feuerwehr Rostock": 175951,  # Neue ID (alt: 118796)
    "Feuerwehr Magdeburg": 56522,
    # OFFLINE: "Feuerwehr Erfurt": 126728,  # Nicht mehr auf presseportal (eigentlich Corporate Entity)
    "Feuerwehr Mainz": 117706,
    "Feuerwehr Wiesbaden": 43563,
    # OFFLINE: "Feuerwehr Saarbrücken": 138226,  # Nicht mehr auf presseportal (eigentlich National Business Daily)
    "Feuerwehr Potsdam": 70069,
    # OFFLINE: "Feuerwehr Kassel": 44150,  # Nicht mehr auf presseportal (eigentlich Polizei Korbach)
    "Feuerwehr Braunschweig": 24217,
    "Feuerwehr Aachen": 11555,
    "Feuerwehr Krefeld": 50666,
    "Feuerwehr Mönchengladbach": 30122,
    "Feuerwehr Gelsenkirchen": 48165,
    "Feuerwehr Hagen": 30834,
    "Feuerwehr Oberhausen": 113653,
    "Feuerwehr Lübeck": 29635,
    "Feuerwehr Oldenburg": 68436,
    "Feuerwehr Osnabrück": 104237,
    "Feuerwehr Darmstadt": 43729,
    "Feuerwehr Heidelberg": 110964,
    "Feuerwehr Ulm": 116242,
    "Feuerwehr Reutlingen": 116246,
    "Feuerwehr Heilbronn": 110965,
    "Feuerwehr Pforzheim": 116243,
    "Feuerwehr Göttingen": 22517,
}

# THW, DLRG, DRK, ASB, Johanniter, Malteser — ENTFERNT
# Grund: Produzieren überwiegend Nicht-Crime-Meldungen
# (Übungen, Blutspende, Schwimmkurse, Tag der offenen Tür)
THW_FEEDS: dict[str, int] = {}
RETTUNG_FEEDS: dict[str, int] = {}

ZOLL_FEEDS: dict[str, int] = {
    # ===================================================================
    # ZOLL — Fahndungsämter
    # ===================================================================
    "Hauptzollamt Frankfurt am Main": 113585,
    "Hauptzollamt München": 113589,
    "Hauptzollamt Hamburg": 113586,
    "Hauptzollamt Köln": 113588,
    "Hauptzollamt Düsseldorf": 113583,
    "Zollfahndungsamt Essen": 12929,
    "Zollfahndungsamt München": 14098,
    "Zollfahndungsamt Frankfurt am Main": 14099,
    "Zollfahndungsamt Hamburg": 14100,
}

# Alle Feeds zusammengefasst
ALL_FEEDS: dict[str, int] = {}
ALL_FEEDS.update(POLIZEI_FEEDS)
ALL_FEEDS.update(FEUERWEHR_FEEDS)
ALL_FEEDS.update(THW_FEEDS)
ALL_FEEDS.update(RETTUNG_FEEDS)
ALL_FEEDS.update(ZOLL_FEEDS)

RSS_URL_TEMPLATE = "https://www.presseportal.de/blaulicht/rss/dienststelle_{feed_id}.rss2"

# ---------------------------------------------------------------------------
# Geocoding-Cache (In-Memory, überlebt einen Server-Restart nicht)
# ---------------------------------------------------------------------------

_geocode_cache: dict[str, Optional[tuple[float, float]]] = {}
_last_nominatim_call = 0.0


async def geocode_city(city: str, session: aiohttp.ClientSession) -> Optional[tuple[float, float]]:
    """
    Stadt/Ort via Nominatim geocoden. Cached, 1 req/sec Rate Limit.
    Gibt (lat, lon) zurück oder None.
    """
    global _last_nominatim_call

    city_key = city.lower().strip()
    if city_key in _geocode_cache:
        return _geocode_cache[city_key]

    # Rate Limit: mindestens 1 Sekunde zwischen Requests
    now = time.time()
    wait = max(0, 1.05 - (now - _last_nominatim_call))
    if wait > 0:
        await asyncio.sleep(wait)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{city}, Deutschland",
        "format": "json",
        "limit": 1,
        "countrycodes": "de",
    }
    headers = {"User-Agent": "NewsWatchdog/1.0 (redaktion@example.de)"}

    try:
        _last_nominatim_call = time.time()
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    _geocode_cache[city_key] = (lat, lon)
                    return (lat, lon)
    except Exception as e:
        log.warning("Nominatim Fehler für '%s': %s", city, e)

    _geocode_cache[city_key] = None
    return None


async def geocode_address(street: str, city: str, session: aiohttp.ClientSession) -> Optional[tuple[float, float]]:
    """
    Straßen-genaues Geocoding via Nominatim structured query.
    Cached, 1 req/sec Rate Limit. Gibt (lat, lon) oder None zurück.
    """
    global _last_nominatim_call

    cache_key = f"{street}|{city}".lower().strip()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    # Rate Limit: mindestens 1 Sekunde zwischen Requests
    now = time.time()
    wait = max(0, 1.05 - (now - _last_nominatim_call))
    if wait > 0:
        await asyncio.sleep(wait)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "street": street,
        "city": city,
        "country": "Deutschland",
        "format": "json",
        "limit": 1,
        "countrycodes": "de",
    }
    headers = {"User-Agent": "NewsWatchdog/1.0 (redaktion@example.de)"}

    try:
        _last_nominatim_call = time.time()
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    _geocode_cache[cache_key] = (lat, lon)
                    log.debug("Geocode Straße OK: '%s, %s' → (%.4f, %.4f)", street, city, lat, lon)
                    return (lat, lon)
    except Exception as e:
        log.warning("Nominatim Straßen-Fehler für '%s, %s': %s", street, city, e)

    _geocode_cache[cache_key] = None
    return None


# ---------------------------------------------------------------------------
# Bekannte Städte-Fallback (für den Fall dass Nominatim ausfällt)
# Massiv erweitert für alle Dienststellen-Städte
# ---------------------------------------------------------------------------

CITY_FALLBACK: dict[str, tuple[float, float]] = {
    # --- Landeshauptstädte ---
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "münchen": (48.1351, 11.5820),
    "munich": (48.1351, 11.5820),
    "köln": (50.9375, 6.9603),
    "frankfurt": (50.1109, 8.6821),
    "frankfurt am main": (50.1109, 8.6821),
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
    "wiesbaden": (50.0782, 8.2398),
    "mainz": (49.9929, 8.2473),
    "freiburg": (47.9990, 7.8421),
    "kiel": (54.3233, 10.1228),
    "rostock": (54.0924, 12.0991),
    "magdeburg": (52.1205, 11.6276),
    "erfurt": (50.9848, 11.0299),
    "potsdam": (52.3906, 13.0645),
    "cottbus": (51.7563, 14.3329),
    "frankfurt (oder)": (52.3471, 14.5506),
    "saarbrücken": (49.2402, 6.9969),
    "chemnitz": (50.8278, 12.9214),

    # --- Großstädte > 100k ---
    "gelsenkirchen": (51.5177, 7.0857),
    "mönchengladbach": (51.1805, 6.4428),
    "aachen": (50.7753, 6.0839),
    "krefeld": (51.3388, 6.5853),
    "hagen": (51.3671, 7.4633),
    "oberhausen": (51.4963, 6.8635),
    "lübeck": (53.8655, 10.6866),
    "oldenburg": (53.1435, 8.2146),
    "osnabrück": (52.2799, 8.0472),
    "darmstadt": (49.8728, 8.6512),
    "heidelberg": (49.3988, 8.6724),
    "recklinghausen": (51.6141, 7.1979),
    "paderborn": (51.7189, 8.7544),
    "ulm": (48.4011, 9.9876),
    "reutlingen": (48.4914, 9.2043),
    "heilbronn": (49.1427, 9.2109),
    "pforzheim": (48.8922, 8.6946),
    "göttingen": (51.5328, 9.9352),
    "braunschweig": (52.2689, 10.5268),
    "kassel": (51.3127, 9.4797),
    "wolfsburg": (52.4227, 10.7865),
    "ludwigsburg": (48.8975, 9.1923),
    "offenbach": (50.0956, 8.7761),
    "gießen": (50.5840, 8.6784),
    "fulda": (50.5528, 9.6778),

    # --- Weitere wichtige Städte ---
    "konstanz": (47.6603, 9.1751),
    "offenburg": (48.4721, 7.9408),
    "tuttlingen": (47.9841, 8.8171),
    "ravensburg": (47.7816, 9.6114),
    "aalen": (48.8364, 10.0926),
    "trier": (49.7497, 6.6372),
    "kaiserslautern": (49.4432, 7.7689),
    "ludwigshafen": (49.4774, 8.4453),
    "koblenz": (50.3569, 7.5890),
    "flensburg": (54.7937, 9.4469),
    "neumünster": (54.0713, 9.9869),
    "itzehoe": (53.9254, 9.5153),
    "ratzeburg": (53.7005, 10.7572),
    "neubrandenburg": (53.5565, 13.2612),
    "schwerin": (53.6355, 11.4012),
    "celle": (52.6224, 10.0807),
    "hildesheim": (52.1508, 9.9511),
    "wilhelmshaven": (53.5220, 8.1256),
    "cuxhaven": (53.8719, 8.6920),
    "nienburg": (52.6365, 9.2151),
    "stade": (53.5977, 9.4743),
    "verden": (52.9191, 9.2355),
    "lüneburg": (53.2464, 10.4115),
    "dessau": (51.8312, 12.2469),
    "dessau-roßlau": (51.8312, 12.2469),
    "halle": (51.4828, 11.9700),
    "stendal": (52.6058, 11.8597),
    "zwickau": (50.7181, 12.4963),
    "görlitz": (51.1526, 14.9870),
    "jena": (50.9272, 11.5892),
    "gera": (50.8810, 12.0833),
    "gotha": (50.9489, 10.7018),
    "saalfeld": (50.6489, 11.3537),
    "nordhausen": (51.5058, 10.7908),
    "suhl": (50.6091, 10.6940),
    "siegen": (50.8748, 8.0243),
    "iserlohn": (51.3758, 7.6948),
    "euskirchen": (50.6610, 6.7872),
    "herford": (52.1145, 8.6734),
    "kleve": (51.7899, 6.1388),
    "mettmann": (51.2505, 6.9715),
    "minden": (52.2887, 8.9168),
    "soest": (51.5711, 8.1059),
    "unna": (51.5348, 7.6889),
    "viersen": (51.2538, 6.3952),
    "wesel": (51.6586, 6.6206),
    "lüdenscheid": (51.2193, 7.6261),
    "bergisch gladbach": (50.9856, 7.1327),

    # --- Bundespolizei-Standorte ---
    "sankt augustin": (50.7707, 7.1863),
    "bad bentheim": (52.3033, 7.1597),
    "klingenthal": (50.3545, 12.4658),
    "ludwigsdorf": (51.1526, 14.9870),

    # --- Sonstige ---
    "worms": (49.6341, 8.3507),
    "speyer": (49.3173, 8.4312),
    "landau": (49.1985, 8.1181),
    "frankenthal": (49.5343, 8.3509),
    "neustadt": (49.3498, 8.1389),
    "pirmasens": (49.2015, 7.6046),
    "zweibrücken": (49.2490, 7.3694),
    "ingolstadt": (48.7665, 11.4258),
    "regensburg": (49.0134, 12.1016),
    "würzburg": (49.7913, 9.9534),
    "bamberg": (49.8988, 10.9028),
    "bayreuth": (49.9427, 11.5761),
    "passau": (48.5748, 13.4609),
    "rosenheim": (47.8571, 12.1181),
    "landshut": (48.5442, 12.1466),
    "kempten": (47.7268, 10.3154),
    "memmingen": (47.9837, 10.1816),
}


def fallback_geocode(text: str) -> Optional[tuple[float, float]]:
    """Versuche den Ort aus dem Text über die Fallback-Tabelle zu finden."""
    text_lower = text.lower()
    # Längere Stadtnamen zuerst prüfen (z.B. "frankfurt am main" vor "frankfurt")
    sorted_cities = sorted(CITY_FALLBACK.keys(), key=len, reverse=True)
    for city in sorted_cities:
        if city in text_lower:
            return CITY_FALLBACK[city]
    return None


# ---------------------------------------------------------------------------
# Dienststellen-Stadt-Mapping (extrahiert Stadt aus dem Dienststellennamen)
# ---------------------------------------------------------------------------

def _extract_city_from_dienststelle(name: str) -> str:
    """Extrahiert den Stadtnamen aus dem Dienststellennamen."""
    # Präfixe entfernen
    prefixes = [
        "Polizeipräsidium ", "Polizeidirektion ", "Polizeiinspektion ",
        "Landespolizeiinspektion ", "Landespolizeipräsidium ",
        "Kreispolizeibehörde ",
        "Polizei ", "Feuerwehr ",
        "Bundespolizeiinspektion ", "Bundespolizeidirektion ",
        "Hauptzollamt ", "Zollfahndungsamt ",
        "THW Landesverband ", "THW ",
        "DLRG ", "DRK ", "ASB ", "Johanniter-", "Malteser ",
    ]
    city = name
    for prefix in prefixes:
        if city.startswith(prefix):
            city = city[len(prefix):]
            break

    # Klammerzusätze entfernen: "Nord (Brandenburg)" -> "Nord"
    city = re.sub(r"\s*\(.*?\)", "", city)

    # Richtungsangaben entfernen die keine Städte sind
    directional = ["Nord", "Süd", "Ost", "West", "Süd/West", "Oberbayern", "Unterfranken",
                   "Oberfranken", "Oberpfalz", "Niederbayern", "Mittelfranken", "Mittelhessen",
                   "Nordhessen", "Osthessen", "Südosthessen", "Westhessen", "Rheinpfalz",
                   "Westpfalz", "Schwaben"]
    if city.strip() in directional:
        return ""

    # "Schwaben Nord" -> "Augsburg" etc. — Spezialfälle
    region_city_map = {
        "Schwaben Nord": "Augsburg",
        "Schwaben Süd/West": "Memmingen",
        "Oberbayern Nord": "Ingolstadt",
        "Oberbayern Süd": "Rosenheim",
        "Unterfranken": "Würzburg",
        "Oberfranken": "Bayreuth",
        "Oberpfalz": "Regensburg",
        "Niederbayern": "Passau",
        "Mittelfranken": "Nürnberg",
        "Mittelhessen": "Gießen",
        "Nordhessen": "Kassel",
        "Osthessen": "Fulda",
        "Südosthessen": "Offenbach",
        "Westhessen": "Wiesbaden",
        "Rheinpfalz": "Ludwigshafen",
        "Westpfalz": "Kaiserslautern",
        "Südhessen": "Darmstadt",
        "Saarland": "Saarbrücken",
    }
    for region, mapped_city in region_city_map.items():
        if region in city:
            return mapped_city

    # Slash-Aufzählungen: "Lüneburg/Lüchow-Dannenberg/Uelzen" -> "Lüneburg"
    if "/" in city:
        city = city.split("/")[0].strip()

    # Bindestrich-Städte beibehalten: "Dessau-Roßlau"
    return city.strip()


# ---------------------------------------------------------------------------
# Polizei-Kennungen (POL-XX, FW-XX) -> Stadt-Mapping
# Massiv erweitert für alle Bundesländer
# ---------------------------------------------------------------------------

KENNUNG_MAP: dict[str, str] = {
    # Baden-Württemberg
    "POL-S": "Stuttgart", "POL-KA": "Karlsruhe", "POL-MA": "Mannheim",
    "POL-FR": "Freiburg", "POL-KN": "Konstanz", "POL-OG": "Offenburg",
    "POL-PF": "Pforzheim", "POL-HN": "Heilbronn", "POL-LB": "Ludwigsburg",
    "POL-RT": "Reutlingen", "POL-UL": "Ulm", "POL-AA": "Aalen",
    "POL-RV": "Ravensburg", "POL-TUT": "Tuttlingen",
    # Bayern
    "POL-M": "München", "POL-N": "Nürnberg",
    # Berlin
    "POL-B": "Berlin",
    # Brandenburg
    "POL-P": "Potsdam",
    # Bremen
    "POL-HB": "Bremen",
    # Hamburg
    "POL-HH": "Hamburg",
    # Hessen
    "POL-F": "Frankfurt", "POL-DA": "Darmstadt", "POL-GI": "Gießen",
    "POL-KS": "Kassel", "POL-OF": "Offenbach", "POL-WI": "Wiesbaden",
    # Mecklenburg-Vorpommern
    "POL-HRO": "Rostock", "POL-NB": "Neubrandenburg",
    # Niedersachsen
    "POL-H": "Hannover", "POL-BS": "Braunschweig", "POL-GÖ": "Göttingen",
    "POL-OL": "Oldenburg", "POL-OS": "Osnabrück", "POL-CE": "Celle",
    "POL-HI": "Hildesheim", "POL-CUX": "Cuxhaven", "POL-STD": "Stade",
    "POL-LG": "Lüneburg",
    # NRW
    "POL-K": "Köln", "POL-D": "Düsseldorf", "POL-DO": "Dortmund",
    "POL-E": "Essen", "POL-DU": "Duisburg", "POL-BO": "Bochum",
    "POL-W": "Wuppertal", "POL-BI": "Bielefeld", "POL-BN": "Bonn",
    "POL-MS": "Münster", "POL-GE": "Gelsenkirchen", "POL-MG": "Mönchengladbach",
    "POL-AC": "Aachen", "POL-KR": "Krefeld", "POL-HA": "Hagen",
    "POL-RE": "Recklinghausen", "POL-PB": "Paderborn",
    "POL-SI": "Siegen", "POL-MK": "Iserlohn", "POL-UN": "Unna",
    "POL-SO": "Soest", "POL-HF": "Herford", "POL-MI": "Minden",
    "POL-KLE": "Kleve", "POL-ME": "Mettmann", "POL-VIE": "Viersen",
    "POL-WES": "Wesel", "POL-EU": "Euskirchen", "POL-BM": "Bergheim",
    "POL-SU": "Siegburg", "POL-GM": "Gummersbach",
    # Rheinland-Pfalz
    "POL-MZ": "Mainz", "POL-KO": "Koblenz", "POL-TR": "Trier",
    "POL-PDLU": "Ludwigshafen", "POL-PDKL": "Kaiserslautern",
    # Saarland
    "POL-SB": "Saarbrücken",
    # Sachsen
    "POL-DD": "Dresden", "POL-L": "Leipzig", "POL-C": "Chemnitz",
    "POL-Z": "Zwickau",
    # Sachsen-Anhalt
    "POL-MD": "Magdeburg", "POL-HAL": "Halle", "POL-DE": "Dessau",
    "POL-SDL": "Stendal",
    # Schleswig-Holstein
    "POL-FL": "Flensburg", "POL-NMS": "Neumünster",
    "POL-IZ": "Itzehoe", "POL-RZ": "Ratzeburg",
    "POL-HL": "Lübeck",
    # Thüringen
    "POL-EF": "Erfurt", "POL-J": "Jena", "POL-G": "Gera",
    "POL-GTH": "Gotha", "POL-SLF": "Saalfeld", "POL-NDH": "Nordhausen",
    "POL-SHL": "Suhl",
    # Feuerwehr-Kennungen
    "FW-B": "Berlin", "FW-M": "München", "FW-HH": "Hamburg",
    "FW-K": "Köln", "FW-F": "Frankfurt", "FW-S": "Stuttgart",
    "FW-D": "Düsseldorf", "FW-L": "Leipzig", "FW-DD": "Dresden",
    "FW-H": "Hannover", "FW-DO": "Dortmund", "FW-E": "Essen",
    "FW-HB": "Bremen", "FW-N": "Nürnberg", "FW-DU": "Duisburg",
    "FW-BO": "Bochum", "FW-W": "Wuppertal", "FW-BN": "Bonn",
    "FW-MS": "Münster", "FW-KA": "Karlsruhe", "FW-MA": "Mannheim",
    # Bundespolizei
    "BPOL": "Berlin", "BPOLD": "Berlin",
}


# ---------------------------------------------------------------------------
# Ort aus Presseportal-Titel extrahieren
# ---------------------------------------------------------------------------

def extract_location(title: str, description: str = "") -> str:
    """
    Presseportal-Titel haben typisch das Format:
    "POL-XX: Irgendein Vorfall" oder "FW-XX: Brand in Musterstadt"
    Der Ortsname steht oft am Anfang der Beschreibung oder im Titel.
    """
    title_upper = title.upper().strip()

    # Schritt 1: Bekannte Polizei-/Feuerwehr-Kennungen matchen
    # Sortiert nach Länge (längste zuerst), damit "POL-HH" vor "POL-H" matcht
    for prefix in sorted(KENNUNG_MAP.keys(), key=len, reverse=True):
        if title_upper.startswith(prefix.upper() + ":") or title_upper.startswith(prefix.upper() + " "):
            return KENNUNG_MAP[prefix]

    # Schritt 2: Ort aus "(ots)" Zeile extrahieren — presseportal nutzt oft
    # "Stadtname (ots) - Beschreibung..." in der Beschreibung
    ots_match = re.search(r"^([A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\s\-/]+?)\s*\(ots\)", description, re.IGNORECASE)
    if ots_match:
        return ots_match.group(1).strip()

    # Schritt 3: Bekannte Städte im Text suchen
    combined = f"{title} {description}"
    result = fallback_geocode(combined)
    if result:
        # Finde den passenden Stadtnamen
        combined_lower = combined.lower()
        sorted_cities = sorted(CITY_FALLBACK.keys(), key=len, reverse=True)
        for city in sorted_cities:
            if city in combined_lower:
                return city.title()

    return ""


# ---------------------------------------------------------------------------
# Straßen-Adress-Extraktion aus Presseportal-Beschreibungen
# ---------------------------------------------------------------------------

# Deutsche Straßen-Suffixe (Regex-escaped)
_STREET_SUFFIXES = (
    r"(?:stra(?:ß|ss)e|str\.|weg|platz|allee|ring|damm|gasse|pfad|ufer|"
    r"brücke|chaussee|steig|bogen|zeile|promenade|markt|graben|anger|hof|wall|tor)"
)

# Hausnummer: 5, 12a, 12-14, 12/14
_HAUSNR = r"(?:\s+\d+[\-/]?\d*\s*[a-zA-Z]?)?"

# Pattern 1: "Musterstraße 5" — Wort(e) + Suffix + opt. Nummer
_PAT_STREET_SIMPLE = re.compile(
    rf"([A-ZÄÖÜ][a-zäöüß]+(?:[-][A-ZÄÖÜ][a-zäöüß]+)?{_STREET_SUFFIXES}{_HAUSNR})",
    re.UNICODE,
)

# Pattern 2: "Am Leopoldplatz", "An der Schillerstraße 5", "Auf dem Marktplatz"
_PAT_STREET_PREP = re.compile(
    rf"((?:Am|An\s+der|An\s+dem|Auf\s+der|Auf\s+dem|Im|In\s+der|In\s+dem|Vor\s+dem|Hinter\s+der|Unter\s+der|Über\s+der|Zum|Zur)\s+[A-ZÄÖÜ][a-zäöüß]+(?:[-][A-ZÄÖÜ][a-zäöüß]+)?{_STREET_SUFFIXES}?{_HAUSNR})",
    re.UNICODE,
)

# Pattern 3: "Berliner Straße 12a", "Kölner Weg 3"
_PAT_STREET_ADJ = re.compile(
    rf"([A-ZÄÖÜ][a-zäöüß]+er\s+(?:Stra(?:ß|ss)e|Str\.|Weg|Platz|Allee|Ring|Damm|Chaussee){_HAUSNR})",
    re.UNICODE,
)

# Pattern 4: "Hauptstr. 5"
_PAT_STREET_ABBR = re.compile(
    rf"([A-ZÄÖÜ][a-zäöüß]+str\.{_HAUSNR})",
    re.UNICODE,
)

# Wörter die fälschlich matchen könnten
_STREET_BLACKLIST = {
    "Polizeistraße", "Pressestelle", "Bundesstraße", "Landesstraße",
    "Kreisstraße", "Staatsstraße", "Autobahnplatz",
}


def extract_street_address(text: str) -> Optional[str]:
    """
    Extrahiert eine Straßen-Adresse (ggf. mit Hausnummer) aus deutschem Text.
    Gibt z.B. "Kottbusser Straße 5" oder "Am Leopoldplatz" zurück, oder None.
    """
    if not text:
        return None

    # Patterns in Prioritätsreihenfolge
    for pattern in [_PAT_STREET_PREP, _PAT_STREET_ADJ, _PAT_STREET_SIMPLE, _PAT_STREET_ABBR]:
        match = pattern.search(text)
        if match:
            street = match.group(1).strip()
            if street not in _STREET_BLACKLIST and len(street) >= 5:
                return street
    return None


# ---------------------------------------------------------------------------
# Scoop-Score berechnen
# ---------------------------------------------------------------------------

SCOOP_KEYWORDS = {
    # Hohe Relevanz (Score-Boost)
    "tot": 3.0, "tödlich": 3.0, "getötet": 3.0, "todesopfer": 3.0, "leiche": 2.5,
    "mord": 3.0, "mordkommission": 2.5, "tötungsdelikt": 3.0,
    "schusswaffe": 2.5, "schüsse": 2.5, "messerangriff": 2.5, "messerattacke": 2.5,
    "großeinsatz": 2.0, "großlage": 2.5, "sek": 2.0, "spezialeinsatzkommando": 2.0,
    "terrorverdacht": 3.0, "bombe": 2.5, "sprengstoff": 2.5, "evakuierung": 2.0,
    "geiselnahme": 3.0, "amoklauf": 3.0,
    # Mittlere Relevanz
    "schwerverletzt": 1.5, "lebensgefährlich": 2.0, "reanimation": 1.5,
    "hubschrauber": 1.5, "rettungshubschrauber": 1.5, "christoph": 1.0,
    "autobahn": 1.0, "vollsperrung": 1.5, "massenkarambolage": 2.0,
    "großbrand": 2.0, "dachstuhlbrand": 1.0, "explosion": 2.0,
    "überfall": 1.5, "raub": 1.0, "bankraub": 2.0,
    "vermisst": 1.0, "kind": 1.0, "flucht": 1.0, "fahndung": 1.5,
    "demo": 0.5, "demonstration": 0.5, "protest": 0.5,
    # Ergänzungen
    "überschwemmung": 2.0, "hochwasser": 2.0, "erdbeben": 2.5,
    "flugzeugabsturz": 3.0, "zugunglück": 3.0, "entgleisung": 2.0,
    "gefahrgut": 1.5, "chemieunfall": 2.0, "gasleck": 1.5,
    "stromausfall": 1.0, "blackout": 1.5,
    "kindesmisshandlung": 2.5, "sexualdelikt": 2.0, "vergewaltigung": 2.5,
    "entführung": 2.5, "vermisstes kind": 3.0,
    "drogenrazzia": 1.5, "razzia": 1.5, "durchsuchung": 1.0,
}


def compute_scoop_score(title: str, description: str = "") -> float:
    """
    Scoop-Score basierend auf dem neuen einheitlichen Scoring-Modul.
    Wird nur als Initialscore verwendet — insert_event() rechnet mit BILD-Info nach.
    """
    import scoring
    return scoring.compute_scoop_score(
        title=title,
        description=description,
        bild_overlap_type="none",  # Wird in insert_event nachberechnet
        zeitpunkt="",
        media_urls=[],
    )


# ---------------------------------------------------------------------------
# Presseportal-Fotos extrahieren
# ---------------------------------------------------------------------------

_photo_semaphore = asyncio.Semaphore(5)


async def _extract_presseportal_photos(
    link: str, session: aiohttp.ClientSession
) -> list[str]:
    """
    Fotos aus einem Presseportal-Artikel extrahieren.
    Holt og:image + Bilder aus dem Artikelbody.
    Filtert Logos/Icons raus, max 5 Fotos.
    """
    if not link or "presseportal.de" not in link:
        return []

    async with _photo_semaphore:
        try:
            async with session.get(
                link,
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"User-Agent": "NewsWatchdog/1.0"},
            ) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()
        except Exception:
            return []

    photos = []

    # og:image Meta-Tag
    og_match = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html)
    if og_match:
        og_url = og_match.group(1)
        if _is_real_photo(og_url):
            photos.append(og_url)

    # Bilder im Artikelbody
    # Presseportal nutzt <img> Tags im .story-content Bereich
    body_match = re.search(
        r'class="[^"]*story[^"]*"[^>]*>(.*?)</(?:article|div class="footer)',
        html,
        re.DOTALL,
    )
    if body_match:
        body_html = body_match.group(1)
        img_matches = re.findall(r'<img[^>]+src="([^"]+)"', body_html)
        for img_url in img_matches:
            if _is_real_photo(img_url) and img_url not in photos:
                photos.append(img_url)
                if len(photos) >= 5:
                    break

    return photos[:5]


def _is_real_photo(url: str) -> bool:
    """Filtert Logos, Icons, Dummy-Bilder und Stock-Grafiken raus.
    Nur echte Einsatzfotos/Fahndungsbilder durchlassen."""
    url_lower = url.lower()

    # Generische Skip-Patterns
    skip_patterns = [
        "logo", "icon", "favicon", "avatar", "banner",
        "tracking", "pixel", "spacer", "button",
        "1x1", "widget", "badge",
    ]
    if any(p in url_lower for p in skip_patterns):
        return False

    # Nur gängige Bildformate
    if not re.search(r"\.(jpg|jpeg|png|webp)", url_lower):
        return False

    # Presseportal Dummy-Bild
    if "presseportal.de/some-default" in url_lower:
        return False

    # Presseportal thumbnail/small = fast immer Behörden-Logos (PI_Stade, PP_SH etc.)
    if "thumbnail/small/" in url_lower:
        return False

    # Bekannte Stock/Kampagnen-Bilder der Polizei (kein echtes Einsatzfoto)
    stock_filenames = [
        "blaulicht", "riegelvor", "riegel_vor", "praevention",
        "waffenverbotszone", "aktenzeichenxy", "codierung",
        "personalwerber", "wiege-aktion", "einbruchschutz",
        "sicherheitstipp", "polizeistern", "erledigt_",
    ]
    if any(s in url_lower for s in stock_filenames):
        return False

    return True


# ---------------------------------------------------------------------------
# Event-Typ bestimmen
# ---------------------------------------------------------------------------

def classify_event_type(title: str, description: str = "") -> str:
    """Event-Typ anhand von Keywords klassifizieren."""
    text = f"{title} {description}".lower()

    fire_kw = ["brand", "feuer", "flammen", "rauch", "feuerwehr", "großbrand", "dachstuhl",
               "explosion", "verpuffung"]
    accident_kw = ["unfall", "verkehrsunfall", "kollision", "zusammenstoß", "karambolage",
                   "auffahrunfall", "vu ", "schwerer unfall"]
    weather_kw = ["unwetter", "sturm", "hochwasser", "überschwemmung", "hagel", "tornado",
                  "starkregen", "orkan"]
    rescue_kw = ["rettung", "vermisst", "suche", "bergung", "wasserrettung", "ertrunken"]

    if any(kw in text for kw in fire_kw):
        return "feuer"
    if any(kw in text for kw in accident_kw):
        return "unfall"
    if any(kw in text for kw in weather_kw):
        return "unwetter"
    if any(kw in text for kw in rescue_kw):
        return "rettung"

    return "polizei"


# ---------------------------------------------------------------------------
# Duplikat-Prüfung (Titel-Ähnlichkeit)
# ---------------------------------------------------------------------------

_recent_titles: list[str] = []
MAX_RECENT = 1000


def is_duplicate_title(title: str, threshold: float = 0.85) -> bool:
    """Prüfe ob ein ähnlicher Titel kürzlich verarbeitet wurde.
    Threshold erhöht auf 0.85 (vorher 0.75) — Polizeimeldungen haben oft
    ähnliche Struktur, das soll keine False Positives erzeugen."""
    title_clean = title.lower().strip()
    for existing in _recent_titles:
        if SequenceMatcher(None, title_clean, existing).ratio() >= threshold:
            return True
    return False


def register_title(title: str):
    """Titel in den Duplikat-Cache aufnehmen."""
    global _recent_titles
    _recent_titles.append(title.lower().strip())
    if len(_recent_titles) > MAX_RECENT:
        _recent_titles = _recent_titles[-MAX_RECENT:]


# ---------------------------------------------------------------------------
# Zeitpunkt-Parsing (robuster als vorher)
# ---------------------------------------------------------------------------

def _parse_time(entry: dict) -> str:
    """RSS-Zeitstempel in ISO-Format umwandeln.

    BUG-FIX: Die alte Version nutzte nur strptime mit 3 festen Formaten.
    presseportal.de liefert RFC 822 Daten die von strptime oft nicht korrekt
    geparst werden (z.B. wegen Locale-Problemen mit Monatsnamen).

    Neue Strategie:
    1. feedparser.published_parsed (struct_time) — meistens vorhanden und korrekt
    2. email.utils.parsedate_to_datetime — RFC 822 Standard-Parser
    3. Diverse strptime-Formate als Fallback
    4. Jetzt-Zeitpunkt als letzter Fallback
    """
    # Strategie 1: feedparser hat es schon geparst
    time_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if time_struct:
        try:
            dt = datetime(*time_struct[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    # Strategie 2: RFC 822 via email.utils (robustester Parser)
    published = entry.get("published", "") or entry.get("updated", "")
    if published:
        try:
            dt = parsedate_to_datetime(published)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

        # Strategie 3: strptime Fallbacks
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%d.%m.%Y %H:%M",
            "%d.%m.%Y %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(published.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# HTML-Bereinigung
# ---------------------------------------------------------------------------

def _clean_html(text: str) -> str:
    """HTML-Tags entfernen."""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Haupt-Scraper-Funktion
# ---------------------------------------------------------------------------

async def scrape_polizei_rss() -> list[dict]:
    """
    Alle konfigurierten Polizei-RSS-Feeds abrufen und Events extrahieren.
    Gibt eine Liste von Event-Dicts zurück.

    BUG-FIX: Vorherige Version gab 0 Events zurück weil:
    1. _parse_time nutzte nur String-Input statt feedparser.published_parsed
    2. Geocoding-Fallback griff nicht zuverlässig
    3. Duplikat-Threshold war zu niedrig (0.75)
    4. Kennung-Map war zu klein (nur 10 Städte)

    Jetzt: Robustes Time-Parsing, erweitertes Geocoding, Dienststellen-Fallback.
    """
    events = []
    feeds_ok = 0
    feeds_fail = 0
    entries_total = 0
    entries_geocoded = 0
    entries_skipped_age = 0
    entries_skipped_dup = 0
    entries_skipped_geo = 0

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=10)  # Max 10 gleichzeitige Verbindungen

    async with aiohttp.ClientSession(connector=connector) as session:

        # Feeds in Batches abrufen (nicht alle gleichzeitig — presseportal Rate Limit)
        feed_items = list(ALL_FEEDS.items())
        batch_size = 10

        for batch_start in range(0, len(feed_items), batch_size):
            batch = feed_items[batch_start:batch_start + batch_size]
            tasks = []
            for name, feed_id in batch:
                tasks.append(_fetch_feed(session, name, feed_id))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (name, feed_id), result in zip(batch, results):
                if isinstance(result, Exception):
                    log.warning("Feed %s Exception: %s", name, result)
                    feeds_fail += 1
                    continue

                if result is None:
                    feeds_fail += 1
                    continue

                feed = result
                feeds_ok += 1
                log.debug("Feed %s: %d Einträge", name, len(feed.entries))

                for entry in feed.entries[:20]:  # Max 20 pro Feed
                    entries_total += 1
                    title = entry.get("title", "").strip()
                    description = entry.get("description", "") or entry.get("summary", "") or ""
                    description = description.strip()
                    link = entry.get("link", "")

                    if not title:
                        continue

                    # Duplikat-Check
                    if is_duplicate_title(title):
                        entries_skipped_dup += 1
                        continue
                    register_title(title)

                    # Zeitpunkt parsen (BUG-FIX: entry-Dict statt nur String)
                    zeitpunkt = _parse_time(entry)

                    # Nur Events der letzten 24 Stunden
                    try:
                        dt = datetime.fromisoformat(zeitpunkt)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                        if age_h > 24:
                            entries_skipped_age += 1
                            continue
                        if age_h < 0:
                            # Zukunfts-Timestamp — wahrscheinlich Timezone-Problem, trotzdem nehmen
                            pass
                    except Exception as e:
                        log.debug("Zeitpunkt-Parse-Fehler: %s für '%s'", e, zeitpunkt)
                        # Bei Parsing-Fehler trotzdem weitermachen

                    # Geocoding — mehrstufiger Fallback mit Straßen-Präzision
                    location = extract_location(title, description)
                    coords = None
                    is_fallback_coords = False

                    # Stufe 0 (NEU): Straßen-genaue Geocodierung
                    street = extract_street_address(description)
                    if not street:
                        street = extract_street_address(title)

                    if street and location:
                        coords = await geocode_address(street, location, session)
                    elif street:
                        dienststelle_city = _extract_city_from_dienststelle(name)
                        if dienststelle_city:
                            coords = await geocode_address(street, dienststelle_city, session)

                    # Stufe 1: Location aus Titel/Beschreibung + Fallback-Tabelle
                    if not coords and location:
                        coords = fallback_geocode(location)
                        if coords:
                            is_fallback_coords = True

                    # Stufe 2: Direkt im kombinierten Text suchen
                    if not coords:
                        coords = fallback_geocode(f"{title} {description}")
                        if coords:
                            is_fallback_coords = True

                    # Stufe 3: Stadt aus dem Dienststellennamen ableiten
                    if not coords:
                        dienststelle_city = _extract_city_from_dienststelle(name)
                        if dienststelle_city:
                            coords = fallback_geocode(dienststelle_city)
                            if coords:
                                is_fallback_coords = True

                    # Stufe 4: Nominatim mit Stadtname (letzter Fallback)
                    if not coords and location:
                        coords = await geocode_city(location, session)

                    if not coords:
                        entries_skipped_geo += 1
                        log.debug("Kein Geocoding für [%s]: %s", name, title[:60])
                        continue

                    entries_geocoded += 1
                    lat, lon = coords

                    # Jitter für Fallback-Koordinaten (±200m), damit
                    # Events in gleicher Stadt nicht übereinander stapeln
                    if is_fallback_coords:
                        lat += random.uniform(-0.002, 0.002)
                        lon += random.uniform(-0.002, 0.002)

                    # Score erst berechnen, Fotos nur bei relevanten Events holen
                    pre_score = compute_scoop_score(title, description)
                    photos = []
                    if pre_score >= 1.5:
                        photos = await _extract_presseportal_photos(link, session)

                    # Event zusammenbauen
                    event = {
                        "titel": title,
                        "typ": classify_event_type(title, description),
                        "lat": lat,
                        "lon": lon,
                        "zeitpunkt": zeitpunkt,
                        "scoop_score": pre_score,
                        "quellen": [name],
                        "zusammenfassung": _clean_html(description)[:500],
                        "media_urls": photos,
                        "link": link,
                        "quelle_typ": "polizei_rss",
                    }
                    events.append(event)

            # Kleine Pause zwischen Batches (presseportal Rate Limit)
            if batch_start + batch_size < len(feed_items):
                await asyncio.sleep(0.5)

    log.info(
        "Polizei RSS: %d Feeds OK, %d Feeds fehlgeschlagen, "
        "%d Einträge gesamt, %d geocoded, "
        "%d zu alt, %d Duplikate, %d kein Geo → %d Events",
        feeds_ok, feeds_fail, entries_total,
        entries_geocoded, entries_skipped_age,
        entries_skipped_dup, entries_skipped_geo,
        len(events),
    )
    return events


async def _fetch_feed(session: aiohttp.ClientSession, name: str, feed_id: int) -> Optional[feedparser.FeedParserDict]:
    """Einzelnen Feed abrufen und parsen."""
    url = RSS_URL_TEMPLATE.format(feed_id=feed_id)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                log.warning("Feed %s HTTP %d", name, resp.status)
                return None
            # BUG-FIX: Content als bytes lesen, nicht als text.
            # feedparser kommt mit bytes+encoding besser klar als mit
            # potentiell falsch dekodiertem text.
            content = await resp.read()
    except Exception as e:
        log.warning("Feed %s Fehler: %s", name, e)
        return None

    return feedparser.parse(content)


# ---------------------------------------------------------------------------
# Statistiken (für Dashboard / Monitoring)
# ---------------------------------------------------------------------------

def get_feed_stats() -> dict:
    """Gibt Statistiken über die konfigurierten Feeds zurück."""
    return {
        "polizei_feeds": len(POLIZEI_FEEDS),
        "feuerwehr_feeds": len(FEUERWEHR_FEEDS),
        "thw_feeds": len(THW_FEEDS),
        "rettung_feeds": len(RETTUNG_FEEDS),
        "zoll_feeds": len(ZOLL_FEEDS),
        "total_feeds": len(ALL_FEEDS),
        "geocode_cache_size": len(_geocode_cache),
        "duplicate_cache_size": len(_recent_titles),
        "fallback_cities": len(CITY_FALLBACK),
    }
