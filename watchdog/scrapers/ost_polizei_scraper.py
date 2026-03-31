#!/usr/bin/env python3
"""
Scraper: Polizei-Pressemeldungen Ost-Deutschland

Berlin, Brandenburg, Sachsen und Sachsen-Anhalt haben presseportal.de
verlassen und bieten keine funktionierenden RSS-Feeds an.
Dieser Scraper holt die Meldungen direkt von den offiziellen
Polizei-Webseiten per HTML-Scraping.

Quellen:
  - berlin.de/polizei/polizeimeldungen/
  - polizei.brandenburg.de/pressemeldungen
  - polizei.sachsen.de/de/polizeiticker-*  (5 PDs)
  - sachsen-anhalt.de/bs/pressemitteilungen/polizei
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

log = logging.getLogger("watchdog.ost_polizei")

# ---------------------------------------------------------------------------
# Duplikat-Cache
# ---------------------------------------------------------------------------

_recent_titles: list[str] = []
MAX_RECENT = 500


def _is_duplicate(title: str, threshold: float = 0.85) -> bool:
    title_clean = title.lower().strip()[:200]
    for existing in _recent_titles:
        if SequenceMatcher(None, title_clean, existing).ratio() >= threshold:
            return True
    return False


def _register_title(title: str):
    global _recent_titles
    _recent_titles.append(title.lower().strip()[:200])
    if len(_recent_titles) > MAX_RECENT:
        _recent_titles = _recent_titles[-MAX_RECENT:]


# ---------------------------------------------------------------------------
# Geocoding Fallback (nur Ost-Deutschland)
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    "berlin": (52.5200, 13.4050),
    "potsdam": (52.3906, 13.0645),
    "cottbus": (51.7563, 14.3329),
    "frankfurt (oder)": (52.3471, 14.5506),
    "brandenburg": (52.4125, 12.5316),
    "oranienburg": (52.7551, 13.2364),
    "bernau": (52.6788, 13.5872),
    "eberswalde": (52.8342, 13.8212),
    "neuruppin": (52.9220, 12.8034),
    "luckenwalde": (52.0893, 13.1686),
    "königs wusterhausen": (52.3018, 13.6310),
    "fürstenwalde": (52.3586, 14.0637),
    "schwedt": (53.0572, 14.2831),
    "prenzlau": (53.3164, 13.8621),
    "senftenberg": (51.5255, 14.0020),
    "finsterwalde": (51.6352, 13.7070),
    "rathenow": (52.6066, 12.3393),
    "nauen": (52.6085, 12.8789),
    "bad belzig": (52.1395, 12.5896),
    "ludwigsfelde": (52.3019, 13.2542),
    "teltow": (52.4032, 13.2668),
    "potsdam-mittelmark": (52.3000, 12.9000),
    "falkensee": (52.5601, 13.0935),
    "henningsdorf": (52.6366, 13.2054),
    "schönefeld": (52.3906, 13.5186),
    "wildau": (52.3219, 13.6281),
    "tempelhof": (52.4681, 13.3854),
    "kreuzberg": (52.4989, 13.4038),
    "neukölln": (52.4812, 13.4353),
    "mitte": (52.5200, 13.4050),
    "charlottenburg": (52.5166, 13.3044),
    "spandau": (52.5345, 13.1963),
    "reinickendorf": (52.5689, 13.3325),
    "pankow": (52.5704, 13.4099),
    "lichtenberg": (52.5162, 13.4970),
    "marzahn": (52.5449, 13.5641),
    "hellersdorf": (52.5369, 13.6035),
    "treptow": (52.4927, 13.4428),
    "köpenick": (52.4474, 13.5752),
    "steglitz": (52.4569, 13.3215),
    "zehlendorf": (52.4342, 13.2586),
    "wedding": (52.5499, 13.3549),
    "prenzlauer berg": (52.5389, 13.4221),
    "friedrichshain": (52.5149, 13.4538),
    "schöneberg": (52.4835, 13.3494),
    "wilmersdorf": (52.4867, 13.3166),
    "dahlem": (52.4573, 13.2883),
    # Havelland / Oberhavel / Barnim
    "strausberg": (52.5790, 13.8877),
    "werneuchen": (52.6319, 13.7383),
    "bad freienwalde": (52.7879, 14.0311),
    # Sachsen
    "dresden": (51.0504, 13.7373),
    "leipzig": (51.3397, 12.3731),
    "chemnitz": (50.8278, 12.9214),
    "zwickau": (50.7184, 12.4964),
    "görlitz": (51.1528, 14.9878),
    "plauen": (50.4953, 12.1381),
    "meißen": (51.1634, 13.4737),
    "bautzen": (51.1814, 14.4244),
    "freiberg": (50.9119, 13.3428),
    "pirna": (50.9613, 13.9398),
    "riesa": (51.3062, 13.2914),
    "radebeul": (51.1042, 13.6516),
    "freital": (51.0082, 13.6488),
    "döbeln": (51.1213, 13.1186),
    "grimma": (51.2366, 12.7277),
    "delitzsch": (51.5255, 12.3428),
    "torgau": (51.5601, 13.0045),
    "eilenburg": (51.4614, 12.6362),
    "hoyerswerda": (51.4364, 14.2360),
    "zittau": (50.8981, 14.8078),
    "kamenz": (51.2690, 14.0929),
    "markkleeberg": (51.2777, 12.3673),
    "bannewitz": (51.0058, 13.7179),
    "sächsische schweiz": (50.9243, 14.1456),
    "osterzgebirge": (50.8500, 13.7000),
    # Sachsen-Anhalt
    "magdeburg": (52.1205, 11.6276),
    "halle": (51.4828, 11.9700),
    "halle (saale)": (51.4828, 11.9700),
    "dessau": (51.8313, 12.2442),
    "dessau-roßlau": (51.8313, 12.2442),
    "wittenberg": (51.8661, 12.6489),
    "stendal": (52.6058, 11.8587),
    "halberstadt": (51.8961, 11.0561),
    "wernigerode": (51.8339, 10.7843),
    "quedlinburg": (51.7874, 11.1490),
    "salzwedel": (52.8526, 11.1532),
    "merseburg": (51.3537, 11.9929),
    "naumburg": (51.1527, 11.8100),
    "weißenfels": (51.2010, 11.9690),
    "sangerhausen": (51.4728, 11.2997),
    "bernburg": (51.7951, 11.7384),
    "aschersleben": (51.7555, 11.4617),
    "schönebeck": (52.0167, 11.7396),
    "wolfen": (51.6614, 12.2767),
    "bitterfeld": (51.6247, 12.3288),
    "burg": (52.2736, 11.8558),
    "gardelegen": (52.5258, 11.3929),
    "börde": (52.1100, 11.4600),
    "benneckenstein": (51.6586, 10.7178),
    "anhalt-bitterfeld": (51.6500, 12.3000),
    "saalekreis": (51.4000, 11.9500),
    "mansfeld-südharz": (51.5000, 11.3000),
    "burgenlandkreis": (51.1800, 11.9000),
    "salzlandkreis": (51.8500, 11.6500),
    "jerichower land": (52.2500, 11.9500),
    "altmarkkreis": (52.8000, 11.1000),
    "harz": (51.8000, 10.9000),
}


def _geocode_text(text: str) -> Optional[tuple[float, float]]:
    """Versuche Koordinaten aus dem Text zu extrahieren."""
    text_lower = text.lower()
    # Längere Namen zuerst
    for city in sorted(CITY_COORDS.keys(), key=len, reverse=True):
        if city in text_lower:
            return CITY_COORDS[city]
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_scoop_score(title: str, description: str = "") -> float:
    import scoring
    return scoring.compute_scoop_score(
        title=title,
        description=description,
        bild_overlap_type="none",
        zeitpunkt="",
        media_urls=[],
    )


def _classify_type(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["brand", "feuer", "flammen", "rauch", "großbrand"]):
        return "feuer"
    if any(kw in text_lower for kw in ["unfall", "verkehrsunfall", "kollision", "karambolage"]):
        return "unfall"
    if any(kw in text_lower for kw in ["unwetter", "sturm", "hochwasser"]):
        return "unwetter"
    if any(kw in text_lower for kw in ["vermisst", "rettung", "suche"]):
        return "rettung"
    return "polizei"


# ---------------------------------------------------------------------------
# HTML bereinigen
# ---------------------------------------------------------------------------

def _clean_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<br\s*/?>", " ", text)
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Berlin: berlin.de/polizei/polizeimeldungen/
# ---------------------------------------------------------------------------

async def _scrape_berlin(session: aiohttp.ClientSession) -> list[dict]:
    """
    Berlin Polizeimeldungen von berlin.de scrapen.

    Struktur: <ul> mit <li> Einträgen.
    Jede <li> hat Datum als Text und einen <a> mit Titel+Link.
    """
    events = []
    url = "https://www.berlin.de/polizei/polizeimeldungen/"

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
                log.warning("Berlin Polizei HTTP %d", resp.status)
                return []
            html = await resp.text()
    except Exception as e:
        log.warning("Berlin Polizei Fehler: %s", e)
        return []

    # CET = UTC+1 (Winterzeit), CEST = UTC+2 (Sommerzeit)
    # Berlin-Zeiten sind immer CET/CEST — wir nehmen UTC+1 als Annäherung
    CET = timezone(timedelta(hours=1))

    # Jede <li> enthält:
    #   <div class="cell nowrap date">16.03.2026 09:50 Uhr</div>
    #   <div class="cell text"><a href="...">Titel</a><span class="category">Bezirk</span></div>
    li_pattern = re.compile(
        r'<li>\s*<div[^>]*class="[^"]*date[^"]*"[^>]*>'
        r'(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})\s*Uhr</div>'
        r'\s*<div[^>]*class="[^"]*text[^"]*"[^>]*>'
        r'\s*<a\s+href="([^"]+)"[^>]*>([^<]+)</a>'
        r'(?:\s*<span[^>]*class="[^"]*category[^"]*"[^>]*>([^<]*)</span>)?',
        re.DOTALL,
    )

    for match in li_pattern.finditer(html):
        date_str = match.group(1).strip()
        link_path = match.group(2).strip()
        title = _clean_html(match.group(3).strip())
        bezirk = _clean_html(match.group(4).strip()) if match.group(4) else ""

        if not title or len(title) < 10:
            continue

        # Zeitpunkt parsen (CET → UTC)
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y %H:%M")
            dt = dt.replace(tzinfo=CET)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            if age_h > 24:
                continue
            zeitpunkt = dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            zeitpunkt = datetime.now(timezone.utc).isoformat()

        # Duplikat-Check
        if _is_duplicate(title):
            continue
        _register_title(title)

        # Link zusammenbauen
        if link_path.startswith("/"):
            link = f"https://www.berlin.de{link_path}"
        else:
            link = link_path

        # Geocoding: Straße → Bezirk → Titel → Berlin-Mitte
        coords = None
        is_fallback = False
        street = extract_street_address(title)
        if street:
            coords = await geocode_address(street, "Berlin", session)
        if not coords and bezirk:
            coords = _geocode_text(bezirk)
            if coords:
                is_fallback = True
        if not coords:
            coords = _geocode_text(title)
            if coords:
                is_fallback = True
        if not coords:
            coords = CITY_COORDS["berlin"]
            is_fallback = True

        lat, lon = coords
        if is_fallback:
            lat += random.uniform(-0.002, 0.002)
            lon += random.uniform(-0.002, 0.002)

        # Zusammenfassung mit Bezirk
        zusammenfassung = f"{title} ({bezirk})" if bezirk else title

        event = {
            "titel": title[:200],
            "typ": _classify_type(title),
            "lat": lat,
            "lon": lon,
            "zeitpunkt": zeitpunkt,
            "scoop_score": _compute_scoop_score(title),
            "quellen": ["Polizei Berlin"],
            "zusammenfassung": zusammenfassung[:500],
            "media_urls": [],
            "link": link,
            "quelle_typ": "ost_polizei",
        }
        events.append(event)

    log.info("Berlin Polizei: %d Meldungen extrahiert", len(events))
    return events


# ---------------------------------------------------------------------------
# Brandenburg: polizei.brandenburg.de/pressemeldungen
# ---------------------------------------------------------------------------

async def _scrape_brandenburg(session: aiohttp.ClientSession) -> list[dict]:
    """
    Brandenburg Polizei-Pressemeldungen scrapen.

    Struktur: <a> Blöcke mit href="/pressemeldung/SLUG/ID",
    Titel und Datum im Link-Inneren.
    """
    events = []
    url = "https://polizei.brandenburg.de/pressemeldungen"

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
                log.warning("Brandenburg Polizei HTTP %d", resp.status)
                return []
            html = await resp.text()
    except Exception as e:
        log.warning("Brandenburg Polizei Fehler: %s", e)
        return []

    CET = timezone(timedelta(hours=1))

    # Pre-Pass: Location (PD - Stadt/Kreis) pro Link-Pfad extrahieren
    # Struktur: <span>(PD\n-\nStadt)</span><a href="/pressemeldung/SLUG/ID">
    loc_pattern = re.compile(
        r'<span>\((\w[\w\s/äöüÄÖÜß-]*?)\s*-\s*(\w[\w\s/äöüÄÖÜß-]*?)\)\s*'
        r'<span/>\s*<a\s+href="(/pressemeldung/[^"]+)"',
        re.DOTALL,
    )
    link_to_location: dict[str, tuple[str, str]] = {}
    for m in loc_pattern.finditer(html):
        pd_name = m.group(1).strip()
        city_or_kreis = m.group(2).strip()
        lpath = m.group(3).strip()
        link_to_location[lpath] = (pd_name, city_or_kreis)

    # Titel + Datum extrahieren
    h4_pattern = re.compile(
        r'<h4>\s*<span>(\d{2}\.\d{2}\.\d{4})</span>\s*,\s*'
        r'<a\s+href="(/pressemeldung/[^"]+)"[^>]*>([^<]+)</a>\s*</h4>',
        re.DOTALL,
    )

    for match in h4_pattern.finditer(html):
        date_str = match.group(1).strip()
        link_path = match.group(2).strip()
        title = _clean_html(match.group(3).strip())

        if not title or len(title) < 5:
            continue

        # Zeitpunkt parsen (CET → UTC)
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
            dt = dt.replace(hour=12, tzinfo=CET)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            if age_h > 24:
                continue
            zeitpunkt = dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            zeitpunkt = datetime.now(timezone.utc).isoformat()

        # Duplikat-Check
        if _is_duplicate(title):
            continue
        _register_title(title)

        # Link zusammenbauen
        link = f"https://polizei.brandenburg.de{link_path}"

        # Location aus Pre-Pass (PD, Stadt/Kreis)
        loc_pd, loc_city = link_to_location.get(link_path, ("", ""))

        # Geocoding: Straße → PD-Stadt → Titel → URL-Slug → PD-Fallback → Potsdam
        coords = None
        is_fallback = False
        street = extract_street_address(title)
        if street:
            geo_city = loc_city or loc_pd or None
            if geo_city:
                coords = await geocode_address(street, geo_city, session)
            if not coords:
                # Stadt aus Titel extrahieren
                for city_name in sorted(CITY_COORDS.keys(), key=len, reverse=True):
                    if city_name in title.lower():
                        coords = await geocode_address(street, city_name.title(), session)
                        break
        if not coords:
            # PD-Stadt als Fallback (viel besser als Potsdam)
            geo_text = f"{loc_pd} {loc_city} {title}"
            coords = _geocode_text(geo_text)
            if coords:
                is_fallback = True
        if not coords:
            slug = link_path.split("/")[2] if len(link_path.split("/")) > 2 else ""
            coords = _geocode_text(slug.replace("-", " "))
            if coords:
                is_fallback = True
        if not coords:
            # Letzer Fallback: PD-Stadt oder Potsdam
            fallback_key = loc_pd.lower() if loc_pd else "potsdam"
            coords = CITY_COORDS.get(fallback_key, CITY_COORDS["potsdam"])
            is_fallback = True

        lat, lon = coords
        if is_fallback:
            lat += random.uniform(-0.005, 0.005)
            lon += random.uniform(-0.005, 0.005)

        # Quelle mit PD anreichern
        quelle = f"Polizei Brandenburg ({loc_pd})" if loc_pd else "Polizei Brandenburg"

        event = {
            "titel": title[:200],
            "typ": _classify_type(title),
            "lat": lat,
            "lon": lon,
            "zeitpunkt": zeitpunkt,
            "scoop_score": _compute_scoop_score(title),
            "quellen": [quelle],
            "zusammenfassung": title[:500],
            "media_urls": [],
            "link": link,
            "quelle_typ": "ost_polizei",
        }
        events.append(event)

    log.info("Brandenburg Polizei: %d Meldungen extrahiert", len(events))
    return events


# ---------------------------------------------------------------------------
# Deutsche Monatsnamen (für Sachsen-Ticker)
# ---------------------------------------------------------------------------

_GERMAN_MONTHS: dict[str, int] = {
    "januar": 1, "februar": 2, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}


# ---------------------------------------------------------------------------
# Sachsen: polizei.sachsen.de Polizeiticker (5 PDs)
# ---------------------------------------------------------------------------

SACHSEN_PDS: list[tuple[str, str, str]] = [
    ("Dresden", "dresden",
     "https://www.polizei.sachsen.de/de/polizeiticker-polizeibericht-polizeidirektion-dresden-34585.html"),
    ("Leipzig", "leipzig",
     "https://www.polizei.sachsen.de/de/polizeiticker-polizeibericht-polizeidirektion-leipzig-34578.html"),
    ("Chemnitz", "chemnitz",
     "https://www.polizei.sachsen.de/de/polizeiticker-polizeibericht-polizeidirektion-chemnitz-34572.html"),
    ("Görlitz", "görlitz",
     "https://www.polizei.sachsen.de/de/polizeiticker-polizeibericht-polizeidirektion-gorlitz-34566.html"),
    ("Zwickau", "zwickau",
     "https://www.polizei.sachsen.de/de/polizeiticker-polizeibericht-polizeidirektion-zwickau-34591.html"),
]


async def _scrape_sachsen_pd(
    session: aiohttp.ClientSession, pd_name: str, fallback_city: str, url: str,
) -> list[dict]:
    """Sachsen Polizeiticker für eine PD scrapen.

    Struktur:
      <div id="a-XXXXX" class="row">
        <h2>DD. Monat | Titel</h2>
        <div class="clearfix"></div>
        <div class="text-col"><p>Body</p></div>
      </div>
    """
    events = []
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
                log.warning("Sachsen %s HTTP %d", pd_name, resp.status)
                return []
            html = await resp.text()
    except Exception as e:
        log.warning("Sachsen %s Fehler: %s", pd_name, e)
        return []

    CET = timezone(timedelta(hours=1))
    now = datetime.now(timezone.utc)
    current_year = now.year

    # Pattern: <h2>DD. Monat [YYYY] | Titel</h2> (Jahr optional, Separator | oder I)
    entry_pattern = re.compile(
        r'<div\s+id="a-\d+"\s+class="row">\s*'
        r'<h2>(\d{1,2})\.\s*(\w+)(?:\s+\d{4})?\s*[|I]\s*(.+?)</h2>'
        r'.*?<div\s+class="text-col">\s*(.*?)</div>\s*</div>',
        re.DOTALL,
    )

    for match in entry_pattern.finditer(html):
        day_str = match.group(1)
        month_str = match.group(2).lower()
        title = _clean_html(match.group(3).strip())
        body_html = match.group(4)

        if not title or len(title) < 10:
            continue

        month_num = _GERMAN_MONTHS.get(month_str)
        if not month_num:
            continue

        # Zeitpunkt: Tag + Monat → aktuelles Jahr, 12:00 CET
        try:
            dt = datetime(current_year, month_num, int(day_str), 12, 0, tzinfo=CET)
            age_h = (now - dt).total_seconds() / 3600
            if age_h > 24:
                continue
            zeitpunkt = dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue

        if _is_duplicate(title):
            continue
        _register_title(title)

        # Body-Text (erste 500 Zeichen)
        body = _clean_html(body_html)[:500]

        # Geocoding
        coords = None
        is_fallback = False
        street = extract_street_address(f"{title} {body[:200]}")
        if street:
            city_hint = None
            for city_name in sorted(CITY_COORDS.keys(), key=len, reverse=True):
                if city_name in (title + " " + body[:200]).lower():
                    city_hint = city_name.title()
                    break
            if city_hint:
                coords = await geocode_address(street, city_hint, session)
        if not coords:
            coords = _geocode_text(title + " " + body[:200])
            if coords:
                is_fallback = True
        if not coords:
            coords = CITY_COORDS.get(fallback_city, (51.05, 13.74))
            is_fallback = True

        lat, lon = coords
        if is_fallback:
            lat += random.uniform(-0.002, 0.002)
            lon += random.uniform(-0.002, 0.002)

        event = {
            "titel": title[:200],
            "typ": _classify_type(title + " " + body[:100]),
            "lat": lat,
            "lon": lon,
            "zeitpunkt": zeitpunkt,
            "scoop_score": _compute_scoop_score(title, body[:200]),
            "quellen": [f"Polizei Sachsen PD {pd_name}"],
            "zusammenfassung": body[:500] if body else title[:500],
            "media_urls": [],
            "link": url,
            "quelle_typ": "ost_polizei",
        }
        events.append(event)

    log.info("Sachsen PD %s: %d Meldungen extrahiert", pd_name, len(events))
    return events


async def _scrape_sachsen(session: aiohttp.ClientSession) -> list[dict]:
    """Alle 5 Sachsen PDs parallel scrapen."""
    tasks = [
        _scrape_sachsen_pd(session, name, city, url)
        for name, city, url in SACHSEN_PDS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    events = []
    for result in results:
        if isinstance(result, Exception):
            log.error("Sachsen PD Exception: %s", result)
            continue
        events.extend(result)
    log.info("Sachsen gesamt: %d Meldungen", len(events))
    return events


# ---------------------------------------------------------------------------
# Sachsen-Anhalt: sachsen-anhalt.de/bs/pressemitteilungen/polizei
# ---------------------------------------------------------------------------

async def _scrape_sachsen_anhalt(session: aiohttp.ClientSession) -> list[dict]:
    """Sachsen-Anhalt Polizei-Pressemeldungen scrapen.

    Struktur (TYPO3 RSS-Include):
      <div class="tx-rssdisplay-newslist">
        <div class="tx-rssdisplay-item-meta">
          <div class="tx-rssdisplay-item-meta-date">DD.MM.YYYY<br/>REF/YYYY</div>
          <div class="tx-rssdisplay-item-meta-header">
            <p>Polizeirevier/Inspektion Name</p>
            <h2><a href="URL">Titel</a></h2>
          </div>
        </div>
        <div class="tx-rssdisplay-item-content">Body-Preview<p><a>weiterlesen</a></p></div>
      </div>
    """
    events = []
    url = "https://www.sachsen-anhalt.de/bs/pressemitteilungen/polizei"

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "de-DE,de;q=0.9",
            },
        ) as resp:
            if resp.status != 200:
                log.warning("Sachsen-Anhalt Polizei HTTP %d", resp.status)
                return []
            html = await resp.text()
    except Exception as e:
        log.warning("Sachsen-Anhalt Polizei Fehler: %s", e)
        return []

    CET = timezone(timedelta(hours=1))

    # Jeder Eintrag ist ein tx-rssdisplay-newslist Block
    entry_pattern = re.compile(
        r'<div\s+class="tx-rssdisplay-newslist">\s*'
        r'<div\s+class="tx-rssdisplay-item-meta">\s*'
        r'<div\s+class="tx-rssdisplay-item-meta-date">\s*'
        r'(\d{2}\.\d{2}\.\d{4})\s*<br\s*/?>.*?</div>\s*'
        r'<div\s+class="tx-rssdisplay-item-meta-header">\s*'
        r'<p>\s*(.*?)\s*</p>\s*'
        r'<h2>\s*(?:<a\s+href="([^"]+)"[^>]*>)?\s*(.*?)\s*(?:</a>)?\s*</h2>'
        r'.*?<div\s+class="tx-rssdisplay-item-content">\s*(.*?)</div>\s*</div>',
        re.DOTALL,
    )

    for match in entry_pattern.finditer(html):
        date_str = match.group(1).strip()
        revier = _clean_html(match.group(2).strip())
        link_path = match.group(3) or ""
        title = _clean_html(match.group(4).strip())
        body_html = match.group(5)

        # Titel oft generisch ("Kriminalitätslage") → mit Revier anreichern
        if not title or len(title) < 5:
            continue

        # Zeitpunkt parsen
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
            dt = dt.replace(hour=12, tzinfo=CET)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            if age_h > 24:
                continue
            zeitpunkt = dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            zeitpunkt = datetime.now(timezone.utc).isoformat()

        # Body bereinigen (ohne "weiterlesen" Link)
        body = _clean_html(re.sub(r'<p>\s*<a[^>]*>weiterlesen</a>\s*</p>', '', body_html))
        body = body[:500]

        # Zusammengesetzter Titel wenn generisch
        display_title = title
        if title.lower() in ("kriminalitätslage", "verkehrslage", "berichtszeitraum"):
            display_title = f"{revier}: {title}"
        elif revier and revier.lower() not in title.lower():
            display_title = f"{title} ({revier})"

        # Duplikat-Check auf zusammengesetzten Titel
        dedup_key = f"{revier} {date_str} {title}"
        if _is_duplicate(dedup_key):
            continue
        _register_title(dedup_key)

        # Link
        if link_path:
            if link_path.startswith("/"):
                link = f"https://www.sachsen-anhalt.de{link_path}"
            else:
                link = link_path
            link = link.replace("&amp;", "&")
        else:
            link = url

        # Geocoding: Revier-Name → Body-Text → Magdeburg-Fallback
        coords = None
        is_fallback = False
        search_text = f"{revier} {title} {body[:200]}"
        street = extract_street_address(search_text)
        if street:
            city_hint = None
            for city_name in sorted(CITY_COORDS.keys(), key=len, reverse=True):
                if city_name in search_text.lower():
                    city_hint = city_name.title()
                    break
            if city_hint:
                coords = await geocode_address(street, city_hint, session)
        if not coords:
            coords = _geocode_text(search_text)
            if coords:
                is_fallback = True
        if not coords:
            coords = CITY_COORDS["magdeburg"]
            is_fallback = True

        lat, lon = coords
        if is_fallback:
            lat += random.uniform(-0.002, 0.002)
            lon += random.uniform(-0.002, 0.002)

        event = {
            "titel": display_title[:200],
            "typ": _classify_type(display_title + " " + body[:100]),
            "lat": lat,
            "lon": lon,
            "zeitpunkt": zeitpunkt,
            "scoop_score": _compute_scoop_score(display_title, body[:200]),
            "quellen": [f"Polizei Sachsen-Anhalt ({revier})"],
            "zusammenfassung": body[:500] if body else display_title[:500],
            "media_urls": [],
            "link": link,
            "quelle_typ": "ost_polizei",
        }
        events.append(event)

    log.info("Sachsen-Anhalt Polizei: %d Meldungen extrahiert", len(events))
    return events


# ---------------------------------------------------------------------------
# Haupt-Scraper-Funktion
# ---------------------------------------------------------------------------

async def scrape_ost_polizei() -> list[dict]:
    """
    Polizeimeldungen aus Ost-Deutschland scrapen.

    Abgedeckt: Berlin, Brandenburg, Sachsen (5 PDs), Sachsen-Anhalt.
    """
    events = []

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=8)

    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            _scrape_berlin(session),
            _scrape_brandenburg(session),
            _scrape_sachsen(session),
            _scrape_sachsen_anhalt(session),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                log.error("Ost-Polizei Scraper Exception: %s", result)
                continue
            events.extend(result)

    log.info("Ost-Polizei gesamt: %d Events (BE+BB+SN+ST)", len(events))
    return events
