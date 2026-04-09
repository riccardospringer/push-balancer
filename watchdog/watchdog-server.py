#!/usr/bin/env python3
"""
News Watchdog Backend — FastAPI Server auf Port 8090
Echtzeit-Aggregation regionaler Breaking News auf einer interaktiven Karte.

Starten: uvicorn watchdog-server:app --host 0.0.0.0 --port 8090 --reload
Oder:    python3 watchdog-server.py
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import bild_checker
import scoring
from boulevard_scorer import compute_boulevard_boost

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watchdog")

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "watchdog.db"

# ---------------------------------------------------------------------------
# SQLite Setup
# ---------------------------------------------------------------------------

def init_db():
    """Datenbank und Tabellen anlegen."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id              TEXT PRIMARY KEY,
            titel           TEXT NOT NULL,
            typ             TEXT NOT NULL DEFAULT 'sonstiges',
            lat             REAL NOT NULL,
            lon             REAL NOT NULL,
            zeitpunkt       TEXT NOT NULL,
            scoop_score     REAL NOT NULL DEFAULT 0.0,
            quellen         TEXT NOT NULL DEFAULT '[]',
            zusammenfassung TEXT NOT NULL DEFAULT '',
            media_urls      TEXT NOT NULL DEFAULT '[]',
            freshness       TEXT NOT NULL DEFAULT 'HOT',
            link            TEXT NOT NULL DEFAULT '',
            quelle_typ      TEXT NOT NULL DEFAULT 'unknown',
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_zeitpunkt ON events(zeitpunkt DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_typ ON events(typ)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_freshness ON events(freshness)
    """)
    conn.commit()
    conn.close()
    log.info("Datenbank initialisiert: %s", DB_PATH)


def get_db() -> sqlite3.Connection:
    """Neue DB-Verbindung holen (pro Request/Task)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Freshness Decay
# ---------------------------------------------------------------------------

FRESHNESS_THRESHOLDS = [
    (1,   "HOT"),      # < 1 Stunde
    (3,   "WARM"),     # < 3 Stunden
    (12,  "COOLING"),  # < 12 Stunden
]
# Alles darüber: COLD — wird bei insert_event() noch akzeptiert (bis 24h),
# aber update_all_freshness() löscht Events >24h aus der DB.


def compute_freshness(zeitpunkt_str: str) -> str:
    """Freshness-Status basierend auf dem Alter des Events berechnen."""
    try:
        if zeitpunkt_str.endswith("Z"):
            zeitpunkt_str = zeitpunkt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(zeitpunkt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        for threshold_h, label in FRESHNESS_THRESHOLDS:
            if age_hours < threshold_h:
                return label
        return "COLD"
    except Exception:
        return "COLD"


def update_all_freshness():
    """Freshness aller Events in der DB aktualisieren + alte Events löschen."""
    conn = get_db()
    try:
        # Events älter als 24h komplett löschen
        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        deleted = conn.execute("DELETE FROM events WHERE zeitpunkt < ?", (cutoff_24h,)).rowcount
        if deleted:
            log.info("Aufgeräumt: %d Events älter als 24h gelöscht", deleted)

        rows = conn.execute("SELECT id, zeitpunkt, freshness FROM events").fetchall()
        updates = []
        for row in rows:
            new_freshness = compute_freshness(row["zeitpunkt"])
            if new_freshness != row["freshness"]:
                updates.append((new_freshness, row["id"]))
        if updates:
            conn.executemany(
                "UPDATE events SET freshness = ?, updated_at = datetime('now') WHERE id = ?",
                updates
            )
        conn.commit()
        if updates:
            log.info("Freshness aktualisiert: %d Events geändert", len(updates))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Event-ID generieren (deterministisch für Duplikat-Check)
# ---------------------------------------------------------------------------

def make_event_id(titel: str, zeitpunkt: str, lat: float, lon: float) -> str:
    """Deterministisches ID-Hashing basierend auf Kernfeldern.

    Koordinaten auf 1 Dezimale gerundet (~11km), damit Geocoding-Jitter
    und Fallback-Varianz nicht zu Duplikaten fuehren.
    """
    raw = f"{titel.lower().strip()}|{zeitpunkt}|{lat:.1f}|{lon:.1f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Event in DB speichern
# ---------------------------------------------------------------------------

import re as _re

# ---------------------------------------------------------------------------
# Crime-Radar Regio-Filter: WHITELIST-Ansatz
# Nur Events die nach lokaler Blaulicht-Meldung aussehen dürfen rein.
# Alles andere (investigativ, national, international, redaktionell) → raus.
# ---------------------------------------------------------------------------

# Quellen die per Definition lokal + vertrauenswürdig sind
_TRUSTED_LOCAL_SOURCES = {"polizei_rss", "ost_polizei"}

# Blaulicht-Signalwörter: Wenn keins davon im Text → kein lokales Blaulicht-Event
_BLAULICHT_KEYWORDS = {
    # Polizei / Kriminalität
    "polizei", "polizeimeldung", "kriminalpolizei", "kripo", "mordkommission",
    "festnahme", "festgenommen", "verhaftet", "durchsuchung", "razzia",
    "fahndung", "tatverdächtig", "beschuldigt", "ermittlung",
    "diebstahl", "einbruch", "raub", "überfall", "raubüberfall",
    "körperverletzung", "schlägerei", "messerangriff", "messerattacke",
    "schüsse", "schusswaffe", "bedrohung", "nötigung",
    "sexualdelikt", "vergewaltigung", "belästigung",
    "betrug", "drogenhandel", "drogen", "btm", "rauschgift",
    "sachbeschädigung", "vandalismus", "brandstiftung",
    "mord", "totschlag", "tötung", "leiche", "leichenfund",
    "vermisst", "vermisste", "vermisstensuche",
    "flucht", "flüchtig", "geflüchtet", "unbekannt entkommen",
    "geiselnahme", "entführung", "amoklauf",
    "sek", "spezialeinsatzkommando", "großeinsatz", "großlage",
    # Verkehr
    "verkehrsunfall", "unfall", "kollision", "zusammenstoß",
    "schwerverletzt", "lebensgefährlich", "tödlich", "getötet", "tot",
    "fahrerflucht", "unfallflucht", "alkohol am steuer", "geisterfahrer",
    "vollsperrung",
    # Feuer / Rettung
    "brand", "feuer", "großbrand", "dachstuhlbrand", "wohnungsbrand",
    "explosion", "verpuffung", "rauchentwicklung",
    "feuerwehr", "feuerwehreinsatz", "rettungseinsatz",
    "rettungshubschrauber", "notarzt", "reanimation",
    "evakuierung", "gefahrgut", "gasleck", "gasaustritt",
    # Waffen / Sprengstoff
    "bombe", "sprengstoff", "kampfmittel", "munition", "granate",
    "bombenentschärfung", "blindgänger",
}

# Harte Blacklist: IMMER rausfiltern, egal was sonst im Text steht
_BLACKLIST_PATTERNS = [
    # International / Geopolitik
    r"\bisrael\b", r"\biran\b", r"\bgaza\b", r"\bukraine\b", r"\brussland\b",
    r"\bnato\b", r"\btrump\b", r"\bbiden\b", r"\bputin\b", r"\bselenskyj\b",
    r"\bnahost\b", r"\bkrieg\b", r"\bchina\b", r"\bpeking\b", r"\bmoskau\b",
    r"\bwashington\b", r"\bpentagon\b", r"\btaliban\b", r"\bhamas\b",
    r"\bhisbollah\b", r"\bsyrien\b", r"\bnordkorea\b", r"\bkreml\b",
    r"\bmullah\b", r"\bregime\b", r"\bsanktion\b",
    # Bundespolitik
    r"\bbundesregierung\b", r"\bbundestag\b", r"\bbundeskanzler\b",
    r"\bministerium\b", r"\bkoalition\b", r"\bampel-",
    # Redaktionell / Investigativ
    r"liveblog", r"liveticker", r"\+\+", r"kommentar:", r"analyse:",
    r"meinung:", r"interview:", r"hintergrund:", r"investigativ",
    r"recherche zeigt", r"studie:", r"forschung:",
    # Sport / Unterhaltung
    r"bundesliga", r"champions.league", r"\bdfb\b", r"nationalmannschaft",
    r"premier.league", r"transfermarkt",
    # Lifestyle / Wissenschaft
    r"klimawandel", r"energiewende", r"\bnasa\b", r"weltraum",
]
_BLACKLIST_RE = _re.compile("|".join(_BLACKLIST_PATTERNS), _re.IGNORECASE)


def _is_local_blaulicht(event: dict) -> bool:
    """
    Prüft ob ein Event eine lokale Blaulicht-Meldung ist.
    Whitelist-Ansatz: Nur rein was NACHWEISLICH lokal + Blaulicht ist.
    """
    quelle_typ = event.get("quelle_typ", "")
    titel = event.get("titel", "")
    zusammenfassung = event.get("zusammenfassung", "")
    text = f"{titel} {zusammenfassung}".lower()

    # Schritt 1: Harte Blacklist — sofort raus
    if _BLACKLIST_RE.search(text):
        return False

    # Schritt 2: Vertrauenswürdige lokale Quellen (Polizei-Presseportale)
    # Diese kommen direkt von Polizei-Dienststellen → per Definition lokal
    if quelle_typ in _TRUSTED_LOCAL_SOURCES:
        return True

    # Schritt 3: Andere Quellen (Twitter, Telegram) — Blaulicht-Keyword erforderlich
    for kw in _BLAULICHT_KEYWORDS:
        if kw in text:
            return True

    # Kein Blaulicht-Signal gefunden → nicht aufnehmen
    return False


def insert_event(event: dict) -> bool:
    """
    Event in die DB einfügen. Gibt True zurück wenn neu, False bei Duplikat.

    Erwartete Keys:
      titel, typ, lat, lon, zeitpunkt, scoop_score, quellen (list),
      zusammenfassung, media_urls (list), link, quelle_typ
    """
    event_id = event.get("id") or make_event_id(
        event["titel"], event["zeitpunkt"], event["lat"], event["lon"]
    )
    freshness = compute_freshness(event["zeitpunkt"])

    # Events älter als 24h gar nicht erst speichern (konsistent mit DB-Cleanup)
    try:
        ts = event["zeitpunkt"]
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        if age_h > 24:
            return False
    except Exception:
        pass

    # Crime-Radar Regio-Filter: Nur lokale Blaulicht-Meldungen zulassen
    quelle_typ = event.get("quelle_typ", "")
    if quelle_typ not in ("dwd", "nina"):
        if not _is_local_blaulicht(event):
            log.debug("Regio-Filter: kein lokales Blaulicht — %s", event["titel"][:80])
            return False

    # Qualitätsfilter: Polizei-Pressestatistiken und Nicht-Events rausfiltern
    titel_lower = event.get("titel", "").lower()
    _non_event_patterns = [
        "kriminalstatistik", "geschwindigkeitskontrolle", "geschwindigkeitsmessung",
        "geschwindigkeitsverstöße", "kontrollwoche", "verkehrsüberwachung",
        "blitzermeldung", "blitzerstandorte", "verkehrssicherheitswoche",
        "presseerklärung", "pressekonferenz", "vorstellung der",
        "sicherheitstipps", "prävention", "einbruchschutz",
        "codier-aktion", "aktionstag", "tag der offenen tür",
        "bilanz der", "jahresbilanz", "halbjahresbilanz",
        "personalwerbung", "bewerbung", "stellenangebot",
        "pks 20",  # Polizeiliche Kriminalstatistik
    ]
    if any(p in titel_lower for p in _non_event_patterns):
        log.debug("Qualitätsfilter: Nicht-Event — %s", event["titel"][:80])
        return False

    # BILD-Overlap prüfen (nur wenn Checker geladen)
    if quelle_typ not in ("nina", "dwd"):
        overlap = bild_checker.check_overlap(event["titel"])
        if overlap["type"] == "exact":
            log.info("BILD-Filter: übersprungen (exact overlap %.2f) — %s",
                      overlap["score"], event["titel"][:60])
            return False

        # Score mit BILD-Info neu berechnen
        base_score = scoring.compute_scoop_score(
            title=event["titel"],
            description=event.get("zusammenfassung", ""),
            bild_overlap_type=overlap["type"],
            zeitpunkt=event["zeitpunkt"],
            media_urls=event.get("media_urls", []),
            category=event.get("typ", ""),
        )
        # Boulevard-Boost: BILD-Relevanz-Muster erkennen
        boost = compute_boulevard_boost(event["titel"], event.get("zusammenfassung", ""))
        event["scoop_score"] = max(0.0, min(10.0, round(base_score + boost, 1)))

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone()
        if existing:
            return False

        # Wetter-Dedup: Gleiche DWD/NINA-Warnung nicht doppelt einfügen
        # Nur 1 Event pro Warnungstyp-Keyword (Frost, Gewitter, Glätte etc.)
        if quelle_typ in ("dwd", "nina") and event.get("typ") == "unwetter":
            # Warnungstyp-Keyword aus dem Titel extrahieren
            _wetter_keywords = [
                "frost", "glätte", "glatteis", "gewitter", "wind", "sturm",
                "starkregen", "regen", "schneefall", "schnee", "tauwetter",
                "hitze", "nebel", "hochwasser", "starkwind", "orkan", "tornado",
            ]
            titel_lower = event["titel"].lower()
            event_wtype = next((kw for kw in _wetter_keywords if kw in titel_lower), None)

            if event_wtype:
                similar = conn.execute(
                    "SELECT id, titel FROM events WHERE quelle_typ = ? AND freshness IN ('HOT', 'WARM', 'COOLING')",
                    (quelle_typ,),
                ).fetchall()
                for row in similar:
                    if event_wtype in row["titel"].lower():
                        return False

        conn.execute("""
            INSERT INTO events (id, titel, typ, lat, lon, zeitpunkt, scoop_score,
                                quellen, zusammenfassung, media_urls, freshness,
                                link, quelle_typ)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            event["titel"],
            event.get("typ", "sonstiges"),
            event["lat"],
            event["lon"],
            event["zeitpunkt"],
            event.get("scoop_score", 0.0),
            json.dumps(event.get("quellen", []), ensure_ascii=False),
            event.get("zusammenfassung", ""),
            json.dumps(event.get("media_urls", []), ensure_ascii=False),
            freshness,
            event.get("link", ""),
            event.get("quelle_typ", "unknown"),
        ))
        conn.commit()
        log.info("Neues Event: [%s] %s (%s)", freshness, event["titel"][:60], event_id)
        return True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Verwaltet aktive WebSocket-Verbindungen für Live-Push."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info("WebSocket verbunden. Aktive Verbindungen: %d", len(self.active))

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        log.info("WebSocket getrennt. Aktive Verbindungen: %d", len(self.active))

    async def broadcast(self, message: dict):
        """Nachricht an alle verbundenen Clients senden."""
        dead = []
        payload = json.dumps(message, ensure_ascii=False)
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Scraper importieren und Scheduler
# ---------------------------------------------------------------------------

# Lazy imports der Scraper-Module
scraper_tasks = []


async def _run_with_timeout(coro, name: str, timeout: float = 60.0) -> list[dict]:
    """Scraper mit Timeout ausführen. Gibt [] bei Fehler/Timeout zurück."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log.error("%s: Timeout nach %.0fs", name, timeout)
        return []
    except Exception as e:
        log.error("%s Fehler: %s", name, e, exc_info=True)
        return []


async def run_scrapers():
    """Alle Scraper parallel ausführen und neue Events über WebSocket pushen."""
    from scrapers.polizei_rss import scrape_polizei_rss
    from scrapers.nina_api import scrape_nina_warnings
    from scrapers.twitter_scraper import scrape_twitter
    from scrapers.telegram_scraper import scrape_telegram
    from scrapers.ost_polizei_scraper import scrape_ost_polizei

    log.info("Starte Scraper-Durchlauf...")
    new_events = []

    # Alle Scraper parallel mit individuellem Timeout
    scraper_tasks = {
        "Polizei RSS":  _run_with_timeout(scrape_polizei_rss(), "Polizei RSS", 90),
        "NINA":         _run_with_timeout(scrape_nina_warnings(), "NINA", 30),
        "Twitter":      _run_with_timeout(scrape_twitter(), "Twitter", 60),
        "Telegram":     _run_with_timeout(scrape_telegram(), "Telegram", 60),
        "Ost-Polizei":  _run_with_timeout(scrape_ost_polizei(), "Ost-Polizei", 60),
    }

    results = await asyncio.gather(*scraper_tasks.values(), return_exceptions=True)

    for name, result in zip(scraper_tasks.keys(), results):
        if isinstance(result, Exception):
            log.error("%s: unerwarteter Fehler: %s", name, result)
            continue

        events = result
        # NINA: Nur Zivilschutz, keine reinen Wetterwarnungen
        if name == "NINA":
            events = [ev for ev in events if ev.get("typ") != "unwetter"]

        before = len(new_events)
        for ev in events:
            if insert_event(ev):
                new_events.append(ev)
        log.info("%s: %d geholt, %d neu", name, len(events), len(new_events) - before)

    # --- YouTube News: Videos mit Events matchen ---
    try:
        from scrapers.youtube_news import fetch_youtube_news, match_videos_to_events
        videos = await fetch_youtube_news()
        if videos:
            # Alle HOT/WARM Events aus DB holen
            conn = get_db()
            try:
                rows = conn.execute(
                    "SELECT id, titel, media_urls FROM events WHERE freshness IN ('HOT', 'WARM')"
                ).fetchall()
                db_events = [{"id": r[0], "titel": r[1], "media_urls": json.loads(r[2] or "[]")} for r in rows]

                matches = match_videos_to_events(videos, db_events)
                updated = 0
                for event_id, urls in matches.items():
                    # Bestehende media_urls ergänzen
                    existing = next((e["media_urls"] for e in db_events if e["id"] == event_id), [])
                    new_urls = [u for u in urls if u not in existing]
                    if new_urls:
                        merged = existing + new_urls
                        conn.execute(
                            "UPDATE events SET media_urls = ? WHERE id = ?",
                            (json.dumps(merged, ensure_ascii=False), event_id),
                        )
                        updated += 1
                conn.commit()
                log.info("YouTube: %d Videos, %d Events gematcht, %d aktualisiert", len(videos), len(matches), updated)
            finally:
                conn.close()
    except Exception as e:
        log.error("YouTube News Fehler: %s", e, exc_info=True)

    # Freshness-Update für alle Events
    update_all_freshness()

    # Neue Events über WebSocket pushen
    if new_events and ws_manager.active:
        for ev in new_events:
            feature = event_to_geojson_feature(ev)
            await ws_manager.broadcast({
                "type": "new_event",
                "feature": feature,
            })
        log.info("%d neue Events an %d Clients gepusht", len(new_events), len(ws_manager.active))

    return len(new_events)


async def scheduler_loop():
    """Alle 5 Minuten Scraper laufen lassen."""
    # Erster Durchlauf sofort nach Start
    await asyncio.sleep(3)
    while True:
        try:
            count = await run_scrapers()
            log.info("Scheduler-Durchlauf abgeschlossen. %d neue Events.", count)
        except Exception as e:
            log.error("Scheduler Fehler: %s", e, exc_info=True)
        await asyncio.sleep(300)  # 5 Minuten


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

async def bild_checker_loop():
    """BILD-Checker alle 10 Minuten refreshen."""
    while True:
        try:
            count = await bild_checker.refresh()
            log.info("BILD-Checker Refresh: %d Titel geladen", count)
        except Exception as e:
            log.error("BILD-Checker Fehler: %s", e)
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup und Shutdown."""
    init_db()
    # BILD-Checker initial laden
    try:
        await bild_checker.refresh()
    except Exception as e:
        log.warning("BILD-Checker Init-Fehler: %s", e)
    # Background-Tasks starten
    task = asyncio.create_task(scheduler_loop())
    bild_task = asyncio.create_task(bild_checker_loop())
    log.info("Watchdog Server gestartet auf Port 8095")
    yield
    task.cancel()
    bild_task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    try:
        await bild_task
    except asyncio.CancelledError:
        pass
    log.info("Watchdog Server heruntergefahren")


app = FastAPI(
    title="News Watchdog",
    description="Echtzeit-Aggregation regionaler Breaking News",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS für lokale Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Hilfsfunktion: DB-Row zu GeoJSON Feature
# ---------------------------------------------------------------------------

def row_to_geojson_feature(row: sqlite3.Row) -> dict:
    """SQLite-Zeile in GeoJSON Feature umwandeln."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row["lon"], row["lat"]],
        },
        "properties": {
            "id": row["id"],
            "titel": row["titel"],
            "typ": row["typ"],
            "zeitpunkt": row["zeitpunkt"],
            "scoop_score": row["scoop_score"],
            "quellen": json.loads(row["quellen"]),
            "zusammenfassung": row["zusammenfassung"],
            "media_urls": json.loads(row["media_urls"]),
            "freshness": row["freshness"],
            "link": row["link"],
            "quelle_typ": row["quelle_typ"],
        },
    }


def event_to_geojson_feature(event: dict) -> dict:
    """Event-Dict in GeoJSON Feature umwandeln (für WebSocket-Push)."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [event["lon"], event["lat"]],
        },
        "properties": {
            "id": event.get("id", ""),
            "titel": event["titel"],
            "typ": event.get("typ", "sonstiges"),
            "zeitpunkt": event["zeitpunkt"],
            "scoop_score": event.get("scoop_score", 0.0),
            "quellen": event.get("quellen", []),
            "zusammenfassung": event.get("zusammenfassung", ""),
            "media_urls": event.get("media_urls", []),
            "freshness": compute_freshness(event["zeitpunkt"]),
            "link": event.get("link", ""),
            "quelle_typ": event.get("quelle_typ", "unknown"),
        },
    }


# ---------------------------------------------------------------------------
# Haversine-Distanz für Radius-Filter
# ---------------------------------------------------------------------------

import math

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine-Distanz in Kilometern."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def get_events(
    typ: Optional[str] = Query(None, description="Event-Typ filtern (polizei, feuer, unfall, unwetter, sonstiges)"),
    zeitraum: int = Query(24, description="Maximales Alter in Stunden (Default: 24h, 0=alle)"),
    lat: Optional[float] = Query(None, description="Breitengrad für Radius-Filter"),
    lon: Optional[float] = Query(None, description="Längengrad für Radius-Filter"),
    radius: Optional[float] = Query(None, description="Radius in km um lat/lon"),
    freshness: Optional[str] = Query(None, description="Freshness-Filter: HOT, WARM, COOLING, COLD"),
    limit: int = Query(500, description="Maximale Anzahl Events"),
):
    """
    GeoJSON FeatureCollection mit allen Events.
    Unterstützt Filter nach Typ, Zeitraum, Radius und Freshness.
    """
    # Freshness vor dem Query aktualisieren
    update_all_freshness()

    conn = get_db()
    try:
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if typ:
            query += " AND typ = ?"
            params.append(typ)

        if zeitraum > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=zeitraum)).isoformat()
            query += " AND zeitpunkt >= ?"
            params.append(cutoff)

        if freshness:
            query += " AND freshness = ?"
            params.append(freshness.upper())

        query += " ORDER BY zeitpunkt DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        # Radius-Filter (post-query, da SQLite kein Geo-Index hat)
        if lat is not None and lon is not None and radius is not None:
            rows = [
                r for r in rows
                if haversine_km(lat, lon, r["lat"], r["lon"]) <= radius
            ]

        features = [row_to_geojson_feature(r) for r in rows]

        return {
            "type": "FeatureCollection",
            "generated": datetime.now(timezone.utc).isoformat(),
            "count": len(features),
            "features": features,
        }
    finally:
        conn.close()


@app.get("/api/stats")
async def get_stats():
    """Statistiken über alle Events."""
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        hot = conn.execute("SELECT COUNT(*) FROM events WHERE freshness = 'HOT'").fetchone()[0]
        warm = conn.execute("SELECT COUNT(*) FROM events WHERE freshness = 'WARM'").fetchone()[0]
        cooling = conn.execute("SELECT COUNT(*) FROM events WHERE freshness = 'COOLING'").fetchone()[0]
        cold = conn.execute("SELECT COUNT(*) FROM events WHERE freshness = 'COLD'").fetchone()[0]

        # Typ-Verteilung
        typ_rows = conn.execute(
            "SELECT typ, COUNT(*) as cnt FROM events GROUP BY typ ORDER BY cnt DESC"
        ).fetchall()

        return {
            "total": total,
            "freshness": {"HOT": hot, "WARM": warm, "COOLING": cooling, "COLD": cold},
            "typen": {r["typ"]: r["cnt"] for r in typ_rows},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        conn.close()


@app.get("/api/trigger-scrape")
async def trigger_scrape():
    """Manueller Scraper-Durchlauf auslösen (Debug-Endpoint)."""
    count = await run_scrapers()
    return {"status": "ok", "neue_events": count}


# ---------------------------------------------------------------------------
# WebSocket für Live-Updates
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """WebSocket-Endpoint für Echtzeit-Event-Push."""
    await ws_manager.connect(ws)
    try:
        # Initial: Alle HOT + WARM Events senden
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE freshness IN ('HOT', 'WARM', 'COOLING') ORDER BY zeitpunkt DESC LIMIT 100"
            ).fetchall()
            features = [row_to_geojson_feature(r) for r in rows]
            await ws.send_text(json.dumps({
                "type": "initial",
                "features": features,
            }, ensure_ascii=False))
        finally:
            conn.close()

        # Keep-alive und auf Client-Nachrichten warten
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Client kann z.B. Ping senden
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Keep-alive Ping
                try:
                    await ws.send_text(json.dumps({"type": "heartbeat", "ts": time.time()}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Frontend ausliefern
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """index.html ausliefern."""
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>News Watchdog</h1><p>Frontend nicht gefunden.</p>")


# ---------------------------------------------------------------------------
# Standalone starten
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8095
    # BIND_HOST default 127.0.0.1 — nur lokal erreichbar.
    # Auf Render/Railway: BIND_HOST=0.0.0.0 in Umgebungsvariablen setzen.
    bind_host = os.environ.get("BIND_HOST", "127.0.0.1")
    uvicorn.run(
        app,
        host=bind_host,
        port=port,
        log_level="info",
    )
