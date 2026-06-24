"""app/config.py — Alle Konfigurationswerte aus Umgebungsvariablen.

Entspricht den frueheren os.environ.get()-Aufrufen aus dem Monolithen.
Beim Import wird automatisch eine .env-Datei im Projektverzeichnis geladen
(selbes Verhalten wie im Monolith).
"""
import os
import socket
import logging
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("push-balancer")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _csv_floats(name: str, default: str = "") -> list[float]:
    out: list[float] = []
    for item in _csv_env(name, default):
        try:
            out.append(float(item.replace(",", ".")))
        except ValueError:
            log.warning("Invalid float in env %s: %r (ignored)", name, item)
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        log.warning("Invalid integer env %s=%r, falling back to %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        log.warning("Invalid float env %s=%r, falling back to %s", name, raw, default)
        return default

# ── .env im Projektverzeichnis laden (identisch zum Monolith) ──────────────
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_ENV = os.path.join(_APP_DIR, ".env")
if os.path.exists(_LOCAL_ENV):
    with open(_LOCAL_ENV) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Server ─────────────────────────────────────────────────────────────────
PORT: int = int(os.environ.get("PORT", "8050"))
ALLOW_INSECURE_SSL: bool = os.environ.get("ALLOW_INSECURE_SSL", "0") == "1"

# ── OpenAI ─────────────────────────────────────────────────────────────────
PAID_EXTERNAL_APIS_ENABLED: bool = _env_flag(
    "PAID_EXTERNAL_APIS_ENABLED",
    False,
)
BACKGROUND_AUTOMATIONS_ENABLED: bool = _env_flag(
    "BACKGROUND_AUTOMATIONS_ENABLED",
    False,
)
HEALTH_ACTIVE_CHECKS_ENABLED: bool = _env_flag(
    "HEALTH_ACTIVE_CHECKS_ENABLED",
    False,
)
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AI_API_KEY", "")
OPENAI_TITLE_GENERATION_ENABLED: bool = _env_flag(
    "OPENAI_TITLE_GENERATION_ENABLED",
    False,
)
OPENAI_TITLE_GENERATION_MODEL: str = os.environ.get(
    "OPENAI_TITLE_GENERATION_MODEL",
    "gpt-4o-mini",
)
OPENAI_TITLE_GENERATION_TIMEOUT_S: float = float(
    os.environ.get("OPENAI_TITLE_GENERATION_TIMEOUT_S", "8.0")
)
OPENAI_TITLE_GENERATION_MAX_TOKENS: int = int(
    os.environ.get("OPENAI_TITLE_GENERATION_MAX_TOKENS", "320")
)
OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR: int = _env_int(
    "OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR",
    0,
)
OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY: int = _env_int(
    "OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY",
    0,
)
OPENAI_BACKFILL_ENABLED: bool = _env_flag(
    "OPENAI_BACKFILL_ENABLED",
    False,
)
OPENAI_PREDICTION_SCORING_ENABLED: bool = _env_flag(
    "OPENAI_PREDICTION_SCORING_ENABLED",
    False,
)
OPENAI_PREDICTION_SCORING_MODEL: str = os.environ.get(
    "OPENAI_PREDICTION_SCORING_MODEL",
    "gpt-4o-mini",
)
OPENAI_PREDICTION_SCORING_TIMEOUT_S: float = float(
    os.environ.get("OPENAI_PREDICTION_SCORING_TIMEOUT_S", "4.0")
)
OPENAI_PREDICTION_SCORING_MAX_TOKENS: int = int(
    os.environ.get("OPENAI_PREDICTION_SCORING_MAX_TOKENS", "60")
)
OPENAI_PREDICTION_SCORING_CACHE_TTL_S: int = int(
    os.environ.get("OPENAI_PREDICTION_SCORING_CACHE_TTL_S", "3600")
)
OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR: int = _env_int(
    "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR",
    0,
)
OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY: int = _env_int(
    "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY",
    0,
)

# ── BILD APIs ──────────────────────────────────────────────────────────────
PUSH_API_BASE: str = os.environ.get("PUSH_API_BASE", "https://push-frontend.bildcms.de")
BILD_SITEMAP: str = os.environ.get("BILD_SITEMAP_URL", "https://www.bild.de/sitemap-news.xml")


def push_api_base_candidates() -> list[str]:
    """Return preferred Push API base URLs with a safe https fallback."""
    candidates: list[str] = []

    def _add(url: str) -> None:
        normalized = url.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(PUSH_API_BASE)

    parsed = urlsplit(PUSH_API_BASE)
    hostname = (parsed.hostname or "").lower()
    if hostname == "push-frontend.bildcms.de":
        alternate_scheme = "https" if parsed.scheme == "http" else "http"
        _add(
            urlunsplit(
                (
                    alternate_scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        )

    return candidates

# ── Sync / Render ──────────────────────────────────────────────────────────
SYNC_SECRET: str = os.environ.get("PUSH_SYNC_SECRET", "")
RENDER_SYNC_URL: str = os.environ.get("RENDER_SYNC_URL", "")

# ── Adobe Analytics ────────────────────────────────────────────────────────
ADOBE_CLIENT_ID: str = os.environ.get("ADOBE_CLIENT_ID", "")
ADOBE_CLIENT_SECRET: str = os.environ.get("ADOBE_CLIENT_SECRET", "")
ADOBE_TRAFFIC_ENABLED: bool = _env_flag(
    "ADOBE_TRAFFIC_ENABLED",
    False,
)
ADOBE_COMPANY_ID: str = os.environ.get("ADOBE_GLOBAL_COMPANY_ID", "axelsp2")
ADOBE_RSID: str = "axelspringerbild"
ADOBE_TOKEN_URL: str = "https://ims-na1.adobelogin.com/ims/token/v3"
ADOBE_API_BASE: str = "https://analytics.adobe.io/api"

# ── Render-Erkennung ───────────────────────────────────────────────────────
IS_RENDER: bool = os.environ.get("RENDER", "").lower() == "true"
ECONOMY_MODE: bool = _env_flag("ECONOMY_MODE", IS_RENDER)
PUSH_LIVE_FETCH_ENABLED: bool = _env_flag(
    "PUSH_LIVE_FETCH_ENABLED",
    not ECONOMY_MODE,
)
LIVE_FEED_FALLBACK_ENABLED: bool = _env_flag(
    "LIVE_FEED_FALLBACK_ENABLED",
    not ECONOMY_MODE,
)
RESEARCH_EXTERNAL_CONTEXT_ENABLED: bool = _env_flag(
    "RESEARCH_EXTERNAL_CONTEXT_ENABLED",
    not ECONOMY_MODE,
)
ARTICLE_PREDICTION_ENRICHMENT_ENABLED: bool = _env_flag(
    "ARTICLE_PREDICTION_ENRICHMENT_ENABLED",
    not ECONOMY_MODE,
)
# ── Dateipfade ─────────────────────────────────────────────────────────────
SERVE_DIR: str = os.path.join(_APP_DIR, "dist-frontend")  # React App Build
# DB_PATH env var → Render nutzt /data (persistent disk), lokal .push_history.db
_PREFERRED_DB_PATH: str = os.environ.get(
    "DB_PATH",
    os.path.join(_APP_DIR, ".push_history.db"),
)


def _resolve_writable_db_path(preferred: str) -> str:
    """Stellt sicher, dass der DB-Pfad beschreibbar ist.

    Auf Render kann `/data` (persistent disk) bei einem Container-Start
    nicht-beschreibbar sein (Permission-Race nach Mount). In dem Fall
    fallen wir auf `/tmp` zurück, damit der Server überhaupt startet.
    Daten sind dort nicht persistent, aber der Service läuft.
    """
    parent = os.path.dirname(preferred) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError:
        pass
    probe = os.path.join(parent, ".__db_writable_probe__")
    try:
        with open(probe, "w") as f:
            f.write("x")
        try:
            os.remove(probe)
        except OSError:
            pass
        return preferred
    except (OSError, PermissionError) as exc:
        fallback = os.path.join("/tmp", os.path.basename(preferred) or ".push_history.db")
        log.warning(
            "DB-Pfad %s nicht beschreibbar (%s) — Fallback auf %s. Daten sind nicht persistent!",
            preferred, exc, fallback,
        )
        try:
            os.makedirs(os.path.dirname(fallback) or "/tmp", exist_ok=True)
            with open(fallback + ".__probe__", "w") as f:
                f.write("x")
            os.remove(fallback + ".__probe__")
            return fallback
        except OSError as exc2:
            log.error("Selbst /tmp nicht beschreibbar (%s) — DB-Init wird crashen", exc2)
            return preferred


PUSH_DB_PATH: str = _resolve_writable_db_path(_PREFERRED_DB_PATH)
PUSH_DB_MAX_DAYS: int = int(os.environ.get("PUSH_DB_MAX_DAYS", "1460"))
PUSH_DB_MAX_ROWS: int = int(
    os.environ.get("PUSH_DB_MAX_ROWS", "5000" if IS_RENDER else "15000")
)
SNAPSHOT_PATH: str = os.environ.get(
    "PUSH_SNAPSHOT_PATH",
    os.path.join(_APP_DIR, "push-snapshot.json"),
)

# ── Railway / Render Domain ────────────────────────────────────────────────
_railway_domain: str = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
_render_domain: str = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")

# ── CORS Origins ───────────────────────────────────────────────────────────
ALLOWED_ORIGINS: list[str] = [
    f"http://localhost:{PORT}",
    f"http://127.0.0.1:{PORT}",
]
try:
    _local_ip = socket.gethostbyname(socket.gethostname())
    if _local_ip and _local_ip != "127.0.0.1":
        ALLOWED_ORIGINS.append(f"http://{_local_ip}:{PORT}")
except Exception:
    pass
if _railway_domain:
    ALLOWED_ORIGINS.append(f"https://{_railway_domain}")
if _render_domain:
    ALLOWED_ORIGINS.append(f"https://{_render_domain}")
else:
    ALLOWED_ORIGINS.append("https://push-balancer.onrender.com")

# Tunnel-Wildcards (Cloudflare, localtunnel, ngrok) nur im DEV_MODE.
# In Produktion sind diese deaktiviert — verhindert CORS-Missbrauch via fremder Tunnel.
_DEV_MODE_RAW = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")
if _DEV_MODE_RAW:
    ALLOWED_ORIGINS += [
        "https://*.trycloudflare.com",
        "https://*.loca.lt",
        "https://*.ngrok-free.app",
    ]

# ── Competitor & International RSS Feeds ──────────────────────────────────
COMPETITOR_FEEDS: dict[str, str] = {
    "welt":       "https://www.welt.de/feeds/latest.rss",
    "spiegel":    "https://www.spiegel.de/schlagzeilen/index.rss",
    "focus":      "https://www.focus.de/rss/",
    "ntv":        "https://www.n-tv.de/rss",
    "tagesschau": "https://www.tagesschau.de/index~rss2.xml",
    "faz":        "https://www.faz.net/rss/aktuell/",
    "sz":         "https://rss.sueddeutsche.de/rss/Topthemen",
    "stern":      "https://www.stern.de/feed/standard/all/",
    "t-online":   "https://www.t-online.de/feed.rss",
    "zeit":       "https://newsfeed.zeit.de/index",
}

INTERNATIONAL_FEEDS: dict[str, str] = {
    "bbc":         "https://feeds.bbci.co.uk/news/rss.xml",
    "guardian":    "https://www.theguardian.com/world/rss",
    "telegraph":   "https://www.telegraph.co.uk/rss.xml",
    "lemonde":     "https://www.lemonde.fr/rss/une.xml",
    "leparisien":  "https://www.leparisien.fr/arc/outboundfeeds/rss/",
    "elpais":      "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "corriere":    "https://xml2.corrieredellasera.it/rss/homepage.xml",
    "ansa":        "https://www.ansa.it/sito/ansait_rss.xml",
    "nzz":         "https://www.nzz.ch/recent.rss",
    "derstandard": "https://www.derstandard.at/rss",
    "aftonbladet": "https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/",
    "nos":         "https://feeds.nos.nl/nosnieuwsalgemeen",
    "rte":         "https://www.rte.ie/feeds/rss/?index=/news/",
    "cnn":         "https://rss.cnn.com/rss/edition.rss",
    "nytimes":     "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "washpost":    "https://feeds.washingtonpost.com/rss/world",
    "reuters":     "https://www.reutersagency.com/feed/",
    "aljazeera":   "https://www.aljazeera.com/xml/rss/all.xml",
    "abc_au":      "https://www.abc.net.au/news/feed/2942460/rss.xml",
    "scmp":        "https://www.scmp.com/rss/91/feed",
    "japantimes":  "https://www.japantimes.co.jp/feed/",
    "timesofind":  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "globo":       "https://g1.globo.com/rss/g1/",
    "abc_news":    "https://abcnews.go.com/abcnews/internationalheadlines",
}

SPORT_COMPETITOR_FEEDS: dict[str, str] = {
    "kicker":        "https://newsfeed.kicker.de/news/aktuell",
    "sportschau":    "https://www.sportschau.de/index~rss2.xml",
    "transfermarkt": "https://www.transfermarkt.de/rss/news",
    "sport_de":      "https://www.sport.de/rss/news/",
    "spiegel_sport": "https://www.spiegel.de/sport/index.rss",
    "faz_sport":     "https://www.faz.net/rss/aktuell/sport/",
    "rp_sport":      "https://rp-online.de/sport/feed.rss",
    "tz_sport":      "https://www.tz.de/sport/rssfeed.rdf",
    "11freunde":     "https://www.11freunde.de/fullarticlerss/index.rss",
}

SPORT_EUROPA_FEEDS: dict[str, str] = {
    "bbc_sport":      "https://feeds.bbci.co.uk/sport/rss.xml",
    "lequipe":        "https://dwh.lequipe.fr/api/edito/rss?path=/",
    "marca":          "https://e00-xlk-ue-marca.uecdn.es/rss/googlenews/portada.xml",
    "gazzetta":       "https://www.gazzetta.it/rss/home.xml",
    "as_sport":       "https://as.com/rss/tags/ultimas_noticias.xml",
    "orf_sport":      "https://rss.orf.at/sport.xml",
    "nzz_sport":      "https://www.nzz.ch/sport.rss",
    "standard_sport": "https://www.derstandard.at/rss/sport",
}

SPORT_GLOBAL_FEEDS: dict[str, str] = {
    "espn":      "https://www.espn.com/espn/rss/news",
    "skysports": "https://www.skysports.com/rss/12040",
    "cbssports": "https://www.cbssports.com/rss/headlines/",
    "yahoo_sport": "https://sports.yahoo.com/rss/",
}

# ── Cache TTL ──────────────────────────────────────────────────────────────
CACHE_TTL: int = 90  # Sekunden
MAX_RESPONSE_SIZE: int = 2 * 1024 * 1024  # 2 MB

# ── Safety ─────────────────────────────────────────────────────────────────
SAFETY_MODE: str = "ADVISORY_ONLY"

# ── Admin API Key (schützt POST-Endpoints: retrain, force-promote etc.) ────
# Setze ADMIN_API_KEY in .env auf einen starken Zufallswert.
# Wenn nicht gesetzt, geben Admin-Endpoints 503 zurück.
ADMIN_API_KEY: str = os.environ.get("ADMIN_API_KEY", "")

# ── Consumer API Key (schützt read-only API für andere Apps) ────────────────
# Wenn nicht gesetzt, bleiben die Consumer-Endpunkte deaktiviert.
CONSUMER_API_KEY: str = os.environ.get("CONSUMER_API_KEY", "")

# ── Dev Mode (Tunnel-Wildcards für CORS nur im lokalen Betrieb) ────────────
DEV_MODE: bool = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")


# ── Interner Zugriff / Netzwerk-Allowlist ─────────────────────────────────
# Auf Render ist der Browser-Zugriff standardmäßig eingeschränkt, bis
# die erlaubten AS-/VPN-Egress-CIDRs explizit gesetzt wurden.
INTERNAL_ACCESS_ENABLED: bool = _env_flag("INTERNAL_ACCESS_ENABLED", IS_RENDER)
INTERNAL_ACCESS_ALLOWED_CIDRS: list[str] = _csv_env(
    "INTERNAL_ACCESS_ALLOWED_CIDRS",
    "127.0.0.1/32,::1/128,145.243.0.0/16,91.220.134.0/24",
)
INTERNAL_ACCESS_EXEMPT_PATHS: list[str] = _csv_env(
    "INTERNAL_ACCESS_EXEMPT_PATHS",
    "/api/health",
)

# ── Microsoft Teams Push Recommendation Alerts ─────────────────────────────
# Disabled by default. Enabling this sends selected article metadata to the
# configured Teams/Power Automate endpoint and requires editorial/privacy approval.
PUSH_TEAMS_ALERTS_ENABLED: bool = _env_flag("PUSH_TEAMS_ALERTS_ENABLED", False)
PUSH_TEAMS_WEBHOOK_URL: str = os.environ.get("PUSH_TEAMS_WEBHOOK_URL", "")
PUSH_TEAMS_MIN_SCORE: float = _env_float("PUSH_TEAMS_MIN_SCORE", 75.0)
# Teams-Reife-Schwelle. PUSH_TEAMS_MIN_TEAMS_SCORE ist der bevorzugte Name,
# PUSH_TEAMS_MIN_ALERT_SCORE bleibt als Alias erhalten.
PUSH_TEAMS_MIN_ALERT_SCORE: float = _env_float(
    "PUSH_TEAMS_MIN_TEAMS_SCORE",
    _env_float("PUSH_TEAMS_MIN_ALERT_SCORE", 78.0),
)
PUSH_TEAMS_SCORE_ONLY_MODE: bool = _env_flag("PUSH_TEAMS_SCORE_ONLY_MODE", False)
PUSH_TEAMS_DASHBOARD_TOP_LIMIT: int = _env_int("PUSH_TEAMS_DASHBOARD_TOP_LIMIT", 20)
PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE: float = _env_float(
    "PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE",
    76.0,
)
PUSH_TEAMS_EDITORIAL_GATE_ENABLED: bool = _env_flag("PUSH_TEAMS_EDITORIAL_GATE_ENABLED", True)
# Ereignis-Gate: nicht-Breaking-Pushes brauchen ein konkretes Nachrichten-Ereignis
# (etwas ist passiert). Ohne Ereignis-Signal -> kein Push. Ersetzt die endlosen
# Stichwortlisten durch eine positive Anforderung; Service/Ratgeber/Teaser fallen raus.
PUSH_TEAMS_EVENT_GATE_ENABLED: bool = _env_flag("PUSH_TEAMS_EVENT_GATE_ENABLED", True)
# KI-generierten Push-Titel (push_title_agent, LLM) in den Teams-Nachrichten nutzen.
# Greift nur, wenn der LLM tatsaechlich verfuegbar ist (OPENAI_API_KEY +
# OPENAI_TITLE_GENERATION_ENABLED + Rate-Budget); sonst sauberer Fallback.
PUSH_TEAMS_LLM_TITLE_ENABLED: bool = _env_flag("PUSH_TEAMS_LLM_TITLE_ENABLED", True)
PUSH_TEAMS_EDITORIAL_TOP_LIMIT: int = _env_int("PUSH_TEAMS_EDITORIAL_TOP_LIMIT", 10)
PUSH_TEAMS_MIN_EDITORIAL_SCORE: float = _env_float("PUSH_TEAMS_MIN_EDITORIAL_SCORE", 74.0)
PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE: float = _env_float(
    "PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE",
    24.0,
)
PUSH_TEAMS_MIN_TIME_FIT_SCORE: float = _env_float("PUSH_TEAMS_MIN_TIME_FIT_SCORE", 4.0)
PUSH_TEAMS_QUIET_HOURS_START: str = os.environ.get("PUSH_TEAMS_QUIET_HOURS_START", "00:00")
PUSH_TEAMS_QUIET_HOURS_END: str = os.environ.get("PUSH_TEAMS_QUIET_HOURS_END", "05:30")
PUSH_TEAMS_MIN_OR: float = _env_float("PUSH_TEAMS_MIN_OR", 5.0)
PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH: int = _env_int(
    "PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH",
    30,
)
PUSH_TEAMS_REALERT_SCORE_DELTA: float = _env_float("PUSH_TEAMS_REALERT_SCORE_DELTA", 8.0)
PUSH_TEAMS_REALERT_OR_DELTA: float = _env_float("PUSH_TEAMS_REALERT_OR_DELTA", 0.75)
# Re-Alert-Cooldown. PUSH_TEAMS_REALERT_COOLDOWN_MINUTES ist der bevorzugte Name,
# PUSH_TEAMS_ALERT_COOLDOWN_MINUTES bleibt als Alias erhalten.
PUSH_TEAMS_ALERT_COOLDOWN_MINUTES: int = _env_int(
    "PUSH_TEAMS_REALERT_COOLDOWN_MINUTES",
    _env_int("PUSH_TEAMS_ALERT_COOLDOWN_MINUTES", 90),
)
PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS: int = _env_int(
    "PUSH_TEAMS_REPEAT_SUPPRESSION_HOURS",
    12,
)
PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES: int = _env_int(
    "PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES",
    45,
)
_DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS = "News,Politik,Wirtschaft,Regional,Digital,Unterhaltung"
PUSH_TEAMS_ALLOWED_SECTIONS: list[str] = _csv_env(
    "PUSH_TEAMS_ALLOWED_SECTIONS",
    _DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS,
) or _csv_env("_PUSH_TEAMS_ALLOWED_SECTIONS_DEFAULT", _DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS)
# Ressorts, die NIE als Teams-Handlungsempfehlung vorgeschlagen werden — auch
# dann nicht, wenn die Allow-Liste leer (= alles erlaubt) ist. Standard: Sport.
PUSH_TEAMS_EXCLUDED_SECTIONS: list[str] = _csv_env(
    "PUSH_TEAMS_EXCLUDED_SECTIONS",
    "Sport",
)
# Tagesziel an Pushes (CvD-Richtwert) und dynamische Schwellenanpassung.
PUSH_TEAMS_TARGET_PUSHES_PER_DAY: int = _env_int("PUSH_TEAMS_TARGET_PUSHES_PER_DAY", 11)
PUSH_TEAMS_MIN_ALERTS_PER_DAY: int = _env_int("PUSH_TEAMS_MIN_ALERTS_PER_DAY", 11)
PUSH_TEAMS_MAX_ALERTS_PER_DAY: int = _env_int("PUSH_TEAMS_MAX_ALERTS_PER_DAY", 11)
PUSH_TEAMS_REQUIRE_VALID_PREDICTION: bool = _env_flag(
    "PUSH_TEAMS_REQUIRE_VALID_PREDICTION",
    False,
)
PUSH_TEAMS_REQUIRE_ARTICLE_FORECAST: bool = _env_flag(
    "PUSH_TEAMS_REQUIRE_ARTICLE_FORECAST",
    True,
)
# Erkennung konstanter Fake-/Default-Prognosen (z. B. globaler Durchschnitt 4.77 %).
# Ein OR-Wert, der sich ueber das Kandidatenfeld wiederholt, ist ein Default und
# wird NICHT als belastbare Prognose gewertet.
PUSH_TEAMS_KNOWN_DEFAULT_FORECASTS: list[float] = _csv_floats(
    "PUSH_TEAMS_KNOWN_DEFAULT_FORECASTS",
    "4.77",
)
PUSH_TEAMS_CONSTANT_FORECAST_MIN_FIELD: int = _env_int(
    "PUSH_TEAMS_CONSTANT_FORECAST_MIN_FIELD",
    3,
)
PUSH_TEAMS_KNOWN_DEFAULT_MIN_FIELD: int = _env_int(
    "PUSH_TEAMS_KNOWN_DEFAULT_MIN_FIELD",
    2,
)
# "Klarer Gewinner"-Regel: ist das Feld unsicher (Top-Kandidat nur knapp vor dem
# Verfolger und selbst nicht eindeutig stark), wird kein Alert gesendet.
# Breaking und eindeutig starke Kandidaten (Editorial >= Schwelle + Buffer) sind
# von der Margin-Pruefung ausgenommen.
# Spekulative/erwartete Lagen ("wohl", "bereitet ... vor", "soll zuruecktreten")
# altern schlecht: die Realitaet kann sie ueberholt haben. Aelter als X Stunden ->
# nicht mehr pushen (wahrscheinlich ueberholt). Frisch -> nur als Risiko markieren.
PUSH_TEAMS_SPECULATIVE_GUARD_ENABLED: bool = _env_flag("PUSH_TEAMS_SPECULATIVE_GUARD_ENABLED", True)
PUSH_TEAMS_SPECULATIVE_MAX_AGE_HOURS: float = _env_float("PUSH_TEAMS_SPECULATIVE_MAX_AGE_HOURS", 3.0)
# Abgleich gegen die (gecachten) Konkurrenz-/International-Feeds: meldet eine
# frischere Quelle die spekulierte Lage bereits als vollzogen (z. B. "Starmer
# tritt zurueck" / "resigns"), wird die Spekulation als ueberholt geblockt.
PUSH_TEAMS_FEED_OVERTAKEN_ENABLED: bool = _env_flag("PUSH_TEAMS_FEED_OVERTAKEN_ENABLED", True)
# Themen-Dublette: ein anderer Artikel zum selben Ereignis wurde bereits per Teams
# gemeldet (z. B. zwei Schlagzeilen zur selben Explosion). Innerhalb des Fensters
# und ueber der Token-Aehnlichkeit -> kein zweiter Alert.
PUSH_TEAMS_TOPIC_DEDUP_HOURS: float = _env_float("PUSH_TEAMS_TOPIC_DEDUP_HOURS", 12.0)
PUSH_TEAMS_TOPIC_DEDUP_SIMILARITY: float = _env_float("PUSH_TEAMS_TOPIC_DEDUP_SIMILARITY", 0.5)
# Abgleich gegen ECHTE Live-Pushes: wurde dieselbe Story bereits real gepusht
# (gleiche Artikel-URL jederzeit im Verlauf, oder dieselbe Story per URL-Slug/
# Titel-Aehnlichkeit innerhalb dieses Fensters), wird sie in Teams nicht erneut
# vorgeschlagen.
PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS: float = _env_float(
    "PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS",
    36.0,
)
PUSH_TEAMS_MIN_SELECTION_MARGIN: float = _env_float("PUSH_TEAMS_MIN_SELECTION_MARGIN", 5.0)
PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER: float = _env_float(
    "PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER",
    6.0,
)
# Visit-Optimierung: Teams soll unter den pushwuerdigen Kandidaten den Vorschlag
# mit dem groessten erwarteten Push-Visit-Potenzial priorisieren. Das Potenzial
# basiert auf Artikel-OR-Prognose x historischer Reichweite je Ressort/Slot.
PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED",
    True,
)
PUSH_TEAMS_VISIT_SELECTION_WEIGHT: float = _env_float(
    "PUSH_TEAMS_VISIT_SELECTION_WEIGHT",
    0.62,
)
PUSH_TEAMS_DEFAULT_REACH: int = _env_int("PUSH_TEAMS_DEFAULT_REACH", 250000)
PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED",
    True,
)
# Maximale Absenkung/Anhebung der Teams-Reife-Schwelle durch die Push-Bestand-Logik.
PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP: float = _env_float(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP",
    10.0,
)
PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE: float = _env_float(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE",
    14.0,
)
# Aktives Push-Fenster (Berlin-Stunden) fuer die Pace-Berechnung des Tagesziels.
PUSH_TEAMS_ACTIVE_HOURS_START: int = _env_int("PUSH_TEAMS_ACTIVE_HOURS_START", 6)
PUSH_TEAMS_ACTIVE_HOURS_END: int = _env_int("PUSH_TEAMS_ACTIVE_HOURS_END", 23)
PUSH_TEAMS_BREAKING_OVERRIDE: bool = _env_flag("PUSH_TEAMS_BREAKING_OVERRIDE", True)
PUSH_TEAMS_BREAKING_MIN_SCORE: float = _env_float("PUSH_TEAMS_BREAKING_MIN_SCORE", 72.0)
PUSH_TEAMS_BREAKING_MIN_OR: float = _env_float("PUSH_TEAMS_BREAKING_MIN_OR", 4.0)
PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH: int = _env_int(
    "PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH",
    10,
)
PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS: int = _env_int("PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS", 24)
PUSH_TEAMS_MAX_PUSHES_LAST_6H: int = _env_int("PUSH_TEAMS_MAX_PUSHES_LAST_6H", 8)
PUSH_TEAMS_CHECK_INTERVAL_SECONDS: int = _env_int("PUSH_TEAMS_CHECK_INTERVAL_SECONDS", 120)
PUSH_TEAMS_CANDIDATE_LIMIT: int = _env_int("PUSH_TEAMS_CANDIDATE_LIMIT", 80)
