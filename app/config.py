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
# Push-Auto-Fetch (bildcms.de → Push-Historie) getrennt schaltbar von den
# schweren Research-/ML-Workern. So kann eine Instanz im AS-Netz (z.B. der
# Next-Pod) Push-Daten 24/7 selbst ziehen, OHNE die CPU-intensiven Research-
# Automationen zu aktivieren. Default = BACKGROUND_AUTOMATIONS_ENABLED
# (rückwärtskompatibel: bestehende Deploys ändern ihr Verhalten nicht).
PUSH_AUTO_FETCH_ENABLED: bool = _env_flag(
    "PUSH_AUTO_FETCH_ENABLED",
    BACKGROUND_AUTOMATIONS_ENABLED,
)
# Render-Sync (Push-Historie -> Render-Relay) getrennt schaltbar. So kann der
# Next-Pod (im AS-Netz, mit AutoFetch) Render 24/7 selbst füttern und damit den
# lokalen Mac-Relay ersetzen. Default = BACKGROUND_AUTOMATIONS_ENABLED
# (rückwärtskompatibel). Braucht zusätzlich RENDER_SYNC_URL + PUSH_SYNC_SECRET.
PUSH_RENDER_SYNC_ENABLED: bool = _env_flag(
    "PUSH_RENDER_SYNC_ENABLED",
    BACKGROUND_AUTOMATIONS_ENABLED,
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
    "gpt-5.6-luna",
)
OPENAI_TITLE_GENERATION_TIMEOUT_S: float = float(
    os.environ.get("OPENAI_TITLE_GENERATION_TIMEOUT_S", "8.0")
)
OPENAI_TITLE_GENERATION_MAX_TOKENS: int = int(
    os.environ.get("OPENAI_TITLE_GENERATION_MAX_TOKENS", "600")
)
OPENAI_TITLE_GENERATION_REASONING_EFFORT: str = os.environ.get(
    "OPENAI_TITLE_GENERATION_REASONING_EFFORT",
    "none",
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
URL_API_BASE: str = os.environ.get("URL_API_BASE", "").rstrip("/")
URL_API_KEY: str = os.environ.get("URL_API_KEY", "")
PUBLIC_ARTICLE_BASE_URL: str = os.environ.get(
    "PUBLIC_ARTICLE_BASE_URL",
    "https://www.bild.de",
).rstrip("/")


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
PUSH_BALANCER_CAPTURED_SCORE_MAX_AGE_SECONDS: int = _env_int(
    "PUSH_BALANCER_CAPTURED_SCORE_MAX_AGE_SECONDS",
    180,
)
# On-Demand-Tagesplan-Builds sind im Economy-Betrieb deaktiviert; der Endpoint
# liefert dann ein leichtes Loading-Payload statt eines teuren Rebuilds.
TAGESPLAN_ON_DEMAND_BUILD_ENABLED: bool = _env_flag("TAGESPLAN_ON_DEMAND_BUILD_ENABLED", False)
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

# Dedicated least-privilege credential for the scheduled Power Automate hand-off.
POWER_AUTOMATE_API_KEY: str = os.environ.get("POWER_AUTOMATE_API_KEY", "")

# Dedicated least-privilege credential for the CMS-ID score lookup.
SCORE_API_KEY: str = os.environ.get("SCORE_API_KEY", "")

# ── Dev Mode (Tunnel-Wildcards für CORS nur im lokalen Betrieb) ────────────
DEV_MODE: bool = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")


# ── Interner Zugriff / Netzwerk-Allowlist ─────────────────────────────────
# Auf Render ist der Browser-Zugriff standardmäßig eingeschränkt, bis
# die erlaubten AS-/VPN-Egress-CIDRs explizit gesetzt wurden. FORCE_INTERNAL_ACCESS
# erzwingt die Allowlist zusaetzlich in der Score-API-Laufzeit (app.score_main).
FORCE_INTERNAL_ACCESS: bool = _env_flag("FORCE_INTERNAL_ACCESS", False)
INTERNAL_ACCESS_ENABLED: bool = _env_flag("INTERNAL_ACCESS_ENABLED", IS_RENDER)
INTERNAL_ACCESS_ALLOWED_CIDRS: list[str] = _csv_env(
    "INTERNAL_ACCESS_ALLOWED_CIDRS",
    "127.0.0.1/32,::1/128,145.243.0.0/16,91.220.134.0/24",
)
INTERNAL_ACCESS_EXEMPT_PATHS: list[str] = _csv_env(
    "INTERNAL_ACCESS_EXEMPT_PATHS",
    "/api/health",
)
# Stable NAT gateway address for the approved BILD Next staging consumer.
# This allowlist applies only to the two read-only score-capture GET routes and
# the exact read-only batch POST; it does not expose the UI, debug, or capture POST.
SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS: list[str] = _csv_env(
    "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS",
    "3.79.136.119/32",
)

# ── Microsoft Teams Push Recommendation Alerts ─────────────────────────────
# Disabled by default. Enabling this sends selected article metadata to the
# configured Teams/Power Automate endpoint and requires editorial/privacy approval.
PUSH_TEAMS_ALERTS_ENABLED: bool = _env_flag("PUSH_TEAMS_ALERTS_ENABLED", False)
# Keep the recommendation policy/API enabled while allowing Power Automate to
# become the sole scheduler and transport owner during a reversible cutover.
PUSH_TEAMS_BACKGROUND_SENDER_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED",
    False,
)
# Power Automate must remain operational without an AS-network relay. When the
# live push history is reachable it is still used for exact duplicate checks;
# otherwise the durable slot and Teams receipt claims prevent repeated posts.
POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY: bool = _env_flag(
    "POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY",
    False,
)
PUSH_TEAMS_WEBHOOK_URL: str = os.environ.get("PUSH_TEAMS_WEBHOOK_URL", "")
# Kanonischer Push-Score fuer die Teams-Auswahl. In Produktion (render.yaml)
# auf Anweisung des Product Owners aktiviert (2026-07-29); lokal default aus.
# Fail-closed: bei Aktivierung gibt es keinen lokalen Score-Fallback.
PUSH_BALANCER_SCORE_API_ENABLED: bool = _env_flag(
    "PUSH_BALANCER_SCORE_API_ENABLED",
    False,
)
PUSH_BALANCER_SCORE_API_BASE_URL: str = os.environ.get(
    "PUSH_BALANCER_SCORE_API_BASE_URL",
    "",
).strip()
PUSH_BALANCER_SCORE_API_KEY: str = os.environ.get(
    "PUSH_BALANCER_SCORE_API_KEY",
    "",
).strip()

# Kandidaten-UI: getrennte, explizit aktivierte Verbindung zur kanonischen
# Editorial-One-Score-API. Diese Werte werden absichtlich nicht mit der
# Teams/Power-Automate-Score-Konfiguration geteilt, damit ein UI-Rollout den
# bestehenden Versandpfad weder umleitet noch blockiert.
EDITORIAL_ONE_SCORE_API_ENABLED: bool = _env_flag(
    "EDITORIAL_ONE_SCORE_API_ENABLED",
    False,
)
EDITORIAL_ONE_SCORE_API_BASE_URL: str = os.environ.get(
    "EDITORIAL_ONE_SCORE_API_BASE_URL",
    "",
).strip()
EDITORIAL_ONE_SCORE_API_KEY: str = os.environ.get(
    "EDITORIAL_ONE_SCORE_API_KEY",
    "",
).strip()
EDITORIAL_ONE_SCORE_API_MAX_AGE_SECONDS: int = _env_int(
    "EDITORIAL_ONE_SCORE_API_MAX_AGE_SECONDS",
    15 * 60,
)
# Selbst-Konsum (Render): Zeigt die Score-API-Basis-URL auf die eigene Instanz
# (Loopback), ist der eigene Server-Key SCORE_API_KEY per Definition der
# richtige Consumer-Key — er gewinnt dann immer. Das macht die kanonische
# Top-1 auf der Instanz möglich, die den Score selbst berechnet, ohne einen
# separat synchronisierten Consumer-Key. Auf Next (externe Basis-URL) ändert
# sich nichts.
_score_api_host = urlsplit(PUSH_BALANCER_SCORE_API_BASE_URL).hostname or ""
if _score_api_host in ("127.0.0.1", "localhost", "::1") and SCORE_API_KEY:
    PUSH_BALANCER_SCORE_API_KEY = SCORE_API_KEY
elif not PUSH_BALANCER_SCORE_API_KEY:
    PUSH_BALANCER_SCORE_API_KEY = SCORE_API_KEY

# Selbst-Konsum-Schalter, der Alt-Env-Werte vollstaendig ignoriert: Render-
# Dashboards koennen historisch materialisierte Blueprint-Werte (z.B.
# PUSH_BALANCER_SCORE_API_ENABLED=false) ueberschatten, sodass render.yaml-
# Aenderungen an diesen Keys wirkungslos bleiben. Der NEUE Key existiert in
# keinem Dashboard-Schatten; ist er gesetzt und ein Server-Key vorhanden,
# erzwingt er die kanonische Selbst-Konsum-Konfiguration (eigene
# /api/v1/scores-Route via Loopback mit dem eigenen SCORE_API_KEY).
PUSH_BALANCER_SCORE_API_SELF_CONSUME: bool = _env_flag(
    "PUSH_BALANCER_SCORE_API_SELF_CONSUME",
    # Render-Blueprint legt neue env-Keys nicht zuverlaessig an (Env-Sync tot/
    # manuell); auf Render ist der Code-Default daher der wirksame Schalter.
    IS_RENDER,
)
if PUSH_BALANCER_SCORE_API_SELF_CONSUME:
    if not SCORE_API_KEY:
        # Ephemerer Prozess-Key genuegt: Server-Route und Consumer-Client leben
        # in derselben Instanz und lesen beide diese Konfiguration.
        import secrets as _secrets
        SCORE_API_KEY = _secrets.token_urlsafe(32)
        os.environ.setdefault("SCORE_API_KEY", SCORE_API_KEY)
    PUSH_BALANCER_SCORE_API_ENABLED = True
    # Loopback-Port = der Port, auf den uvicorn tatsaechlich gebunden ist.
    # NICHT config.PORT verwenden: Render injiziert PORT=10000, waehrend das
    # Dockerfile uvicorn fest auf 8050 startet — mit PORT waere der
    # Selbstkonsum ein Connection-refused auf 127.0.0.1:10000 (alle Scores
    # 'unavailable'). 8050 ist der Dockerfile-/Dev-Standard; abweichende
    # Setups uebersteuern per PUSH_BALANCER_SELF_CONSUME_PORT.
    PUSH_BALANCER_SELF_CONSUME_PORT: int = _env_int(
        "PUSH_BALANCER_SELF_CONSUME_PORT",
        8050,
    )
    PUSH_BALANCER_SCORE_API_BASE_URL = (
        f"http://127.0.0.1:{PUSH_BALANCER_SELF_CONSUME_PORT}"
    )
    PUSH_BALANCER_SCORE_API_KEY = SCORE_API_KEY
PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS: float = _env_float(
    "PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS",
    2.5,
)
PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS: float = _env_float(
    "PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS",
    35.0,
)
PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS: float = _env_float(
    "PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS",
    45.0,
)
PUSH_BALANCER_SCORE_API_MAX_AGE_SECONDS: int = _env_int(
    "PUSH_BALANCER_SCORE_API_MAX_AGE_SECONDS",
    8 * 3600,
)
PUSH_BALANCER_SCORE_API_MAX_CONCURRENCY: int = _env_int(
    "PUSH_BALANCER_SCORE_API_MAX_CONCURRENCY",
    16,
)
PUSH_BALANCER_SCORE_API_MAX_RETRIES: int = _env_int(
    "PUSH_BALANCER_SCORE_API_MAX_RETRIES",
    1,
)
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
PUSH_TEAMS_QUIET_HOURS_START: str = os.environ.get("PUSH_TEAMS_QUIET_HOURS_START", "23:00")
PUSH_TEAMS_QUIET_HOURS_END: str = os.environ.get("PUSH_TEAMS_QUIET_HOURS_END", "06:00")
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
    30,
)
# Nach einem gesendeten Teams-Hinweis steigt die rohe Push-Score-Schwelle
# zunaechst bis zum Peak und faellt anschliessend bis zum Ende des Fensters
# linear auf PUSH_TEAMS_MIN_SCORE zurueck. Der harte Cooldown bleibt separat.
PUSH_TEAMS_POST_SEND_THRESHOLD_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_POST_SEND_THRESHOLD_ENABLED",
    True,
)
PUSH_TEAMS_POST_SEND_PEAK_SCORE: float = _env_float(
    "PUSH_TEAMS_POST_SEND_PEAK_SCORE",
    80.0,
)
PUSH_TEAMS_POST_SEND_DECAY_MINUTES: int = _env_int(
    "PUSH_TEAMS_POST_SEND_DECAY_MINUTES",
    90,
)
# Strikt groesser als dieser kanonische Push Score darf weiche Qualitaets- und
# Ermuedungsgates ueberstimmen. Ruhezeit, Timing, Fakten, Aktualitaet,
# Cooldown, Ressort, Tageslimit sowie Live-Push- und Teams-Dubletten bleiben hart.
PUSH_TEAMS_HIGH_SCORE_ALWAYS_THRESHOLD: float = _env_float(
    "PUSH_TEAMS_HIGH_SCORE_ALWAYS_THRESHOLD",
    80.0,
)
# Nicht konfigurierbare Produktregel: Alle tatsaechlich gesendeten Live-Pushes
# zaehlen zum Tagesvolumen und steuern Pacing, Mindestabstand und Tageslimit der
# Teams-Empfehlungen. Eine identische bereits live gepushte Artikel-URL oder
# CMS-ID bleibt zusaetzlich gesperrt.
PUSH_TEAMS_INDEPENDENT_PACING_ENABLED: bool = False
_DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS = (
    "News,Politik,Wirtschaft,Geld,Regional,Digital,Unterhaltung,Sport,Leben-Wissen"
)
PUSH_TEAMS_ALLOWED_SECTIONS: list[str] = _csv_env(
    "PUSH_TEAMS_ALLOWED_SECTIONS",
    _DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS,
) or _csv_env("_PUSH_TEAMS_ALLOWED_SECTIONS_DEFAULT", _DEFAULT_PUSH_TEAMS_ALLOWED_SECTIONS)
# Ressorts, die NIE als Teams-Handlungsempfehlung vorgeschlagen werden — auch
# dann nicht, wenn die Allow-Liste leer (= alles erlaubt) ist. Sport wird ueber
# ein eigenes Ereignis- und Timing-Gate bewertet und deshalb nicht pauschal gesperrt.
PUSH_TEAMS_EXCLUDED_SECTIONS: list[str] = _csv_env(
    "PUSH_TEAMS_EXCLUDED_SECTIONS",
    "",
)
# Public article URLs that were recommended through an approved operational
# recovery outside the application transport. They participate only in exact
# Teams deduplication and contain no recipient or employee data.
PUSH_TEAMS_EXTERNALLY_RECOMMENDED_URLS: list[str] = _csv_env(
    "PUSH_TEAMS_EXTERNALLY_RECOMMENDED_URLS",
    "",
)
# Tagesvolumen: mindestens 11, maximal 15 Pushes pro Tag. Alle gesendeten
# Live-Pushes zaehlen mit; das Pacing arbeitet auf dem echten Push-Bestand.
PUSH_TEAMS_TARGET_PUSHES_PER_DAY: int = _env_int("PUSH_TEAMS_TARGET_PUSHES_PER_DAY", 11)
PUSH_TEAMS_MIN_ALERTS_PER_DAY: int = _env_int("PUSH_TEAMS_MIN_ALERTS_PER_DAY", 11)
PUSH_TEAMS_MAX_ALERTS_PER_DAY: int = _env_int("PUSH_TEAMS_MAX_ALERTS_PER_DAY", 15)
# Redaktioneller Tagesplan fuer Teams: nicht jeder Slot muss ein Sofort-Alert sein,
# aber der CvD soll einen vollstaendigen, transparent priorisierten Tagesplan sehen.
PUSH_TEAMS_DAILY_PLAN_MIN_ITEMS: int = _env_int("PUSH_TEAMS_DAILY_PLAN_MIN_ITEMS", 11)
PUSH_TEAMS_DAILY_PLAN_MAX_ITEMS: int = _env_int("PUSH_TEAMS_DAILY_PLAN_MAX_ITEMS", 15)
# Verbindliche Live-Entscheidungslogik: 06:15 und 06:45 sind taegliche
# Basisfenster. Jede rote/gelbe Wochentagszelle bekommt ebenfalls zwei frische
# Top-1-Entscheidungen um :15 und :45. Bis mindestens 15 werden die besten
# verbleibenden :45-Fenster ergaenzt; an starken Tagen sind bis zu 18 verbindlich.
PUSH_TEAMS_SLOT_GATE_ENABLED: bool = _env_flag("PUSH_TEAMS_SLOT_GATE_ENABLED", True)
PUSH_TEAMS_SLOT_DEADLINE_MINUTE: int = _env_int("PUSH_TEAMS_SLOT_DEADLINE_MINUTE", 45)
PUSH_TEAMS_PEAK_SLOT_MIN_OR: float = _env_float("PUSH_TEAMS_PEAK_SLOT_MIN_OR", 6.0)
# Date-scoped recovery delay for the remaining binding slots after a late
# operational recovery.  The date makes the override self-expiring so the
# normal deterministic schedule resumes automatically on the next day.
PUSH_TEAMS_SLOT_DELAY_DATE: str = os.environ.get(
    "PUSH_TEAMS_SLOT_DELAY_DATE",
    "",
).strip()
PUSH_TEAMS_SLOT_DELAY_FROM: str = os.environ.get(
    "PUSH_TEAMS_SLOT_DELAY_FROM",
    "",
).strip()
PUSH_TEAMS_SLOT_DELAY_MINUTES: int = _env_int(
    "PUSH_TEAMS_SLOT_DELAY_MINUTES",
    0,
)
PUSH_TEAMS_EARLY_EXCEPTIONAL_SCORE: float = _env_float(
    "PUSH_TEAMS_EARLY_EXCEPTIONAL_SCORE",
    88.0,
)
PUSH_TEAMS_EARLY_EXCEPTIONAL_ALERT_SCORE: float = _env_float(
    "PUSH_TEAMS_EARLY_EXCEPTIONAL_ALERT_SCORE",
    86.0,
)
PUSH_TEAMS_EARLY_EXCEPTIONAL_EDITORIAL_SCORE: float = _env_float(
    "PUSH_TEAMS_EARLY_EXCEPTIONAL_EDITORIAL_SCORE",
    80.0,
)
PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE: float = _env_float(
    "PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE",
    75.0,
)
PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE: float = _env_float(
    "PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE",
    73.0,
)
PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE: float = _env_float(
    "PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE",
    69.0,
)
# Ein kompakter Tagesfahrplan wird einmal pro Berliner Kalendertag gesendet.
# Der persistente Claim verhindert Doppelversand bei Restart oder mehreren Workern.
PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED",
    False,
)
PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME: str = os.environ.get(
    "PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME",
    "05:45",
)
# Legacy Heartbeat-Konfiguration bleibt nur fuer rueckwaertskompatible Deployments
# lesbar. Artikel-Heartbeats sind durch die Versand-Policy hart gesperrt: Eine
# Empfehlung darf ausschliesslich den regulaeren Slot- und Freigabepfad nutzen.
PUSH_TEAMS_HEARTBEAT_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_HEARTBEAT_ENABLED",
    False,
)
PUSH_TEAMS_HEARTBEAT_MAX_SILENCE_MINUTES: int = _env_int(
    "PUSH_TEAMS_HEARTBEAT_MAX_SILENCE_MINUTES",
    90,
)
# Ein per Heartbeat (unter Alarm-Schwelle) bereits gemeldeter Artikel wird sonst
# wie ein echter Alert dauerhaft gesperrt. Steigt sein Push-Score spaeter um
# mindestens diesen Betrag ueber den Heartbeat-Score UND ueber die Alarm-Schwelle,
# darf die eskalierende Story regulaer als Hard-Alert erneut in den Channel.
PUSH_TEAMS_HEARTBEAT_ESCALATION_MARGIN: float = float(
    _env_int("PUSH_TEAMS_HEARTBEAT_ESCALATION_MARGIN", 8)
)
# Nachrichtentyp 2: Jeder tatsaechlich versendete Live-Push der Redaktion wird
# als eigene Teams-Nachricht gespiegelt und fliesst sofort in die weitere Planung.
# Nur 🔵 PUSH-EMPFEHLUNGEN im Kanal (User-Vorgabe 2026-08-04). Default False,
# weil Render-Blueprint-Env-Sync neue Keys nicht zuverlaessig anlegt — der
# Code-Default ist der einzige garantiert wirksame Schalter.
PUSH_TEAMS_LIVE_PUSH_POSTS_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_LIVE_PUSH_POSTS_ENABLED",
    False,
)
# Nur Live-Pushes, die juenger als dieses Fenster sind, werden noch als Nachricht
# gepostet (aeltere zaehlen trotzdem zum Tagesvolumen). Verhindert eine Flut
# historischer Posts nach Neustart oder Ausfall.
PUSH_TEAMS_LIVE_PUSH_POST_LOOKBACK_MINUTES: int = _env_int(
    "PUSH_TEAMS_LIVE_PUSH_POST_LOOKBACK_MINUTES",
    90,
)
PUSH_TEAMS_LIVE_PUSH_POSTS_PER_CYCLE: int = _env_int(
    "PUSH_TEAMS_LIVE_PUSH_POSTS_PER_CYCLE",
    3,
)
# Zustellungs-Robustheit: transiente Webhook-Fehler (Timeout, Verbindungsabbruch,
# 408/429/5xx) werden mit exponentiellem Backoff wiederholt. Permanente 4xx
# werden nicht wiederholt. Schlaegt der Versand endgueltig fehl, bleibt der
# Kandidat nur kurz gesperrt (statt den ganzen Tag), damit ein Netzwerk-Blip
# keine Story verbrennt.
PUSH_TEAMS_WEBHOOK_MAX_ATTEMPTS: int = _env_int("PUSH_TEAMS_WEBHOOK_MAX_ATTEMPTS", 3)
PUSH_TEAMS_WEBHOOK_RETRY_BACKOFF_SECONDS: float = _env_float(
    "PUSH_TEAMS_WEBHOOK_RETRY_BACKOFF_SECONDS",
    1.5,
)
PUSH_TEAMS_TRANSPORT_FAILURE_COOLDOWN_MINUTES: int = _env_int(
    "PUSH_TEAMS_TRANSPORT_FAILURE_COOLDOWN_MINUTES",
    20,
)
# Watchdog: laeuft laenger als diese Frist kein Worker-Zyklus, gilt der Kanal
# als gestoert (Health degraded) und der Supervisor startet den Thread neu.
PUSH_TEAMS_WORKER_STALL_SECONDS: int = _env_int("PUSH_TEAMS_WORKER_STALL_SECONDS", 600)

# Timeliness-Eskalation: eine brandaktuelle Top-Story (hoher Push-Balancer-Score
# UND sehr frisch veroeffentlicht) darf sofort raus, ohne auf den naechsten
# Raster-Slot zu warten. Das ist bewusst an Qualitaet + Frische gekoppelt und
# NIE an Volumendruck - so bleibt der Off-Raster-Schutz gegen schwache Fueller
# unangetastet. Ein Live-Event (z. B. eine Parade) ist damit schnell, ohne dass
# eine formale Eilmeldung noetig ist.
PUSH_TEAMS_HOT_FRESH_ENABLED: bool = _env_flag("PUSH_TEAMS_HOT_FRESH_ENABLED", True)
PUSH_TEAMS_HOT_FRESH_MIN_SCORE: float = _env_float("PUSH_TEAMS_HOT_FRESH_MIN_SCORE", 85.0)
PUSH_TEAMS_HOT_FRESH_MAX_AGE_MINUTES: float = _env_float(
    "PUSH_TEAMS_HOT_FRESH_MAX_AGE_MINUTES",
    20.0,
)
# Eskalation nur, wenn der naechste Raster-Slot weiter als dieser Abstand
# entfernt ist - eine brandaktuelle Story soll nicht stundenlang warten, aber
# wenn der naechste Slot ohnehin bald kommt, altert sie kaum. Trennt das
# Live-Event (grosse Luecke) sauber vom normalen Takt (kleine Luecke).
PUSH_TEAMS_HOT_FRESH_MIN_GAP_MINUTES: float = _env_float(
    "PUSH_TEAMS_HOT_FRESH_MIN_GAP_MINUTES",
    180.0,
)
# Sport-Zielkorridor: ungefaehr ein Drittel des Tagesvolumens (4-5 von 11-15).
# Weicher Korridor, keine starre Pflicht. Die Praeferenz-Bandbreite begrenzt,
# wie viele Score-Punkte ein Sport-Kandidat maximal schwaecher sein darf, um bei
# Unterdeckung dennoch vorgezogen zu werden - ein deutlich staerkerer News-Push
# wird nie verdraengt.
PUSH_TEAMS_SPORT_MIN_PER_DAY: int = _env_int("PUSH_TEAMS_SPORT_MIN_PER_DAY", 4)
PUSH_TEAMS_SPORT_MAX_PER_DAY: int = _env_int("PUSH_TEAMS_SPORT_MAX_PER_DAY", 5)
PUSH_TEAMS_SPORT_SHARE_LOW: float = _env_float("PUSH_TEAMS_SPORT_SHARE_LOW", 0.27)
PUSH_TEAMS_SPORT_SHARE_HIGH: float = _env_float("PUSH_TEAMS_SPORT_SHARE_HIGH", 0.40)
PUSH_TEAMS_SPORT_PREFERENCE_BAND: float = _env_float(
    "PUSH_TEAMS_SPORT_PREFERENCE_BAND",
    4.0,
)
# LLM-Slot-Fit: Vor dem Versand prueft ein LLM, ob der empfohlene Push inhaltlich
# in den AKTUELLEN Zeitslot passt oder auf eine spaetere Hot Hour warten sollte
# (basierend auf den historischen Opening-Rate-Baselines pro Stunde/Wochentag).
# Zeitkritische/Breaking-Meldungen werden nie zurueckgehalten. Fail-safe: ohne
# LLM-Key nur Annotation, kein Zurueckhalten.
PUSH_TEAMS_SLOT_FIT_LLM_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_SLOT_FIT_LLM_ENABLED",
    True,
)
# Ein deutlich besserer Slot rechtfertigt Zurueckhalten nur, wenn er innerhalb
# dieser Stunden kommt (sonst sofort senden).
PUSH_TEAMS_SLOT_FIT_MAX_DEFER_HOURS: int = _env_int(
    "PUSH_TEAMS_SLOT_FIT_MAX_DEFER_HOURS",
    3,
)
# Meldungen aelter als dies werden nie fuer einen besseren Slot zurueckgehalten
# (Aktualitaet schlaegt Timing-Optimierung).
PUSH_TEAMS_SLOT_FIT_MAX_ARTICLE_AGE_HOURS: float = float(
    _env_int("PUSH_TEAMS_SLOT_FIT_MAX_ARTICLE_AGE_HOURS", 3)
)
# Lokales, deterministisches Pruefkollegium vor jedem Teams-Versand. Die
# Spezialpruefer teilen nur einen fluechtigen Artikel-Snapshot und rufen weder
# externe Modelle noch weitere Cloud-Dienste auf.
PUSH_TEAMS_AGENT_REVIEW_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_AGENT_REVIEW_ENABLED",
    False,
)
PUSH_TEAMS_AGENT_REVIEW_MIN_EVIDENCE_APPROVALS: int = _env_int(
    "PUSH_TEAMS_AGENT_REVIEW_MIN_EVIDENCE_APPROVALS",
    3,
)
PUSH_TEAMS_AGENT_REVIEW_MIN_CONSENSUS_SCORE: float = _env_float(
    "PUSH_TEAMS_AGENT_REVIEW_MIN_CONSENSUS_SCORE",
    60.0,
)
PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY: float = _env_float(
    "PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY",
    72.0,
)
PUSH_TEAMS_AGENT_REVIEW_MAX_LATENCY_MS: int = _env_int(
    "PUSH_TEAMS_AGENT_REVIEW_MAX_LATENCY_MS",
    50,
)
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
# Abgleich gegen ECHTE Live-Pushes: gleiche Artikel-URLs oder CMS-IDs sind ein hartes
# Dubletten-Gate. Aehnliche URL-Slugs/Titel bleiben Vergleichssignale, damit eine
# neue Entwicklung unter einer anderen Artikel-URL nicht pauschal gesperrt wird.
PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS: float = _env_float(
    "PUSH_TEAMS_PUSHED_TOPIC_WINDOW_HOURS",
    36.0,
)
PUSH_TEAMS_MIN_SELECTION_MARGIN: float = _env_float("PUSH_TEAMS_MIN_SELECTION_MARGIN", 5.0)
PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER: float = _env_float(
    "PUSH_TEAMS_SELECTION_CLEAR_EDITORIAL_BUFFER",
    6.0,
)
# Score-dominierte Response-Optimierung: Der Push-Score bleibt das klare
# Leitsignal. OR x Reichweite differenziert nur innerhalb des engen Score-Felds
# und ist zur Laufzeit auf 10 Prozent begrenzt.
PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_VISIT_OPTIMIZATION_ENABLED",
    True,
)
PUSH_TEAMS_VISIT_SELECTION_WEIGHT: float = _env_float(
    "PUSH_TEAMS_VISIT_SELECTION_WEIGHT",
    0.10,
)
PUSH_TEAMS_DEFAULT_REACH: int = _env_int("PUSH_TEAMS_DEFAULT_REACH", 250000)
PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED: bool = _env_flag(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_ENABLED",
    True,
)
# Legacy-Kompatibilitaet: Rueckstand senkt die Schwelle nicht mehr; Vorsprung
# darf sie weiterhin zum Schutz vor Push-Muedigkeit anheben.
PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP: float = _env_float(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_DROP",
    0.0,
)
PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE: float = _env_float(
    "PUSH_TEAMS_DYNAMIC_THRESHOLD_MAX_RISE",
    14.0,
)
# Aktives Push-Fenster (Berlin-Stunden) fuer Slot-Planung und Pace-Berechnung.
# Ende 22: die letzte planbare Stunde ist 22:xx, ab 23:00 gilt die Ruhezeit.
PUSH_TEAMS_ACTIVE_HOURS_START: int = _env_int("PUSH_TEAMS_ACTIVE_HOURS_START", 6)
PUSH_TEAMS_ACTIVE_HOURS_END: int = _env_int("PUSH_TEAMS_ACTIVE_HOURS_END", 22)
PUSH_TEAMS_BREAKING_OVERRIDE: bool = _env_flag("PUSH_TEAMS_BREAKING_OVERRIDE", True)
PUSH_TEAMS_BREAKING_MIN_SCORE: float = _env_float("PUSH_TEAMS_BREAKING_MIN_SCORE", 72.0)
PUSH_TEAMS_BREAKING_MIN_OR: float = _env_float("PUSH_TEAMS_BREAKING_MIN_OR", 4.0)
PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH: int = _env_int(
    "PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH",
    45,
)
PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS: int = _env_int("PUSH_TEAMS_MAX_ARTICLE_AGE_HOURS", 24)
PUSH_TEAMS_MAX_PUSHES_LAST_6H: int = _env_int("PUSH_TEAMS_MAX_PUSHES_LAST_6H", 8)
PUSH_TEAMS_CHECK_INTERVAL_SECONDS: int = _env_int("PUSH_TEAMS_CHECK_INTERVAL_SECONDS", 60)
PUSH_TEAMS_CANDIDATE_LIMIT: int = _env_int("PUSH_TEAMS_CANDIDATE_LIMIT", 80)
TRUSTED_PROXY_CIDRS: list[str] = _csv_env(
    "TRUSTED_PROXY_CIDRS",
    "127.0.0.1/32,::1/128",
)
