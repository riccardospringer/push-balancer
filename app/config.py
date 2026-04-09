"""app/config.py — Alle Konfigurationswerte aus Umgebungsvariablen.

Entspricht den os.environ.get()-Aufrufen aus push-balancer-server.py.
Beim Import wird automatisch eine .env-Datei im Projektverzeichnis geladen
(selbes Verhalten wie im Monolith).
"""
import os
import socket
import logging

log = logging.getLogger("push-balancer")

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
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AI_API_KEY", "")

# ── BILD APIs ──────────────────────────────────────────────────────────────
PUSH_API_BASE: str = os.environ.get("PUSH_API_BASE", "http://push-frontend.bildcms.de")
BILD_SITEMAP: str = os.environ.get("BILD_SITEMAP_URL", "https://www.bild.de/sitemap-news.xml")

# ── Sync / Render ──────────────────────────────────────────────────────────
SYNC_SECRET: str = os.environ.get("PUSH_SYNC_SECRET", "bild-push-sync-2026")
RENDER_SYNC_URL: str = os.environ.get("RENDER_SYNC_URL", "")

# ── Adobe Analytics ────────────────────────────────────────────────────────
ADOBE_CLIENT_ID: str = os.environ.get("ADOBE_CLIENT_ID", "")
ADOBE_CLIENT_SECRET: str = os.environ.get("ADOBE_CLIENT_SECRET", "")
ADOBE_COMPANY_ID: str = os.environ.get("ADOBE_GLOBAL_COMPANY_ID", "axelsp2")
ADOBE_RSID: str = "axelspringerbild"
ADOBE_TOKEN_URL: str = "https://ims-na1.adobelogin.com/ims/token/v3"
ADOBE_API_BASE: str = "https://analytics.adobe.io/api"

# ── Render-Erkennung ───────────────────────────────────────────────────────
IS_RENDER: bool = os.environ.get("RENDER", "").lower() == "true"

# ── Dateipfade ─────────────────────────────────────────────────────────────
SERVE_DIR: str = os.path.join(_APP_DIR, "dist-frontend")  # React App Build
# DB_PATH env var → Render nutzt /data (persistent disk), lokal .push_history.db
PUSH_DB_PATH: str = os.environ.get(
    "DB_PATH",
    os.path.join(_APP_DIR, ".push_history.db"),
)
SNAPSHOT_PATH: str = os.path.join(_APP_DIR, "push-snapshot.json")

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
# Wenn nicht gesetzt, sind Admin-Endpoints im lokalen Betrieb offen (Legacy).
# In Produktionsumgebungen MUSS dieser Wert gesetzt werden!
ADMIN_API_KEY: str = os.environ.get("ADMIN_API_KEY", "")

# ── Dev Mode (Tunnel-Wildcards für CORS nur im lokalen Betrieb) ────────────
DEV_MODE: bool = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")
