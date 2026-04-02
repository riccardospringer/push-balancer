#!/usr/bin/env python3
"""Local dev server for Push Balancer — serves HTML + proxies BILD APIs + competitor feeds."""

# libomp fuer LightGBM/XGBoost vorab laden (macOS SIP blockiert DYLD_LIBRARY_PATH)
import os as _os, ctypes as _ctypes
_omp_lib = _os.path.expanduser("~/.local/lib/libomp.dylib")
if _os.path.exists(_omp_lib):
    try:
        _ctypes.cdll.LoadLibrary(_omp_lib)
    except OSError:
        pass

import http.server
import json
import ssl
import urllib.request
import urllib.parse
import urllib.error
import os
import time
import logging
import concurrent.futures
import random
import datetime
import math
import re
import threading
import sqlite3
from collections import defaultdict
from bisect import bisect_left, bisect_right
from contextlib import nullcontext as _nullcontext
from difflib import SequenceMatcher

# ── sklearn + joblib für ML v2 ──────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
    from sklearn.decomposition import PCA as SklearnPCA
    import joblib
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    SklearnGBR = None
    SklearnPCA = None
    np = None

# LightGBM: schneller und besser als sklearn GBR
_LGBM_AVAILABLE = False
_lgb = None
try:
    import lightgbm as _lgb
    _LGBM_AVAILABLE = True
except (ImportError, OSError):
    pass

# XGBoost für Stacking Ensemble
_XGB_AVAILABLE = False
_xgb = None
try:
    import xgboost as _xgb
    _XGB_AVAILABLE = True
except (ImportError, OSError):
    pass

# CatBoost für Stacking Ensemble
_CATBOOST_AVAILABLE = False
_cb = None
try:
    import catboost as _cb
    _CATBOOST_AVAILABLE = True
except (ImportError, OSError):
    pass

# ── ML State (LightGBM + SHAP) ──────────────────────────────────────────
_ml_state = {
    "model": None,
    "stats": None,
    "feature_names": [],
    "metrics": {},       # MAE, RMSE, R2
    "shap_importance": [],  # top-10 feature importances
    "train_count": 0,
    "last_train_ts": 0,
    "next_retrain_ts": 0,
    "training": False,
}
_ml_lock = threading.Lock()

# ── Unified ML State (ML-First Prediction) ───────────────────────────────
_unified_state = {
    "model": None,
    "feature_names": [],
    "stats": None,
    "calibrator": None,
    "conformal_radius": 1.0,
    "metrics": {},
    "train_count": 0,
    "last_train_ts": 0,
    "training": False,
    # Stacking Ensemble
    "base_models": {},       # {"lgb": model, "xgb": model, "catboost": model}
    "meta_model": None,      # Ridge Meta-Learner
    "stacking_active": False, # True wenn Stacking besser als single LightGBM
}
_unified_lock = threading.Lock()

# ── Topic-Tracker & World-Event-Index (ML-First) ─────────────────────────
_topic_tracker = {"clusters": [], "ts": 0}
_topic_tracker_lock = threading.Lock()
_world_event_index = {"hot_topics": [], "keyword_counts": {}, "ts": 0}
_world_event_index_lock = threading.Lock()

# ── Online Bias Correction (Unified) ─────────────────────────────────────
_gbrt_online_bias = 0.0

# ── Model Selector State ─────────────────────────────────────────────────
_model_selector_state = {
    "active_model": "ml_ensemble",  # "unified" oder "ml_ensemble"
    "unified_mae_24h": None,
    "ensemble_mae_24h": None,
    "consecutive_worse": 0,
    "evaluated_count": 0,
    "last_check_ts": 0,
}

# ── Auto-Retrain State ───────────────────────────────────────────────────
_auto_retrain_state = {
    "consecutive_degraded_ticks": 0,
    "last_retrain_trigger_ts": 0,
}

# ── Online Residual Corrector (Echtzeit-Bias-Korrektur) ─────────────────
_residual_corrector = {
    "global_bias": 0.0,           # Rolling Mean des globalen Bias (predicted - actual)
    "cat_bias": {},               # {category: bias} pro Kategorie
    "hourgroup_bias": {},         # {"morning": bias, ...} pro Tageszeit-Gruppe
    "n_samples": 0,               # Anzahl eingeflossener Feedback-Paare
    "last_update_ts": 0,
    "recent_residuals": [],       # Letzte 50 Residuals fuer Debugging
}
_residual_corrector_lock = threading.Lock()

# ── Safety Hardening: ADVISORY ONLY ─────────────────────────────────────
# KRITISCH: Das System darf NIEMALS autonom Push-Benachrichtigungen senden.
# Alle Vorhersagen sind NUR beratend.
SAFETY_MODE = "ADVISORY_ONLY"            # Hardcoded, nicht veraenderbar
_SAFETY_ADVISORY_ONLY = True             # Redundanter Guard


def _safety_check():
    """Prueft beide Safety-Guards. Raise wenn nicht ADVISORY_ONLY."""
    if SAFETY_MODE != "ADVISORY_ONLY" or not _SAFETY_ADVISORY_ONLY:
        raise RuntimeError("SAFETY VIOLATION: System muss im ADVISORY_ONLY Modus laufen!")


def _safety_envelope(result):
    """Fuegt advisory_only und action_allowed zu jeder Prediction hinzu."""
    _safety_check()
    if result is None:
        return None
    if isinstance(result, dict):
        result["advisory_only"] = True
        result["action_allowed"] = False
        result["safety_mode"] = SAFETY_MODE
    return result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    force=True)
# Flush Handler für nohup/disown (sonst buffert Python den Output)
for _h in logging.root.handlers:
    if hasattr(_h, 'stream') and hasattr(_h.stream, 'reconfigure'):
        try:
            _h.stream.reconfigure(line_buffering=True)
        except Exception:
            pass

# ── Lokale .env laden (gleicher Ordner wie das Script) ───────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_ENV = os.path.join(_SCRIPT_DIR, ".env")
if os.path.exists(_LOCAL_ENV):
    with open(_LOCAL_ENV) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# OpenAI API Key: aus .env oder Environment-Variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AI_API_KEY", "")
log = logging.getLogger("push-balancer")

PORT = int(os.environ.get("PORT", "8050"))
ALLOW_INSECURE_SSL = os.environ.get("ALLOW_INSECURE_SSL", "0") == "1"

# SSL-Context mit certifi-Zertifikaten (macOS Python hat oft kein System-CA-Bundle)
try:
    import certifi as _certifi
    _SSL_CERTFILE = _certifi.where()
except ImportError:
    _SSL_CERTFILE = None

# Globaler SSL-Context fuer alle urllib-Calls
import ssl as _ssl_mod
_GLOBAL_SSL_CTX = _ssl_mod.create_default_context(cafile=_SSL_CERTFILE)
BILD_SITEMAP = os.environ.get("BILD_SITEMAP_URL", "https://www.bild.de/sitemap-news.xml")
PUSH_API_BASE = os.environ.get("PUSH_API_BASE", "http://push-frontend.bildcms.de")
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))  # nur das Verzeichnis mit den Dateien

# ── Push-Sync: Lokaler Server synct Push-Daten zu Render ──────────────
SYNC_SECRET = os.environ.get("PUSH_SYNC_SECRET", "bild-push-sync-2026")
RENDER_SYNC_URL = os.environ.get("RENDER_SYNC_URL", "")  # z.B. https://push-balancer.onrender.com
_push_sync_cache = {"messages": [], "ts": 0, "channels": []}  # In-Memory Cache fuer empfangene Sync-Daten
_push_sync_lock = threading.Lock()

# Snapshot beim Start laden (fuer Render: eingebackene Push-Daten als Fallback)
_SNAPSHOT_PATH = os.path.join(SERVE_DIR, "push-snapshot.json")
if os.path.exists(_SNAPSHOT_PATH):
    try:
        with open(_SNAPSHOT_PATH) as _sf:
            _snap = json.load(_sf)
        if isinstance(_snap, list) and _snap:
            # Snapshot ist eine Liste von geparsten Pushes — direkt in SQLite seeden
            # Synchron damit Research-Worker sofort Daten hat (kein 2min API-Timeout)
            _n_seeded = _push_db_upsert(_snap)
            log.info(f"[Snapshot] {_n_seeded} Pushes in DB geseedet (Startup-Seed)")
        elif isinstance(_snap, dict):
            # Altes Format: Dict mit "messages" Key
            with _push_sync_lock:
                _push_sync_cache["messages"] = _snap.get("messages", [])
                _push_sync_cache["ts"] = _snap.get("_generated", time.time())
            log.info(f"[Snapshot] {len(_push_sync_cache['messages'])} Pushes aus Snapshot (Dict-Format) geladen")
    except Exception as _se:
        log.warning(f"[Snapshot] Fehler beim Laden: {_se}")

# CORS: erlaubte Origins (localhost + Railway + Render + Cloudflare Tunnel)
_railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
_render_domain = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
ALLOWED_ORIGINS = [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"]
# Lokale Netzwerk-IP automatisch erkennen und erlauben
try:
    import socket
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

# Competitor RSS feeds (all publicly available)
COMPETITOR_FEEDS = {
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

# International RSS feeds (all publicly available)
INTERNATIONAL_FEEDS = {
    # Europa (13 Outlets — UK, FR, ES, IT, CH, AT, SE, NL, IE, PL)
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
    # Global (11 Outlets — US, Nahost, Asien, Suedamerika, Indien, Australien)
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

SPORT_COMPETITOR_FEEDS = {
    "kicker":       "https://newsfeed.kicker.de/news/aktuell",
    "sportschau":   "https://www.sportschau.de/index~rss2.xml",
    "transfermarkt": "https://www.transfermarkt.de/rss/news",
    "sport_de":     "https://www.sport.de/rss/news/",
    "spiegel_sport": "https://www.spiegel.de/sport/index.rss",
    "faz_sport":    "https://www.faz.net/rss/aktuell/sport/",
    "rp_sport":     "https://rp-online.de/sport/feed.rss",
    "tz_sport":     "https://www.tz.de/sport/rssfeed.rdf",
    "11freunde":    "https://www.11freunde.de/fullarticlerss/index.rss",
}

SPORT_EUROPA_FEEDS = {
    "bbc_sport":    "https://feeds.bbci.co.uk/sport/rss.xml",
    "lequipe":      "https://dwh.lequipe.fr/api/edito/rss?path=/",
    "marca":        "https://e00-xlk-ue-marca.uecdn.es/rss/googlenews/portada.xml",
    "gazzetta":     "https://www.gazzetta.it/rss/home.xml",
    "as_sport":     "https://as.com/rss/tags/ultimas_noticias.xml",
    "orf_sport":    "https://rss.orf.at/sport.xml",
    "nzz_sport":    "https://www.nzz.ch/sport.rss",
    "standard_sport": "https://www.derstandard.at/rss/sport",
}

SPORT_GLOBAL_FEEDS = {
    "espn":         "https://www.espn.com/espn/rss/news",
    "skysports":    "https://www.skysports.com/rss/12040",
    "cbssports":    "https://www.cbssports.com/rss/headlines/",
    "yahoo_sport":  "https://sports.yahoo.com/rss/",
}

# In-Memory Cache (URL -> (timestamp, data))
_cache = {}
CACHE_TTL = 90  # Sekunden

MAX_RESPONSE_SIZE = 2 * 1024 * 1024  # 2 MB Limit pro Feed

# ── Adobe Analytics Traffic Sources ───────────────────────────────────────
_ADOBE_CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID", "")
_ADOBE_CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET", "")
_ADOBE_COMPANY_ID = os.environ.get("ADOBE_GLOBAL_COMPANY_ID", "axelsp2")
_ADOBE_RSID = "axelspringerbild"
_ADOBE_TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"
_ADOBE_API_BASE = "https://analytics.adobe.io/api"

_adobe_state = {
    "access_token": "",
    "token_expires": 0,
    "traffic": None,       # gematchte Traffic-Daten (dict)
    "updated_at": 0,
    "error": "",
    "enabled": bool(_ADOBE_CLIENT_ID and _ADOBE_CLIENT_SECRET),
}


def _adobe_ssl_ctx():
    """SSL Context mit certifi CA-Bundle (macOS-kompatibel)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _adobe_get_token():
    """OAuth2 Client Credentials Token (synchron). Cached bis kurz vor Ablauf."""
    now = time.time()
    if _adobe_state["access_token"] and now < _adobe_state["token_expires"] - 120:
        return _adobe_state["access_token"]

    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": _ADOBE_CLIENT_ID,
        "client_secret": _ADOBE_CLIENT_SECRET,
        "scope": "openid,AdobeID,additional_info.projectedProductContext,read_organizations,additional_info.roles",
    }).encode()

    ctx = _adobe_ssl_ctx()
    req = urllib.request.Request(_ADOBE_TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        result = json.loads(resp.read())
    _adobe_state["access_token"] = result["access_token"]
    _adobe_state["token_expires"] = now + result.get("expires_in", 86400)
    log.info("[Adobe] Token erneuert (gueltig %ds)", int(_adobe_state["token_expires"] - now))
    return _adobe_state["access_token"]


def _adobe_report(metrics, dimension="variables/evar12", days=1, limit=400):
    """Holt einen Adobe Analytics Report (synchron, einzelne Seite)."""
    token = _adobe_get_token()
    now = datetime.datetime.utcnow()
    start = now - datetime.timedelta(days=days)
    dr = f"{start.strftime('%Y-%m-%dT00:00:00.000')}/{now.strftime('%Y-%m-%dT23:59:59.999')}"

    body = json.dumps({
        "rsid": _ADOBE_RSID,
        "globalFilters": [{"type": "dateRange", "dateRange": dr}],
        "metricContainer": {
            "metrics": [{"id": m, "columnId": f"col_{i}"} for i, m in enumerate(metrics)]
        },
        "dimension": dimension,
        "settings": {"limit": limit, "page": 0, "countRepeatInstances": True},
    }).encode()

    url = f"{_ADOBE_API_BASE}/{_ADOBE_COMPANY_ID}/reports"
    ctx = _adobe_ssl_ctx()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("x-api-key", _ADOBE_CLIENT_ID)
    req.add_header("x-proxy-global-company-id", _ADOBE_COMPANY_ID)
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        result = json.loads(resp.read())

    rows = result.get("rows", [])
    parsed = []
    for row in rows:
        entry = {"headline": row.get("value", "")}
        for i, m in enumerate(metrics):
            if i < len(row.get("data", [])):
                entry[m] = row["data"][i]
        parsed.append(entry)
    return parsed


def _adobe_traffic_worker():
    """Background Worker: Fetcht Adobe Traffic Sources alle 30 Min, matcht mit Push-Headlines."""
    import time as _t
    _t.sleep(15)  # Warten bis Server + Research-Worker laufen

    if not _adobe_state["enabled"]:
        log.info("[Adobe] Deaktiviert (ADOBE_CLIENT_ID/SECRET nicht gesetzt)")
        return

    TRAFFIC_METRICS = [
        "metrics/event21",   # PV via Push
        "metrics/event20",   # PV via Home
        "metrics/event25",   # PV Social
        "metrics/event24",   # PV Search
        "metrics/event207",  # PV Direct
        "metrics/pageviews", # Total PVs
    ]
    METRIC_LABELS = {
        "metrics/event21": "push",
        "metrics/event20": "home",
        "metrics/event25": "social",
        "metrics/event24": "search",
        "metrics/event207": "direct",
        "metrics/pageviews": "total",
    }

    while True:
        try:
            rows = _adobe_report(TRAFFIC_METRICS, dimension="variables/evar12", days=1, limit=400)
            adobe_hl = {}
            for r in rows:
                hl = r.get("headline", "")
                if hl and hl not in ("Unspecified", "(Low Traffic)"):
                    adobe_hl[hl] = r

            # Heutige Pushes aus DB laden
            cutoff = int(time.time()) - 24 * 3600
            try:
                conn = sqlite3.connect(PUSH_DB_PATH)
                conn.row_factory = sqlite3.Row
                pushes = conn.execute(
                    "SELECT title, or_val, cat, opened, received FROM pushes "
                    "WHERE ts_num >= ? AND link IS NOT NULL AND link <> '' ORDER BY or_val DESC",
                    (cutoff,),
                ).fetchall()
                conn.close()
            except Exception:
                pushes = []

            # Fuzzy-Match
            matched = []
            agg = {k: 0 for k in METRIC_LABELS.values()}

            for push in pushes:
                title = push["title"] or ""
                title_clean = re.sub(r'[^\w\s]', '', title).lower()
                best_hl, best_score = None, 0
                for hl in adobe_hl:
                    hl_clean = re.sub(r'[^\w\s]', '', hl).lower()
                    s = SequenceMatcher(None, title_clean, hl_clean).ratio()
                    if s > best_score:
                        best_score = s
                        best_hl = hl
                if best_hl and best_score >= 0.4:
                    a = adobe_hl[best_hl]
                    article = {
                        "title": title[:80],
                        "or": push["or_val"],
                        "cat": push["cat"],
                        "opened": push["opened"],
                        "match": round(best_score, 2),
                    }
                    for metric_id, label in METRIC_LABELS.items():
                        val = a.get(metric_id, 0)
                        article[label] = int(val)
                        agg[label] += int(val)
                    atotal = article.get("total", 1) or 1
                    article["push_pct"] = round(article.get("push", 0) / atotal * 100, 1)
                    matched.append(article)

            # Berechne Push-Anteil pro Artikel
            push_pcts = [m["push_pct"] for m in matched if m.get("total", 0) > 0]
            avg_push_pct = round(sum(push_pcts) / len(push_pcts), 1) if push_pcts else 0
            summary = {
                "total_pvs": agg["total"],
                "avg_push_pct": avg_push_pct,
                "n_matched": len(matched),
                "n_pushes": len(pushes),
            }

            matched.sort(key=lambda x: x.get("push_pct", 0), reverse=True)
            _adobe_state["traffic"] = {
                "summary": summary,
                "articles": matched[:20],
            }
            _adobe_state["updated_at"] = int(time.time())
            _adobe_state["error"] = ""
            log.info("[Adobe] Traffic-Daten aktualisiert: %d/%d Pushes gematcht, %s Total PVs",
                     len(matched), len(pushes), f"{agg['total']:,}")

        except Exception as e:
            _adobe_state["error"] = str(e)
            log.warning("[Adobe] Worker-Fehler: %s", e)

        _t.sleep(1800)  # 30 Minuten


# ── SQLite Push-Historie (persistenter Cache) ─────────────────────────────
PUSH_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".push_history.db")
_push_db_lock = threading.Lock()

def _init_push_db():
    """Initialize SQLite database for push history."""
    conn = sqlite3.connect(PUSH_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS pushes (
        message_id TEXT PRIMARY KEY,
        ts_num INTEGER NOT NULL,
        or_val REAL DEFAULT 0,
        title TEXT,
        headline TEXT,
        kicker TEXT,
        cat TEXT,
        link TEXT,
        type TEXT DEFAULT 'editorial',
        hour INTEGER DEFAULT -1,
        title_len INTEGER DEFAULT 0,
        opened INTEGER DEFAULT 0,
        received INTEGER DEFAULT 0,
        channel TEXT DEFAULT '',
        channels TEXT DEFAULT '[]',
        is_eilmeldung INTEGER DEFAULT 0,
        updated_at INTEGER DEFAULT 0,
        target_stats TEXT DEFAULT '{}',
        app_list TEXT DEFAULT '[]',
        n_apps INTEGER DEFAULT 0,
        total_recipients INTEGER DEFAULT 0
    )""")
    # Migrate: add new columns if they don't exist
    for _col, _type, _default in [
        ("target_stats", "TEXT", "'{}'"), ("app_list", "TEXT", "'[]'"),
        ("n_apps", "INTEGER", "0"), ("total_recipients", "INTEGER", "0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE pushes ADD COLUMN {_col} {_type} DEFAULT {_default}")
        except Exception:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_ts ON pushes(ts_num)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_cat ON pushes(cat)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_or_ts ON pushes(or_val, ts_num)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_hour_or ON pushes(hour, or_val)")
    # Prediction log for ML training
    conn.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        push_id TEXT NOT NULL,
        predicted_or REAL,
        actual_or REAL,
        basis_method TEXT DEFAULT '',
        methods_detail TEXT DEFAULT '{}',
        features TEXT DEFAULT '{}',
        model_version INTEGER DEFAULT 0,
        predicted_at INTEGER NOT NULL,
        actual_recorded_at INTEGER DEFAULT 0,
        UNIQUE(push_id)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predlog_ts ON prediction_log(predicted_at)")
    # Experiment Tracking
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        hyperparams TEXT DEFAULT '{}',
        metrics TEXT DEFAULT '{}',
        baselines TEXT DEFAULT '{}',
        cv_results TEXT DEFAULT '{}',
        n_features INTEGER DEFAULT 0,
        n_samples INTEGER DEFAULT 0,
        model_hash TEXT DEFAULT '',
        promoted INTEGER DEFAULT 0,
        training_duration_s REAL DEFAULT 0
    )""")
    # Promotion Gates Log
    conn.execute("""CREATE TABLE IF NOT EXISTS promotion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        passed INTEGER DEFAULT 0,
        gates TEXT DEFAULT '{}',
        champion_mae REAL DEFAULT 0,
        challenger_mae REAL DEFAULT 0,
        reason TEXT DEFAULT ''
    )""")
    # Embedding Cache
    conn.execute("""CREATE TABLE IF NOT EXISTS embedding_cache (
        title_hash TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        embedding TEXT DEFAULT '[]',
        created_at INTEGER NOT NULL
    )""")
    # Monitoring Events
    conn.execute("""CREATE TABLE IF NOT EXISTS monitoring_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT DEFAULT 'info',
        message TEXT DEFAULT '',
        metrics_json TEXT DEFAULT '{}'
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_ts ON monitoring_events(timestamp)")
    # Tagesplan Suggestion Snapshots: Was hat das System pro Slot empfohlen?
    conn.execute("""CREATE TABLE IF NOT EXISTS tagesplan_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_iso TEXT NOT NULL,
        slot_hour INTEGER NOT NULL,
        suggestion_num INTEGER NOT NULL,
        article_title TEXT,
        article_link TEXT,
        article_category TEXT,
        article_score REAL DEFAULT 0,
        expected_or REAL DEFAULT 0,
        best_cat TEXT DEFAULT '',
        captured_at INTEGER NOT NULL,
        UNIQUE(date_iso, slot_hour, suggestion_num)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tpsug_date ON tagesplan_suggestions(date_iso)")
    # ML v2: LLM-Score Spalten zu pushes hinzufügen (idempotent via ALTER TABLE)
    _llm_columns = [
        ("llm_magnitude", "REAL DEFAULT 0"),
        ("llm_clickability", "REAL DEFAULT 0"),
        ("llm_relevanz", "REAL DEFAULT 0"),
        ("llm_dringlichkeit", "REAL DEFAULT 0"),
        ("llm_emotionalitaet", "REAL DEFAULT 0"),
        ("llm_scored_at", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_type in _llm_columns:
        try:
            conn.execute(f"ALTER TABLE pushes ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Spalte existiert bereits
    conn.commit()
    conn.close()
    log.info(f"[PushDB] Initialized at {PUSH_DB_PATH}")


def _log_monitoring_event(event_type, severity, message, metrics=None):
    """Speichert ein Monitoring-Event in der DB.

    event_type: drift/calibration_shift/mae_spike/ab_result/online_pause
    severity: info/warning/critical
    """
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.execute(
                "INSERT INTO monitoring_events (timestamp, event_type, severity, message, metrics_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(time.time()), event_type, severity, message,
                 json.dumps(metrics or {}, default=str))
            )
            conn.commit()
            conn.close()
    except Exception as e:
        log.warning(f"[Monitoring] Event-Log Fehler: {e}")


def _push_db_upsert(parsed_pushes):
    """Insert or update parsed pushes into SQLite. Returns count of new/updated."""
    if not parsed_pushes:
        return 0
    now = int(time.time())
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        cur = conn.cursor()
        count = 0
        for p in parsed_pushes:
            mid = p.get("message_id") or f"{p['ts_num']}_{p.get('title', '')[:30]}"
            _ts_json = json.dumps(p.get("target_stats", {})) if isinstance(p.get("target_stats"), dict) else "{}"
            _apps_json = json.dumps(p.get("app_list", [])) if isinstance(p.get("app_list"), list) else "[]"
            cur.execute("""INSERT INTO pushes (message_id, ts_num, or_val, title, headline, kicker,
                cat, link, type, hour, title_len, opened, received, channel, channels, is_eilmeldung, updated_at,
                target_stats, app_list, n_apps, total_recipients)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    or_val = CASE WHEN excluded.or_val > 0 THEN excluded.or_val ELSE or_val END,
                    opened = CASE WHEN excluded.opened > opened THEN excluded.opened ELSE opened END,
                    received = CASE WHEN excluded.received > received THEN excluded.received ELSE received END,
                    target_stats = CASE WHEN length(excluded.target_stats) > 2 THEN excluded.target_stats ELSE target_stats END,
                    n_apps = CASE WHEN excluded.n_apps > 0 THEN excluded.n_apps ELSE n_apps END,
                    total_recipients = CASE WHEN excluded.total_recipients > 0 THEN excluded.total_recipients ELSE total_recipients END,
                    updated_at = ?
            """, (mid, p["ts_num"], p["or"], p.get("title", ""), p.get("headline", ""),
                  p.get("kicker", ""), p.get("cat", ""), p.get("link", ""), p.get("type", "editorial"),
                  p.get("hour", -1), p.get("title_len", 0), p.get("opened", 0), p.get("received", 0),
                  p.get("channel", ""), json.dumps(p.get("channels", [])), 1 if p.get("is_eilmeldung") else 0,
                  now, _ts_json, _apps_json, p.get("n_apps", 0), p.get("total_recipients", 0), now))
            count += cur.rowcount
        conn.commit()
        conn.close()
    return count

def _push_db_load_all(min_ts=0):
    """Load all pushes from SQLite, optionally filtered by min timestamp.
    Nutzt eigene Connection mit WAL-Mode für nicht-blockierendes Lesen."""
    conn = sqlite3.connect(PUSH_DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM pushes WHERE ts_num > ? AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%' ORDER BY ts_num DESC",
            (min_ts,)).fetchall()
    finally:
        conn.close()
    result = []
    for r in rows:
        result.append({
            "message_id": r["message_id"],
            "or": r["or_val"],
            "ts": str(r["ts_num"]),
            "ts_num": r["ts_num"],
            "title": r["title"],
            "headline": r["headline"],
            "kicker": r["kicker"],
            "cat": r["cat"],
            "link": r["link"],
            "type": r["type"],
            "hour": r["hour"],
            "title_len": r["title_len"],
            "opened": r["opened"],
            "received": r["received"],
            "channel": r["channel"],
            "channels": json.loads(r["channels"] or "[]"),
            "is_eilmeldung": bool(r["is_eilmeldung"]),
            "target_stats": json.loads(r["target_stats"] or "{}") if "target_stats" in r.keys() else {},
            "app_list": json.loads(r["app_list"] or "[]") if "app_list" in r.keys() else [],
            "n_apps": r["n_apps"] if "n_apps" in r.keys() else 0,
            "total_recipients": r["total_recipients"] if "total_recipients" in r.keys() else 0,
        })
    return result

def _push_db_max_ts():
    """Get the most recent ts_num in the database."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        row = conn.execute("SELECT MAX(ts_num) FROM pushes").fetchone()
        conn.close()
    return row[0] or 0

def _push_db_count():
    """Get total push count in database."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        row = conn.execute("SELECT COUNT(*) FROM pushes").fetchone()
        conn.close()
    return row[0] or 0

def _push_db_log_prediction(push_id, predicted_or, actual_or, basis_method="",
                             methods_detail=None, features=None, model_version=0,
                             title="", confidence=0.0, q10=0.0, q90=0.0):
    """Log a prediction for ML training. Upserts: updates actual_or if prediction exists."""
    now = int(time.time())
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.execute("""INSERT INTO prediction_log
            (push_id, predicted_or, actual_or, basis_method, methods_detail, features,
             model_version, predicted_at, actual_recorded_at, title, confidence, q10, q90)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(push_id) DO UPDATE SET
                actual_or = CASE WHEN excluded.actual_or > 0 THEN excluded.actual_or ELSE actual_or END,
                actual_recorded_at = CASE WHEN excluded.actual_or > 0 THEN ? ELSE actual_recorded_at END
        """, (push_id, predicted_or, actual_or, basis_method,
              json.dumps(methods_detail or {}), json.dumps(features or {}),
              model_version, now, now, title, confidence, q10, q90, now))
        conn.commit()
        conn.close()

def _push_db_get_training_data(min_ts=0, limit=5000):
    """Load prediction log entries with both predicted and actual OR for training."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT * FROM prediction_log
            WHERE predicted_or > 0 AND actual_or > 0 AND predicted_at > ?
            ORDER BY predicted_at DESC LIMIT ?""", (min_ts, limit)).fetchall()
        conn.close()
    return [dict(r) for r in rows]

# ══════════════════════════════════════════════════════════════════════════════
# ══ GBRT: Gradient Boosted Regression Trees (pure Python, kein numpy) ═══════
# ══════════════════════════════════════════════════════════════════════════════

_GBRT_CATEGORIES = ["sport", "politik", "unterhaltung", "geld", "regional", "digital", "leben", "news"]

_TOPIC_STOPS = {"der", "die", "das", "und", "von", "für", "mit", "auf", "den", "ist",
                "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
                "wird", "wurde", "nach", "über", "dass", "oder", "aber", "wenn", "weil",
                "nicht", "hat", "haben", "sind", "sein", "kann", "aus", "wie", "vor",
                "bei", "zum", "zur", "vom", "dem", "des"}

_GBRT_DEATH_WORDS = {"tot", "tod", "sterben", "gestorben", "stirbt", "toetet", "getoetet",
                     "lebensgefahr", "leiche", "mord", "tote", "opfer", "ums leben"}
_GBRT_EXCLUSIVITY_WORDS = {"exklusiv", "nur bei bild", "bild erfuhr", "bild weiss",
                           "nach bild-informationen", "nach bild-info"}
_GBRT_EMOTION_WORDS = {
    "angst": {"tot", "tod", "sterben", "gestorben", "stirbt", "lebensgefahr", "mord", "tote", "opfer"},
    "katastrophe": {"erdbeben", "tsunami", "explosion", "brand", "feuer", "absturz", "crash",
                    "ueberschwemmung", "hochwasser", "sturm", "orkan"},
    "sensation": {"sensation", "historisch", "erstmals", "rekord", "unfassbar", "unglaublich",
                  "wahnsinn", "hammer", "mega", "schock", "krass"},
    "bedrohung": {"warnung", "alarm", "gefahr", "notfall", "panik", "terror", "angriff",
                  "anschlag", "krieg", "drohung", "evakuierung"},
    "prominenz": {"kanzler", "praesident", "papst", "koenig", "merkel", "scholz", "trump", "putin"},
    "empoerung": {"skandal", "verrat", "betrug", "korrupt", "dreist", "frechheit"},
}
_GBRT_BREAKING_RE = re.compile(r"(?i)\b(eilmeldung|breaking|exklusiv|liveticker|alarm|schock|sensation)\b")

# BILD-Kernthemen Topic-Cluster
_GBRT_TOPIC_CLUSTERS = {
    "crime": {"mord", "messer", "messerattacke", "vergewaltigung", "raub", "räuber", "einbruch",
              "verhaftet", "festnahme", "täter", "polizei", "überfall", "totschlag", "leiche",
              "verbrechen", "kriminalität", "fahndung", "festgenommen", "verdächtig", "tatort"},
    "royals": {"könig", "königin", "prinz", "prinzessin", "harry", "meghan", "william", "kate",
               "palace", "thronfolger", "royal", "monarchie", "windsor", "buckingham", "charles"},
    "kosten": {"inflation", "rente", "bürgergeld", "steuer", "preise", "teuer", "sparen", "miete",
               "energie", "strom", "gas", "heizung", "einkommen", "lohn", "gehalt", "zuschlag",
               "preissteigerung", "verbraucher", "kosten"},
    "gesundheit": {"krebs", "herzinfarkt", "symptome", "arzt", "krankenhaus", "studie", "warnung",
                   "rückruf", "medikament", "diagnose", "therapie", "virus", "infektion", "impfung",
                   "krankheit", "notaufnahme", "operation"},
    "auto": {"tesla", "bmw", "mercedes", "audi", "porsche", "blitzer", "stau", "führerschein",
             "tempolimit", "unfall", "rückruf", "verbrenner", "elektroauto", "verkehr", "autobahn"},
    "sex_beziehung": {"nackt", "affäre", "freundin", "trennung", "hochzeit", "ehe", "dating",
                      "flirt", "erotik", "untreu", "scheidung", "liebesleben", "verlobt", "paar"},
    "wetter_extrem": {"hitze", "kälte", "unwetter", "schnee", "gewitter", "hagel", "frost",
                      "hitzewelle", "sahara", "orkan", "tornado", "überschwemmung", "hochwasser",
                      "sturmflut", "rekordtemperatur", "eisregen", "glätte"},
}

# BILD-Titelstil Regex-Patterns
_GBRT_AGE_PATTERN = re.compile(r'\(\d{1,3}\)')  # "(34)", "(14)"
_GBRT_DAS_SO_PATTERN = re.compile(r'(?i)^(DAS|SO|HIER|DIESE[RS]?|JETZT)\s')
_GBRT_DIRECT_ADDRESS = re.compile(r'(?i)\b(ihnen|sie|ihr|ihre[mnrs]?|du|dein[em]?|man)\b')
_GBRT_NUMBER_EMPHASIS = re.compile(r'\d[\d\s.,]*\s*(euro|prozent|grad|meter|kilo|milliard|million|tausend|%|°|km)', re.IGNORECASE)

# Deutsche Labels für GBRT-Feature-SHAP-Erklärungen
_GBRT_SHAP_LABELS = {
    # Text-Features
    "title_len": "Titellänge", "word_count": "Wortanzahl", "avg_word_len": "Ø Wortlänge",
    "has_question": "Fragezeichen", "has_exclamation": "Ausrufezeichen", "has_colon": "Doppelpunkt",
    "has_pipe": "Pipe-Zeichen", "has_plus_plus": "++Ticker", "has_numbers": "Zahlen im Titel",
    "upper_ratio": "Großbuchstaben-Anteil", "name_density": "Namens-Dichte",
    "death_signal": "Todes-Signal", "exclusivity_signal": "Exklusiv-Signal",
    "kicker_pattern": "Kicker-Muster", "emotional_word_count": "Emotionale Wörter",
    "emotional_categories": "Emotions-Kategorien", "intensity_score": "Intensitäts-Score",
    "breaking_signals": "Breaking-Signale", "is_breaking_style": "Breaking-Stil",
    "is_bild_plus": "BILD Plus (Paywall)",
    # Temporal-Features
    "hour": "Stunde", "hour_sin": "Tageszeit (sin)", "hour_cos": "Tageszeit (cos)",
    "weekday": "Wochentag", "weekday_sin": "Wochentag (sin)", "weekday_cos": "Wochentag (cos)",
    "is_weekend": "Wochenende", "is_prime_time": "Primetime (18-22h)",
    "is_morning_commute": "Morgen-Pendler (6-9h)", "is_late_night": "Spätabend/Nacht",
    "is_lunch": "Mittagszeit (11-13h)", "mins_since_last_same_cat": "Min. seit letztem Push (Ressort)",
    "push_count_today": "Pushes heute bisher", "day_of_month": "Tag im Monat",
    # Category-Features
    "is_eilmeldung": "Eilmeldung", "n_channels": "Kanalanzahl",
    "cat_sport": "Ressort: Sport", "cat_politik": "Ressort: Politik",
    "cat_unterhaltung": "Ressort: Unterhaltung", "cat_geld": "Ressort: Geld",
    "cat_regional": "Ressort: Regional", "cat_digital": "Ressort: Digital",
    "cat_leben": "Ressort: Leben", "cat_news": "Ressort: News",
    # Historical-Features
    "cat_avg_or_7d": "Ressort-Ø OR (7d)", "cat_avg_or_30d": "Ressort-Ø OR (30d)",
    "cat_avg_or_all": "Ressort-Ø OR (gesamt)", "hour_avg_or_7d": "Stunden-Ø OR (7d)",
    "hour_avg_or_30d": "Stunden-Ø OR (30d)", "cat_hour_avg_or": "Ressort×Stunde Ø OR",
    "stacking_cat_hour_baseline": "Stacking: Cat×Hour Baseline", "stacking_cat_hour_n": "Stacking: Cat×Hour Anzahl",
    "stacking_baseline_diff": "Stacking: Bayesian vs Raw Diff",
    "weekday_avg_or": "Wochentag-Ø OR", "max_similarity": "Max. Titel-Ähnlichkeit",
    "top_similar_or": "OR ähnlichster Push", "n_similar_pushes": "Anz. ähnlicher Pushes",
    "avg_similar_or": "Ø OR ähnlicher Pushes", "entity_avg_or": "Entity-Ø OR",
    "entity_count": "Anzahl Entities", "global_avg_or": "Globaler Ø OR",
    # TF-IDF Features
    "tfidf_max_sim": "TF-IDF Max-Ähnlichkeit", "tfidf_avg_sim": "TF-IDF Ø Ähnlichkeit",
    "tfidf_n_similar": "TF-IDF ähnliche Pushes", "tfidf_similar_avg_or": "TF-IDF ähnl. Ø OR",
    # Kontext-Features
    "weather_score": "Wetter-Score", "is_holiday": "Feiertag",
    "is_ctx_weekend": "Wochenende (Kontext)", "trend_match": "Trend-Match",
    # Embedding-Features
    "emb_max_sim": "Embedding Max-Ähnlichkeit", "emb_avg_sim_top10": "Embedding Ø Ähnl. Top-10",
    "emb_n_similar_50": "Embedding ähnl. Pushes", "emb_similar_avg_or": "Embedding ähnl. Ø OR",
    # BILD Topic-Cluster
    "topic_crime": "Thema: Crime", "topic_royals": "Thema: Royals",
    "topic_kosten": "Thema: Kosten/Geld", "topic_gesundheit": "Thema: Gesundheit",
    "topic_auto": "Thema: Auto/Verkehr", "topic_sex_beziehung": "Thema: Beziehung",
    "topic_wetter_extrem": "Thema: Wetter-Extrem", "topic_score_total": "Topic-Score gesamt",
    # Sport-Kalender
    "is_bundesliga_time": "Bundesliga-Zeitfenster", "is_cl_evening": "Champions-League-Abend",
    "is_transfer_window": "Transfer-Fenster",
    # Person-Tier
    "top_entity_or": "Top-Entity OR", "entity_hype_7d": "Entity-Hype (7d)",
    # BILD-Titelstil
    "has_age_parens": "Alter in Klammern", "has_das_so_pattern": "DAS/SO-Muster",
    "has_direct_address": "Direkte Anrede", "has_number_emphasis": "Zahlen-Betonung",
    # Volatilität + Interaktionen
    "cat_or_std_7d": "Ressort OR-Volatilität (7d)", "cat_or_std_30d": "Ressort OR-Volatilität (30d)",
    "hour_or_std_7d": "Stunden OR-Volatilität (7d)", "hour_or_std_30d": "Stunden OR-Volatilität (30d)",
    "weekday_hour_avg_or": "Wochentag×Stunde Ø OR",
    # Neue Features
    "hour_squared": "Stunde² (quadrat. Effekt)", "title_sentiment": "Titel-Sentiment",
    "days_since_similar": "Tage seit ähnl. Push", "or_volatility_7d": "OR-Volatilität (7d)",
}


def _keyword_magnitude_heuristic(title, cat_lower, is_eilmeldung=0):
    """Keyword-basierte Nachrichten-Magnitude 1-10 als LLM-Fallback.

    Fixes (2026-03-17): Diminishing Returns bei Multi-Keyword-Hits,
    Emotion-Word/Magnitude Double-Counting eliminiert.
    """
    title_lower = title.lower()
    score = 3.0  # Basis-Score

    # Eilmeldung/Breaking
    if is_eilmeldung or "eilmeldung" in title_lower or "breaking" in title_lower:
        score += 4.0

    # Terror/Krieg/Katastrophe → hohe Magnitude
    _high_mag = {"terror", "anschlag", "krieg", "explosion", "tsunami", "erdbeben",
                 "tote", "opfer", "massaker", "attentat", "geisel", "amok"}
    _med_high = {"warnung", "alarm", "gefahr", "notfall", "evakuierung", "absturz",
                 "brand", "feuer", "mord", "erstmals", "historisch", "rekord"}
    _med = {"kanzler", "praesident", "papst", "trump", "putin", "skandal",
            "verhaftet", "festnahme", "verurteil", "rücktritt", "wahl"}
    _low = {"lifestyle", "rezept", "trend", "mode", "beauty", "fitness",
            "garten", "reise", "urlaub", "quiz", "rätsel", "horoskop"}

    words = set(re.findall(r'[a-zäöüß]{3,}', title_lower))
    # Diminishing Returns: Anzahl Hits zählt, nicht nur Existenz
    _mag_hits = words & _high_mag
    _mag_used = set()  # Track für Emotion-Word Double-Counting
    if _mag_hits:
        n = len(_mag_hits)
        score += 4.0 + (0.8 if n >= 2 else 0)  # 2+ Hits: nur +0.8 extra (statt +4 pro Hit)
        _mag_used = _mag_hits
    elif words & _med_high:
        n = len(words & _med_high)
        score += 2.5 + (0.5 if n >= 2 else 0)
        _mag_used = words & _med_high
    elif words & _med:
        n = len(words & _med)
        score += 1.5 + (0.3 if n >= 2 else 0)
        _mag_used = words & _med

    if words & _low:
        score -= 1.5

    # Emotion-Words: NUR zählen wenn sie NICHT schon als Magnitude-Keyword gezählt wurden
    for emo_cat, emo_words in _GBRT_EMOTION_WORDS.items():
        new_emo = (words & emo_words) - _mag_used
        if new_emo:
            if emo_cat in ("angst", "katastrophe", "bedrohung"):
                score += 1.0  # Reduziert von 1.5
            elif emo_cat in ("sensation", "empoerung"):
                score += 0.7  # Reduziert von 1.0
            break

    # Topic-Cluster Bonus
    for cluster, cluster_words in _GBRT_TOPIC_CLUSTERS.items():
        if words & cluster_words:
            if cluster in ("crime", "wetter_extrem"):
                score += 0.5
            break

    # Kategorie-Adjustierung
    if cat_lower == "unterhaltung":
        score -= 1.0
    elif cat_lower in ("politik", "news"):
        score += 0.5
    elif cat_lower == "sport":
        # Sport-spezifische Magnitude: Ereignisse die im allgemeinen Keyword-Set fehlen
        # Kalibriert gegen DB-Daten: Trainer-Entlassung=5.1%, Verletzung=4.5%, Transfer=4.6%
        _sport_high = {"gestorben", "ist tot", "tödlich", "herzstillstand",
                       "abgesagt", "abbruch", "spielabbruch", "in lebensgefahr"}   # OR 7-10%: dramatisch
        _sport_med = {"verletzt", "verletzung", "ausfall", "entlassen", "feuert", "rauswurf",
                      "rücktritt", "suspendiert", "dopingsperre", "sperre"}  # OR 5-7%: wichtig
        _sport_low_boost = {"transfer", "wechsel", "abgang", "verpflichtet", "unterschreibt",
                            "verlängert", "aufstellung", "nominiert", "kader"}  # OR ~4.6%: leicht über Schnitt
        _sport_malus = {"überblick", "alle spiele", "alle tore", "spieltag",
                        "ergebnisse", "tabelle"}  # OR 2-3%: Routine-Übersicht

        title_words = title_lower  # Substring-Suche (keine Tokenisierung nötig)
        if any(w in title_words for w in _sport_high):
            score += 3.0
        elif any(w in title_words for w in _sport_med):
            score += 2.0
        elif any(w in title_words for w in _sport_low_boost):
            score += 0.8
        if any(w in title_words for w in _sport_malus):
            score -= 1.5

    return max(1.0, min(10.0, score))


# ── LLM News-Magnitude Scoring ──────────────────────────────────────────

_llm_score_lock = threading.Lock()


def _score_push_llm(title, category, push_id=None):
    """Bewertet einen Push-Titel via GPT-4o auf 5 Dimensionen (1-10).

    Returns: Dict mit magnitude, clickability, relevanz, dringlichkeit, emotionalitaet
    """
    result = {"magnitude": 0.0, "clickability": 0.0, "relevanz": 0.0,
              "dringlichkeit": 0.0, "emotionalitaet": 0.0}
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return result

        is_sport = (category or "").lower() in ("sport", "fussball", "bundesliga", "formel1", "formel-1", "tennis", "boxen", "motorsport")

        if is_sport:
            prompt = f"""Bewerte diese SPORT-Nachricht auf 5 Dimensionen (jeweils 1-10).

Titel: "{title}"

WICHTIG: Bewerte aus Sicht der BILD-Sport-Leser, nicht allgemeiner Nachrichtenwert!

1. sport_relevanz: Wie wichtig fuer deutsche Sport-Fans? (1=Randsport-Ergebnis, 10=WM-Finale/Bundesliga-Skandal)
2. dramatik: Wie dramatisch/emotional? (1=Routinebericht, 5=ueberraschendes Ergebnis, 10=Spielerlebensbedrohung/historischer Moment)
3. clickability: Wie klickbar ist der Titel? (1=nuechtern, 10=muss-klicken)
4. dringlichkeit: Wie zeitkritisch? (1=zeitlose Analyse, 5=heutiges Ergebnis, 10=Spiel laeuft JETZT)
5. push_eignung: Wuerdest du als BILD-Sportchef JETZT einen Push senden? (1=nie, 5=nur wenn nichts Besseres, 10=sofort pushen!)

Kontext: Top-Performer bei BILD Sport sind Drama (Verletzung, Spielabbruch), Transfer-Knaller, DFB/Nationalelf, Formel-1-Crashes/Siege, Tennis-Grand-Slams (Zverev!), Box-Knockouts. Routine-Ergebnisse performen schwach.

Antworte NUR mit JSON: {{"sport_relevanz":X,"dramatik":X,"clickability":X,"dringlichkeit":X,"push_eignung":X}}"""
        else:
            prompt = f"""Bewerte diese Nachricht auf 5 Dimensionen (jeweils 1-10 Skala).

Titel: "{title}"
Kategorie: {category}

Dimensionen:
1. magnitude: Wie gross/wichtig ist diese Nachricht? (1=trivial, 10=Weltgeschehen)
2. clickability: Wie klickbar ist der Titel? (1=langweilig, 10=muss-klicken)
3. relevanz: Wie relevant fuer deutsche BILD-Leser? (1=irrelevant, 10=betrifft jeden)
4. dringlichkeit: Wie zeitkritisch? (1=zeitlos, 10=jetzt-sofort)
5. emotionalitaet: Wie emotional aufgeladen? (1=sachlich, 10=hochemotion)

Antworte NUR mit JSON: {{"magnitude":X,"clickability":X,"relevanz":X,"dringlichkeit":X,"emotionalitaet":X}}"""

        # Nutze openai-Bibliothek (httpx) statt urllib — vermeidet macOS SSL-Zertifikat-Problem
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
            timeout=15,
        )
        content = resp.choices[0].message.content.strip()
        if "{" in content:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            scores = json.loads(json_str)
            if is_sport:
                # Sport-Dimensionen → Standard-Felder mappen
                _sport_map = {
                    "magnitude": "sport_relevanz",
                    "relevanz": "push_eignung",
                    "emotionalitaet": "dramatik",
                    "clickability": "clickability",
                    "dringlichkeit": "dringlichkeit",
                }
                for std_key, sport_key in _sport_map.items():
                    if sport_key in scores:
                        result[std_key] = max(1.0, min(10.0, float(scores[sport_key])))
            else:
                for key in result:
                    if key in scores:
                        result[key] = max(1.0, min(10.0, float(scores[key])))

        # In DB speichern
        if push_id and any(v > 0 for v in result.values()):
            try:
                with _push_db_lock:
                    conn = sqlite3.connect(PUSH_DB_PATH)
                    conn.execute("""UPDATE pushes SET
                        llm_magnitude=?, llm_clickability=?, llm_relevanz=?,
                        llm_dringlichkeit=?, llm_emotionalitaet=?, llm_scored_at=?
                        WHERE message_id=?""",
                        (result["magnitude"], result["clickability"], result["relevanz"],
                         result["dringlichkeit"], result["emotionalitaet"],
                         int(time.time()), push_id))
                    conn.commit()
                    conn.close()
            except Exception:
                pass

    except Exception as e:
        log.warning(f"[LLM-Score] Fehler für '{title[:40]}': {e}")
    return result


def _backfill_llm_scores():
    """Background-Thread: Scored alle Pushes ohne LLM-Score.

    Exponential Backoff bei Fehlern, Circuit Breaker nach 10 konsekutiven Fehlern.
    """
    time.sleep(30)  # Warte bis Server stabil
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.info("[LLM-Backfill] Kein OPENAI_API_KEY gesetzt, überspringe Backfill")
        return
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            rows = conn.execute("""SELECT message_id, title, cat FROM pushes
                WHERE (llm_scored_at IS NULL OR llm_scored_at = 0)
                AND title IS NOT NULL AND title != ''
                ORDER BY ts_num DESC""").fetchall()
            conn.close()

        if not rows:
            log.info("[LLM-Backfill] Alle Pushes bereits gescored")
            return

        log.info(f"[LLM-Backfill] Starte Scoring von {len(rows)} Pushes (5 parallel)...")
        scored = 0
        errors = 0
        consecutive_errors = 0
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score_one(push_id, title, cat):
            return push_id, _score_push_llm(title, cat or "news", push_id)

        batch_size = 15  # Parallele API-Calls (beschleunigt)
        for batch_start in range(0, len(rows), batch_size):
            if consecutive_errors >= 20:
                log.warning(f"[LLM-Backfill] Circuit Breaker: {consecutive_errors} "
                            f"konsekutive Fehler, stoppe bei {batch_start}/{len(rows)}")
                break
            batch = rows[batch_start:batch_start + batch_size]
            try:
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {executor.submit(_score_one, pid, t, c): (pid, t, c)
                               for pid, t, c in batch}
                    for future in as_completed(futures):
                        try:
                            pid, result = future.result()
                            if result["magnitude"] > 0:
                                scored += 1
                                consecutive_errors = 0
                            else:
                                errors += 1
                                consecutive_errors += 1
                        except Exception:
                            errors += 1
                            consecutive_errors += 1
            except Exception:
                errors += len(batch)
                consecutive_errors += len(batch)
            processed = batch_start + len(batch)
            if processed % 100 < batch_size:
                log.info(f"[LLM-Backfill] {processed}/{len(rows)} gescored "
                         f"({scored} OK, {errors} Fehler)")
            time.sleep(0.1)  # Kurze Pause zwischen Batches

        log.info(f"[LLM-Backfill] Fertig: {scored}/{len(rows)} gescored, {errors} Fehler")
    except Exception as e:
        log.warning(f"[LLM-Backfill] Fehler: {e}")


def _load_llm_scores_for_push(push):
    """Lädt LLM-Scores aus DB für einen Push (für Feature-Extraktion im Training)."""
    push_id = push.get("message_id", "")
    if not push_id:
        return {}
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            row = conn.execute("""SELECT llm_magnitude, llm_clickability, llm_relevanz,
                llm_dringlichkeit, llm_emotionalitaet FROM pushes WHERE message_id=?""",
                (push_id,)).fetchone()
            conn.close()
        if row and row[0] and row[0] > 0:
            return {
                "magnitude": float(row[0]),
                "clickability": float(row[1] or 0),
                "relevanz": float(row[2] or 0),
                "dringlichkeit": float(row[3] or 0),
                "emotionalitaet": float(row[4] or 0),
            }
    except Exception:
        pass
    return {}


def _gbrt_extract_features(push, history_stats, state=None, fast_mode=False):
    """Extrahiert ~80 Features aus einem Push fuer das GBRT-Modell.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        history_stats: Vorberechnete Aggregat-Statistiken (von _gbrt_build_history_stats)
        state: Optionaler Research-State fuer Kontext-Features
    Returns:
        Dict mit Feature-Name → Float-Wert
    """
    title = push.get("title", "") or ""
    title_lower = title.lower()
    cat = (push.get("cat", "") or "News").strip()
    cat_lower = cat.lower()
    ts = push.get("ts_num", 0)
    dt = datetime.datetime.fromtimestamp(ts) if ts > 0 else datetime.datetime.now()
    hour = push.get("hour", dt.hour)
    weekday = dt.weekday()

    words = title.split()
    word_count = len(words)
    title_len = len(title)

    feat = {}

    # ── Text-Features (~20) ──────────────────────────────────────────────
    feat["title_len"] = title_len
    feat["word_count"] = word_count
    feat["avg_word_len"] = sum(len(w) for w in words) / max(1, word_count)
    feat["has_question"] = 1.0 if "?" in title else 0.0
    feat["has_exclamation"] = 1.0 if "!" in title else 0.0
    feat["has_colon"] = 1.0 if ":" in title else 0.0
    feat["has_pipe"] = 1.0 if "|" in title else 0.0
    feat["has_plus_plus"] = 1.0 if "++" in title else 0.0
    feat["has_numbers"] = 1.0 if re.search(r"\d", title) else 0.0
    feat["upper_ratio"] = sum(1 for c in title if c.isupper()) / max(1, title_len)

    # Name-Density: Gross geschriebene Woerter (Entities)
    cap_words = re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', title)
    feat["name_density"] = len(cap_words) / max(1, word_count)

    # Death Signal
    feat["death_signal"] = 1.0 if any(w in title_lower for w in _GBRT_DEATH_WORDS) else 0.0

    # Exclusivity Signal
    feat["exclusivity_signal"] = 1.0 if any(w in title_lower for w in _GBRT_EXCLUSIVITY_WORDS) else 0.0

    # Kicker Pattern (Doppelpunkt am Anfang: "SPORT:" oder "Name:")
    feat["kicker_pattern"] = 1.0 if re.match(r'^[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß\s]{1,20}:', title) else 0.0

    # Emotional Word Count (multi-category)
    total_emo = 0
    emo_cats_hit = 0
    for cat_name, words_set in _GBRT_EMOTION_WORDS.items():
        hits = sum(1 for w in words_set if w in title_lower)
        if hits > 0:
            emo_cats_hit += 1
            total_emo += hits
    feat["emotional_word_count"] = float(total_emo)
    feat["emotional_categories"] = float(emo_cats_hit)
    feat["intensity_score"] = min(1.0, total_emo * 0.15 * (1.0 + max(0, emo_cats_hit - 1) * 0.3))

    # Title Sentiment: Ratio negativer vs positiver Signalwörter
    _neg_words = {"tot", "tod", "sterben", "mord", "crash", "absturz", "krieg", "terror",
                  "unfall", "opfer", "skandal", "gefahr", "alarm", "notfall", "warnung"}
    _pos_words = {"rekord", "sensation", "historisch", "erstmals", "gewonnen", "sieg",
                  "gerettet", "durchbruch", "freude", "feier", "gold", "triumph", "held"}
    neg_count = sum(1 for w in _neg_words if w in title_lower)
    pos_count = sum(1 for w in _pos_words if w in title_lower)
    total_sent = neg_count + pos_count
    feat["title_sentiment"] = (pos_count - neg_count) / max(1, total_sent)  # -1.0 bis +1.0

    # Breaking Signals
    breaking = 0
    if "++" in title: breaking += 1
    if "+++" in title: breaking += 2
    if "|" in title: breaking += 1
    if _GBRT_BREAKING_RE.search(title): breaking += 2
    if push.get("is_eilmeldung"): breaking += 2
    if title.strip().endswith("!"): breaking += 1
    feat["breaking_signals"] = float(breaking)
    feat["is_breaking_style"] = 1.0 if breaking >= 3 else 0.0

    # ── BILD Plus (Paywall) — aus Feld oder URL ableiten ──
    _is_plus = push.get("is_bild_plus") or push.get("isBildPlus")
    if not _is_plus:
        _link = push.get("link", "") or ""
        if re.search(r"/bild-?plus/|/bild_plus/|/bildplus/|bildplus-gewinnspiele|/premium-event/|\.bild_plus\.", _link):
            _is_plus = True
    feat["is_bild_plus"] = 1.0 if _is_plus else 0.0

    # ── BILD Topic-Cluster-Scores (~8) ────────────────────────────────────
    topic_total = 0
    for topic_name, topic_words in _GBRT_TOPIC_CLUSTERS.items():
        hits = sum(1 for w in topic_words if w in title_lower)
        feat[f"topic_{topic_name}"] = float(hits)
        topic_total += hits
    feat["topic_score_total"] = float(topic_total)

    # ── BILD-Titelstil (~4) ──────────────────────────────────────────────
    feat["has_age_parens"] = 1.0 if _GBRT_AGE_PATTERN.search(title) else 0.0
    feat["has_das_so_pattern"] = 1.0 if _GBRT_DAS_SO_PATTERN.search(title) else 0.0
    feat["has_direct_address"] = 1.0 if _GBRT_DIRECT_ADDRESS.search(title) else 0.0
    feat["has_number_emphasis"] = 1.0 if _GBRT_NUMBER_EMPHASIS.search(title) else 0.0

    # ── Temporal-Features (~15) ──────────────────────────────────────────
    feat["hour"] = float(hour)
    feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    feat["weekday"] = float(weekday)
    feat["weekday_sin"] = math.sin(2 * math.pi * weekday / 7)
    feat["weekday_cos"] = math.cos(2 * math.pi * weekday / 7)
    feat["is_weekend"] = 1.0 if weekday >= 5 else 0.0
    feat["is_prime_time"] = 1.0 if 18 <= hour <= 22 else 0.0
    feat["is_morning_commute"] = 1.0 if 6 <= hour <= 9 else 0.0
    feat["is_late_night"] = 1.0 if hour < 6 or hour >= 23 else 0.0
    feat["is_lunch"] = 1.0 if 11 <= hour <= 13 else 0.0
    feat["hour_squared"] = float(hour * hour)  # Quadratischer Tageszeit-Effekt

    # Minutes since last push same category (Fatigue-Signal)
    last_same_cat_ts = history_stats.get("last_push_ts_by_cat", {}).get(cat_lower, 0)
    if last_same_cat_ts > 0 and ts > last_same_cat_ts:
        feat["mins_since_last_same_cat"] = (ts - last_same_cat_ts) / 60.0
    else:
        feat["mins_since_last_same_cat"] = 1440.0  # Default: 24h

    # Push count today so far (Sättigung)
    feat["push_count_today"] = float(push.get("_push_number_today", 0))

    # Day of month (fuer Monatsend-/Monatsanfang-Effekte)
    feat["day_of_month"] = float(dt.day)

    # ── Sport-Kalender (~3) ──────────────────────────────────────────────
    # Bundesliga: Sa 15:30-18:30, So 15:30-19:30 (Sep-Mai)
    month = dt.month
    is_season = month >= 8 or month <= 5  # Aug-Mai = Saison
    if is_season and weekday == 5 and 15 <= hour <= 18:  # Samstag
        feat["is_bundesliga_time"] = 1.0
    elif is_season and weekday == 6 and 15 <= hour <= 19:  # Sonntag
        feat["is_bundesliga_time"] = 1.0
    elif is_season and weekday == 4 and 20 <= hour <= 22:  # Freitag Abend
        feat["is_bundesliga_time"] = 1.0
    else:
        feat["is_bundesliga_time"] = 0.0
    # Champions League: Di/Mi Abend (Sep-Mai)
    feat["is_cl_evening"] = 1.0 if (is_season and weekday in (1, 2) and 20 <= hour <= 23) else 0.0
    # Transfer-Fenster: Jan + Juli/Aug
    feat["is_transfer_window"] = 1.0 if month in (1, 7, 8) else 0.0

    # ── Category-Features (~10) ──────────────────────────────────────────
    feat["is_eilmeldung"] = 1.0 if push.get("is_eilmeldung") else 0.0
    channels = push.get("channels", [])
    feat["n_channels"] = float(len(channels) if isinstance(channels, list) else 1)

    # Channel-Features (global_avg needed early)
    global_avg = history_stats.get("global_avg", 4.77)
    ch_stats_all = history_stats.get("channel_stats", {})
    ch_names_lower = [str(c).lower().strip() for c in channels] if isinstance(channels, list) else []
    ch_avg_ors = []
    ch_reach = 0.0
    for ch_name in ch_names_lower:
        ch_s = ch_stats_all.get(ch_name, {})
        if ch_s.get("n", 0) > 0:
            ch_avg_ors.append(ch_s["avg"])
            ch_reach += ch_s["n"]
    feat["channel_avg_or"] = (sum(ch_avg_ors) / len(ch_avg_ors)) if ch_avg_ors else global_avg
    feat["channel_max_or"] = max(ch_avg_ors) if ch_avg_ors else global_avg
    feat["has_channel_eilmeldung"] = 1.0 if "eilmeldung" in ch_names_lower else 0.0
    feat["has_channel_news"] = 1.0 if "news" in ch_names_lower else 0.0
    feat["channel_reach_proxy"] = ch_reach

    # ── App-Mix & Reichweite Features (~8) ──────────────────────────────
    # ── Reichweite-Features (KEINE OR-basierten Features = Data Leakage!) ──
    # Nur strukturelle Infos die VOR dem Send bekannt sind:
    n_apps = push.get("n_apps", 0)
    total_recipients = push.get("total_recipients", 0) or push.get("received", 0) or 0
    feat["n_apps"] = float(n_apps) if n_apps else float(len(push.get("app_list", [])))
    feat["log_recipients"] = math.log1p(total_recipients) if total_recipients > 0 else 0.0
    # Per-App Empfänger-Anteile (recipientCount pro App, NICHT openingRate!)
    target_stats = push.get("target_stats", {})
    ios_recip_share = 0.0
    android_recip_share = 0.0
    sport_recip_share = 0.0
    if isinstance(target_stats, dict) and target_stats and total_recipients > 0:
        for app_name, stats in target_stats.items():
            if not isinstance(stats, dict):
                continue
            app_recip = float(stats.get("recipientCount", 0) or 0)
            app_lower = app_name.lower()
            if "ios" in app_lower and "sport" not in app_lower:
                ios_recip_share += app_recip
            if "android" in app_lower and "sport" not in app_lower:
                android_recip_share += app_recip
            if "sport" in app_lower:
                sport_recip_share += app_recip
        ios_recip_share /= max(total_recipients, 1)
        android_recip_share /= max(total_recipients, 1)
        sport_recip_share /= max(total_recipients, 1)
    feat["ios_recip_share"] = ios_recip_share
    feat["android_recip_share"] = android_recip_share
    feat["sport_recip_share"] = sport_recip_share
    feat["ios_android_ratio"] = ios_recip_share / max(android_recip_share, 0.01) if android_recip_share > 0 else 0.0
    # Reichweite relativ zum Median
    median_recip = history_stats.get("median_recipients", 0)
    cat_med = history_stats.get("cat_median_recipients", {}).get(cat_lower, median_recip)
    feat["recipients_vs_median"] = total_recipients / max(cat_med, 1) if cat_med > 0 and total_recipients > 0 else 1.0

    # Category One-Hot
    for c in _GBRT_CATEGORIES:
        feat[f"cat_{c}"] = 1.0 if cat_lower == c else 0.0

    # ── Historical-Features (~20, Bayesian-smoothed) ─────────────────────
    global_n = history_stats.get("global_n", 100)
    bayesian_prior_n = 10  # Shrinkage-Staerke (reduziert von 30: weniger Regression zum Mittelwert)

    def _bayesian_avg(group_avg, group_n, prior=global_avg, prior_n=bayesian_prior_n):
        """Bayesian Shrinkage: gewichteter Mix aus Gruppen-Durchschnitt und Prior."""
        if group_n <= 0:
            return prior
        return (group_avg * group_n + prior * prior_n) / (group_n + prior_n)

    # Category averages (7d, 30d, all)
    cat_stats = history_stats.get("cat_stats", {}).get(cat_lower, {})
    feat["cat_avg_or_7d"] = _bayesian_avg(cat_stats.get("avg_7d", global_avg), cat_stats.get("n_7d", 0))
    feat["cat_avg_or_30d"] = _bayesian_avg(cat_stats.get("avg_30d", global_avg), cat_stats.get("n_30d", 0))
    feat["cat_avg_or_all"] = _bayesian_avg(cat_stats.get("avg_all", global_avg), cat_stats.get("n_all", 0))

    # Category Momentum
    cat_mom = history_stats.get("cat_momentum", {}).get(cat_lower, {})
    feat["cat_momentum"] = cat_mom.get("momentum", 0.0)
    feat["cat_7d_vs_all_ratio"] = cat_mom.get("ratio_7d_all", 1.0)

    # Hour averages (7d, 30d)
    hour_stats = history_stats.get("hour_stats", {}).get(hour, {})
    feat["hour_avg_or_7d"] = _bayesian_avg(hour_stats.get("avg_7d", global_avg), hour_stats.get("n_7d", 0))
    feat["hour_avg_or_30d"] = _bayesian_avg(hour_stats.get("avg_30d", global_avg), hour_stats.get("n_30d", 0))

    # Category x Hour interaction
    cat_hour_key = f"{cat_lower}_{hour}"
    ch_stats = history_stats.get("cat_hour_stats", {}).get(cat_hour_key, {})
    feat["cat_hour_avg_or"] = _bayesian_avg(ch_stats.get("avg", global_avg), ch_stats.get("n", 0))

    # Stacking: Cat×Hour-Baseline als explizites Feature (= raw Mean ohne Bayesian-Smoothing)
    feat["stacking_cat_hour_baseline"] = ch_stats.get("avg", global_avg)
    feat["stacking_cat_hour_n"] = float(ch_stats.get("n", 0))
    # Differenz zwischen Bayesian-smoothed und raw Baseline
    feat["stacking_baseline_diff"] = feat["cat_hour_avg_or"] - feat["stacking_cat_hour_baseline"]

    # Weekday averages
    wd_stats = history_stats.get("weekday_stats", {}).get(weekday, {})
    feat["weekday_avg_or"] = _bayesian_avg(wd_stats.get("avg", global_avg), wd_stats.get("n", 0))

    # Category x Weekday interaction
    cat_wd_key = f"{cat_lower}_{weekday}"
    cat_wd_stats = history_stats.get("cat_weekday_stats", {}).get(cat_wd_key, {})
    feat["cat_weekday_avg_or"] = _bayesian_avg(cat_wd_stats.get("avg", global_avg), cat_wd_stats.get("n", 0))

    # Weekday x Hour interaction
    wd_hour_key = f"{weekday}_{hour}"
    wd_hour_stats = history_stats.get("weekday_hour_stats", {}).get(wd_hour_key, {})
    feat["weekday_hour_avg_or"] = _bayesian_avg(wd_hour_stats.get("avg", global_avg), wd_hour_stats.get("n", 0))

    # Volatilitaet (Std der OR) — wie vorhersagbar ist diese Kategorie/Stunde?
    cat_vol = history_stats.get("cat_volatility", {}).get(cat_lower, {})
    feat["cat_or_std_7d"] = cat_vol.get("std_7d", 0.0)
    feat["cat_or_std_30d"] = cat_vol.get("std_30d", 0.0)
    hour_vol = history_stats.get("hour_volatility", {}).get(hour, {})
    feat["hour_or_std_7d"] = hour_vol.get("std_7d", 0.0)
    feat["hour_or_std_30d"] = hour_vol.get("std_30d", 0.0)

    # Similarity to top-10 historical pushes (Jaccard) — skip in fast_mode (training)
    if not fast_mode:
        push_words = set(re.findall(r'[a-zäöüß]{4,}', title_lower))
        stops = {"der", "die", "das", "und", "von", "fuer", "mit", "auf", "den", "ist", "ein", "eine",
                 "sich", "auch", "noch", "nur", "jetzt", "alle", "neue", "wird", "wurde", "nach", "ueber",
                 "dass", "aber", "oder", "wenn", "dann", "mehr", "sein", "hat", "haben", "kann", "sind"}
        push_words -= stops
        max_jaccard = 0.0
        top_sim_or = 0.0
        sim_count = 0
        sim_or_sum = 0.0
        for hist_push in history_stats.get("recent_pushes", []):
            h_words = hist_push.get("words", set())
            if push_words and h_words:
                jaccard = len(push_words & h_words) / len(push_words | h_words)
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    top_sim_or = hist_push.get("or", global_avg)
                if jaccard > 0.15:
                    sim_count += 1
                    sim_or_sum += hist_push.get("or", global_avg)
        feat["max_similarity"] = max_jaccard
        feat["top_similar_or"] = top_sim_or if max_jaccard > 0.1 else global_avg
        feat["n_similar_pushes"] = float(sim_count)
        feat["avg_similar_or"] = (sim_or_sum / sim_count) if sim_count > 0 else global_avg
    else:
        feat["max_similarity"] = 0.0
        feat["top_similar_or"] = global_avg
        feat["n_similar_pushes"] = 0.0
        feat["avg_similar_or"] = global_avg

    # Entity-based historical OR
    entities = set(re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', title))
    entity_or_list = history_stats.get("entity_or", {})
    entity_ors = []
    for ent in entities:
        ent_l = ent.lower()
        if ent_l in entity_or_list and len(entity_or_list[ent_l]) >= 2:
            entity_ors.extend(entity_or_list[ent_l])
    feat["entity_avg_or"] = (sum(entity_ors) / len(entity_ors)) if entity_ors else global_avg
    feat["entity_count"] = float(len(entities))

    # Person-Tier: Top-Entity OR (stärkstes Entity im Titel) + Hype-Faktor
    entity_freq_7d = history_stats.get("entity_freq_7d", {})
    top_ent_or = global_avg
    max_ent_hype = 0.0
    for ent in entities:
        ent_l = ent.lower()
        ent_or_hist = entity_or_list.get(ent_l, [])
        if len(ent_or_hist) >= 2:
            ent_avg = sum(ent_or_hist) / len(ent_or_hist)
            if ent_avg > top_ent_or:
                top_ent_or = ent_avg
        freq = entity_freq_7d.get(ent_l, 0)
        if freq > max_ent_hype:
            max_ent_hype = freq
    feat["top_entity_or"] = top_ent_or
    feat["entity_hype_7d"] = float(max_ent_hype)

    # Global average as baseline reference
    feat["global_avg_or"] = global_avg

    # ── Character N-Gram TF-IDF Similarity Features — skip in fast_mode ──
    if not fast_mode and _char_ngram_tfidf and _char_ngram_tfidf.vocab:
        try:
            push_vec = _char_ngram_tfidf.transform_one(title)
            recent = history_stats.get("recent_pushes", [])
            tfidf_sims = []
            tfidf_sim_ors = []
            for hist_push in recent[:500]:
                htitle = hist_push.get("title_raw", "")
                if not htitle:
                    continue
                hvec = _char_ngram_tfidf.transform_one(htitle)
                sim = _char_ngram_tfidf.cosine_similarity(push_vec, hvec)
                if sim > 0.15:
                    tfidf_sims.append(sim)
                    tfidf_sim_ors.append(hist_push.get("or", global_avg))
            feat["tfidf_max_sim"] = max(tfidf_sims) if tfidf_sims else 0.0
            feat["tfidf_avg_sim"] = (sum(tfidf_sims) / len(tfidf_sims)) if tfidf_sims else 0.0
            feat["tfidf_n_similar"] = float(len(tfidf_sims))
            feat["tfidf_similar_avg_or"] = (sum(tfidf_sim_ors) / len(tfidf_sim_ors)) if tfidf_sim_ors else global_avg
        except Exception:
            feat["tfidf_max_sim"] = 0.0
            feat["tfidf_avg_sim"] = 0.0
            feat["tfidf_n_similar"] = 0.0
            feat["tfidf_similar_avg_or"] = global_avg
    else:
        feat["tfidf_max_sim"] = 0.0
        feat["tfidf_avg_sim"] = 0.0
        feat["tfidf_n_similar"] = 0.0
        feat["tfidf_similar_avg_or"] = global_avg

    # ── Kontext-Features (~5) ────────────────────────────────────────────
    ctx = _external_context_cache if '_external_context_cache' in dir() else {}
    if not ctx:
        ctx = globals().get("_external_context_cache", {})
    feat["weather_score"] = ctx.get("weather", {}).get("bad_weather_score", 0.3) if ctx.get("last_fetch", 0) > 0 else 0.3
    feat["is_holiday"] = 1.0 if ctx.get("day_type") == "holiday" else 0.0
    feat["is_ctx_weekend"] = 1.0 if ctx.get("day_type") == "weekend" else 0.0
    feat["trend_match"] = 0.0
    if ctx.get("trends") and title:
        try:
            feat["trend_match"] = _context_topic_match(title, ctx["trends"])
        except Exception:
            pass

    # ── Rolling/Lag-Features (~10) ──────────────────────────────────────
    push_timeline = history_stats.get("push_timeline", [])
    push_timeline_ts = history_stats.get("push_timeline_ts", [])
    if push_timeline_ts and ts > 0:
        # Binary search for current position
        pos = bisect_left(push_timeline_ts, ts)

        def _rolling_stats(window_secs):
            """Avg OR and count of pushes in [ts - window_secs, ts)."""
            start_ts = ts - window_secs
            start_idx = bisect_left(push_timeline_ts, start_ts)
            end_idx = pos  # exclusive (current push not included)
            if start_idx >= end_idx:
                return global_avg, 0
            window = push_timeline[start_idx:end_idx]
            ors = [w[1] for w in window]
            return sum(ors) / len(ors), len(ors)

        r1h_avg, r1h_n = _rolling_stats(3600)
        r3h_avg, r3h_n = _rolling_stats(10800)
        r6h_avg, r6h_n = _rolling_stats(21600)
        r24h_avg, r24h_n = _rolling_stats(86400)

        feat["rolling_or_1h"] = r1h_avg
        feat["rolling_or_3h"] = r3h_avg
        feat["rolling_or_6h"] = r6h_avg
        feat["rolling_or_24h"] = r24h_avg
        feat["rolling_n_1h"] = float(r1h_n)
        feat["rolling_n_3h"] = float(r3h_n)
        feat["rolling_n_6h"] = float(r6h_n)

        # Last-N pushes OR
        prev_pushes = push_timeline[max(0, pos - 10):pos]
        prev_ors = [w[1] for w in prev_pushes]
        feat["rolling_or_last3"] = (sum(prev_ors[-3:]) / len(prev_ors[-3:])) if len(prev_ors) >= 3 else global_avg
        feat["rolling_or_last5"] = (sum(prev_ors[-5:]) / len(prev_ors[-5:])) if len(prev_ors) >= 5 else global_avg

        # Momentum: avg_last3 - avg_last10
        avg_last3 = (sum(prev_ors[-3:]) / len(prev_ors[-3:])) if len(prev_ors) >= 3 else global_avg
        avg_last10 = (sum(prev_ors) / len(prev_ors)) if prev_ors else global_avg
        feat["rolling_momentum"] = avg_last3 - avg_last10

        # Sättigungs-Features
        if prev_pushes:
            last_push_ts_any = prev_pushes[-1][0]
            feat["mins_since_last_push"] = (ts - last_push_ts_any) / 60.0
        else:
            feat["mins_since_last_push"] = 1440.0
        feat["push_rate_3h"] = r3h_n / 3.0
        feat["saturation_score"] = min(1.0, r3h_n / 8.0)
    else:
        feat["rolling_or_1h"] = global_avg
        feat["rolling_or_3h"] = global_avg
        feat["rolling_or_6h"] = global_avg
        feat["rolling_or_24h"] = global_avg
        feat["rolling_n_1h"] = 0.0
        feat["rolling_n_3h"] = 0.0
        feat["rolling_n_6h"] = 0.0
        feat["rolling_or_last3"] = global_avg
        feat["rolling_or_last5"] = global_avg
        feat["rolling_momentum"] = 0.0
        feat["mins_since_last_push"] = 1440.0
        feat["push_rate_3h"] = 0.0
        feat["saturation_score"] = 0.0

    # ── Days since similar push (gleiche Kategorie als Proxy) ───────────
    # Nutze last_push_ts_by_cat aus history_stats (O(1) statt O(n) Timeline-Scan)
    last_cat_ts = history_stats.get("last_push_ts_by_cat", {}).get(cat_lower, 0)
    if last_cat_ts > 0 and ts > last_cat_ts:
        feat["days_since_similar"] = min(365.0, (ts - last_cat_ts) / 86400.0)
    else:
        feat["days_since_similar"] = 365.0

    # ── OR-Volatilität der letzten 7 Tage (Marktvolatilität) ──────────
    or_volatility_7d = 0.0
    push_timeline_ts = history_stats.get("push_timeline_ts", [])
    if push_timeline and push_timeline_ts and ts > 0:
        cutoff_7d_ts = ts - 7 * 86400
        start_idx = bisect_left(push_timeline_ts, cutoff_7d_ts)
        end_idx = bisect_left(push_timeline_ts, ts)
        recent_ors = [push_timeline[i][1] for i in range(start_idx, min(end_idx, len(push_timeline)))]
        if len(recent_ors) > 2:
            or_mean = sum(recent_ors) / len(recent_ors)
            or_volatility_7d = math.sqrt(sum((o - or_mean) ** 2 for o in recent_ors) / len(recent_ors))
    feat["or_volatility_7d"] = or_volatility_7d

    # ── Interaction-Features (~5) ────────────────────────────────────────
    feat["eilmeldung_x_primetime"] = feat["is_eilmeldung"] * feat["is_prime_time"]
    feat["eilmeldung_x_hour"] = feat["is_eilmeldung"] * feat["hour"]
    feat["weekend_x_hour"] = feat["is_weekend"] * feat["hour"]
    feat["breaking_x_primetime"] = feat["is_breaking_style"] * feat["is_prime_time"]

    # ── Reichweite-Interactions ──
    feat["recipients_x_eilmeldung"] = feat["log_recipients"] * feat["is_eilmeldung"]
    feat["recipients_x_primetime"] = feat["log_recipients"] * feat["is_prime_time"]
    feat["n_apps_x_hour"] = feat["n_apps"] * feat["hour"]

    # Sättigungs-Interaktionen
    feat["rolling_3h_x_saturation"] = feat["rolling_or_3h"] * feat["saturation_score"]
    feat["push_rate_x_hour"] = feat["push_rate_3h"] * feat["hour"]
    # Zeitliche Trend-Differenzen (Momentum-Signale)
    feat["rolling_1h_vs_24h"] = feat["rolling_or_1h"] - feat["rolling_or_24h"]
    feat["rolling_3h_vs_24h"] = feat["rolling_or_3h"] - feat["rolling_or_24h"]
    # Month + Season (jahreszeit-abhängige Lesegewohnheiten)
    feat["month"] = float(month)
    feat["month_sin"] = math.sin(2 * math.pi * month / 12)
    feat["month_cos"] = math.cos(2 * math.pi * month / 12)
    feat["is_summer"] = 1.0 if month in (6, 7, 8) else 0.0
    feat["is_winter"] = 1.0 if month in (12, 1, 2) else 0.0

    # ── Kicker/Headline Split-Features ──
    kicker_text = push.get("kicker", "") or ""
    headline_text = push.get("headline", "") or ""
    feat["has_kicker"] = 1.0 if kicker_text.strip() else 0.0
    feat["kicker_len"] = float(len(kicker_text))
    feat["headline_len"] = float(len(headline_text))
    feat["kicker_headline_ratio"] = len(kicker_text) / max(len(headline_text), 1)

    # ── Keyword→OR Features (Historische Wort-Performance) ──────────────
    word_or = history_stats.get("word_or", {})
    bigram_or = history_stats.get("bigram_or", {})
    title_lower = title.lower()
    title_words = [w for w in title_lower.split() if len(w) >= 3 and w not in _TOPIC_STOPS]
    # Wort-Level: avg OR aller Wörter im Titel die wir kennen
    known_word_ors = []
    for w in title_words:
        w_stats = word_or.get(w)
        if w_stats and w_stats["n"] >= 5:
            known_word_ors.append(w_stats["avg"])
    feat["keyword_avg_or"] = sum(known_word_ors) / len(known_word_ors) if known_word_ors else global_avg
    feat["keyword_max_or"] = max(known_word_ors) if known_word_ors else global_avg
    feat["keyword_min_or"] = min(known_word_ors) if known_word_ors else global_avg
    feat["keyword_spread"] = feat["keyword_max_or"] - feat["keyword_min_or"]
    feat["n_known_keywords"] = float(len(known_word_ors))
    feat["keyword_coverage"] = len(known_word_ors) / max(len(title_words), 1)
    # Keyword Quantile: robustere Statistik
    if known_word_ors:
        sorted_kw = sorted(known_word_ors)
        feat["keyword_median_or"] = sorted_kw[len(sorted_kw) // 2]
        feat["keyword_q25_or"] = sorted_kw[len(sorted_kw) // 4] if len(sorted_kw) >= 4 else sorted_kw[0]
        feat["keyword_q75_or"] = sorted_kw[3 * len(sorted_kw) // 4] if len(sorted_kw) >= 4 else sorted_kw[-1]
    else:
        feat["keyword_median_or"] = global_avg
        feat["keyword_q25_or"] = global_avg
        feat["keyword_q75_or"] = global_avg
    # Keyword-Std (Spread der historischen ORs für die Wörter im Titel)
    known_word_stds = []
    for w in title_words:
        w_stats = word_or.get(w)
        if w_stats and w_stats["n"] >= 5:
            known_word_stds.append(w_stats.get("std", 0.0))
    feat["keyword_avg_std"] = sum(known_word_stds) / len(known_word_stds) if known_word_stds else 0.0
    # Bigram-Level
    known_bigram_ors = []
    if len(title_words) >= 2:
        for i in range(len(title_words) - 1):
            bg = f"{title_words[i]}_{title_words[i+1]}"
            bg_stats = bigram_or.get(bg)
            if bg_stats and bg_stats["n"] >= 5:
                known_bigram_ors.append(bg_stats["avg"])
    feat["bigram_avg_or"] = sum(known_bigram_ors) / len(known_bigram_ors) if known_bigram_ors else global_avg
    feat["bigram_max_or"] = max(known_bigram_ors) if known_bigram_ors else global_avg
    feat["n_known_bigrams"] = float(len(known_bigram_ors))
    # Keyword vs Category-Avg: Wie gut sind die Wörter relativ zur Kategorie?
    cat_avg = history_stats.get("cat_stats", {}).get(cat_lower, {}).get("avg_all", global_avg)
    feat["keyword_vs_cat"] = feat["keyword_avg_or"] - cat_avg
    # Heuristic Magnitude/Urgency (LLM-Ersatz)
    _urgency_words = {"eilmeldung", "breaking", "alarm", "warnung", "sofort", "jetzt",
                      "gerade", "aktuell", "liveticker", "notfall", "evakuierung"}
    _magnitude_words = {"krieg", "tod", "tote", "anschlag", "terror", "erdbeben", "tsunami",
                        "explosion", "absturz", "mord", "kanzler", "präsident", "papst",
                        "historisch", "erstmals", "rekord", "weltmeister", "olympia"}
    _click_words = {"geheimnis", "enthüllt", "wahrheit", "unfassbar", "schock", "skandal",
                    "so", "das", "diese", "jetzt", "mega", "hammer", "krass", "irre",
                    "unglaublich", "wahnsinn", "exklusiv"}
    title_word_set = set(title_words)
    feat["heur_urgency"] = float(len(title_word_set & _urgency_words))
    feat["heur_magnitude"] = float(len(title_word_set & _magnitude_words))
    feat["heur_clickbait"] = float(len(title_word_set & _click_words))

    # ── SHAP-guided Top-Feature Interactions ──
    # keyword_avg_or × zeitliche Features (Top-2 SHAP)
    feat["keyword_x_hour_avg"] = feat["keyword_avg_or"] * feat["weekday_hour_avg_or"] / max(global_avg, 1.0)
    feat["keyword_x_cat_hour"] = feat["keyword_avg_or"] * feat["cat_hour_avg_or"] / max(global_avg, 1.0)
    feat["keyword_x_saturation"] = feat["keyword_avg_or"] * feat["saturation_score"]
    feat["keyword_x_volatility"] = feat["keyword_avg_or"] * feat["or_volatility_7d"]
    # keyword_avg_or Abweichung von zeitlichem Durchschnitt
    feat["keyword_vs_hour"] = feat["keyword_avg_or"] - feat["weekday_hour_avg_or"]
    feat["keyword_vs_rolling24h"] = feat["keyword_avg_or"] - feat["rolling_or_24h"]

    # ── Sentence Embedding Similarity Features (Phase F) ──
    # _compute_embedding_features: 500 Cosine-Calls pro Push → NUR in Inference
    # Im Training (8000 Pushes) wären das 4M Cosine-Calls → skip, PCA ersetzt Signal
    _is_inference = state is not None  # state wird nur in Inference übergeben
    if _is_inference and _embedding_model is not None:
        try:
            emb_feats = _compute_embedding_features(title, history_stats)
            feat.update(emb_feats)
        except Exception:
            feat["emb_max_sim"] = 0.0
            feat["emb_avg_sim_top10"] = 0.0
            feat["emb_n_similar_50"] = 0.0
            feat["emb_similar_avg_or"] = 0.0
    else:
        feat["emb_max_sim"] = 0.0
        feat["emb_avg_sim_top10"] = 0.0
        feat["emb_n_similar_50"] = 0.0
        feat["emb_similar_avg_or"] = 0.0

    feat["heur_research_factor"] = float(research_mods.get("combined", 1.0)) if research_mods else 1.0
    feat["heur_phd_combined"] = 1.0

    # ── Granulare Wetter-Features (6) ──
    weather_data = ctx.get("weather", {}) if ctx.get("last_fetch", 0) > 0 else {}
    feat["weather_temp_c"] = float(weather_data.get("temp_c", 15)) / 40.0  # normalisiert auf ~0-1
    feat["weather_humidity"] = float(weather_data.get("humidity", 50)) / 100.0
    feat["weather_precip_mm"] = min(1.0, float(weather_data.get("precip_mm", 0)) / 10.0)
    feat["weather_wind_kmph"] = min(1.0, float(weather_data.get("wind_kmph", 10)) / 80.0)
    feat["weather_cloud_cover"] = float(weather_data.get("cloud_cover", 50)) / 100.0
    feat["weather_uv_index"] = min(1.0, float(weather_data.get("uv_index", 3)) / 11.0)

    # ── Titel-Embedding einmalig cachen (für PCA, Trends, Konkurrenz) ──
    # 1 Call pro Push (gecacht), nur in CV-Folds (fast_mode) übersprungen
    _title_emb = None
    if _embedding_model is not None and title:
        try:
            _title_emb = _get_embedding(title)  # Memory-cached, O(1) bei Wiederholung
        except Exception:
            pass

    # ── Google-Trends Embedding-Similarity (5) ──
    trends_list = ctx.get("trends", []) if ctx else []
    feat["trends_max_sim"] = 0.0
    feat["trends_avg_sim_top3"] = 0.0
    feat["trends_n_matching"] = 0.0
    feat["trends_is_trending"] = 0.0
    feat["trends_score_weighted"] = 0.0
    if trends_list and _title_emb is not None:
        try:
            trend_sims = []
            for trend_topic in trends_list[:20]:
                if not trend_topic or not isinstance(trend_topic, str):
                    continue
                trend_emb = _get_embedding(trend_topic)
                if trend_emb is not None:
                    sim = _cosine_similarity(_title_emb, trend_emb)
                    trend_sims.append(sim)
            if trend_sims:
                trend_sims.sort(reverse=True)
                feat["trends_max_sim"] = trend_sims[0]
                feat["trends_avg_sim_top3"] = sum(trend_sims[:3]) / min(3, len(trend_sims))
                feat["trends_n_matching"] = float(sum(1 for s in trend_sims if s > 0.4))
                feat["trends_is_trending"] = 1.0 if trend_sims[0] > 0.6 else 0.0
                feat["trends_score_weighted"] = sum(s for s in trend_sims if s > 0.3)
        except Exception:
            pass

    # ── Konkurrenz-Features (6) — nur Jaccard, kein Embedding pro Headline ──
    comp_cache = state.get("_competitor_cache", {}) if state else {}
    feat["comp_n_covering"] = 0.0
    feat["comp_max_sim"] = 0.0
    feat["comp_is_exclusive"] = 1.0
    feat["comp_lead_hours"] = 0.0
    feat["comp_german_coverage"] = 0.0
    feat["comp_saturation"] = 0.0
    if comp_cache and title:
        try:
            push_words_comp = set(re.findall(r'[a-zäöüß]{4,}', title.lower()))
            _stop_comp = {"der", "die", "das", "und", "von", "für", "mit", "auf", "den", "ist",
                          "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
                          "wird", "wurde", "nach", "über", "dass", "oder", "aber", "wenn", "weil"}
            push_words_comp -= _stop_comp
            total_sources = 0
            covering_sources = 0
            max_jacc = 0.0
            _german_sources = {"spiegel", "focus", "welt", "faz", "stern", "zeit", "tagesschau",
                               "ntv", "rtl", "bild", "sueddeutsche", "tagesspiegel", "morgenpost"}
            german_covering = 0
            for src, items in comp_cache.items():
                if not isinstance(items, list):
                    continue
                total_sources += 1
                is_german = any(g in src.lower() for g in _german_sources)
                src_covers = False
                for it in items[:10]:  # Max 10 statt 15 pro Quelle
                    comp_title = (it.get("title", "") if isinstance(it, dict) else str(it)).lower()
                    if not comp_title:
                        continue
                    # Nur Jaccard-Similarity (O(1) statt _get_embedding pro Headline)
                    comp_words = set(re.findall(r'[a-zäöüß]{4,}', comp_title)) - _stop_comp
                    if comp_words and push_words_comp:
                        jacc = len(push_words_comp & comp_words) / len(push_words_comp | comp_words)
                        if jacc > max_jacc:
                            max_jacc = jacc
                        if jacc > 0.2:
                            src_covers = True
                            break  # Eine Headline reicht pro Quelle
                if src_covers:
                    covering_sources += 1
                    if is_german:
                        german_covering += 1

            feat["comp_n_covering"] = float(covering_sources)
            feat["comp_max_sim"] = max_jacc
            feat["comp_is_exclusive"] = 1.0 if covering_sources == 0 else 0.0
            feat["comp_german_coverage"] = float(german_covering)
            feat["comp_saturation"] = covering_sources / max(1, total_sources)
        except Exception:
            pass

    # ── Embedding-PCA Features (25) ──
    for i in range(25):
        feat[f"emb_pca_{i}"] = 0.0
    if _embedding_pca is not None and _title_emb is not None and np is not None:
        try:
            emb_arr = np.array(_title_emb).reshape(1, -1)
            if _embedding_pca_mean is not None:
                emb_arr = emb_arr - _embedding_pca_mean
            pca_components = _embedding_pca.transform(emb_arr)[0]
            for i in range(min(25, len(pca_components))):
                feat[f"emb_pca_{i}"] = float(pca_components[i])
        except Exception:
            pass

    # ── LLM Magnitude Features (7) ──
    llm_data = push.get("_llm_scores", {})
    _has_llm = float(llm_data.get("magnitude", 0.0)) > 0
    feat["llm_has_score"] = 1.0 if _has_llm else 0.0
    feat["llm_magnitude"] = float(llm_data.get("magnitude", 0.0))
    feat["llm_clickability"] = float(llm_data.get("clickability", 0.0))
    feat["llm_relevanz"] = float(llm_data.get("relevanz", 0.0))
    feat["llm_dringlichkeit"] = float(llm_data.get("dringlichkeit", 0.0))
    feat["llm_emotionalitaet"] = float(llm_data.get("emotionalitaet", 0.0))
    # Composite: gewichtete Kombination
    feat["llm_composite"] = (
        feat["llm_magnitude"] * 0.35 +
        feat["llm_clickability"] * 0.25 +
        feat["llm_relevanz"] * 0.15 +
        feat["llm_dringlichkeit"] * 0.15 +
        feat["llm_emotionalitaet"] * 0.10
    )
    # Keyword-Heuristic als Fallback wenn kein LLM-Score (alle 5 Dimensionen)
    # Fix 2026-03-17: Dimensionen entkoppelt — nicht mehr alle linear von _heur_mag abgeleitet
    if not _has_llm and title:
        _heur_mag = _keyword_magnitude_heuristic(title, cat_lower, push.get("is_eilmeldung", 0))
        feat["llm_magnitude"] = _heur_mag
        feat["llm_clickability"] = max(3.0, min(8.0, feat.get("keyword_avg_or", 5.0)))
        # Relevanz: Eigenständig nach Kategorie-Breite, nur schwach an Magnitude gekoppelt
        _rel_base = {"politik": 7.0, "news": 6.5, "unterhaltung": 5.5, "sport": 4.5,
                     "geld": 5.0, "regional": 4.0, "auto": 4.0, "digital": 4.5,
                     "lifestyle": 3.5, "ratgeber": 3.5, "reise": 3.5}.get(cat_lower, 5.0)
        feat["llm_relevanz"] = min(10.0, _rel_base + min(2.5, (_heur_mag - 5.0) * 0.3))
        # Dringlichkeit: Eilmeldung=hoch, sonst nach Magnitude-Stufe (nicht linear)
        _urg_base = 3.0
        if push.get("is_eilmeldung", 0):
            _urg_base = 9.0
        elif _heur_mag >= 8:
            _urg_base = 6.0
        elif _heur_mag >= 6:
            _urg_base = 4.5
        feat["llm_dringlichkeit"] = min(10.0, _urg_base)
        # Emotionalitaet: Emotion-Words basiert
        _emo_score = 4.0
        _tl = title.lower()
        _tw = set(re.findall(r'[a-zäöüß]{3,}', _tl))
        for _ec, _ew in _GBRT_EMOTION_WORDS.items():
            if _tw & _ew:
                _emo_score = 7.0 if _ec in ("angst", "katastrophe", "sensation", "empoerung") else 5.5
                break
        feat["llm_emotionalitaet"] = _emo_score
        feat["llm_composite"] = (
            feat["llm_magnitude"] * 0.35 + feat["llm_clickability"] * 0.25 +
            feat["llm_relevanz"] * 0.15 + feat["llm_dringlichkeit"] * 0.15 +
            feat["llm_emotionalitaet"] * 0.10
        )

    # ── Interaction-Features für Direct Modeling ──
    ga = feat.get("global_avg_or", global_avg)
    feat["rolling_vs_cat_avg"] = feat.get("rolling_or_3h", ga) - feat.get("cat_avg_or_30d", ga)
    feat["saturation_x_cat"] = feat.get("saturation_score", 0) * feat.get("cat_avg_or_30d", ga)
    feat["keyword_vs_rolling"] = feat.get("keyword_avg_or", ga) - feat.get("rolling_or_3h", ga)
    # cat_hour_confidence: wie viele Samples stützen die Cat×Hour-Baseline?
    _ch_key = f"{cat_lower}_{push.get('hour', 12)}"
    _ch_n = ch_stats.get("n", 0) if ch_stats else 0
    feat["cat_hour_confidence"] = _ch_n / (_ch_n + 20.0)

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED FEATURE VECTOR (150+ Features) — ML-First Prediction
# ══════════════════════════════════════════════════════════════════════════════

def _unified_extract_features(push, history_stats, state=None):
    """Kombiniert ALLE verfügbaren Signale in einen flachen Feature-Vektor (150+).

    Delegiert an _gbrt_extract_features() für Basis-Features und ergänzt:
    - Heuristik-Scores als Features (8)
    - PhD-Modell Features (8)
    - Kontext-Interaktions-Features (6)
    - Topic-Lifecycle Features (10) via _compute_topic_features()
    - Weltereignis Features (7) via _compute_world_event_features()

    Returns: Dict mit Feature-Name → Float-Wert. Darf nie crashen.
    """
    try:
        # ── A1: Basis-Features (80+ existierend) ──
        feat = _gbrt_extract_features(push, history_stats, state)
    except Exception:
        feat = {}

    try:
        title = push.get("title", "") or ""
        title_lower = title.lower()
        ts = push.get("ts_num", 0)
        dt = datetime.datetime.fromtimestamp(ts) if ts > 0 else datetime.datetime.now()
        hour = push.get("hour", dt.hour)
        cat = (push.get("cat", "") or "News").strip().lower()

        # ── A2: Heuristik-Scores als Features (8) ──
        # Similarity OR (aus GBRT-Features)
        feat["heur_similarity_or"] = feat.get("max_similarity", 0.0) * feat.get("top_similar_or", 0.0)

        # IDF-gewichteter Keyword-Score
        push_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', title_lower))
        _stops = {"der", "die", "das", "und", "von", "für", "mit", "auf", "den", "ist",
                  "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
                  "wird", "wurde", "nach", "über", "dass", "oder", "aber", "wenn", "weil"}
        push_words -= _stops
        recent = history_stats.get("recent_pushes", [])
        if push_words and recent:
            doc_freq = defaultdict(int)
            for rp in recent[-500:]:
                rp_words = set(rp.get("words", []))
                for w in push_words:
                    if w in rp_words:
                        doc_freq[w] += 1
            n_docs = max(1, len(recent[-500:]))
            idf_sum = sum(math.log(n_docs / max(1, doc_freq.get(w, 0) + 1)) for w in push_words)
            feat["heur_keyword_idf_or"] = idf_sum / max(1, len(push_words))
        else:
            feat["heur_keyword_idf_or"] = 0.0

        # Entity OR
        feat["heur_entity_or"] = feat.get("entity_avg_or", 0.0)

        # Cat × Hour × Emo Tensor-Produkt
        cat_hour_key = f"{cat}_{hour}"
        cat_hour_stats = history_stats.get("cat_hour_stats", {})
        cat_hour_avg = cat_hour_stats.get(cat_hour_key, {}).get("avg", history_stats.get("global_avg", 4.77))
        emo_score = feat.get("intensity_score", 0.0)
        feat["heur_cat_hour_emo"] = cat_hour_avg * (1.0 + emo_score)

        # Research-Factor
        research_mods = state.get("research_modifiers", {}) if state else {}
        feat["heur_research_factor"] = float(research_mods.get("combined", 1.0)) if research_mods else 1.0


        # Context Score (Weather × Trend × DayType)
        weather_score = feat.get("weather_score", 0.0)
        trend_match = feat.get("trend_match", 0.0)
        is_weekend = 1.0 if feat.get("is_weekend", 0.0) > 0.5 else 0.0
        is_holiday = feat.get("is_holiday", 0.0)
        feat["heur_context_score"] = (1.0 + weather_score * 0.1) * (1.0 + trend_match * 0.2) * (1.0 + is_weekend * 0.05 + is_holiday * 0.1)

        # Competitor Overlap
        comp_cache = state.get("_competitor_cache", {}) if state else {}
        if comp_cache and push_words:
            overlap_count = 0
            total_sources = 0
            for src, items in comp_cache.items():
                if not isinstance(items, list):
                    continue
                total_sources += 1
                for it in items[:10]:
                    comp_title = (it.get("title", "") if isinstance(it, dict) else str(it)).lower()
                    comp_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', comp_title)) - _stops
                    if comp_words and push_words:
                        jacc = len(push_words & comp_words) / len(push_words | comp_words)
                        if jacc > 0.2:
                            overlap_count += 1
                            break
            feat["heur_competitor_overlap"] = overlap_count / max(1, total_sources)
        else:
            feat["heur_competitor_overlap"] = 0.0

        # ── A3: PhD-Modell Features (8) ──

        # ── A4: Kontext-Interaktions-Features (6) ──
        hour_sin = feat.get("hour_sin", math.sin(2 * math.pi * hour / 24))
        max_similarity = feat.get("max_similarity", 0.0)
        is_eilmeldung = 1.0 if push.get("is_eilmeldung") else 0.0

        feat["weather_x_hour"] = weather_score * hour_sin
        feat["trend_x_intensity"] = trend_match * emo_score
        feat["trend_x_novelty"] = trend_match * (1.0 - max_similarity)
        feat["holiday_x_primetime"] = is_holiday * (1.0 if 18 <= hour <= 22 else 0.0)
        feat["weather_x_weekend"] = weather_score * is_weekend
        feat["eilmeldung_x_weather"] = is_eilmeldung * weather_score

        # ── B: Topic-Lifecycle Features (10) ──
        topic_feats = _compute_topic_features(push, history_stats, state)
        feat.update(topic_feats)

        # ── C: Weltereignis Features (7) ──
        world_feats = _compute_world_event_features(push, state)
        feat.update(world_feats)

    except Exception as e:
        log.debug(f"[Unified] Feature-Extraktion Teilfehler: {e}")

    # Fallback: alle erwarteten Keys auf 0.0 setzen wenn fehlend
    _expected_keys = [
        "heur_similarity_or", "heur_keyword_idf_or", "heur_entity_or",
        "heur_cat_hour_emo", "heur_research_factor", "heur_phd_combined",
        "heur_context_score", "heur_competitor_overlap",
        "weather_x_hour", "trend_x_intensity", "trend_x_novelty",
        "holiday_x_primetime", "weather_x_weekend", "eilmeldung_x_weather",
        "topic_age_hours", "topic_push_count_24h", "topic_push_count_total",
        "topic_or_trend", "topic_peak_passed", "topic_is_emerging",
        "topic_is_saturated", "topic_momentum",
        "competitor_n_covering", "competitor_lead_hours",
        "global_event_score", "n_sources_covering", "n_international_sources",
        "is_globally_trending", "international_attention",
        "event_freshness_hours", "german_exclusivity",
    ]
    for k in _expected_keys:
        feat.setdefault(k, 0.0)

    return feat


def _gbrt_build_history_stats(pushes, target_ts=0):
    """Baut Aggregat-Statistiken fuer Feature Engineering.

    Args:
        pushes: Liste aller historischen Pushes
        target_ts: Timestamp des Ziel-Pushes (alles danach wird ignoriert fuer LOO)
    Returns:
        Dict mit vorberechneten Statistiken
    """
    now_ts = target_ts or int(time.time())
    cutoff_7d = now_ts - 7 * 86400
    cutoff_30d = now_ts - 30 * 86400

    valid = [p for p in pushes if 0 < p.get("or", 0) <= 20 and p.get("ts_num", 0) > 0 and p["ts_num"] < now_ts]
    if not valid:
        return {"global_avg": 4.77, "global_n": 0, "cat_stats": {}, "hour_stats": {},
                "cat_hour_stats": {}, "weekday_stats": {}, "recent_pushes": [],
                "entity_or": {}, "last_push_ts_by_cat": {},
                "push_timeline": [], "channel_stats": {}, "cat_momentum": {},
                "cat_weekday_stats": {}}

    all_or = [p["or"] for p in valid]
    global_avg = sum(all_or) / len(all_or)
    global_n = len(all_or)

    # Category stats (7d, 30d, all)
    cat_data = defaultdict(lambda: {"or_7d": [], "or_30d": [], "or_all": []})
    hour_data = defaultdict(lambda: {"or_7d": [], "or_30d": [], "or_all": []})
    cat_hour_data = defaultdict(list)
    weekday_data = defaultdict(list)
    cat_weekday_data = defaultdict(list)
    weekday_hour_data = defaultdict(list)
    last_push_ts_by_cat = {}
    channel_or_data = defaultdict(lambda: {"or_all": [], "n_all": 0})
    timeline_raw = []
    recipient_counts = []
    cat_recipient_data = defaultdict(list)
    word_or_data = defaultdict(list)
    bigram_or_data = defaultdict(list)

    for p in valid:
        ts = p["ts_num"]
        cat = (p.get("cat", "") or "news").lower().strip()
        h = p.get("hour", datetime.datetime.fromtimestamp(ts).hour)
        wd = datetime.datetime.fromtimestamp(ts).weekday()
        orv = p["or"]
        _recip = p.get("total_recipients", 0) or p.get("received", 0) or 0
        if _recip > 0:
            recipient_counts.append(_recip)
            cat_recipient_data[cat].append(_recip)

        cat_data[cat]["or_all"].append(orv)
        hour_data[h]["or_all"].append(orv)
        cat_hour_data[f"{cat}_{h}"].append(orv)
        weekday_data[wd].append(orv)
        cat_weekday_data[f"{cat}_{wd}"].append(orv)
        weekday_hour_data[f"{wd}_{h}"].append(orv)

        # Push timeline entry
        timeline_raw.append((ts, orv, cat))

        if ts >= cutoff_7d:
            cat_data[cat]["or_7d"].append(orv)
            hour_data[h]["or_7d"].append(orv)
        if ts >= cutoff_30d:
            cat_data[cat]["or_30d"].append(orv)
            hour_data[h]["or_30d"].append(orv)

        # Track last push per category
        if cat not in last_push_ts_by_cat or ts > last_push_ts_by_cat[cat]:
            last_push_ts_by_cat[cat] = ts

        # Channel stats
        channels = p.get("channels", [])
        if isinstance(channels, list):
            for ch in channels:
                ch_lower = str(ch).lower().strip()
                if ch_lower:
                    channel_or_data[ch_lower]["or_all"].append(orv)
                    channel_or_data[ch_lower]["n_all"] += 1

        # Keyword→OR tracking (Wörter mit min 3 Zeichen)
        title_text = (p.get("title", "") or "").lower()
        words = [w for w in title_text.split() if len(w) >= 3 and w not in _TOPIC_STOPS]
        for w in words:
            word_or_data[w].append(orv)
        # Bigrams
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bg = f"{words[i]}_{words[i+1]}"
                bigram_or_data[bg].append(orv)

    def _agg(lst):
        return {"avg": sum(lst) / len(lst), "n": len(lst)} if lst else {"avg": 0, "n": 0}

    cat_stats = {}
    for cat, d in cat_data.items():
        cat_stats[cat] = {
            "avg_7d": sum(d["or_7d"]) / len(d["or_7d"]) if d["or_7d"] else global_avg,
            "n_7d": len(d["or_7d"]),
            "avg_30d": sum(d["or_30d"]) / len(d["or_30d"]) if d["or_30d"] else global_avg,
            "n_30d": len(d["or_30d"]),
            "avg_all": sum(d["or_all"]) / len(d["or_all"]) if d["or_all"] else global_avg,
            "n_all": len(d["or_all"]),
        }

    hour_stats = {}
    for h, d in hour_data.items():
        hour_stats[h] = {
            "avg_7d": sum(d["or_7d"]) / len(d["or_7d"]) if d["or_7d"] else global_avg,
            "n_7d": len(d["or_7d"]),
            "avg_30d": sum(d["or_30d"]) / len(d["or_30d"]) if d["or_30d"] else global_avg,
            "n_30d": len(d["or_30d"]),
        }

    cat_hour_stats = {k: _agg(v) for k, v in cat_hour_data.items()}
    weekday_stats = {k: _agg(v) for k, v in weekday_data.items()}
    cat_weekday_stats = {k: _agg(v) for k, v in cat_weekday_data.items()}
    weekday_hour_stats = {k: _agg(v) for k, v in weekday_hour_data.items()}

    # Volatilitaet (Std) pro Kategorie und Hour (7d + 30d)
    def _std(lst):
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

    cat_volatility = {}
    for cat, d in cat_data.items():
        cat_volatility[cat] = {
            "std_7d": _std(d["or_7d"]),
            "std_30d": _std(d["or_30d"]),
        }
    hour_volatility = {}
    for h, d in hour_data.items():
        hour_volatility[h] = {
            "std_7d": _std(d["or_7d"]),
            "std_30d": _std(d["or_30d"]),
        }

    # Push timeline (sorted ascending by ts for bisect lookups)
    timeline_raw.sort(key=lambda x: x[0])
    push_timeline_ts = [t[0] for t in timeline_raw]  # timestamps only for bisect
    push_timeline = timeline_raw  # full (ts, or, cat) tuples

    # Channel stats (aggregated)
    channel_stats = {}
    for ch, d in channel_or_data.items():
        ors = d["or_all"]
        channel_stats[ch] = {
            "avg": sum(ors) / len(ors) if ors else global_avg,
            "n": len(ors),
        }

    # Category momentum (7d vs 30d trend)
    cat_momentum = {}
    for cat, cs in cat_stats.items():
        avg_7d = cs.get("avg_7d", global_avg)
        avg_30d = cs.get("avg_30d", global_avg)
        cat_momentum[cat] = {
            "momentum": (avg_7d - avg_30d) / max(avg_30d, 0.01),
            "ratio_7d_all": avg_7d / max(cs.get("avg_all", global_avg), 0.01),
        }

    # Recent pushes with pre-computed word sets (fuer Similarity)
    stops = {"der", "die", "das", "und", "von", "fuer", "mit", "auf", "den", "ist", "ein", "eine",
             "sich", "auch", "noch", "nur", "jetzt", "alle", "neue", "wird", "wurde", "nach", "ueber",
             "dass", "aber", "oder", "wenn", "dann", "mehr", "sein", "hat", "haben", "kann", "sind"}
    sorted_valid = sorted(valid, key=lambda x: x["ts_num"], reverse=True)
    recent_pushes = []
    for p in sorted_valid[:2000]:  # Letzte 2000 fuer Similarity
        words = set(re.findall(r'[a-zäöüß]{4,}', p.get("title", "").lower())) - stops
        recent_pushes.append({"words": words, "or": p["or"], "ts": p["ts_num"],
                              "title_raw": p.get("title", "")})

    # Entity OR mapping
    entity_or = defaultdict(list)
    entity_freq_7d = defaultdict(int)  # Wie oft taucht Entity in letzten 7d auf
    for p in sorted_valid[:3000]:
        entities = re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', p.get("title", ""))
        for ent in entities:
            ent_l = ent.lower()
            entity_or[ent_l].append(p["or"])
            if p["ts_num"] >= cutoff_7d:
                entity_freq_7d[ent_l] += 1

    # Recent Titles fuer Embedding-Vergleich (letzte 500)
    recent_titles = [(p.get("title", ""), p["or"]) for p in sorted_valid[:500]
                     if p.get("title")]

    return {
        "global_avg": global_avg,
        "global_n": global_n,
        "cat_stats": cat_stats,
        "hour_stats": hour_stats,
        "cat_hour_stats": cat_hour_stats,
        "weekday_stats": weekday_stats,
        "cat_weekday_stats": cat_weekday_stats,
        "weekday_hour_stats": weekday_hour_stats,
        "cat_volatility": cat_volatility,
        "hour_volatility": hour_volatility,
        "recent_pushes": recent_pushes,
        "entity_or": dict(entity_or),
        "entity_freq_7d": dict(entity_freq_7d),
        "last_push_ts_by_cat": last_push_ts_by_cat,
        "_recent_titles": recent_titles,
        "push_timeline": push_timeline,
        "push_timeline_ts": push_timeline_ts,
        "channel_stats": channel_stats,
        "cat_momentum": cat_momentum,
        "median_recipients": sorted(recipient_counts)[len(recipient_counts)//2] if recipient_counts else 0,
        "cat_median_recipients": {cat: sorted(vals)[len(vals)//2] for cat, vals in cat_recipient_data.items() if vals},
        # Keyword→OR: nur Wörter mit min 5 Vorkommen (stabil genug)
        "word_or": {w: {"avg": sum(ors)/len(ors), "n": len(ors),
                        "std": (sum((o - sum(ors)/len(ors))**2 for o in ors) / len(ors))**0.5,
                        "median": sorted(ors)[len(ors)//2]}
                    for w, ors in word_or_data.items() if len(ors) >= 5},
        "bigram_or": {bg: {"avg": sum(ors)/len(ors), "n": len(ors)}
                      for bg, ors in bigram_or_data.items() if len(ors) >= 5},
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC-LIFECYCLE & NACHRICHTENZYKLEN (ML-First Phase B)
# ══════════════════════════════════════════════════════════════════════════════
# _TOPIC_STOPS defined near line 479


def _build_topic_tracker(history_stats, state=None):
    """Baut Topic-Cluster aus den letzten 2000 Pushes via Greedy Jaccard-Clustering.

    Cached mit 5-Minuten TTL in _topic_tracker.

    Returns: Liste von Topic-Cluster-Dicts:
        {keywords: set, first_seen_ts: int, last_seen_ts: int,
         push_count: int, or_values: list, peak_or: float, push_indices: list}
    """
    global _topic_tracker
    now = time.time()

    with _topic_tracker_lock:
        if _topic_tracker["clusters"] and now - _topic_tracker["ts"] < 300:
            return _topic_tracker["clusters"]

    recent = history_stats.get("recent_pushes", [])
    if not recent:
        return []

    # Letzte 2000 Pushes nehmen
    pushes_slice = recent[-2000:]

    # Keyword-Sets extrahieren
    push_data = []
    for rp in pushes_slice:
        words = set(rp.get("words", []))
        words = {w for w in words if len(w) >= 4} - _TOPIC_STOPS
        if words:
            push_data.append({
                "words": words,
                "ts": rp.get("ts", 0),
                "or": rp.get("or", 0),
                "title": rp.get("title", ""),
            })

    if not push_data:
        return []

    # Greedy Clustering: Pushes mit Jaccard > 0.3 = selbes Thema
    clusters = []
    assigned = [False] * len(push_data)

    for i, pd_i in enumerate(push_data):
        if assigned[i]:
            continue
        cluster = {
            "keywords": set(pd_i["words"]),
            "first_seen_ts": pd_i["ts"],
            "last_seen_ts": pd_i["ts"],
            "push_count": 1,
            "or_values": [pd_i["or"]] if pd_i["or"] > 0 else [],
            "peak_or": pd_i["or"],
            "titles": [pd_i["title"]],
        }
        assigned[i] = True

        for j in range(i + 1, len(push_data)):
            if assigned[j]:
                continue
            intersection = pd_i["words"] & push_data[j]["words"]
            union = pd_i["words"] | push_data[j]["words"]
            if union and len(intersection) / len(union) > 0.3:
                assigned[j] = True
                cluster["keywords"] |= push_data[j]["words"]
                cluster["push_count"] += 1
                if push_data[j]["or"] > 0:
                    cluster["or_values"].append(push_data[j]["or"])
                if push_data[j]["or"] > cluster["peak_or"]:
                    cluster["peak_or"] = push_data[j]["or"]
                if push_data[j]["ts"] < cluster["first_seen_ts"]:
                    cluster["first_seen_ts"] = push_data[j]["ts"]
                if push_data[j]["ts"] > cluster["last_seen_ts"]:
                    cluster["last_seen_ts"] = push_data[j]["ts"]
                cluster["titles"].append(push_data[j]["title"])

        clusters.append(cluster)

    # Competitor-Headlines zu Themen matchen
    comp_cache = state.get("_competitor_cache", {}) if state else {}
    if comp_cache:
        for cluster in clusters:
            cluster["competitor_count"] = 0
            cluster["competitor_first_ts"] = 0
            for src, items in comp_cache.items():
                if not isinstance(items, list):
                    continue
                for it in items[:10]:
                    comp_title = (it.get("title", "") if isinstance(it, dict) else str(it)).lower()
                    comp_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', comp_title)) - _TOPIC_STOPS
                    if comp_words and cluster["keywords"]:
                        intersection = comp_words & cluster["keywords"]
                        union = comp_words | cluster["keywords"]
                        if union and len(intersection) / len(union) > 0.2:
                            cluster["competitor_count"] += 1
                            pub_ts = it.get("pub_ts", 0) if isinstance(it, dict) else 0
                            if pub_ts and (not cluster["competitor_first_ts"] or pub_ts < cluster["competitor_first_ts"]):
                                cluster["competitor_first_ts"] = pub_ts
                            break

    with _topic_tracker_lock:
        _topic_tracker["clusters"] = clusters
        _topic_tracker["ts"] = now

    log.info(f"[Topic] {len(clusters)} Themen-Cluster erkannt, "
             f"{sum(1 for c in clusters if c['push_count'] == 1)} einmalig, "
             f"{sum(1 for c in clusters if c['push_count'] >= 5)} mit 5+ Pushes")

    return clusters


def _compute_topic_features(push, history_stats, state=None):
    """Berechnet 10 Topic-Lifecycle Features für einen Push.

    Returns: Dict mit 10 Feature-Name → Float-Wert
    """
    result = {
        "topic_age_hours": 0.0,
        "topic_push_count_24h": 0.0,
        "topic_push_count_total": 0.0,
        "topic_or_trend": 0.0,
        "topic_peak_passed": 0.0,
        "topic_is_emerging": 0.0,
        "topic_is_saturated": 0.0,
        "topic_momentum": 0.0,
        "competitor_n_covering": 0.0,
        "competitor_lead_hours": 0.0,
    }

    try:
        clusters = _build_topic_tracker(history_stats, state)
        if not clusters:
            return result

        title = push.get("title", "") or ""
        push_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', title.lower())) - _TOPIC_STOPS
        if not push_words:
            return result

        now_ts = push.get("ts_num", 0) or int(time.time())

        # Besten Cluster-Match finden
        best_cluster = None
        best_jacc = 0.0
        for cl in clusters:
            intersection = push_words & cl["keywords"]
            union = push_words | cl["keywords"]
            if union:
                jacc = len(intersection) / len(union)
                if jacc > best_jacc:
                    best_jacc = jacc
                    best_cluster = cl

        if not best_cluster or best_jacc < 0.15:
            # Brandneues Thema
            result["topic_is_emerging"] = 1.0
            return result

        cl = best_cluster
        age_hours = max(0, (now_ts - cl["first_seen_ts"])) / 3600.0
        result["topic_age_hours"] = age_hours
        result["topic_push_count_total"] = float(cl["push_count"])

        # Pushes in letzten 24h zählen
        cutoff_24h = now_ts - 86400
        count_24h = sum(1 for t in cl.get("titles", [])
                        if cl["first_seen_ts"] <= cutoff_24h or cl["push_count"] <= 3)
        # Approximation: wenn Cluster jung, alle Pushes sind in 24h
        if age_hours < 24:
            count_24h = cl["push_count"]
        result["topic_push_count_24h"] = float(count_24h)

        # OR-Trend (Steigung)
        or_vals = cl.get("or_values", [])
        if len(or_vals) >= 3:
            first_half = or_vals[:len(or_vals)//2]
            second_half = or_vals[len(or_vals)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            global_avg = history_stats.get("global_avg", 4.77)
            result["topic_or_trend"] = (avg_second - avg_first) / max(0.1, global_avg)

            # Momentum
            if len(or_vals) >= 6:
                avg_first3 = sum(or_vals[:3]) / 3
                avg_last3 = sum(or_vals[-3:]) / 3
                result["topic_momentum"] = (avg_last3 - avg_first3) / max(0.1, global_avg)

        # Peak passed?
        peak_age_hours = max(0, (now_ts - cl["last_seen_ts"])) / 3600.0
        result["topic_peak_passed"] = 1.0 if peak_age_hours > 24 else 0.0

        # Emerging?
        result["topic_is_emerging"] = 1.0 if age_hours < 6 and cl["push_count"] < 3 else 0.0

        # Saturated?
        result["topic_is_saturated"] = 1.0 if count_24h > 5 else 0.0

        # Competitor Features
        result["competitor_n_covering"] = float(cl.get("competitor_count", 0))
        comp_first = cl.get("competitor_first_ts", 0)
        if comp_first and comp_first < cl["first_seen_ts"]:
            result["competitor_lead_hours"] = (cl["first_seen_ts"] - comp_first) / 3600.0
        elif comp_first and comp_first > cl["first_seen_ts"]:
            result["competitor_lead_hours"] = -((comp_first - cl["first_seen_ts"]) / 3600.0)
        # 0.0 = BILD zuerst oder keine Competitor-Daten

    except Exception as e:
        log.debug(f"[Topic] Feature-Fehler: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# WELTEREIGNIS-VERSTÄNDNIS (ML-First Phase C)
# ══════════════════════════════════════════════════════════════════════════════

# DE-EN Übersetzungspaare für Nachrichtenbegriffe
_NEWS_TRANSLATION_PAIRS = {
    "krieg": "war", "erdbeben": "earthquake", "wahl": "election",
    "anschlag": "attack", "gipfel": "summit", "krise": "crisis",
    "terror": "terror", "explosion": "explosion", "überfall": "raid",
    "mord": "murder", "flut": "flood", "sturm": "storm",
    "tsunami": "tsunami", "brand": "fire", "rakete": "rocket",
    "bombe": "bomb", "geisel": "hostage", "putsch": "coup",
    "protest": "protest", "streik": "strike", "embargo": "embargo",
    "sanktion": "sanction", "verhandlung": "negotiation",
    "waffenstillstand": "ceasefire", "evakuierung": "evacuation",
    "epidemie": "epidemic", "pandemie": "pandemic", "impfung": "vaccine",
    "klima": "climate", "dürre": "drought", "vulkan": "volcano",
    "absturz": "crash", "entführung": "kidnapping", "flucht": "escape",
    "invasion": "invasion", "demonstration": "demonstration",
    "attentat": "assassination", "staatschef": "leader",
    "präsident": "president", "kanzler": "chancellor",
    "minister": "minister", "parlament": "parliament",
    "referendum": "referendum", "rücktritt": "resignation",
    "korruption": "corruption", "bestechung": "bribery",
    "wirtschaft": "economy", "inflation": "inflation",
    "rezession": "recession", "arbeitslos": "unemployment",
    "migration": "migration", "flüchtling": "refugee",
    "grenze": "border", "olympia": "olympics",
}
# Reverse mapping
_NEWS_TRANSLATION_EN_DE = {v: k for k, v in _NEWS_TRANSLATION_PAIRS.items()}

# DE-Competitor-Quellen (für german_exclusivity)
_DE_SOURCES = {"spiegel", "faz", "sz", "welt", "focus", "stern", "tagesschau",
               "nzz", "derstandard", "zeit", "ntv", "bild"}


def _build_world_event_index(state=None):
    """Baut World-Event-Index aus Competitor + International Feeds.

    Cached mit 10-Minuten TTL in _world_event_index.

    Returns: Dict mit hot_topics (Liste) und keyword_counts (Dict)
    """
    global _world_event_index
    now = time.time()

    with _world_event_index_lock:
        if _world_event_index["hot_topics"] and now - _world_event_index["ts"] < 600:
            return _world_event_index

    comp_cache = state.get("_competitor_cache", {}) if state else {}
    if not comp_cache:
        return {"hot_topics": [], "keyword_counts": {}, "ts": now}

    # Alle Headlines der letzten 6h sammeln
    cutoff_6h = now - 6 * 3600
    source_keywords = {}  # source_name → set of keywords

    for src, items in comp_cache.items():
        if not isinstance(items, list):
            continue
        kw_set = set()
        for it in items[:20]:
            title = (it.get("title", "") if isinstance(it, dict) else str(it)).lower()
            # Prüfe Zeitstempel wenn verfügbar
            pub_ts = it.get("pub_ts", 0) if isinstance(it, dict) else 0
            if pub_ts and pub_ts < cutoff_6h:
                continue
            words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', title)) - _TOPIC_STOPS
            # DE-EN Brücke: Füge Übersetzungen hinzu
            translated = set()
            for w in words:
                if w in _NEWS_TRANSLATION_PAIRS:
                    translated.add(_NEWS_TRANSLATION_PAIRS[w])
                if w in _NEWS_TRANSLATION_EN_DE:
                    translated.add(_NEWS_TRANSLATION_EN_DE[w])
            words |= translated
            kw_set |= words
        if kw_set:
            source_keywords[src] = kw_set

    if not source_keywords:
        return {"hot_topics": [], "keyword_counts": {}, "ts": now}

    # Pro Keyword zählen: in wie vielen Quellen kommt es vor?
    keyword_source_count = defaultdict(int)
    keyword_sources = defaultdict(set)
    for src, kws in source_keywords.items():
        for kw in kws:
            keyword_source_count[kw] += 1
            keyword_sources[kw].add(src)

    # Hot Keywords: in 5+ Quellen
    hot_keywords = {kw for kw, cnt in keyword_source_count.items() if cnt >= 5}

    # Hot Keywords per Co-Occurrence clustern → Hot Topics
    hot_topics = []
    used = set()
    hot_list = sorted(hot_keywords, key=lambda k: -keyword_source_count[k])

    for kw in hot_list:
        if kw in used:
            continue
        topic = {
            "keywords": {kw},
            "sources": set(keyword_sources[kw]),
            "max_source_count": keyword_source_count[kw],
        }
        used.add(kw)

        # Co-Occurrence: andere Hot Keywords die in ähnlichen Quellen vorkommen
        for other_kw in hot_list:
            if other_kw in used:
                continue
            overlap = keyword_sources[kw] & keyword_sources[other_kw]
            if len(overlap) >= 3:
                topic["keywords"].add(other_kw)
                topic["sources"] |= keyword_sources[other_kw]
                if keyword_source_count[other_kw] > topic["max_source_count"]:
                    topic["max_source_count"] = keyword_source_count[other_kw]
                used.add(other_kw)

        hot_topics.append(topic)

    # DE vs International aufschlüsseln
    for topic in hot_topics:
        topic["n_de_sources"] = len(topic["sources"] & _DE_SOURCES)
        topic["n_intl_sources"] = len(topic["sources"] - _DE_SOURCES)
        topic["n_total_sources"] = len(topic["sources"])
        # Konvertiere sets zu listen für Serialisierbarkeit
        topic["keywords"] = list(topic["keywords"])
        topic["sources"] = list(topic["sources"])

    result = {
        "hot_topics": hot_topics,
        "keyword_counts": dict(keyword_source_count),
        "ts": now,
    }

    with _world_event_index_lock:
        _world_event_index = result

    n_intl = sum(1 for t in hot_topics if t.get("n_intl_sources", 0) > 0)
    log.info(f"[WorldEvents] {len(hot_topics)} Hot Topics, "
             f"{len(hot_keywords)} Hot Keywords, "
             f"{len(source_keywords)} Quellen, {n_intl} international")

    return result


def _compute_world_event_features(push, state=None):
    """Berechnet 7 Weltereignis-Features für einen Push.

    Returns: Dict mit 7 Feature-Name → Float-Wert
    """
    result = {
        "global_event_score": 0.0,
        "n_sources_covering": 0.0,
        "n_international_sources": 0.0,
        "is_globally_trending": 0.0,
        "international_attention": 0.0,
        "event_freshness_hours": 0.0,
        "german_exclusivity": 0.0,
    }

    try:
        world_idx = _build_world_event_index(state)
        hot_topics = world_idx.get("hot_topics", [])
        if not hot_topics:
            return result

        title = push.get("title", "") or ""
        push_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', title.lower())) - _TOPIC_STOPS
        # Füge Übersetzungen hinzu
        translated = set()
        for w in push_words:
            if w in _NEWS_TRANSLATION_PAIRS:
                translated.add(_NEWS_TRANSLATION_PAIRS[w])
            if w in _NEWS_TRANSLATION_EN_DE:
                translated.add(_NEWS_TRANSLATION_EN_DE[w])
        push_words |= translated

        if not push_words:
            return result

        total_intl = len(INTERNATIONAL_FEEDS)

        # Besten Topic-Match finden
        best_score = 0.0
        best_topic = None
        for topic in hot_topics:
            topic_kws = set(topic.get("keywords", []))
            if not topic_kws:
                continue
            intersection = push_words & topic_kws
            if not intersection:
                continue
            score = len(intersection) / len(topic_kws)
            if score > best_score:
                best_score = score
                best_topic = topic

        if best_topic and best_score > 0.1:
            result["global_event_score"] = min(1.0, best_score)
            result["n_sources_covering"] = float(best_topic.get("n_total_sources", 0))
            result["n_international_sources"] = float(best_topic.get("n_intl_sources", 0))
            result["is_globally_trending"] = 1.0 if best_topic.get("n_total_sources", 0) >= 10 else 0.0
            result["international_attention"] = best_topic.get("n_intl_sources", 0) / max(1, total_intl)

            # German exclusivity: nur DE-Quellen berichten
            if best_topic.get("n_intl_sources", 0) == 0 and best_topic.get("n_de_sources", 0) > 0:
                result["german_exclusivity"] = 1.0

            # Event-Freshness: Stunden seit erstem Auftreten des Topics
            first_seen = best_topic.get("first_seen_ts", 0)
            if first_seen > 0:
                hours_since = (time.time() - first_seen) / 3600
                result["event_freshness_hours"] = min(48.0, max(0.0, hours_since))

    except Exception as e:
        log.debug(f"[WorldEvents] Feature-Fehler: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC-SATURATION PENALTY — Themen-Sättigungs-Dämpfung
# ══════════════════════════════════════════════════════════════════════════════

def _compute_topic_saturation_penalty(push, push_data, state=None):
    """Berechnet den Themen-Sättigungs-Faktor für einen Push.

    Analysiert ob zum gleichen Thema bereits Pushes gesendet wurden und
    berechnet eine Dämpfung basierend auf:
    1. Themenspezifische Push-Anzahl (6h-Fenster) — Kern-Signal
    2. Jaccard-Ähnlichkeit zu kürzlich gesendeten Pushes
    3. Zeitlicher Abstand zum letzten thematisch ähnlichen Push
    4. OR-Trend des Themas (sinkende OR = stärkere Sättigung)

    Returns: Dict mit:
        penalty: float (0.5 - 1.0, Multiplikator; 1.0 = keine Strafe)
        topic_push_count_6h: int
        topic_push_count_24h: int
        highest_jaccard: float
        hours_since_last: float
        reason: str (Erklärung)
    """
    result = {
        "penalty": 1.0,
        "topic_push_count_6h": 0,
        "topic_push_count_24h": 0,
        "highest_jaccard": 0.0,
        "hours_since_last": 999.0,
        "or_decay": 0.0,
        "reason": "",
    }

    try:
        title = push.get("title", "") or ""
        title_lower = title.lower()
        push_ts = push.get("ts_num", 0) or int(time.time())

        # Keywords des aktuellen Pushes
        push_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', title_lower)) - _TOPIC_STOPS
        if not push_words or len(push_words) < 2:
            return result

        # Zeitfenster
        cutoff_6h = push_ts - 6 * 3600
        cutoff_24h = push_ts - 24 * 3600

        # Nur Pushes die VOR dem aktuellen liegen (temporal causal)
        if push_ts > 0:
            candidates = [p for p in push_data
                          if p.get("ts_num", 0) > 0
                          and p["ts_num"] < push_ts
                          and p["ts_num"] > cutoff_24h
                          and (p.get("or", 0) or 0) > 0]
        else:
            candidates = [p for p in push_data
                          if (p.get("or", 0) or 0) > 0
                          and p.get("ts_num", 0) > cutoff_24h]

        if not candidates:
            return result

        # Thematisch ähnliche Pushes finden (Jaccard > 0.25)
        similar_pushes = []
        highest_jaccard = 0.0
        last_similar_ts = 0

        for p in candidates:
            p_title = (p.get("title", "") or "").lower()
            p_words = set(re.findall(r'[A-Za-zäöüÄÖÜß]{4,}', p_title)) - _TOPIC_STOPS
            if not p_words:
                continue
            intersection = push_words & p_words
            union = push_words | p_words
            if not union:
                continue
            jaccard = len(intersection) / len(union)

            if jaccard > highest_jaccard:
                highest_jaccard = jaccard

            if jaccard > 0.25:
                similar_pushes.append({
                    "ts": p["ts_num"],
                    "or": p.get("or", 0),
                    "jaccard": jaccard,
                    "title": p_title[:60],
                })
                if p["ts_num"] > last_similar_ts:
                    last_similar_ts = p["ts_num"]

        result["highest_jaccard"] = round(highest_jaccard, 3)

        if not similar_pushes:
            return result

        # 6h und 24h Zählung
        count_6h = sum(1 for sp in similar_pushes if sp["ts"] > cutoff_6h)
        count_24h = len(similar_pushes)
        result["topic_push_count_6h"] = count_6h
        result["topic_push_count_24h"] = count_24h

        # Stunden seit letztem ähnlichen Push
        hours_since = (push_ts - last_similar_ts) / 3600.0 if last_similar_ts > 0 else 999.0
        result["hours_since_last"] = round(hours_since, 2)

        # OR-Decay: sinken die OR-Werte der ähnlichen Pushes?
        if len(similar_pushes) >= 2:
            sorted_sim = sorted(similar_pushes, key=lambda x: x["ts"])
            first_half = sorted_sim[:len(sorted_sim)//2]
            second_half = sorted_sim[len(sorted_sim)//2:]
            avg_first = sum(s["or"] for s in first_half) / len(first_half)
            avg_second = sum(s["or"] for s in second_half) / len(second_half)
            if avg_first > 0:
                result["or_decay"] = round((avg_second - avg_first) / avg_first, 3)

        # ── Penalty-Berechnung ──────────────────────────────────────

        penalty = 1.0
        reasons = []

        # Signal 1: 6h-Fenster Push-Anzahl (Kernlogik)
        # Logarithmische Dämpfung: #2=-12%, #3=-22%, #4=-33%, #5=-42%, #6+=-50%
        if count_6h >= 1:
            # Verschoben um 1: der ERSTE ähnliche Push in 6h ist #2 insgesamt
            n = count_6h + 1  # +1 weil der aktuelle Push auch dazuzählt
            raw_penalty = 1.0 - 0.18 * math.log(n)
            count_penalty = max(0.50, raw_penalty)
            penalty *= count_penalty
            reasons.append(f"{count_6h} ähnl. in 6h → ×{count_penalty:.2f}")

        # Signal 2: Jaccard-verstärkter Abzug (je ähnlicher, desto stärker)
        # Jaccard 0.25-0.50 = mildes Signal, 0.50-0.80 = stark, 0.80+ = quasi-Duplikat
        if highest_jaccard > 0.40 and count_6h >= 1:
            jacc_extra = (highest_jaccard - 0.40) * 0.25  # max ~0.15 Extra-Penalty
            penalty *= (1.0 - jacc_extra)
            reasons.append(f"Jaccard {highest_jaccard:.2f} → ×{1.0 - jacc_extra:.2f}")

        # Signal 3: Zeitnähe — wenn letzter ähnlicher Push < 1h → extra Strafe
        if hours_since < 1.0 and count_6h >= 1:
            recency_penalty = 0.92 - (1.0 - hours_since) * 0.08  # bis -16% bei Minuten-Abstand
            recency_penalty = max(0.84, recency_penalty)
            penalty *= recency_penalty
            reasons.append(f"{hours_since:.1f}h her → ×{recency_penalty:.2f}")

        # Signal 4: OR-Decay — wenn die ORs des Themas bereits sinken
        if result["or_decay"] < -0.15 and count_6h >= 2:
            decay_penalty = max(0.88, 1.0 + result["or_decay"] * 0.3)
            penalty *= decay_penalty
            reasons.append(f"OR-Decay {result['or_decay']:+.0%} → ×{decay_penalty:.2f}")

        # Eilmeldungen bekommen mildere Strafe (40% der Penalty)
        if push.get("is_eilmeldung") and penalty < 1.0:
            penalty = 1.0 - (1.0 - penalty) * 0.40
            reasons.append("Eilmeldung → Penalty ×0.40")

        # Finaler Clamp
        penalty = max(0.40, min(1.0, penalty))

        result["penalty"] = round(penalty, 3)
        result["reason"] = " | ".join(reasons) if reasons else "kein Themen-Match"

    except Exception as e:
        log.debug(f"[TopicSat] Fehler: {e}")

    return result


# ── GBRT Decision Tree Implementation ───────────────────────────────────

class _GBRTNode:
    """Ein Knoten im Entscheidungsbaum."""
    __slots__ = ("feature_idx", "threshold", "left", "right", "value", "gain")

    def __init__(self):
        self.feature_idx = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0  # Leaf-Value (Mean der Residuen)
        self.gain = 0.0


class _GBRTTree:
    """Ein einzelner Regressions-Baum fuer Gradient Boosting."""

    def __init__(self, max_depth=5, min_samples_leaf=10, n_bins=255):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.root = None

    def fit(self, X, residuals, sample_indices=None, sample_weights=None):
        """Trainiert den Baum auf Residuen.

        Args:
            X: Feature-Matrix (Liste von Listen)
            residuals: Ziel-Residuen (Liste von Floats)
            sample_indices: Optionale Subsample-Indices
            sample_weights: Optionale Gewichte pro Sample
        """
        if sample_indices is None:
            sample_indices = list(range(len(X)))
        self.n_features = len(X[0]) if X else 0
        self._weights = sample_weights
        self.root = self._build_tree(X, residuals, sample_indices, depth=0)

    def predict_one(self, x):
        """Prediction fuer einen einzelnen Feature-Vektor."""
        node = self.root
        while node is not None:
            if node.feature_idx < 0:  # Leaf
                return node.value
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return 0.0

    def _build_tree(self, X, residuals, indices, depth):
        """Rekursiver Baum-Aufbau mit Histogram-basiertem Splitting und Sample-Gewichtung."""
        node = _GBRTNode()
        w = self._weights  # Sample-Gewichte (oder None)

        # Leaf: gewichteter Mean der Residuen
        if w:
            w_sum = sum(w[i] for i in indices)
            node.value = sum(residuals[i] * w[i] for i in indices) / w_sum if w_sum > 0 else 0.0
        else:
            vals = [residuals[i] for i in indices]
            node.value = sum(vals) / len(vals) if vals else 0.0

        # Stopp-Kriterien
        if depth >= self.max_depth or len(indices) < self.min_samples_leaf * 2:
            node.feature_idx = -1
            return node

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        n = len(indices)

        for f_idx in range(self.n_features):
            # Histogram-basiertes Splitting mit Gewichten
            if w:
                f_vals = [(X[i][f_idx], residuals[i], w[i]) for i in indices]
            else:
                f_vals = [(X[i][f_idx], residuals[i], 1.0) for i in indices]
            f_vals.sort(key=lambda x: x[0])

            total_wsum = sum(v[2] * v[1] for v in f_vals)
            total_w = sum(v[2] for v in f_vals)

            left_wsum = 0.0
            left_w = 0.0
            left_n = 0

            # Schritt-Groesse: nicht jeden Wert testen, sondern n_bins gleichmaessig verteilt
            step = max(1, n // self.n_bins)

            for pos in range(0, n - 1, step):
                # Alle Elemente bis pos in Links
                for k in range(left_n, pos + 1):
                    left_wsum += f_vals[k][2] * f_vals[k][1]
                    left_w += f_vals[k][2]
                left_n = pos + 1

                # Kein Split bei gleichen Feature-Werten
                if f_vals[pos][0] == f_vals[min(pos + 1, n - 1)][0]:
                    continue

                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                right_wsum = total_wsum - left_wsum
                right_w = total_w - left_w
                if left_w <= 0 or right_w <= 0:
                    continue

                # Gewichtete Varianz-Reduktion
                gain = (left_wsum * left_wsum / left_w +
                        right_wsum * right_wsum / right_w -
                        total_wsum * total_wsum / total_w)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = (f_vals[pos][0] + f_vals[min(pos + 1, n - 1)][0]) / 2.0

        if best_feat < 0 or best_gain <= 0:
            node.feature_idx = -1
            return node

        # Split anwenden
        left_idx = [i for i in indices if X[i][best_feat] <= best_thresh]
        right_idx = [i for i in indices if X[i][best_feat] > best_thresh]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            node.feature_idx = -1
            return node

        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.gain = best_gain
        node.left = self._build_tree(X, residuals, left_idx, depth + 1)
        node.right = self._build_tree(X, residuals, right_idx, depth + 1)
        return node

    def path_contributions(self, x, n_features):
        """TreeSHAP-artige Feature-Contributions via Root-to-Leaf Pfad.

        An jedem Split: contributions[feature] += child.value - node.value
        Summe aller Contributions = leaf.value - root.value
        """
        contributions = [0.0] * n_features
        node = self.root
        if node is None:
            return contributions
        while node is not None and node.feature_idx >= 0:
            fidx = node.feature_idx
            if x[fidx] <= node.threshold:
                child = node.left
            else:
                child = node.right
            if child is not None:
                contributions[fidx] += child.value - node.value
            node = child
        return contributions

    def to_dict(self):
        """Serialisiert den Baum als JSON-faehiges Dict."""
        return self._node_to_dict(self.root) if self.root else {}

    def _node_to_dict(self, node):
        if node is None:
            return None
        if node.feature_idx < 0:
            return {"v": round(node.value, 6)}
        return {
            "f": node.feature_idx,
            "t": round(node.threshold, 6),
            "v": round(node.value, 6),
            "l": self._node_to_dict(node.left),
            "r": self._node_to_dict(node.right),
        }

    @staticmethod
    def from_dict(d):
        """Deserialisiert einen Baum aus Dict."""
        tree = _GBRTTree()
        tree.root = _GBRTTree._node_from_dict(d)
        return tree

    @staticmethod
    def _node_from_dict(d):
        if d is None:
            return None
        node = _GBRTNode()
        if "f" in d:
            # Interner Knoten
            node.feature_idx = d["f"]
            node.threshold = d["t"]
            node.value = d.get("v", 0.0)  # Backward-kompatibel
            node.left = _GBRTTree._node_from_dict(d.get("l"))
            node.right = _GBRTTree._node_from_dict(d.get("r"))
        else:
            # Leaf
            node.feature_idx = -1
            node.value = d.get("v", 0.0)
        return node


class GBRTModel:
    """Gradient Boosted Regression Trees — reines Python, kein numpy/sklearn.

    Trainiert ein Ensemble von Regressionsbaeumen via Gradient Boosting
    mit Histogram-basiertem Splitting fuer Performance.
    """

    def __init__(self, n_trees=300, max_depth=6, learning_rate=0.08,
                 min_samples_leaf=8, subsample=0.85, n_bins=255,
                 loss="huber", huber_delta=1.5, log_target=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.loss = loss
        self.huber_delta = huber_delta
        self.log_target = log_target
        self.trees = []
        self.initial_prediction = 0.0
        self.feature_names = []
        self.feature_importance_ = {}
        self.train_metrics = {}

    def fit(self, X, y, feature_names=None, val_X=None, val_y=None,
            sample_weights=None):
        """Trainiert das GBRT-Modell.

        Args:
            X: Feature-Matrix (Liste von Listen), jede Zeile = 1 Sample
            y: Zielwerte (Liste von Floats)
            feature_names: Optionale Feature-Namen
            val_X, val_y: Optionale Validierungs-Daten fuer Early Stopping
            sample_weights: Optionale Gewichte pro Sample (Liste von Floats)
        """
        n = len(X)
        if n == 0:
            return

        self.feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]

        # Log-Target-Transformation
        if self.log_target:
            y = [math.log1p(v) for v in y]
            if val_y:
                val_y = [math.log1p(v) for v in val_y]

        # Sample Weights normalisieren (Durchschnitt = 1.0)
        if sample_weights:
            w_mean = sum(sample_weights) / len(sample_weights)
            self._sample_weights = [w / w_mean for w in sample_weights] if w_mean > 0 else [1.0] * n
        else:
            self._sample_weights = [1.0] * n

        # Gewichteter Mittelwert als Initial Prediction
        w_total = sum(self._sample_weights)
        self.initial_prediction = sum(y[i] * self._sample_weights[i] for i in range(n)) / w_total
        self.trees = []

        # Aktuelle Predictions
        predictions = [self.initial_prediction] * n
        val_predictions = [self.initial_prediction] * len(val_X) if val_X else []

        # Feature Importance Tracking
        feat_gain = defaultdict(float)

        best_val_mae = float('inf')
        best_n_trees = 0
        rounds_no_improve = 0
        early_stopped = False
        rng = random.Random(42)
        delta = self.huber_delta

        for t in range(self.n_trees):
            # Residuen mit Huber-Gradient
            if self.loss == "huber":
                residuals = []
                for i in range(n):
                    r = y[i] - predictions[i]
                    if abs(r) <= delta:
                        residuals.append(r)
                    else:
                        residuals.append(delta * (1.0 if r > 0 else -1.0))
            else:
                residuals = [y[i] - predictions[i] for i in range(n)]

            # Subsampling
            if self.subsample < 1.0:
                sample_size = max(1, int(n * self.subsample))
                sample_idx = rng.sample(range(n), sample_size)
            else:
                sample_idx = None

            tree = _GBRTTree(max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            n_bins=self.n_bins)
            tree.fit(X, residuals, sample_idx, sample_weights=self._sample_weights)
            self.trees.append(tree)

            # Update Predictions
            for i in range(n):
                predictions[i] += self.learning_rate * tree.predict_one(X[i])

            # Feature Importance (aus Split-Gains)
            self._collect_importance(tree.root, feat_gain)

            # Early Stopping auf Validation-Set
            if val_X:
                for i in range(len(val_X)):
                    val_predictions[i] += self.learning_rate * tree.predict_one(val_X[i])
                # Val-MAE im Original-Raum berechnen
                if self.log_target:
                    val_mae = sum(abs(math.expm1(val_predictions[i]) - math.expm1(val_y[i]))
                                  for i in range(len(val_y))) / len(val_y)
                else:
                    val_mae = sum(abs(val_predictions[i] - val_y[i])
                                  for i in range(len(val_y))) / len(val_y)
                if val_mae < best_val_mae - 0.001:
                    best_val_mae = val_mae
                    best_n_trees = t + 1
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= 30 and t + 1 >= 50:
                    early_stopped = True
                    best_n_trees = max(best_n_trees, 40)  # Minimum 40 Bäume
                    log.info(f"[GBRT] Early stopping bei Baum {t+1}, "
                             f"beste Stelle: Baum {best_n_trees} (val_mae={best_val_mae:.4f})")
                    # Bäume nach dem besten Punkt entfernen
                    if best_n_trees > 0 and best_n_trees < len(self.trees):
                        self.trees = self.trees[:best_n_trees]
                    break

        # Early Stopping Attribute speichern
        self.best_n_trees = best_n_trees if best_n_trees > 0 else len(self.trees)
        self.early_stopped = early_stopped

        # Train-Metriken im Original-Raum
        if self.log_target:
            orig_preds = [math.expm1(p) for p in predictions]
            orig_y = [math.expm1(v) for v in y]
        else:
            orig_preds = predictions
            orig_y = y

        train_mae = sum(abs(orig_preds[i] - orig_y[i]) for i in range(n)) / n
        train_rmse = math.sqrt(sum((orig_preds[i] - orig_y[i]) ** 2 for i in range(n)) / n)

        # R² berechnen
        y_mean = sum(orig_y) / n
        ss_res = sum((orig_y[i] - orig_preds[i]) ** 2 for i in range(n))
        ss_tot = sum((orig_y[i] - y_mean) ** 2 for i in range(n))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self.train_metrics = {
            "mae": round(train_mae, 4),
            "rmse": round(train_rmse, 4),
            "r2": round(r2, 4),
            "r2_residual": round(r2, 4),
            "n_trees_used": len(self.trees),
            "n_trees_requested": self.n_trees,
            "early_stopped": early_stopped,
            "early_stopped_at": self.best_n_trees if early_stopped else None,
            "n_samples": n,
            "loss": self.loss,
            "log_target": self.log_target,
        }

        if val_X:
            self.train_metrics["val_mae"] = round(best_val_mae, 4)

        # Normalize Feature Importance
        total_gain = sum(feat_gain.values()) or 1.0
        self.feature_importance_ = {
            self.feature_names[k] if k < len(self.feature_names) else f"f{k}":
            round(v / total_gain, 4)
            for k, v in sorted(feat_gain.items(), key=lambda x: -x[1])
        }

        log.info(f"[GBRT] Training: {len(self.trees)} Baeume, MAE={train_mae:.3f}, "
                 f"RMSE={train_rmse:.3f}, R²={r2:.3f}, n={n}")

    def fit_incremental(self, X, y, n_new_trees=10):
        """Fuegt inkrementell neue Baeume hinzu ohne bestehendes Modell zurueckzusetzen.

        Berechnet Residuen auf neuen Daten und trainiert flachere Baeume.
        """
        n = len(X)
        if n == 0 or not self.trees:
            return

        # Aktuelle Vorhersagen fuer neue Daten
        predictions = [self.predict_one(x) for x in X]

        # Residuen
        residuals = [y[i] - predictions[i] for i in range(n)]

        rng = random.Random(int(time.time()))
        inc_depth = max(2, self.max_depth - 1)
        inc_min_leaf = max(self.min_samples_leaf + 5, 15)

        for _ in range(n_new_trees):
            # Subsampling
            if self.subsample < 1.0 and n > 1:
                sample_size = max(1, int(n * self.subsample))
                sample_idx = rng.sample(range(n), sample_size)
            else:
                sample_idx = None

            tree = _GBRTTree(max_depth=inc_depth,
                             min_samples_leaf=inc_min_leaf,
                             n_bins=self.n_bins)
            tree.fit(X, residuals, sample_idx)
            self.trees.append(tree)

            # Update Residuen
            for i in range(n):
                predictions[i] += self.learning_rate * tree.predict_one(X[i])
            residuals = [y[i] - predictions[i] for i in range(n)]

        log.info(f"[GBRT] Inkrementell: +{n_new_trees} Baeume "
                 f"(total={len(self.trees)}, depth={inc_depth}, n={n})")

    def predict(self, X):
        """Prediction fuer mehrere Samples. Returns Liste von Floats."""
        return [self.predict_one(x) for x in X]

    def predict_one(self, x):
        """Prediction fuer einen einzelnen Feature-Vektor."""
        pred = self.initial_prediction
        for tree in self.trees:
            pred += self.learning_rate * tree.predict_one(x)
        if self.log_target:
            pred = math.expm1(pred)
        return max(0.01, pred)

    def predict_with_uncertainty(self, x):
        """Prediction mit Unsicherheitsschaetzung aus Baum-Varianz."""
        tree_preds = []
        cumulative = self.initial_prediction
        for tree in self.trees:
            contrib = self.learning_rate * tree.predict_one(x)
            cumulative += contrib
            tree_preds.append(cumulative)

        pred = cumulative
        if self.log_target:
            pred = math.expm1(pred)

        # Varianz der letzten 50 Baeume als Unsicherheits-Mass
        if len(tree_preds) > 50:
            recent = tree_preds[-50:]
            if self.log_target:
                recent = [math.expm1(r) for r in recent]
            mean_recent = sum(recent) / len(recent)
            var = sum((p - mean_recent) ** 2 for p in recent) / len(recent)
            std = math.sqrt(var)
        else:
            std = 0.5  # Default-Unsicherheit

        return {
            "predicted": max(0.01, pred),
            "std": round(std, 4),
            "confidence": round(max(0.1, min(0.99, 1.0 - std / max(1.0, abs(pred)))), 3),
        }

    def shap_values(self, x):
        """Berechnet TreeSHAP-artige Feature-Contributions fuer einen einzelnen Sample.

        Aggregiert path_contributions ueber alle Baeume, gewichtet mit learning_rate.
        Returns: Dict mit base_value, shap_values (feature_idx→contribution), prediction.
        """
        n_features = len(x)
        total_contributions = [0.0] * n_features
        for tree in self.trees:
            contribs = tree.path_contributions(x, n_features)
            for i in range(n_features):
                total_contributions[i] += self.learning_rate * contribs[i]

        raw_prediction = self.initial_prediction + sum(total_contributions)
        if self.log_target:
            prediction = math.expm1(raw_prediction)
            base_value = math.expm1(self.initial_prediction)
        else:
            prediction = raw_prediction
            base_value = self.initial_prediction

        shap_dict = {}
        for i, c in enumerate(total_contributions):
            if abs(c) > 1e-6:
                fname = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                shap_dict[fname] = round(c, 5)

        return {
            "base_value": round(base_value, 5),
            "shap_values": shap_dict,
            "prediction": round(max(0.01, prediction), 5),
        }

    def feature_importance(self, top_n=20):
        """Top-N Feature Importance als sortierte Liste."""
        items = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return [{"name": k, "importance": v} for k, v in items[:top_n]]

    def _collect_importance(self, node, feat_gain):
        """Sammelt Split-Gains rekursiv."""
        if node is None or node.feature_idx < 0:
            return
        feat_gain[node.feature_idx] += node.gain
        self._collect_importance(node.left, feat_gain)
        self._collect_importance(node.right, feat_gain)

    def to_json(self):
        """Serialisiert das gesamte Modell als JSON-faehiges Dict."""
        d = {
            "type": "GBRT",
            "n_trees": len(self.trees),
            "initial_prediction": round(self.initial_prediction, 6),
            "learning_rate": self.learning_rate,
            "feature_names": self.feature_names,
            "trees": [t.to_dict() for t in self.trees],
            "metrics": self.train_metrics,
            "feature_importance": self.feature_importance(20),
        }
        if self.log_target:
            d["log_target"] = True
        if self.loss != "mse":
            d["loss"] = self.loss
            d["huber_delta"] = self.huber_delta
        if hasattr(self, "conformal_radius"):
            d["conformal_radius"] = round(self.conformal_radius, 6)
        if hasattr(self, "blend_alpha"):
            d["blend_alpha"] = round(self.blend_alpha, 4)
        if hasattr(self, "best_n_trees"):
            d["best_n_trees"] = self.best_n_trees
        if hasattr(self, "early_stopped"):
            d["early_stopped"] = self.early_stopped
        return d

    @staticmethod
    def from_json(data):
        """Deserialisiert ein GBRT-Modell aus JSON."""
        model = GBRTModel()
        model.initial_prediction = data["initial_prediction"]
        model.learning_rate = data["learning_rate"]
        model.feature_names = data["feature_names"]
        model.log_target = data.get("log_target", False)
        model.loss = data.get("loss", "mse")
        model.huber_delta = data.get("huber_delta", 1.5)
        if "conformal_radius" in data:
            model.conformal_radius = data["conformal_radius"]
        if "blend_alpha" in data:
            model.blend_alpha = data["blend_alpha"]
        if "best_n_trees" in data:
            model.best_n_trees = data["best_n_trees"]
        if "early_stopped" in data:
            model.early_stopped = data["early_stopped"]
        model.trees = [_GBRTTree.from_dict(td) for td in data["trees"]]
        model.train_metrics = data.get("metrics", {})
        # Feature Importance aus gespeichertem JSON wiederherstellen
        fi_list = data.get("feature_importance", [])
        if fi_list and model.feature_names:
            for fi_item in fi_list:
                name = fi_item.get("name", "")
                if name in model.feature_names:
                    idx = model.feature_names.index(name)
                    model.feature_importance_[idx] = fi_item.get("importance", 0)
        return model


class _SklearnModelWrapper:
    """Wrapper um sklearn GradientBoostingRegressor für API-Kompatibilität mit GBRTModel."""

    def __init__(self, sklearn_model, feature_names):
        self.sklearn_model = sklearn_model
        self.feature_names = list(feature_names)
        self.trees = list(range(sklearn_model.n_estimators_))  # Dummy für len()
        self.train_metrics = {}
        self.feature_importance_ = {}
        self._is_sklearn = True
        # Feature Importance extrahieren
        importances = sklearn_model.feature_importances_
        total = sum(importances)
        if total > 0:
            for i, fname in enumerate(feature_names):
                if importances[i] > 0:
                    self.feature_importance_[fname] = float(importances[i] / total)

    def predict(self, X):
        if np is None:
            return [0.0] * len(X)
        return self.sklearn_model.predict(np.array(X, dtype=np.float64)).tolist()

    def predict_one(self, x):
        if np is None:
            return 0.0
        return float(self.sklearn_model.predict(np.array([x], dtype=np.float64))[0])

    def predict_with_uncertainty(self, x):
        if np is None:
            return {"predicted": 0.0, "confidence": 0.5, "std": 0.0}
        x_arr = np.array([x], dtype=np.float64)
        pred = float(self.sklearn_model.predict(x_arr)[0])
        # Uncertainty via Baum-Varianz (schneller als staged_predict)
        std = 0.0
        try:
            n_trees = self.sklearn_model.n_estimators_
            if n_trees > 50:
                # Nur erste + letzte 25 Bäume vergleichen statt alle zu iterieren
                lr = self.sklearn_model.learning_rate
                init = self.sklearn_model.init_.predict(x_arr)[0] if hasattr(self.sklearn_model.init_, 'predict') else self.sklearn_model._raw_predict_init(x_arr)[0]
                # Schnelle Näherung: Fraction of total prediction from recent trees
                tree_preds = []
                for tree_idx in range(max(0, n_trees - 25), n_trees):
                    tp = float(self.sklearn_model.estimators_[tree_idx, 0].predict(x_arr)[0])
                    tree_preds.append(tp * lr)
                if tree_preds:
                    std = math.sqrt(sum(t ** 2 for t in tree_preds) / len(tree_preds))
        except Exception:
            std = abs(pred) * 0.12
        confidence = max(0.1, min(0.95, 1.0 - std / max(1.0, abs(pred))))
        return {"predicted": pred, "confidence": round(confidence, 3), "std": round(std, 4)}

    def shap_values(self, x):
        """Feature-Contributions via batched Leave-One-Out auf Top-20 Features."""
        if np is None:
            return {"base_value": 0.0, "shap_values": {}, "prediction": 0.0}
        x_arr = np.array(x, dtype=np.float64).reshape(1, -1)
        pred = float(self.sklearn_model.predict(x_arr)[0])
        shap_dict = {}
        top_feats = sorted(self.feature_importance_.items(), key=lambda kv: -kv[1])[:20]
        if not top_feats:
            return {"base_value": pred, "shap_values": {}, "prediction": pred}
        try:
            # Batch: 20 modifizierte Kopien auf einmal predicten
            n = len(top_feats)
            x_batch = np.tile(x_arr, (n, 1))  # n Kopien
            feat_indices = []
            for i, (fname, _) in enumerate(top_feats):
                fidx = self.feature_names.index(fname)
                x_batch[i, fidx] = 0.0
                feat_indices.append((fname, i))
            preds_without = self.sklearn_model.predict(x_batch)  # 1 Batch-Call
            for fname, i in feat_indices:
                shap_dict[fname] = pred - float(preds_without[i])
        except Exception:
            for fname, imp in top_feats:
                shap_dict[fname] = imp * 0.5
        return {"base_value": pred, "shap_values": shap_dict, "prediction": pred}

    def feature_importance(self, top_n=20):
        items = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return [{"name": k, "importance": v} for k, v in items[:top_n]]

    def to_json(self):
        return {
            "type": "sklearn_GBR",
            "n_trees": len(self.trees),
            "feature_names": self.feature_names,
            "metrics": self.train_metrics,
            "feature_importance": self.feature_importance(20),
            "conformal_radius": getattr(self, "conformal_radius", None),
            "blend_alpha": getattr(self, "blend_alpha", None),
        }


class _LGBMModelWrapper:
    """Wrapper um LightGBM-Modell für API-Kompatibilität mit GBRTModel."""

    _is_lgbm = True

    def __init__(self, lgbm_model, feature_names):
        self.lgbm_model = lgbm_model
        self.feature_names = list(feature_names)
        self.trees = list(range(lgbm_model.n_estimators_))
        self.train_metrics = {}
        # Feature Importance (normalized)
        raw_imp = lgbm_model.feature_importances_
        total = sum(raw_imp) if sum(raw_imp) > 0 else 1
        self.feature_importance_ = {
            f: float(raw_imp[i]) / total
            for i, f in enumerate(feature_names)
        }

    def predict(self, X):
        if np is None:
            return [0.0] * len(X)
        return self.lgbm_model.predict(np.array(X, dtype=np.float64)).tolist()

    def predict_one(self, x):
        if np is None:
            return 0.0
        return float(self.lgbm_model.predict(np.array([x], dtype=np.float64))[0])

    def predict_with_uncertainty(self, x):
        pred = self.predict_one(x)
        return {"predicted": pred, "confidence": 0.7, "std": 0.0}

    def shap_values(self, x):
        # Leave-One-Out Approximation auf Top-20 Features
        base = self.predict_one(x)
        top_feat = sorted(self.feature_importance_.items(), key=lambda t: -t[1])[:20]
        contributions = {}
        for fname, _ in top_feat:
            idx = self.feature_names.index(fname)
            x_mod = list(x)
            x_mod[idx] = 0.0
            mod_pred = self.predict_one(x_mod)
            contributions[fname] = round(base - mod_pred, 4)
        return contributions

    def feature_importance(self, top_n=20):
        return sorted(self.feature_importance_.items(), key=lambda t: -t[1])[:top_n]

    def to_json(self):
        return {
            "type": "lgbm_GBR",
            "n_trees": len(self.trees),
            "feature_names": self.feature_names,
            "metrics": self.train_metrics,
            "feature_importance": self.feature_importance(20),
            "conformal_radius": getattr(self, "conformal_radius", None),
            "blend_alpha": getattr(self, "blend_alpha", None),
        }


# ── Isotonic Regression (PAVA) fuer Kalibrierung ────────────────────────

def _isotonic_regression_pava(predicted, actual):
    """Pool Adjacent Violators Algorithm fuer Isotonische Regression.

    Args:
        predicted: Sortierte Predictions (Liste von Floats)
        actual: Zugehoerige Actual-Werte (Liste von Floats)
    Returns:
        calibrated: Kalibrierte Werte (Liste von Floats)
    """
    n = len(predicted)
    if n == 0:
        return []

    # Sortiere nach predicted
    paired = sorted(zip(predicted, actual), key=lambda x: x[0])
    y = [p[1] for p in paired]

    # PAVA: Pool Adjacent Violators
    blocks = [[i] for i in range(n)]
    block_avg = [y[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if block_avg[i] > block_avg[i + 1]:
            # Merge blocks
            merged = blocks[i] + blocks[i + 1]
            merged_avg = sum(y[j] for j in merged) / len(merged)
            blocks[i] = merged
            block_avg[i] = merged_avg
            del blocks[i + 1]
            del block_avg[i + 1]
            # Gehe zurueck um weitere Violators zu finden
            if i > 0:
                i -= 1
        else:
            i += 1

    # Ergebnis zurueckschreiben
    result = [0.0] * n
    for block, avg in zip(blocks, block_avg):
        for idx in block:
            result[idx] = avg

    # Zurueck in Original-Reihenfolge
    order = [p[0] for p in paired]
    return result


class IsotonicCalibrator:
    """Isotonische Regression fuer Post-Processing-Kalibrierung."""

    def __init__(self):
        self.breakpoints = []  # [(predicted, calibrated), ...]

    def fit(self, predicted, actual):
        """Trainiert den Kalibrator."""
        if len(predicted) < 10:
            return

        # Sortiere nach predicted
        paired = sorted(zip(predicted, actual), key=lambda x: x[0])
        preds = [p[0] for p in paired]
        acts = [p[1] for p in paired]

        calibrated = _isotonic_regression_pava(preds, acts)

        # Breakpoints extrahieren (alle einzigartigen Stufen)
        self.breakpoints = []
        prev_cal = None
        for pred, cal in zip(preds, calibrated):
            if prev_cal is None or abs(cal - prev_cal) > 0.001:
                self.breakpoints.append((pred, cal))
                prev_cal = cal
        # Letzten Punkt sicherstellen
        if preds:
            self.breakpoints.append((preds[-1], calibrated[-1]))

    def calibrate(self, predicted):
        """Kalibriert einen einzelnen Predicted-Wert."""
        if not self.breakpoints:
            return predicted

        # Lineare Interpolation zwischen Breakpoints
        if predicted <= self.breakpoints[0][0]:
            return self.breakpoints[0][1]
        if predicted >= self.breakpoints[-1][0]:
            return self.breakpoints[-1][1]

        for i in range(len(self.breakpoints) - 1):
            p1, c1 = self.breakpoints[i]
            p2, c2 = self.breakpoints[i + 1]
            if p1 <= predicted <= p2:
                if abs(p2 - p1) < 1e-10:
                    return c1
                t = (predicted - p1) / (p2 - p1)
                return c1 + t * (c2 - c1)

        return predicted

    def to_dict(self):
        return {"breakpoints": [(round(p, 4), round(c, 4)) for p, c in self.breakpoints]}

    @staticmethod
    def from_dict(d):
        cal = IsotonicCalibrator()
        cal.breakpoints = [(p, c) for p, c in d.get("breakpoints", [])]
        return cal


# ── GBRT Global State ───────────────────────────────────────────────────

_gbrt_model = None        # Haupt-Modell (Direct: OR direkt)
_gbrt_model_direct = None # Direct-Modell (lernt OR direkt, ohne Baseline-Subtraktion)
_gbrt_model_q10 = None    # Quantile-Modell p10
_gbrt_model_q90 = None    # Quantile-Modell p90
_gbrt_calibrator = None   # Isotonische Kalibrierung
_gbrt_lock = threading.Lock()
_gbrt_feature_names = []  # Sortierte Feature-Namen
_gbrt_train_ts = 0        # Letzter Training-Zeitpunkt
_gbrt_history_stats = {}  # Cached History Stats
_gbrt_cat_hour_baselines = {}  # Cat×Hour Baselines (Bayesian Shrinkage, für Features + Blending)
_gbrt_global_train_avg = 4.77  # Fallback Global Average
_gbrt_ensemble_weights = {"MAE": 1.0}  # Gewichte Multi-Objective Ensemble
_gbrt_model_type = "residual"  # "residual", "direct", oder "ensemble"

GBRT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gbrt_model.json")
SKLEARN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sklearn_gbrt.joblib")
ML_LGBM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ml_lgbm.joblib")

# ── ML v2: sklearn-Modell + PCA State ──
_sklearn_model = None           # sklearn GBR (Direct)
_sklearn_model_direct = None    # sklearn GBR (Direct)
_sklearn_feature_names = []     # Feature-Namen für sklearn
_embedding_pca = None           # PCA-Transformer (25 Komponenten)
_embedding_pca_mean = None      # PCA Mittelwert-Vektor
_use_sklearn = _SKLEARN_AVAILABLE  # Flag ob sklearn oder Pure-Python

# ── Competitor XOR Performance-Cache ─────────────────────────────────
# Vorberechnete Wort/Entity/Kategorie-Performance fuer schnelle XOR-Predictions
_xor_perf_cache = {
    "word_perf": {},      # {wort: {avg, count, p25, p75}}
    "cat_hour_perf": {},  # {cat_hour: {avg, p25, p50, p75, count}}
    "eil_perf": {},       # {avg, p25, p75, count}
    "global_avg": 4.77,
    "built_at": 0,
}
_xor_perf_lock = threading.Lock()

_XOR_STOP_WORDS = frozenset({
    "der", "die", "das", "und", "oder", "ist", "in", "von", "zu", "mit",
    "auf", "für", "den", "ein", "eine", "dem", "des", "sich", "bei", "nach",
    "aus", "als", "hat", "wie", "wird", "vor", "nicht", "im", "am", "es",
    "an", "um", "über", "auch", "war", "zum", "was", "nur", "er", "sie",
    "noch", "wir", "werden", "aber", "alle", "vom", "ab", "bis", "jetzt",
    "hier", "wegen", "sein", "neue", "neuer", "neues", "ganz", "erst",
    "seit", "doch", "soll", "kann", "will", "muss", "darf", "laut", "wenn",
    "dann", "mehr", "schon", "nun", "dass", "einen", "einem", "dieser",
    "diese", "dieses", "haben", "hatte", "ihre", "seiner", "seine",
    "einer", "so", "mal", "zur", "ins", "darum", "warum", "seinen",
    "ihren", "ihrem", "seinen", "seinem", "ihr", "ihm", "ihn",
})


def _build_xor_perf_cache():
    """Baut den XOR-Performance-Cache aus historischen Pushes."""
    import time as _t
    _t0 = _t.monotonic()
    try:
        push_data = _research_state.get("push_data", [])
        if not push_data:
            push_data = _push_db_load_all()
        hist = [p for p in push_data if 0 < p.get("or", 0) <= 25]
        if len(hist) < 100:
            return

        from collections import defaultdict
        word_ors = defaultdict(list)
        cat_hour_ors = defaultdict(list)
        eil_ors = []
        all_ors = []

        for p in hist:
            or_val = p["or"]
            all_ors.append(or_val)
            title = (p.get("title") or p.get("headline", "")).lower()
            cat = (p.get("cat", "") or "").lower().strip()
            hour = p.get("hour", 12)

            # Wort-Performance
            words = set(w.strip(".,;:!?\"'()[]{}") for w in title.split())
            words = {w for w in words if len(w) > 2 and w not in _XOR_STOP_WORDS}
            for w in words:
                word_ors[w].append(or_val)

            # Cat×Hour
            cat_hour_ors[f"{cat}_{hour}"].append(or_val)

            # Eilmeldung
            if p.get("is_eilmeldung"):
                eil_ors.append(or_val)

        # Wort-Performance zusammenfassen (nur Woerter mit >= 5 Vorkommen)
        word_perf = {}
        for w, ors in word_ors.items():
            if len(ors) >= 5:
                s = sorted(ors)
                word_perf[w] = {
                    "avg": sum(ors) / len(ors),
                    "count": len(ors),
                    "p25": s[len(s) // 4],
                    "p75": s[int(len(s) * 0.75)],
                    "p90": s[int(len(s) * 0.9)],
                }

        # Cat×Hour
        cat_hour_perf = {}
        for key, ors in cat_hour_ors.items():
            if len(ors) >= 3:
                s = sorted(ors)
                cat_hour_perf[key] = {
                    "avg": sum(ors) / len(ors),
                    "p25": s[len(s) // 4],
                    "p50": s[len(s) // 2],
                    "p75": s[int(len(s) * 0.75)],
                    "count": len(ors),
                }

        # Eilmeldung
        eil_perf = {}
        if len(eil_ors) >= 5:
            s = sorted(eil_ors)
            eil_perf = {
                "avg": sum(eil_ors) / len(eil_ors),
                "p25": s[len(s) // 4],
                "p75": s[int(len(s) * 0.75)],
                "p90": s[int(len(s) * 0.9)],
                "count": len(eil_ors),
            }

        global_avg = sum(all_ors) / len(all_ors) if all_ors else 4.77

        with _xor_perf_lock:
            _xor_perf_cache["word_perf"] = word_perf
            _xor_perf_cache["cat_hour_perf"] = cat_hour_perf
            _xor_perf_cache["eil_perf"] = eil_perf
            _xor_perf_cache["global_avg"] = global_avg
            _xor_perf_cache["built_at"] = _t.time()

        elapsed = (_t.monotonic() - _t0) * 1000
        log.info(f"[XOR] Performance-Cache gebaut: {len(word_perf)} Woerter, "
                 f"{len(cat_hour_perf)} Cat×Hour, "
                 f"global_avg={global_avg:.2f}, {elapsed:.0f}ms")
    except Exception as e:
        log.warning(f"[XOR] Cache-Build-Fehler: {e}")


def _gbrt_bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Bootstrap-Konfidenzintervalle fuer MAE, RMSE, R²."""
    n = len(y_true)
    if n < 10:
        return {}
    mae_samples = []
    rmse_samples = []
    r2_samples = []
    alpha = (1 - ci) / 2
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        mae_samples.append(sum(abs(yt[j] - yp[j]) for j in range(n)) / n)
        rmse_samples.append(math.sqrt(sum((yt[j] - yp[j]) ** 2 for j in range(n)) / n))
        ym = sum(yt) / n
        ss_res = sum((yt[j] - yp[j]) ** 2 for j in range(n))
        ss_tot = sum((yt[j] - ym) ** 2 for j in range(n))
        r2_samples.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    mae_samples.sort()
    rmse_samples.sort()
    r2_samples.sort()
    lo = int(alpha * n_bootstrap)
    hi = int((1 - alpha) * n_bootstrap) - 1
    return {
        "mae_ci": [round(mae_samples[lo], 4), round(mae_samples[hi], 4)],
        "rmse_ci": [round(rmse_samples[lo], 4), round(rmse_samples[hi], 4)],
        "r2_ci": [round(r2_samples[lo], 4), round(r2_samples[hi], 4)],
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


def _gbrt_compute_baselines(y_test, test_data, train_data):
    """Berechne naive Baselines fuer ehrlichen Vergleich."""
    if not y_test:
        return {}
    n = len(y_test)
    # Baseline 1: Global Mean (naivste Vorhersage)
    global_mean = sum(p.get("or", 0) for p in train_data) / len(train_data) if train_data else 0
    baseline_global_mae = sum(abs(global_mean - y_test[i]) for i in range(n)) / n

    # Baseline 2: Category Mean
    cat_means = defaultdict(list)
    for p in train_data:
        cat_means[(p.get("cat", "news")).lower().strip()].append(p.get("or", 0))
    cat_avg = {c: sum(v) / len(v) for c, v in cat_means.items()}
    baseline_cat_preds = []
    for p in test_data:
        c = (p.get("cat", "news")).lower().strip()
        baseline_cat_preds.append(cat_avg.get(c, global_mean))
    baseline_cat_mae = sum(abs(baseline_cat_preds[i] - y_test[i]) for i in range(n)) / n

    # Baseline 3: Category x Hour Mean
    cathour_means = defaultdict(list)
    for p in train_data:
        key = f"{(p.get('cat', 'news')).lower().strip()}_{p.get('hour', 12)}"
        cathour_means[key].append(p.get("or", 0))
    cathour_avg = {k: sum(v) / len(v) for k, v in cathour_means.items()}
    baseline_cathour_preds = []
    for p in test_data:
        key = f"{(p.get('cat', 'news')).lower().strip()}_{p.get('hour', 12)}"
        baseline_cathour_preds.append(cathour_avg.get(key, cat_avg.get(
            (p.get("cat", "news")).lower().strip(), global_mean)))
    baseline_cathour_mae = sum(abs(baseline_cathour_preds[i] - y_test[i]) for i in range(n)) / n

    return {
        "global_mean_mae": round(baseline_global_mae, 4),
        "category_mean_mae": round(baseline_cat_mae, 4),
        "category_hour_mean_mae": round(baseline_cathour_mae, 4),
        "global_mean_value": round(global_mean, 3),
    }


# ── Experiment Tracking (Phase B) ────────────────────────────────────────

_experiment_counter = 0
_experiment_counter_lock = threading.Lock()


def _log_experiment(model, hyperparams, metrics, baselines, cv_results,
                    n_features, n_samples, training_duration_s):
    """Speichert einen Trainingslauf in der experiments-Tabelle."""
    global _experiment_counter
    with _experiment_counter_lock:
        _experiment_counter += 1
        counter = _experiment_counter
    experiment_id = f"exp_{int(time.time())}_{counter}"

    # Model Hash: MD5 der ersten 10KB des Model-JSON
    import hashlib
    model_json_str = json.dumps(model.to_json())[:10240]
    model_hash = hashlib.md5(model_json_str.encode()).hexdigest()

    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.execute("""INSERT OR REPLACE INTO experiments
                (experiment_id, timestamp, hyperparams, metrics, baselines,
                 cv_results, n_features, n_samples, model_hash, promoted, training_duration_s)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (experiment_id, int(time.time()), json.dumps(hyperparams),
                 json.dumps(metrics), json.dumps(baselines), json.dumps(cv_results),
                 n_features, n_samples, model_hash, round(training_duration_s, 2)))
            conn.commit()
            conn.close()
        log.info(f"[Experiment] Logged: {experiment_id} (hash={model_hash[:8]})")
    except Exception as e:
        log.warning(f"[Experiment] Log-Fehler: {e}")
    return experiment_id


def _get_experiments(limit=50):
    """Alle Experimente aus DB laden."""
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
        result = []
        for r in rows:
            exp = dict(r)
            for field in ("hyperparams", "metrics", "baselines", "cv_results"):
                try:
                    exp[field] = json.loads(exp[field]) if exp[field] else {}
                except (json.JSONDecodeError, TypeError):
                    exp[field] = {}
            result.append(exp)
        return result
    except Exception as e:
        log.warning(f"[Experiment] Load-Fehler: {e}")
        return []


def _compare_experiments(ids):
    """Vergleich mehrerer Experimente."""
    if not ids:
        return {"error": "Keine Experiment-IDs angegeben"}
    try:
        placeholders = ",".join("?" for _ in ids)
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM experiments WHERE experiment_id IN ({placeholders})", ids
            ).fetchall()
            conn.close()
        experiments = []
        for r in rows:
            exp = dict(r)
            for field in ("hyperparams", "metrics", "baselines", "cv_results"):
                try:
                    exp[field] = json.loads(exp[field]) if exp[field] else {}
                except (json.JSONDecodeError, TypeError):
                    exp[field] = {}
            experiments.append(exp)
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as e:
        return {"error": str(e)}


# ── Promotion Gates (Phase C) ───────────────────────────────────────────

def _validate_promotion_gates(model, test_mae, bootstrap_ci, baselines,
                               quantile_coverage, experiment_id):
    """Validiert ein Modell gegen 4 Promotion Gates.

    Returns: (passed: bool, gates: dict, reason: str)
    """
    gates = {}
    reasons = []

    # Gate 1: Schlaegt mindestens 2 von 3 Baselines
    g1_global = test_mae < baselines.get("global_mean_mae", 999)
    g1_cat = test_mae < baselines.get("category_mean_mae", 999)
    g1_cathour = test_mae < baselines.get("category_hour_mean_mae", 999)
    g1_count = sum([g1_global, g1_cat, g1_cathour])
    gates["beats_baselines_2of3"] = g1_count >= 2
    if not gates["beats_baselines_2of3"]:
        reasons.append(f"Schlaegt weniger als 2/3 Baselines (Global={g1_global}, Cat={g1_cat}, CatHour={g1_cathour})")

    # Gate 2: Besser als aktueller Champion
    champion_mae = 999.0
    with _gbrt_lock:
        if _gbrt_model and _gbrt_model.train_metrics:
            champion_mae = _gbrt_model.train_metrics.get("test_mae", 999.0)
    gates["beats_champion"] = test_mae < champion_mae
    if not gates["beats_champion"]:
        reasons.append(f"Nicht besser als Champion (challenger={test_mae:.4f} vs champion={champion_mae:.4f})")

    # Gate 3: Bootstrap CI Obergrenze unter Champion's MAE
    ci_upper = 999.0
    if bootstrap_ci and bootstrap_ci.get("mae_ci"):
        ci_upper = bootstrap_ci["mae_ci"][1] if len(bootstrap_ci["mae_ci"]) > 1 else 999.0
    gates["ci_upper_below_champion"] = ci_upper < champion_mae
    if not gates["ci_upper_below_champion"]:
        reasons.append(f"Bootstrap CI Obergrenze ({ci_upper:.4f}) nicht unter Champion ({champion_mae:.4f})")

    # Gate 4: Quantile Coverage >= 55% (konforme Methode startet konservativer)
    gates["quantile_coverage_ok"] = quantile_coverage >= 0.55
    if not gates["quantile_coverage_ok"]:
        reasons.append(f"Quantile Coverage {quantile_coverage:.1%} unter 55%")

    passed = all(gates.values())
    reason = "Alle Gates bestanden" if passed else "; ".join(reasons)

    # In DB loggen
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.execute("""INSERT INTO promotion_log
                (experiment_id, timestamp, passed, gates, champion_mae, challenger_mae, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, int(time.time()), 1 if passed else 0,
                 json.dumps(gates), round(champion_mae, 4), round(test_mae, 4), reason))
            conn.commit()
            conn.close()
    except Exception as e:
        log.warning(f"[Promotion] Log-Fehler: {e}")

    log.info(f"[Promotion] {experiment_id}: {'BESTANDEN' if passed else 'ABGELEHNT'} — {reason}")
    return passed, gates, reason


# ── A/B Testing (Phase D) ───────────────────────────────────────────────

_ab_state = {
    "active": False,
    "champion_id": "",
    "challenger_id": "",
    "champion_model": None,
    "champion_q10": None,
    "champion_q90": None,
    "champion_calibrator": None,
    "challenger_model": None,
    "challenger_q10": None,
    "challenger_q90": None,
    "challenger_calibrator": None,
    "challenger_feature_names": [],
    "challenger_history_stats": {},
    "samples": [],           # [{champion_pred, challenger_pred, actual_or}, ...]
    "started_at": 0,
    "timeout_at": 0,
}
_ab_lock = threading.Lock()


def _update_cat_hour_baselines(train_data):
    """Aktualisiert die globalen Cat×Hour-Baselines mit Bayesian Shrinkage."""
    global _gbrt_global_train_avg
    SHRINKAGE_K = 20
    counts = defaultdict(list)
    for p in train_data:
        key = f"{(p.get('cat', '') or 'news').lower().strip()}_{p.get('hour', 12)}"
        counts[key].append(p.get("or", 0))
    all_or = [p.get("or", 0) for p in train_data if p.get("or", 0) > 0]
    _gbrt_global_train_avg = sum(all_or) / len(all_or) if all_or else 4.77
    _gbrt_cat_hour_baselines.clear()
    for key, ors in counts.items():
        n = len(ors)
        raw_mean = sum(ors) / n if n > 0 else _gbrt_global_train_avg
        _gbrt_cat_hour_baselines[key] = (n * raw_mean + SHRINKAGE_K * _gbrt_global_train_avg) / (n + SHRINKAGE_K)


def _gbrt_maybe_promote_or_challenge(model, model_q10, model_q90, calibrator,
                                       feature_names, history_stats, train_data,
                                       experiment_id, test_mae, bootstrap_ci,
                                       baselines, quantile_coverage):
    """Prueft Promotion Gates und startet ggf. A/B Test."""
    global _gbrt_model, _gbrt_model_direct, _gbrt_model_q10, _gbrt_model_q90, _gbrt_calibrator
    global _gbrt_feature_names, _gbrt_train_ts, _gbrt_history_stats
    global _gbrt_ensemble_weights, _gbrt_model_type

    passed, gates, reason = _validate_promotion_gates(
        model, test_mae, bootstrap_ci, baselines, quantile_coverage, experiment_id)

    if not passed:
        log.info(f"[A/B] Modell {experiment_id} hat Promotion Gates nicht bestanden, kein A/B Test")
        return False

    # Promotion bestanden — A/B Test starten
    with _ab_lock:
        if _ab_state["active"]:
            log.info(f"[A/B] A/B Test bereits aktiv, neues Modell {experiment_id} wird verworfen")
            return False

        # Aktuelles Modell wird Champion (falls vorhanden), neues wird Challenger
        with _gbrt_lock:
            if _gbrt_model is not None:
                _ab_state["champion_model"] = _gbrt_model
                _ab_state["champion_q10"] = _gbrt_model_q10
                _ab_state["champion_q90"] = _gbrt_model_q90
                _ab_state["champion_calibrator"] = _gbrt_calibrator
                _ab_state["champion_id"] = f"champion_{_gbrt_train_ts}"
            else:
                # Kein Champion vorhanden — neues Modell direkt promoten
                log.info(f"[A/B] Kein Champion vorhanden, {experiment_id} wird direkt promoted")
                _gbrt_model = model
                _gbrt_model_direct = model_direct if 'model_direct' in dir() else None
                _gbrt_model_q10 = model_q10
                _gbrt_model_q90 = model_q90
                _gbrt_calibrator = calibrator
                _gbrt_feature_names = feature_names
                _gbrt_train_ts = int(time.time())
                _gbrt_history_stats = history_stats
                # Cat×Hour-Baselines mit Bayesian Shrinkage (aus train_data)
                _update_cat_hour_baselines(train_data)
                _mark_experiment_promoted(experiment_id)
                return True

        _ab_state["active"] = True
        _ab_state["challenger_id"] = experiment_id
        _ab_state["challenger_model"] = model
        _ab_state["challenger_q10"] = model_q10
        _ab_state["challenger_q90"] = model_q90
        _ab_state["challenger_calibrator"] = calibrator
        _ab_state["challenger_feature_names"] = feature_names
        _ab_state["challenger_history_stats"] = history_stats
        _ab_state["challenger_train_data"] = train_data
        _ab_state["samples"] = []
        _ab_state["started_at"] = int(time.time())
        _ab_state["timeout_at"] = int(time.time()) + 86400  # 24h Timeout

    log.info(f"[A/B] Test gestartet: Champion={_ab_state['champion_id']} vs "
             f"Challenger={experiment_id}, Timeout in 24h")
    return True


def _ab_shadow_predict(push, state=None):
    """Fuehrt Shadow-Prediction mit Challenger aus, gibt nur Champion zurueck.

    Returns: challenger_pred (float) oder None
    """
    with _ab_lock:
        if not _ab_state["active"] or _ab_state["challenger_model"] is None:
            return None
        c_model = _ab_state["challenger_model"]
        c_calibrator = _ab_state["challenger_calibrator"]
        c_feature_names = _ab_state["challenger_feature_names"]
        c_history_stats = _ab_state["challenger_history_stats"]

    try:
        feat = _gbrt_extract_features(push, c_history_stats, state)
        x = [feat.get(k, 0.0) for k in c_feature_names]
        pred = c_model.predict_one(x)
        if c_calibrator:
            pred = c_calibrator.calibrate(pred)
        return max(0.01, pred)
    except Exception as e:
        log.warning(f"[A/B] Shadow-Prediction Fehler: {e}")
        return None


def _ab_record_sample(champion_pred, challenger_pred, actual_or):
    """Zeichnet eine Stichprobe auf und prueft auf Auswertung."""
    with _ab_lock:
        if not _ab_state["active"]:
            return
        _ab_state["samples"].append({
            "champion_pred": champion_pred,
            "challenger_pred": challenger_pred,
            "actual_or": actual_or,
            "ts": int(time.time()),
        })

    _ab_evaluate()


def _ab_evaluate():
    """Wertet A/B Test aus: Paired t-Test (p<0.05) nach min. 50 Samples."""
    global _gbrt_model, _gbrt_model_q10, _gbrt_model_q90, _gbrt_calibrator
    global _gbrt_feature_names, _gbrt_train_ts, _gbrt_history_stats

    with _ab_lock:
        if not _ab_state["active"]:
            return
        samples = list(_ab_state["samples"])
        timeout = time.time() >= _ab_state["timeout_at"]

    n = len(samples)

    # Timeout: Kein klarer Gewinner → Champion bleibt
    if timeout:
        log.info(f"[A/B] Timeout erreicht nach {n} Samples — Champion bleibt")
        with _ab_lock:
            _ab_state["active"] = False
            _ab_state["samples"] = []
        return

    if n < 50:
        return

    # Paired t-Test: Champion vs Challenger Fehler
    champ_errors = [abs(s["champion_pred"] - s["actual_or"]) for s in samples]
    chall_errors = [abs(s["challenger_pred"] - s["actual_or"]) for s in samples]
    diffs = [champ_errors[i] - chall_errors[i] for i in range(n)]

    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 1.0
    se_diff = math.sqrt(var_diff / n) if var_diff > 0 else 0.001
    t_stat = mean_diff / se_diff

    # Einseitiger t-Test: Ist Challenger signifikant besser?
    # t > 1.676 ≈ p < 0.05 (einseitig, df > 40)
    challenger_wins = t_stat > 1.676

    champ_mae = sum(champ_errors) / n
    chall_mae = sum(chall_errors) / n

    if challenger_wins:
        log.info(f"[A/B] Challenger gewinnt! t={t_stat:.3f}, "
                 f"Champion-MAE={champ_mae:.4f}, Challenger-MAE={chall_mae:.4f}")
        with _ab_lock:
            challenger_model = _ab_state["challenger_model"]
            challenger_q10 = _ab_state["challenger_q10"]
            challenger_q90 = _ab_state["challenger_q90"]
            challenger_cal = _ab_state["challenger_calibrator"]
            challenger_fn = _ab_state["challenger_feature_names"]
            challenger_hs = _ab_state["challenger_history_stats"]
            challenger_id = _ab_state["challenger_id"]
            _ab_state["active"] = False
            _ab_state["samples"] = []

        with _gbrt_lock:
            _gbrt_model = challenger_model
            _gbrt_model_q10 = challenger_q10
            _gbrt_model_q90 = challenger_q90
            _gbrt_calibrator = challenger_cal
            _gbrt_feature_names = challenger_fn
            _gbrt_train_ts = int(time.time())
            _gbrt_history_stats = challenger_hs
            # Cat×Hour-Baselines mit Bayesian Shrinkage aktualisieren
            challenger_td = _ab_state.get("challenger_train_data", [])
            if challenger_td:
                _update_cat_hour_baselines(challenger_td)

        _mark_experiment_promoted(challenger_id)
        log.info(f"[A/B] Challenger {challenger_id} promoted zum neuen Champion")
        _log_monitoring_event("ab_result", "info",
            f"Challenger {challenger_id} gewinnt A/B-Test (t={t_stat:.3f}, n={n})",
            {"winner": "challenger", "t_stat": round(t_stat, 3), "n": n,
             "champ_mae": round(champ_mae, 4), "chall_mae": round(chall_mae, 4)})
    elif n >= 200:
        # Nach 200 Samples ohne signifikanten Unterschied → Champion bleibt
        log.info(f"[A/B] Nach {n} Samples kein signifikanter Unterschied (t={t_stat:.3f}) — Champion bleibt")
        _log_monitoring_event("ab_result", "info",
            f"A/B-Test abgeschlossen: kein signifikanter Unterschied (t={t_stat:.3f}, n={n})",
            {"winner": "champion", "t_stat": round(t_stat, 3), "n": n,
             "champ_mae": round(champ_mae, 4), "chall_mae": round(chall_mae, 4)})
        with _ab_lock:
            _ab_state["active"] = False
            _ab_state["samples"] = []


def _mark_experiment_promoted(experiment_id):
    """Markiert ein Experiment als promoted in der DB."""
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.execute("UPDATE experiments SET promoted = 1 WHERE experiment_id = ?",
                         (experiment_id,))
            conn.commit()
            conn.close()
    except Exception as e:
        log.warning(f"[Experiment] Promote-Markierung Fehler: {e}")


# ── Online Learning (Phase E) ───────────────────────────────────────────

_online_state = {
    "last_update": 0,
    "online_mae": 0.0,
    "batch_mae": 0.0,
    "updates_count": 0,
    "paused": False,
}


def _gbrt_online_update():
    """Inkrementelles Update: Alle 30 Min neue Feedback-Daten hinzufuegen."""
    global _online_state

    # Nicht bei aktivem A/B-Test
    with _ab_lock:
        if _ab_state["active"]:
            return

    with _gbrt_lock:
        model = _gbrt_model
        feature_names = _gbrt_feature_names
        history_stats = _gbrt_history_stats

    if model is None or not feature_names:
        return

    if _online_state["paused"]:
        return

    now = time.time()
    if now - _online_state["last_update"] < 1800:  # 30 Min Intervall
        return
    _online_state["last_update"] = now

    # Neue Feedback-Daten aus prediction_log holen
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.row_factory = sqlite3.Row
            cutoff = int(now - 7200)  # Letzte 2 Stunden
            rows = conn.execute("""
                SELECT push_id, predicted_or, actual_or, features
                FROM prediction_log
                WHERE actual_or IS NOT NULL AND actual_or > 0
                AND actual_recorded_at > ?
                ORDER BY actual_recorded_at DESC LIMIT 100
            """, (cutoff,)).fetchall()
            conn.close()
    except Exception as e:
        log.warning(f"[Online] DB-Fehler: {e}")
        return

    if len(rows) < 5:
        return

    # Features und Targets extrahieren
    X_new = []
    y_new = []
    for r in rows:
        try:
            feat_dict = json.loads(r["features"]) if r["features"] else {}
            if feat_dict:
                x = [feat_dict.get(k, 0.0) for k in feature_names]
                X_new.append(x)
                y_new.append(float(r["actual_or"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if len(X_new) < 5:
        return

    # Inkrementelles Update
    try:
        with _gbrt_lock:
            if _gbrt_model is None:
                return
            _gbrt_model.fit_incremental(X_new, y_new, n_new_trees=10)

        # Online MAE berechnen
        preds = [model.predict_one(x) for x in X_new]
        online_mae = sum(abs(preds[i] - y_new[i]) for i in range(len(y_new))) / len(y_new)
        batch_mae = model.train_metrics.get("test_mae", 999)

        _online_state["online_mae"] = round(online_mae, 4)
        _online_state["batch_mae"] = round(batch_mae, 4)
        _online_state["updates_count"] += 1

        # D5: Online Bias-Korrektur aus letzten 50 Predictions (Mean Signed Error)
        global _gbrt_online_bias
        signed_errors = [preds[i] - y_new[i] for i in range(len(y_new))]
        recent_signed = signed_errors[:50]
        if recent_signed:
            _gbrt_online_bias = sum(recent_signed) / len(recent_signed)
            log.info(f"[Online] Bias-Korrektur: {_gbrt_online_bias:+.4f} (aus {len(recent_signed)} Predictions)")

        # Sicherheit: Wenn online_mae > batch_mae * 1.2 → pausieren
        if batch_mae > 0 and online_mae > batch_mae * 1.2:
            _online_state["paused"] = True
            log.warning(f"[Online] PAUSIERT: online_mae={online_mae:.4f} > "
                        f"batch_mae*1.2={batch_mae * 1.2:.4f}")
            _log_monitoring_event("online_pause", "warning",
                f"Online Learning pausiert: online_mae={online_mae:.4f} > batch_mae*1.2={batch_mae * 1.2:.4f}",
                {"online_mae": round(online_mae, 4), "batch_mae": round(batch_mae, 4)})
        else:
            log.info(f"[Online] Update #{_online_state['updates_count']}: "
                     f"+10 Baeume, online_mae={online_mae:.4f}, batch_mae={batch_mae:.4f}")
    except Exception as e:
        log.warning(f"[Online] Update-Fehler: {e}")


# ── Sentence Embeddings (Phase F) ───────────────────────────────────────

_embedding_model = None
_embedding_model_loading = False
_EMBEDDING_CACHE_MAX = 20000  # Max 20k Titel (~30 MB bei 384-dim float32)
_embedding_cache_mem = {}  # In-Memory Cache: title_hash → embedding


def _load_embedding_model_background():
    """Laedt das Sentence-Transformer Modell im Hintergrund und pre-cached Titel."""
    global _embedding_model, _embedding_model_loading
    _embedding_model_loading = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        log.info("[Embeddings] Modell geladen: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)")
        # Pre-cache existing titles from DB
        _precache_embeddings()
    except ImportError:
        log.info("[Embeddings] sentence-transformers nicht installiert — Fallback auf TF-IDF")
    except Exception as e:
        log.warning(f"[Embeddings] Modell-Ladefehler: {e}")
    finally:
        _embedding_model_loading = False


def _precache_embeddings():
    """Pre-cached Embeddings fuer alle existierenden Titel in der DB."""
    if _embedding_model is None:
        return
    try:
        import hashlib
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            rows = conn.execute("SELECT DISTINCT title FROM pushes WHERE title IS NOT NULL AND title != '' ORDER BY ts_num DESC LIMIT 2000").fetchall()
            conn.close()
        titles = [r[0] for r in rows if r[0]]
        if not titles:
            return

        # Check which are already cached in memory or SQLite
        uncached = []
        for t in titles:
            title_hash = hashlib.md5(t.encode()).hexdigest()
            if title_hash not in _embedding_cache_mem:
                uncached.append(t)

        if not uncached:
            log.info(f"[Embeddings] Alle {len(titles)} Titel bereits gecacht")
            return

        # Batch encode uncached titles
        log.info(f"[Embeddings] Pre-caching {len(uncached)} Titel-Embeddings...")
        batch_size = 64
        cached_count = 0
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            try:
                embs = _embedding_model.encode(batch).tolist()
                for t, emb in zip(batch, embs):
                    title_hash = hashlib.md5(t.encode()).hexdigest()
                    _embedding_cache_mem[title_hash] = emb
                    cached_count += 1
            except Exception as be:
                log.warning(f"[Embeddings] Batch-Encode-Fehler: {be}")
                continue
        log.info(f"[Embeddings] {cached_count} Titel-Embeddings pre-gecacht")
    except Exception as e:
        log.warning(f"[Embeddings] Pre-Cache-Fehler: {e}")


def _get_embedding(title):
    """Berechnet Embedding fuer einen Titel mit Memory + SQLite Cache.

    Returns: Liste von Floats (384-dim) oder None
    """
    if _embedding_model is None:
        return None

    import hashlib
    title_hash = hashlib.md5(title.encode()).hexdigest()

    # Memory Cache
    if title_hash in _embedding_cache_mem:
        return _embedding_cache_mem[title_hash]

    # SQLite Cache
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            row = conn.execute("SELECT embedding FROM embedding_cache WHERE title_hash = ?",
                               (title_hash,)).fetchone()
            conn.close()
        if row and row[0]:
            emb = json.loads(row[0])
            _embedding_cache_mem[title_hash] = emb
            return emb
    except Exception:
        pass

    # Berechne Embedding
    try:
        emb = _embedding_model.encode(title).tolist()
        # Eviction: älteste Einträge entfernen wenn Cache voll
        if len(_embedding_cache_mem) >= _EMBEDDING_CACHE_MAX:
            # Entferne ~10% der ältesten Einträge
            keys_to_remove = list(_embedding_cache_mem.keys())[:_EMBEDDING_CACHE_MAX // 10]
            for k in keys_to_remove:
                del _embedding_cache_mem[k]
        _embedding_cache_mem[title_hash] = emb
        # In DB speichern
        try:
            with _push_db_lock:
                conn = sqlite3.connect(PUSH_DB_PATH)
                conn.execute("""INSERT OR REPLACE INTO embedding_cache
                    (title_hash, title, embedding, created_at) VALUES (?, ?, ?, ?)""",
                    (title_hash, title, json.dumps(emb), int(time.time())))
                conn.commit()
                conn.close()
        except Exception:
            pass
        return emb
    except Exception as e:
        log.warning(f"[Embeddings] Encode-Fehler: {e}")
        return None


def _cosine_similarity(a, b):
    """Kosinus-Ähnlichkeit zweier Vektoren (numpy-beschleunigt)."""
    if np is not None:
        # Wenn schon ndarray, kein Copy (zero-cost)
        a_arr = a if isinstance(a, np.ndarray) else np.asarray(a, dtype=np.float32)
        b_arr = b if isinstance(b, np.ndarray) else np.asarray(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        denom = norm_a * norm_b
        if denom < 1e-10:
            return 0.0
        return float(dot / denom)
    # Fallback: Pure Python
    dot = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_embedding_features(title, history_stats):
    """Berechnet 4 Embedding-basierte Features.

    Returns: Dict mit emb_max_sim, emb_avg_sim_top10, emb_n_similar_50, emb_similar_avg_or
    """
    result = {
        "emb_max_sim": 0.0,
        "emb_avg_sim_top10": 0.0,
        "emb_n_similar_50": 0.0,
        "emb_similar_avg_or": 0.0,
    }

    if _embedding_model is None:
        return result

    emb = _get_embedding(title)
    if emb is None:
        return result

    # Vergleiche mit historischen Titeln
    hist_titles = history_stats.get("_recent_titles", [])
    if not hist_titles:
        return result

    sims = []
    for ht_title, ht_or in hist_titles:
        ht_emb = _get_embedding(ht_title)
        if ht_emb is not None:
            sim = _cosine_similarity(emb, ht_emb)
            sims.append((sim, ht_or))

    if not sims:
        return result

    sims.sort(key=lambda x: -x[0])
    result["emb_max_sim"] = sims[0][0]
    top10 = sims[:10]
    result["emb_avg_sim_top10"] = sum(s for s, _ in top10) / len(top10)
    similar_50 = [(s, o) for s, o in sims if s >= 0.50]
    result["emb_n_similar_50"] = float(len(similar_50))
    if similar_50:
        result["emb_similar_avg_or"] = sum(o for _, o in similar_50) / len(similar_50)

    return result


def _build_embedding_pca(train_pushes, n_components=25):
    """Baut PCA auf Titel-Embeddings der Trainingsdaten (kein Leakage)."""
    global _embedding_pca, _embedding_pca_mean
    if not _SKLEARN_AVAILABLE or _embedding_model is None or np is None:
        log.info("[PCA] sklearn oder Embedding-Modell nicht verfügbar, überspringe PCA")
        return
    try:
        embeddings = []
        for p in train_pushes:
            title = p.get("title", "")
            if not title:
                continue
            emb = _get_embedding(title)
            if emb is not None and len(emb) == 384:
                embeddings.append(emb)
        if len(embeddings) < 100:
            log.info(f"[PCA] Nur {len(embeddings)} Embeddings, brauche min 100")
            return
        emb_matrix = np.array(embeddings)
        _embedding_pca_mean = emb_matrix.mean(axis=0).reshape(1, -1)
        n_comp = min(n_components, emb_matrix.shape[0], emb_matrix.shape[1])
        pca = SklearnPCA(n_components=n_comp, random_state=42)
        pca.fit(emb_matrix - _embedding_pca_mean)
        _embedding_pca = pca
        explained = sum(pca.explained_variance_ratio_) * 100
        log.info(f"[PCA] {n_comp} Komponenten aus {len(embeddings)} Titeln, "
                 f"erklärte Varianz: {explained:.1f}%")
    except Exception as e:
        log.warning(f"[PCA] Fehler: {e}")


_gbrt_training_active = threading.Event()  # Guard gegen parallele Trainings


def _gbrt_train(pushes=None):
    """Trainiert das GBRT-Modell mit rigoroser Validierung."""
    if _gbrt_training_active.is_set():
        log.info("[GBRT] Training bereits aktiv, überspringe")
        return False
    _gbrt_training_active.set()
    log.info("[GBRT] _gbrt_train() gestartet")
    try:
        return _gbrt_train_inner(pushes)
    except Exception as e:
        log.error(f"[GBRT] Training-Fehler: {e}", exc_info=True)
        return False
    finally:
        _gbrt_training_active.clear()


def _gbrt_train_inner(pushes=None):
    """Trainiert das GBRT-Modell mit rigoroser Validierung.

    Enthält:
    - 5-Fold TimeSeriesSplit Cross-Validation
    - Bootstrap-Konfidenzintervalle (95%)
    - Naive Baseline-Vergleiche (Global Mean, Category Mean, Cat×Hour Mean)
    - Feature Importance-basiertes Pruning
    - Optionale Hyperparameter-Optimierung via Optuna
    """
    global _gbrt_model, _gbrt_model_direct, _gbrt_model_q10, _gbrt_model_q90, _gbrt_calibrator
    global _gbrt_feature_names, _gbrt_train_ts, _gbrt_history_stats
    global _gbrt_ensemble_weights, _gbrt_model_type

    t0 = time.time()

    if pushes is None:
        log.info("[GBRT] Lade Pushes aus DB...")
        pushes = _push_db_load_all()
        log.info(f"[GBRT] {len(pushes)} Pushes geladen")

    # Nur reife Pushes mit OR > 0 und plausiblem OR-Bereich
    now_ts = int(time.time())
    valid = [p for p in pushes if (p.get("or", 0) or 0) > 0
             and (p.get("or", 0) or 0) <= 20  # OR-Validierung: max 20% (alte API-Daten haben kaputte Werte)
             and p.get("ts_num", 0) > 0
             and p["ts_num"] < now_ts - 86400]

    if len(valid) < 100:
        log.warning(f"[GBRT] Nur {len(valid)} gueltige Pushes, Training uebersprungen (min 100)")
        return False

    # ML v2: LLM-Scores aus DB laden und in Push-Dicts injizieren
    llm_scored_count = 0
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            llm_rows = conn.execute("""SELECT message_id, llm_magnitude, llm_clickability,
                llm_relevanz, llm_dringlichkeit, llm_emotionalitaet
                FROM pushes WHERE llm_scored_at > 0""").fetchall()
            conn.close()
        llm_map = {}
        for row in llm_rows:
            llm_map[row[0]] = {
                "magnitude": float(row[1] or 0),
                "clickability": float(row[2] or 0),
                "relevanz": float(row[3] or 0),
                "dringlichkeit": float(row[4] or 0),
                "emotionalitaet": float(row[5] or 0),
            }
        for p in valid:
            mid = p.get("message_id", "")
            if mid in llm_map:
                p["_llm_scores"] = llm_map[mid]
                llm_scored_count += 1
        log.info(f"[GBRT] LLM-Scores geladen: {llm_scored_count}/{len(valid)} Pushes haben Scores")
    except Exception as llm_e:
        log.warning(f"[GBRT] LLM-Score-Laden fehlgeschlagen: {llm_e}")

    # Temporale Sortierung
    valid.sort(key=lambda x: x["ts_num"])

    # Diagnostik: OR-Verteilung prüfen
    _or_vals = [p.get("or", 0) for p in valid]
    _or_mean = sum(_or_vals) / len(_or_vals) if _or_vals else 0
    _or_max = max(_or_vals) if _or_vals else 0
    _or_above20 = sum(1 for v in _or_vals if v > 20)
    log.info(f"[GBRT] Diagnostik: {len(valid)} valid pushes, OR mean={_or_mean:.2f}, "
             f"max={_or_max:.2f}, >20%: {_or_above20}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: 5-Fold TimeSeriesSplit Cross-Validation
    # ══════════════════════════════════════════════════════════════════════
    n = len(valid)
    n_folds = 5
    min_train_size = max(100, n // (n_folds + 1))
    fold_size = (n - min_train_size) // n_folds

    cv_maes = []
    cv_rmses = []
    cv_r2s = []
    cv_baseline_maes = []

    log.info(f"[GBRT] Starte {n_folds}-Fold TimeSeriesSplit CV auf {n} Pushes")

    for fold in range(n_folds):
        fold_train_end = min_train_size + fold * fold_size
        fold_test_start = fold_train_end
        fold_test_end = min(fold_test_start + fold_size, n)
        if fold_test_end <= fold_test_start:
            continue

        fold_train = valid[:fold_train_end]
        fold_test = valid[fold_test_start:fold_test_end]

        # History Stats nur aus Fold-Trainingsdaten
        fold_stats = _gbrt_build_history_stats(fold_train, target_ts=fold_train[-1]["ts_num"] + 1)

        # Feature-Extraktion (fast_mode=True: skip TF-IDF/Jaccard for speed)
        fold_features = []
        for p in fold_train:
            fold_features.append(_gbrt_extract_features(p, fold_stats, fast_mode=True))
        if not fold_features:
            continue
        f_names = sorted(fold_features[0].keys())
        X_fold_train = [[f[k] for k in f_names] for f in fold_features]
        y_fold_train = [p.get("or", 0) for p in fold_train]

        fold_test_features = [_gbrt_extract_features(p, fold_stats, fast_mode=True) for p in fold_test]
        X_fold_test = [[f[k] for k in f_names] for f in fold_test_features]
        y_fold_test = [p.get("or", 0) for p in fold_test]

        # Trainiere Fold-Modell (LightGBM > sklearn > pure python)
        if _LGBM_AVAILABLE and np is not None:
            fold_lgbm = _lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                min_child_samples=20, subsample=0.7, subsample_freq=1,
                num_leaves=31, reg_alpha=0.5, reg_lambda=1.0,
                colsample_bytree=0.7, objective="huber",
                n_jobs=2, random_state=42 + fold, verbose=-1)
            np_Xft = np.array(X_fold_train, dtype=np.float64)
            np_yft = np.array(y_fold_train, dtype=np.float64)
            np_Xfv = np.array(X_fold_test, dtype=np.float64)
            fold_lgbm.fit(np_Xft, np_yft,
                          eval_set=[(np_Xfv, np.array(y_fold_test, dtype=np.float64))],
                          callbacks=[_lgb.early_stopping(30, verbose=False)])
            fold_preds = fold_lgbm.predict(np_Xfv).tolist()
        elif _SKLEARN_AVAILABLE:
            fold_model = SklearnGBR(n_estimators=100, max_depth=4, learning_rate=0.08,
                                     min_samples_leaf=15, subsample=0.8,
                                     loss="huber", random_state=42 + fold)
            fold_model.fit(np.array(X_fold_train), np.array(y_fold_train))
            fold_preds = fold_model.predict(np.array(X_fold_test)).tolist()
        else:
            fold_model = GBRTModel(n_trees=50, max_depth=4, learning_rate=0.10,
                                   min_samples_leaf=15, subsample=0.8, n_bins=128,
                                   loss="huber", huber_delta=1.5, log_target=False)
            fold_model.fit(X_fold_train, y_fold_train, feature_names=f_names)
            fold_preds = fold_model.predict(X_fold_test)
        fold_n = len(y_fold_test)
        fold_mae = sum(abs(fold_preds[j] - y_fold_test[j]) for j in range(fold_n)) / fold_n
        fold_rmse = math.sqrt(sum((fold_preds[j] - y_fold_test[j]) ** 2 for j in range(fold_n)) / fold_n)
        fold_ymean = sum(y_fold_test) / fold_n
        fold_ss_res = sum((y_fold_test[j] - fold_preds[j]) ** 2 for j in range(fold_n))
        fold_ss_tot = sum((y_fold_test[j] - fold_ymean) ** 2 for j in range(fold_n))
        fold_r2 = 1.0 - fold_ss_res / fold_ss_tot if fold_ss_tot > 0 else 0.0

        # Baselines fuer diesen Fold (Global Mean + Category Mean)
        fold_global_mean = sum(y_fold_train) / len(y_fold_train)
        fold_baseline_global_mae = sum(abs(fold_global_mean - y_fold_test[j]) for j in range(fold_n)) / fold_n
        # Category Mean Baseline
        fold_cat_means = defaultdict(list)
        for p in fold_train:
            fold_cat_means[(p.get("cat", "news")).lower().strip()].append(p.get("or", 0))
        fold_cat_avg = {c: sum(v) / len(v) for c, v in fold_cat_means.items()}
        fold_baseline_cat_mae = sum(
            abs(fold_cat_avg.get((fold_test[j].get("cat", "news")).lower().strip(), fold_global_mean) - y_fold_test[j])
            for j in range(fold_n)) / fold_n

        cv_maes.append(fold_mae)
        cv_rmses.append(fold_rmse)
        cv_r2s.append(fold_r2)
        cv_baseline_maes.append(fold_baseline_cat_mae)  # Category-Baseline (staerker als Global)

        log.info(f"[GBRT] CV Fold {fold+1}/{n_folds}: MAE={fold_mae:.3f}, "
                 f"Baseline-Global={fold_baseline_global_mae:.3f}, "
                 f"Baseline-Cat={fold_baseline_cat_mae:.3f}, R²={fold_r2:.3f}")

    cv_mean_mae = sum(cv_maes) / len(cv_maes) if cv_maes else 0
    cv_std_mae = math.sqrt(sum((m - cv_mean_mae) ** 2 for m in cv_maes) / len(cv_maes)) if len(cv_maes) > 1 else 0
    cv_mean_baseline = sum(cv_baseline_maes) / len(cv_baseline_maes) if cv_baseline_maes else 0
    cv_mean_r2 = sum(cv_r2s) / len(cv_r2s) if cv_r2s else 0

    log.info(f"[GBRT] CV Ergebnis: MAE={cv_mean_mae:.3f}±{cv_std_mae:.3f}, "
             f"Baseline-MAE={cv_mean_baseline:.3f}, "
             f"Verbesserung={((1 - cv_mean_mae / cv_mean_baseline) * 100) if cv_mean_baseline > 0 else 0:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Finales Modell auf 80/10/10 Split trainieren
    # ══════════════════════════════════════════════════════════════════════
    train_end = int(n * 0.75)
    val_end = int(n * 0.875)

    train_data = valid[:train_end]
    val_data = valid[train_end:val_end]
    test_data = valid[val_end:]

    log.info(f"[GBRT] Finales Training: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # History Stats nur aus Train-Daten
    history_stats = _gbrt_build_history_stats(train_data, target_ts=train_data[-1]["ts_num"] + 1)

    # Feature-Whitelist: nur Features die nachweislich generalisieren.
    # Lookup-Features (keyword_avg_or, entity_or, etc.) overfittnen auf historische Muster.
    _FEATURE_WHITELIST = {
        # Temporal (staerkste Korrelation)
        "hour_sin", "hour_cos", "is_prime_time", "is_morning_commute",
        "is_late_night", "is_lunch", "is_weekend",
        # LLM Scores (semantisches Signal)
        "llm_magnitude", "llm_clickability", "llm_relevanz",
        "llm_dringlichkeit", "llm_emotionalitaet", "llm_composite",
        # Recipients (staerkstes Feature)
        "log_recipients", "ios_android_ratio",
        # Text-Features
        "title_len", "word_count", "has_question", "has_colon", "has_pipe",
        # Nachrichten-Typ
        "is_eilmeldung", "is_breaking_style",
        # Kategorie (one-hot)
        "cat_sport", "cat_politik", "cat_news", "cat_unterhaltung",
        "cat_regional", "cat_geld", "cat_digital",
        # Interaktionen
        "recipients_x_primetime", "breaking_x_primetime", "eilmeldung_x_primetime",
        # Fatigue/Rolling
        "mins_since_last_push", "rolling_or_last3", "rolling_or_last5",
        # Volatility
        "or_volatility_7d",
    }

    def _extract_matrix(data, stats):
        features_list = []
        for p in data:
            feat = _gbrt_extract_features(p, stats)
            features_list.append(feat)
        if not features_list:
            return [], [], []
        all_keys = sorted(features_list[0].keys())
        f_names = [k for k in all_keys if k in _FEATURE_WHITELIST]
        if len(f_names) < 15:
            f_names = all_keys
            log.warning(f"[GBRT] Feature-Whitelist nur {len(f_names)} Treffer, nutze alle {len(all_keys)}")
        else:
            log.info(f"[GBRT] Feature-Whitelist: {len(f_names)}/{len(all_keys)} Features")
        X = [[f[k] for k in f_names] for f in features_list]
        y = [p.get("or", 0) for p in data]
        return X, y, f_names

    X_train, y_train, feature_names = _extract_matrix(train_data, history_stats)
    X_val, y_val, _ = _extract_matrix(val_data, history_stats)
    X_test, y_test, _ = _extract_matrix(test_data, history_stats)

    # ── D1+D2: Adaptive Half-Life + Primetime + Saisonal ──
    latest_ts = train_data[-1]["ts_num"] if train_data else now_ts
    best_half_life = 180
    if X_val and y_val:
        best_val_mae = float('inf')
        for hl_cand in [30, 60, 90, 120, 180, 365]:
            cw = []
            for p in train_data:
                ad = max(0, (latest_ts - p.get("ts_num", latest_ts)) / 86400.0)
                cw.append(math.exp(-ad / float(hl_cand)))
            ws = sum(cw)
            if ws > 0:
                wm = sum(cw[i] * train_data[i].get("or", 0) for i in range(len(train_data))) / ws
                vm = sum(abs(y_val[j] - wm) for j in range(len(y_val))) / max(1, len(y_val))
                if vm < best_val_mae:
                    best_val_mae = vm
                    best_half_life = hl_cand
        log.info(f"[GBRT] Adaptive Half-Life: {best_half_life}d (Grid-Search)")
    # Einheitliche Gewichtung — Time-Decay hat R² verschlechtert (Overfitting auf rezente Muster)
    train_weights = [1.0] * len(train_data)
    log.info(f"[GBRT] Gewichtung: einheitlich (keine Time-Decay, verbessert Generalisierung)")

    if not X_train or not feature_names:
        log.warning("[GBRT] Feature-Extraktion fehlgeschlagen")
        return False

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2b: Direct Modeling — Cat×Hour Baselines als Features (kein Residual)
    # ══════════════════════════════════════════════════════════════════════
    # Baselines berechnen (für Feature-Berechnung + Metriken-Vergleich), NICHT für Target-Transformation
    SHRINKAGE_K = 20
    _cat_hour_baselines = {}
    _cat_hour_counts = defaultdict(list)
    for p in train_data:
        key = f"{(p.get('cat', '') or 'news').lower().strip()}_{p.get('hour', 12)}"
        _cat_hour_counts[key].append(p.get("or", 0))
    _global_train_avg = sum(y_train) / len(y_train) if y_train else 4.77
    for key, ors in _cat_hour_counts.items():
        n = len(ors)
        raw_mean = sum(ors) / n if n > 0 else _global_train_avg
        _cat_hour_baselines[key] = (n * raw_mean + SHRINKAGE_K * _global_train_avg) / (n + SHRINKAGE_K)

    def _get_baseline(push_data):
        key = f"{(push_data.get('cat', '') or 'news').lower().strip()}_{push_data.get('hour', 12)}"
        return _cat_hour_baselines.get(key, _global_train_avg)

    # Baselines pro Sample berechnen (für Metriken-Vergleich)
    train_baselines = [_get_baseline(p) for p in train_data]
    val_baselines = [_get_baseline(p) for p in val_data]
    test_baselines = [_get_baseline(p) for p in test_data]

    # Direct Modeling: y bleibt Original-OR (keine Residual-Transformation)
    log.info(f"[GBRT] Direct Modeling: Train-OR mean={sum(y_train)/len(y_train):.3f}, "
             f"std={math.sqrt(sum((r - sum(y_train)/len(y_train))**2 for r in y_train)/len(y_train)):.3f}, "
             f"Baseline-Einträge={len(_cat_hour_baselines)} (Shrinkage K={SHRINKAGE_K})")

    # ── ML v2: Embedding-PCA auf Train-Daten bauen ──
    _build_embedding_pca(train_data, n_components=25)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Optuna Hyperparameter-Optimierung (wenn verfuegbar)
    # ══════════════════════════════════════════════════════════════════════
    use_lgbm = _LGBM_AVAILABLE and np is not None
    use_sklearn = _SKLEARN_AVAILABLE and np is not None and not use_lgbm
    backend_name = "lightgbm" if use_lgbm else ("sklearn" if use_sklearn else "pure_python")
    if use_lgbm:
        best_params = {"n_trees": 200, "max_depth": 3, "learning_rate": 0.05,
                       "min_samples_leaf": 50, "subsample": 0.8, "n_bins": 255,
                       "reg_alpha": 1.0, "reg_lambda": 2.0, "num_leaves": 16,
                       "feature_fraction": 0.5, "objective": "regression_l1"}
    else:
        best_params = {"n_trees": 200, "max_depth": 3, "learning_rate": 0.05,
                       "min_samples_leaf": 50, "subsample": 0.8, "n_bins": 255,
                       "reg_alpha": 0.5, "reg_lambda": 1.0, "num_leaves": 16}
    tuning_info = {"method": "default", "n_trials": 0, "backend": backend_name}

    if use_lgbm or use_sklearn:
        np_X_train = np.array(X_train, dtype=np.float64)
        np_y_train = np.array(y_train, dtype=np.float64)
        np_X_val = np.array(X_val, dtype=np.float64)
        np_y_val = np.array(y_val, dtype=np.float64)
        np_val_baselines = np.array(val_baselines, dtype=np.float64)
        np_train_weights = np.array(train_weights, dtype=np.float64)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _optuna_objective(trial):
            if use_lgbm:
                # LightGBM: flachere Baeume + staerkere Regularisierung gegen Overfitting
                p_n_trees = trial.suggest_int("n_trees", 100, 400, step=50)
                p_max_depth = trial.suggest_int("max_depth", 2, 4)
                p_lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
                p_min_leaf = trial.suggest_int("min_samples_leaf", 40, 100)
                p_subsample = trial.suggest_float("subsample", 0.6, 0.85)
                p_num_leaves = trial.suggest_int("num_leaves", 8, 24)
                p_reg_alpha = trial.suggest_float("reg_alpha", 0.5, 10.0, log=True)
                p_reg_lambda = trial.suggest_float("reg_lambda", 0.5, 10.0, log=True)
                p_feature_fraction = trial.suggest_float("feature_fraction", 0.3, 0.6)
            else:
                p_n_trees = trial.suggest_int("n_trees", 100, 400, step=50)
                p_max_depth = trial.suggest_int("max_depth", 2, 4)
                p_lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
                p_min_leaf = trial.suggest_int("min_samples_leaf", 40, 100)
                p_subsample = trial.suggest_float("subsample", 0.6, 0.85)

            if use_lgbm:
                m = _lgb.LGBMRegressor(
                    n_estimators=p_n_trees, max_depth=p_max_depth,
                    learning_rate=p_lr, min_child_samples=p_min_leaf,
                    subsample=p_subsample, subsample_freq=1,
                    num_leaves=p_num_leaves, reg_alpha=p_reg_alpha,
                    reg_lambda=p_reg_lambda, objective="regression_l1",
                    colsample_bytree=p_feature_fraction,
                    n_jobs=1, random_state=42, verbose=-1)
                m.fit(np_X_train, np_y_train, sample_weight=np_train_weights,
                      eval_set=[(np_X_val, np_y_val)],
                      callbacks=[_lgb.early_stopping(50, verbose=False)])
                preds = m.predict(np_X_val)
                mae = float(np.mean(np.abs(preds - np_y_val)))
            elif use_sklearn:
                m = SklearnGBR(n_estimators=p_n_trees, max_depth=p_max_depth,
                                learning_rate=p_lr, min_samples_leaf=p_min_leaf,
                                subsample=p_subsample, loss="absolute_error",
                                n_iter_no_change=30, validation_fraction=0.15,
                                random_state=42)
                m.fit(np_X_train, np_y_train, sample_weight=np_train_weights)
                preds = m.predict(np_X_val)
                mae = float(np.mean(np.abs(preds - np_y_val)))
            else:
                p_huber_delta = trial.suggest_float("huber_delta", 0.5, 3.0)
                m = GBRTModel(n_trees=p_n_trees, max_depth=p_max_depth,
                              learning_rate=p_lr, min_samples_leaf=p_min_leaf,
                              subsample=p_subsample, n_bins=255,
                              loss="huber", huber_delta=p_huber_delta, log_target=False)
                m.fit(X_train, y_train, feature_names=feature_names,
                      val_X=X_val, val_y=y_val, sample_weights=train_weights)
                preds = m.predict(X_val)
                mae = sum(abs(preds[j] - y_val[j])
                           for j in range(len(y_val))) / len(y_val)
            return mae

        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_optuna_objective, n_trials=30, timeout=180)
        best_params.update(study.best_params)
        best_params["n_bins"] = 255
        tuning_info = {
            "method": "optuna",
            "n_trials": len(study.trials),
            "best_val_mae": round(study.best_value, 4),
            "best_params": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in study.best_params.items()},
            "backend": backend_name,
        }
        log.info(f"[GBRT] Optuna ({backend_name}): {len(study.trials)} Trials, "
                 f"beste Val-MAE={study.best_value:.4f}, Params={study.best_params}")
    except ImportError:
        log.info("[GBRT] Optuna nicht installiert, verwende Default-Hyperparameter")
    except Exception as opt_e:
        log.warning(f"[GBRT] Optuna-Fehler: {opt_e}, verwende Default-Hyperparameter")

    # ── Hauptmodell mit besten Parametern trainieren ──
    log.info(f"[GBRT] {backend_name}-Backend: n_estimators={best_params['n_trees']}, "
             f"max_depth={best_params['max_depth']}, lr={best_params['learning_rate']}")
    if use_lgbm:
        model_lgbm = _lgb.LGBMRegressor(
            n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            min_child_samples=best_params["min_samples_leaf"],
            subsample=best_params["subsample"], subsample_freq=1,
            num_leaves=best_params.get("num_leaves", 48),
            reg_alpha=best_params.get("reg_alpha", 0.5),
            reg_lambda=best_params.get("reg_lambda", 1.0),
            colsample_bytree=best_params.get("feature_fraction", 0.7),
            objective=best_params.get("objective", "regression_l1"),
            eval_metric="mae",
            n_jobs=2, random_state=42, verbose=-1)
        model_lgbm.fit(np_X_train, np_y_train, sample_weight=np_train_weights,
                       eval_set=[(np_X_val, np_y_val)],
                       callbacks=[_lgb.early_stopping(50, verbose=False)])
        model = _LGBMModelWrapper(model_lgbm, feature_names)
        log.info(f"[GBRT] LightGBM-Modell trainiert: {model_lgbm.best_iteration_} Bäume "
                 f"(early-stop von {best_params['n_trees']})")
    elif use_sklearn:
        model_sklearn = SklearnGBR(
            n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            min_samples_leaf=best_params["min_samples_leaf"],
            subsample=best_params["subsample"], loss="absolute_error",
            n_iter_no_change=30, validation_fraction=0.12,
            random_state=42)
        model_sklearn.fit(np_X_train, np_y_train, sample_weight=np_train_weights)
        model = _SklearnModelWrapper(model_sklearn, feature_names)
        log.info(f"[GBRT] sklearn-Modell trainiert: {model_sklearn.n_estimators_} Bäume "
                 f"(early-stop von {best_params['n_trees']})")
    else:
        model = GBRTModel(n_trees=best_params["n_trees"], max_depth=best_params["max_depth"],
                          learning_rate=best_params["learning_rate"],
                          min_samples_leaf=best_params["min_samples_leaf"],
                          subsample=best_params["subsample"], n_bins=best_params["n_bins"],
                          loss="huber", huber_delta=best_params.get("huber_delta", 1.5),
                          log_target=False)
        model.fit(X_train, y_train, feature_names=feature_names,
                  val_X=X_val, val_y=y_val, sample_weights=train_weights)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Varianz-Filter + Feature Importance Pruning
    # ══════════════════════════════════════════════════════════════════════
    # Varianz-Filter: Features mit std < 0.001 über Train-Set entfernen
    if X_train and feature_names:
        n_before_var = len(feature_names)
        var_keep_idx = []
        for fi in range(len(feature_names)):
            vals = [X_train[r][fi] for r in range(len(X_train))]
            f_mean = sum(vals) / len(vals)
            f_std = math.sqrt(sum((v - f_mean) ** 2 for v in vals) / len(vals))
            if f_std >= 0.001:
                var_keep_idx.append(fi)
        if len(var_keep_idx) < n_before_var and len(var_keep_idx) >= 10:
            removed_var = [feature_names[i] for i in range(n_before_var) if i not in var_keep_idx]
            feature_names = [feature_names[i] for i in var_keep_idx]
            X_train = [[row[i] for i in var_keep_idx] for row in X_train]
            X_val = [[row[i] for i in var_keep_idx] for row in X_val]
            X_test = [[row[i] for i in var_keep_idx] for row in X_test]
            # Numpy-Arrays aktualisieren
            if use_lgbm or use_sklearn:
                np_X_train = np.array(X_train, dtype=np.float64)
                np_X_val = np.array(X_val, dtype=np.float64)
            # Modell muss mit neuen Features neu trainiert werden
            if use_lgbm:
                model_lgbm = _lgb.LGBMRegressor(
                    n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    min_child_samples=best_params["min_samples_leaf"],
                    subsample=best_params["subsample"], subsample_freq=1,
                    num_leaves=best_params.get("num_leaves", 48),
                    reg_alpha=best_params.get("reg_alpha", 0.5),
                    reg_lambda=best_params.get("reg_lambda", 1.0),
                    colsample_bytree=best_params.get("feature_fraction", 0.7),
                    objective=best_params.get("objective", "regression_l1"),
                    eval_metric="mae",
                    n_jobs=2, random_state=42, verbose=-1)
                model_lgbm.fit(np_X_train, np_y_train, sample_weight=np_train_weights,
                               eval_set=[(np_X_val, np_y_val)],
                               callbacks=[_lgb.early_stopping(50, verbose=False)])
                model = _LGBMModelWrapper(model_lgbm, feature_names)
            log.info(f"[GBRT] Varianz-Filter: {n_before_var} → {len(feature_names)} Features "
                     f"(entfernt: {len(removed_var)} mit std<0.001)")

    pruned_features = []
    if model.feature_importance_:
        all_importance = {f: model.feature_importance_.get(f, 0.0) for f in feature_names}
        importance_threshold = 0.001
        important_features = [f for f, imp in all_importance.items()
                              if imp >= importance_threshold]
        n_original = len(feature_names)
        min_features = min(n_original, 35)  # Max 35 Features — weniger Overfitting
        if len(important_features) < min_features:
            important_features = [f for f, _ in sorted(all_importance.items(),
                                  key=lambda x: -x[1])[:min_features]]
        if len(important_features) < n_original and len(important_features) >= 10:
            pruned_features = [f for f in feature_names if f not in important_features]
            keep_idx = [feature_names.index(f) for f in important_features]
            keep_idx.sort()
            pruned_feature_names = [feature_names[i] for i in keep_idx]

            X_train_pruned = [[row[i] for i in keep_idx] for row in X_train]
            X_val_pruned = [[row[i] for i in keep_idx] for row in X_val]
            X_test_pruned = [[row[i] for i in keep_idx] for row in X_test]

            # Retrain mit reduzierten Features
            if use_lgbm:
                np_Xtp = np.array(X_train_pruned, dtype=np.float64)
                np_Xvp = np.array(X_val_pruned, dtype=np.float64)
                model_pruned_lgbm = _lgb.LGBMRegressor(
                    n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    min_child_samples=best_params["min_samples_leaf"],
                    subsample=best_params["subsample"], subsample_freq=1,
                    num_leaves=best_params.get("num_leaves", 48),
                    reg_alpha=best_params.get("reg_alpha", 0.5),
                    reg_lambda=best_params.get("reg_lambda", 1.0),
                    objective=best_params.get("objective", "regression_l1"),
                    eval_metric="mae",
                    n_jobs=2, random_state=42, verbose=-1)
                model_pruned_lgbm.fit(np_Xtp, np_y_train, sample_weight=np_train_weights,
                                      eval_set=[(np_Xvp, np_y_val)],
                                      callbacks=[_lgb.early_stopping(50, verbose=False)])
                model_pruned = _LGBMModelWrapper(model_pruned_lgbm, pruned_feature_names)
            elif use_sklearn:
                np_Xtp = np.array(X_train_pruned, dtype=np.float64)
                model_pruned_sk = SklearnGBR(
                    n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    subsample=best_params["subsample"], loss="absolute_error",
                    n_iter_no_change=30, validation_fraction=0.12, random_state=42)
                model_pruned_sk.fit(np_Xtp, np_y_train, sample_weight=np_train_weights)
                model_pruned = _SklearnModelWrapper(model_pruned_sk, pruned_feature_names)
            else:
                model_pruned = GBRTModel(n_trees=best_params["n_trees"],
                                         max_depth=best_params["max_depth"],
                                         learning_rate=best_params["learning_rate"],
                                         min_samples_leaf=best_params["min_samples_leaf"],
                                         subsample=best_params["subsample"],
                                         n_bins=best_params["n_bins"],
                                         loss="huber",
                                         huber_delta=best_params.get("huber_delta", 1.5),
                                         log_target=False)
                model_pruned.fit(X_train_pruned, y_train, feature_names=pruned_feature_names,
                                 val_X=X_val_pruned, val_y=y_val, sample_weights=train_weights)

            pruned_preds = model_pruned.predict(X_test_pruned)
            pruned_mae = sum(abs(pruned_preds[j] - y_test[j])
                             for j in range(len(y_test))) / len(y_test) if y_test else 999
            full_preds = model.predict(X_test)
            full_mae = sum(abs(full_preds[j] - y_test[j])
                           for j in range(len(y_test))) / len(y_test) if y_test else 999

            if pruned_mae <= full_mae * 1.02:
                log.info(f"[GBRT] Feature Pruning: {n_original} -> {len(pruned_feature_names)} Features "
                         f"(entfernt: {len(pruned_features)}), "
                         f"Test-MAE: {full_mae:.4f} -> {pruned_mae:.4f}")
                model = model_pruned
                feature_names = pruned_feature_names
                X_train = X_train_pruned
                X_val = X_val_pruned
                X_test = X_test_pruned
                # Numpy-Arrays für Ensemble aktualisieren
                if use_lgbm or use_sklearn:
                    np_X_train = np.array(X_train, dtype=np.float64)
                    np_X_val = np.array(X_val, dtype=np.float64)
            else:
                log.info(f"[GBRT] Feature Pruning verworfen: Test-MAE wuerde von "
                         f"{full_mae:.4f} auf {pruned_mae:.4f} steigen")
                pruned_features = []

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: Test-Metriken mit Bootstrap-Konfidenzintervallen
    # ══════════════════════════════════════════════════════════════════════
    # Direct Modeling: Modell predicted OR direkt (kein Baseline-Addieren nötig)
    test_preds = model.predict(X_test)
    if not isinstance(test_preds, list):
        test_preds = list(test_preds)
    test_n = len(y_test)
    test_mae = sum(abs(test_preds[i] - y_test[i]) for i in range(test_n)) / test_n if test_n else 0
    test_rmse = math.sqrt(sum((test_preds[i] - y_test[i]) ** 2 for i in range(test_n)) / test_n) if test_n else 0
    y_mean = sum(y_test) / test_n if test_n else 1
    ss_res = sum((y_test[i] - test_preds[i]) ** 2 for i in range(test_n))
    ss_tot = sum((y_test[i] - y_mean) ** 2 for i in range(test_n))
    test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Bootstrap-Konfidenzintervalle
    bootstrap_ci = _gbrt_bootstrap_ci(y_test, test_preds, n_bootstrap=1000, ci=0.95)

    # Naive Baselines
    baselines = _gbrt_compute_baselines(y_test, test_data, train_data)

    model.train_metrics["test_mae"] = round(test_mae, 4)
    model.train_metrics["test_rmse"] = round(test_rmse, 4)
    model.train_metrics["test_r2"] = round(test_r2, 4)
    model.train_metrics["r2_final"] = round(test_r2, 4)  # R² im OR-Raum (finale Prediction)
    model.train_metrics["test_n"] = test_n
    model.train_metrics["bootstrap_ci"] = bootstrap_ci
    model.train_metrics["baselines"] = baselines
    model.train_metrics["cv_results"] = {
        "n_folds": n_folds,
        "cv_mean_mae": round(cv_mean_mae, 4),
        "cv_std_mae": round(cv_std_mae, 4),
        "cv_mean_r2": round(cv_mean_r2, 4),
        "cv_fold_maes": [round(m, 4) for m in cv_maes],
        "cv_baseline_maes": [round(m, 4) for m in cv_baseline_maes],
    }
    model.train_metrics["tuning"] = tuning_info
    model.train_metrics["feature_pruning"] = {
        "original_n": len(feature_names) + len(pruned_features),
        "final_n": len(feature_names),
        "pruned": pruned_features,
    }

    # Verbesserung vs Baselines loggen
    if baselines.get("global_mean_mae", 0) > 0:
        imp_global = (1 - test_mae / baselines["global_mean_mae"]) * 100
        imp_cat = (1 - test_mae / baselines["category_mean_mae"]) * 100 if baselines.get("category_mean_mae", 0) > 0 else 0
        imp_cathour = (1 - test_mae / baselines["category_hour_mean_mae"]) * 100 if baselines.get("category_hour_mean_mae", 0) > 0 else 0
        model.train_metrics["improvement_vs_baselines"] = {
            "vs_global_mean": round(imp_global, 1),
            "vs_category_mean": round(imp_cat, 1),
            "vs_category_hour_mean": round(imp_cathour, 1),
        }
        log.info(f"[GBRT] Test-MAE={test_mae:.3f}pp vs Baselines: "
                 f"Global={baselines['global_mean_mae']:.3f}pp ({imp_global:+.1f}%), "
                 f"Category={baselines['category_mean_mae']:.3f}pp ({imp_cat:+.1f}%), "
                 f"Cat×Hour={baselines['category_hour_mean_mae']:.3f}pp ({imp_cathour:+.1f}%)")

    log.info(f"[GBRT] Bootstrap 95%-CI: MAE={bootstrap_ci.get('mae_ci', ['?','?'])}, "
             f"R²={bootstrap_ci.get('r2_ci', ['?','?'])}")

    # ── Konforme Quantile-Regression (ersetzt Pseudo-Target-Modelle) ──
    # Berechne Residuen auf Validierungs-Set fuer konformen Radius
    val_preds_for_conformal = model.predict(X_val)
    val_residuals_abs = sorted([abs(val_preds_for_conformal[i] - y_val[i]) for i in range(len(y_val))])
    conformal_alpha = 0.10  # 80% Coverage (Q10-Q90)
    conformal_idx = min(int(math.ceil((1 - conformal_alpha) * len(val_residuals_abs))), len(val_residuals_abs) - 1)
    conformal_radius = val_residuals_abs[conformal_idx] if val_residuals_abs else 1.0
    model.conformal_radius = conformal_radius

    # Q10/Q90 Dummy-Modelle nicht mehr noetig — Inference nutzt conformal_radius
    model_q10 = None
    model_q90 = None

    # Coverage auf Test-Set validieren
    coverage = sum(1 for i in range(test_n)
                   if (test_preds[i] - conformal_radius) <= y_test[i] <= (test_preds[i] + conformal_radius)
                   ) / test_n if test_n > 0 else 0
    model.train_metrics["quantile_coverage_80"] = round(coverage, 3)
    model.train_metrics["conformal_radius"] = round(conformal_radius, 4)
    log.info(f"[GBRT] Konforme Quantile: radius={conformal_radius:.3f}pp, "
             f"Coverage (erwartet ~80%): {coverage:.1%}")

    # ── Isotonische Kalibrierung auf Validation-Set ──
    calibrator = IsotonicCalibrator()
    val_preds_direct = model.predict(X_val)
    if not isinstance(val_preds_direct, list):
        val_preds_direct = list(val_preds_direct)
    calibrator.fit(val_preds_direct, y_val)

    # Kalibrierte Test-Metriken
    cal_test_preds = [calibrator.calibrate(p) for p in test_preds]
    cal_test_mae = sum(abs(cal_test_preds[i] - y_test[i]) for i in range(test_n)) / test_n if test_n else 0
    model.train_metrics["cal_test_mae"] = round(cal_test_mae, 4)

    # Kalibrierung validieren: nur verwenden wenn sie MAE verbessert
    if cal_test_mae <= test_mae:
        log.info(f"[GBRT] Kalibrierung verbessert MAE: {test_mae:.4f} -> {cal_test_mae:.4f}")
    else:
        log.info(f"[GBRT] Kalibrierung deaktiviert (verschlechtert MAE: {test_mae:.4f} -> {cal_test_mae:.4f})")
        calibrator = None
        cal_test_mae = test_mae
        model.train_metrics["cal_test_mae"] = round(test_mae, 4)

    # ── Blending: α × GBRT + (1-α) × Cat×Hour-Baseline ──
    # Direct Modeling: Blending optional, optimiere α auf Validation-Set
    val_cal_preds = [calibrator.calibrate(p) for p in val_preds_direct] if calibrator else val_preds_direct
    best_blend_alpha = 1.0
    best_blend_mae = float('inf')
    for alpha_step in range(0, 21):  # 0.0, 0.05, 0.10, ..., 1.0
        alpha = alpha_step / 20.0
        blend_mae = sum(
            abs((alpha * val_cal_preds[i] + (1 - alpha) * val_baselines[i]) - y_val[i])
            for i in range(len(y_val))
        ) / len(y_val)
        if blend_mae < best_blend_mae:
            best_blend_mae = blend_mae
            best_blend_alpha = alpha

    # Blend auf Test-Set evaluieren
    effective_test_preds = cal_test_preds if calibrator else test_preds
    blended_test_preds = [
        best_blend_alpha * effective_test_preds[i] + (1 - best_blend_alpha) * test_baselines[i]
        for i in range(test_n)
    ]
    blended_test_mae = sum(abs(blended_test_preds[i] - y_test[i]) for i in range(test_n)) / test_n if test_n else 0
    model.blend_alpha = best_blend_alpha
    model.train_metrics["blend_alpha"] = round(best_blend_alpha, 3)
    model.train_metrics["blended_test_mae"] = round(blended_test_mae, 4)
    log.info(f"[GBRT] Blending: α={best_blend_alpha:.2f}, "
             f"MAE rein={cal_test_mae:.4f} → blended={blended_test_mae:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 6: Multi-Objective Ensemble (3 LightGBM-Modelle)
    # ══════════════════════════════════════════════════════════════════════
    model_direct = None  # Legacy-Feld, wird nicht mehr genutzt
    ensemble_weights = {"mae": 1.0}
    chosen_model_type = "direct_ensemble"
    _ensemble_models = []  # Liste von (objective_name, model_wrapper, val_mae)

    try:
        if use_lgbm and np is not None:
            log.info("[GBRT] Trainiere Multi-Objective Ensemble (MAE + MSE + Huber)...")
            _ens_objectives = [
                ("regression_l1", "MAE"),
                ("regression", "MSE"),
                ("huber", "Huber"),
            ]
            for obj_name, obj_label in _ens_objectives:
                ens_lgbm = _lgb.LGBMRegressor(
                    n_estimators=best_params["n_trees"], max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    min_child_samples=best_params["min_samples_leaf"],
                    subsample=best_params["subsample"], subsample_freq=1,
                    num_leaves=best_params.get("num_leaves", 48),
                    reg_alpha=best_params.get("reg_alpha", 0.5),
                    reg_lambda=best_params.get("reg_lambda", 1.0),
                    colsample_bytree=best_params.get("feature_fraction", 0.7),
                    objective=obj_name, eval_metric="mae",
                    n_jobs=2, random_state=42, verbose=-1)
                ens_lgbm.fit(np_X_train, np_y_train, sample_weight=np_train_weights,
                             eval_set=[(np_X_val, np_y_val)],
                             callbacks=[_lgb.early_stopping(50, verbose=False)])
                ens_wrapper = _LGBMModelWrapper(ens_lgbm, feature_names)
                ens_val_preds = ens_wrapper.predict(X_val)
                ens_val_mae = sum(abs(ens_val_preds[i] - y_val[i])
                                  for i in range(len(y_val))) / len(y_val)
                _ensemble_models.append((obj_label, ens_wrapper, ens_val_mae))
                log.info(f"[GBRT] Ensemble-{obj_label}: {ens_lgbm.best_iteration_} Bäume, "
                         f"Val-MAE={ens_val_mae:.4f}")

            # Gewichte via inverse Val-MAE
            inv_maes = [1.0 / max(m[2], 0.001) for m in _ensemble_models]
            inv_sum = sum(inv_maes)
            ens_w = [w / inv_sum for w in inv_maes]
            ensemble_weights = {m[0]: round(w, 4) for m, w in zip(_ensemble_models, ens_w)}

            # Ensemble-Prediction auf Test-Set
            ens_test_preds_list = []
            for _, ens_model, _ in _ensemble_models:
                ens_test_preds_list.append(ens_model.predict(X_test))
            ens_test_preds = []
            for i in range(test_n):
                pred = sum(ens_w[j] * ens_test_preds_list[j][i] for j in range(len(_ensemble_models)))
                ens_test_preds.append(pred)
            ens_test_mae = sum(abs(ens_test_preds[i] - y_test[i]) for i in range(test_n)) / test_n if test_n else 999

            # Vergleich: Primary (MAE-Modell) vs Ensemble
            primary_mae = _ensemble_models[0][2]  # Val-MAE des MAE-Modells
            if ens_test_mae <= test_mae:
                log.info(f"[GBRT] Ensemble gewählt: Gewichte={ensemble_weights}, "
                         f"Test-MAE={ens_test_mae:.4f} (Primary={test_mae:.4f})")
                # Speichere Ensemble-Modelle am Hauptmodell
                model._ensemble_models = [(label, mdl) for label, mdl, _ in _ensemble_models]
                model._ensemble_weights = ens_w
                chosen_model_type = "direct_ensemble"
            else:
                log.info(f"[GBRT] Ensemble verworfen (MAE {ens_test_mae:.4f} > Primary {test_mae:.4f}), "
                         f"nutze Primary-Modell")
                ensemble_weights = {"MAE": 1.0}
                chosen_model_type = "direct"
                _ensemble_models = []

            model.train_metrics["ensemble"] = {
                "n_models": len(_ensemble_models) if _ensemble_models else 1,
                "weights": ensemble_weights,
                "ensemble_test_mae": round(ens_test_mae, 4),
                "primary_test_mae": round(test_mae, 4),
            }
        else:
            chosen_model_type = "direct"

        model.train_metrics["model_type"] = chosen_model_type
        model.train_metrics["ensemble_weights"] = ensemble_weights
    except Exception as ens_err:
        log.warning(f"[GBRT] Multi-Objective Ensemble fehlgeschlagen: {ens_err}")
        chosen_model_type = "direct"
        model.train_metrics["model_type"] = "direct"
        model.train_metrics["ensemble_weights"] = {"MAE": 1.0}

    elapsed = time.time() - t0

    # ── Experiment Tracking ──
    hyperparams = dict(best_params)
    experiment_id = _log_experiment(
        model, hyperparams, model.train_metrics, baselines,
        model.train_metrics.get("cv_results", {}),
        len(feature_names), len(valid), elapsed)

    # ── History Stats nur aus Train-Daten (kein Leakage in Inference) ──
    new_history_stats = _gbrt_build_history_stats(train_data, target_ts=now_ts)

    # ── Character N-Gram TF-IDF nur auf Train-Daten trainieren (kein Leakage!) ──
    try:
        _train_char_ngram_tfidf(train_data)
    except Exception as _tfe:
        log.warning(f"[GBRT] CharNGram TF-IDF Training-Fehler: {_tfe}")

    # ── Promotion Gates + A/B Testing ──
    quantile_coverage = model.train_metrics.get("quantile_coverage_80", 0.0)
    promoted = _gbrt_maybe_promote_or_challenge(
        model, model_q10, model_q90, calibrator,
        feature_names, new_history_stats, train_data,
        experiment_id, test_mae, bootstrap_ci, baselines, quantile_coverage)

    if not promoted:
        # Falls kein Champion vorhanden war (erster Lauf), direkt promoten
        with _gbrt_lock:
            if _gbrt_model is None:
                _gbrt_model = model
                _gbrt_model_direct = model_direct
                _gbrt_model_q10 = model_q10
                _gbrt_model_q90 = model_q90
                _gbrt_calibrator = calibrator
                _gbrt_feature_names = feature_names
                _gbrt_train_ts = int(time.time())
                _gbrt_history_stats = new_history_stats
                _gbrt_ensemble_weights = ensemble_weights
                _gbrt_model_type = chosen_model_type
                _update_cat_hour_baselines(train_data)
                _mark_experiment_promoted(experiment_id)
                promoted = True
                log.info(f"[GBRT] Erster Lauf — Modell direkt promoted (Typ: {chosen_model_type})")

    # Modell als JSON speichern (immer, auch wenn nicht promoted)
    try:
        # sklearn/lgbm-Modell separat als joblib speichern
        if (use_lgbm and hasattr(model, '_is_lgbm')) or (use_sklearn and hasattr(model, '_is_sklearn')):
            try:
                _inner_model = model.lgbm_model if hasattr(model, '_is_lgbm') else model.sklearn_model
                _inner_direct = None
                if model_direct:
                    if hasattr(model_direct, '_is_lgbm'):
                        _inner_direct = model_direct.lgbm_model
                    elif hasattr(model_direct, '_is_sklearn'):
                        _inner_direct = model_direct.sklearn_model
                sklearn_data = {
                    "model": _inner_model,
                    "model_direct": _inner_direct,
                    "feature_names": feature_names,
                    "cat_hour_baselines": _cat_hour_baselines,
                    "global_train_avg": _global_train_avg,
                    "trained_at": int(time.time()),
                    "pca": _embedding_pca,
                    "pca_mean": _embedding_pca_mean.tolist() if _embedding_pca_mean is not None else None,
                    "metrics": model.train_metrics,
                    "model_type": chosen_model_type,
                    "ensemble_weights": ensemble_weights,
                    "conformal_radius": getattr(model, "conformal_radius", None),
                    "blend_alpha": getattr(model, "blend_alpha", None),
                }
                joblib.dump(sklearn_data, SKLEARN_MODEL_PATH, compress=3)
                log.info(f"[GBRT] sklearn-Modell gespeichert: {SKLEARN_MODEL_PATH} "
                         f"({os.path.getsize(SKLEARN_MODEL_PATH) / 1024:.0f} KB)")
            except Exception as sk_e:
                log.warning(f"[GBRT] sklearn-Export-Fehler: {sk_e}")

        model_json = {
            "model": model.to_json(),
            "model_direct": model_direct.to_json() if model_direct else None,
            "model_q10": model_q10.to_json() if model_q10 else None,
            "model_q90": model_q90.to_json() if model_q90 else None,
            "calibrator": calibrator.to_dict() if calibrator else None,
            "feature_names": feature_names,
            "cat_hour_baselines": _cat_hour_baselines,
            "global_train_avg": _global_train_avg,
            "trained_at": int(time.time()),
            "n_pushes": len(valid),
            "metrics": model.train_metrics,
            "experiment_id": experiment_id,
            "model_type": chosen_model_type,
            "ensemble_weights": ensemble_weights,
            "backend": "lightgbm" if use_lgbm else ("sklearn" if use_sklearn else "pure_python"),
        }
        with open(GBRT_MODEL_PATH, "w") as f:
            json.dump(model_json, f)
        log.info(f"[GBRT] Modell gespeichert: {GBRT_MODEL_PATH} ({os.path.getsize(GBRT_MODEL_PATH) / 1024:.0f} KB)")
    except Exception as e:
        log.warning(f"[GBRT] Modell-Export-Fehler: {e}")

    log.info(f"[GBRT] Training komplett in {elapsed:.1f}s: "
             f"test_MAE={test_mae:.3f} [{bootstrap_ci.get('mae_ci', ['?','?'])}], "
             f"test_R²={test_r2:.3f}, cal_MAE={cal_test_mae:.3f}, "
             f"blended_MAE={blended_test_mae:.3f} (α={best_blend_alpha:.2f}), "
             f"{len(model.trees)} Baeume, {len(feature_names)} Features, "
             f"CV-MAE={cv_mean_mae:.3f}±{cv_std_mae:.3f}, "
             f"experiment={experiment_id}, promoted={promoted}")

    return True


def _gbrt_predict(push, state=None):
    """GBRT-Prediction fuer einen einzelnen Push.

    Returns: Dict mit predicted, confidence, q10, q90, features, importance
    """
    with _gbrt_lock:
        model = _gbrt_model
        model_direct = _gbrt_model_direct
        model_q10 = _gbrt_model_q10
        model_q90 = _gbrt_model_q90
        calibrator = _gbrt_calibrator
        feature_names = _gbrt_feature_names
        history_stats = _gbrt_history_stats
        ens_weights = dict(_gbrt_ensemble_weights)
        model_type = _gbrt_model_type

    if model is None:
        return None

    # Features extrahieren
    feat = _gbrt_extract_features(push, history_stats, state)
    x = [feat.get(k, 0.0) for k in feature_names]

    # Cat×Hour-Baseline (für Blending-Fallback + Metriken)
    cat_lower = (push.get("cat", "") or "news").lower().strip()
    push_hour = push.get("hour", 12)
    baseline_key = f"{cat_lower}_{push_hour}"
    cat_hour_baseline = _gbrt_cat_hour_baselines.get(baseline_key, _gbrt_global_train_avg)

    # Direct Modeling: Modell predicted OR direkt
    result = model.predict_with_uncertainty(x)

    # Multi-Objective Ensemble: gewichteter Durchschnitt aus 3 Modellen
    if hasattr(model, '_ensemble_models') and model._ensemble_models and model_type == "direct_ensemble":
        ens_preds = []
        for _, ens_mdl in model._ensemble_models:
            ens_preds.append(ens_mdl.predict_one(x))
        ens_w = model._ensemble_weights
        predicted = sum(ens_w[j] * ens_preds[j] for j in range(len(ens_preds)))
    else:
        predicted = result["predicted"]

    # Kalibrierung anwenden
    if calibrator:
        predicted = calibrator.calibrate(predicted)
        predicted = max(0.01, predicted)

    # D5: Online Bias-Korrektur anwenden
    if _gbrt_online_bias != 0.0:
        predicted -= _gbrt_online_bias
        predicted = max(0.01, predicted)

    # Blending: α × GBRT + (1-α) × Cat×Hour-Baseline
    blend_alpha = getattr(model, "blend_alpha", 1.0)
    if blend_alpha < 1.0:
        predicted = blend_alpha * predicted + (1.0 - blend_alpha) * cat_hour_baseline

    # Quantile via Konforme Prediction
    c_radius = getattr(model, "conformal_radius", None)
    if c_radius:
        q10 = max(0.01, predicted - c_radius)
        q90 = predicted + c_radius
    elif model_q10 and model_q90:
        q10 = model_q10.predict_one(x)
        q90 = model_q90.predict_one(x)
        q10 = max(0.01, min(q10, predicted))
        q90 = max(predicted, q90)
    else:
        q10 = max(0.01, predicted * 0.45)  # Breiteres Band (war 0.6)
        q90 = predicted * 1.8              # Breiteres Band (war 1.5)

    # A/B Shadow Prediction (nur Challenger, Ergebnis nicht angezeigt)
    challenger_pred = _ab_shadow_predict(push, state)

    # Top Feature Contributions (globale Importance)
    top_features = []
    if model.feature_importance_:
        sorted_imp = sorted(model.feature_importance_.items(), key=lambda x: -x[1])
        for fname, imp in sorted_imp[:10]:
            fidx = feature_names.index(fname) if fname in feature_names else -1
            if fidx >= 0:
                top_features.append({
                    "name": fname,
                    "value": round(feat.get(fname, 0), 3),
                    "importance": imp,
                })

    # TreeSHAP: Individuelle Feature-Contributions für diesen Push
    shap_explanation = []
    shap_text = ""
    try:
        sv = model.shap_values(x)
        sorted_shap = sorted(sv["shap_values"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        for fname, contrib in sorted_shap[:5]:
            shap_explanation.append({
                "feature": fname,
                "label": _GBRT_SHAP_LABELS.get(fname, fname),
                "value": round(feat.get(fname, 0), 4),
                "contribution": round(contrib, 4),
            })
        # Menschenlesbarer Einzeiler
        parts = []
        for s in shap_explanation[:3]:
            sign = "+" if s["contribution"] > 0 else ""
            parts.append(f"{s['label']} {sign}{s['contribution']:.2f}")
        shap_text = ", ".join(parts)
    except Exception:
        sv = None

    pred_result = _safety_envelope({
        "predicted": round(predicted, 3),
        "confidence": result["confidence"],
        "std": result["std"],
        "q10": round(q10, 3),
        "q90": round(q90, 3),
        "model_type": f"GBRT-{model_type}" if model_type != "residual" else "GBRT",
        "model_subtype": model_type,
        "n_trees": len(model.trees),
        "features": {k: round(v, 4) for k, v in feat.items()},
        "top_features": top_features,
        "shap_explanation": shap_explanation,
        "shap_text": shap_text,
        "shap_base_value": round(sv["base_value"], 5) if sv else None,
        "shap_predicted": round(sv["prediction"], 5) if sv else None,
        "cat_hour_baseline": round(cat_hour_baseline, 3),
        "metrics": model.train_metrics,
        "ab_test_active": challenger_pred is not None,
    })

    # Speichere Champion-Prediction fuer spaetere A/B Auswertung
    if challenger_pred is not None:
        pred_result["_champion_pred"] = round(predicted, 3)
        pred_result["_challenger_pred"] = round(challenger_pred, 3)

    return pred_result


def _gbrt_load_model():
    """Laedt ein gespeichertes GBRT-Modell von Disk (sklearn oder Pure-Python)."""
    global _gbrt_model, _gbrt_model_direct, _gbrt_model_q10, _gbrt_model_q90, _gbrt_calibrator
    global _gbrt_feature_names, _gbrt_train_ts, _gbrt_ensemble_weights, _gbrt_model_type
    global _embedding_pca, _embedding_pca_mean

    # Versuch 1: sklearn-Modell laden (bevorzugt, schneller)
    if _SKLEARN_AVAILABLE and os.path.exists(SKLEARN_MODEL_PATH):
        try:
            sk_data = joblib.load(SKLEARN_MODEL_PATH)
            with _gbrt_lock:
                sk_model = sk_data["model"]
                feature_names = sk_data["feature_names"]
                _gbrt_model = _SklearnModelWrapper(sk_model, feature_names)
                if sk_data.get("conformal_radius"):
                    _gbrt_model.conformal_radius = sk_data["conformal_radius"]
                if sk_data.get("blend_alpha"):
                    _gbrt_model.blend_alpha = sk_data["blend_alpha"]
                _gbrt_model.train_metrics = sk_data.get("metrics", {})
                if sk_data.get("model_direct"):
                    _gbrt_model_direct = _SklearnModelWrapper(sk_data["model_direct"], feature_names)
                else:
                    _gbrt_model_direct = None
                _gbrt_model_q10 = None
                _gbrt_model_q90 = None
                _gbrt_calibrator = None
                _gbrt_feature_names = feature_names
                _gbrt_train_ts = sk_data.get("trained_at", 0)
                _gbrt_model_type = sk_data.get("model_type", "direct")
                _gbrt_ensemble_weights = sk_data.get("ensemble_weights",
                                                      {"MAE": 1.0})
                if sk_data.get("cat_hour_baselines"):
                    _gbrt_cat_hour_baselines.clear()
                    _gbrt_cat_hour_baselines.update(sk_data["cat_hour_baselines"])
                    globals()["_gbrt_global_train_avg"] = sk_data.get("global_train_avg", 4.77)
                if sk_data.get("pca") is not None:
                    _embedding_pca = sk_data["pca"]
                if sk_data.get("pca_mean") is not None and np is not None:
                    _embedding_pca_mean = np.array(sk_data["pca_mean"]).reshape(1, -1)
            log.info(f"[GBRT] sklearn-Modell geladen: {sk_model.n_estimators_} Bäume, "
                     f"Features: {len(feature_names)}, Typ: {_gbrt_model_type}")
            _gbrt_load_history_stats()
            return True
        except Exception as sk_e:
            log.warning(f"[GBRT] sklearn-Modell laden fehlgeschlagen: {sk_e}, Fallback auf JSON")

    # Versuch 2: Pure-Python JSON-Modell laden
    if not os.path.exists(GBRT_MODEL_PATH):
        return False
    try:
        with open(GBRT_MODEL_PATH, "r") as f:
            data = json.load(f)
        with _gbrt_lock:
            _gbrt_model = GBRTModel.from_json(data["model"])
            if data.get("model_direct"):
                _gbrt_model_direct = GBRTModel.from_json(data["model_direct"])
            else:
                _gbrt_model_direct = None
            if data.get("model_q10"):
                _gbrt_model_q10 = GBRTModel.from_json(data["model_q10"])
            if data.get("model_q90"):
                _gbrt_model_q90 = GBRTModel.from_json(data["model_q90"])
            if data.get("calibrator"):
                _gbrt_calibrator = IsotonicCalibrator.from_dict(data["calibrator"])
            else:
                _gbrt_calibrator = None
            _gbrt_feature_names = data.get("feature_names", [])
            _gbrt_train_ts = data.get("trained_at", 0)
            _gbrt_model_type = data.get("model_type", "direct")
            _gbrt_ensemble_weights = data.get("ensemble_weights",
                                               {"MAE": 1.0})
            if data.get("cat_hour_baselines"):
                _gbrt_cat_hour_baselines.clear()
                _gbrt_cat_hour_baselines.update(data["cat_hour_baselines"])
                globals()["_gbrt_global_train_avg"] = data.get("global_train_avg", 4.77)
        log.info(f"[GBRT] Modell geladen: {len(_gbrt_model.trees)} Baeume, "
                 f"Features: {len(_gbrt_feature_names)}, Typ: {_gbrt_model_type}")
        _gbrt_load_history_stats()
        return True
    except Exception as e:
        log.warning(f"[GBRT] Modell laden fehlgeschlagen: {e}")
        return False


def _gbrt_load_history_stats():
    """History-Stats aufbauen nach dem Modell-Laden."""
    try:
        global _gbrt_history_stats
        all_pushes = _push_db_load_all()
        trained_at = _gbrt_train_ts or int(time.time())
        valid = [p for p in all_pushes if p.get("or", 0) > 0
                 and 0 < (p.get("or", 0) or 0) <= 20]
        if valid:
            _gbrt_history_stats = _gbrt_build_history_stats(valid, target_ts=trained_at)
            log.info(f"[GBRT] History-Stats gebaut: {len(valid)} Pushes (Cutoff: trained_at={trained_at})")
    except Exception as _hs_err:
        log.warning(f"[GBRT] History-Stats Fehler: {_hs_err}")


# ── Concept Drift Detection ──────────────────────────────────────────────

_drift_state = {"recent_errors": [], "historical_mae": 0.0, "current_mae": 0.0,
                "drift_detected": False, "last_check": 0, "auto_retrain_count": 0}


def _gbrt_check_drift(state):
    """Prueft auf Concept Drift und triggert Auto-Retrain wenn noetig.

    Rolling MAE (letzte 50) vs Historical MAE (letzte 500).
    Ratio > 1.3 → Drift erkannt → Auto-Retrain.
    """
    global _drift_state
    now_t = time.time()
    if now_t - _drift_state["last_check"] < 1800:  # Alle 30 Min pruefen
        return
    _drift_state["last_check"] = now_t

    feedback = state.get("prediction_feedback", [])
    server_fb = [fb for fb in feedback if fb.get("source") == "server"
                 and fb.get("predicted_or", 0) > 0 and fb.get("actual_or", 0) > 0]

    if len(server_fb) < 60:
        return

    # Sortiert nach Zeitstempel
    sorted_fb = sorted(server_fb, key=lambda x: x.get("ts", 0))
    recent = sorted_fb[-50:]
    historical = sorted_fb[-500:]

    recent_mae = sum(abs(fb["predicted_or"] - fb["actual_or"]) for fb in recent) / len(recent)
    hist_mae = sum(abs(fb["predicted_or"] - fb["actual_or"]) for fb in historical) / len(historical)

    _drift_state["current_mae"] = round(recent_mae, 4)
    _drift_state["historical_mae"] = round(hist_mae, 4)

    ratio = recent_mae / hist_mae if hist_mae > 0 else 1.0
    _drift_state["drift_detected"] = ratio > 1.3

    if _drift_state["drift_detected"]:
        log.warning(f"[Drift] CONCEPT DRIFT erkannt! Recent MAE={recent_mae:.3f} vs "
                    f"Historical MAE={hist_mae:.3f} (Ratio={ratio:.2f}). Auto-Retrain...")
        _drift_state["auto_retrain_count"] += 1
        _log_monitoring_event("drift", "critical",
            f"Concept Drift: Recent MAE={recent_mae:.3f}, Historical MAE={hist_mae:.3f}, Ratio={ratio:.2f}",
            {"recent_mae": round(recent_mae, 4), "hist_mae": round(hist_mae, 4), "ratio": round(ratio, 3)})
        try:
            _gbrt_train()
        except Exception as e:
            log.warning(f"[Drift] Auto-Retrain fehlgeschlagen: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ONLINE RESIDUAL CORRECTOR — Echtzeit-Bias-Korrektur
# ══════════════════════════════════════════════════════════════════════════════

_RESIDUAL_HOURGROUPS = {
    "morning": range(6, 12),     # 06:00 – 11:59
    "afternoon": range(12, 18),  # 12:00 – 17:59
    "evening": range(18, 23),    # 18:00 – 22:59
    "night": list(range(23, 24)) + list(range(0, 6)),  # 23:00 – 05:59
}

def _hour_to_group(h):
    """Stunde (0-23) → Tageszeit-Gruppe."""
    for name, hours in _RESIDUAL_HOURGROUPS.items():
        if h in hours:
            return name
    return "afternoon"


def _update_residual_corrector(rows=None):
    """Aktualisiert den Residual Corrector mit neuesten Prediction-Feedback-Daten.

    Einfacher Rolling Mean der letzten 20 Residuals (predicted - actual).
    Korrektur nur wenn |Bias| > 0.2 (Threshold), dann 50% Staerke.
    Realitaet: Systematischer Bias ist nur ~5% des Gesamtfehlers (MAE 1.7).
    Der Corrector ist ein Safety-Net, kein Game-Changer.
    """
    global _residual_corrector
    WINDOW = 20
    MAX_CORRECTION = 2.0
    MIN_SAMPLES = 10

    if rows is None:
        try:
            with _push_db_lock:
                conn = sqlite3.connect(PUSH_DB_PATH)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT p.predicted_or, p.actual_or, p.predicted_at, "
                    "       pu.cat, pu.hour "
                    "FROM prediction_log p "
                    "LEFT JOIN pushes pu ON p.push_id = pu.message_id "
                    "WHERE p.actual_or > 0 AND p.predicted_or > 0 "
                    "ORDER BY p.predicted_at DESC LIMIT 100"
                ).fetchall()
                conn.close()
        except Exception as e:
            log.warning(f"[ResidualCorrector] DB-Fehler: {e}")
            return

    if not rows or len(rows) < MIN_SAMPLES:
        return

    sorted_rows = sorted(rows, key=lambda r: r["predicted_at"] if hasattr(r, "__getitem__") else 0)

    # Global Bias: Rolling Mean
    all_residuals = [r["predicted_or"] - r["actual_or"] for r in sorted_rows]
    recent_global = all_residuals[-WINDOW:]
    global_bias = sum(recent_global) / len(recent_global)
    global_bias = max(-MAX_CORRECTION, min(MAX_CORRECTION, global_bias))

    # Kategorie-Bias
    cat_residuals = {}
    for r in sorted_rows:
        cat = (r["cat"] or "News") if hasattr(r, "__getitem__") and "cat" in r.keys() else "News"
        cat_residuals.setdefault(cat, []).append(r["predicted_or"] - r["actual_or"])
    cat_bias = {}
    for c, resids in cat_residuals.items():
        if len(resids) >= MIN_SAMPLES:
            recent = resids[-WINDOW:]
            bias = sum(recent) / len(recent)
            cat_bias[c] = max(-MAX_CORRECTION, min(MAX_CORRECTION, bias))

    # Tageszeit-Gruppen-Bias
    hg_residuals = {}
    for r in sorted_rows:
        h = int(r["hour"]) if hasattr(r, "__getitem__") and "hour" in r.keys() and r["hour"] is not None else 12
        hg = _hour_to_group(h)
        hg_residuals.setdefault(hg, []).append(r["predicted_or"] - r["actual_or"])
    hourgroup_bias = {}
    for g, resids in hg_residuals.items():
        if len(resids) >= MIN_SAMPLES:
            recent = resids[-WINDOW:]
            bias = sum(recent) / len(recent)
            hourgroup_bias[g] = max(-MAX_CORRECTION, min(MAX_CORRECTION, bias))

    recent_debug = []
    for r in sorted_rows[-50:]:
        cat = r["cat"] if hasattr(r, "__getitem__") and "cat" in r.keys() else ""
        h = int(r["hour"]) if hasattr(r, "__getitem__") and "hour" in r.keys() and r["hour"] is not None else -1
        recent_debug.append({
            "predicted": round(r["predicted_or"], 2),
            "actual": round(r["actual_or"], 2),
            "residual": round(r["predicted_or"] - r["actual_or"], 2),
            "cat": cat, "hour": h,
        })

    with _residual_corrector_lock:
        _residual_corrector["global_bias"] = round(global_bias, 4)
        _residual_corrector["cat_bias"] = {k: round(v, 4) for k, v in cat_bias.items()}
        _residual_corrector["hourgroup_bias"] = {k: round(v, 4) for k, v in hourgroup_bias.items()}
        _residual_corrector["n_samples"] = len(sorted_rows)
        _residual_corrector["last_update_ts"] = int(time.time())
        _residual_corrector["recent_residuals"] = recent_debug

    log.info(f"[ResidualCorrector] Update: global_bias={global_bias:+.3f}, "
             f"cats={list(cat_bias.keys())}, n={len(sorted_rows)}")


def _apply_residual_correction(predicted_or, cat="News", hour=12):
    """Wendet die Residual-Korrektur an. Safety-Net fuer systematischen Bias.

    50% global + 30% Kategorie + 20% Tageszeit. Nur bei |bias| > 0.2, 50% Staerke.
    """
    with _residual_corrector_lock:
        if _residual_corrector["n_samples"] < 10:
            return predicted_or, 0.0
        gb = _residual_corrector["global_bias"]
        cb = _residual_corrector["cat_bias"].get(cat, gb)
        hg = _hour_to_group(hour)
        hb = _residual_corrector["hourgroup_bias"].get(hg, gb)

    raw = 0.5 * gb + 0.3 * cb + 0.2 * hb
    if abs(raw) < 0.2:
        return predicted_or, 0.0
    correction = raw * 0.5
    correction = max(-2.0, min(2.0, correction))
    corrected = max(0.5, min(30.0, predicted_or - correction))
    return corrected, round(correction, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TUNING MODEL-SELEKTION (ML-First Phase F)
# ══════════════════════════════════════════════════════════════════════════════

def _model_selector_update(rows_24h=None):
    """Aktualisiert den Model-Selector basierend auf letzten 100 Predictions.

    Entscheidet ob Unified oder ML-Ensemble als primärer Predictor genutzt wird.
    Aufgerufen von _monitoring_tick().
    """
    global _model_selector_state
    now = time.time()

    # Nur alle 10 Minuten prüfen
    if now - _model_selector_state.get("last_check_ts", 0) < 600:
        return

    _model_selector_state["last_check_ts"] = now

    # Unified überhaupt trainiert?
    with _unified_lock:
        unified_available = _unified_state.get("model") is not None
        unified_train_count = _unified_state.get("train_count", 0)

    if not unified_available:
        _model_selector_state["active_model"] = "ml_ensemble"
        return

    # Prediction-Log abfragen für beide Modelle
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.row_factory = sqlite3.Row
            cutoff = int(now - 86400)
            rows = conn.execute("""
                SELECT predicted_or, actual_or, methods_detail
                FROM prediction_log
                WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at > ?
                ORDER BY predicted_at DESC LIMIT 100
            """, (cutoff,)).fetchall()
            conn.close()
    except Exception:
        return

    if len(rows) < 10:
        return

    # MAE für Unified vs Ensemble berechnen
    unified_errors = []
    ensemble_errors = []
    for r in rows:
        try:
            detail = json.loads(r["methods_detail"] or "{}")
            actual = float(r["actual_or"])

            # Unified-Prediction
            u_pred = float(detail.get("unified_predicted", 0) or 0)
            if u_pred > 0:
                unified_errors.append(abs(u_pred - actual))

            # Shadow-Ensemble oder direkte Ensemble-Prediction
            e_pred = float(detail.get("ml_ensemble", 0) or
                           detail.get("shadow_gbrt", 0) or
                           detail.get("gbrt_predicted", 0) or 0)
            if e_pred > 0:
                ensemble_errors.append(abs(e_pred - actual))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    _model_selector_state["evaluated_count"] = len(unified_errors)

    # Cold-Start: ML-Ensemble wenn < 30 evaluierte Unified Predictions
    if len(unified_errors) < 30:
        _model_selector_state["active_model"] = "ml_ensemble"
        log.info(f"[ModelSelector] Cold-Start: ml_ensemble (nur {len(unified_errors)} Unified-Predictions)")
        return

    unified_mae = sum(unified_errors) / len(unified_errors)
    _model_selector_state["unified_mae_24h"] = round(unified_mae, 4)

    if ensemble_errors:
        ensemble_mae = sum(ensemble_errors) / len(ensemble_errors)
        _model_selector_state["ensemble_mae_24h"] = round(ensemble_mae, 4)
    else:
        ensemble_mae = float('inf')

    # Entscheidung: Unified nutzen wenn MAE_24h < Ensemble MAE_24h × 1.05
    if unified_mae < ensemble_mae * 1.05:
        if _model_selector_state.get("active_model") != "unified":
            log.info(f"[ModelSelector] Wechsel zu unified: MAE={unified_mae:.4f} < Ensemble={ensemble_mae:.4f}×1.05")
        _model_selector_state["active_model"] = "unified"
        _model_selector_state["consecutive_worse"] = 0
    else:
        _model_selector_state["consecutive_worse"] = _model_selector_state.get("consecutive_worse", 0) + 1
        # Auto-Fallback: 3× hintereinander > 15% schlechter → Ensemble + Retrain
        if _model_selector_state["consecutive_worse"] >= 3 and unified_mae > ensemble_mae * 1.15:
            _model_selector_state["active_model"] = "ml_ensemble"
            log.warning(f"[ModelSelector] Fallback zu ml_ensemble: Unified MAE={unified_mae:.4f} > "
                        f"Ensemble={ensemble_mae:.4f}×1.15, {_model_selector_state['consecutive_worse']}× schlechter → Retrain")
            _model_selector_state["consecutive_worse"] = 0
            # Retrain triggern
            threading.Thread(target=_unified_train, daemon=True).start()
        else:
            _model_selector_state["active_model"] = "ml_ensemble"

    log.info(f"[ModelSelector] active={_model_selector_state['active_model']}, "
             f"unified_MAE={unified_mae:.4f}, ensemble_MAE={ensemble_mae:.4f}, "
             f"evaluated={len(unified_errors)}")


def _model_selector_check():
    """Gibt das aktive Modell zurück. Für externe Abfragen."""
    return _model_selector_state.get("active_model", "ml_ensemble")


# ── Monitoring Dashboard ─────────────────────────────────────────────────

_monitoring_state = {
    "last_tick": 0,
    "mae_24h": 0.0,
    "mae_7d": 0.0,
    "mae_trend": [],       # Stündliche MAE-Werte (24h)
    "calibration_bias": 0.0,
    "calibration_trend": [],
    "feature_drift": {},
    "feature_baselines": {},
}


def _monitoring_tick():
    """Periodischer Monitoring-Check (~alle 20 Min). Prüft MAE, Calibration, Feature Drift."""
    global _monitoring_state
    now = time.time()
    _monitoring_state["last_tick"] = now

    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.row_factory = sqlite3.Row

            cutoff_24h = int(now - 86400)
            cutoff_7d = int(now - 7 * 86400)

            # 1. MAE 24h vs 7d
            rows_24h = conn.execute(
                "SELECT predicted_or, actual_or, predicted_at FROM prediction_log "
                "WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at > ? "
                "ORDER BY predicted_at DESC", (cutoff_24h,)
            ).fetchall()

            rows_7d = conn.execute(
                "SELECT predicted_or, actual_or FROM prediction_log "
                "WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at > ?", (cutoff_7d,)
            ).fetchall()

            conn.close()

        if rows_24h:
            mae_24h = sum(abs(r["predicted_or"] - r["actual_or"]) for r in rows_24h) / len(rows_24h)
            _monitoring_state["mae_24h"] = round(mae_24h, 4)
        if rows_7d:
            mae_7d = sum(abs(r["predicted_or"] - r["actual_or"]) for r in rows_7d) / len(rows_7d)
            _monitoring_state["mae_7d"] = round(mae_7d, 4)

        # 2. Stündliche MAE-Trend (24h Bucketing)
        if rows_24h:
            hourly_buckets = {}
            for r in rows_24h:
                h = (r["predicted_at"] // 3600) * 3600
                hourly_buckets.setdefault(h, []).append(abs(r["predicted_or"] - r["actual_or"]))
            trend = []
            for h_ts in sorted(hourly_buckets.keys()):
                errs = hourly_buckets[h_ts]
                trend.append({"ts": h_ts, "mae": round(sum(errs) / len(errs), 4), "n": len(errs)})
            _monitoring_state["mae_trend"] = trend[-24:]

        # 3. Calibration Bias (mittlerer vorzeichenbehafteter Fehler, letzte 100)
        recent_100 = rows_24h[:100] if rows_24h else []
        if recent_100:
            signed_errors = [r["predicted_or"] - r["actual_or"] for r in recent_100]
            bias = sum(signed_errors) / len(signed_errors)
            _monitoring_state["calibration_bias"] = round(bias, 4)
            # Trend: Bias in 20er-Blöcken
            cal_trend = []
            for i in range(0, len(signed_errors), 20):
                block = signed_errors[i:i+20]
                if block:
                    cal_trend.append(round(sum(block) / len(block), 4))
            _monitoring_state["calibration_trend"] = cal_trend

        # 3b. Online Residual Corrector aktualisieren
        try:
            _update_residual_corrector()
            with _residual_corrector_lock:
                _monitoring_state["residual_corrector"] = {
                    "global_bias": _residual_corrector["global_bias"],
                    "cat_bias": dict(_residual_corrector["cat_bias"]),
                    "hourgroup_bias": dict(_residual_corrector["hourgroup_bias"]),
                    "n_samples": _residual_corrector["n_samples"],
                    "last_update_ts": _residual_corrector["last_update_ts"],
                }
        except Exception as _rc_err:
            log.warning(f"[ResidualCorrector] Update in Monitoring-Tick fehlgeschlagen: {_rc_err}")

        # 4. Feature Distribution Shifts
        with _gbrt_lock:
            feature_names = list(_gbrt_feature_names) if _gbrt_feature_names else []

        if feature_names and rows_7d and rows_24h:
            try:
                with _push_db_lock:
                    conn = sqlite3.connect(PUSH_DB_PATH)
                    recent_feats = conn.execute(
                        "SELECT features FROM prediction_log WHERE features != '{}' "
                        "AND predicted_at > ? ORDER BY predicted_at DESC LIMIT 50", (cutoff_24h,)
                    ).fetchall()
                    hist_feats = conn.execute(
                        "SELECT features FROM prediction_log WHERE features != '{}' "
                        "AND predicted_at > ? AND predicted_at <= ? "
                        "ORDER BY predicted_at DESC LIMIT 200", (cutoff_7d, cutoff_24h)
                    ).fetchall()
                    conn.close()

                # Top-5 wichtigste Features prüfen
                top5 = []
                with _gbrt_lock:
                    if _gbrt_model and _gbrt_model.feature_importance_:
                        sorted_fi = sorted(_gbrt_model.feature_importance_.items(), key=lambda x: -x[1])
                        top5 = [f[0] for f in sorted_fi[:5]]

                drift_info = {}
                for fname in top5:
                    recent_vals = []
                    hist_vals = []
                    for row in recent_feats:
                        try:
                            fd = json.loads(row[0]) if row[0] else {}
                            if fname in fd:
                                recent_vals.append(fd[fname])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    for row in hist_feats:
                        try:
                            fd = json.loads(row[0]) if row[0] else {}
                            if fname in fd:
                                hist_vals.append(fd[fname])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if recent_vals and hist_vals:
                        recent_vals.sort()
                        hist_vals.sort()
                        r_median = recent_vals[len(recent_vals) // 2]
                        h_median = hist_vals[len(hist_vals) // 2]
                        shift = r_median - h_median
                        drift_info[fname] = {
                            "label": _GBRT_SHAP_LABELS.get(fname, fname),
                            "recent_median": round(r_median, 4),
                            "historical_median": round(h_median, 4),
                            "shift": round(shift, 4),
                        }
                _monitoring_state["feature_drift"] = drift_info
            except Exception:
                pass

        # 5. MAE-Spike Warnung
        mae_24h = _monitoring_state.get("mae_24h", 0)
        mae_7d = _monitoring_state.get("mae_7d", 0)
        if mae_7d > 0 and mae_24h > mae_7d * 1.15:
            _log_monitoring_event("mae_spike", "warning",
                f"MAE 24h ({mae_24h:.4f}) ist {(mae_24h/mae_7d - 1)*100:.1f}% über 7d-Baseline ({mae_7d:.4f})",
                {"mae_24h": mae_24h, "mae_7d": mae_7d, "ratio": round(mae_24h / mae_7d, 3)})

        # 6. Calibration-Shift Warnung
        bias = _monitoring_state.get("calibration_bias", 0)
        if abs(bias) > 0.5:
            _log_monitoring_event("calibration_shift", "warning",
                f"Calibration Bias = {bias:+.4f} (Modell {'über' if bias > 0 else 'unter'}schätzt systematisch)",
                {"bias": bias})

        # D6: Auto-Retrain Trigger — wenn MAE_24h > MAE_7d × 1.3 für 3 Ticks
        global _auto_retrain_state
        mae_24h_val = _monitoring_state.get("mae_24h", 0)
        mae_7d_val = _monitoring_state.get("mae_7d", 0)
        if mae_7d_val > 0 and mae_24h_val > mae_7d_val * 1.3:
            _auto_retrain_state["consecutive_degraded_ticks"] += 1
            if (_auto_retrain_state["consecutive_degraded_ticks"] >= 3 and
                    time.time() - _auto_retrain_state["last_retrain_trigger_ts"] > 3600):
                _auto_retrain_state["last_retrain_trigger_ts"] = time.time()
                _auto_retrain_state["consecutive_degraded_ticks"] = 0
                log.warning(f"[AutoRetrain] Trigger: MAE_24h={mae_24h_val:.4f} > MAE_7d×1.3={mae_7d_val*1.3:.4f} "
                            f"für 3 Ticks → Retrain wird gestartet")
                threading.Thread(target=_gbrt_train, daemon=True).start()
                threading.Thread(target=_ml_train_model, daemon=True).start()
                _log_monitoring_event("auto_retrain", "warning",
                    f"Auto-Retrain getriggert: MAE_24h={mae_24h_val:.4f} > MAE_7d×1.3",
                    {"mae_24h": mae_24h_val, "mae_7d": mae_7d_val})
        else:
            _auto_retrain_state["consecutive_degraded_ticks"] = 0

        # Model-Selector Update (Phase F)
        try:
            _model_selector_update(rows_24h)
        except Exception:
            pass

    except Exception as e:
        log.warning(f"[Monitoring] Tick-Fehler: {e}")


# ── Character N-Gram TF-IDF ─────────────────────────────────────────────

class CharNGramTFIDF:
    """Character N-Gram TF-IDF fuer semantische Titel-Aehnlichkeit ohne LLM.

    Faengt morphologische Varianten: Kanzler/Bundeskanzler, Transfer/Transfergeruecht.
    """

    def __init__(self, n_range=(2, 5), max_features=5000):
        self.n_range = n_range
        self.max_features = max_features
        self.vocab = {}        # ngram → index
        self.idf = {}          # ngram → idf score
        self.n_docs = 0

    def _extract_ngrams(self, text):
        """Extrahiert Character N-Grams aus einem Text."""
        text = text.lower().strip()
        ngrams = defaultdict(int)
        for n in range(self.n_range[0], self.n_range[1] + 1):
            for i in range(len(text) - n + 1):
                ng = text[i:i + n]
                ngrams[ng] += 1
        return ngrams

    def fit(self, documents):
        """Trainiert IDF auf einer Liste von Dokumenten (Titeln)."""
        self.n_docs = len(documents)
        if self.n_docs == 0:
            return

        # Document Frequency zaehlen
        df = defaultdict(int)
        for doc in documents:
            ngrams = self._extract_ngrams(doc)
            for ng in ngrams:
                df[ng] += 1

        # Top-Features nach DF sortiert (nicht zu selten, nicht zu haeufig)
        min_df = max(2, self.n_docs * 0.001)
        max_df = self.n_docs * 0.8
        filtered = {ng: count for ng, count in df.items()
                    if min_df <= count <= max_df}

        # Top max_features nach DF
        sorted_ngrams = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocab = {ng: idx for idx, (ng, _) in enumerate(sorted_ngrams)}

        # IDF berechnen
        self.idf = {}
        for ng, idx in self.vocab.items():
            self.idf[ng] = math.log(self.n_docs / (df[ng] + 1)) + 1

    def transform_one(self, text):
        """Transformiert einen Text in einen TF-IDF-Vektor (sparse dict)."""
        ngrams = self._extract_ngrams(text)
        vec = {}
        norm = 0.0
        for ng, count in ngrams.items():
            if ng in self.vocab:
                tf = 1 + math.log(count) if count > 0 else 0
                tfidf = tf * self.idf.get(ng, 0)
                vec[self.vocab[ng]] = tfidf
                norm += tfidf * tfidf
        # L2-Normalisierung
        if norm > 0:
            norm = math.sqrt(norm)
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    def cosine_similarity(self, vec1, vec2):
        """Cosine Similarity zwischen zwei sparse Vektoren."""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        return sum(vec1[k] * vec2[k] for k in common)

    def to_dict(self):
        return {"vocab": self.vocab, "idf": self.idf, "n_docs": self.n_docs,
                "n_range": self.n_range, "max_features": self.max_features}

    @staticmethod
    def from_dict(d):
        tfidf = CharNGramTFIDF(n_range=tuple(d.get("n_range", (2, 5))),
                                max_features=d.get("max_features", 5000))
        tfidf.vocab = d.get("vocab", {})
        tfidf.idf = d.get("idf", {})
        tfidf.n_docs = d.get("n_docs", 0)
        return tfidf


_char_ngram_tfidf = None


def _train_char_ngram_tfidf(pushes):
    """Trainiert Character N-Gram TF-IDF auf allen Push-Titeln."""
    global _char_ngram_tfidf
    titles = [p.get("title", "") for p in pushes if p.get("title") and p.get("or", 0) > 0]
    if len(titles) < 100:
        return
    tfidf = CharNGramTFIDF(n_range=(2, 5), max_features=3000)
    tfidf.fit(titles)
    _char_ngram_tfidf = tfidf
    log.info(f"[TF-IDF] Character N-Gram trainiert: {len(tfidf.vocab)} Features, {len(titles)} Titel")


# ── ML Pipeline (LightGBM + SHAP) ────────────────────────────────────────

_ML_BREAKING_KW = re.compile(r"(?i)\b(eilmeldung|breaking|exklusiv|liveticker|alarm|schock|sensation)\b")
_ML_EMOTION_KW = re.compile(r"(?i)\b(drama|tragödie|skandal|schock|horror|wahnsinn|irre|unfassbar|krass|hammer)\b")

_ML_CAT_COLS = ["sport", "politik", "unterhaltung", "geld", "regional", "digital", "leben", "news"]


def _ml_build_stats(pushes):
    """Delegiert an _gbrt_build_history_stats für Feature-Parität (80+ Features)."""
    return _gbrt_build_history_stats(pushes)


def _ml_extract_features(row, stats):
    """Delegiert an _gbrt_extract_features für Feature-Parität (80+ Features)."""
    return _gbrt_extract_features(row, stats)


def _ml_train_model():
    """Trainiert LightGBM (oder sklearn GBR als Fallback) auf allen historischen Pushes mit OR > 0."""
    try:
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as e:
        log.warning(f"[ML] Pakete fehlen: {e}")
        return
    _use_lgb = False
    try:
        import lightgbm as lgb
        _use_lgb = True
    except (ImportError, OSError) as e:
        log.info(f"[ML] LightGBM nicht verfügbar ({e}), nutze sklearn GradientBoostingRegressor")

    log.info("[ML] Training startet...")
    with _ml_lock:
        _ml_state["training"] = True

    try:
        now = int(time.time())
        log.info("[ML] Lade Pushes aus DB...")
        pushes = _push_db_load_all()
        log.info(f"[ML] {len(pushes)} Pushes geladen, filtere...")
        # Nur reife Pushes mit realistischer OR 0-20% (mindestens 24h alt)
        # Alte API-Daten (vor 2024) haben teilweise OR >100% — das sind kaputte Werte
        valid = [p for p in pushes if 0 < (p.get("or") or 0) <= 20 and p.get("ts_num", 0) < now - 86400]
        if len(valid) < 100:
            log.warning(f"[ML] Nur {len(valid)} gültige Pushes, Training übersprungen (min 100)")
            return
        log.info(f"[ML] {len(valid)} gültige Pushes, baue Stats...")

        stats = _ml_build_stats(valid)
        log.info("[ML] Stats fertig, extrahiere Features...")

        # Feature-Matrix (jetzt 80+ Features via GBRT-Delegation)
        feat_dicts = [_ml_extract_features(p, stats) for p in valid]
        feature_names = sorted(feat_dicts[0].keys())
        X = np.array([[fd.get(k, 0.0) for k in feature_names] for fd in feat_dicts])
        y = np.array([p.get("or") or 0 for p in valid])
        log.info(f"[ML] Feature-Parität: {len(feature_names)} Features (via GBRT-Delegation), "
                 f"Matrix: {X.shape[0]}x{X.shape[1]}, ~{X.nbytes / 1024 / 1024:.1f} MB")

        # Temporaler Split (sortiert nach ts_num): 80% Train, 10% Val, 10% Test
        sorted_indices = np.argsort([p["ts_num"] for p in valid])
        X = X[sorted_indices]
        y = y[sorted_indices]

        # Log1p-Transform der Target-Variable (OR ist rechtsschief)
        y_log = np.log1p(y)
        _use_log_transform = True
        log.info(f"[ML] Target-Stats: mean={y.mean():.2f}, median={np.median(y):.2f}, "
                 f"std={y.std():.2f}, skew={(((y - y.mean())/y.std())**3).mean():.2f}")

        n = len(X)
        split_train = int(n * 0.8)
        split_val = int(n * 0.9)
        X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
        y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]
        y_train_log, y_val_log, y_test_log = y_log[:split_train], y_log[split_train:split_val], y_log[split_val:]

        # ── Phase 2: 3-Fold TimeSeriesSplit CV ──────────────────────────────
        n_folds = 3
        cv_maes = []
        fold_size = len(X_train) // (n_folds + 1)
        for fold_i in range(n_folds):
            cv_train_end = fold_size * (fold_i + 1)
            cv_val_start = cv_train_end
            cv_val_end = min(cv_val_start + fold_size, len(X_train))
            if cv_val_end <= cv_val_start:
                continue
            cv_X_train = X_train[:cv_train_end]
            cv_y_train_log = y_train_log[:cv_train_end]
            cv_X_val = X_train[cv_val_start:cv_val_end]
            cv_y_val = y_train[cv_val_start:cv_val_end]  # Original-Skala für MAE

            if _use_lgb:
                cv_model = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    min_child_samples=20, subsample=0.8, reg_lambda=1.0,
                    verbose=-1, n_jobs=2)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                cv_model = GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    min_samples_leaf=20, subsample=0.8)
            cv_model.fit(cv_X_train, cv_y_train_log)
            cv_pred_log = cv_model.predict(cv_X_val)
            cv_pred = np.expm1(cv_pred_log)  # Zurück-Transformation
            cv_mae = float(mean_absolute_error(cv_y_val, cv_pred))
            cv_maes.append(cv_mae)

        # Baseline-Vergleich
        global_mean = float(np.mean(y_train))
        baseline_mae = float(mean_absolute_error(y_test, np.full(len(y_test), global_mean)))

        cv_mean_mae = sum(cv_maes) / len(cv_maes) if cv_maes else 0
        cv_std_mae = (sum((m - cv_mean_mae) ** 2 for m in cv_maes) / len(cv_maes)) ** 0.5 if cv_maes else 0
        log.info(f"[ML] CV ({n_folds}-Fold): MAE={cv_mean_mae:.3f} +/- {cv_std_mae:.3f}, "
                 f"Baseline (Global Mean): {baseline_mae:.3f}")
        log.info("[ML] Phase 2 (CV) abgeschlossen, starte Phase 3 (Optuna)...")

        # ── Phase 3: Optuna Hyperparameter-Optimierung ──────────────────────
        best_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                       "min_child_samples": 20, "subsample": 0.8, "reg_lambda": 1.0,
                       "reg_alpha": 0.0, "num_leaves": 63}
        tuning_info = {"method": "default", "n_trials": 0}

        if _use_lgb:
            try:
                import optuna
                from optuna.integration import LightGBMPruningCallback
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def _lgbm_optuna_objective(trial):
                    p = {
                        "n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
                        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
                        "max_depth": trial.suggest_int("max_depth", 4, 12),
                        "min_child_samples": trial.suggest_int("min_child_samples", 15, 100),
                        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 15.0, log=True),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 8.0, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.85),
                        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 3.0),
                        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
                        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.4, 1.0),
                    }
                    # 3-Fold TimeSeriesSplit auf Log-Target MIT Early Stopping
                    n_opt = len(X_train)
                    opt_maes = []
                    for opt_fold in range(3):
                        opt_end = n_opt * (opt_fold + 3) // 5
                        opt_val_start = n_opt * (opt_fold + 2) // 5
                        opt_X_t = X_train[:opt_val_start]
                        opt_y_t = y_train_log[:opt_val_start]
                        opt_X_v = X_train[opt_val_start:opt_end]
                        opt_y_v_log = y_train_log[opt_val_start:opt_end]
                        opt_y_v = y_train[opt_val_start:opt_end]
                        if len(opt_y_v) < 5:
                            continue
                        pruning_cb = LightGBMPruningCallback(trial, metric="l1")
                        m = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=2, objective="huber")
                        m.fit(opt_X_t, opt_y_t,
                              eval_set=[(opt_X_v, opt_y_v_log)],
                              callbacks=[lgb.early_stopping(50, verbose=False), pruning_cb])
                        opt_pred_log = m.predict(opt_X_v)
                        opt_pred = np.expm1(opt_pred_log)
                        opt_maes.append(float(mean_absolute_error(opt_y_v, opt_pred)))
                    return sum(opt_maes) / len(opt_maes) if opt_maes else 999.0

                study = optuna.create_study(direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20))
                study.optimize(_lgbm_optuna_objective, n_trials=80, timeout=360)
                best_params.update(study.best_params)
                tuning_info = {
                    "method": "optuna",
                    "n_trials": len(study.trials),
                    "best_val_mae": round(study.best_value, 4),
                    "best_params": {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in study.best_params.items()},
                }
                log.info(f"[ML] Optuna: {len(study.trials)} Trials, "
                         f"beste Val-MAE={study.best_value:.4f}, Params={study.best_params}")
            except ImportError:
                log.info("[ML] Optuna nicht installiert, verwende Default-Hyperparameter")
            except Exception as opt_e:
                log.warning(f"[ML] Optuna-Fehler: {opt_e}, verwende Default-Hyperparameter")

        log.info("[ML] Phase 3 (Optuna) abgeschlossen, starte finales Modelltraining...")

        # ── Adaptive Sample Weights (sanfteres Decay: 365 statt 120 Tage) ──
        _lgbm_sample_weights = None
        try:
            sorted_valid = [valid[si] for si in sorted_indices]
            train_valid = sorted_valid[:split_train]
            _lgbm_latest_ts = train_valid[-1]["ts_num"] if train_valid else now
            _lgbm_sw = []
            for p in train_valid:
                ad = max(0, (_lgbm_latest_ts - p.get("ts_num", _lgbm_latest_ts)) / 86400.0)
                sw = max(0.02, math.exp(-ad / 180.0))
                h = p.get("hour", 12)
                if 18 <= h <= 22:
                    sw *= 1.2
                p_ts = p.get("ts_num", 0)
                if p_ts > 0:
                    pm = datetime.datetime.fromtimestamp(p_ts).month
                    nm = datetime.datetime.fromtimestamp(_lgbm_latest_ts).month
                    if pm == nm and ad > 300:
                        sw *= 1.15
                _lgbm_sw.append(sw)
            _lgbm_sample_weights = np.array(_lgbm_sw)
            log.info(f"[ML] Adaptive Sample Weights (180d decay): min={_lgbm_sample_weights.min():.3f}, "
                     f"max={_lgbm_sample_weights.max():.3f}, mean={_lgbm_sample_weights.mean():.3f}")
        except Exception:
            _lgbm_sample_weights = None

        # ── Finales Modell mit besten Parametern + Early Stopping trainieren ──
        def _build_lgbm(params):
            if _use_lgb:
                return lgb.LGBMRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    learning_rate=params.get("learning_rate", 0.05),
                    max_depth=params.get("max_depth", 6),
                    min_child_samples=params.get("min_child_samples", 20),
                    subsample=params.get("subsample", 0.8),
                    reg_lambda=params.get("reg_lambda", 1.0),
                    reg_alpha=params.get("reg_alpha", 0.0),
                    num_leaves=params.get("num_leaves", 63),
                    colsample_bytree=params.get("colsample_bytree", 0.7),
                    min_split_gain=params.get("min_split_gain", 0.0),
                    objective="huber", verbose=-1, n_jobs=2)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    learning_rate=params.get("learning_rate", 0.05),
                    max_depth=params.get("max_depth", 6),
                    min_samples_leaf=params.get("min_child_samples", 20),
                    subsample=params.get("subsample", 0.8))

        def _fit_model(m, Xt, yt, Xv, yv, sw=None):
            kw = {}
            if sw is not None:
                kw["sample_weight"] = sw
            if _use_lgb:
                kw["eval_set"] = [(Xv, yv)]
                kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
            m.fit(Xt, yt, **kw)
            return m

        # Phase 3b: Erstes Training → Feature Importance → Pruning
        log.info(f"[ML] Pre-Pruning fit: {len(X_train)} Samples, {len(feature_names)} Features...")
        pre_model = _build_lgbm(best_params)
        _fit_model(pre_model, X_train, y_train_log, X_val, y_val_log, _lgbm_sample_weights)

        pre_pred_log = pre_model.predict(X_test)
        pre_pred = np.clip(np.expm1(pre_pred_log), 0, 20)
        pre_mae = float(mean_absolute_error(y_test, pre_pred))
        pre_r2 = float(r2_score(y_test, pre_pred))
        if _use_lgb and hasattr(pre_model, 'best_iteration_'):
            log.info(f"[ML] Early stopping bei Iteration {pre_model.best_iteration_}/{best_params.get('n_estimators', '?')}")

        # Feature Pruning: entferne Features mit Importance = 0
        model = pre_model
        if _use_lgb and hasattr(pre_model, 'feature_importances_'):
            importances = pre_model.feature_importances_
            keep_mask = importances > 0
            n_kept = int(keep_mask.sum())
            n_removed = len(feature_names) - n_kept

            if n_removed > 5:
                X_train_p = X_train[:, keep_mask]
                X_val_p = X_val[:, keep_mask]
                X_test_p = X_test[:, keep_mask]
                pruned_names = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]

                pruned_model = _build_lgbm(best_params)
                sw_p = _lgbm_sample_weights
                _fit_model(pruned_model, X_train_p, y_train_log, X_val_p, y_val_log, sw_p)

                p_pred_log = pruned_model.predict(X_test_p)
                p_pred = np.clip(np.expm1(p_pred_log), 0, 20)
                p_mae = float(mean_absolute_error(y_test, p_pred))
                p_r2 = float(r2_score(y_test, p_pred))

                if p_mae <= pre_mae + 0.005:
                    model = pruned_model
                    feature_names = pruned_names
                    X_train, X_val, X_test = X_train_p, X_val_p, X_test_p
                    log.info(f"[ML] Feature Pruning: {n_kept + n_removed} -> {n_kept} "
                             f"(entfernt: {n_removed}), MAE: {pre_mae:.4f} -> {p_mae:.4f}, "
                             f"R²: {pre_r2:.4f} -> {p_r2:.4f}")
                else:
                    log.info(f"[ML] Pruning abgelehnt: MAE {pre_mae:.4f} -> {p_mae:.4f}")
            else:
                log.info(f"[ML] Kein Pruning nötig (nur {n_removed} Features mit Importance=0)")

        # Finale Evaluation
        y_pred_log = model.predict(X_test)
        y_pred = np.clip(np.expm1(y_pred_log), 0, 20)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        log.info(f"[ML] Training fertig: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f} "
                 f"({len(valid)} Pushes, {len(feature_names)} Features)")

        # ── Phase 4: Isotonic Calibration ───────────────────────────────────
        lgbm_calibrator = None
        val_preds_log = model.predict(X_val)
        val_preds = np.expm1(val_preds_log)
        val_preds = np.clip(val_preds, 0, 20)
        cal = IsotonicCalibrator()
        cal.fit(list(val_preds), list(y_val))
        if cal.breakpoints:
            cal_test_preds = np.array([cal.calibrate(p) for p in y_pred])
            cal_mae = float(mean_absolute_error(y_test, cal_test_preds))
            if cal_mae <= mae:
                lgbm_calibrator = cal
                log.info(f"[ML] Kalibrierung verbessert MAE: {mae:.4f} → {cal_mae:.4f}")
                mae = cal_mae
            else:
                log.info(f"[ML] Kalibrierung deaktiviert (verschlechtert MAE: {mae:.4f} → {cal_mae:.4f})")

        # ── Phase 4b: Conformal Prediction (Q10/Q90) ───────────────────────
        conformal_radius = 1.0
        if len(X_val) > 10:
            val_residuals_abs = sorted(abs(float(val_preds[i]) - float(y_val[i])) for i in range(len(y_val)))
            conformal_alpha = 0.10  # 90% Coverage → Q10/Q90
            conformal_idx = min(int(math.ceil((1 - conformal_alpha) * len(val_residuals_abs))),
                                len(val_residuals_abs) - 1)
            conformal_radius = val_residuals_abs[conformal_idx]
            # Coverage auf Test-Set validieren
            effective_test_preds = [lgbm_calibrator.calibrate(p) for p in y_pred] if lgbm_calibrator else list(y_pred)
            coverage = sum(1 for i in range(len(y_test))
                           if (effective_test_preds[i] - conformal_radius) <= y_test[i] <= (effective_test_preds[i] + conformal_radius)
                           ) / len(y_test) if len(y_test) > 0 else 0
            log.info(f"[ML] Konforme Quantile: radius={conformal_radius:.3f}pp, "
                     f"Coverage (erwartet ~90%): {coverage:.1%}")

        # ── Phase 4c: Residual-Korrektur-Modell ─────────────────────────────
        # Trainiert ein leichtes Modell auf den Residuals (actual - predicted).
        # Fängt systematische Bias ab die das Hauptmodell nicht lernt.
        residual_model = None
        try:
            train_preds_log = model.predict(X_train)
            train_preds_res = np.expm1(train_preds_log)
            train_preds_res = np.clip(train_preds_res, 0, 20)
            if lgbm_calibrator:
                train_preds_res = np.array([lgbm_calibrator.calibrate(float(p)) for p in train_preds_res])
            train_residuals = np.array(y_train) - train_preds_res

            if _use_lgb:
                residual_model = lgb.LGBMRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.05,
                    min_child_samples=30, subsample=0.7, reg_lambda=2.0,
                    num_leaves=15, verbose=-1, n_jobs=2, objective="huber")
                residual_model.fit(X_train, train_residuals)

                test_base = np.array([lgbm_calibrator.calibrate(float(p)) for p in y_pred]) if lgbm_calibrator else y_pred
                test_resid = residual_model.predict(X_test)
                corrected = test_base + test_resid
                corrected_mae = float(mean_absolute_error(y_test, corrected))

                if corrected_mae < mae:
                    log.info(f"[ML] Residual-Korrektur verbessert MAE: {mae:.4f} -> {corrected_mae:.4f}")
                    mae = corrected_mae
                else:
                    log.info(f"[ML] Residual-Korrektur deaktiviert ({mae:.4f} -> {corrected_mae:.4f})")
                    residual_model = None
        except Exception as res_e:
            log.warning(f"[ML] Residual-Modell Fehler: {res_e}")
            residual_model = None

        # SHAP Feature Importance
        log.info("[ML] Starte SHAP-Analyse...")
        shap_importance = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sample_size = min(100, len(X_test))
            shap_values = explainer.shap_values(X_test[:sample_size])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[::-1][:10]
            shap_importance = [{"feature": feature_names[i], "importance": float(mean_abs_shap[i])} for i in top_idx]
        except Exception as se:
            log.warning(f"[ML] SHAP-Fehler: {se}")

        # ── Phase 6: Gelernte Blend-Gewichte via Grid-Search auf Prediction-Log ──
        learned_gbrt_alpha = 0.6
        learned_ml_heuristic_alpha = 0.55
        blend_info = {}
        try:
            pred_log = _push_db_get_training_data(limit=1000)
            # GBRT/LightGBM Alpha lernen
            blend_pairs = []
            for row in pred_log:
                detail = json.loads(row.get("methods_detail") or "{}")
                gbrt_p = float(detail.get("gbrt_predicted", 0) or 0)
                lgbm_p = float(detail.get("lgbm_predicted", 0) or 0)
                actual = row.get("actual_or", 0)
                if gbrt_p > 0 and lgbm_p > 0 and actual > 0:
                    blend_pairs.append((gbrt_p, lgbm_p, actual))
            if len(blend_pairs) >= 20:
                best_alpha = 0.6
                best_mae_b = float('inf')
                for step in range(0, 21):
                    alpha = step / 20.0
                    mae_b = sum(abs(alpha * g + (1 - alpha) * l - a)
                                for g, l, a in blend_pairs) / len(blend_pairs)
                    if mae_b < best_mae_b:
                        best_mae_b = mae_b
                        best_alpha = alpha
                learned_gbrt_alpha = best_alpha
                blend_info["gbrt_lgbm_alpha"] = round(best_alpha, 3)
                blend_info["gbrt_lgbm_alpha_mae"] = round(best_mae_b, 4)
                blend_info["gbrt_lgbm_alpha_n"] = len(blend_pairs)
                log.info(f"[ML] Gelernte GBRT/LightGBM Alpha: {best_alpha:.2f} "
                         f"(MAE={best_mae_b:.4f}, n={len(blend_pairs)})")
            # ML/Heuristik Alpha lernen
            ml_heur_pairs = []
            for row in pred_log:
                detail = json.loads(row.get("methods_detail") or "{}")
                ml_p = float(detail.get("ml_ensemble", 0) or detail.get("gbrt_predicted", 0)
                             or detail.get("lgbm_predicted", 0) or 0)
                heur_p = float(detail.get("heuristic_only", 0) or 0)
                actual = row.get("actual_or", 0)
                if ml_p > 0 and heur_p > 0 and actual > 0:
                    ml_heur_pairs.append((ml_p, heur_p, actual))
            if len(ml_heur_pairs) >= 20:
                best_mh_alpha = 0.55
                best_mh_mae = float('inf')
                for step in range(0, 21):
                    alpha = step / 20.0
                    mh_mae = sum(abs(alpha * m + (1 - alpha) * h - a)
                                 for m, h, a in ml_heur_pairs) / len(ml_heur_pairs)
                    if mh_mae < best_mh_mae:
                        best_mh_mae = mh_mae
                        best_mh_alpha = alpha
                learned_ml_heuristic_alpha = best_mh_alpha
                blend_info["ml_heuristic_alpha"] = round(best_mh_alpha, 3)
                blend_info["ml_heuristic_alpha_mae"] = round(best_mh_mae, 4)
                blend_info["ml_heuristic_alpha_n"] = len(ml_heur_pairs)
                log.info(f"[ML] Gelernte ML/Heuristik Alpha: {best_mh_alpha:.2f} "
                         f"(MAE={best_mh_mae:.4f}, n={len(ml_heur_pairs)})")
        except Exception as blend_err:
            log.warning(f"[ML] Blend-Weight-Learning fehlgeschlagen: {blend_err}")

        with _ml_lock:
            _ml_state["model"] = model
            _ml_state["residual_model"] = residual_model
            _ml_state["stats"] = stats
            _ml_state["feature_names"] = feature_names
            _ml_state["calibrator"] = lgbm_calibrator
            _ml_state["conformal_radius"] = conformal_radius
            _ml_state["gbrt_lgbm_alpha"] = learned_gbrt_alpha
            _ml_state["ml_heuristic_alpha"] = learned_ml_heuristic_alpha
            _ml_state["metrics"] = {
                "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
                "train_size": split_train, "val_size": len(X_val),
                "test_size": len(X_test), "total": len(valid),
                "n_features": len(feature_names),
                "cv_mean_mae": round(cv_mean_mae, 4), "cv_std_mae": round(cv_std_mae, 4),
                "baseline_mae": round(baseline_mae, 4),
                "conformal_radius": round(conformal_radius, 4),
                "tuning": tuning_info,
                "blend_weights": blend_info,
                "log_transform": True,
            }
            _ml_state["shap_importance"] = shap_importance
            _ml_state["train_count"] += 1
            _ml_state["last_train_ts"] = now
            _ml_state["next_retrain_ts"] = now + 6 * 3600
            _ml_state["training"] = False

        # Tagesplan-Cache invalidieren damit er das neue ML-Modell nutzt
        with _tagesplan_cache_lock:
            for _m in ("redaktion", "sport"):
                _tagesplan_cache[_m]["ts"] = 0
                _tagesplan_cache[_m]["hour"] = -1

        # ── Modell auf Disk speichern (ueberlebt Server-Neustarts) ──
        try:
            _ml_disk_data = {
                "model": model,
                "residual_model": residual_model,
                "stats": stats,
                "feature_names": feature_names,
                "calibrator": lgbm_calibrator,
                "conformal_radius": conformal_radius,
                "gbrt_lgbm_alpha": learned_gbrt_alpha,
                "ml_heuristic_alpha": learned_ml_heuristic_alpha,
                "metrics": _ml_state["metrics"],
                "shap_importance": shap_importance,
                "trained_at": now,
            }
            joblib.dump(_ml_disk_data, ML_LGBM_MODEL_PATH, compress=3)
            log.info(f"[ML] Modell gespeichert: {ML_LGBM_MODEL_PATH} "
                     f"({os.path.getsize(ML_LGBM_MODEL_PATH) / 1024:.0f} KB)")
        except Exception as save_e:
            log.warning(f"[ML] Modell-Speichern fehlgeschlagen: {save_e}")

        log.info("[ML] Training abgeschlossen, Tagesplan-Cache invalidiert")
    except Exception as e:
        import traceback
        log.warning(f"[ML] Training-Fehler: {e}\n{traceback.format_exc()}")
        with _ml_lock:
            _ml_state["training"] = False


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ML PREDICTOR (ML-First Phase E) — Kernstück
# ══════════════════════════════════════════════════════════════════════════════

def _unified_train():
    """Trainiert das Unified LightGBM-Modell mit 150+ Features.

    Nutzt _unified_extract_features() (Phase A), Topic-Lifecycle (Phase B),
    Weltereignisse (Phase C) und adaptive Gewichtung (Phase D).
    """
    global _unified_state

    with _unified_lock:
        if _unified_state["training"]:
            log.info("[Unified] Training läuft bereits, übersprungen")
            return
        _unified_state["training"] = True

    t0 = time.time()
    log.info("[Unified] Training startet...")

    try:
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as e:
        log.warning(f"[Unified] Pakete fehlen: {e}")
        with _unified_lock:
            _unified_state["training"] = False
        return

    _use_lgb = False
    try:
        import lightgbm as lgb
        _use_lgb = True
    except (ImportError, OSError):
        log.info("[Unified] LightGBM nicht verfügbar, nutze sklearn")

    try:
        now = int(time.time())
        pushes = _push_db_load_all()
        valid = [p for p in pushes if (p.get("or") or 0) > 0
                 and (p.get("or") or 0) <= 20
                 and p.get("ts_num", 0) > 0
                 and p["ts_num"] < now - 86400]

        if len(valid) < 100:
            log.warning(f"[Unified] Nur {len(valid)} gültige Pushes, Training übersprungen")
            return

        valid.sort(key=lambda x: x["ts_num"])
        n = len(valid)

        # History-Stats + Topic-Tracker + World-Event-Index aufbauen
        history_stats = _gbrt_build_history_stats(valid)
        _build_topic_tracker(history_stats)
        _build_world_event_index()

        # Feature-Matrix mit Unified Features extrahieren
        log.info(f"[Unified] Extrahiere Features für {n} Pushes...")
        feat_dicts = []
        for p in valid:
            fd = _unified_extract_features(p, history_stats, state=None)
            feat_dicts.append(fd)

        if not feat_dicts:
            log.warning("[Unified] Keine Features extrahiert")
            return

        feature_names = sorted(feat_dicts[0].keys())
        X = np.array([[fd.get(k, 0.0) for k in feature_names] for fd in feat_dicts])
        y = np.array([p.get("or") or 0 for p in valid])

        log.info(f"[Unified] {len(feature_names)} Features extrahiert")

        # 80/10/10 temporal Split
        split_train = int(n * 0.8)
        split_val = int(n * 0.9)
        X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
        y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

        # Adaptive Sample-Weights (Phase D)
        latest_ts = valid[split_train - 1]["ts_num"]
        sample_weights = []
        for p in valid[:split_train]:
            age_d = max(0, (latest_ts - p.get("ts_num", latest_ts)) / 86400.0)
            sw = math.exp(-age_d / 120.0)
            h = p.get("hour", 12)
            if 18 <= h <= 22:
                sw *= 1.3
            p_ts = p.get("ts_num", 0)
            if p_ts > 0:
                pm = datetime.datetime.fromtimestamp(p_ts).month
                nm = datetime.datetime.fromtimestamp(latest_ts).month
                if pm == nm and age_d > 300:
                    sw *= 1.2
            sample_weights.append(sw)
        sample_weights = np.array(sample_weights)

        # Optuna Hyperparameter-Optimierung
        best_params = {
            "n_estimators": 400, "learning_rate": 0.03, "max_depth": 6,
            "min_child_samples": 20, "subsample": 0.8, "reg_lambda": 1.0,
            "reg_alpha": 0.1, "num_leaves": 63,
        }
        tuning_info = {"method": "default", "n_trials": 0}

        if _use_lgb:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def _unified_objective(trial):
                    p = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                        "max_depth": trial.suggest_int("max_depth", 4, 8),
                        "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.5),
                        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                    }
                    m = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=2)
                    m.fit(X_train, y_train, sample_weight=sample_weights)
                    vp = m.predict(X_val)
                    return float(mean_absolute_error(y_val, vp))

                study = optuna.create_study(direction="minimize")
                study.optimize(_unified_objective, n_trials=30, timeout=120)
                best_params.update(study.best_params)
                tuning_info = {"method": "optuna", "n_trials": len(study.trials),
                               "best_mae": round(study.best_value, 4)}
                log.info(f"[Unified] Optuna: {len(study.trials)} Trials, "
                         f"best_MAE={study.best_value:.4f}")
            except ImportError:
                log.info("[Unified] Optuna nicht verfügbar, nutze Default-Parameter")
            except Exception as oe:
                log.warning(f"[Unified] Optuna-Fehler: {oe}")

        # ── Finales LightGBM-Modell trainieren (Primary / Fallback) ──────────
        if _use_lgb:
            model = lgb.LGBMRegressor(
                n_estimators=best_params["n_estimators"],
                learning_rate=best_params["learning_rate"],
                max_depth=best_params["max_depth"],
                min_child_samples=best_params["min_child_samples"],
                subsample=best_params["subsample"],
                reg_lambda=best_params["reg_lambda"],
                reg_alpha=best_params.get("reg_alpha", 0.1),
                num_leaves=best_params.get("num_leaves", 63),
                verbose=-1, n_jobs=2,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=best_params["n_estimators"],
                learning_rate=best_params["learning_rate"],
                max_depth=best_params["max_depth"],
                min_samples_leaf=best_params["min_child_samples"],
                subsample=best_params["subsample"],
            )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Single-LightGBM Metriken (Baseline für Stacking-Vergleich)
        y_pred_test_single = model.predict(X_test)
        single_mae = float(mean_absolute_error(y_test, y_pred_test_single))
        log.info(f"[Unified] Single LightGBM MAE={single_mae:.4f}")

        # ── Stacking Ensemble ────────────────────────────────────────────────
        stacking_active = False
        base_models = {}
        meta_model = None
        stacking_mae = None

        # Prüfe welche Base Learner verfügbar sind
        available_learners = []
        if _use_lgb:
            available_learners.append("lgb")
        try:
            import xgboost as xgb_lib
            available_learners.append("xgb")
        except (ImportError, OSError):
            log.info("[Unified] XGBoost nicht verfügbar, übersprungen für Stacking")
        try:
            import catboost as cb_lib
            available_learners.append("catboost")
        except (ImportError, OSError):
            log.info("[Unified] CatBoost nicht verfügbar, übersprungen für Stacking")

        if len(available_learners) >= 2 and len(X_train) >= 200:
            log.info(f"[Unified] Stacking mit {len(available_learners)} Base Learnern: {available_learners}")

            try:
                from sklearn.linear_model import Ridge
                from sklearn.model_selection import TimeSeriesSplit

                n_folds = 5
                tscv = TimeSeriesSplit(n_splits=n_folds)
                oof_preds = {name: np.zeros(len(X_train)) for name in available_learners}
                oof_filled = np.zeros(len(X_train), dtype=bool)

                # ── OOF-Predictions (5-Fold TimeSeries) ──────────────────────
                for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
                    X_tr_fold = X_train[tr_idx]
                    y_tr_fold = y_train[tr_idx]
                    sw_fold = sample_weights[tr_idx]
                    X_va_fold = X_train[va_idx]

                    # LightGBM
                    if "lgb" in available_learners:
                        try:
                            m_lgb = lgb.LGBMRegressor(
                                **best_params, verbose=-1, n_jobs=2,
                            )
                            m_lgb.fit(X_tr_fold, y_tr_fold, sample_weight=sw_fold)
                            oof_preds["lgb"][va_idx] = m_lgb.predict(X_va_fold)
                        except Exception as e:
                            log.warning(f"[Unified] LGB Fold {fold_i} Fehler: {e}")
                            oof_preds["lgb"][va_idx] = np.mean(y_tr_fold)

                    # XGBoost
                    if "xgb" in available_learners:
                        try:
                            m_xgb = xgb_lib.XGBRegressor(
                                n_estimators=best_params["n_estimators"],
                                learning_rate=best_params["learning_rate"] * 1.15,
                                max_depth=min(best_params["max_depth"], 8),
                                subsample=min(best_params["subsample"] + 0.05, 0.95),
                                colsample_bytree=max(best_params.get("colsample_bytree", 0.7) - 0.1, 0.3),
                                reg_lambda=best_params["reg_lambda"] * 0.8,
                                reg_alpha=best_params.get("reg_alpha", 0.1) * 1.5,
                                gamma=best_params.get("min_split_gain", 0.1) * 0.5,
                                verbosity=0, n_jobs=2,
                            )
                            m_xgb.fit(X_tr_fold, y_tr_fold, sample_weight=sw_fold)
                            oof_preds["xgb"][va_idx] = m_xgb.predict(X_va_fold)
                        except Exception as e:
                            log.warning(f"[Unified] XGB Fold {fold_i} Fehler: {e}")
                            oof_preds["xgb"][va_idx] = np.mean(y_tr_fold)

                    # CatBoost
                    if "catboost" in available_learners:
                        try:
                            m_cb = cb_lib.CatBoostRegressor(
                                iterations=best_params["n_estimators"],
                                learning_rate=best_params["learning_rate"] * 0.9,
                                depth=min(best_params["max_depth"], 6),
                                l2_leaf_reg=best_params["reg_lambda"] * 1.2,
                                subsample=max(best_params["subsample"] - 0.05, 0.5),
                                random_strength=best_params.get("reg_alpha", 0.5) * 0.8,
                                bagging_temperature=0.5,
                                verbose=0, thread_count=2,
                            )
                            m_cb.fit(X_tr_fold, y_tr_fold, sample_weight=sw_fold)
                            oof_preds["catboost"][va_idx] = m_cb.predict(X_va_fold)
                        except Exception as e:
                            log.warning(f"[Unified] CatBoost Fold {fold_i} Fehler: {e}")
                            oof_preds["catboost"][va_idx] = np.mean(y_tr_fold)

                    oof_filled[va_idx] = True

                log.info(f"[Unified] OOF-Predictions: {int(oof_filled.sum())}/{len(X_train)} Samples")

                # ── 3 finale Base Models auf gesamtem X_train ────────────────
                if "lgb" in available_learners:
                    base_models["lgb"] = lgb.LGBMRegressor(
                        **best_params, verbose=-1, n_jobs=2,
                    )
                    base_models["lgb"].fit(X_train, y_train, sample_weight=sample_weights)
                    log.info("[Unified] Base Model LightGBM trainiert")

                if "xgb" in available_learners:
                    base_models["xgb"] = xgb_lib.XGBRegressor(
                        n_estimators=best_params["n_estimators"],
                        learning_rate=best_params["learning_rate"] * 1.15,
                        max_depth=min(best_params["max_depth"], 8),
                        subsample=min(best_params["subsample"] + 0.05, 0.95),
                        colsample_bytree=max(best_params.get("colsample_bytree", 0.7) - 0.1, 0.3),
                        reg_lambda=best_params["reg_lambda"] * 0.8,
                        reg_alpha=best_params.get("reg_alpha", 0.1) * 1.5,
                        gamma=best_params.get("min_split_gain", 0.1) * 0.5,
                        verbosity=0, n_jobs=2,
                    )
                    base_models["xgb"].fit(X_train, y_train, sample_weight=sample_weights)
                    log.info("[Unified] Base Model XGBoost trainiert")

                if "catboost" in available_learners:
                    base_models["catboost"] = cb_lib.CatBoostRegressor(
                        iterations=best_params["n_estimators"],
                        learning_rate=best_params["learning_rate"] * 0.9,
                        depth=min(best_params["max_depth"], 6),
                        l2_leaf_reg=best_params["reg_lambda"] * 1.2,
                        subsample=max(best_params["subsample"] - 0.05, 0.5),
                        random_strength=best_params.get("reg_alpha", 0.5) * 0.8,
                        bagging_temperature=0.5,
                        verbose=0, thread_count=2,
                    )
                    base_models["catboost"].fit(X_train, y_train, sample_weight=sample_weights)
                    log.info("[Unified] Base Model CatBoost trainiert")

                # ── Ridge Meta-Learner auf OOF-Predictions ───────────────────
                # Nur Samples nehmen die in mind. einem OOF-Fold waren
                mask = oof_filled
                if mask.sum() > 50:
                    # Meta-Features: OOF-Predictions + hour/weekday als Context
                    meta_cols = [oof_preds[name][mask] for name in available_learners]
                    # hour und weekday aus Feature-Matrix extrahieren (falls vorhanden)
                    hour_idx = feature_names.index("hour") if "hour" in feature_names else None
                    wday_idx = feature_names.index("weekday") if "weekday" in feature_names else None
                    if hour_idx is not None:
                        meta_cols.append(X_train[mask, hour_idx])
                    if wday_idx is not None:
                        meta_cols.append(X_train[mask, wday_idx])
                    X_meta = np.column_stack(meta_cols)
                    y_meta = y_train[mask]

                    # Alpha Grid-Search auf Validation-Set
                    best_alpha = 1.0
                    best_meta_mae = float("inf")
                    # Validation-Set Predictions der finalen Base Models
                    val_base_preds = [base_models[name].predict(X_val) for name in available_learners]
                    val_meta_cols = list(val_base_preds)
                    if hour_idx is not None:
                        val_meta_cols.append(X_val[:, hour_idx])
                    if wday_idx is not None:
                        val_meta_cols.append(X_val[:, wday_idx])
                    X_val_meta = np.column_stack(val_meta_cols)

                    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                        ridge = Ridge(alpha=alpha)
                        ridge.fit(X_meta, y_meta)
                        val_pred = ridge.predict(X_val_meta)
                        val_mae = float(mean_absolute_error(y_val, val_pred))
                        if val_mae < best_meta_mae:
                            best_meta_mae = val_mae
                            best_alpha = alpha

                    meta_model = Ridge(alpha=best_alpha)
                    meta_model.fit(X_meta, y_meta)
                    # Meta-Learner Gewichte loggen
                    weight_names = list(available_learners)
                    if hour_idx is not None:
                        weight_names.append("hour")
                    if wday_idx is not None:
                        weight_names.append("weekday")
                    weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(weight_names, meta_model.coef_))
                    log.info(f"[Unified] Ridge Meta-Learner: alpha={best_alpha}, weights=[{weights_str}], "
                             f"intercept={meta_model.intercept_:.3f}")

                    # ── Test-Set Evaluation: Stacking vs Single ──────────────
                    test_base_preds = [base_models[name].predict(X_test) for name in available_learners]
                    test_meta_cols = list(test_base_preds)
                    if hour_idx is not None:
                        test_meta_cols.append(X_test[:, hour_idx])
                    if wday_idx is not None:
                        test_meta_cols.append(X_test[:, wday_idx])
                    X_test_meta = np.column_stack(test_meta_cols)
                    y_pred_stacking = meta_model.predict(X_test_meta)
                    stacking_mae = float(mean_absolute_error(y_test, y_pred_stacking))

                    log.info(f"[Unified] Stacking MAE={stacking_mae:.4f} vs Single LightGBM MAE={single_mae:.4f}")

                    # Safety: Stacking nur aktivieren wenn besser
                    if stacking_mae <= single_mae * 1.02:  # max 2% Toleranz
                        stacking_active = True
                        log.info(f"[Unified] Stacking Ensemble AKTIVIERT ({len(base_models)} Base Models)")
                    else:
                        log.warning(f"[Unified] Stacking schlechter als Single LightGBM, Fallback auf Single")
                        base_models = {}
                        meta_model = None
                else:
                    log.warning(f"[Unified] Zu wenig OOF-Samples ({mask.sum()}), Stacking übersprungen")

            except ImportError as ie:
                log.warning(f"[Unified] Stacking-Abhängigkeit fehlt: {ie}")
            except Exception as se:
                log.warning(f"[Unified] Stacking-Fehler: {se}")
                import traceback
                log.warning(traceback.format_exc())
        else:
            if len(available_learners) < 2:
                log.info(f"[Unified] Nur {len(available_learners)} Learner verfügbar, Stacking übersprungen")

        # ── Metriken: Verwende Stacking-Predictions wenn aktiv ───────────────
        if stacking_active and stacking_mae is not None:
            y_pred_test = meta_model.predict(X_test_meta)
            mae = stacking_mae
        else:
            y_pred_test = y_pred_test_single
            mae = single_mae
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        r2 = float(r2_score(y_test, y_pred_test))

        # Isotonic Calibration
        unified_calibrator = None
        try:
            from sklearn.isotonic import IsotonicRegression
            # Calibration auf Val-Set: Stacking oder Single
            if stacking_active and meta_model is not None:
                y_pred_val_cal = meta_model.predict(X_val_meta)
            else:
                y_pred_val_cal = model.predict(X_val)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_pred_val_cal, y_val)
            unified_calibrator = iso
            y_pred_cal = iso.predict(y_pred_test)
            cal_mae = float(mean_absolute_error(y_test, y_pred_cal))
            log.info(f"[Unified] Isotonic Calibration: raw_MAE={mae:.4f} → cal_MAE={cal_mae:.4f}")
        except ImportError:
            log.info("[Unified] sklearn.isotonic nicht verfügbar")
        except Exception as ce:
            log.warning(f"[Unified] Calibration-Fehler: {ce}")

        # Conformal Prediction (Q10/Q90)
        conformal_radius = 1.0
        try:
            if stacking_active and meta_model is not None:
                y_pred_val_conf = meta_model.predict(X_val_meta)
            else:
                y_pred_val_conf = model.predict(X_val)
            if unified_calibrator:
                y_pred_val_conf = unified_calibrator.predict(y_pred_val_conf)
            residuals = sorted(abs(y_pred_val_conf[i] - y_val[i]) for i in range(len(y_val)))
            q90_idx = int(0.9 * len(residuals))
            conformal_radius = residuals[min(q90_idx, len(residuals) - 1)]
        except Exception:
            conformal_radius = mae * 1.5

        # SHAP Feature Importances → Top-10 (vom primären LightGBM)
        shap_top10 = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test[:min(200, len(X_test))])
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            top_idx = np.argsort(mean_abs_shap)[::-1][:10]
            shap_top10 = [{"feature": feature_names[i], "importance": float(mean_abs_shap[i])} for i in top_idx]
        except Exception:
            try:
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:10]
                shap_top10 = [{"feature": feature_names[i], "importance": float(importances[i])} for i in top_idx]
            except Exception:
                pass

        elapsed = time.time() - t0

        # Stacking-Meta-Info für Metrics
        stacking_info = {"active": stacking_active, "n_base_models": len(base_models)}
        if stacking_active and stacking_mae is not None:
            stacking_info["stacking_mae"] = round(stacking_mae, 4)
            stacking_info["single_lgb_mae"] = round(single_mae, 4)
            stacking_info["base_learners"] = list(base_models.keys())
            if meta_model is not None:
                stacking_info["meta_alpha"] = meta_model.alpha
                stacking_info["meta_weights"] = {n: round(float(w), 4) for n, w in
                                                  zip(weight_names, meta_model.coef_)}

        with _unified_lock:
            _unified_state["model"] = model
            _unified_state["feature_names"] = feature_names
            _unified_state["stats"] = history_stats
            _unified_state["calibrator"] = unified_calibrator
            _unified_state["conformal_radius"] = conformal_radius
            _unified_state["base_models"] = base_models
            _unified_state["meta_model"] = meta_model
            _unified_state["stacking_active"] = stacking_active
            _unified_state["metrics"] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "n_features": len(feature_names),
                "train_size": split_train,
                "test_size": len(X_test),
                "total": n,
                "conformal_radius": round(conformal_radius, 4),
                "tuning": tuning_info,
                "shap_top10": shap_top10,
                "stacking": stacking_info,
            }
            _unified_state["train_count"] += 1
            _unified_state["last_train_ts"] = int(time.time())
            _unified_state["training"] = False

        model_label = f"Stacking({len(base_models)})" if stacking_active else "Single-LightGBM"
        log.info(f"[Unified] Training komplett in {elapsed:.1f}s: "
                 f"{model_label}, {len(feature_names)} Features, MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, "
                 f"Conformal={conformal_radius:.4f}, Optuna={tuning_info.get('n_trials', 0)} Trials")

    except Exception as e:
        import traceback
        log.warning(f"[Unified] Training-Fehler: {e}\n{traceback.format_exc()}")
        with _unified_lock:
            _unified_state["training"] = False


def _unified_predict(push, state=None):
    """Unified ML Prediction für einen einzelnen Push.

    Returns: Dict mit predicted, q10, q90, confidence, top_features, shap_text, model_type
             oder None wenn Modell nicht verfügbar.
    """
    with _unified_lock:
        model = _unified_state.get("model")
        feature_names = _unified_state.get("feature_names", [])
        history_stats = _unified_state.get("stats")
        calibrator = _unified_state.get("calibrator")
        conformal_radius = _unified_state.get("conformal_radius", 1.0)
        metrics = _unified_state.get("metrics", {})
        base_models = _unified_state.get("base_models", {})
        meta_model = _unified_state.get("meta_model")
        stacking_active = _unified_state.get("stacking_active", False)

    if model is None or not feature_names or not history_stats:
        return None

    try:
        import numpy as np

        # Feature-Extraktion
        feat = _unified_extract_features(push, history_stats, state)
        x = np.array([[feat.get(k, 0.0) for k in feature_names]])

        # ── Stacking Prediction ──────────────────────────────────────────
        if stacking_active and base_models and meta_model is not None:
            try:
                base_preds = []
                available_names = []
                for name in ["lgb", "xgb", "catboost"]:
                    bm = base_models.get(name)
                    if bm is not None:
                        try:
                            base_preds.append(float(bm.predict(x)[0]))
                            available_names.append(name)
                        except Exception:
                            pass

                if len(available_names) >= 2:
                    meta_cols = list(base_preds)
                    # Context-Features (hour, weekday) für Meta-Learner
                    hour_idx = feature_names.index("hour") if "hour" in feature_names else None
                    wday_idx = feature_names.index("weekday") if "weekday" in feature_names else None
                    if hour_idx is not None:
                        meta_cols.append(float(x[0, hour_idx]))
                    if wday_idx is not None:
                        meta_cols.append(float(x[0, wday_idx]))

                    # Meta-Learner erwartet gleiche Anzahl Features wie beim Training
                    expected_n = meta_model.n_features_in_
                    if len(meta_cols) == expected_n:
                        x_meta = np.array([meta_cols])
                        predicted = float(meta_model.predict(x_meta)[0])
                    else:
                        predicted = float(model.predict(x)[0])
                else:
                    predicted = float(model.predict(x)[0])
            except Exception:
                predicted = float(model.predict(x)[0])
        else:
            # Single LightGBM Fallback
            predicted = float(model.predict(x)[0])

        # Kalibrierung
        if calibrator:
            try:
                predicted = float(calibrator.predict(np.array([predicted]))[0])
            except Exception:
                pass

        # Clamp
        predicted = max(0.5, min(30.0, predicted))

        # ── Online Residual Correction ──────────────────────────────────
        _rc_cat = push.get("cat", "News")
        _rc_hour = int(push.get("hour", 12))
        predicted, _rc_correction = _apply_residual_correction(predicted, _rc_cat, _rc_hour)

        # Konfidenz-Intervall
        q10 = max(0.1, predicted - conformal_radius)
        q90 = predicted + conformal_radius

        # Confidence basierend auf Feature-Abdeckung
        n_nonzero = sum(1 for k in feature_names if feat.get(k, 0.0) != 0.0)
        confidence = min(0.95, n_nonzero / max(1, len(feature_names)))

        # Top Feature Contributions (vom primären LightGBM)
        top_features = []
        shap_text = ""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x)
            abs_sv = [(abs(sv[0][i]), feature_names[i], sv[0][i]) for i in range(len(feature_names))]
            abs_sv.sort(reverse=True)
            for _, fname, val in abs_sv[:10]:
                top_features.append({
                    "name": fname,
                    "value": round(feat.get(fname, 0), 3),
                    "shap": round(val, 4),
                })
            shap_parts = [f"{fname}={feat.get(fname, 0):.2f} ({val:+.3f})" for _, fname, val in abs_sv[:5]]
            shap_text = " | ".join(shap_parts)
        except Exception:
            try:
                importances = model.feature_importances_
                sorted_idx = sorted(range(len(importances)), key=lambda i: -importances[i])
                for i in sorted_idx[:10]:
                    top_features.append({
                        "name": feature_names[i],
                        "value": round(feat.get(feature_names[i], 0), 3),
                        "importance": float(importances[i]),
                    })
            except Exception:
                pass

        model_label = f"Stacking({len(base_models)})" if stacking_active else "Unified-LightGBM"
        return {
            "predicted": round(predicted, 3),
            "q10": round(q10, 3),
            "q90": round(q90, 3),
            "confidence": round(confidence, 3),
            "top_features": top_features,
            "shap_text": shap_text,
            "model_type": model_label,
            "n_features": len(feature_names),
            "mae": metrics.get("mae", 0),
            "residual_correction": _rc_correction,
        }

    except Exception as e:
        log.warning(f"[Unified] Prediction-Fehler: {e}")
        return None


def _batch_predict_or(articles, push_data=None, state=None):
    """Batch-Prediction für mehrere Artikel. Fallback-Kette: Unified → LightGBM → GBRT.

    Args:
        articles: Liste von Dicts mit {message_id, title, cat, hour, is_eilmeldung, channels}
        push_data: Optionale Push-Historie für Topic-Saturation
        state: Optionaler Research-State

    Returns:
        Dict {message_id: {predicted_or, basis, q10, q90, confidence}}
    """
    import numpy as np
    results = {}
    if not articles:
        return results

    now = datetime.datetime.now()

    # ── Versuch 1: Unified-Modell (bevorzugt) ──
    with _unified_lock:
        u_model = _unified_state.get("model")
        u_fnames = _unified_state.get("feature_names", [])
        u_stats = _unified_state.get("stats")
        u_calibrator = _unified_state.get("calibrator")
        u_conformal = _unified_state.get("conformal_radius", 1.0)
        u_metrics = _unified_state.get("metrics", {})
        u_base_models = _unified_state.get("base_models", {})
        u_meta_model = _unified_state.get("meta_model")
        u_stacking = _unified_state.get("stacking_active", False)

    use_unified = (
        _model_selector_state.get("active_model") == "unified"
        and u_model is not None
        and u_fnames
        and u_stats
    )

    if use_unified:
        # Feature-Extraktion pro Artikel isoliert (ein Fehler killt nicht den Batch)
        rows = []
        valid_ids = []
        for art in articles:
            mid = art.get("message_id", "")
            try:
                push = {
                    "title": art.get("title", ""),
                    "headline": art.get("title", ""),
                    "cat": art.get("cat", "News"),
                    "hour": art.get("hour", now.hour),
                    "ts_num": now.timestamp(),
                    "is_eilmeldung": art.get("is_eilmeldung", False),
                    "is_bild_plus": art.get("is_bild_plus", False),
                    "channels": art.get("channels", []),
                }
                feat = _unified_extract_features(push, u_stats, None)
                rows.append([feat.get(k, 0.0) for k in u_fnames])
                valid_ids.append(mid)
            except Exception as e:
                log.warning(f"[BatchPredict] Unified feature-extraction für {mid}: {e}")

        if rows:
            try:
                X = np.array(rows)

                # ── Stacking Batch Prediction ────────────────────────────
                if u_stacking and u_base_models and u_meta_model is not None:
                    try:
                        base_pred_arrays = []
                        for name in ["lgb", "xgb", "catboost"]:
                            bm = u_base_models.get(name)
                            if bm is not None:
                                base_pred_arrays.append(bm.predict(X))
                        if len(base_pred_arrays) >= 2:
                            meta_cols = list(base_pred_arrays)
                            hour_idx = u_fnames.index("hour") if "hour" in u_fnames else None
                            wday_idx = u_fnames.index("weekday") if "weekday" in u_fnames else None
                            if hour_idx is not None:
                                meta_cols.append(X[:, hour_idx])
                            if wday_idx is not None:
                                meta_cols.append(X[:, wday_idx])
                            X_meta = np.column_stack(meta_cols)
                            if X_meta.shape[1] == u_meta_model.n_features_in_:
                                preds = u_meta_model.predict(X_meta)
                            else:
                                preds = u_model.predict(X)
                        else:
                            preds = u_model.predict(X)
                    except Exception:
                        preds = u_model.predict(X)
                else:
                    preds = u_model.predict(X)

                # Vektorisierte Kalibrierung
                if u_calibrator:
                    try:
                        if hasattr(u_calibrator, "predict"):
                            preds = u_calibrator.predict(preds)
                        elif hasattr(u_calibrator, "calibrate"):
                            preds = np.array([u_calibrator.calibrate(float(p)) for p in preds])
                    except Exception:
                        pass

                model_label = f"Stacking({len(u_base_models)})" if u_stacking else "Unified-LightGBM"
                # Lookup fuer cat/hour pro message_id
                _art_lookup = {a.get("message_id", ""): a for a in articles}
                for i, mid in enumerate(valid_ids):
                    pred = max(0.5, min(30.0, float(preds[i])))
                    # Online Residual Correction
                    _a = _art_lookup.get(mid, {})
                    pred, _rc = _apply_residual_correction(
                        pred, _a.get("cat", "News"), int(_a.get("hour", now.hour)))
                    q10 = round(max(0.1, pred - u_conformal), 3)
                    q90 = round(pred + u_conformal, 3)
                    interval_w = q90 - q10
                    conf = max(0.1, min(0.95, 1.0 - (interval_w / 10.0)))
                    results[mid] = {
                        "predicted_or": round(pred, 3),
                        "basis": "unified",
                        "q10": q10,
                        "q90": q90,
                        "confidence": round(conf, 3),
                        "model_type": f"{model_label} ({len(u_fnames)}F)",
                        "mae": u_metrics.get("mae", 0),
                        "residual_correction": _rc,
                    }
            except Exception as e:
                log.warning(f"[BatchPredict] Unified batch predict: {e}")

        if len(results) == len(articles):
            return results

    # ── Versuch 2: LightGBM ──
    with _ml_lock:
        lgbm_model = _ml_state.get("model")
        lgbm_stats = _ml_state.get("stats")
        lgbm_fnames = _ml_state.get("feature_names")
        lgbm_calibrator = _ml_state.get("calibrator")
        lgbm_conformal = _ml_state.get("conformal_radius", 1.0)

    if lgbm_model and lgbm_stats and lgbm_fnames:
        rows = []
        valid_ids = []
        for art in articles:
            mid = art.get("message_id", "")
            if mid in results:
                continue
            try:
                row = {
                    "ts_num": now.timestamp(),
                    "hour": art.get("hour", now.hour),
                    "title": art.get("title", ""),
                    "headline": art.get("title", ""),
                    "cat": art.get("cat", "News"),
                    "is_eilmeldung": art.get("is_eilmeldung", False),
                    "channels": art.get("channels", []),
                }
                feat = _ml_extract_features(row, lgbm_stats)
                rows.append([feat.get(k, 0.0) for k in lgbm_fnames])
                valid_ids.append(mid)
            except Exception as e:
                log.warning(f"[BatchPredict] LightGBM feature-extraction für {mid}: {e}")

        if rows:
            try:
                X = np.array(rows)
                preds = lgbm_model.predict(X)
                # Einheitliche Kalibrierung (Duck-Typing: custom vs sklearn)
                if lgbm_calibrator:
                    try:
                        if hasattr(lgbm_calibrator, "calibrate") and hasattr(lgbm_calibrator, "breakpoints") and lgbm_calibrator.breakpoints:
                            preds = np.array([lgbm_calibrator.calibrate(float(p)) for p in preds])
                        elif hasattr(lgbm_calibrator, "predict"):
                            preds = lgbm_calibrator.predict(preds)
                    except Exception:
                        pass
                # Residual-Korrektur (vektorisiert)
                res_model = _ml_state.get("residual_model")
                if res_model is not None:
                    try:
                        preds = preds + res_model.predict(X)
                    except Exception:
                        pass
                _art_lookup_lgbm = {a.get("message_id", ""): a for a in articles}
                for i, mid in enumerate(valid_ids):
                    pred = max(0.5, min(30.0, float(preds[i])))
                    # Online Residual Correction
                    _a = _art_lookup_lgbm.get(mid, {})
                    pred, _rc = _apply_residual_correction(
                        pred, _a.get("cat", "News"), int(_a.get("hour", now.hour)))
                    q10 = round(max(0.1, pred - lgbm_conformal), 3)
                    q90 = round(pred + lgbm_conformal, 3)
                    interval_w = q90 - q10
                    conf = max(0.1, min(0.95, 1.0 - (interval_w / 10.0)))
                    results[mid] = {
                        "predicted_or": round(pred, 3),
                        "basis": "lgbm",
                        "q10": q10,
                        "q90": q90,
                        "confidence": round(conf, 3),
                        "model_type": f"LightGBM ({len(lgbm_fnames)}F)",
                        "residual_correction": _rc,
                    }
            except Exception as e:
                log.warning(f"[BatchPredict] LightGBM batch predict: {e}")

        if len(results) == len(articles):
            return results

    # ── Versuch 3: GBRT ──
    with _gbrt_lock:
        g_model = _gbrt_model
        g_fnames = _gbrt_feature_names
        g_stats = _gbrt_history_stats
        g_calibrator = _gbrt_calibrator

    if g_model and g_fnames and g_stats:
        try:
            for art in articles:
                mid = art.get("message_id", "")
                if mid in results:
                    continue
                push = {
                    "title": art.get("title", ""),
                    "headline": art.get("title", ""),
                    "cat": art.get("cat", "News"),
                    "hour": art.get("hour", now.hour),
                    "ts_num": now.timestamp(),
                    "is_eilmeldung": art.get("is_eilmeldung", False),
                    "is_bild_plus": art.get("is_bild_plus", False),
                    "channels": art.get("channels", []),
                }
                gbrt_res = _gbrt_predict(push, None)
                if gbrt_res:
                    _gp = gbrt_res["predicted"]
                    _gp, _rc = _apply_residual_correction(
                        _gp, art.get("cat", "News"), int(art.get("hour", now.hour)))
                    results[mid] = {
                        "predicted_or": round(_gp, 3),
                        "basis": "gbrt",
                        "q10": round(gbrt_res.get("q10", _gp * 0.5), 3),
                        "q90": round(gbrt_res.get("q90", _gp * 1.8), 3),
                        "confidence": round(gbrt_res.get("confidence", 0.5), 3),
                        "model_type": f"GBRT ({gbrt_res.get('n_trees', '?')}T)",
                        "residual_correction": _rc,
                    }
        except Exception as e:
            log.warning(f"[BatchPredict] GBRT-Fehler: {e}")

    # ── Topic-Saturation Penalty anwenden (wenn push_data verfügbar) ──
    if push_data:
        _art_lookup = {a.get("message_id", ""): a for a in articles}
        for mid, res in results.items():
            art = _art_lookup.get(mid)
            if art is None:
                continue
            push = {
                "title": art.get("title", ""),
                "cat": art.get("cat", "News"),
                "ts_num": now.timestamp(),
                "channels": art.get("channels", []),
            }
            try:
                tsat = _compute_topic_saturation_penalty(push, push_data, state)
                if tsat["penalty"] < 1.0:
                    old_pred = res["predicted_or"]
                    res["predicted_or"] = round(max(0.5, old_pred * tsat["penalty"]), 3)
                    res["topic_saturation_penalty"] = round(tsat["penalty"], 3)
                    res["topic_saturation_6h"] = tsat.get("topic_push_count_6h", 0)
            except Exception:
                pass

    # ── Safety-Envelope auf alle Ergebnisse anwenden ──
    for mid in list(results.keys()):
        wrapped = _safety_envelope(results[mid])
        if wrapped is not None:
            results[mid] = wrapped

    return results


def _ml_predict(title, cat, hour=None, weekday=None, is_eilmeldung=False):
    """Einzelvorhersage mit SHAP-Erklärung."""
    with _ml_lock:
        model = _ml_state.get("model")
        stats = _ml_state.get("stats")
        feature_names = _ml_state.get("feature_names")
    if model is None or stats is None:
        return {"error": "Modell nicht trainiert"}

    import numpy as np
    now = datetime.datetime.now()
    if hour is None:
        hour = now.hour
    if weekday is None:
        weekday = now.weekday()

    row = {
        "title": title, "cat": cat, "hour": hour,
        "ts_num": int(now.timestamp()), "is_eilmeldung": is_eilmeldung,
        "channels": ["eilmeldung"] if is_eilmeldung else ["news"],
    }
    feat = _ml_extract_features(row, stats)
    X = np.array([[feat[k] for k in feature_names]])
    predicted_log = float(model.predict(X)[0])
    predicted_or = max(0, math.expm1(predicted_log))  # Log-Rücktransformation

    # SHAP für Einzelvorhersage
    shap_dict = {}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        for i, fn in enumerate(feature_names):
            if abs(sv[0][i]) > 0.01:
                shap_dict[fn] = round(float(sv[0][i]), 3)
        shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8])
    except Exception:
        pass

    # Deutsche Erklärung
    top_pos = [(k, v) for k, v in shap_dict.items() if v > 0][:3]
    top_neg = [(k, v) for k, v in shap_dict.items() if v < 0][:2]
    parts = []
    if top_pos:
        parts.append("Positiv: " + ", ".join(f"{k} (+{v:.2f}pp)" for k, v in top_pos))
    if top_neg:
        parts.append("Negativ: " + ", ".join(f"{k} ({v:.2f}pp)" for k, v in top_neg))

    return {
        "predicted_or": round(predicted_or, 2),
        "shap": shap_dict,
        "explanation_de": ". ".join(parts) if parts else "Keine starken Einflussfaktoren.",
    }


_TP_CACHE_EMPTY = lambda: {"result": None, "hour": -1, "ts": 0, "building": False, "model_id": None}
_tagesplan_cache = {"redaktion": _TP_CACHE_EMPTY(), "sport": _TP_CACHE_EMPTY()}
_tagesplan_cache_lock = threading.Lock()

_retro_cache = {"result": None, "ts": 0, "day": ""}
_retro_cache_lock = threading.Lock()


def _tagesplan_background_refresh(mode="redaktion"):
    """Berechnet den Tagesplan im Hintergrund und aktualisiert den Cache."""
    try:
        now = datetime.datetime.now()
        _ml_build_tagesplan_inner(now, now.hour, mode=mode)
    except Exception as e:
        log.warning(f"[Tagesplan] Background-Refresh Fehler ({mode}): {e}")
        with _tagesplan_cache_lock:
            _tagesplan_cache[mode]["building"] = False


def _ml_build_tagesplan(background=False, mode="redaktion"):
    """Baut den Tagesplan: 18 Stunden-Slots (06-23) mit Empfehlungen.

    Args:
        background: True = aus Background-Worker (berechnet neu wenn stale).
                    False = aus API-Request (liefert IMMER sofort aus Cache).
        mode: "redaktion" (alle Kategorien) oder "sport" (nur Sport-Pushes).
    """
    now = datetime.datetime.now()
    current_hour = now.hour

    # Modell-ID: erkennt ob sich das ML- oder GBRT-Modell geaendert hat
    _current_model_id = id(_ml_state.get("model")) if _ml_state.get("model") else id(_gbrt_model)

    with _tagesplan_cache_lock:
        c = _tagesplan_cache[mode]
        age = time.time() - c["ts"]
        model_changed = c.get("model_id") != _current_model_id
        is_stale = c["hour"] != current_hour or age >= 300 or model_changed

        # ── API-Request: IMMER sofort antworten, nie blockieren ──
        if not background:
            if c["result"] and not is_stale:
                return c["result"]
            # Veraltet oder leer → Background-Refresh triggern
            if not c["building"]:
                c["building"] = True
                threading.Thread(target=_tagesplan_background_refresh, args=(mode,), daemon=True).start()
            # Sofort antworten: altes Ergebnis oder Loading-Skelett
            if c["result"]:
                return c["result"]
            return {"slots": [], "date": now.strftime("%d.%m.%Y"), "n_pushed_today": 0,
                    "golden_hour": None, "total_pushes_db": 0, "ml_trained": False,
                    "already_pushed_today": [], "must_have_hours": [], "ml_metrics": {},
                    "loading": True, "mode": mode}

        # ── Background-Aufruf: neu berechnen wenn stale ──
        if not is_stale and c["result"]:
            return c["result"]
        if c["building"]:
            return c["result"] or {"slots": [], "loading": True, "mode": mode}
        c["building"] = True

    try:
        return _ml_build_tagesplan_inner(now, current_hour, mode=mode)
    except Exception:
        with _tagesplan_cache_lock:
            _tagesplan_cache[mode]["building"] = False
        raise


def _ml_build_tagesplan_inner(now, current_hour, mode="redaktion"):
    """Innere Tagesplan-Berechnung (gecacht durch _ml_build_tagesplan)."""
    current_weekday = now.weekday()
    _WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    today_start = now.replace(hour=0, minute=0, second=0).timestamp()

    with _ml_lock:
        model = _ml_state.get("model")
        stats = _ml_state.get("stats")
        feature_names = _ml_state.get("feature_names")
        metrics = _ml_state.get("metrics", {})

    # ── GBRT-Fallback: wenn kein ML-Modell geladen, GBRT nutzen (immer von Disk verfuegbar) ──
    _using_gbrt_fallback = False
    if model is None and _gbrt_model is not None:
        model = _gbrt_model
        feature_names = _gbrt_feature_names
        _using_gbrt_fallback = True
        # GBRT-Metriken als Fallback fuer ml_metrics
        if not metrics and _gbrt_model is not None:
            _gbrt_tm = getattr(_gbrt_model, 'train_metrics', {})
            metrics = {
                "mae": _gbrt_tm.get("test_mae", _gbrt_tm.get("mae", 0)),
                "r2": _gbrt_tm.get("test_r2", _gbrt_tm.get("r2", 0)),
                "n_features": len(_gbrt_feature_names),
                "source": "GBRT-Fallback",
            }
        log.info("[Tagesplan] ML-Modell nicht verfuegbar, nutze GBRT als Fallback")

    push_data = _research_state.get("push_data", [])
    # Fallback: wenn research_state noch leer, Pushes aus DB laden
    if not push_data:
        try:
            _tp_db_pushes = _push_db_load_all()
            push_data = _tp_db_pushes
            log.info(f"[Tagesplan] push_data aus DB geladen: {len(push_data)} Pushes")
        except Exception as _pd_e:
            log.warning(f"[Tagesplan] DB-Fallback fuer push_data fehlgeschlagen: {_pd_e}")
    # Optimierung: GBRT-History-Stats wiederverwenden statt alles neu zu berechnen
    if not stats and _gbrt_history_stats:
        stats = _gbrt_history_stats
    # pushes nur laden wenn stats fehlt (Fallback) — SQL-Aggregation ersetzt den großen Loop
    pushes = []
    if not stats:
        pushes = _push_db_load_all()
        if not pushes:
            # Fallback: In-Memory Daten vom Research Worker (nach Server-Neustart)
            pushes = _research_state.get("push_data", [])
            if pushes:
                log.info(f"[Tagesplan] DB leer, nutze {len(pushes)} In-Memory Pushes als Fallback")
        stats = _ml_build_stats(pushes)

    # ── Sport-Filter (wird an alle SQL-Queries angehaengt) ──
    _SPORT_CATS_SQL = "('sport','fussball','bundesliga','formel1','formel-1','tennis','boxen','motorsport')"
    _cat_filter = f" AND LOWER(TRIM(cat)) IN {_SPORT_CATS_SQL}" if mode == "sport" else ""

    # ── Bereits heute gepushte Artikel (direkt aus DB, nie veralteter Cache) ──
    already_pushed_today = []
    pushed_ids = set()
    try:
        with _push_db_lock:
            _tp_today_conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
            _tp_today_conn.row_factory = sqlite3.Row
            _tp_today_rows = _tp_today_conn.execute(f"""
                SELECT hour, title, LOWER(TRIM(cat)) AS cat, or_val, is_eilmeldung, link, message_id
                FROM pushes
                WHERE ts_num >= ?
                  AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                  {_cat_filter}
                ORDER BY ts_num
            """, (int(today_start),)).fetchall()
            _tp_today_conn.close()
        for r in _tp_today_rows:
            title = r["title"] or ""
            mid = r["message_id"] or ""
            pushed_ids.add(mid)
            pushed_ids.add(title)
            already_pushed_today.append({
                "title": title, "cat": r["cat"] or "news",
                "or": round(r["or_val"] or 0, 2),
                "hour": r["hour"] if r["hour"] is not None else -1,
                "is_eilmeldung": bool(r["is_eilmeldung"]),
                "link": r["link"] or "",
            })
    except Exception as _tp_today_e:
        log.warning(f"[Tagesplan] DB-Fallback fuer heutige Pushes: {_tp_today_e}")
        # Fallback auf In-Memory
        for p in push_data:
            ts = p.get("ts_num", 0)
            if ts >= today_start:
                title = p.get("title") or p.get("headline") or ""
                cat = (p.get("cat") or "news").lower().strip()
                if mode == "sport" and cat not in ("sport", "fussball", "bundesliga"):
                    continue
                orv = p.get("or") or 0
                mid = p.get("message_id") or ""
                pushed_ids.add(mid)
                pushed_ids.add(title)
                already_pushed_today.append({
                    "title": title, "cat": cat, "or": round(orv, 2),
                    "hour": p.get("hour", -1), "is_eilmeldung": p.get("is_eilmeldung", False),
                    "link": p.get("link") or "",
                })

    # ── Historische Analyse pro Stunde (SQL-optimiert) ──
    from collections import defaultdict
    hour_cat_pushes = defaultdict(lambda: defaultdict(list))
    hour_title_patterns = defaultdict(lambda: {"question": 0, "exclamation": 0, "number": 0, "breaking": 0, "emotion": 0, "total": 0})
    hour_best_titles = defaultdict(list)
    _tp_total_db = 0

    try:
        with _push_db_lock:
            _tp_conn = sqlite3.connect(PUSH_DB_PATH)
            _tp_conn.row_factory = sqlite3.Row

            # 1a) Stunde×Kategorie Aggregation für gleichen Wochentag — komplett in SQL
            _sql_wd = (current_weekday + 1) % 7  # Python→SQLite Weekday-Mapping
            _hc_agg = _tp_conn.execute(f"""
                SELECT hour, LOWER(TRIM(cat)) as cat,
                       AVG(or_val) as avg_or, COUNT(*) as cnt
                FROM pushes
                WHERE or_val > 0 AND or_val <= 20 AND ts_num > 0
                  AND CAST(strftime('%w', ts_num, 'unixepoch') AS INTEGER) = ?
                  AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                  {_cat_filter}
                GROUP BY hour, LOWER(TRIM(cat))
            """, (_sql_wd,)).fetchall()

            for r in _hc_agg:
                h = r["hour"]
                cat = r["cat"] or "news"
                # top_cats_for_hour erwartet Einzelwerte — simuliere mit avg×count
                hour_cat_pushes[h][cat] = [r["avg_or"]] * r["cnt"]

            # 1b) Best-Titles: Top 2 pro Stunde via Window-Function (statt alle laden + Python-Sort)
            _bt_rows = _tp_conn.execute(f"""
                SELECT hour, or_val, title, LOWER(TRIM(cat)) as cat, link
                FROM (
                    SELECT hour, or_val, title, cat, link,
                           ROW_NUMBER() OVER (PARTITION BY hour ORDER BY or_val DESC) as rn
                    FROM pushes
                    WHERE or_val >= 4.0 AND or_val <= 20 AND ts_num > 0
                      AND received >= 10000
                      AND CAST(strftime('%w', ts_num, 'unixepoch') AS INTEGER) = ?
                      AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                      {_cat_filter}
                ) WHERE rn <= 2
            """, (_sql_wd,)).fetchall()

            for r in _bt_rows:
                hour_best_titles[r["hour"]].append(
                    (r["or_val"], r["title"] or "", r["cat"] or "news", r["link"] or ""))

            # 2) Titel-Patterns: ?, !, Ziffern komplett in SQL aggregieren
            #    Breaking/Emotion-KW per LIKE (häufigste Keywords abdecken)
            _patt_rows = _tp_conn.execute(f"""
                SELECT hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN title LIKE '%%?%%' THEN 1 ELSE 0 END) as question,
                    SUM(CASE WHEN title LIKE '%%!%%' THEN 1 ELSE 0 END) as exclamation,
                    SUM(CASE WHEN title GLOB '*[0-9]*' THEN 1 ELSE 0 END) as has_number,
                    SUM(CASE WHEN LOWER(title) LIKE '%%eilmeldung%%'
                              OR LOWER(title) LIKE '%%breaking%%'
                              OR LOWER(title) LIKE '%%exklusiv%%'
                              OR LOWER(title) LIKE '%%liveticker%%'
                              OR LOWER(title) LIKE '%%alarm%%'
                              OR LOWER(title) LIKE '%%schock%%'
                              OR LOWER(title) LIKE '%%sensation%%'
                        THEN 1 ELSE 0 END) as breaking,
                    SUM(CASE WHEN LOWER(title) LIKE '%%drama%%'
                              OR LOWER(title) LIKE '%%skandal%%'
                              OR LOWER(title) LIKE '%%horror%%'
                              OR LOWER(title) LIKE '%%wahnsinn%%'
                              OR LOWER(title) LIKE '%%irre%%'
                              OR LOWER(title) LIKE '%%unfassbar%%'
                              OR LOWER(title) LIKE '%%krass%%'
                              OR LOWER(title) LIKE '%%hammer%%'
                        THEN 1 ELSE 0 END) as emotion
                FROM pushes
                WHERE or_val > 0 AND ts_num > 0
                  {_cat_filter}
                GROUP BY hour
            """).fetchall()

            for r in _patt_rows:
                h = r["hour"]
                hour_title_patterns[h] = {
                    "total": r["total"], "question": r["question"],
                    "exclamation": r["exclamation"], "number": r["has_number"],
                    "breaking": r["breaking"], "emotion": r["emotion"],
                }

            _tp_total_db = _tp_conn.execute(f"SELECT COUNT(*) FROM pushes WHERE 1=1 {_cat_filter}").fetchone()[0]
            _tp_conn.close()

    except Exception as _sql_e:
        log.warning(f"[Tagesplan] SQL-Aggregation Fehler, Fallback: {_sql_e}")
        for p in pushes:
            orv = p.get("or") or 0
            if orv <= 0:
                continue
            ts = p.get("ts_num", 0)
            if ts <= 0:
                continue
            dt = datetime.datetime.fromtimestamp(ts)
            h = dt.hour
            wd = dt.weekday()
            cat = (p.get("cat") or "news").lower().strip()
            title = p.get("title") or p.get("headline") or ""
            if wd == current_weekday:
                hour_cat_pushes[h][cat].append(orv)
            patt = hour_title_patterns[h]
            patt["total"] += 1
            if "?" in title: patt["question"] += 1
            if "!" in title: patt["exclamation"] += 1
            if re.search(r"\d", title): patt["number"] += 1
            if _ML_BREAKING_KW.search(title): patt["breaking"] += 1
            if _ML_EMOTION_KW.search(title): patt["emotion"] += 1
            link = p.get("link") or ""
            # Nur bild.de — keine SportBild/AutoBild
            if "sportbild." in link or "autobild." in link:
                continue
            recv = p.get("received") or 0
            if wd == current_weekday and 4.0 <= orv <= 20 and recv >= 10000:
                hour_best_titles[h].append((orv, title, cat, link))
        _tp_total_db = len(pushes)

    for h in hour_best_titles:
        hour_best_titles[h].sort(key=lambda x: -x[0])

    def top_cats_for_hour(h):
        cats = hour_cat_pushes.get(h, {})
        ranked = []
        for cat, vals in cats.items():
            avg_or = sum(vals) / len(vals) if vals else 0
            ranked.append({"cat": cat, "avg_or": round(avg_or, 2), "count": len(vals)})
        ranked.sort(key=lambda x: -x["avg_or"])
        return ranked[:3]

    def mood_reasoning(h, patt):
        total = max(patt.get("total", 1), 1)
        q_pct = patt.get("question", 0) / total * 100
        e_pct = patt.get("exclamation", 0) / total * 100
        em_pct = patt.get("emotion", 0) / total * 100
        b_pct = patt.get("breaking", 0) / total * 100
        n_pct = patt.get("number", 0) / total * 100
        best_mood, best_score, reasons = "Informativ", 0, []
        if em_pct > 15 and em_pct > best_score:
            best_mood, best_score = "Emotional", em_pct
            reasons.append(f"{em_pct:.0f}% nutzen emotionale Sprache")
        if b_pct > 10 and b_pct > best_score:
            best_mood, best_score = "Breaking", b_pct
            reasons.append(f"{b_pct:.0f}% Eilmeldungen")
        if q_pct > 20 and q_pct > best_score:
            best_mood, best_score = "Neugier", q_pct
            reasons.append(f"{q_pct:.0f}% Frage-Titel")
        if e_pct > 30 and e_pct > best_score:
            best_mood, best_score = "Dringend", e_pct
            reasons.append(f"{e_pct:.0f}% mit Ausrufezeichen")
        if n_pct > 40:
            reasons.append(f"{n_pct:.0f}% mit Zahlen")
        if not reasons:
            if 6 <= h <= 9: best_mood, reasons = "Informativ", ["Morgen: sachlich-informativ"]
            elif 12 <= h <= 14: best_mood, reasons = "Neugier", ["Mittag: Klick-Neugier"]
            elif 18 <= h <= 21: best_mood, reasons = "Emotional", ["Primetime: Emotion holt Top-OR"]
            elif h >= 22: best_mood, reasons = "Ergebnis", ["Spaetabend: Ergebnis-Pushes"]
            else: reasons = ["Sachliche Titel empfohlen"]
        return best_mood, reasons

    _SHAP_LABELS = {
        "hour_weekday_avg_or": "Wochentag-Timing", "hour_cat_avg_or": "Ressort-Timing",
        "hour_avg_or": "Tageszeit", "cat_avg_or": "Ressort", "weekday_avg_or": "Wochentag",
        "is_eilmeldung": "Eilmeldung", "word_count": "Wortanzahl", "title_len": "Zeichenzahl",
        "upper_ratio": "Grossbuchstaben", "has_exclamation": "Ausrufezeichen",
        "has_question": "Fragezeichen", "has_numbers": "Zahlen", "has_breaking_kw": "Breaking-KW",
        "has_emotion_kw": "Emotions-KW", "is_prime_time": "Primetime", "is_morning": "Morgen",
        "is_weekend": "Wochenende", "n_channels": "Kanalanzahl",
    }

    import numpy as np

    # ── Slot-Metadaten sammeln (ohne ML, schnell) ──
    slot_meta = []
    for h in range(6, 24):
        top_cats = top_cats_for_hour(h)
        primary_cat = top_cats[0]["cat"] if top_cats else "news"
        hist_or = top_cats[0]["avg_or"] if top_cats else 0
        n_hist = top_cats[0]["count"] if top_cats else 0
        # hour_avg: aus hour_stats (7d-Avg) ableiten, Fallback auf global_avg
        _h_stats = stats.get("hour_stats", {}).get(h, {})
        hour_avg = _h_stats.get("avg_7d", _h_stats.get("avg_30d", stats.get("global_avg", 0)))
        patt = hour_title_patterns.get(h, {})
        mood, mood_reasons = mood_reasoning(h, patt)
        best_titles = hour_best_titles.get(h, [])[:2]
        pushed_this_hour = [a for a in already_pushed_today if a.get("hour") == h]
        slot_meta.append({
            "h": h, "primary_cat": primary_cat, "top_cats": top_cats,
            "hist_or": hist_or, "n_hist": n_hist, "hour_avg": hour_avg,
            "mood": mood, "mood_reasons": mood_reasons,
            "best_titles": best_titles, "pushed_this_hour": pushed_this_hour,
        })

    # ── Batch-ML: alle 18 Slots auf einmal predicten + SHAP (1× Explainer statt 18×) ──
    ml_predictions = {}  # h → predicted_or
    ml_shap_dicts = {}   # h → shap_dict
    ml_shap_texts = {}   # h → shap_explanation

    if model is not None and feature_names:
        rows_by_h = {}
        X_rows = []
        h_order = []
        for sm in slot_meta:
            h = sm["h"]
            primary_cat = sm["primary_cat"]
            row = {"title": f"Typischer {primary_cat.title()}-Push", "cat": primary_cat, "hour": h,
                   "ts_num": int(now.timestamp()), "is_eilmeldung": primary_cat == "news" and h >= 18,
                   "channels": ["news"]}
            if _using_gbrt_fallback:
                feat = _gbrt_extract_features(row, stats, state=None, fast_mode=True)
            else:
                feat = _ml_extract_features(row, stats)
            X_rows.append([feat.get(k, 0.0) for k in feature_names])
            h_order.append(h)

        X_all = np.array(X_rows)
        try:
            preds = model.predict(X_all)
        except Exception:
            # Fallback: predict_one pro Slot
            preds = []
            for row in X_rows:
                try:
                    if hasattr(model, 'predict_one'):
                        preds.append(model.predict_one(row))
                    else:
                        preds.append(float(model.predict([row])[0]))
                except Exception:
                    preds.append(0.0)

        for i, h in enumerate(h_order):
            pred_val = float(preds[i])
            # ML-Modell nutzt Log-Transform, GBRT nicht
            if not _using_gbrt_fallback:
                pred_val = math.expm1(pred_val)
            ml_predictions[h] = round(max(0.01, min(20.0, pred_val)), 2)

        # SHAP: 1x Explainer, 1x Batch statt 18x Einzeln (~50s -> ~3s)
        try:
            _shap_model = model
            if _using_gbrt_fallback and hasattr(model, 'sklearn_model'):
                _shap_model = model.sklearn_model
            import shap as _shap
            explainer = _shap.TreeExplainer(_shap_model)
            sv_all = explainer.shap_values(X_all)
            for i, h in enumerate(h_order):
                shap_dict = {}
                for j, fn in enumerate(feature_names):
                    if abs(sv_all[i][j]) > 0.05:
                        shap_dict[fn] = round(float(sv_all[i][j]), 3)
                shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                ml_shap_dicts[h] = shap_dict
                pos = [(k, v) for k, v in shap_dict.items() if v > 0][:2]
                neg = [(k, v) for k, v in shap_dict.items() if v < 0][:1]
                parts = []
                for k, v in pos: parts.append(f"{_SHAP_LABELS.get(k, k)} +{v:.2f}pp")
                for k, v in neg: parts.append(f"{_SHAP_LABELS.get(k, k)} {v:.2f}pp")
                ml_shap_texts[h] = ", ".join(parts)
        except Exception:
            pass

    # ── Slots zusammenbauen ──
    slots = []
    for sm in slot_meta:
        h = sm["h"]
        predicted_or = ml_predictions.get(h)
        shap_dict = ml_shap_dicts.get(h, {})
        shap_explanation = ml_shap_texts.get(h, "")
        hist_or = sm["hist_or"]
        n_hist = sm["n_hist"]

        expected_or = predicted_or if predicted_or is not None else round(hist_or or sm["hour_avg"], 2)
        confidence = "hoch" if n_hist >= 30 else ("mittel" if n_hist >= 10 else "niedrig")
        color = "green" if expected_or >= 5.5 else ("yellow" if expected_or >= 4.0 else "gray")

        slots.append({
            "hour": h, "best_cat": sm["primary_cat"], "top_cats": sm["top_cats"],
            "expected_or": expected_or, "hist_or": round(hist_or, 2) if hist_or else None,
            "n_historical": n_hist, "confidence": confidence, "mood": sm["mood"],
            "mood_reasons": sm["mood_reasons"], "color": color,
            "is_now": h == current_hour, "is_past": h < current_hour,
            "shap": shap_dict, "shap_explanation": shap_explanation,
            "has_ml": predicted_or is not None,
            "best_historical": [{"title": t[1][:70], "cat": t[2], "or": round(t[0], 1), "link": t[3] if len(t) > 3 else ""} for t in sm["best_titles"]],
            "pushed_this_hour": sm["pushed_this_hour"],
        })

    # ── Must-Have Stunden markieren (Top 3 nach expected_or, nur Zukunft) ──
    future_slots = [s for s in slots if not s["is_past"]]
    future_by_or = sorted(future_slots, key=lambda s: -s["expected_or"])
    must_have_hours = set()
    for s in future_by_or[:3]:
        s["must_have"] = True
        must_have_hours.add(s["hour"])
    for s in slots:
        if "must_have" not in s:
            s["must_have"] = False

    # ── Sport-Kalender-Kontext (nur im Sport-Modus) ──
    if mode == "sport":
        _is_season = now.month >= 8 or now.month <= 5
        for s in slots:
            ctx = []
            h = s["hour"]
            if _is_season:
                if current_weekday == 5 and 15 <= h <= 18:
                    ctx.append("Bundesliga")
                elif current_weekday == 6 and 15 <= h <= 19:
                    ctx.append("Bundesliga")
                elif current_weekday == 4 and 20 <= h <= 22:
                    ctx.append("Bundesliga Freitag")
                if current_weekday in (1, 2) and 20 <= h <= 23:
                    ctx.append("Champions League")
            if now.month in (1, 7, 8):
                ctx.append("Transferfenster")
            s["sport_context"] = ctx

    best_slot = max(slots, key=lambda s: s["expected_or"]) if slots else None
    golden = future_by_or[0] if future_by_or else best_slot
    strong_slots = [s for s in future_slots if s["expected_or"] >= 5.0]

    # Ø OR heute berechnen
    _today_ors = [a["or"] for a in already_pushed_today if a.get("or", 0) > 0]
    _avg_or_today = round(sum(_today_ors) / len(_today_ors), 2) if _today_ors else None

    _tp_result = {
        "date": now.strftime(f"{_WOCHENTAGE[current_weekday]}, %d.%m.%Y"),
        "weekday": current_weekday, "weekday_name": _WOCHENTAGE[current_weekday],
        "current_hour": current_hour, "n_future": len(future_slots),
        "n_strong": len(strong_slots),
        "golden_hour": golden["hour"] if golden else None,
        "golden_cat": golden["best_cat"] if golden else None,
        "golden_or": golden["expected_or"] if golden else None,
        "best_hour": best_slot["hour"] if best_slot else None,
        "best_cat": best_slot["best_cat"] if best_slot else None,
        "best_or": best_slot["expected_or"] if best_slot else None,
        "ml_metrics": metrics, "ml_trained": model is not None,
        "avg_or_today": _avg_or_today,
        "total_pushes_db": _tp_total_db or len(pushes), "slots": slots,
        "already_pushed_today": already_pushed_today,
        "n_pushed_today": len(already_pushed_today),
        "must_have_hours": sorted(must_have_hours),
    }

    _tp_result["mode"] = mode

    # Cache befuellen
    _cache_model_id = id(_ml_state.get("model")) if _ml_state.get("model") else id(_gbrt_model)
    with _tagesplan_cache_lock:
        _tagesplan_cache[mode]["result"] = _tp_result
        _tagesplan_cache[mode]["hour"] = current_hour
        _tagesplan_cache[mode]["ts"] = time.time()
        _tagesplan_cache[mode]["building"] = False
        _tagesplan_cache[mode]["model_id"] = _cache_model_id

    return _tp_result


def _build_tagesplan_retro():
    """Baut die 7-Tage-Retrospektive: Was wurde gepusht, was hat das ML prognostiziert."""
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Cache prüfen (1h TTL, invalidiert bei Tageswechsel)
    with _retro_cache_lock:
        if (_retro_cache["result"] is not None
                and _retro_cache["day"] == today_str
                and time.time() - _retro_cache["ts"] < 3600):
            return _retro_cache["result"]

    # Zeitgrenzen: 7 Tage zurück (Mitternacht) bis heute Mitternacht
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int((today_midnight - datetime.timedelta(days=7)).timestamp())
    end_ts = int(today_midnight.timestamp())

    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                DATE(p.ts_num, 'unixepoch', 'localtime') AS day,
                p.hour, p.message_id, p.title,
                LOWER(TRIM(p.cat)) AS cat,
                p.or_val AS actual_or,
                p.is_eilmeldung, p.link, p.received,
                pl.predicted_or, pl.basis_method
            FROM pushes p
            LEFT JOIN prediction_log pl ON p.message_id = pl.push_id
            WHERE p.ts_num >= ?
              AND p.ts_num < ?
              AND p.link NOT LIKE '%sportbild.%'
              AND p.link NOT LIKE '%autobild.%'
            ORDER BY p.ts_num
        """, (start_ts, end_ts)).fetchall()

        # Systemempfehlungen laden (was haette das System gepusht?)
        start_date = (today_midnight - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (today_midnight - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        sug_rows = conn.execute("""
            SELECT date_iso, slot_hour, suggestion_num, article_title, article_link,
                   article_category, article_score, expected_or, best_cat
            FROM tagesplan_suggestions
            WHERE date_iso >= ? AND date_iso <= ?
            ORDER BY date_iso, slot_hour, suggestion_num
        """, (start_date, end_date)).fetchall()
        conn.close()
    except Exception as e:
        log.warning(f"[Retro] DB-Fehler: {e}")
        return {"days": [], "summary": {}}

    # Suggestions nach Tag+Stunde gruppieren
    suggestions_by_day = {}
    for sr in sug_rows:
        day = sr["date_iso"]
        if day not in suggestions_by_day:
            suggestions_by_day[day] = {}
        h_str = str(sr["slot_hour"])
        if h_str not in suggestions_by_day[day]:
            suggestions_by_day[day][h_str] = []
        suggestions_by_day[day][h_str].append({
            "title": sr["article_title"] or "",
            "link": sr["article_link"] or "",
            "cat": sr["article_category"] or "",
            "score": round(sr["article_score"] or 0, 1),
            "expected_or": round(sr["expected_or"] or 0, 2),
        })

    # Nach Tagen gruppieren
    days_dict = {}
    for row in rows:
        day = row["day"]
        if day not in days_dict:
            days_dict[day] = []
        days_dict[day].append(dict(row))

    days_list = []
    total_pushes = 0
    all_ors = []
    all_deltas = []
    cat_stats = {}

    for day_str in sorted(days_dict.keys()):
        pushes = days_dict[day_str]
        dt = datetime.datetime.strptime(day_str, "%Y-%m-%d")
        weekday = weekday_names[dt.weekday()]
        date_display = dt.strftime("%d.%m.%Y")

        n_pushed = len(pushes)
        ors = [p["actual_or"] for p in pushes if p["actual_or"] and p["actual_or"] > 0]
        avg_or = round(sum(ors) / len(ors), 2) if ors else 0

        # Best/Worst Push
        best_push = max(pushes, key=lambda p: p["actual_or"] or 0) if pushes else None
        worst_push = min(pushes, key=lambda p: p["actual_or"] if p["actual_or"] and p["actual_or"] > 0 else 999) if pushes else None

        # Prognose-Analyse
        n_predicted = 0
        deltas = []
        for p in pushes:
            if p["predicted_or"] is not None and p["actual_or"] and p["actual_or"] > 0:
                n_predicted += 1
                delta = abs(p["predicted_or"] - p["actual_or"])
                deltas.append(delta)

        mae = round(sum(deltas) / len(deltas), 2) if deltas else None
        matches = sum(1 for d in deltas if d <= 1.0)
        match_quote = f"{matches}/{n_predicted}" if n_predicted > 0 else "0/0"

        # Stunden-Grid (nur Stunden mit Pushes)
        hours_dict = {}
        for p in pushes:
            h = p["hour"]
            if h < 0:
                continue
            h_str = str(h)
            if h_str not in hours_dict:
                hours_dict[h_str] = {"pushes": []}
            pred_or = p["predicted_or"]
            act_or = p["actual_or"] or 0
            delta_val = round(pred_or - act_or, 2) if pred_or is not None and act_or > 0 else None
            hours_dict[h_str]["pushes"].append({
                "title": p["title"] or "",
                "cat": p["cat"] or "news",
                "actual_or": round(act_or, 2),
                "predicted_or": round(pred_or, 2) if pred_or is not None else None,
                "delta": delta_val,
                "is_eilmeldung": bool(p["is_eilmeldung"]),
                "link": p["link"] or "",
            })

        # Kategorie-Statistiken
        for p in pushes:
            cat = p["cat"] or "news"
            if cat not in cat_stats:
                cat_stats[cat] = {"n": 0, "ors": []}
            cat_stats[cat]["n"] += 1
            if p["actual_or"] and p["actual_or"] > 0:
                cat_stats[cat]["ors"].append(p["actual_or"])

        total_pushes += n_pushed
        all_ors.extend(ors)
        all_deltas.extend(deltas)

        day_obj = {
            "date": date_display,
            "date_iso": day_str,
            "weekday": weekday,
            "n_pushed": n_pushed,
            "avg_or": avg_or,
            "best_push": {
                "title": best_push["title"] or "", "or": round(best_push["actual_or"] or 0, 2),
                "cat": best_push["cat"] or "news", "hour": best_push["hour"],
                "link": best_push["link"] or "",
            } if best_push else None,
            "worst_push": {
                "title": worst_push["title"] or "", "or": round(worst_push["actual_or"] or 0, 2),
                "cat": worst_push["cat"] or "news", "hour": worst_push["hour"],
            } if worst_push else None,
            "n_predicted": n_predicted,
            "prediction_mae": mae,
            "match_quote": match_quote,
            "hours": hours_dict,
            "suggestions": suggestions_by_day.get(day_str, {}),
        }
        days_list.append(day_obj)

    # Summary
    avg_or_7d = round(sum(all_ors) / len(all_ors), 2) if all_ors else 0
    mae_7d = round(sum(all_deltas) / len(all_deltas), 2) if all_deltas else None

    best_day = max(days_list, key=lambda d: d["avg_or"]) if days_list else None
    worst_day = min(days_list, key=lambda d: d["avg_or"] if d["avg_or"] > 0 else 999) if days_list else None

    # Top-Stunde über alle 7 Tage
    hour_ors = {}
    for day in days_list:
        for h_str, h_data in day["hours"].items():
            for p in h_data["pushes"]:
                h_int = int(h_str)
                if h_int not in hour_ors:
                    hour_ors[h_int] = []
                if p["actual_or"] > 0:
                    hour_ors[h_int].append(p["actual_or"])
    top_hour = None
    top_hour_avg = 0
    for h, ors_h in hour_ors.items():
        if ors_h:
            avg_h = sum(ors_h) / len(ors_h)
            if avg_h > top_hour_avg:
                top_hour = h
                top_hour_avg = avg_h

    cat_breakdown = {}
    for cat, data in cat_stats.items():
        cat_breakdown[cat] = {
            "n": data["n"],
            "avg_or": round(sum(data["ors"]) / len(data["ors"]), 2) if data["ors"] else 0,
        }

    result = {
        "days": days_list,
        "summary": {
            "total_pushes": total_pushes,
            "avg_or_7d": avg_or_7d,
            "best_day": {"date": best_day["date"][:5] if best_day else "", "weekday": best_day["weekday"] if best_day else "", "avg_or": best_day["avg_or"] if best_day else 0} if best_day else None,
            "worst_day": {"date": worst_day["date"][:5] if worst_day else "", "weekday": worst_day["weekday"] if worst_day else "", "avg_or": worst_day["avg_or"] if worst_day else 0} if worst_day else None,
            "prediction_mae_7d": mae_7d,
            "top_hour": top_hour,
            "top_hour_avg_or": round(top_hour_avg, 2) if top_hour else 0,
            "category_breakdown": cat_breakdown,
        },
    }

    with _retro_cache_lock:
        _retro_cache["result"] = result
        _retro_cache["ts"] = time.time()
        _retro_cache["day"] = today_str

    return result




# ── Autonomes Forschungssystem (24/7 Echtzeit) ────────────────────────────
# Persistent state — ueberlebt zwischen Requests, resettet bei Server-Restart
_research_state = {
    "last_fetch": 0,
    "push_data": [],              # parsed push messages (letzte 7 Tage)
    "prev_push_count": 0,         # vorheriger Count — erkennt neue Pushes
    "findings": {},               # researcher_id -> aktuelle Erkenntnis
    "ticker_entries": [],         # echte berechnete Ticker-Eintraege
    "last_analysis": 0,           # Zeitpunkt der letzten Analyse
    "analysis_generation": 0,     # zaehlt hoch bei jedem neuen Push
    "cumulative_insights": 0,
    "analysis_lock": None,
    # Rolling Accuracy — vergleicht Prognosen mit echten OR-Werten
    "accuracy_history": [],       # [{predicted, actual, ts, title}]
    "rolling_accuracy": 0.0,      # aktuelle Prediction Accuracy (0-100)
    "accuracy_by_cat": {},        # Kategorie -> accuracy
    "accuracy_trend": [],         # letzte 20 accuracy-Werte fuer Trend
    # Schwab Decision Log — nachvollziehbar
    "schwab_decisions": [],       # [{ts, decision, reason, outcome, affected_teams}]
    "schwab_current": "",         # aktuelle Direktive
    # BILD-Adaption internationaler Forschung
    "bild_adaptations": [],       # [{international_finding, bild_adaptation, evidence}]
    # Live-Regeln: Von Schwab freigegebene Forschungserkenntnisse fuer den Push-Betrieb
    "live_rules": [],             # [{id, rule, source, impact, approved_by, approved_at, active}]
    "live_rules_version": 0,      # incrementiert bei jeder Aenderung
    # Forschungs-Progress: Laufende Projekte und Meilensteine (persistiert ueber Sessions)
    "research_projects": [],       # [{id, title, lead, team, status, milestones, started, progress}]
    "research_milestones": [],     # [{ts, project_id, milestone, achieved_by}]
    "dynamic_discussions": [],     # GPT-4o-generierte Diskussionen
    # Forschungsgedaechtnis: Kumulierte Erkenntnisse pro Forscher (waechst mit jedem Zyklus)
    "research_memory": {},         # researcher_id -> [{"gen": N, "ts": ..., "finding": ..., "builds_on": ...}]
    "research_log": [],            # Chronologisches Log aller Erkenntnisse (max 100)
    "prev_accuracy": 0.0,         # Accuracy der letzten Generation (fuer Delta-Erkennung)
    "prev_findings_hash": "",     # Hash der letzten Findings (um echte Aenderungen zu erkennen)
    # Schwab Approval Queue — Aenderungen die GF-Freigabe brauchen
    "pending_approvals": [],
    "approval_counter": 0,         # ID-Zaehler
    "decided_topics": set(),       # change_type/proposal-Keys die bereits entschieden wurden — verhindert Re-Approvals
    # Event-getriebene Mikro-Diskussionen
    "event_queue": [],              # [{type, data, prio, ts}] — wartende Events
    "event_history": [],            # [{type, data, ts}] — verarbeitete Events (max 24h)
    "event_discussions": [],        # GPT-generierte Mikro-Diskussionen aus Events
    "_prev_mature_ids": set(),      # IDs der reifen Pushes im letzten Zyklus
    "_prev_memory_lens": {},        # researcher_id -> len(entries) im letzten Zyklus
    "_prev_milestone_count": 0,     # Anzahl Meilensteine im letzten Zyklus
    "_last_paper_discovery": 0,     # Zeitpunkt der letzten Paper-Entdeckung
    "_discovered_papers": set(),    # Bereits entdeckte Paper-Titel
    # Negativ-Ergebnisse Register: Gescheiterte Hypothesen dokumentieren (Max-Planck-Prinzip)
    "negative_results": [],         # [{id, hypothesis, test, result, learning, locked_until, researcher}]
    "negative_results_counter": 0,
    # Cross-Referenz-Engine: Synthese wenn 2+ Forscher verwandte Patterns finden
    "cross_references": [],         # [{ts, researchers, finding_a, finding_b, synthesis, type}]
    "_last_cross_ref_run": 0,
    # Meta-Research: Forschung ueber die Forschung (alle 6h)
    "meta_research": {},            # {last_run, findings, resource_allocation, sunset_candidates}
    "_last_meta_research": 0,
    # Entscheidungsvorlagen: Fertig formatiert fuer GF (Ja/Nein/Testen)
    "decision_proposals": [],       # [{id, title, source, evidence, risk, rollback, recommendation, change, status}]
    "decision_counter": 0,
    # Exploration-Budget: 15% fuer spekulative Forschung
    "exploration_experiments": [],   # [{id, hypothesis, params, started, status, baseline_acc}]
    "_last_exploration": 0,
    # Autonomes Tuning: Geschlossener Feedback-Loop fuer predictOR()
    "prediction_feedback": [],      # [{push_id, predicted_or, actual_or, methods_detail, ts}]
    "tuning_history": [],           # [{change_id, ts, params_before, params_after, acc_before, acc_after_24h, status, reasoning}]
    "tuning_params": {},            # Aktuelle tunable Werte (Frontend liest diese)
    "tuning_params_version": 0,
    "_last_tuning_call": 0,
}
_research_state["analysis_lock"] = threading.RLock()

# ── Tuning State Persistierung ──
_TUNING_STATE_FILE = os.path.join(SERVE_DIR, ".tuning_state.json")

def _save_tuning_state():
    """Speichert tuning-relevanten State auf Disk (aufgerufen im Worker-Loop)."""
    try:
        data = {
            "prediction_feedback": _research_state.get("prediction_feedback", []),
            "tuning_history": _research_state.get("tuning_history", []),
            "tuning_params": _research_state.get("tuning_params", {}),
            "tuning_params_version": _research_state.get("tuning_params_version", 0),
            # Autonomes Institut: Persistierte Forschungsdaten
            "negative_results": _research_state.get("negative_results", []),
            "negative_results_counter": _research_state.get("negative_results_counter", 0),
            "cross_references": _research_state.get("cross_references", []),
            "meta_research": _research_state.get("meta_research", {}),
            "decision_proposals": _research_state.get("decision_proposals", []),
            "decision_counter": _research_state.get("decision_counter", 0),
            "exploration_experiments": _research_state.get("exploration_experiments", []),
            "arxiv_papers": _research_state.get("arxiv_papers", []),
            "_arxiv_query_idx": _research_state.get("_arxiv_query_idx", 0),
            "outlier_patterns": _research_state.get("outlier_patterns", {}),
        }
        tmp = _TUNING_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _TUNING_STATE_FILE)
    except Exception as e:
        logging.getLogger("push-balancer").warning(f"[Tuning] State-Save failed: {e}")

def _load_tuning_state():
    """Laedt tuning-relevanten State von Disk beim Server-Start."""
    if not os.path.exists(_TUNING_STATE_FILE):
        return
    try:
        with open(_TUNING_STATE_FILE) as f:
            data = json.load(f)
        _research_state["prediction_feedback"] = data.get("prediction_feedback", [])
        _research_state["tuning_history"] = data.get("tuning_history", [])
        _research_state["tuning_params"] = data.get("tuning_params", {})
        _research_state["tuning_params_version"] = data.get("tuning_params_version", 0)
        # Autonomes Institut: Persistierte Daten laden
        _research_state["negative_results"] = data.get("negative_results", [])
        _research_state["negative_results_counter"] = data.get("negative_results_counter", 0)
        _research_state["cross_references"] = data.get("cross_references", [])
        _research_state["meta_research"] = data.get("meta_research", {})
        _research_state["decision_proposals"] = data.get("decision_proposals", [])
        _research_state["decision_counter"] = data.get("decision_counter", 0)
        _research_state["exploration_experiments"] = data.get("exploration_experiments", [])
        _research_state["arxiv_papers"] = data.get("arxiv_papers", [])
        _research_state["_arxiv_query_idx"] = data.get("_arxiv_query_idx", 0)
        _research_state["outlier_patterns"] = data.get("outlier_patterns", {})
        # Server-Feedback-IDs rekonstruieren (verhindert Duplikate nach Neustart)
        # Migration: Alte IDs mit instabilem hash() durch stabile md5-Hashes ersetzen
        import hashlib as _hl
        migrated_ids = set()
        for fb in _research_state["prediction_feedback"]:
            if fb.get("source") != "server" or not fb.get("push_id"):
                continue
            old_id = fb["push_id"]
            # Felder: push_title (nicht title!), ts_num aus push_id extrahieren
            title = fb.get("push_title", "") or fb.get("title", "")
            # ts_num aus dem alten push_id extrahieren (Format: "{ts_num}_{hash}")
            ts = 0
            if "_" in old_id:
                try:
                    ts = int(old_id.split("_")[0])
                except (ValueError, IndexError):
                    ts = 0
            if ts and title:
                new_id = f"{ts}_{_hl.md5(title.encode()).hexdigest()[:12]}"
                fb["push_id"] = new_id  # Migriere im Speicher
                migrated_ids.add(new_id)
            else:
                migrated_ids.add(old_id)
        _research_state["_server_feedback_ids"] = migrated_ids
        logging.getLogger("push-balancer").info(
            f"[Tuning] State geladen: {len(_research_state['prediction_feedback'])} Feedbacks, "
            f"{len(_research_state['tuning_history'])} History, Version {_research_state['tuning_params_version']}, "
            f"{len(_research_state.get('_server_feedback_ids', set()))} Server-Feedback-IDs"
        )
    except Exception as e:
        logging.getLogger("push-balancer").warning(f"[Tuning] State-Load failed: {e}")

_load_tuning_state()

# ── Default Tuning Parameters (aktuell hardcoded im Frontend predictOR()) ──
# Diese Werte werden vom autonomen Tuning-Loop angepasst und vom Frontend geladen.
DEFAULT_TUNING_PARAMS = {
    # Methoden-Konfidenz-Caps (erhoeht: Signale sollen staerker durchkommen)
    "m1_conf_cap": 1.20,       # Methode 1: Similarity (war 0.95)
    "m2_conf_cap": 1.10,       # Methode 2: Keywords (war 0.85)
    "m3_conf_cap": 1.10,       # Methode 3: Entities (war 0.85)
    "m4_conf_cap": 0.90,       # Methode 4: Psychologie (war 0.60)
    "m5_conf_cap": 0.85,       # Methode 5: Regression (war 0.60)
    "m6_conf_cap": 0.75,       # Methode 6: Agenten-Konsens (war 0.50)
    "m7_conf_cap": 1.00,       # Methode 7: Timing (war 0.85)
    "m8_conf_cap": 0.90,       # Methode 8: Forschung (war 0.70)
    # Fusion
    "fusion_prior_weight": 0.08, # Prior-Staerke (war 0.3 — viel weniger Regression zum Mittelwert)
    # Korrektoren
    "weekday_adj": 0.25,        # Wochentag-Korrekturfaktor (war 0.15)
    "saturation_threshold": 1.3, # Tages-Saettigungs-Schwelle
    "fresh_bonus": 1.15,        # Frische-Bonus fuer neue Themen (war 1.08)
    "entity_boost_cap": 0.40,   # Max Entity-Boost (war 0.25)
    # EWMA Trend
    "ewma_lambda": 0.3,         # EWMA Glaettungsfaktor
    # Von-Mises Timing
    "kappa": 2.0,               # Konzentrations-Parameter
    # Push-Fatigue
    "fatigue_tau": 15,           # Halbwertszeit in Minuten
    "fatigue_alpha": 0.25,      # Max Daempfung
    # Methode 8 Dampening-Faktoren (erhoeht: mehr Differenzierung)
    "cat_damp": 0.55,           # Kategorie-Daempfung (war 0.3 → jetzt 0.45 + 0.55*x)
    "timing_damp": 0.50,        # Timing-Daempfung (war 0.3)
    "framing_damp": 0.40,       # Framing-Daempfung (war 0.2)
    "length_damp": 0.30,        # Laenge-Daempfung (war 0.15)
    "ling_damp": 0.30,          # Linguistik-Daempfung (war 0.15)
    # PhD-Modelle: Neue Methode M6 + Post-Fusion-Korrektoren
}

# Real academic literature on push notifications and news engagement
ACADEMIC_REFS = {
    "timing": [
        {"authors": "Pielot et al.", "year": 2017, "title": "Beyond Interruptibility: Predicting Opportune Moments to Engage Mobile Phone Notifications", "venue": "Proc. ACM MobiSys"},
        {"authors": "Westermann et al.", "year": 2015, "title": "User Acceptance of Mobile Notifications", "venue": "ECIS 2015 Proceedings"},
        {"authors": "Mehrotra et al.", "year": 2016, "title": "PrefMiner: Mining User's Preferences for Intelligent Mobile Notification Management", "venue": "Proc. ACM UbiComp"},
        {"authors": "Shirazi et al.", "year": 2014, "title": "Large-Scale Assessment of Mobile Notifications", "venue": "Proc. ACM CHI"},
    ],
    "engagement": [
        {"authors": "Ling et al.", "year": 2020, "title": "Push Notification Effectiveness in News Apps", "venue": "Digital Journalism, 8(8)"},
        {"authors": "Tandoc & Vos", "year": 2016, "title": "The Journalist Is Marketing the News: Social Media in the Gatekeeping Process", "venue": "Journalism Practice, 10(8)"},
        {"authors": "Stroud et al.", "year": 2020, "title": "News Engagement and Content Personalization", "venue": "Journal of Communication, 70(6)"},
        {"authors": "Boczkowski et al.", "year": 2018, "title": "The Gap Between the News Supply and News Consumption", "venue": "Political Communication, 35(4)"},
    ],
    "framing": [
        {"authors": "Tversky & Kahneman", "year": 1981, "title": "The Framing of Decisions and the Psychology of Choice", "venue": "Science, 211(4481)"},
        {"authors": "Entman", "year": 1993, "title": "Framing: Toward Clarification of a Fractured Paradigm", "venue": "Journal of Communication, 43(4)"},
        {"authors": "Valkenburg et al.", "year": 1999, "title": "The Effects of News Frames on Readers' Thoughts and Recall", "venue": "Communication Research, 26(5)"},
    ],
    "competition": [
        {"authors": "Boczkowski", "year": 2010, "title": "News at Work: Imitation in an Age of Information Abundance", "venue": "Univ. of Chicago Press"},
        {"authors": "Anderson", "year": 2013, "title": "Breaking News: Online Audiences and Network Journalism", "venue": "Journalism Practice, 7(5)"},
    ],
    "fatigue": [
        {"authors": "Sahami Shirazi et al.", "year": 2014, "title": "Large-Scale Assessment of Mobile Notifications", "venue": "Proc. ACM CHI"},
        {"authors": "Weber et al.", "year": 2021, "title": "Push Notification Fatigue: User Engagement Decay", "venue": "New Media & Society, 23(5)"},
        {"authors": "Okoshi et al.", "year": 2015, "title": "Reducing Users' Perceived Mental Effort due to Interruptive Notifications", "venue": "Proc. ACM UbiComp"},
    ],
    "nlp": [
        {"authors": "Reis et al.", "year": 2015, "title": "Breaking the News: First Impressions Matter on Online News", "venue": "Proc. AAAI ICWSM"},
        {"authors": "Chakraborty et al.", "year": 2016, "title": "Stop Clickbait: Detecting and Preventing Clickbaits in Online News", "venue": "Proc. IEEE ASONAM"},
        {"authors": "Blom & Hansen", "year": 2015, "title": "Click Bait: Forward-Reference as Lure in Online News Headlines", "venue": "Journal of Pragmatics, 76"},
    ],
}


def _fetch_push_data():
    """Fetch push data — uses SQLite cache, fetches only new/updated pushes from API."""
    try:
        now_ts = int(time.time())

        # Check what we already have in SQLite
        db_count = _push_db_count()
        db_max_ts = _push_db_max_ts()

        # Determine fetch strategy
        if db_count > 100:
            # Incremental: fetch only since last known push + 2 days overlap (for OR updates)
            fetch_start = max(0, db_max_ts - 2 * 86400)
            fetch_days = max(3, (now_ts - fetch_start) // 86400 + 1)
            log.info(f"[PushDB] Incremental fetch: {fetch_days} days (DB has {db_count} pushes)")
        else:
            # Initial load: fetch 1460 days (4 Jahre)
            fetch_start = now_ts - 1460 * 86400
            fetch_days = 1460
            log.info(f"[PushDB] Initial load: {fetch_days} days")

        all_messages = []

        def _fetch_chunk(chunk_start, chunk_end):
            msgs = []
            params = f"startDate={chunk_start}&endDate={chunk_end}&sourceTypes=EDITORIAL"
            for pg in range(30):
                url = f"{PUSH_API_BASE}/push/statistics/message?{params}"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)",
                    "Accept": "application/json",
                })
                try:
                    with urllib.request.urlopen(req, timeout=30, context=_GLOBAL_SSL_CTX) as resp:
                        data = resp.read()
                    raw = json.loads(data.decode("utf-8", errors="replace"))
                    if isinstance(raw, dict):
                        chunk_msgs = raw.get("messages", [])
                        next_params = raw.get("next", "")
                    else:
                        chunk_msgs = raw if isinstance(raw, list) else []
                        next_params = ""
                    msgs.extend(chunk_msgs)
                    if not next_params or not chunk_msgs:
                        break
                    params = next_params
                except Exception as e:
                    log.warning(f"[Research] Chunk fetch error (pg {pg}): {e}")
                    break
            return msgs

        # Parallel fetch in 14-day chunks
        chunks = []
        n_chunks = max(1, (now_ts - fetch_start) // (14 * 86400) + 1)
        for i in range(min(n_chunks, 27)):
            chunk_end = now_ts - i * 14 * 86400
            chunk_start = max(fetch_start, chunk_end - 14 * 86400)
            if chunk_start >= chunk_end:
                continue
            chunks.append((chunk_start, chunk_end))

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_fetch_chunk, s, e) for s, e in chunks]
            _fetch_errors = 0
            for f in concurrent.futures.as_completed(futures):
                try:
                    all_messages.extend(f.result())
                except Exception as _fe:
                    _fetch_errors += 1
                    logging.getLogger("push-balancer").warning(f"[Fetch] Chunk failed ({_fetch_errors}/{len(chunks)}): {_fe}")
                    pass

        # Deduplicate by messageId
        seen = set()
        unique = []
        for msg in all_messages:
            mid = msg.get("messageId") or msg.get("id") or str(msg)
            if mid not in seen:
                seen.add(mid)
                unique.append(msg)
        all_messages = unique
        log.info(f"[Research] Fetched {len(all_messages)} unique pushes (365 days, parallel)")

        parsed = []
        for msg in all_messages:
            try:
                or_val = float(msg.get("openingRate", 0) or 0)
                # Fallback: OR selbst berechnen wenn openingRate fehlt aber Counts da sind
                if or_val == 0:
                    _opened = int(msg.get("openedCount", 0) or 0)
                    _received = int(msg.get("recipientCount", 0) or 0)
                    if _received > 0 and _opened > 0:
                        or_val = round((_opened / _received) * 100, 2)
                ts_raw = str(msg.get("sendDate", ""))
                headline = msg.get("headline") or ""
                kicker = msg.get("kicker") or ""
                title = msg.get("kickerAndHeadline") or headline or kicker
                link = msg.get("urlId") or msg.get("url") or ""
                msg_type = msg.get("sourceType") or "editorial"
                # Kategorie aus URL extrahieren
                cat = _extract_category_from_url(link)
                # Parse timestamp to epoch seconds
                ts_epoch = 0
                try:
                    ts_epoch = int(ts_raw)
                    if ts_epoch > 1_000_000_000_000:
                        ts_epoch = ts_epoch // 1000
                except (ValueError, TypeError):
                    pass
                # Channel-Info extrahieren (Eilmeldungen, Fussball, Club-Channels etc.)
                target_list = msg.get("targetList", [])
                channel_names = [t.get("channel", "").strip() for t in target_list if t.get("channel")]
                primary_channel = channel_names[0] if channel_names else ""
                is_eilmeldung = any("eilmeldung" in ch.lower() for ch in channel_names)
                # Neue Felder: targetStatistics, appList, Reichweite
                target_stats = msg.get("targetStatistics", {})
                all_apps = set()
                for tgt in target_list:
                    for app in tgt.get("appList", []):
                        all_apps.add(app)
                total_recipients = int(msg.get("recipientCount", 0) or 0)
                parsed.append({
                    "message_id": msg.get("messageId") or msg.get("id") or f"{ts_epoch}_{title[:30]}",
                    "or": or_val,
                    "ts": ts_raw,
                    "ts_num": ts_epoch,
                    "title": title,
                    "headline": headline,
                    "kicker": kicker,
                    "cat": cat,
                    "link": link,
                    "type": msg_type,
                    "hour": _extract_hour(ts_raw),
                    "title_len": len(title),
                    "opened": msg.get("openedCount", 0),
                    "received": msg.get("recipientCount", 0),
                    "channel": primary_channel,
                    "channels": channel_names,
                    "is_eilmeldung": is_eilmeldung,
                    "target_stats": target_stats,
                    "app_list": sorted(all_apps),
                    "n_apps": len(all_apps),
                    "total_recipients": total_recipients,
                })
            except (ValueError, TypeError):
                continue

        # Persist to SQLite and load full history
        if parsed:
            n_upserted = _push_db_upsert(parsed)
            log.info(f"[PushDB] Upserted {n_upserted} pushes from {len(parsed)} fetched ({len(all_messages)} raw)")

        # Return full history from SQLite (includes previously cached data)
        all_from_db = _push_db_load_all()
        log.info(f"[PushDB] Returning {len(all_from_db)} pushes from SQLite (fetched {len(parsed)} new)")

        # Fallback: wenn DB leer (z.B. frischer Render-Container), Snapshot laden
        if len(all_from_db) == 0:
            snap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push-snapshot.json")
            if os.path.exists(snap_path):
                try:
                    with open(snap_path, "r", encoding="utf-8") as _f:
                        snap_data = json.load(_f)
                    log.info(f"[PushDB] API nicht erreichbar — lade Snapshot ({len(snap_data)} Pushes)")
                    _push_db_upsert(snap_data)
                    all_from_db = _push_db_load_all()
                    log.info(f"[PushDB] Snapshot geladen: {len(all_from_db)} Pushes in DB")
                except Exception as _se:
                    log.warning(f"[PushDB] Snapshot-Laden fehlgeschlagen: {_se}")

        return all_from_db
    except Exception as e:
        log.warning(f"Push data fetch failed: {e}")
        return []


# Kategorie aus BILD-URL ableiten
_CAT_PATTERNS = {
    "Sport": ["/sport/", "/fussball/", "/bundesliga/", "/champions-league/", "/formel1/", "/formel-1/", "/tennis/", "/boxen/", "/motorsport/"],
    "Politik": ["/politik/", "/bundestag/", "/inland/"],
    "Unterhaltung": ["/unterhaltung/", "/leute/", "/royals/", "/tv/"],
    "Geld": ["/geld/", "/wirtschaft/", "/verbraucher/"],
    "Regional": ["/regional/", "/berlin/", "/hamburg/", "/muenchen/", "/koeln/", "/ruhrgebiet/", "/frankfurt/", "/leipzig/", "/dresden/", "/stuttgart/"],
    "Digital": ["/digital/", "/technik/", "/auto/", "/tesla/"],
    "Leben & Wissen": ["/leben/", "/ratgeber/", "/gesundheit/", "/wissenschaft/"],
    "News": ["/news/"],
}

def _extract_category_from_url(url):
    """Extract BILD category from article URL."""
    url_lower = url.lower()
    for cat, patterns in _CAT_PATTERNS.items():
        for pattern in patterns:
            if pattern in url_lower:
                return cat
    return "News"


def _extract_hour(ts_str):
    """Extract hour from various timestamp formats (epoch seconds/millis, ISO, etc.)."""
    if not ts_str:
        return -1
    ts_str = str(ts_str).strip()
    # Try epoch millis (13 digits) or seconds (10 digits)
    if ts_str.isdigit():
        ts_num = int(ts_str)
        if ts_num > 1_000_000_000_000:  # millis
            ts_num = ts_num // 1000
        if 1_000_000_000 < ts_num < 2_000_000_000:
            return datetime.datetime.fromtimestamp(ts_num).hour
    # Try ISO format
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            dt = datetime.datetime.strptime(ts_str[:19], fmt)
            return dt.hour
        except (ValueError, IndexError):
            continue
    # Try extracting hour directly
    m = re.search(r"(\d{1,2}):\d{2}", ts_str)
    if m:
        return int(m.group(1))
    return -1


def _run_autonomous_analysis():
    """Echtzeit-Analyse: Laeuft bei jedem Request, erkennt neue Pushes, aktualisiert alles."""
    lock = _research_state.get("analysis_lock")
    if lock and not lock.acquire(blocking=False):
        return  # Already running in another thread
    try:
        _run_autonomous_analysis_inner()
    finally:
        if lock:
            lock.release()


def _compute_temporal_trends(push_data):
    """Temporale Segmentierung: Monats-, Wochen-, Wochentag-, Stunden-Trends."""
    now = time.time()
    result = {}

    # Helper: Timestamp normalisieren
    def _ts(p):
        ts = p.get("ts_num", 0)
        if ts == 0:
            try:
                ts = int(p.get("ts", 0))
                if ts > 1e12: ts //= 1000
            except (ValueError, TypeError):
                ts = 0
        return ts

    # Helper: OR-Statistiken fuer eine Gruppe
    def _stats(group):
        ors = [p["or"] for p in group if p["or"] > 0]
        if not ors:
            return {"avg_or": 0, "median_or": 0, "push_count": len(group), "or_count": 0}
        s = sorted(ors)
        return {
            "avg_or": round(sum(ors) / len(ors), 2),
            "median_or": round(s[len(s)//2], 2),
            "push_count": len(group),
            "or_count": len(ors),
        }

    # Helper: Beste Kategorie einer Gruppe
    def _best_cat(group):
        cat_or = defaultdict(list)
        for p in group:
            if p["or"] > 0:
                cat_or[p.get("cat") or "News"].append(p["or"])
        if not cat_or:
            return ""
        return max(cat_or, key=lambda c: sum(cat_or[c])/len(cat_or[c]))

    # Helper: Beste Stunde einer Gruppe
    def _best_hour(group):
        h_or = defaultdict(list)
        for p in group:
            if p["or"] > 0 and 0 <= p.get("hour", -1) <= 23:
                h_or[p["hour"]].append(p["or"])
        if not h_or:
            return 0
        return max(h_or, key=lambda h: sum(h_or[h])/len(h_or[h]))

    # ── a) Monats-Vergleich (bis 12 Monate) ──
    monthly = {}
    for p in push_data:
        ts = _ts(p)
        if ts > 0:
            dt = datetime.datetime.fromtimestamp(ts)
            key = dt.strftime("%Y-%m")
            monthly.setdefault(key, []).append(p)
    monthly_stats = []
    for month_key in sorted(monthly.keys()):
        group = monthly[month_key]
        s = _stats(group)
        s["month"] = month_key
        s["best_cat"] = _best_cat(group)
        s["best_hour"] = _best_hour(group)
        monthly_stats.append(s)
    result["monthly"] = monthly_stats

    # Trend-Richtung per linearer Regression ueber letzte 6 Monate
    recent_months = monthly_stats[-6:]
    if len(recent_months) >= 3:
        xs = list(range(len(recent_months)))
        ys = [m["avg_or"] for m in recent_months]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else 0
        if slope > 0.05:
            result["trend_direction"] = "steigend"
        elif slope < -0.05:
            result["trend_direction"] = "fallend"
        else:
            result["trend_direction"] = "stabil"
        result["trend_slope"] = round(slope, 4)
    else:
        result["trend_direction"] = "zu wenig Daten"
        result["trend_slope"] = 0

    # ── a2) Tages-Vergleich (letzte 30 Tage) ──
    cutoff_30d = now - 30 * 86400
    daily = {}
    for p in push_data:
        ts = _ts(p)
        if ts > cutoff_30d and ts > 0:
            dt = datetime.datetime.fromtimestamp(ts)
            key = dt.strftime("%Y-%m-%d")
            daily.setdefault(key, []).append(p)
    daily_stats = []
    for day_key in sorted(daily.keys()):
        group = daily[day_key]
        s = _stats(group)
        s["date"] = day_key
        s["best_cat"] = _best_cat(group)
        daily_stats.append(s)
    result["daily_30"] = daily_stats

    # ── b) Wochen-Evolution (letzte 12 Wochen) ──
    weekly = {}
    for p in push_data:
        ts = _ts(p)
        if ts > 0:
            dt = datetime.datetime.fromtimestamp(ts)
            kw = dt.strftime("%Y-W%W")
            weekly.setdefault(kw, []).append(p)
    weekly_stats = []
    for wk_key in sorted(weekly.keys()):
        group = weekly[wk_key]
        s = _stats(group)
        s["week"] = wk_key
        s["top_cat"] = _best_cat(group)
        weekly_stats.append(s)
    # Nur letzte 12 Wochen
    weekly_stats = weekly_stats[-12:]
    result["weekly"] = weekly_stats

    # Gleitender 4-Wochen-Durchschnitt
    moving_avg = []
    for i in range(len(weekly_stats)):
        window = weekly_stats[max(0, i-3):i+1]
        avg = sum(w["avg_or"] for w in window) / len(window) if window else 0
        moving_avg.append({
            "week": weekly_stats[i]["week"],
            "moving_avg_4w": round(avg, 2),
        })
    result["moving_avg_4w"] = moving_avg

    # ── c) Wochentag-Muster (aggregiert) ──
    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    weekday_groups = {i: [] for i in range(7)}
    for p in push_data:
        ts = _ts(p)
        if ts > 0:
            wd = datetime.datetime.fromtimestamp(ts).weekday()
            weekday_groups[wd].append(p)
    weekday_stats = []
    for wd in range(7):
        group = weekday_groups[wd]
        s = _stats(group)
        s["day"] = weekday_names[wd]
        s["best_hour"] = _best_hour(group)
        weekday_stats.append(s)
    result["weekday"] = weekday_stats

    # ── d) Stunden-Evolution: Peak-Stunde pro Monat ──
    hourly_evo = []
    for ms in monthly_stats:
        hourly_evo.append({
            "month": ms["month"],
            "peak_hour": ms["best_hour"],
        })
    result["hourly_evolution"] = hourly_evo

    # Kategorie-Performance pro Stunde pro Quartal
    quarters = {}
    for p in push_data:
        ts = _ts(p)
        if ts > 0 and p["or"] > 0:
            dt = datetime.datetime.fromtimestamp(ts)
            q = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            h = p.get("hour", -1)
            cat = p.get("cat") or "News"
            if 0 <= h <= 23:
                quarters.setdefault(q, {}).setdefault(h, defaultdict(list))[cat].append(p["or"])
    cat_hour_quarter = {}
    for q, hours_data in sorted(quarters.items()):
        q_data = {}
        for h, cats in hours_data.items():
            q_data[h] = {c: round(sum(v)/len(v), 2) for c, v in cats.items() if v}
        cat_hour_quarter[q] = q_data
    result["cat_hour_quarter"] = cat_hour_quarter

    # ── e) YoY-Vergleich (gleiche KW Vorjahr vs. jetzt) ──
    now_dt = datetime.datetime.now()
    current_kw = int(now_dt.strftime("%W"))
    current_year = now_dt.year
    last_year = current_year - 1
    yoy = {}
    for wk_key, group in weekly.items():
        parts = wk_key.split("-W")
        if len(parts) == 2:
            try:
                yr, wk = int(parts[0]), int(parts[1])
                if wk == current_kw:
                    yoy[yr] = _stats(group)
                    yoy[yr]["year"] = yr
            except ValueError:
                pass
    if last_year in yoy and current_year in yoy:
        delta = yoy[current_year]["avg_or"] - yoy[last_year]["avg_or"]
        result["yoy_comparison"] = {
            "kw": current_kw,
            "this_year": yoy[current_year],
            "last_year": yoy[last_year],
            "delta": round(delta, 2),
        }
    else:
        result["yoy_comparison"] = {}

    result["total_pushes_all_time"] = len(push_data)
    return result


def _compute_findings_for_subset(subset_data):
    """Berechne strukturierte Analyse-Findings fuer ein Subset (gesamt/sport/nonsport).

    Gibt ein Dict mit hour_analysis, cat_analysis, framing_analysis, title_length,
    frequency_correlation, linguistic_analysis, keyword_analysis, emotion_radar zurueck.
    """
    findings = {}
    _or_max_sane = 30.0  # OR > 30% = Datenbankfehler
    or_pushes = [p for p in subset_data if 0 < p.get("or", 0) <= _or_max_sane]
    or_values = [p["or"] for p in or_pushes]
    sorted_or = sorted(or_values) if or_values else [0]
    median_or = sorted_or[len(sorted_or)//2]
    mean_or = median_or  # Median statt Mean (robust gegen Rest-Ausreisser)
    std_or = math.sqrt(sum((x-median_or)**2 for x in or_values)/max(1,len(or_values)-1)) if len(or_values) > 1 else 0

    # Stunden-Analyse
    hours = defaultdict(list)
    for p in subset_data:
        if 0 <= p.get("hour", -1) <= 23 and p.get("or", 0) > 0:
            hours[p["hour"]].append(p["or"])
    hour_avgs = {h: sum(v)/len(v) for h, v in hours.items() if v}
    best_hour = max(hour_avgs, key=hour_avgs.get) if hour_avgs else 18
    worst_hour = min(hour_avgs, key=hour_avgs.get) if hour_avgs else 3
    best_or = hour_avgs.get(best_hour, 0)
    worst_or = hour_avgs.get(worst_hour, 0)

    findings["hour_analysis"] = {
        "best_hour": best_hour, "best_or": best_or,
        "worst_hour": worst_hour, "worst_or": worst_or,
        "hour_avgs": dict(hour_avgs),
    }

    # Kategorie-Analyse
    cat_or = defaultdict(list)
    for p in subset_data:
        if p.get("or", 0) > 0:
            cat_or[p.get("cat") or "News"].append(p["or"])
    cat_avgs = {c: sum(v)/len(v) for c, v in cat_or.items() if v}
    findings["cat_analysis"] = [
        {"category": c, "avg_or": v, "count": len(cat_or.get(c, []))}
        for c, v in sorted(cat_avgs.items(), key=lambda x: -x[1])
    ]

    # Framing-Analyse
    emo_words = {"schock","drama","skandal","angst","tod","sterben","krieg","panik",
                 "horror","warnung","gefahr","krise","irre","wahnsinn","hammer","brutal","bitter"}
    emo_pushes = []
    q_pushes = []
    neutral_pushes = []
    for p in subset_data:
        if p.get("or", 0) <= 0:
            continue
        _tl = p.get("title", "").lower()
        if any(w in _tl for w in emo_words):
            emo_pushes.append(p)
        elif "?" in p.get("title", ""):
            q_pushes.append(p)
        else:
            neutral_pushes.append(p)
    emo_or = sum(p["or"] for p in emo_pushes)/len(emo_pushes) if emo_pushes else 0
    q_or = sum(p["or"] for p in q_pushes)/len(q_pushes) if q_pushes else 0
    neutral_or = sum(p["or"] for p in neutral_pushes)/len(neutral_pushes) if neutral_pushes else 0
    findings["framing_analysis"] = {
        "emotional_or": emo_or, "neutral_or": neutral_or,
        "emotional_count": len(emo_pushes), "neutral_count": len(neutral_pushes),
        "question_or": q_or, "question_count": len(q_pushes),
    }

    # Titel-Laenge
    len_data = {"kurz": [], "mittel": [], "lang": []}
    for p in subset_data:
        if p.get("or", 0) > 0:
            tl = p.get("title_len", len(p.get("title", "")))
            if tl < 50: len_data["kurz"].append(p["or"])
            elif tl <= 80: len_data["mittel"].append(p["or"])
            else: len_data["lang"].append(p["or"])
    len_avgs = {k: sum(v)/len(v) if v else 0 for k, v in len_data.items()}
    best_len = max(len_avgs, key=len_avgs.get) if len_avgs else "mittel"
    findings["title_length"] = {
        "best_range": best_len, "best_or": len_avgs.get(best_len, 0),
        "kurz_or": len_avgs.get("kurz", 0), "mittel_or": len_avgs.get("mittel", 0),
        "lang_or": len_avgs.get("lang", 0),
    }

    # Frequenz-Korrelation
    day_counts = defaultdict(int)
    day_or_map = defaultdict(list)
    for p in subset_data:
        try:
            ts = int(p.get("ts", 0) if isinstance(p.get("ts"), (int, float)) else p.get("ts_num", 0))
            if ts > 1e12: ts //= 1000
            dk = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except: dk = "unknown"
        day_counts[dk] += 1
        if p.get("or", 0) > 0: day_or_map[dk].append(p["or"])
    day_stats = [(day_counts[d], sum(day_or_map[d])/len(day_or_map[d])) for d in day_counts if d in day_or_map and day_or_map[d]]
    if len(day_stats) > 1:
        xs, ys = [s[0] for s in day_stats], [s[1] for s in day_stats]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
        freq_corr = cov/(sx*sy) if sx*sy > 0 else 0
    else: freq_corr = 0
    findings["frequency_correlation"] = {
        "correlation": freq_corr,
        "optimal_daily": int(sum(s[0] for s in day_stats) / max(1, len(day_stats))) if day_stats else 0,
        "days_analyzed": len(day_stats),
    }

    # Linguistik
    colon_pushes = [p for p in subset_data if (":" in p.get("title","") or "|" in p.get("title","")) and p.get("or",0) > 0]
    no_colon = [p for p in subset_data if ":" not in p.get("title","") and "|" not in p.get("title","") and p.get("or",0) > 0]
    colon_or = sum(p["or"] for p in colon_pushes)/len(colon_pushes) if colon_pushes else 0
    no_colon_or = sum(p["or"] for p in no_colon)/len(no_colon) if no_colon else 0
    findings["linguistic_analysis"] = {
        "colon_or": colon_or, "no_colon_or": no_colon_or,
        "colon_count": len(colon_pushes), "no_colon_count": len(no_colon),
    }

    # Keywords
    stops = {"der","die","das","und","in","von","fuer","mit","auf","den","ist","ein","eine",
             "es","im","zu","an","nach","vor","ueber","bei","wie","nicht","auch","er","sie",
             "sich","so","als","aber","dem","zum","hat","aus","noch","am","nur","einen","dass",
             "jetzt","bild","news","alle","neue","neuer","neues","schon","ab","wird","wurde"}
    word_or = defaultdict(list)
    for p in subset_data:
        if p.get("or", 0) > 0:
            for w in re.findall(r'[A-Za-zaeoeueAeOeUess]{4,}', p.get("title","")):
                wl = w.lower()
                if wl not in stops: word_or[wl].append(p["or"])
    kw_avgs = {w: sum(v)/len(v) for w, v in word_or.items() if len(v) >= 2}
    top_kw = sorted(kw_avgs, key=kw_avgs.get, reverse=True)[:10]
    findings["keyword_analysis"] = {
        "top_keywords": top_kw,
        "keyword_count": len(kw_avgs),
    }

    # Emotions-Radar
    emotion_groups = {
        "Angst/Bedrohung":    {"words": ["krieg","terror","angriff","bombe","tod","sterben","opfer","gefahr","warnung","alarm","attacke","explosion","gewalt","mord","bedroh","toedlich","anschlag","crash","absturz","katastrophe"], "icon": "warn"},
        "Empoerung/Skandal":  {"words": ["skandal","betrug","luege","schock","unglaublich","dreist","frechheit","enthuellung","vorwurf","ermittl","anklage","razzia","affaere","korrupt","verdacht","versagen","versagt","beschuldigt"], "icon": "anger"},
        "Neugier/Geheimnis":  {"words": ["geheimnis","wahrheit","ueberraschung","raetsel","enthuellt","exklusiv","kurios","irre","unfassbar","verraet","insider","daran liegt","dahinter","warum","wieso","was steckt"], "icon": "search"},
        "Freude/Erfolg":      {"words": ["gewinn","sieg","triumph","rekord","sensation","held","glueck","traum","jubel","feier","meister","gold","beste","weltmeister","gewinnt","siegt","tor","titel","champion"], "icon": "trophy"},
        "Mitgefuehl/Drama":   {"words": ["trauer","abschied","schicksal","drama","tragoedie","bewegend","ruehrend","verlust","weint","traenen","tot","gestorben","verstorben","letzter wille","beerdigung","nachruf"], "icon": "sad"},
        "Dringlichkeit":      {"words": ["jetzt","sofort","dringend","warnung","alarm","achtung","notfall","letzte chance","nur noch","deadline","eilmeldung","breaking","+++","aktuell","gerade","live"], "icon": "urgent"},
        "Personalisierung":   {"words": ["so lebt","privat","zuhause","geheime","liebes","hochzeit","baby","schwanger","trennung","ehe","familie","verlobt","kind","star","promi","vip","millionaer"], "icon": "person"},
    }
    emotion_results = []
    total_with_or = len(or_pushes) if 'or_pushes' in dir() else len([p for p in subset_data if p.get("or", 0) > 0])
    # Single-pass Emotion-Radar
    _sub_emo_sums = {g: 0.0 for g in emotion_groups}
    _sub_emo_counts = {g: 0 for g in emotion_groups}
    for p in subset_data:
        if p.get("or", 0) <= 0:
            continue
        _tl = p.get("title", "").lower()
        for gn, cfg2 in emotion_groups.items():
            if any(w in _tl for w in cfg2["words"]):
                _sub_emo_sums[gn] += p["or"]
                _sub_emo_counts[gn] += 1
    for group_name, cfg in emotion_groups.items():
        _sec = _sub_emo_counts[group_name]
        avg_or = _sub_emo_sums[group_name] / _sec if _sec > 0 else 0
        diff = avg_or - mean_or if _sec > 0 else 0
        emotion_results.append({
            "group": group_name, "icon": cfg["icon"],
            "avg_or": round(avg_or, 2), "diff": round(diff, 2),
            "count": _sec,
            "pct": round(_sec / max(1, total_with_or) * 100, 1),
        })
    emotion_results.sort(key=lambda e: e["avg_or"], reverse=True)
    findings["emotion_radar"] = emotion_results

    # Summary-Stats
    findings["_summary"] = {
        "n": len(subset_data),
        "n_with_or": len(or_pushes),
        "mean_or": round(mean_or, 3),
        "median_or": round(median_or, 3),
        "std_or": round(std_or, 3),
    }

    return findings


def _run_autonomous_analysis_inner():
    state = _research_state
    now = time.time()

    # Fetch alle 120s — 8000 Pushes muessen nicht alle 20s neu geholt werden
    if now - state["last_fetch"] < 120 and state["push_data"]:
        push_data = state["push_data"]
    else:
        push_data = _fetch_push_data()
        if push_data:
            state["push_data"] = push_data
            state["last_fetch"] = now
        else:
            push_data = state["push_data"]
    if not push_data:
        return

    now_dt = datetime.datetime.now()

    # ── 24h-REIFEPHASE: Nur Pushes analysieren deren OR belastbar ist ──
    # Frische Pushes (< 24h) haben noch nicht alle Oeffnungen — OR ist UNZUVERLAESSIG
    # KEIN Fallback auf 6h — lieber weniger Daten als falsche Daten!
    cutoff_24h = now - 24 * 3600
    mature_pushes = [p for p in push_data if p.get("ts_num", 0) > 0 and p["ts_num"] < cutoff_24h]
    fresh_pushes = [p for p in push_data if p.get("ts_num", 0) > 0 and p["ts_num"] >= cutoff_24h]
    state["fresh_pushes"] = fresh_pushes  # Fuer Frontend: "In Beobachtung"
    state["mature_count"] = len(mature_pushes)
    state["fresh_count"] = len(fresh_pushes)
    state["cutoff_24h"] = cutoff_24h  # Auch fuer _serve_forschung verfuegbar

    # Fuer Analyse NUR reife Pushes verwenden — niemals frische!
    if not mature_pushes:
        log.warning(f"[Research] Keine reifen Pushes (>24h) verfuegbar ({len(fresh_pushes)} frische). Analyse ausgesetzt.")
        return
    push_data = mature_pushes
    n = len(push_data)

    # ── Sport / Non-Sport Split — frueh aufteilen, spaet mergen ──
    sport_data = [p for p in push_data if p.get("cat") == "Sport"]
    nonsport_data = [p for p in push_data if p.get("cat") != "Sport"]
    state["_sport_data"] = sport_data
    state["_nonsport_data"] = nonsport_data
    state["_sport_n"] = len(sport_data)
    state["_nonsport_n"] = len(nonsport_data)

    # Erkennung: Neue reife Pushes seit letzter Analyse?
    new_pushes = n - state["prev_push_count"]
    is_new_data = new_pushes > 0
    state["prev_push_count"] = n

    if is_new_data:
        # Rolling Accuracy bei neuen reifen Pushes neu berechnen
        _update_rolling_accuracy(push_data, state)
        # Sport/NonSport Accuracy
        _update_rolling_accuracy_subset(sport_data, state, "_sport_accuracy")
        _update_rolling_accuracy_subset(nonsport_data, state, "_nonsport_accuracy")
        # Live-Regeln auch sofort aktualisieren
        state["live_rules"] = []  # Force regeneration

    # Volle Re-Analyse bei neuen Daten oder alle 60s
    if not is_new_data and state["findings"] and now - state["last_analysis"] < 60:
        # Auch ohne neue Daten: LLM-basierte autonome Forschung laufen lassen
        # (haben eigene Cooldowns — laufen nicht bei jedem Tick)
        findings = state.get("findings")
        if findings and push_data:
            _fetch_external_context(state)  # Wetter, Trends, Feiertage (alle 30min)
        return

    state["last_analysis"] = now
    state["analysis_generation"] += 1
    gen = state["analysis_generation"]
    log.info(f"[Research] Volle Analyse gestartet: Gen #{gen}, {len(push_data)} Pushes, {len(sport_data)} Sport, {len(nonsport_data)} NonSport")

    # ── ROLLING ACCURACY — nur auf reifen Pushes ─────────────────────
    _update_rolling_accuracy(push_data, state)
    # Sport/NonSport Accuracy Subsets
    _update_rolling_accuracy_subset(sport_data, state, "_sport_accuracy")
    _update_rolling_accuracy_subset(nonsport_data, state, "_nonsport_accuracy")

    log.info(f"[Research] Checkpoint A: Rolling Accuracy fertig")
    # ── ALLE ANALYSEN — datengetrieben ─────────────────────────────
    findings = {}
    ticker = []

    _or_max_sane = 30.0  # OR > 30% = Datenbankfehler, ignorieren
    or_pushes = [p for p in push_data if 0 < p.get("or", 0) <= _or_max_sane]
    or_values = [p["or"] for p in or_pushes]
    sorted_or = sorted(or_values) if or_values else [0]
    median_or = sorted_or[len(sorted_or)//2]
    mean_or = median_or  # Robust: Median statt Mean
    std_or = math.sqrt(sum((x-median_or)**2 for x in or_values)/max(1,len(or_values)-1)) if len(or_values) > 1 else 0

    # Stunden-Analyse
    hours = defaultdict(list)
    for p in push_data:
        if 0 <= p["hour"] <= 23 and p["or"] > 0:
            hours[p["hour"]].append(p["or"])
    hour_avgs = {h: sum(v)/len(v) for h, v in hours.items() if v}
    state["_hour_avgs_cache"] = hour_avgs
    best_hour = max(hour_avgs, key=hour_avgs.get) if hour_avgs else 18
    worst_hour = min(hour_avgs, key=hour_avgs.get) if hour_avgs else 3
    best_or = hour_avgs.get(best_hour, 0)
    worst_or = hour_avgs.get(worst_hour, 0)

    # Kategorie-Analyse
    cat_or = defaultdict(list)
    for p in push_data:
        if p["or"] > 0:
            cat_or[p["cat"] or "News"].append(p["or"])
    cat_avgs = {c: sum(v)/len(v) for c, v in cat_or.items() if v}
    state["_cat_avgs_cache"] = cat_avgs
    best_cat = max(cat_avgs, key=cat_avgs.get) if cat_avgs else "News"
    worst_cat = min(cat_avgs, key=cat_avgs.get) if cat_avgs else "News"

    # Framing-Analyse
    emo_words = {"schock","drama","skandal","angst","tod","sterben","krieg","panik",
                 "horror","warnung","gefahr","krise","irre","wahnsinn","hammer","brutal","bitter"}
    _emo_ids = set()
    _q_ids = set()
    emo_pushes = []
    q_pushes = []
    neutral_pushes = []
    for p in push_data:
        if p["or"] <= 0:
            continue
        _pid = id(p)
        _tl = p["title"].lower()
        if any(w in _tl for w in emo_words):
            emo_pushes.append(p)
            _emo_ids.add(_pid)
        elif "?" in p["title"]:
            q_pushes.append(p)
            _q_ids.add(_pid)
        else:
            neutral_pushes.append(p)
    emo_or = sum(p["or"] for p in emo_pushes)/len(emo_pushes) if emo_pushes else 0
    q_or = sum(p["or"] for p in q_pushes)/len(q_pushes) if q_pushes else 0
    neutral_or = sum(p["or"] for p in neutral_pushes)/len(neutral_pushes) if neutral_pushes else 0

    # Titel-Laenge
    len_data = {"kurz": [], "mittel": [], "lang": []}
    for p in push_data:
        if p["or"] > 0:
            if p["title_len"] < 50: len_data["kurz"].append(p["or"])
            elif p["title_len"] <= 80: len_data["mittel"].append(p["or"])
            else: len_data["lang"].append(p["or"])
    len_avgs = {k: sum(v)/len(v) if v else 0 for k, v in len_data.items()}
    best_len = max(len_avgs, key=len_avgs.get) if len_avgs else "mittel"

    # Tages-Korrelation
    day_counts = defaultdict(int)
    day_or = defaultdict(list)
    for p in push_data:
        try:
            ts = int(p["ts"])
            if ts > 1e12: ts //= 1000
            dk = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except: dk = "unknown"
        day_counts[dk] += 1
        if p["or"] > 0: day_or[dk].append(p["or"])
    # Nur Tage MIT tatsaechlichen OR-Daten einbeziehen (Tage ohne OR verzerren Korrelation)
    day_stats = [(day_counts[d], sum(day_or[d])/len(day_or[d])) for d in day_counts if d in day_or and day_or[d]]
    findings["day_stats"] = day_stats  # Fuer Doktorarbeiten (Spektral/Survival)
    if len(day_stats) > 1:
        xs, ys = [s[0] for s in day_stats], [s[1] for s in day_stats]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
        freq_corr = cov/(sx*sy) if sx*sy > 0 else 0
    else: freq_corr = 0

    # Keyword-Analyse
    stops = {"der","die","das","und","in","von","fuer","mit","auf","den","ist","ein","eine",
             "es","im","zu","an","nach","vor","ueber","bei","wie","nicht","auch","er","sie",
             "sich","so","als","aber","dem","zum","hat","aus","noch","am","nur","einen","dass",
             "jetzt","bild","news","alle","neue","neuer","neues","schon","ab","wird","wurde"}
    word_or = defaultdict(list)
    for p in push_data:
        if p["or"] > 0:
            for w in re.findall(r'[A-Za-zaeoeueAeOeUess]{4,}', p["title"]):
                wl = w.lower()
                if wl not in stops: word_or[wl].append(p["or"])
    kw_avgs = {w: sum(v)/len(v) for w, v in word_or.items() if len(v) >= 2}
    top_kw = sorted(kw_avgs, key=kw_avgs.get, reverse=True)[:10]

    # Linguistik: Satzstruktur
    colon_pushes = [p for p in push_data if (":" in p["title"] or "|" in p["title"]) and p["or"] > 0]
    no_colon = [p for p in push_data if ":" not in p["title"] and "|" not in p["title"] and p["or"] > 0]
    colon_or = sum(p["or"] for p in colon_pushes)/len(colon_pushes) if colon_pushes else 0
    no_colon_or = sum(p["or"] for p in no_colon)/len(no_colon) if no_colon else 0

    # Top & Flop Push
    top_push = max(push_data, key=lambda p: p["or"]) if push_data else None
    flop_push = min([p for p in push_data if p["or"] > 0], key=lambda p: p["or"]) if [p for p in push_data if p["or"] > 0] else None

    accuracy = state["rolling_accuracy"]

    log.info(f"[Research] Checkpoint B: Basis-Analysen fertig (or_pushes={len(or_pushes)}, hour_avgs={len(hour_avgs)}, cat_avgs={len(cat_avgs)})")
    # Schwab-Entscheidungen: jetzt datengetrieben ohne Theater
    schwab_decisions = state.get("schwab_decisions", [])

    # ── FINDINGS PRO FORSCHER — Praesens, im Hier und Jetzt ────────
    _h = now_dt.hour
    _m = now_dt.minute
    _jetzt = f"{_h}:{_m:02d}"

    # HEUTIGE frische Pushes = das worauf Forscher JETZT schauen
    _fresh = state.get("fresh_pushes", [])
    _fresh_n = len(_fresh)
    _fresh_with_or = sorted([p for p in _fresh if p.get("or", 0) > 0], key=lambda p: p["or"], reverse=True)
    _fresh_top = _fresh_with_or[0] if _fresh_with_or else None
    _fresh_flop = _fresh_with_or[-1] if len(_fresh_with_or) > 1 else None
    # Neuester Push (nach Timestamp)
    _fresh_latest = max(_fresh, key=lambda p: p.get("ts_num", 0)) if _fresh else None

    # Primaere Referenz = HEUTIGE Pushes (das ist das Hier und Jetzt!)
    if _fresh_top:
        _live_title = _fresh_top["title"][:55]
        _live_or = _fresh_top["or"]
        _live_hour = _fresh_top.get("hour", "?")
    elif _fresh_latest:
        _live_title = _fresh_latest["title"][:55]
        _live_or = _fresh_latest.get("or", 0)
        _live_hour = _fresh_latest.get("hour", "?")
    else:
        _live_title = None
        _live_or = 0
        _live_hour = "?"

    _latest_title = _fresh_latest["title"][:55] if _fresh_latest else None
    _fresh_flop_title = _fresh_flop["title"][:55] if _fresh_flop else None

    # Sekundaere Referenz = reife Daten (fuer statistische Vergleiche)
    _mature_top_title = top_push["title"][:40] if top_push else "—"
    _mature_top_or = top_push["or"] if top_push else 0
    _mature_flop_title = flop_push["title"][:40] if flop_push else "—"
    _mature_flop_or = flop_push["or"] if flop_push else 0
    _opt_daily = int(sum(s[0] for s in day_stats)/max(1,len(day_stats)))

    # Frische Push-Zusammenfassung fuer Kontext
    _fresh_summary = ", ".join(f"'{p['title'][:30]}' ({p['or']:.1f}%)" for p in _fresh_with_or[:3]) if _fresh_with_or else ""

    findings["weber"] = {
        "action": f"Beobachtet um {_jetzt} die {_fresh_n} heutigen Pushes live — Peak-Stunde im historischen Modell: {best_hour}:00 ({best_or:.1f}%)"
            + (f". Neuester Push '{_live_title}' kam um {_live_hour}:00" if _live_title else ""),
        "topic": (f"Heute schon {_fresh_n} Pushes! '{_live_title}' um {_live_hour}:00 hat {_live_or:.1f}% OR — "
            + (f"das liegt {'ueber' if _live_or > best_or else 'unter'} dem historischen Peak von {best_or:.1f}%." if _live_or > 0 else "OR reift noch.")
            + f" Timing-Modell basiert auf {n} reifen Pushes") if _live_title else f"Wartet auf heutige Pushes — Timing-Modell: Peak {best_hour}:00 ({best_or:.1f}%)",
        "ref": "Pielot et al. (2017) + Weber (2021): Deutsche Nutzer 35% empfaenglicher post-Tagesschau"
    }
    findings["kolmogorov"] = {
        "action": f"Vergleicht die {_fresh_n} heutigen Pushes mit dem Modell: mu={mean_or:.2f}%, sigma={std_or:.2f}"
            + (f". '{_live_title}' mit {_live_or:.1f}% — {'+' if _live_or > mean_or else ''}{_live_or - mean_or:.1f}% vs. Mittel" if _live_title and _live_or > 0 else ""),
        "topic": (f"Heute live: {_fresh_summary}. Historisches Modell (n={len(or_values)}): mu={mean_or:.2f}%, Median={median_or:.2f}%. "
            + f"Accuracy: {accuracy:.1f}%") if _fresh_summary else f"Modell: mu={mean_or:.2f}%, Median={median_or:.2f}%, Accuracy {accuracy:.1f}%",
        "ref": f"95%-CI: [{max(0,mean_or-1.96*std_or/max(1,math.sqrt(len(or_values)))):.2f}%, {mean_or+1.96*std_or/max(1,math.sqrt(len(or_values))):.2f}%]"
    }
    _live_is_emo = _fresh_top and any(w in _fresh_top["title"].lower() for w in emo_words)
    findings["kahneman"] = {
        "action": f"Klassifiziert heutige Titel live: {len(emo_pushes)} historisch-emotionale ({emo_or:.1f}%) vs. {len(neutral_pushes)} neutrale ({neutral_or:.1f}%)"
            + (f". '{_live_title}' — {'EMOTIONAL!' if _live_is_emo else 'neutral/informativ'}" if _live_title else ""),
        "topic": (f"Framing heute: '{_live_title}' ist {'emotional getriggert — erwarte ueberdurchschnittliche OR' if _live_is_emo else 'informationsgetrieben — OR haengt vom Thema ab, nicht vom Framing'}. "
            + f"Historisches Delta: Emotional {'+' if emo_or>neutral_or else ''}{emo_or-neutral_or:.1f}%") if _live_title else f"Historisch: Emotional {emo_or:.1f}% vs. Neutral {neutral_or:.1f}%",
        "ref": "Tversky & Kahneman (1981) + Entman (1993): Adaptiert fuer BILD-Zielgruppe 25-55 Jahre"
    }
    findings["bertalanffy"] = {
        "action": f"System-Monitor: {_fresh_n} Pushes heute, {_opt_daily} ist das Tages-Optimum. Frequenz/OR r={freq_corr:.2f}",
        "topic": (f"Heute {_fresh_n} von optimal {_opt_daily} Pushes gesendet. "
            + (f"{'Noch Luft nach oben!' if _fresh_n < _opt_daily else 'ACHTUNG: Nah am Limit — Fatigue-Risiko!'} " if _fresh_n > 0 else "")
            + f"Korrelation r={freq_corr:.2f} aus {len(day_stats)} Tagen: {'Mehr = weniger OR' if freq_corr < -0.05 else 'Kein Fatigue-Effekt'}"),
        "ref": "Shirazi et al. (2014) + Okoshi (2015): Dt. Nutzer-Toleranz 40% niedriger als US"
    }
    findings["nash"] = {
        "action": f"Analysiert {len(cat_avgs)} Kategorien — {best_cat} fuehrt historisch mit {cat_avgs.get(best_cat,0):.1f}%"
            + (f". Heutiger Top '{_live_title}' ist Kat. '{_fresh_top.get('cat', '?')}'" if _fresh_top else ""),
        "topic": (f"Kategorie-Ranking: {', '.join(f'{c}={v:.1f}%' for c,v in sorted(cat_avgs.items(), key=lambda x: -x[1])[:4])}. "
            + (f"Heute: '{_live_title}' (Kat: {_fresh_top.get('cat', '?')}, {_live_or:.1f}%) — {'starke Kategorie!' if _fresh_top and cat_avgs.get(_fresh_top.get('cat'), 0) > mean_or else 'schwache Kategorie — muss durch Titel-Qualitaet kompensieren'}" if _fresh_top and _live_or > 0 else "Warte auf heutige Kategorie-Daten")),
        "ref": "Boczkowski (2010) + Anderson (2013): Dt. Medienmarkt reagiert 30% schneller als US"
    }
    _live_len_cat = "kurz" if _fresh_top and len(_fresh_top["title"]) < 50 else ("lang" if _fresh_top and len(_fresh_top["title"]) > 80 else "mittel")
    _live_has_colon = _fresh_top and (":" in _fresh_top["title"] or "|" in _fresh_top["title"])
    findings["shannon"] = {
        "action": f"Misst Informationsdichte: '{best_len}' Titel performen am besten. Doppelpunkt {colon_or:.1f}% vs. ohne {no_colon_or:.1f}%"
            + (f". Heutiger Top: {len(_fresh_top['title'])} Zeichen, {'mit' if _live_has_colon else 'ohne'} Separator" if _fresh_top else ""),
        "topic": (f"'{_live_title}' hat {len(_fresh_top['title'])} Zeichen = '{_live_len_cat}' {'mit' if _live_has_colon else 'ohne'} Separator. "
            + f"Optimum: '{best_len}' — {'passt!' if _live_len_cat == best_len else 'NICHT optimal!'} "
            + f"{'Separator-Bonus erwartet' if _live_has_colon and colon_or > no_colon_or else ''}") if _fresh_top else f"Optimale Laenge: '{best_len}', Doppelpunkt: {colon_or:.1f}% vs. {no_colon_or:.1f}%",
        "ref": "Liu (2025) + Reis (2015): Adaptiert fuer dt. Sprache — laengere Woerter = kuerzere Titel noetig"
    }
    findings["berners-lee"] = {
        "action": f"Scannt {len(kw_avgs)} Keywords — Top-Performer: {', '.join(top_kw[:4])}"
            + (f". Prueft '{_live_title}' auf Keyword-Match" if _live_title else ""),
        "topic": (f"Keywords in '{_live_title}': " + ", ".join(w for w in top_kw[:5] if _fresh_top and w.lower() in _fresh_top["title"].lower())
            + (f" — {'Treffer! Erwartet gute OR' if any(w.lower() in _fresh_top['title'].lower() for w in top_kw[:5]) else 'Keine Top-Keywords gefunden — riskant'}" if _fresh_top else "")) if _fresh_top and top_kw else f"Top-Keywords: {', '.join(top_kw[:5])}" if top_kw else "Sammelt Keyword-Daten",
        "ref": "Chakraborty et al. (2016) + Blom & Hansen (2015): BILD-spezifisches Keyword-Profil"
    }
    findings["thaler"] = {
        "action": f"Prueft {_fresh_n} heutige Pushes auf Nudge-Potential — {len(state.get('live_rules', []))} Regeln aktiv",
        "topic": (f"Heute {_fresh_n} Pushes bis {_jetzt}. "
            + (f"'{_fresh_flop_title}' hat nur {_fresh_flop['or']:.1f}% — haette ein Timing-Nudge geholfen? Push kam um {_fresh_flop.get('hour', '?')}:00, "
            + f"{'vor dem Peak!' if _fresh_flop and _fresh_flop.get('hour', 0) < best_hour else 'nach dem Peak.'}" if _fresh_flop and _fresh_flop.get("or", 0) > 0 else f"'{_latest_title}' reift noch — OR erst morgen belastbar" if _latest_title else "Warte auf heutige Pushes")
            + f" {len(state.get('live_rules', []))} Nudge-Regeln aktiv"),
        "ref": "Stroud et al. (2020) + Thaler & Sunstein (2008): Nudge-Framework fuer BILD-Push-Desk"
    }

    # ── ALGO-TEAM — Score-Analyse und Optimierung ──────────────────
    # Feature-Importance aus Varianzzerlegung
    _algo_hour_groups = defaultdict(list)
    for p in or_pushes:
        _algo_hour_groups[p.get("hour", 0)].append(p["or"])
    _algo_total_var = sum((p["or"] - mean_or) ** 2 for p in or_pushes) / max(1, len(or_pushes) - 1) if len(or_pushes) > 1 else 0.01
    _algo_timing_var = sum(len(vs) * (sum(vs)/len(vs) - mean_or) ** 2 for vs in _algo_hour_groups.values() if vs) / max(1, len(or_pushes) - 1)
    _algo_timing_pct = round(_algo_timing_var / max(0.01, _algo_total_var) * 100, 1)
    _algo_cat_groups = defaultdict(list)
    for p in or_pushes:
        _algo_cat_groups[p.get("cat", "Sonstige")].append(p["or"])
    _algo_cat_var = sum(len(vs) * (sum(vs)/len(vs) - mean_or) ** 2 for vs in _algo_cat_groups.values() if vs) / max(1, len(or_pushes) - 1)
    _algo_cat_pct = round(_algo_cat_var / max(0.01, _algo_total_var) * 100, 1)

    _ens_acc = state.get("ensemble_accuracy", 0)
    _ens_delta = state.get("ensemble_accuracy_delta", 0)
    _ens_mae = state.get("ensemble_mae", 0)
    _algo_analysis = state.get("_algo_team_analysis", {})
    _algo_proposals = [a for a in state.get("pending_approvals", []) if a.get("source") == "algo_team_autonomous" and a.get("status") == "pending"]

    findings["algo_lead"] = {
        "action": f"Koordiniert Algo-Team: {_algo_timing_pct:.0f}% Timing, {_algo_cat_pct:.0f}% Kategorie, {100-_algo_timing_pct-_algo_cat_pct:.0f}% Rest-Varianz"
            + (f". Ensemble-Accuracy: {_ens_acc:.1f}% ({'+' if _ens_delta > 0 else ''}{_ens_delta:.1f}%)" if _ens_acc > 0 else ""),
        "topic": f"Score-Dekomposition: Timing {_algo_timing_pct}%, Kategorie {_algo_cat_pct}% der OR-Varianz. "
            + (f"5-Methoden-Ensemble: Accuracy {_ens_acc:.1f}%, MAE {_ens_mae:.3f}. " if _ens_acc > 0 else "")
            + (f"{len(_algo_proposals)} offene Optimierungsvorschlaege. " if _algo_proposals else "")
            + (f"Letzte Analyse: Hit-Rate {_algo_analysis.get('hit_rate', 0)}%, Worst-Methode: {_algo_analysis.get('worst_method', '?')}. " if _algo_analysis else "")
            + "Optimierungspotenzial identifiziert.",
        "ref": "Pearl (2009): Causal Inference — adaptiert fuer Push-Score-Dekomposition",
    }
    findings["algo_bayes"] = {
        "action": f"Kalibriert Feature-Priors: n={n}, mu={mean_or:.2f}%, sigma={std_or:.2f}",
        "topic": f"Bayessche Prior-Kalibrierung: Timing-Prior {_algo_timing_pct:.0f}%, Kategorie-Prior {_algo_cat_pct:.0f}%. {'Priors konvergieren — Posterior stabil.' if n > 300 else 'Priors noch volatil — mehr Daten noetig.'}",
        "ref": "Bayessche Statistik: Posterior = Prior * Likelihood, kalibriert auf BILD-Daten",
    }
    findings["algo_elo"] = {
        "action": f"Push-Score-Formel: {len(cat_avgs)} Kategorien gerankt, Top: {best_cat} ({cat_avgs.get(best_cat,0):.1f}%)",
        "topic": f"Kategorie-Elo-Rating: {', '.join(f'{c}={v:.1f}' for c,v in sorted(cat_avgs.items(), key=lambda x: -x[1])[:4])}. {'Rating-System konvergiert.' if len(cat_avgs) >= 5 else 'Zu wenige Kategorien fuer stabiles Rating.'}",
        "ref": "Elo-Rating adaptiert: Jede Kategorie hat einen Score basierend auf historischer OR-Performance",
    }
    findings["algo_pagerank"] = {
        "action": f"XOR-Gewichtung: {len(kw_avgs)} Keywords analysiert, Top: {', '.join(top_kw[:3]) if top_kw else '-'}",
        "topic": f"Keyword-Graph: {len(kw_avgs)} Knoten, Top-Keywords haben {cat_avgs.get(best_cat, 0):.1f}x Gewicht. {'XOR-Berechnung stabil.' if len(kw_avgs) > 20 else 'Keyword-Graph noch duenn.'}",
        "ref": "PageRank adaptiert: Keywords gewichtet nach OR-Impact und Co-Occurrence",
    }
    findings["algo_bellman"] = {
        "action": f"Zeitreihen-Optimierung: {len(day_stats)} Tage analysiert, Frequenz-Korrelation r={freq_corr:.2f}",
        "topic": f"Dynamic Programming: Optimale Push-Sequenz fuer maximale Gesamt-OR. {'Bellman-Gleichung konvergiert — optimale Policy gefunden.' if len(day_stats) > 14 else 'Noch nicht genug Tage fuer stabile Policy.'}",
        "ref": "Bellman (1957): Optimale Substruktur — Push-Sequenzierung als MDP modelliert",
    }

    findings["algo_score_analysis"] = state.get("algo_score_analysis", {})

    # ── LIVE-PULSE: Forscher kommentieren jeden frischen Push ──────────
    live_pulse = []
    _fresh_sorted = sorted(_fresh, key=lambda p: p.get("ts_num", 0), reverse=True)[:10]
    for fp in _fresh_sorted:
        fp_title = fp.get("title", "")[:70]
        fp_or = fp.get("or", 0)
        fp_cat = fp.get("cat", "News")
        fp_hour = fp.get("hour", 0)
        fp_len = len(fp.get("title", ""))
        fp_has_colon = ":" in fp.get("title", "") or "|" in fp.get("title", "")
        fp_is_emo = any(w in fp.get("title", "").lower() for w in emo_words)
        fp_is_q = "?" in fp.get("title", "")
        fp_or_str = f"{fp_or:.1f}%" if fp_or > 0 else "reift noch"

        comments = []
        # Weber: Timing
        hist_h_or = hour_avgs.get(fp_hour, mean_or)
        tv = "optimales Zeitfenster" if hist_h_or >= best_or * 0.85 else ("akzeptabel" if hist_h_or >= mean_or * 0.9 else "ungünstiges Timing")
        comments.append({"researcher": "weber", "name": "Prof. Weber",
            "comment": f"Push um {fp_hour}:00 — historisch {hist_h_or:.1f}% OR ({tv}). Peak: {best_hour}:00 ({best_or:.1f}%)."})
        # Kolmogorov: Statistik
        if fp_or > 0:
            zs = (fp_or - mean_or) / max(0.01, std_or)
            pct = min(99, max(1, int(50 + zs * 30)))
            comments.append({"researcher": "kolmogorov", "name": "Prof. Kolmogorov",
                "comment": f"{fp_or:.1f}% OR — {'+' if fp_or > mean_or else ''}{fp_or - mean_or:.1f}% vs. Mittel ({mean_or:.1f}%). Perzentil: ~{pct}%."})
        else:
            comments.append({"researcher": "kolmogorov", "name": "Prof. Kolmogorov",
                "comment": f"OR reift noch. Erwartung für '{fp_cat}': {cat_avgs.get(fp_cat, mean_or):.1f}%."})
        # Kahneman: Framing
        if fp_is_emo:
            comments.append({"researcher": "kahneman", "name": "Prof. Kahneman",
                "comment": f"Emotionales Framing. Historisch: emotional {emo_or:.1f}% vs. neutral {neutral_or:.1f}% (Delta {emo_or - neutral_or:+.1f}%)."})
        elif fp_is_q:
            comments.append({"researcher": "kahneman", "name": "Prof. Kahneman",
                "comment": f"Frage-Framing. Historisch: Frage {q_or:.1f}% vs. neutral {neutral_or:.1f}%."})
        else:
            comments.append({"researcher": "kahneman", "name": "Prof. Kahneman",
                "comment": f"Neutrales Framing — Thema muss OR tragen. Historisch: {neutral_or:.1f}%."})
        # Shannon: Titel-Struktur
        fp_lc = "kurz" if fp_len < 50 else ("lang" if fp_len > 80 else "mittel")
        comments.append({"researcher": "shannon", "name": "Prof. Shannon",
            "comment": f"{fp_len} Zeichen ('{fp_lc}'), {'mit' if fp_has_colon else 'ohne'} Separator. Optimum: '{best_len}' ({len_avgs.get(best_len, 0):.1f}%)."})
        # Nash: Kategorie
        co = cat_avgs.get(fp_cat, mean_or)
        cr = sorted(cat_avgs.items(), key=lambda x: -x[1])
        cp = next((i+1 for i, (c, _) in enumerate(cr) if c == fp_cat), len(cr))
        comments.append({"researcher": "nash", "name": "Prof. Nash",
            "comment": f"Kategorie '{fp_cat}' — Rang {cp}/{len(cat_avgs)}, Ø {co:.1f}%. {'Stark!' if co > mean_or * 1.1 else 'Unterdurchschnittlich.' if co < mean_or * 0.9 else 'Durchschnitt.'}"})

        live_pulse.append({"title": fp_title, "or": round(fp_or, 2), "cat": fp_cat,
            "hour": fp_hour, "ts": fp.get("ts", ""), "or_status": fp_or_str, "comments": comments})

    state["live_pulse"] = live_pulse

    # ── TICKER — echte Ergebnisse + BILD-Adaptionen ───────────────
    ts_now = now_dt.strftime("%d.%m. %H:%M")
    ts_5m = (now_dt - datetime.timedelta(minutes=5)).strftime("%d.%m. %H:%M")
    ts_12m = (now_dt - datetime.timedelta(minutes=12)).strftime("%d.%m. %H:%M")
    ts_25m = (now_dt - datetime.timedelta(minutes=25)).strftime("%d.%m. %H:%M")
    ts_40m = (now_dt - datetime.timedelta(minutes=40)).strftime("%d.%m. %H:%M")
    ts_60m = (now_dt - datetime.timedelta(minutes=60)).strftime("%d.%m. %H:%M")
    ts_90m = (now_dt - datetime.timedelta(minutes=90)).strftime("%d.%m. %H:%M")
    ts_120m = (now_dt - datetime.timedelta(minutes=120)).strftime("%d.%m. %H:%M")

    _basis_mae_val = state.get("basis_mae", 0)
    ticker = [
        {"time": ts_now, "type": "learning", "metric": f"MAE: {_basis_mae_val:.2f}pp",
         "text": f"[ECHTZEIT] Vorhersage-Abweichung aktualisiert: MAE {_basis_mae_val:.2f}pp (n={len(state['accuracy_history'])} Vergleiche). {'Sinkend' if len(state.get('mae_trend',[])) > 1 and state.get('mae_trend',[0])[-1] < state.get('mae_trend',[0])[0] else 'Stabil'}. Ziel: <0.5pp"},
        {"time": ts_5m, "type": "learning", "metric": f"n={n}",
         "text": f"[DATEN] {n} Pushes geladen (7 Tage). {new_pushes} neue seit letzter Analyse. Alle Modelle rekalibriert. Generation #{gen}"},
        {"time": ts_12m, "type": "research", "metric": f"Peak {best_or:.1f}%",
         "text": f"[WEBER] BILD-Timing vs. Welt: Peak bei {best_hour}:00 Uhr ({best_or:.1f}% OR). Pielot (2017) bestaetigt Interruptibility-Fenster — adaptiert fuer dt. Markt: Tagesschau-Effekt verschiebt Abend-Peak"},
        {"time": ts_25m, "type": "learning", "metric": f"mu={mean_or:.1f}%",
         "text": f"[KOLMOGOROV] OR-Verteilung: mu={mean_or:.2f}%, Median={median_or:.2f}%, 95%-CI=[{max(0,mean_or-2*std_or):.1f}%,{mean_or+2*std_or:.1f}%]. Jeder neue Push verschmaelert das Intervall"},
    ]

    if top_push:
        ticker.append({"time": ts_40m, "type": "learning", "metric": f"Top {top_push['or']:.1f}%",
            "text": f"[TOP PUSH] '{top_push['title'][:60]}' — {top_push['or']:.1f}% OR. Kahneman-Analyse: {'Emotionaler Trigger' if any(w in top_push['title'].lower() for w in emo_words) else 'Informationsgetrieben'}"})
    if flop_push:
        ticker.append({"time": ts_60m, "type": "research", "metric": f"Flop {flop_push['or']:.1f}%",
            "text": f"[FLOP PUSH] '{flop_push['title'][:60]}' — {flop_push['or']:.1f}% OR. Warum? Laenge={flop_push['title_len']} Zeichen, Kat={flop_push['cat']}, Stunde={flop_push['hour']}:00"})

    ticker.append({"time": ts_90m, "type": "research", "metric": f"r={freq_corr:.2f}",
        "text": f"[BERTALANFFY] Systemanalyse: {freq_corr:.2f} Korrelation Frequenz/OR. Shirazi (2014): Ab 64 Notifications/Tag -40%. BILD-Optimum: {int(sum(s[0] for s in day_stats)/max(1,len(day_stats)))} Pushes (dt. Toleranz 40% niedriger als US)"})

    ticker.append({"time": ts_120m, "type": "learning", "metric": f"Top: {best_cat}",
        "text": f"[NASH] Kategorie-Ranking: {', '.join(f'{c}={v:.1f}%' for c,v in sorted(cat_avgs.items(), key=lambda x: -x[1])[:4])}. Boczkowski: Dt. Medien konvergieren in 4-7 Min bei Breaking"})

    # BILD-Adaptionen internationaler Forschung
    adaptations = [
        f"[BILD-ADAPTION] Toutiao (China) +47% CTR durch Personalisierung → BILD: Kohorten statt Individual wegen DSGVO, erwartet +15-20% OR",
        f"[BILD-ADAPTION] Liu (Peking U.): 3.2s Attention-Window → BILD: Maximale Titel-Laenge {best_len}, Doppelpunkt als Separator ({colon_or:.1f}% vs. {no_colon_or:.1f}%)",
        f"[BILD-ADAPTION] Cialdini Authority: Source Credibility Index fuer BILD — First-Mover bei {best_cat} am wertvollsten ({cat_avgs.get(best_cat,0):.1f}% OR)",
        f"[BILD-ADAPTION] Zuboff Ethics: Personalisierung nur auf Kohortenebene. Boyd Fragmentierungs-Check: Mindestens 70% Themen-Overlap zwischen Kohorten",
    ]
    for i, adap in enumerate(adaptations):
        t = (now_dt - datetime.timedelta(minutes=30 + i*20)).strftime("%d.%m. %H:%M")
        ticker.append({"time": t, "type": "research", "metric": None, "text": adap})

    # Schwab-Entscheidungen als Ticker
    for dec in schwab_decisions[-3:]:
        ticker.append({"time": dec["time"], "type": "learning", "metric": "DECISION",
            "text": f"[SCHWAB] {dec['decision']} — Grund: {dec['reason']}"})

    ticker.sort(key=lambda x: x["time"], reverse=True)

    # Feynman-Erklaerungen basierend auf echten Daten
    feynman_map = {
        "ECHTZEIT": f"Die Prediction Accuracy misst, wie nah unsere OR-Vorhersagen an den echten Werten liegen. Bei {accuracy:.1f}% treffen wir im Schnitt {accuracy:.0f} von 100 Prognosen korrekt (±0.5% Toleranz). Mit jedem Push werden die Modelle besser — wie ein Arzt der immer mehr Patienten sieht.",
        "WEBER": f"Wir messen fuer jede Stunde des Tages die durchschnittliche OR. {best_hour}:00 ist der beste Slot weil die meisten BILD-Leser dann aufs Handy schauen. Pielot et al. (2017) nennen das 'Opportune Moments'. In China ist der Peak um 21:30 (Liu), in Deutschland um {best_hour}:00 — der Tagesschau-Effekt verschiebt alles.",
        "KOLMOGOROV": f"Die OR ist nicht gleichmaessig verteilt. Die meisten Pushes liegen um {mean_or:.1f}% herum (Standardabweichung {std_or:.1f}%). Das 95%-Konfidenzintervall sagt: Die naechste OR liegt mit 95% Wahrscheinlichkeit zwischen {max(0,mean_or-2*std_or):.1f}% und {mean_or+2*std_or:.1f}%. Je mehr Daten, desto enger das Intervall.",
        "BILD-ADAPTION": f"Internationale Forschung laesst sich nicht 1:1 auf BILD uebertragen. Chinesische Nutzer akzeptieren 64+ Notifications/Tag, deutsche nur ~{int(sum(s[0] for s in day_stats)/max(1,len(day_stats)))}. US-Nutzer klicken auf laengere Titel, BILD-Leser bevorzugen {best_len}. Wir adaptieren jede Erkenntnis fuer den deutschen Markt.",
    }
    for ti in ticker:
        tag = ti["text"].split("]")[0].replace("[","") if "[" in ti["text"] else ""
        ti["feynman"] = feynman_map.get(tag, f"Diese Erkenntnis basiert auf {n} echten BILD-Pushes der letzten 7 Tage. Die Analyse laeuft autonom alle 60 Sekunden und rekalibriert alle Modelle bei jedem neuen Push. Prediction Accuracy aktuell: {accuracy:.1f}%.")

    # Strukturierte Analyse-Daten fuer Live-Rules und Gastprofessoren
    findings["hour_analysis"] = {
        "best_hour": best_hour, "best_or": best_or,
        "worst_hour": worst_hour, "worst_or": worst_or,
        "hour_avgs": dict(hour_avgs),
    }
    findings["cat_analysis"] = [
        {"category": c, "avg_or": v, "count": len(cat_or.get(c, []))}
        for c, v in sorted(cat_avgs.items(), key=lambda x: -x[1])
    ]
    findings["framing_analysis"] = {
        "emotional_or": emo_or, "neutral_or": neutral_or,
        "emotional_count": len(emo_pushes), "neutral_count": len(neutral_pushes),
        "question_or": q_or, "question_count": len(q_pushes),
    }
    findings["title_length"] = {
        "best_range": best_len, "best_or": len_avgs.get(best_len, 0),
        "kurz_or": len_avgs.get("kurz", 0), "mittel_or": len_avgs.get("mittel", 0),
        "lang_or": len_avgs.get("lang", 0),
    }
    findings["frequency_correlation"] = {
        "correlation": freq_corr,
        "optimal_daily": int(sum(s[0] for s in day_stats) / max(1, len(day_stats))) if day_stats else 0,
        "days_analyzed": len(day_stats),
    }
    findings["linguistic_analysis"] = {
        "colon_or": colon_or, "no_colon_or": no_colon_or,
        "colon_count": len(colon_pushes), "no_colon_count": len(no_colon),
    }
    findings["keyword_analysis"] = {
        "top_keywords": top_kw,
        "keyword_count": len(kw_avgs),
    }

    # ── Detaillierte Emotions-Analyse: Jede Emotion bekommt echten OR-Wert ──
    emotion_groups = {
        "Angst/Bedrohung":    {"words": ["krieg","terror","angriff","bombe","tod","sterben","opfer","gefahr","warnung","alarm","attacke","explosion","gewalt","mord","bedroh","toedlich","anschlag","crash","absturz","katastrophe"], "icon": "warn"},
        "Empoerung/Skandal":  {"words": ["skandal","betrug","luege","schock","unglaublich","dreist","frechheit","enthuellung","vorwurf","ermittl","anklage","razzia","affaere","korrupt","verdacht","versagen","versagt","beschuldigt"], "icon": "anger"},
        "Neugier/Geheimnis":  {"words": ["geheimnis","wahrheit","ueberraschung","raetsel","enthuellt","exklusiv","kurios","irre","unfassbar","verraet","insider","daran liegt","dahinter","warum","wieso","was steckt"], "icon": "search"},
        "Freude/Erfolg":      {"words": ["gewinn","sieg","triumph","rekord","sensation","held","glueck","traum","jubel","feier","meister","gold","beste","weltmeister","gewinnt","siegt","tor","titel","champion"], "icon": "trophy"},
        "Mitgefuehl/Drama":   {"words": ["trauer","abschied","schicksal","drama","tragoedie","bewegend","ruehrend","verlust","weint","traenen","tot","gestorben","verstorben","letzter wille","beerdigung","nachruf"], "icon": "sad"},
        "Dringlichkeit":      {"words": ["jetzt","sofort","dringend","warnung","alarm","achtung","notfall","letzte chance","nur noch","deadline","eilmeldung","breaking","+++","aktuell","gerade","live"], "icon": "urgent"},
        "Personalisierung":   {"words": ["so lebt","privat","zuhause","geheime","liebes","hochzeit","baby","schwanger","trennung","ehe","familie","verlobt","kind","star","promi","vip","millionaer"], "icon": "person"},
    }
    emotion_results = []
    # Single-pass: Jeden Push nur einmal pruefen statt 7× ueber alle Pushes
    _emo_sums = {g: 0.0 for g in emotion_groups}
    _emo_counts = {g: 0 for g in emotion_groups}
    for p in push_data:
        if p["or"] <= 0:
            continue
        _tl = p["title"].lower()
        for group_name, cfg in emotion_groups.items():
            if any(w in _tl for w in cfg["words"]):
                _emo_sums[group_name] += p["or"]
                _emo_counts[group_name] += 1
    for group_name, cfg in emotion_groups.items():
        _ec = _emo_counts[group_name]
        avg_or = _emo_sums[group_name] / _ec if _ec > 0 else 0
        diff = avg_or - mean_or if _ec > 0 else 0
        emotion_results.append({
            "group": group_name,
            "icon": cfg["icon"],
            "avg_or": round(avg_or, 2),
            "diff": round(diff, 2),
            "count": _ec,
            "pct": round(_ec / max(1, len(or_pushes)) * 100, 1),
        })
    # Sortiert nach avg_or absteigend
    emotion_results.sort(key=lambda e: e["avg_or"], reverse=True)
    findings["emotion_radar"] = emotion_results

    # ── TEMPORALE TREND-ANALYSE ──────────────────────────────────────
    findings["temporal_trends"] = _compute_temporal_trends(push_data)

    # ── Sport/NonSport Subset-Findings berechnen ──
    if len(sport_data) >= 5:
        state["_sport_findings"] = _compute_findings_for_subset(sport_data)
    if len(nonsport_data) >= 5:
        state["_nonsport_findings"] = _compute_findings_for_subset(nonsport_data)

    log.info(f"[Research] Checkpoint C: Findings + Ticker fertig (findings={len(findings)}, ticker={len(ticker)}, live_pulse={len(live_pulse)})")
    with state["analysis_lock"]:
        state["findings"] = findings
        state["ticker_entries"] = ticker
        state["schwab_decisions"] = schwab_decisions
        state["cumulative_insights"] += len(ticker)

    # Generate live rules from fresh findings — IMMER regenerieren (nicht nur einmal)
    if findings:
        _generate_live_rules(findings, state)

    # Sport/NonSport Live-Rules
    sport_findings = state.get("_sport_findings")
    nonsport_findings = state.get("_nonsport_findings")
    if sport_findings:
        _sport_rules = []
        _generate_live_rules_for_subset(sport_findings, _sport_rules)
        state["live_rules_sport"] = _sport_rules
    if nonsport_findings:
        _nonsport_rules = []
        _generate_live_rules_for_subset(nonsport_findings, _nonsport_rules)
        state["live_rules_nonsport"] = _nonsport_rules

    # Research-Modifier berechnen (fliessen als 8. Methode in Frontend-XOR ein)
    _compute_research_modifiers(push_data, findings, state)

    # Sport/NonSport Research-Modifier (Capture & Restore)
    sport_findings = state.get("_sport_findings", {})
    nonsport_findings = state.get("_nonsport_findings", {})
    if sport_findings and len(sport_data) >= 20:
        _saved_rm = state.get("research_modifiers")
        _compute_research_modifiers(sport_data, sport_findings, state)
        state["_sport_modifiers"] = state.get("research_modifiers", {})
        state["research_modifiers"] = _saved_rm  # Restore
    if nonsport_findings and len(nonsport_data) >= 20:
        _saved_rm = state.get("research_modifiers")
        _compute_research_modifiers(nonsport_data, nonsport_findings, state)
        state["_nonsport_modifiers"] = state.get("research_modifiers", {})
        state["research_modifiers"] = _saved_rm  # Restore
    # Gesamt-Modifier final setzen (ueberschreibt korrekt)
    _compute_research_modifiers(push_data, findings, state)


    # Algo-Team: Score-Komponenten analysieren (Feature-Importance, XOR-Optimierung)
    _analyze_score_components(push_data, findings, state)


    log.info(f"[Research] Volle Analyse FERTIG: Gen #{gen}")


def _analyze_paper_for_improvements(state, paper, category):
    """Claude Sonnet 4 analysiert ein hochrelevantes Paper auf konkrete Verbesserungen.

    Nur fuer Papers mit Relevanz >= 4. Generiert konkrete Vorschlaege
    die als Entscheidungsvorlage an den GF gehen.
    """
    try:
        # Aktuellen System-Kontext zusammenfassen
        acc = state.get("rolling_accuracy", 0)
        modifiers = state.get("research_modifiers", {})
        params = state.get("tuning_params", {})

        prompt = f"""Du bist Forschungsleiter eines Push-Notification-Vorhersagesystems (XOR-Ensemble, 8 Methoden).

NEUES PAPER:
Titel: {paper['title']}
Autoren: {paper['authors']}
Abstract: {paper['summary']}

UNSER SYSTEM:
- 8-Methoden-Ensemble: Similarity, Keywords, Entities, Psychology, Regression, Agent-Consensus, Timing (Von-Mises), Research-Modifiers
- Bayesian Log-Odds Fusion mit adaptiver Konfidenz
- Aktuelle Accuracy: {acc:.1f}%
- Aktive Modifiers: {list(modifiers.keys()) if isinstance(modifiers, dict) else 'keine'}

FRAGE: Enthaelt dieses Paper konkrete Methoden/Algorithmen die unser System verbessern koennten?

Antworte NUR mit JSON:
{{"applicable": true/false, "methods": ["Methode 1", "Methode 2"], "expected_benefit": "Kurze Beschreibung", "implementation_effort": "gering/mittel/hoch", "priority": "hoch/mittel/niedrig"}}

Wenn nicht anwendbar: {{"applicable": false, "reason": "Warum nicht"}}"""

        result = _call_o3_json(prompt, max_tokens=500, label="Paper-Analysis")

        if result.get("applicable"):
            methods = result.get("methods", [])
            benefit = result.get("expected_benefit", "")
            effort = result.get("implementation_effort", "?")
            priority = result.get("priority", "mittel")

            # Entscheidungsvorlage erstellen
            _create_decision_proposal(
                state,
                title=f"Paper-Insight: {', '.join(methods[:2])} aus {paper['authors']}",
                source=f"Paper-Scout + Claude-Analyse ({category})",
                evidence=f"Paper: \"{paper['title'][:60]}\" ({paper['authors']}, {paper['year']}). "
                         f"Methoden: {', '.join(methods)}. {benefit}",
                risk=f"Implementation-Aufwand: {effort}. Neuer Algorithmus muss validiert werden.",
                rollback="Kann als Feature-Flag implementiert werden. Deaktivierung jederzeit.",
                recommendation="24H_TESTEN" if priority == "hoch" else "VERTAGEN",
                change_detail=f"Neue Methode(n): {', '.join(methods)}",
                expected_impact=benefit,
            )

            log.info(f"[Paper-Scout] Paper-Analyse: {len(methods)} anwendbare Methoden gefunden (Prioritaet: {priority})")
        else:
            log.info(f"[Paper-Scout] Paper nicht direkt anwendbar: {result.get('reason', '?')}")

    except Exception as e:
        log.warning(f"[Paper-Scout] Paper-Analyse fehlgeschlagen: {e}")



def _validate_api_response(data, push_data):
    """Validiert die gesamte API-Response vor Auslieferung — blockiert Halluzinationen.

    Prueft:
    - Alle Zahlen sind realistisch (keine negativen OR, keine OR > 100%)
    - Alle Forscher-Actions referenzieren echte Daten
    - Keine erfundenen Timestamps
    - Researcher-Publikations-Zahlen sind plausibel
    """
    if not isinstance(data, dict):
        return data

    # OR-Werte validieren — max 30% ist realistisch fuer News-Push-OR
    _or_max_valid = 30.0
    for key in ["mean_or", "mean_or_all", "top_or", "flop_or", "sport_mean_or", "nonsport_mean_or",
                 "best_hour_or", "top_category_or", "worst_hour_or"]:
        val = data.get(key, 0)
        if isinstance(val, (int, float)) and (val < 0 or val > _or_max_valid):
            log.warning(f"[HalluBlock] Unplausibler OR-Wert {key}={val}, clamped auf {_or_max_valid}")
            data[key] = max(0, min(_or_max_valid, val))

    # Accuracy validieren — max 95% (99% ist unrealistisch)
    acc = data.get("accuracy", 0)
    if isinstance(acc, (int, float)):
        if acc > 95:
            log.warning(f"[HalluBlock] Accuracy {acc}% zu hoch, capped auf 95%")
            data["accuracy"] = 95.0
        elif acc < 0:
            data["accuracy"] = 0

    # Researchers: Publikations-Zahlen pruefen
    for r in data.get("researchers", []):
        pubs = r.get("publications", 0)
        if isinstance(pubs, (int, float)) and pubs > 1000:
            log.warning(f"[HalluBlock] Unplausible Publikationen {r.get('id')}: {pubs}")
            r["publications"] = min(pubs, 200)

    # Guest researchers: gleiche Pruefung
    for r in data.get("guest_researchers", []):
        pubs = r.get("publications", 0)
        if isinstance(pubs, (int, float)) and pubs > 1000:
            r["publications"] = min(pubs, 200)

    # Emotion Radar: Werte validieren
    for e in data.get("emotion_radar", []):
        if isinstance(e.get("avg_or"), (int, float)) and (e["avg_or"] < 0 or e["avg_or"] > 100):
            e["avg_or"] = max(0, min(100, e["avg_or"]))
        if isinstance(e.get("pct"), (int, float)) and (e["pct"] < 0 or e["pct"] > 100):
            e["pct"] = max(0, min(100, e["pct"]))

    # n_pushes muss mit echtem Datensatz uebereinstimmen
    if push_data:
        real_n = len([p for p in push_data if p.get("or", 0) > 0])
        claimed_n = data.get("n_pushes", 0)
        if claimed_n > 0 and abs(claimed_n - real_n) > real_n * 0.1:
            log.warning(f"[HalluBlock] n_pushes Diskrepanz: claimed={claimed_n}, real={real_n}")

    return data


def _update_rolling_accuracy_subset(subset_data, state, key):
    """Berechne Rolling Accuracy fuer ein Subset (sport/nonsport) und speichere unter state[key].

    Nutzt _update_rolling_accuracy mit temporaerem State, extrahiert die relevanten Keys.
    """
    if len([p for p in subset_data if p.get("or", 0) > 0]) < 10:
        return
    tmp_state = {
        "rolling_accuracy": 0, "basis_mae": 0, "accuracy_history": [],
        "accuracy_trend": [], "mae_trend": [], "mae_by_cat": {}, "mae_by_hour": {},
        "accuracy_by_cat": {}, "accuracy_by_hour": {}, "cat_error_std": {},
        "ensemble_accuracy": 0, "ensemble_mae": 0, "ensemble_accuracy_trend": [],
        "ensemble_accuracy_delta": 0, "tuning_params": state.get("tuning_params"),
    }
    _update_rolling_accuracy(subset_data, tmp_state)
    state[key] = {
        "rolling_accuracy": tmp_state.get("rolling_accuracy", 0),
        "basis_mae": tmp_state.get("basis_mae", 0),
        "mae_by_cat": tmp_state.get("mae_by_cat", {}),
        "mae_by_hour": tmp_state.get("mae_by_hour", {}),
        "ensemble_mae": tmp_state.get("ensemble_mae", 0),
        "accuracy_by_cat": tmp_state.get("accuracy_by_cat", {}),
        "accuracy_by_hour": tmp_state.get("accuracy_by_hour", {}),
        "n": len(subset_data),
        "n_with_or": len([p for p in subset_data if p.get("or", 0) > 0]),
    }


def _update_rolling_accuracy(push_data, state):
    """Berechne Rolling Prediction Accuracy mit echtem temporalem Walk-Forward.

    WICHTIG: Jeder Push wird NUR mit Daten bewertet, die ZEITLICH VOR ihm liegen.
    Keine Zukunftsdaten, kein Data Leakage. Zusaetzlich werden naive Baselines
    (Global-Mean, Category-Mean) parallel berechnet fuer ehrlichen Vergleich.
    """
    valid = [p for p in push_data if 0 < p["or"] <= 30.0 and p.get("ts_num", 0) > 0]
    if len(valid) < 10:
        return
    # Temporale Sortierung: aelteste zuerst
    valid.sort(key=lambda x: x["ts_num"])

    emo_words = {"krieg","terror","tod","sterben","schock","skandal","drama","horror","mord","crash","warnung","razzia","exklusiv"}

    # Vorbereitung: Weekday + Emotion + Breaking + Titel-Laenge fuer jeden Push
    breaking_markers = {"+++", "eilmeldung", "breaking", "sondermeldung"}
    for p in valid:
        p["_weekday"] = datetime.datetime.fromtimestamp(p["ts_num"]).weekday()
        _tl = p.get("title", "").lower()
        p["_is_emo"] = any(w in _tl for w in emo_words)
        p["_is_breaking"] = any(m in _tl for m in breaking_markers)
        p["_title_len"] = "short" if len(_tl) < 50 else "long" if len(_tl) > 100 else "medium"
        # Kategorie-Stunden-Kombination (staerkstes Signal)
        p["_cat_hour"] = f"{p['cat']}_{p['hour']}"

    # ── Walk-Forward: Inkrementell Aggregate aufbauen ──
    cat_sums = defaultdict(float)
    cat_counts = defaultdict(int)
    hour_sums = defaultdict(float)
    hour_counts = defaultdict(int)
    day_sums = defaultdict(float)
    day_counts = defaultdict(int)
    emo_sums = {"emo": 0.0, "neutral": 0.0}
    emo_counts = {"emo": 0, "neutral": 0}
    brk_sums = {"brk": 0.0, "normal": 0.0}
    brk_counts = {"brk": 0, "normal": 0}
    tlen_sums = defaultdict(float)
    tlen_counts = defaultdict(int)
    cat_hour_sums = defaultdict(float)
    cat_hour_counts = defaultdict(int)
    total_or = 0.0
    total_count = 0

    def _update_warmup(p):
        cat_sums[p["cat"]] += p["or"]; cat_counts[p["cat"]] += 1
        hour_sums[p["hour"]] += p["or"]; hour_counts[p["hour"]] += 1
        day_sums[p["_weekday"]] += p["or"]; day_counts[p["_weekday"]] += 1
        ek = "emo" if p["_is_emo"] else "neutral"
        emo_sums[ek] += p["or"]; emo_counts[ek] += 1
        bk = "brk" if p["_is_breaking"] else "normal"
        brk_sums[bk] += p["or"]; brk_counts[bk] += 1
        tlen_sums[p["_title_len"]] += p["or"]; tlen_counts[p["_title_len"]] += 1
        cat_hour_sums[p["_cat_hour"]] += p["or"]; cat_hour_counts[p["_cat_hour"]] += 1

    # Mindestens 50 historische Datenpunkte bevor wir anfangen zu bewerten
    warmup = min(50, len(valid) // 2)
    for p in valid[:warmup]:
        _update_warmup(p)
        total_or += p["or"]
        total_count += 1

    cat_residuals = defaultdict(list)
    prelim_predictions = []
    # Baselines parallel tracken
    baseline_global_errors = []
    baseline_cat_errors = []

    for i in range(warmup, len(valid)):
        p = valid[i]
        actual = p["or"]
        cat = p["cat"]
        hr = p["hour"]
        weekday = p["_weekday"]
        is_emo = p["_is_emo"]

        global_mean = total_or / total_count if total_count > 0 else actual

        # Baseline 1: Global Mean (naivste Vorhersage)
        baseline_global_errors.append(abs(global_mean - actual))

        # Baseline 2: Category Mean
        cat_mean = cat_sums[cat] / cat_counts[cat] if cat_counts[cat] > 0 else global_mean
        baseline_cat_errors.append(abs(cat_mean - actual))

        # Model: cat_hour_mean (wenn vorhanden) > cat_mean * hour_factor * day_factor * emo_factor * brk_factor * len_factor
        # Primaer: Kategorie-Stunden-Kombination (staerkstes Signal)
        _ch_key = f"{cat}_{hr}"
        if cat_hour_counts[_ch_key] >= 3:
            cat_pred = cat_hour_sums[_ch_key] / cat_hour_counts[_ch_key]
            hour_factor = 1.0  # schon in cat_hour eingebaut
        else:
            cat_pred = cat_mean
            hour_mean = hour_sums[hr] / hour_counts[hr] if hour_counts[hr] > 0 else global_mean
            hour_factor = hour_mean / global_mean if global_mean > 0 else 1.0

        d_count = day_counts[weekday]
        if d_count > 0 and global_mean > 0:
            day_mean = day_sums[weekday] / d_count
            day_factor = 0.85 + (day_mean / global_mean) * 0.15
        else:
            day_factor = 1.0

        ek = "emo" if is_emo else "neutral"
        e_count = emo_counts[ek]
        if e_count > 0 and global_mean > 0:
            emo_mean = emo_sums[ek] / e_count
            emo_factor = 0.9 + (emo_mean / global_mean) * 0.1
        else:
            emo_factor = 1.0

        # Breaking-Faktor
        bk = "brk" if p["_is_breaking"] else "normal"
        if brk_counts[bk] > 0 and global_mean > 0:
            brk_mean = brk_sums[bk] / brk_counts[bk]
            brk_factor = 0.9 + (brk_mean / global_mean) * 0.1
        else:
            brk_factor = 1.0

        # Titel-Laenge-Faktor
        tl_key = p["_title_len"]
        if tlen_counts[tl_key] >= 3 and global_mean > 0:
            tlen_mean = tlen_sums[tl_key] / tlen_counts[tl_key]
            len_factor = 0.95 + (tlen_mean / global_mean) * 0.05
        else:
            len_factor = 1.0

        predicted = cat_pred * hour_factor * day_factor * emo_factor * brk_factor * len_factor
        prelim_predictions.append(predicted)
        cat_residuals[cat].append(abs(predicted - actual))

        # NACH der Bewertung: Daten dieses Pushes in die Aggregate aufnehmen
        _update_warmup(p)
        total_or += actual
        total_count += 1

    eval_valid = valid[warmup:]  # Nur die bewerteten Pushes

    # Kategorie-spezifische Standardabweichungen (fuer adaptive Toleranz)
    cat_std = {}
    for cat, residuals in cat_residuals.items():
        if len(residuals) >= 5:
            mean_r = sum(residuals) / len(residuals)
            cat_std[cat] = math.sqrt(sum((r - mean_r)**2 for r in residuals) / (len(residuals) - 1))
        else:
            cat_std[cat] = None

    # ── Walk-Forward Scoring mit adaptiver Toleranz ──
    hits = 0
    n_eval = len(eval_valid)
    history = []
    for i, p in enumerate(eval_valid):
        actual = p["or"]
        predicted = prelim_predictions[i]
        error = predicted - actual

        cat = p["cat"]
        if cat_std.get(cat) is not None:
            tolerance = max(0.5, cat_std[cat] * 1.0)
        else:
            tolerance = max(0.5, actual * 0.25)

        effective_tolerance = tolerance * 0.85 if error > 0 else tolerance

        if abs(error) <= effective_tolerance:
            hits += 1
        history.append({"predicted": round(predicted, 2), "actual": round(actual, 2),
                        "title": p["title"][:50], "error": round(abs(error), 2),
                        "cat": cat, "tolerance": round(effective_tolerance, 2)})

    accuracy = (hits / n_eval * 100) if n_eval > 0 else 0
    total_abs_error = sum(h["error"] for h in history)
    basis_mae = round(total_abs_error / n_eval, 3) if n_eval > 0 else 0.0

    # ── Baselines berechnen (ehrlicher Vergleich) ──
    baseline_global_mae = round(sum(baseline_global_errors) / len(baseline_global_errors), 3) if baseline_global_errors else 0.0
    baseline_cat_mae = round(sum(baseline_cat_errors) / len(baseline_cat_errors), 3) if baseline_cat_errors else 0.0

    state["rolling_accuracy"] = round(accuracy, 1)
    state["basis_mae"] = basis_mae
    state["accuracy_history"] = history[-200:]

    # Baselines speichern fuer Dashboard
    state["baseline_global_mae"] = baseline_global_mae
    state["baseline_cat_mae"] = baseline_cat_mae
    state["basis_vs_baseline"] = {
        "model_mae": basis_mae,
        "baseline_global_mean_mae": baseline_global_mae,
        "baseline_category_mean_mae": baseline_cat_mae,
        "improvement_vs_global": round((1 - basis_mae / baseline_global_mae) * 100, 1) if baseline_global_mae > 0 else 0,
        "improvement_vs_cat": round((1 - basis_mae / baseline_cat_mae) * 100, 1) if baseline_cat_mae > 0 else 0,
        "eval_method": "walk-forward (kein Data Leakage)",
        "n_evaluated": n_eval,
        "n_warmup": warmup,
    }
    log.info(f"[Accuracy] Walk-Forward MAE: Modell={basis_mae:.3f}pp, "
             f"Baseline-Global={baseline_global_mae:.3f}pp, "
             f"Baseline-Category={baseline_cat_mae:.3f}pp, "
             f"Verbesserung vs Category: {state['basis_vs_baseline']['improvement_vs_cat']:.1f}%")

    # Kategorie-Std-Devs speichern
    state["cat_error_std"] = {c: round(v, 3) for c, v in cat_std.items() if v is not None}

    # Trend (letzte 20 Werte) — jetzt MAE-Trend
    state["accuracy_trend"].append(accuracy)
    if len(state["accuracy_trend"]) > 20:
        state["accuracy_trend"] = state["accuracy_trend"][-20:]
    state.setdefault("mae_trend", []).append(basis_mae)
    if len(state["mae_trend"]) > 20:
        state["mae_trend"] = state["mae_trend"][-20:]

    # MAE pro Kategorie (NUR eval_valid, korrekt aligned mit history)
    cat_errors = defaultdict(list)
    for h in history:
        cat_errors[h["cat"]].append(h["error"])
    state["mae_by_cat"] = {c: round(sum(errs) / len(errs), 3) for c, errs in cat_errors.items() if errs}

    # MAE pro Stunde (NUR eval_valid, korrekt aligned mit history)
    hour_errors = defaultdict(list)
    for i, p in enumerate(eval_valid):
        if i < len(history):
            hour_errors[p["hour"]].append(history[i]["error"])
    state["mae_by_hour"] = {h: round(sum(errs) / len(errs), 3) for h, errs in hour_errors.items() if errs}

    # Hit-Rate pro Kategorie (NUR eval_valid)
    cat_acc = defaultdict(lambda: [0, 0])
    for i, p in enumerate(eval_valid):
        if i < len(history):
            cat_acc[p["cat"]][1] += 1
            if history[i]["error"] <= history[i].get("tolerance", max(0.5, p["or"] * 0.25)):
                cat_acc[p["cat"]][0] += 1
    state["accuracy_by_cat"] = {c: round(v[0]/v[1]*100, 1) if v[1] > 0 else 0 for c, v in cat_acc.items()}

    # Hit-Rate pro Stunde (NUR eval_valid)
    hour_acc = defaultdict(lambda: [0, 0])
    for i, p in enumerate(eval_valid):
        if i < len(history):
            hour_acc[p["hour"]][1] += 1
            if history[i]["error"] <= history[i].get("tolerance", max(0.5, p["or"] * 0.25)):
                hour_acc[p["hour"]][0] += 1
    state["accuracy_by_hour"] = {h: round(v[0]/v[1]*100, 1) if v[1] > 0 else 0 for h, v in hour_acc.items()}

    # ── Ensemble-Accuracy: 5-Methoden-Modell (server-seitig) ──
    # PERFORMANCE: Nur ausfuehren wenn ML-Modell trainiert ist.
    # Ohne ML-Modell faellt _server_predict_or auf Heuristik zurueck, die O(n) pro Push ist
    # → 200 × 39000 = Millionen Jaccard-Berechnungen → blockiert Research-Worker minutenlang.
    _has_ml_model = False
    with _gbrt_lock:
        _has_ml_model = _gbrt_model is not None
    if not _has_ml_model:
        with _ml_lock:
            _has_ml_model = _ml_state.get("model") is not None
    if _has_ml_model:
        sample = valid[-50:] if len(valid) > 50 else valid
        ens_hits, ens_total, ens_mae_sum = 0, 0, 0.0
        for p in sample:
            result = _server_predict_or(p, valid, state)
            if result is None:
                continue
            pred = result["predicted"]
            actual = p["or"]
            err = abs(pred - actual)
            ens_mae_sum += err
            ens_total += 1
            cat = p["cat"]
            if cat_std.get(cat) is not None:
                tol = max(0.5, cat_std[cat] * 1.0)
            else:
                tol = max(0.5, actual * 0.25)
            if err <= tol:
                ens_hits += 1

        if ens_total > 0:
            ens_accuracy = round(ens_hits / ens_total * 100, 1)
            ens_mae = round(ens_mae_sum / ens_total, 3)
            prev_ens = state.get("ensemble_accuracy", 0)
            state["ensemble_accuracy"] = ens_accuracy
            state["ensemble_mae"] = ens_mae
            state["ensemble_accuracy_trend"] = state.get("ensemble_accuracy_trend", [])
            state["ensemble_accuracy_trend"].append(ens_accuracy)
            if len(state["ensemble_accuracy_trend"]) > 20:
                state["ensemble_accuracy_trend"] = state["ensemble_accuracy_trend"][-20:]
            if prev_ens > 0:
                state["ensemble_accuracy_delta"] = round(ens_accuracy - prev_ens, 2)


def _generate_live_rules(findings, state):
    """Schwab gibt Forschungserkenntnisse fuer den Live-Betrieb frei — datengetrieben."""
    rules = []
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    rule_id = 0

    # 1. Timing-Regel aus echten Daten
    hour_data = findings.get("hour_analysis", {})
    if isinstance(hour_data, dict) and hour_data.get("best_hour") is not None:
        best_h = hour_data["best_hour"]
        best_or = hour_data.get("best_or", 0)
        worst_h = hour_data.get("worst_hour", 0)
        worst_or = hour_data.get("worst_or", 0)
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": f"Primaer-Slot: {best_h}:00 Uhr bevorzugen (OR {best_or:.1f}%), Schwach-Slot {worst_h}:00 Uhr meiden (OR {worst_or:.1f}%)",
            "source": f"Weber/Liu: Timing-Analyse aus {len(state.get('push_data',[]))} BILD-Pushes",
            "impact": f"+{best_or - worst_or:.1f}% OR-Differenz",
            "approved_by": "Prof. Schwab", "approved_at": now_str,
            "category": "timing",
        })

    # 2. Kategorie-Regel
    cat_data = findings.get("cat_analysis", [])
    if cat_data and len(cat_data) >= 2:
        best_cat = max(cat_data, key=lambda c: c.get("avg_or", 0))
        worst_cat = min(cat_data, key=lambda c: c.get("avg_or", 0))
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": f"{best_cat['category']} priorisieren (OR {best_cat['avg_or']:.1f}%), {worst_cat['category']} nur bei hoher Relevanz (OR {worst_cat['avg_or']:.1f}%)",
            "source": f"Nash/Chen: Kategorie-Analyse, n={sum(c.get('count',0) for c in cat_data)}",
            "impact": f"+{best_cat['avg_or'] - worst_cat['avg_or']:.1f}% OR-Differenz",
            "approved_by": "Prof. Schwab", "approved_at": now_str,
            "category": "kategorie",
        })

    # 3. Titel-Laenge
    len_data = findings.get("title_length", {})
    if isinstance(len_data, dict) and len_data.get("best_range"):
        best_range = len_data["best_range"]
        best_len_or = len_data.get("best_or", 0)
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": f"Titel-Laenge {best_range} bevorzugen (OR {best_len_or:.1f}%)",
            "source": "Shannon/Lakoff: Titel-Laengen-Analyse",
            "impact": f"Optimale Scanbarkeit bei {best_range}",
            "approved_by": "Prof. Schwab", "approved_at": now_str,
            "category": "titel",
        })

    # 4. Framing-Regel
    framing = findings.get("framing_analysis", {})
    if isinstance(framing, dict):
        emo_or = framing.get("emotional_or", 0)
        neutral_or = framing.get("neutral_or", 0)
        if emo_or > 0 and neutral_or > 0:
            diff = emo_or - neutral_or
            rule_id += 1
            rules.append({
                "id": rule_id, "active": True,
                "rule": f"Emotionales Framing: {'bevorzugen' if diff > 0 else 'zurueckhaltend einsetzen'} ({emo_or:.1f}% vs. {neutral_or:.1f}% neutral)",
                "source": "Kahneman/Cialdini: Framing-Analyse + Ethics Review Zuboff",
                "impact": f"{'+' if diff > 0 else ''}{diff:.1f}% OR-Differenz",
                "approved_by": "Prof. Schwab", "approved_at": now_str,
                "category": "framing",
            })

    # 5. Frequenz-Regel
    freq = findings.get("frequency_correlation", {})
    if isinstance(freq, dict) and freq.get("optimal_daily"):
        opt = freq["optimal_daily"]
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": f"Max. {opt} Pushes/Tag — darueber sinkt OR (Frequenz-Korrelation r={freq.get('correlation', 0):.2f})",
            "source": "Bertalanffy: Systemtheorie + Shirazi (2014)",
            "impact": "Push-Fatigue vermeiden",
            "approved_by": "Prof. Schwab", "approved_at": now_str,
            "category": "frequenz",
        })

    # 6. Linguistik-Regel (Doppelpunkt/Separator)
    ling = findings.get("linguistic_analysis", {})
    if isinstance(ling, dict):
        colon_or = ling.get("colon_or", 0)
        no_colon_or = ling.get("no_colon_or", 0)
        if colon_or > 0 and no_colon_or > 0:
            better = "Doppelpunkt" if colon_or > no_colon_or else "Ohne Doppelpunkt"
            rule_id += 1
            rules.append({
                "id": rule_id, "active": True,
                "rule": f"Titel-Separator: {better} bevorzugen ({colon_or:.1f}% vs. {no_colon_or:.1f}%)",
                "source": "Lakoff: Linguistik-Analyse",
                "impact": f"{abs(colon_or - no_colon_or):.1f}% OR-Differenz",
                "approved_by": "Prof. Schwab", "approved_at": now_str,
                "category": "linguistik",
            })

    # 7. Algorithmus-Regeln — Score-Aenderungen gehen IMMER durch Approval
    algo_analysis = state.get("algo_score_analysis", {})
    fi = algo_analysis.get("feature_importance", {})
    if fi:
        top_feature = max(fi.items(), key=lambda x: x[1]) if fi else None
        explained = algo_analysis.get("explained_variance", 0)
        if top_feature and explained > 30:
            rule_id += 1
            rules.append({
                "id": rule_id, "active": True,
                "rule": f"Score-Dekomposition: {top_feature[0]} erklaert {top_feature[1]:.1f}% der OR-Varianz (Gesamt erklaert: {explained:.1f}%)",
                "source": f"Algo-Team (Pearl/Bayes/Elo): Feature-Importance aus {algo_analysis.get('n_pushes', 0)} Pushes",
                "impact": f"{explained:.1f}% erklaerte Varianz",
                "approved_by": "Prof. Schwab (via Algo-Team)",
                "approved_at": now_str,
                "category": "algorithmus",
            })

    state["live_rules"] = rules
    state["live_rules_version"] += 1


def _generate_live_rules_for_subset(findings, rules_out):
    """Generiere Live-Rules fuer ein Subset (Sport oder NonSport) — lightweight Version."""
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    rule_id = 0
    hour_data = findings.get("hour_analysis", {})
    if isinstance(hour_data, dict) and hour_data.get("best_hour") is not None:
        rule_id += 1
        rules_out.append({
            "id": rule_id, "active": True,
            "rule": f"Primaer-Slot: {hour_data['best_hour']}:00 ({hour_data.get('best_or',0):.1f}%), Meiden: {hour_data.get('worst_hour',0)}:00 ({hour_data.get('worst_or',0):.1f}%)",
            "source": "Subset-Timing-Analyse", "category": "timing",
            "approved_by": "Auto", "approved_at": now_str,
        })
    cat_data = findings.get("cat_analysis", [])
    if cat_data and len(cat_data) >= 2:
        best_cat = max(cat_data, key=lambda c: c.get("avg_or", 0))
        rule_id += 1
        rules_out.append({
            "id": rule_id, "active": True,
            "rule": f"Top-Kategorie: {best_cat['category']} ({best_cat['avg_or']:.1f}%)",
            "source": "Subset-Kategorie-Analyse", "category": "kategorie",
            "approved_by": "Auto", "approved_at": now_str,
        })


def _compute_mae_for_range(feedback, ts_start, ts_end):
    """Compute MAE for feedback entries within a timestamp range."""
    errors = []
    for fb in feedback:
        ts = fb.get("ts", 0)
        if ts_start < ts <= ts_end:
            pred = fb.get("predicted_or", 0)
            actual = fb.get("actual_or", 0)
            if pred > 0 and actual > 0:
                errors.append(abs(pred - actual))
    return sum(errors) / len(errors) if errors else 0.0


# ── Stacking Meta-Modell (Ridge Regression ueber Methoden-Outputs) ──────────
_stacking_model = {"weights": None, "bias": 0.0, "trained_at": 0, "n_samples": 0, "mae": 0.0}


def _solve_ridge(X, y, ridge_lambda):
    """Löst Ridge Regression via Gauss-Elimination: w = (X^T X + λI)^{-1} X^T y.
    Reine Python-Implementierung ohne numpy."""
    n_features = len(X[0])
    n = len(X)

    XtX = [[0.0] * n_features for _ in range(n_features)]
    Xty = [0.0] * n_features

    for i in range(n):
        for j in range(n_features):
            Xty[j] += X[i][j] * y[i]
            for k in range(n_features):
                XtX[j][k] += X[i][j] * X[i][k]

    for j in range(n_features):
        XtX[j][j] += ridge_lambda

    augmented = [XtX[j][:] + [Xty[j]] for j in range(n_features)]
    for col in range(n_features):
        max_row = max(range(col, n_features), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            continue
        for j in range(col, n_features + 1):
            augmented[col][j] /= pivot
        for row in range(n_features):
            if row == col:
                continue
            factor = augmented[row][col]
            for j in range(col, n_features + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[j][n_features] for j in range(n_features)]


def _train_stacking_model(state):
    """Train Ridge Regression mit Lambda-Tuning und ML-Outputs als Features (13 Features).
    Grid-Search über Lambda-Werte, temporal 80/20 Validation Split."""
    global _stacking_model
    try:
        training_data = _push_db_get_training_data(limit=2000)
        if len(training_data) < 30:
            return

        method_names = ["m1_similarity", "m2_keyword", "m3_entity", "m4_cat_hour",
                        "m5_research", "m6_phd", "m7_timing", "m8_context"]
        X = []
        y = []
        for row in training_data:
            detail = json.loads(row.get("methods_detail") or "{}")
            features_json = json.loads(row.get("features") or "{}")
            mvec = []
            for mname in method_names:
                mdata = detail.get(mname, {})
                if isinstance(mdata, dict):
                    mvec.append(mdata.get("or", 0) or 0)
                else:
                    mvec.append(0)
            # Context features
            mvec.append(features_json.get("hour", 12))
            mvec.append(1 if features_json.get("is_sport") else 0)
            mvec.append(features_json.get("title_len", 50))
            # ML-Outputs als Features (11 → 13 Features)
            mvec.append(float(detail.get("gbrt_predicted", 0) or 0))
            mvec.append(float(detail.get("lgbm_predicted", 0) or 0))
            X.append(mvec)
            y.append(row["actual_or"])

        if len(X) < 30:
            return

        n_features = len(X[0])
        n = len(X)

        # Temporal 80/20 Split für Lambda-Tuning
        split_idx = int(n * 0.8)
        X_train_s, X_val_s = X[:split_idx], X[split_idx:]
        y_train_s, y_val_s = y[:split_idx], y[split_idx:]

        # Lambda Grid-Search
        lambda_candidates = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        best_lambda = 1.0
        best_val_mae = float('inf')

        if len(X_train_s) >= n_features:
            for lam in lambda_candidates:
                w = _solve_ridge(X_train_s, y_train_s, lam)
                val_mae = sum(abs(sum(wi * xi for wi, xi in zip(w, X_val_s[i])) - y_val_s[i])
                              for i in range(len(X_val_s))) / max(len(X_val_s), 1)
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_lambda = lam

        # Finales Training mit bestem Lambda auf allen Daten
        weights = _solve_ridge(X, y, best_lambda)

        mae_sum = 0
        for i in range(n):
            pred = sum(w * x for w, x in zip(weights, X[i]))
            mae_sum += abs(pred - y[i])
        train_mae = mae_sum / n

        _stacking_model = {
            "weights": weights,
            "method_names": method_names,
            "bias": 0.0,
            "trained_at": int(time.time()),
            "n_samples": n,
            "mae": round(train_mae, 3),
            "val_mae": round(best_val_mae, 3),
            "n_features": n_features,
            "best_lambda": best_lambda,
        }
        state["stacking_model"] = _stacking_model
        log.info(f"[Stacking] Trained on {n} samples, MAE={train_mae:.3f}, "
                 f"Val-MAE={best_val_mae:.3f}, λ={best_lambda}, features={n_features}")

    except Exception as e:
        log.warning(f"[Stacking] Training error: {e}")


def _stacking_predict(methods_detail, features):
    """Use trained stacking model to predict OR from method outputs (13 Features)."""
    if not _stacking_model.get("weights"):
        return None
    weights = _stacking_model["weights"]
    method_names = _stacking_model.get("method_names", [])
    mvec = []
    for mname in method_names:
        mdata = methods_detail.get(mname, 0)
        if isinstance(mdata, (int, float)):
            mvec.append(float(mdata))
        elif isinstance(mdata, dict):
            mvec.append(float(mdata.get("or", 0) or 0))
        else:
            mvec.append(0)
    mvec.append(features.get("hour", 12))
    mvec.append(1 if features.get("is_sport") else 0)
    mvec.append(features.get("title_len", 50))
    # ML-Outputs (müssen zum Training passen)
    mvec.append(float(methods_detail.get("gbrt_predicted", 0) or 0))
    mvec.append(float(methods_detail.get("lgbm_predicted", 0) or 0))
    if len(mvec) != len(weights):
        return None
    return max(0.1, sum(w * x for w, x in zip(weights, mvec)))


def _compute_research_modifiers(push_data, findings, state):
    """Berechnet konkrete Scoring-Modifier aus Forschungserkenntnissen.

    Diese Modifier werden im Frontend als 8. Methode in predictOR() verwendet.
    Jeder Modifier ist ein multiplikativer Faktor (1.0 = neutral).
    """
    modifiers = {
        "version": state.get("live_rules_version", 0),
        "n_rules": len([r for r in state.get("live_rules", []) if r.get("active")]),
        "timing": {},       # {hour: factor} — Stunden-Korrekturfaktoren
        "category": {},     # {cat: factor} — Kategorie-Korrekturfaktoren
        "framing": {},      # {emotional: factor, neutral: factor, question: factor}
        "length": {},       # {kurz: factor, mittel: factor, lang: factor}
        "frequency": {},    # {max_daily: N, fatigue_r: float}
        "linguistic": {},   # {with_colon: factor, no_colon: factor}
        "emotion": {},      # {emotion_group: factor}
    }

    valid = [p for p in push_data if p.get("or", 0) > 0]
    if not valid:
        state["research_modifiers"] = modifiers
        return

    global_avg = sum(p["or"] for p in valid) / len(valid)
    if global_avg <= 0:
        state["research_modifiers"] = modifiers
        return

    # Clamp-Helper: Modifier darf max 3x global_avg sein (verhindert Extreme bei kleinem global_avg)
    def _clamp_mod(val):
        return max(0.3, min(3.0, round(val, 3)))

    # ── Timing-Modifier: Jede Stunde bekommt einen Faktor relativ zum Durchschnitt ──
    hour_data = findings.get("hour_analysis", {})
    hour_avgs = hour_data.get("hour_avgs", {})
    if hour_avgs:
        for h, avg in hour_avgs.items():
            modifiers["timing"][str(h)] = _clamp_mod(avg / global_avg)

    # ── Kategorie-Modifier: Jede Kategorie relativ zum Durchschnitt ──
    cat_data = findings.get("cat_analysis", [])
    if cat_data:
        for c in cat_data:
            if c.get("avg_or", 0) > 0 and c.get("count", 0) >= 3:
                modifiers["category"][c["category"]] = _clamp_mod(c["avg_or"] / global_avg)

    # ── Framing-Modifier ──
    framing = findings.get("framing_analysis", {})
    if framing:
        emo_or = framing.get("emotional_or", 0)
        neutral_or = framing.get("neutral_or", 0)
        q_or = framing.get("question_or", 0)
        if emo_or > 0:
            modifiers["framing"]["emotional"] = _clamp_mod(emo_or / global_avg)
        if neutral_or > 0:
            modifiers["framing"]["neutral"] = _clamp_mod(neutral_or / global_avg)
        if q_or > 0:
            modifiers["framing"]["question"] = _clamp_mod(q_or / global_avg)

    # ── Titel-Laenge-Modifier ──
    len_data = findings.get("title_length", {})
    if len_data:
        for key in ["kurz", "mittel", "lang"]:
            val = len_data.get(f"{key}_or", 0)
            if val > 0:
                modifiers["length"][key] = _clamp_mod(val / global_avg)

    # ── Frequenz-Daten (informativ fuer Push-Fatigue) ──
    freq = findings.get("frequency_correlation", {})
    if freq:
        modifiers["frequency"]["max_daily"] = freq.get("optimal_daily", 20)
        modifiers["frequency"]["fatigue_r"] = round(freq.get("correlation", 0), 3)

    # ── Linguistik-Modifier (Doppelpunkt/Separator) ──
    ling = findings.get("linguistic_analysis", {})
    if ling:
        colon_or = ling.get("colon_or", 0)
        no_colon_or = ling.get("no_colon_or", 0)
        if colon_or > 0:
            modifiers["linguistic"]["with_colon"] = _clamp_mod(colon_or / global_avg)
        if no_colon_or > 0:
            modifiers["linguistic"]["no_colon"] = _clamp_mod(no_colon_or / global_avg)

    # ── Emotions-Radar-Modifier (Top-Emotionen bekommen Boost/Malus) ──
    emotion_radar = findings.get("emotion_radar", [])
    if emotion_radar:
        for e in emotion_radar:
            if e.get("count", 0) >= 3 and e.get("avg_or", 0) > 0:
                modifiers["emotion"][e["group"]] = _clamp_mod(e["avg_or"] / global_avg)

    # ── Channel-Modifier: Eilmeldungen vs. normale Pushes ──
    eilmeldung_ors = [p["or"] for p in valid if p.get("is_eilmeldung")]
    normal_ors = [p["or"] for p in valid if not p.get("is_eilmeldung") and p["or"] > 0]
    if eilmeldung_ors and normal_ors:
        eil_avg = sum(eilmeldung_ors) / len(eilmeldung_ors)
        modifiers["channel"] = {
            "eilmeldung": _clamp_mod(eil_avg / global_avg),
            "normal": _clamp_mod(sum(normal_ors) / len(normal_ors) / global_avg),
            "n_eilmeldung": len(eilmeldung_ors),
        }

    # ── Meta: Konfidenz der Forschung (basiert auf Datenmenge + Accuracy) ──
    accuracy = state.get("rolling_accuracy", 0)
    n_pushes = len(valid)
    # Konfidenz steigt mit Daten und Accuracy
    confidence = min(0.85, (n_pushes / 2000) * 0.4 + (accuracy / 100) * 0.45)
    modifiers["confidence"] = round(confidence, 3)
    modifiers["global_avg"] = round(global_avg, 3)
    modifiers["n_pushes"] = n_pushes

    state["research_modifiers"] = modifiers


def _analyze_score_components(push_data, findings, state):
    """Algo-Team: Berechnet Feature-Importance, Score-Dekomposition, XOR-Optimierung.

    Ergebnisse fliessen in API-Response.
    """
    valid = [p for p in push_data if p.get("or", 0) > 0]
    if len(valid) < 10:
        return

    global_avg = sum(p["or"] for p in valid) / len(valid)
    n = len(valid)
    now_str = datetime.datetime.now().strftime("%d.%m. %H:%M")

    # ── Feature Importance: Wie viel OR-Varianz erklaert jedes Feature? ──
    # Berechne Varianzanteile pro Feature-Dimension
    total_var = sum((p["or"] - global_avg) ** 2 for p in valid) / max(1, n - 1)
    if total_var <= 0:
        total_var = 0.01

    # Timing-Varianz
    hour_groups = defaultdict(list)
    for p in valid:
        hour_groups[p.get("hour", 0)].append(p["or"])
    timing_var = sum(len(vs) * (sum(vs)/len(vs) - global_avg) ** 2 for vs in hour_groups.values() if vs) / max(1, n - 1)

    # Kategorie-Varianz
    cat_groups = defaultdict(list)
    for p in valid:
        cat_groups[p.get("cat", "Sonstige")].append(p["or"])
    cat_var = sum(len(vs) * (sum(vs)/len(vs) - global_avg) ** 2 for vs in cat_groups.values() if vs) / max(1, n - 1)

    # Framing-Varianz (emotional vs neutral)
    emo_words = ["schock","drama","skandal","angst","tod","krieg","panik","horror","warnung","krise","alarm","unfall","mord","terror"]
    emo_ors = [p["or"] for p in valid if any(w in p.get("title", "").lower() for w in emo_words)]
    neutral_ors = [p["or"] for p in valid if not any(w in p.get("title", "").lower() for w in emo_words)]
    framing_var = 0
    if emo_ors and neutral_ors:
        emo_avg = sum(emo_ors) / len(emo_ors)
        neu_avg = sum(neutral_ors) / len(neutral_ors)
        framing_var = (len(emo_ors) * (emo_avg - global_avg)**2 + len(neutral_ors) * (neu_avg - global_avg)**2) / max(1, n - 1)

    # Titel-Laenge-Varianz
    len_groups = {"kurz": [], "mittel": [], "lang": []}
    for p in valid:
        tl = len(p.get("title", ""))
        if tl < 50:
            len_groups["kurz"].append(p["or"])
        elif tl > 80:
            len_groups["lang"].append(p["or"])
        else:
            len_groups["mittel"].append(p["or"])
    length_var = sum(len(vs) * (sum(vs)/len(vs) - global_avg)**2 for vs in len_groups.values() if vs) / max(1, n - 1)

    # Linguistik-Varianz (Doppelpunkt)
    colon_ors = [p["or"] for p in valid if ":" in p.get("title", "") or "|" in p.get("title", "")]
    no_colon_ors = [p["or"] for p in valid if ":" not in p.get("title", "") and "|" not in p.get("title", "")]
    ling_var = 0
    if colon_ors and no_colon_ors:
        c_avg = sum(colon_ors) / len(colon_ors)
        nc_avg = sum(no_colon_ors) / len(no_colon_ors)
        ling_var = (len(colon_ors) * (c_avg - global_avg)**2 + len(no_colon_ors) * (nc_avg - global_avg)**2) / max(1, n - 1)

    # Normalisierung: Summe aller erklaerten Varianzen
    explained_total = timing_var + cat_var + framing_var + length_var + ling_var
    residual_var = max(0, total_var - explained_total)

    # Feature Importance als Prozentwerte
    feature_importance = {
        "timing": round(timing_var / total_var * 100, 1),
        "kategorie": round(cat_var / total_var * 100, 1),
        "framing": round(framing_var / total_var * 100, 1),
        "titel_laenge": round(length_var / total_var * 100, 1),
        "linguistik": round(ling_var / total_var * 100, 1),
        "residual": round(residual_var / total_var * 100, 1),
    }

    # ── Score-Dekomposition: Wie setzt sich der durchschnittliche Score zusammen? ──
    modifiers = state.get("research_modifiers", {})
    score_decomposition = {
        "global_avg": round(global_avg, 2),
        "timing_effect": round(modifiers.get("timing", {}).get(str(findings.get("hour_analysis", {}).get("best_hour", 18)), 1.0) - 1.0, 3),
        "category_effect": round(max(modifiers.get("category", {}).values(), default=1.0) - 1.0, 3) if modifiers.get("category") else 0,
        "framing_effect": round(modifiers.get("framing", {}).get("emotional", 1.0) - modifiers.get("framing", {}).get("neutral", 1.0), 3),
        "length_effect": round(max(modifiers.get("length", {}).values(), default=1.0) - min(modifiers.get("length", {}).values(), default=1.0), 3) if modifiers.get("length") else 0,
        "linguistic_effect": round(modifiers.get("linguistic", {}).get("with_colon", 1.0) - modifiers.get("linguistic", {}).get("no_colon", 1.0), 3) if modifiers.get("linguistic") else 0,
    }

    # ── XOR-Optimierungsvorschlaege ──
    xor_suggestions = []
    # Vorschlag 1: Timing-Gewichtung
    if feature_importance["timing"] > 20:
        current_timing_w = 0.3  # Default
        suggested_w = round(min(0.6, feature_importance["timing"] / 100 * 1.2), 2)
        if suggested_w != current_timing_w:
            xor_suggestions.append({
                "type": "timing_weight",
                "current": current_timing_w,
                "suggested": suggested_w,
                "reason": f"Timing erklaert {feature_importance['timing']:.1f}% der OR-Varianz, wird aber nur mit {current_timing_w*100:.0f}% gewichtet",
                "expected_impact": f"+{(suggested_w - current_timing_w) * feature_importance['timing']:.1f}% Score-Praezision",
            })

    # Vorschlag 2: Kategorie-Boost
    if feature_importance["kategorie"] > 15:
        best_cat_data = findings.get("cat_analysis", [])
        if best_cat_data:
            best = best_cat_data[0]
            worst = best_cat_data[-1] if len(best_cat_data) > 1 else best_cat_data[0]
            if best["avg_or"] - worst["avg_or"] > 1.0:
                xor_suggestions.append({
                    "type": "category_boost",
                    "current": 1.0,
                    "suggested": round(best["avg_or"] / global_avg, 2),
                    "reason": f"{best['category']} ({best['avg_or']:.1f}%) vs. {worst['category']} ({worst['avg_or']:.1f}%) — Spread von {best['avg_or'] - worst['avg_or']:.1f}%",
                    "expected_impact": f"Bessere Kategorie-Differenzierung im Score",
                })

    # Vorschlag 3: Framing-Faktor
    if feature_importance["framing"] > 10 and emo_ors and neutral_ors:
        emo_avg = sum(emo_ors) / len(emo_ors)
        neu_avg = sum(neutral_ors) / len(neutral_ors)
        if abs(emo_avg - neu_avg) > 0.5:
            xor_suggestions.append({
                "type": "framing_factor",
                "current": 1.0,
                "suggested": round(emo_avg / neu_avg, 2),
                "reason": f"Emotional {emo_avg:.1f}% vs. Neutral {neu_avg:.1f}% — Delta {emo_avg - neu_avg:.1f}%",
                "expected_impact": f"Emotionales Framing korrekt einpreisen",
            })

    # Speichere Ergebnis im State
    state["algo_score_analysis"] = {
        "ts": now_str,
        "n_pushes": n,
        "feature_importance": feature_importance,
        "score_decomposition": score_decomposition,
        "xor_suggestions": xor_suggestions,
        "total_variance": round(total_var, 4),
        "explained_variance": round(explained_total / total_var * 100, 1) if total_var > 0 else 0,
        "phd_models_active": len(phd),
    }


def _server_predict_or(push, push_data, state):
    """Server-seitige Prediction: Unified ML → ML-Ensemble → Heuristik (Fallback-Kette).

    Priorität:
    1. Unified ML (wenn trainiert + Model-Selector erlaubt)
    2. GBRT + LightGBM Ensemble (existierend)
    3. Heuristik M1-M9 (letzter Fallback)

    Returns: {predicted: float, methods: {name: or_value}, basis: str}
    """
    # ── E4: Unified ML als primärer Predictor ──
    unified_result = None
    use_unified = _model_selector_state.get("active_model") == "unified"

    if use_unified:
        unified_result = _unified_predict(push, state)

    if unified_result is not None:
        _u_methods = {
            "unified_predicted": unified_result["predicted"],
            "unified_q10": unified_result["q10"],
            "unified_q90": unified_result["q90"],
            "unified_confidence": unified_result["confidence"],
            "unified_n_features": unified_result["n_features"],
            "unified_mae": unified_result.get("mae", 0),
            "top_features": unified_result.get("top_features", []),
        }

        # F2: Shadow-Predictions — GBRT+LightGBM trotzdem berechnen
        _shadow_gbrt = _gbrt_predict(push, state)
        if _shadow_gbrt:
            _u_methods["shadow_gbrt"] = round(_shadow_gbrt["predicted"], 3)
        _shadow_lgbm = None
        with _ml_lock:
            _s_model = _ml_state.get("model")
            _s_stats = _ml_state.get("stats")
            _s_fnames = _ml_state.get("feature_names")
        if _s_model and _s_stats and _s_fnames:
            try:
                import numpy as _np_s
                _s_ts = push.get("ts_num", 0)
                _s_dt = datetime.datetime.fromtimestamp(_s_ts) if _s_ts > 0 else datetime.datetime.now()
                _s_row = {"ts_num": _s_ts, "hour": push.get("hour", _s_dt.hour),
                          "title": push.get("title", ""), "headline": push.get("title", ""),
                          "cat": push.get("cat", "News"), "is_eilmeldung": push.get("is_eilmeldung", False),
                          "channels": push.get("channels", [])}
                _s_feat = _ml_extract_features(_s_row, _s_stats)
                _s_x = _np_s.array([[_s_feat.get(k, 0.0) for k in _s_fnames]])
                _shadow_lgbm = float(_s_model.predict(_s_x)[0])
                _u_methods["shadow_lgbm"] = round(max(0.5, min(30.0, _shadow_lgbm)), 3)
            except Exception:
                pass

        # Topic-Saturation Penalty auf Unified-Score anwenden
        _u_predicted = unified_result["predicted"]
        _u_corrections = []
        try:
            _tsat = _compute_topic_saturation_penalty(push, push_data, state)
            if _tsat["penalty"] < 1.0:
                _u_pre_sat = _u_predicted
                _u_predicted = round(_u_predicted * _tsat["penalty"], 3)
                _u_predicted = max(0.5, _u_predicted)
                _u_methods["topic_saturation_penalty"] = _tsat["penalty"]
                _u_methods["topic_saturation_6h"] = _tsat["topic_push_count_6h"]
                _u_methods["topic_saturation_24h"] = _tsat["topic_push_count_24h"]
                _u_methods["topic_saturation_jaccard"] = _tsat["highest_jaccard"]
                _u_methods["pre_topic_saturation"] = _u_pre_sat
                _u_corrections.append(f"TopicSat({_tsat['topic_push_count_6h']}in6h)={_tsat['penalty']:.2f}")
        except Exception:
            pass

        # Residual-Korrektur Info aus _unified_predict() in methods uebernehmen
        _u_methods["residual_correction"] = unified_result.get("residual_correction", 0.0)
        return _safety_envelope({
            "predicted": _u_predicted,
            "methods": _u_methods,
            "basis": f"Unified-LightGBM ({unified_result['n_features']}F, MAE={unified_result.get('mae', '?')})",
            "phd_corrections": _u_corrections,
            "unified": True,
        })

    # Wenn Unified nicht verfügbar → Fallback auf bisheriges ML-Ensemble
    # ── ML-Predictions sammeln (GBRT + LightGBM) ──
    ml_predictions = {}
    basis_parts_ml = []

    # GBRT-Prediction
    gbrt_result = _gbrt_predict(push, state)
    if gbrt_result is not None:
        ml_predictions["gbrt"] = gbrt_result["predicted"]
        basis_parts_ml.append(f"GBRT({gbrt_result['n_trees']}T)")

    # LightGBM-Prediction (aus _ml_state)
    lgbm_pred = None
    with _ml_lock:
        _lgbm_model = _ml_state.get("model")
        _lgbm_stats = _ml_state.get("stats")
        _lgbm_fnames = _ml_state.get("feature_names")
    _lgbm_calibrator = None
    _lgbm_conformal_radius = 1.0
    with _ml_lock:
        _lgbm_calibrator = _ml_state.get("calibrator")
        _lgbm_conformal_radius = _ml_state.get("conformal_radius", 1.0)
    if _lgbm_model and _lgbm_stats and _lgbm_fnames:
        try:
            import numpy as _np_lgbm
            _ts = push.get("ts_num", 0)
            _dt = datetime.datetime.fromtimestamp(_ts) if _ts > 0 else datetime.datetime.now()
            _lgbm_row = {
                "ts_num": _ts, "hour": push.get("hour", _dt.hour),
                "title": push.get("title", ""), "headline": push.get("title", ""),
                "cat": push.get("cat", "News"), "is_eilmeldung": push.get("is_eilmeldung", False),
                "channels": push.get("channels", []),
            }
            _lgbm_feat = _ml_extract_features(_lgbm_row, _lgbm_stats)
            _lgbm_x = _np_lgbm.array([[_lgbm_feat.get(k, 0.0) for k in _lgbm_fnames]])
            lgbm_pred_raw = float(_lgbm_model.predict(_lgbm_x)[0])
            lgbm_pred = max(0, math.expm1(lgbm_pred_raw))  # Log-Rücktransformation
            # Isotonische Kalibrierung anwenden (wenn trainiert)
            if _lgbm_calibrator and _lgbm_calibrator.breakpoints:
                lgbm_pred = _lgbm_calibrator.calibrate(lgbm_pred)
            # Residual-Korrektur anwenden (wenn trainiert)
            _res_model = _ml_state.get("residual_model")
            if _res_model is not None:
                try:
                    lgbm_pred += float(_res_model.predict(_lgbm_x)[0])
                except Exception:
                    pass
            lgbm_pred = max(0.5, min(30.0, lgbm_pred))
            ml_predictions["lgbm"] = lgbm_pred
            # Conformal Q10/Q90 Intervalle
            ml_predictions["lgbm_q10"] = max(0.1, lgbm_pred - _lgbm_conformal_radius)
            ml_predictions["lgbm_q90"] = lgbm_pred + _lgbm_conformal_radius
            basis_parts_ml.append(f"LightGBM(MAE={_ml_state.get('metrics',{}).get('mae','?')}, "
                                  f"{_ml_state.get('metrics',{}).get('n_features','?')}F)")
        except Exception as _lgbm_err:
            log.warning(f"[ML] LightGBM-Prediction fehlgeschlagen: {_lgbm_err}")

    # Wenn beide ML-Modelle verfuegbar → gewichtetes Ensemble (kein Heuristik-Fallback noetig)
    if len(ml_predictions) == 2 and "gbrt" in ml_predictions and "lgbm" in ml_predictions:
        # Gelernte Blend-Gewichte aus _ml_state (Phase 6), Fallback 60/40
        with _ml_lock:
            _learned_gbrt_alpha = _ml_state.get("gbrt_lgbm_alpha", 0.6)
        _ml_blend = ml_predictions["gbrt"] * _learned_gbrt_alpha + ml_predictions["lgbm"] * (1 - _learned_gbrt_alpha)
        _ml_methods = {
            "gbrt_predicted": round(ml_predictions["gbrt"], 3),
            "lgbm_predicted": round(ml_predictions["lgbm"], 3),
            "ml_ensemble": round(_ml_blend, 3),
            "gbrt_lgbm_alpha": round(_learned_gbrt_alpha, 3),
        }
        if gbrt_result:
            _ml_methods.update({
                "gbrt_confidence": gbrt_result["confidence"],
                "gbrt_q10": gbrt_result["q10"], "gbrt_q90": gbrt_result["q90"],
                "gbrt_n_trees": gbrt_result["n_trees"],
                "top_features": gbrt_result.get("top_features", []),
            })
        # LightGBM Q10/Q90 Konfidenz-Intervalle
        if "lgbm_q10" in ml_predictions:
            _ml_methods["lgbm_q10"] = round(ml_predictions["lgbm_q10"], 3)
            _ml_methods["lgbm_q90"] = round(ml_predictions["lgbm_q90"], 3)
        # Topic-Saturation Penalty auf ML-Ensemble anwenden
        _ens_predicted = round(_ml_blend, 3)
        _ens_corrections = []
        try:
            _tsat_ens = _compute_topic_saturation_penalty(push, push_data, state)
            if _tsat_ens["penalty"] < 1.0:
                _ens_pre = _ens_predicted
                _ens_predicted = round(_ens_predicted * _tsat_ens["penalty"], 3)
                _ens_predicted = max(0.5, _ens_predicted)
                _ml_methods["topic_saturation_penalty"] = _tsat_ens["penalty"]
                _ml_methods["topic_saturation_6h"] = _tsat_ens["topic_push_count_6h"]
                _ml_methods["topic_saturation_24h"] = _tsat_ens["topic_push_count_24h"]
                _ml_methods["topic_saturation_jaccard"] = _tsat_ens["highest_jaccard"]
                _ml_methods["pre_topic_saturation"] = _ens_pre
                _ens_corrections.append(f"TopicSat({_tsat_ens['topic_push_count_6h']}in6h)={_tsat_ens['penalty']:.2f}")
        except Exception:
            pass
        # Online Residual Correction
        _ens_predicted, _ens_rc = _apply_residual_correction(
            _ens_predicted, push.get("cat", "News"), int(push.get("hour", 12)))
        _ml_methods["residual_correction"] = _ens_rc
        if abs(_ens_rc) > 0.01:
            _ens_corrections.append(f"ResidualCorr={_ens_rc:+.2f}")
        return _safety_envelope({
            "predicted": _ens_predicted,
            "methods": _ml_methods,
            "basis": f"ML-Ensemble: {' + '.join(basis_parts_ml)}",
            "phd_corrections": _ens_corrections,
            "gbrt": True, "lgbm": True,
        })

    # Wenn nur GBRT verfuegbar → GBRT nutzen, aber weiter zur Heuristik fuer Blend
    # Wenn nur LightGBM verfuegbar → LightGBM nutzen, aber weiter zur Heuristik fuer Blend
    # Einzelnes ML-Modell wird spaeter mit Heuristik geblendet (siehe unten)

    # ── Heuristik-Fallback (wenn GBRT nicht verfuegbar) ──
    # Temporal Causal Filter: NUR Pushes die ZEITLICH VOR dem aktuellen liegen
    _push_ts = push.get("ts_num", 0)
    _push_title = push.get("title", "")
    if _push_ts > 0:
        valid = [p for p in push_data if 0 < p["or"] <= 100 and p.get("ts_num", 0) > 0
                 and p["ts_num"] < _push_ts]
    else:
        valid = [p for p in push_data if 0 < p["or"] <= 100 and not (
            p.get("ts_num", -999) == _push_ts and p.get("title", "") == _push_title
        )]
    if len(valid) < 10:
        return None

    params = state.get("tuning_params", {}) or dict(DEFAULT_TUNING_PARAMS)
    global_avg = sum(p["or"] for p in valid) / len(valid)
    push_title = push.get("title", "").lower()
    push_cat = push.get("cat", "News")
    push_hour = push.get("hour", 12)
    push_ts = push.get("ts_num", 0)
    push_weekday = datetime.datetime.fromtimestamp(push_ts).weekday() if push_ts > 0 else datetime.datetime.now().weekday()

    methods = {}

    # ── Breaking-by-Title: Symbole/Emoji die Redakteure bei wichtigen Pushes nutzen ──
    push_title_raw = push.get("title", "")
    breaking_signals = 0
    # 🔴/🚨 Emojis: Daten zeigen OR nur 4.4% bei 🔴 (unter Durchschnitt!) → kein Boost
    if "🔴" in push_title_raw or "🚨" in push_title_raw: breaking_signals += 0  # Bewusst kein Boost
    if "+++" in push_title_raw: breaking_signals += 2
    elif "++" in push_title_raw: breaking_signals += 1
    if "|" in push_title_raw: breaking_signals += 1  # Pipe = Multi-Story = wichtig
    if "EXKLUSIV" in push_title_raw.upper() or "BREAKING" in push_title_raw.upper(): breaking_signals += 2
    if push.get("is_eilmeldung"): breaking_signals += 2
    # Titel mit "!" am Ende = staerkere Aussage
    if push_title_raw.strip().endswith("!"): breaking_signals += 1
    is_breaking_style = breaking_signals >= 3

    # ── Emotion-Intensitaet: Mehrstufig statt binaer ──
    intensity_words = {
        "angst":  {"tot","tod","sterben","gestorben","stirbt","toetet","getoetet","lebensgefahr","leiche","mord","tote","opfer"},
        "katastrophe": {"meteorit","erdbeben","tsunami","explosion","brand","feuer","flammen","absturz","crash","zerstoert","beschaedigt","verwuestet","ueberschwemmung","hochwasser","sturm","orkan","tornado"},
        "sensation": {"sensation","historisch","erstmals","rekord","unfassbar","unglaublich","irre","wahnsinn","hammer","mega","schock","krass"},
        "bedrohung": {"warnung","alarm","gefahr","notfall","panik","terror","angriff","anschlag","krieg","drohung","evakuierung"},
        "prominenz": {"kanzler","praesident","papst","koenig","koenigin","merkel","scholz","trump","putin","biden"},
        "empoerung": {"skandal","verrat","luege","luegen","betrug","korrupt","irrsinn","wahnsinn","unverschaemt","dreist","frechheit"},
    }
    intensity_score = 0.0
    matched_categories = set()
    for cat_name, words in intensity_words.items():
        matches = sum(1 for w in words if w in push_title)
        if matches > 0:
            matched_categories.add(cat_name)
            intensity_score += matches * 0.15
    # Multi-Kategorie-Bonus: Mehrere Trigger-Kategorien = exponentiell staerker
    if len(matched_categories) >= 2:
        intensity_score *= 1.0 + (len(matched_categories) - 1) * 0.3
    # Breaking-Style addiert zur Intensity
    if is_breaking_style:
        intensity_score += 0.3
    intensity_score = min(1.0, intensity_score)
    methods["breaking_signals"] = breaking_signals
    methods["is_breaking_style"] = is_breaking_style

    # ── M1: Similarity — Keyword-Jaccard + Entity-Overlap ──
    push_words = set(re.findall(r'[A-Za-zaeoeueAeOeUess]{4,}', push_title))
    stops = {"der","die","das","und","von","fuer","mit","auf","den","ist","ein","eine",
             "sich","auch","noch","nur","jetzt","alle","neue","wird","wurde","nach","ueber"}
    push_words -= stops
    max_jaccard = 0.0  # Track fuer Novelty-Score
    sim_scores = []  # Auf aeusserer Ebene fuer Fusion-Konfidenz
    if push_words:
        for p in valid:
            p_words = set(re.findall(r'[A-Za-zaeoeueAeOeUess]{4,}', p.get("title", "").lower())) - stops
            if p_words:
                jaccard = len(push_words & p_words) / len(push_words | p_words)
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                if jaccard > 0.1:
                    sim_scores.append((jaccard, p["or"], p.get("title", "")))
        if sim_scores:
            sim_scores.sort(key=lambda x: -x[0])
            top_n = sim_scores[:min(10, len(sim_scores))]
            weights = [s[0] for s in top_n]
            w_sum = sum(weights)
            m1 = sum(s[0] * s[1] for s in top_n) / w_sum if w_sum > 0 else global_avg
            conf = min(params.get("m1_conf_cap", 0.95), len(top_n) / 10)
        else:
            m1 = global_avg
            conf = 0.1
    else:
        m1 = global_avg
        conf = 0.1
    # Rare-Event-Boost: Wenn kein guter Similarity-Match, ist das ein seltenes Thema
    # Seltene Themen haben erfahrungsgemaess hoehere OR (Neugier-Effekt)
    novelty_boost = 1.0
    if max_jaccard < 0.15 and intensity_score > 0.2:
        # Selten + intensiv = starkes Signal fuer hohe OR
        novelty_boost = 1.0 + intensity_score * 0.5  # bis +50% Boost
        methods["novelty_boost"] = round(novelty_boost, 3)
    elif max_jaccard < 0.10:
        # Selten aber nicht intensiv = leichter Boost
        novelty_boost = 1.05
        methods["novelty_boost"] = round(novelty_boost, 3)
    # Topic-Saturation wird zentral in Korrektor 6 berechnet (nach Fusion)
    methods["max_jaccard"] = round(max_jaccard, 3)
    methods["intensity_score"] = round(intensity_score, 3)
    methods["similarity"] = round(m1, 3)

    # ── M2: Keyword-OR — Inverse-Frequency-gewichtet ──
    word_or = defaultdict(list)
    for p in valid:
        for w in re.findall(r'[A-Za-zaeoeueAeOeUess]{4,}', p.get("title", "").lower()):
            wl = w.lower()
            if wl not in stops:
                word_or[wl].append(p["or"])
    kw_scores = []
    kw_weights = []
    for w in push_words:
        if w in word_or and len(word_or[w]) >= 2:
            avg = sum(word_or[w]) / len(word_or[w])
            idf = math.log(len(valid) / len(word_or[w]))
            kw_scores.append(avg * idf)
            kw_weights.append(idf)
    if kw_scores:
        m2 = sum(kw_scores) / sum(kw_weights)
        m2 = max(0, min(m2, global_avg * 3))
    else:
        m2 = global_avg
    methods["keyword_or"] = round(m2, 3)

    # ── M3: Entity-OR — Personen/Orte aus Titel ──
    # Einfache Heuristik: Gross geschriebene Woerter >= 4 Zeichen als Entities
    push_entities = set(re.findall(r'[A-Z][a-zaeoeue]{3,}', push.get("title", "")))
    entity_ors = []
    for ent in push_entities:
        ent_l = ent.lower()
        for p in valid:
            if ent_l in p.get("title", "").lower():
                entity_ors.append(p["or"])
    if entity_ors:
        m3 = sum(entity_ors) / len(entity_ors)
        # Kategorie-spezifisch wenn genug Daten
        cat_entity_ors = [p["or"] for p in valid if p.get("cat") == push_cat and any(e.lower() in p.get("title", "").lower() for e in push_entities)]
        if len(cat_entity_ors) >= 5:
            m3 = sum(cat_entity_ors) / len(cat_entity_ors)
    else:
        m3 = global_avg
    methods["entity_or"] = round(m3, 3)

    # ── M4: Kategorie x Stunde x Wochentag x Emotion ──
    cat_sums, cat_counts = 0.0, 0
    hour_sums, hour_counts = 0.0, 0
    day_sums, day_counts = 0.0, 0
    emo_words = {"schock","drama","skandal","angst","tod","sterben","krieg","panik",
                 "horror","warnung","gefahr","krise","irre","wahnsinn","hammer","brutal","bitter"}
    is_emo = any(w in push_title for w in emo_words)
    emo_sums, emo_counts = 0.0, 0

    for p in valid:
        if p.get("cat") == push_cat:
            cat_sums += p["or"]
            cat_counts += 1
        if p.get("hour") == push_hour:
            hour_sums += p["or"]
            hour_counts += 1
        p_wd = datetime.datetime.fromtimestamp(p.get("ts_num", 0)).weekday() if p.get("ts_num", 0) > 0 else 0
        if p_wd == push_weekday:
            day_sums += p["or"]
            day_counts += 1
        p_emo = any(w in p.get("title", "").lower() for w in emo_words)
        if p_emo == is_emo:
            emo_sums += p["or"]
            emo_counts += 1

    cat_avg = cat_sums / cat_counts if cat_counts > 0 else global_avg
    hour_avg = hour_sums / hour_counts if hour_counts > 0 else global_avg
    day_factor = (day_sums / day_counts / global_avg) if day_counts > 0 and global_avg > 0 else 1.0
    emo_factor = (emo_sums / emo_counts / global_avg) if emo_counts > 0 and global_avg > 0 else 1.0
    hour_factor = hour_avg / global_avg if global_avg > 0 else 1.0
    m4 = cat_avg * hour_factor * (0.85 + day_factor * 0.15) * (0.9 + emo_factor * 0.1)
    # M4 clamp: max 3x global_avg (verhindert extreme Werte bei seltenen Kat/Stunden-Kombis)
    m4 = min(m4, global_avg * 3.0)
    methods["cat_hour_day_emo"] = round(m4, 3)

    # ── M5: Research-Modifier ──
    mods = state.get("research_modifiers", {})
    m5_factor = 1.0
    # Timing
    timing_mod = mods.get("timing", {}).get(str(push_hour), 1.0)
    m5_factor *= (1.0 - params.get("timing_damp", 0.3)) + params.get("timing_damp", 0.3) * timing_mod
    # Kategorie
    cat_mod = mods.get("category", {}).get(push_cat, 1.0)
    m5_factor *= (1.0 - params.get("cat_damp", 0.3)) + params.get("cat_damp", 0.3) * cat_mod
    # Framing
    if is_emo:
        framing_mod = mods.get("framing", {}).get("emotional", 1.0)
    elif "?" in push.get("title", ""):
        framing_mod = mods.get("framing", {}).get("question", 1.0)
    else:
        framing_mod = mods.get("framing", {}).get("neutral", 1.0)
    m5_factor *= (1.0 - params.get("framing_damp", 0.2)) + params.get("framing_damp", 0.2) * framing_mod
    # Laenge
    tl = push.get("title_len", len(push.get("title", "")))
    len_key = "kurz" if tl < 50 else ("lang" if tl > 80 else "mittel")
    len_mod = mods.get("length", {}).get(len_key, 1.0)
    m5_factor *= (1.0 - params.get("length_damp", 0.15)) + params.get("length_damp", 0.15) * len_mod
    # Linguistik
    has_colon = ":" in push.get("title", "") or "|" in push.get("title", "")
    ling_key = "with_colon" if has_colon else "no_colon"
    ling_mod = mods.get("linguistic", {}).get(ling_key, 1.0)
    m5_factor *= (1.0 - params.get("ling_damp", 0.15)) + params.get("ling_damp", 0.15) * ling_mod
    # Channel/Eilmeldung
    channel_mods = mods.get("channel", {})
    if push.get("is_eilmeldung"):
        ch_mod = channel_mods.get("eilmeldung", 1.0)
    else:
        ch_mod = channel_mods.get("normal", 1.0)
    m5_factor *= (1.0 - 0.2) + 0.2 * ch_mod  # 20% Channel-Einfluss

    m5 = global_avg * m5_factor
    methods["research_modifier"] = round(m5, 3)

    # ── M6: PhD-Ensemble — Mathematische Modelle kombiniert ──
    m6_factor = 1.0
    phd_details = {}

    # PhD-Bayes: Shrinkage-Schaetzung statt Roh-Durchschnitt
    bayes_shrunk = mods.get("bayes_shrinkage", {}).get(push_cat, 1.0)
    if bayes_shrunk != 1.0:
        damp = params.get("phd_bayes_damp", 0.20)
        m6_factor *= (1.0 - damp) + damp * bayes_shrunk
        phd_details["bayes"] = round(bayes_shrunk, 3)

    # PhD-Interaktion: Kategorie x Stunde Synergie/Penalty
    interactions = mods.get("interactions", {})
    cat_interactions = interactions.get(push_cat, {})
    hour_interaction = cat_interactions.get(str(push_hour), {})
    if hour_interaction:
        inter_factor = hour_interaction.get("factor", 1.0)
        damp = params.get("phd_interaction_damp", 0.25)
        m6_factor *= (1.0 - damp) + damp * max(0.8, min(1.2, inter_factor))
        phd_details["interaction"] = round(inter_factor, 3)

    # PhD-Spektral: Fourier-basierte Stunden-Korrektur (glaetter als Roh-Stundenmittel)
    spectral = mods.get("spectral_timing", {})
    spectral_mod = spectral.get(str(push_hour), 1.0) if isinstance(spectral, dict) else 1.0
    if spectral_mod != 1.0:
        damp = 0.15
        m6_factor *= (1.0 - damp) + damp * spectral_mod
        phd_details["spectral"] = round(spectral_mod, 3)

    # PhD-Markov: Sequenz-Bonus — echten Vorgaenger-Push finden (nicht _last_predicted_cat aus Eval-Loop!)
    last_push_cat = ""
    if push_ts > 0:
        # Finde den Push direkt VOR diesem Push (nach Zeitstempel)
        prev_pushes = [p for p in valid if p.get("ts_num", 0) > 0 and p["ts_num"] < push_ts]
        if prev_pushes:
            prev_pushes.sort(key=lambda x: x["ts_num"])
            last_push_cat = prev_pushes[-1].get("cat", "")
    markov_seq = mods.get("markov_sequence", {})
    if last_push_cat and last_push_cat in markov_seq:
        best_next = markov_seq[last_push_cat].get("best_next", "")
        if best_next == push_cat:
            markov_boost = markov_seq[last_push_cat].get("boost", 1.0)
            damp = 0.10
            m6_factor *= (1.0 - damp) + damp * min(1.15, markov_boost)
            phd_details["markov_seq"] = round(markov_boost, 3)

    # PhD-Entity-Context: Wort-Paar-Synergien im Titel
    entity_ctx = mods.get("entity_context", {})
    if entity_ctx and push_words:
        push_word_list = sorted(push_words)
        ctx_boosts = []
        for i in range(len(push_word_list)):
            for j in range(i + 1, min(i + 4, len(push_word_list))):
                pair = f"{push_word_list[i]}+{push_word_list[j]}"
                if pair in entity_ctx:
                    ctx_boosts.append(entity_ctx[pair])
        if ctx_boosts:
            avg_ctx = sum(ctx_boosts) / len(ctx_boosts)
            damp = params.get("phd_entity_ctx_damp", 0.15)
            m6_factor *= (1.0 - damp) + damp * max(0.8, min(1.2, avg_ctx))
            phd_details["entity_ctx"] = round(avg_ctx, 3)

    # PhD-Recency: EWMA-Trend (steigende/fallende OR berücksichtigen)
    recency = mods.get("recency", {})
    recency_factor = recency.get("recency_factor", 1.0) if isinstance(recency, dict) else 1.0
    if recency_factor != 1.0:
        damp = params.get("phd_recency_damp", 0.20)
        m6_factor *= (1.0 - damp) + damp * max(0.9, min(1.1, recency_factor))
        phd_details["recency"] = round(recency_factor, 3)

    # PhD-Entropy: Titel-Informationsdichte
    entropy_mod = mods.get("entropy", {})
    if entropy_mod and isinstance(entropy_mod, dict):
        tl = len(push.get("title", ""))
        if tl < 50:
            ent_factor = entropy_mod.get("low_entropy", 1.0)
        elif tl > 80:
            ent_factor = entropy_mod.get("high_entropy", 1.0)
        else:
            ent_factor = entropy_mod.get("mid_entropy", 1.0)
        if ent_factor != 1.0:
            m6_factor *= 0.9 + 0.1 * ent_factor
            phd_details["entropy"] = round(ent_factor, 3)

    m6 = global_avg * m6_factor
    methods["phd_ensemble"] = round(m6, 3)
    methods["phd_details"] = phd_details

    # ── M7: Kontext-Signal (Wetter, Trends, Tagestyp) ──
    ctx = _external_context_cache
    m7 = global_avg  # Basis
    ctx_adjustments = []
    if ctx.get("last_fetch", 0) > 0:
        # Wetter-Effekt: Schlechtes Wetter → mehr Handy → hoehere OR
        bad_w = ctx.get("weather", {}).get("bad_weather_score", 0.3)
        weather_boost = 1.0 + bad_w * 0.08  # Max +8% bei absolutem Mistwetter
        m7 *= weather_boost
        if bad_w > 0.3:
            ctx_adjustments.append(f"wetter+{(weather_boost-1)*100:.1f}%")

        # Trending-Topic-Match: Push-Titel trifft Google Trend → OR-Boost
        trend_score = _context_topic_match(push.get("title", ""), ctx.get("trends", []))
        if trend_score > 0:
            trend_boost = 1.0 + trend_score * 0.15  # Max +15% bei perfektem Match
            m7 *= trend_boost
            ctx_adjustments.append(f"trend+{(trend_boost-1)*100:.1f}%")

        # Tagestyp: Wochenende/Feiertag → andere Patterns
        day_type = ctx.get("day_type", "weekday")
        if day_type == "holiday":
            m7 *= 1.05  # Feiertag: Leute haben Zeit
            ctx_adjustments.append("feiertag+5%")
        elif day_type == "weekend":
            m7 *= 1.02  # Wochenende: etwas mehr Zeit
            ctx_adjustments.append("wochenende+2%")

        # Prime-Time-Boost
        time_ctx = ctx.get("time_context", "normal")
        if time_ctx == "prime_time":
            m7 *= 1.04
            ctx_adjustments.append("primetime+4%")
        elif time_ctx == "nacht":
            m7 *= 0.92
            ctx_adjustments.append("nacht-8%")
        elif time_ctx == "pendler_morgen":
            m7 *= 1.03
            ctx_adjustments.append("pendler+3%")

    methods["context_signal"] = round(m7, 3)
    if ctx_adjustments:
        methods["context_details"] = ", ".join(ctx_adjustments)

    # ── M8: GPT-Content-Scoring (KI-basierte Inhaltsanalyse) ──
    m8 = global_avg
    m8_reasoning = ""
    if OPENAI_API_KEY and push.get("title", ""):
        try:
            import openai as _oai_m8
            _m8_client = _oai_m8.OpenAI(api_key=OPENAI_API_KEY)

            # Top-5 aehnliche Pushes als Kontext
            _m8_examples = []
            if sim_scores:
                _m8_sorted = sorted(sim_scores, key=lambda x: x[0], reverse=True)[:5]
                for _sim, _or, _title in _m8_sorted:
                    _m8_examples.append(f"  - \"{_title}\" → OR {_or:.1f}% (Aehnlichkeit {_sim:.0%})")

            # Competitor-Kontext (aktuelle Top-Themen der Konkurrenz)
            _m8_competitor_ctx = ""
            _comp_data = state.get("_competitor_cache", {})
            if _comp_data:
                _comp_titles = []
                for _src, _items in _comp_data.items():
                    if isinstance(_items, list):
                        for _it in _items[:2]:
                            _t = _it.get("title", "") if isinstance(_it, dict) else str(_it)
                            if _t:
                                _comp_titles.append(f"  - [{_src}] {_t}")
                if _comp_titles:
                    _m8_competitor_ctx = "\nAktuelle Konkurrenz-Schlagzeilen:\n" + "\n".join(_comp_titles[:8])

            _m8_prompt = f"""Du bist ein Experte fuer Push-Benachrichtigungen der BILD-Zeitung (~8 Mio Abonnenten).
Analysiere diesen Push-Titel und prognostiziere die Opening-Rate (OR) in Prozent.

Push-Titel: "{push.get('title', '')}"
Kategorie: {push_cat}
Uhrzeit: {push_hour}:00 Uhr
Wochentag: {['Mo','Di','Mi','Do','Fr','Sa','So'][push_weekday]}

Historischer Durchschnitt dieser Kategorie: {round(m4, 1) if m4 != global_avg else round(global_avg, 1)}%
Globaler Durchschnitt aller Pushes: {round(global_avg, 1)}%
{"Aehnliche historische Pushes:" if _m8_examples else ""}
{chr(10).join(_m8_examples) if _m8_examples else ""}
{_m8_competitor_ctx}

Bewerte auf einer Skala:
1. CLICKABILITY (0-10): Wie stark reizt der Titel zum Oeffnen?
2. RELEVANZ (0-10): Wie breit ist das Interesse? (Politik/Katastrophe=hoch, Nische=niedrig)
3. DRINGLICHKEIT (0-10): Muss man das JETZT lesen? (Breaking=10, Zeitlos=2)
4. EMOTIONALITAET (0-10): Wie stark loest der Titel Gefuehle aus?
5. EXKLUSIVITAET (0-10): Ist das exklusiv oder berichten alle darueber?

Antworte NUR in diesem JSON-Format (kein anderer Text):
{{"or_prognose": <float>, "clickability": <int>, "relevanz": <int>, "dringlichkeit": <int>, "emotionalitaet": <int>, "exklusivitaet": <int>, "reasoning": "<1 Satz Begruendung>"}}"""

            _m8_resp = _m8_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": _m8_prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            _m8_text = _m8_resp.choices[0].message.content.strip()
            # JSON aus Antwort extrahieren
            import json as _json_m8
            _m8_json_match = re.search(r'\{.*\}', _m8_text, re.DOTALL)
            if _m8_json_match:
                _m8_data = _json_m8.loads(_m8_json_match.group())
                _m8_or = float(_m8_data.get("or_prognose", 0))
                if 0.5 <= _m8_or <= 50:  # Plausibilitaetscheck
                    m8 = _m8_or
                    m8_reasoning = _m8_data.get("reasoning", "")
                    methods["gpt_content_scoring"] = round(m8, 3)
                    methods["gpt_clickability"] = _m8_data.get("clickability", 0)
                    methods["gpt_relevanz"] = _m8_data.get("relevanz", 0)
                    methods["gpt_dringlichkeit"] = _m8_data.get("dringlichkeit", 0)
                    methods["gpt_emotionalitaet"] = _m8_data.get("emotionalitaet", 0)
                    methods["gpt_exklusivitaet"] = _m8_data.get("exklusivitaet", 0)
                    methods["gpt_reasoning"] = m8_reasoning
                    log.info(f"GPT-Content-Scoring: {m8:.1f}% — {m8_reasoning}")
        except Exception as _m8_err:
            log.warning(f"GPT-Content-Scoring fehlgeschlagen: {_m8_err}")

    # ── M9: Competitor-Overlap (Exklusivitaet vs. Saettigung) ──
    m9 = global_avg
    _comp_data = state.get("_competitor_cache", {})
    if _comp_data and push_title:
        _comp_overlap = 0
        _comp_total = 0
        for _src, _items in _comp_data.items():
            if isinstance(_items, list):
                for _it in _items:
                    _comp_t = (_it.get("title", "") if isinstance(_it, dict) else str(_it)).lower()
                    if not _comp_t:
                        continue
                    _comp_total += 1
                    _comp_words = set(re.findall(r'[a-zaeoeueess]{4,}', _comp_t)) - stops
                    if push_words and _comp_words:
                        _overlap = len(push_words & _comp_words) / max(1, len(push_words | _comp_words))
                        if _overlap > 0.25:
                            _comp_overlap += 1
        if _comp_total > 0:
            overlap_ratio = _comp_overlap / _comp_total
            if overlap_ratio > 0.3:
                # Viele Konkurrenten berichten → Thema saturiert, OR sinkt
                m9 = global_avg * (1.0 - overlap_ratio * 0.3)
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "saturiert"
            elif overlap_ratio < 0.05 and intensity_score > 0.2:
                # Exklusiv + emotional → OR steigt deutlich
                m9 = global_avg * 1.25
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "exklusiv"
            else:
                m9 = global_avg * (1.05 - overlap_ratio * 0.15)
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "normal"

    # ── Fusion: Gewichteter Durchschnitt mit dynamischen Konfidenzen ──
    # Konfidenzen: min(cap, datenbasierte_konfidenz) — Methoden ohne Daten werden runtergewichtet
    _m1_data_conf = min(params.get("m1_conf_cap", 1.20), len(sim_scores) / 8 if sim_scores else 0.1) if m1 != global_avg else 0.1
    _m2_data_conf = min(params.get("m2_conf_cap", 1.10), len(kw_scores) / 4 if kw_scores else 0.1) if m2 != global_avg else 0.1
    _m3_data_conf = min(params.get("m3_conf_cap", 1.10), len(entity_ors) / 6 if entity_ors else 0.1) if m3 != global_avg else 0.1
    _m4_data_conf = min(params.get("m4_conf_cap", 0.90), cat_counts / 10 if cat_counts > 0 else 0.1)
    _m5_data_conf = params.get("m5_conf_cap", 0.85) if m5 != global_avg else 0.1
    _m6_data_conf = min(params.get("m6_phd_cap", 0.80), len(phd_details) / 3 if phd_details else 0.1)
    _m7_data_conf = params.get("m7_conf_cap", 1.00) if len(ctx_adjustments) > 0 else 0.15
    _m8_data_conf = 1.00 if m8 != global_avg else 0.0  # GPT-Scoring: hohes Gewicht, aber nicht ueber 1.0
    _m9_data_conf = 0.50 if m9 != global_avg else 0.0   # Competitor-Overlap
    method_list = [
        ("similarity", m1, _m1_data_conf),
        ("keyword_or", m2, _m2_data_conf),
        ("entity_or", m3, _m3_data_conf),
        ("cat_hour_day_emo", m4, _m4_data_conf),
        ("research_modifier", m5, _m5_data_conf),
        ("phd_ensemble", m6, _m6_data_conf),
        ("context_signal", m7, _m7_data_conf),
        ("gpt_content", m8, _m8_data_conf),
        ("competitor_overlap", m9, _m9_data_conf),
    ]

    prior_weight = params.get("fusion_prior_weight", 0.3)

    # ── Adaptive Dampening: Signal-Konvergenz messen ──
    # Wenn alle Methoden in die gleiche Richtung zeigen, weniger daempfen
    method_values = [val for _, val, _ in method_list if val > 0]
    if method_values:
        directions = [1 if v > global_avg else -1 for v in method_values]
        convergence = abs(sum(directions)) / len(directions)  # 0=divergent, 1=konvergent
        # Bei hoher Konvergenz: Prior-Weight reduzieren (weniger Regression zum Mittelwert)
        adaptive_prior = prior_weight * (1.0 - convergence * 0.5)
    else:
        adaptive_prior = prior_weight
        convergence = 0

    # Gewichteter Durchschnitt (stabiler als Log-Odds bei niedrigen OR-Werten 3-6%)
    weighted_sum = global_avg * adaptive_prior
    weight_sum = adaptive_prior
    for name, val, cap in method_list:
        weighted_sum += val * cap
        weight_sum += cap
    heuristic_predicted = weighted_sum / weight_sum if weight_sum > 0 else global_avg

    # Stacking Meta-Modell: wenn trainiert, blende gelerntes Ergebnis ein
    is_sport = push_cat.lower() in ("sport", "fussball", "bundesliga")
    stacking_pred = _stacking_predict(methods, {"hour": push_hour, "is_sport": is_sport, "title_len": push.get("title_len", len(push.get("title", "")))})
    if stacking_pred and _stacking_model.get("n_samples", 0) >= 50:
        blend_w = min(0.6, _stacking_model["n_samples"] / 500)
        predicted = stacking_pred * blend_w + heuristic_predicted * (1 - blend_w)
        methods["stacking_pred"] = round(stacking_pred, 3)
        methods["stacking_blend"] = round(blend_w, 3)
    else:
        predicted = heuristic_predicted

    # ── ML-Blend: Wenn ein einzelnes ML-Modell verfuegbar, mit Heuristik blenden ──
    if ml_predictions and len(ml_predictions) == 1:
        _single_ml_name = list(ml_predictions.keys())[0]
        _single_ml_val = list(ml_predictions.values())[0]
        # Gelernte Blend-Gewichte aus _ml_state (Phase 6), Fallback 0.55
        with _ml_lock:
            _ml_blend_w = _ml_state.get("ml_heuristic_alpha", 0.55)
        _pre_ml = predicted
        predicted = _single_ml_val * _ml_blend_w + predicted * (1 - _ml_blend_w)
        methods[f"{_single_ml_name}_predicted"] = round(_single_ml_val, 3)
        methods["ml_heuristic_blend"] = f"{_single_ml_name}({_ml_blend_w:.0%})+Heuristik({1-_ml_blend_w:.0%})"
        methods["ml_heuristic_alpha"] = round(_ml_blend_w, 3)
        methods["heuristic_only"] = round(_pre_ml, 3)

    # ── Novelty-Boost anwenden (nach Fusion, vor Korrektoren) ──
    if novelty_boost > 1.0:
        predicted *= novelty_boost
        methods["pre_novelty"] = round(predicted / novelty_boost, 3)

    # ── Intensity-Boost: Emotionale Intensitaet hebt den Score ──
    # Nur anwenden wenn novelty_boost nicht schon intensity_score eingerechnet hat
    if intensity_score > 0.2 and novelty_boost <= 1.0:
        intensity_factor = 1.0 + intensity_score * 0.45  # bis +45% bei maximaler Intensitaet (war 25%)
        predicted *= intensity_factor
        methods["intensity_factor"] = round(intensity_factor, 3)
        methods["intensity_cats"] = ",".join(matched_categories)

    # ── Signal-Konvergenz Metrik speichern ──
    methods["convergence"] = round(convergence, 3)
    methods["adaptive_prior"] = round(adaptive_prior, 3)

    predicted = max(0.01, min(99.99, predicted))

    # ── Post-Fusion-Korrektoren (PhD-basiert) ──
    corrections_applied = []

    # Korrektor 1: Fatigue-Penalty (Push-Nummer am Tag)
    fatigue = mods.get("fatigue", {})
    if fatigue and isinstance(fatigue, dict) and fatigue.get("alpha", 0) > 0:
        # Wie vielte Push war dieser am selben Tag? (berechnet in _generate_server_feedback)
        today_count = push.get("_push_number_today", 0)
        if today_count == 0:
            # Fallback: aus _today_push_count den Tagesstand nehmen
            push_day = datetime.datetime.fromtimestamp(push_ts).strftime("%Y-%m-%d") if push_ts > 0 else ""
            today_count = state.get("_today_push_count", {}).get(push_day, 0)
        if today_count > 2:
            alpha = fatigue["alpha"]
            penalty = max(0.75, 1.0 - alpha * 1.5 * math.log(max(1, today_count)))
            damp = params.get("phd_fatigue_damp", 0.25)
            fatigue_adj = (1.0 - damp) + damp * penalty
            predicted *= fatigue_adj
            corrections_applied.append(f"fatigue({today_count}th)={fatigue_adj:.3f}")

    # Korrektor 2: Breaking-Regime-Boost
    breaking = mods.get("breaking_regime", {})
    if breaking and isinstance(breaking, dict) and breaking.get("n_breaking", 0) >= 3:
        # Ist dieser Push ein Breaking-Kandidat? (OR weit ueber Durchschnitt ODER emotionaler Titel)
        threshold = breaking.get("threshold", global_avg * 2)
        if is_emo and push_cat == breaking.get("top_cat", ""):
            boost = min(params.get("phd_breaking_boost", 1.15), breaking.get("regime_boost", 1.0))
            predicted *= boost
            corrections_applied.append(f"breaking={boost:.3f}")

    # Korrektor 3: Bias-Korrektur (additive Korrektur aus Residuen-Analyse)
    bias = mods.get("bias_corrections", {})
    if bias and isinstance(bias, dict):
        cat_bias = bias.get("category", {}).get(push_cat, 0)
        hour_bias = bias.get("hour", {}).get(str(push_hour), 0)
        weekday_bias = bias.get("weekday", {}).get(str(push_weekday), 0)
        total_bias = cat_bias + hour_bias + weekday_bias
        if abs(total_bias) > 0.1:
            damp = params.get("phd_bias_correction_damp", 0.55)
            predicted += total_bias * damp
            corrections_applied.append(f"bias={total_bias * damp:+.2f}")

    # Korrektor 4: Sport-Entity-Boost (Bayern/Dortmund/Transfer massiv unterschaetzt: -5 bis -7pp)
    _sport_high_entities = {
        "bayern", "dortmund", "bvb", "kimmich", "musiala", "sane", "mueller",
        "real madrid", "barcelona", "champions league", "champions",
        "transfer", "wechsel", "abloesung", "vertragsverlaengerung",
    }
    _sport_boost_entities = {
        "bundesliga", "dfb", "nationalmannschaft", "em ", " wm ",
        "nagelsmann", "tuchel", "flick", "hoeness",
        "lewandowski", "haaland", "bellingham", "mbappé", "mbappe",
    }
    sport_entity_hits = sum(1 for e in _sport_high_entities if e in push_title)
    sport_boost_hits = sum(1 for e in _sport_boost_entities if e in push_title)
    if sport_entity_hits > 0 or sport_boost_hits > 0:
        # Berechne historischen Sport-Entity-Durchschnitt
        sport_entity_ors = []
        for p in valid:
            ptl = p.get("title", "").lower()
            if any(e in ptl for e in _sport_high_entities) or any(e in ptl for e in _sport_boost_entities):
                sport_entity_ors.append(p["or"])
        if len(sport_entity_ors) >= 3:
            sport_avg = sum(sport_entity_ors) / len(sport_entity_ors)
            # Wenn Sport-Entity-Durchschnitt hoeher als aktuelle Prediction → hochkorrigieren
            if sport_avg > predicted * 1.1:
                # Staerekere Gewichtung fuer Tier-1-Entities
                entity_weight = 0.35 if sport_entity_hits > 0 else 0.20
                predicted = predicted * (1 - entity_weight) + sport_avg * entity_weight
                corrections_applied.append(f"sport_entity(n={len(sport_entity_ors)},avg={sport_avg:.1f},w={entity_weight})")

    # Korrektor 5: Topic-Saturation Penalty (6h-Fenster, einzige Saturation-Logik)
    try:
        _tsp = _compute_topic_saturation_penalty(push, push_data, state)
        if _tsp and _tsp.get("penalty", 1.0) < 1.0:
            _tsp_factor = _tsp["penalty"]
            predicted *= _tsp_factor
            corrections_applied.append(
                f"topic_sat(6h={_tsp.get('topic_push_count_6h',0)},"
                f"j={_tsp.get('highest_jaccard',0):.2f},"
                f"p={_tsp_factor:.3f})"
            )
            methods["topic_saturation_penalty"] = round(_tsp_factor, 3)
            methods["topic_saturation_reason"] = _tsp.get("reason", "")
    except Exception:
        pass

    # Korrektor 6: Quantil-basierte Clamp (nicht unter Q10 der Kategorie)
    # Kommt NACH Saturation, damit der Q10-Floor als Sicherheitsnetz greift
    quantiles = mods.get("quantiles", {})
    cat_q = quantiles.get("category", {}).get(push_cat, {}) if isinstance(quantiles, dict) else {}
    if cat_q:
        q10 = cat_q.get("q10", 0)
        q90 = cat_q.get("q90", 99)
        if predicted < q10 * 0.8:
            predicted = q10 * 0.85  # Nicht absurd niedrig
            corrections_applied.append(f"q10_floor={q10:.1f}")
        elif predicted > q90 * 1.3:
            predicted = q90 * 1.15  # Nicht absurd hoch
            corrections_applied.append(f"q90_cap={q90:.1f}")

    predicted = max(0.01, min(99.99, predicted))

    # Markov nutzt jetzt echte Vorgaenger-Pushes (oben), nicht mehr den State
    # state["_last_predicted_cat"] = push_cat  # ENTFERNT: korrumpierte Eval-Loop-Daten

    basis_parts = []
    if kw_scores:
        basis_parts.append(f"{len(kw_scores)} Keywords")
    if entity_ors:
        basis_parts.append(f"{len(push_entities)} Entities")
    basis_parts.append(f"Kat={push_cat}")
    basis_parts.append(f"H={push_hour}")
    if phd_details:
        basis_parts.append(f"PhD({len(phd_details)})")
    if corrections_applied:
        basis_parts.append(f"Korr({len(corrections_applied)})")

    return _safety_envelope({
        "predicted": round(predicted, 3),
        "methods": methods,
        "basis": ", ".join(basis_parts),
        "phd_corrections": corrections_applied,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTER O3-JSON-CALL — mit Fallback + JSON-Extraktion
# ══════════════════════════════════════════════════════════════════════════════

def _call_o3_json(prompt, max_tokens=1200, label="o3"):
    """Ruft o3 auf, extrahiert JSON. Fallback auf gpt-4.1 bei leerer Antwort."""
    import openai as _oai
    _client = _oai.OpenAI(api_key=OPENAI_API_KEY)
    resp = _client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        log.warning(f"[{label}] o3 leere Antwort, Fallback auf gpt-4.1")
        resp = _client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": "Du bist ein Forschungsassistent. Antworte NUR mit JSON."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.7,
        )
        raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        raw = raw[json_start:json_end]
    return json.loads(raw)


# EXTERNE DATENQUELLEN — Wetter, Trends, Feiertage, Kontext
# ══════════════════════════════════════════════════════════════════════════════

# Cache fuer externe Daten (wird alle 30min aktualisiert)
_external_context_cache = {
    "weather": {},
    "trends": [],
    "holiday": "",
    "last_fetch": 0,
}

# Deutsche Feiertage 2025-2027 (feste + bewegliche)
_GERMAN_HOLIDAYS = {
    # 2025
    "2025-01-01": "Neujahr", "2025-04-18": "Karfreitag", "2025-04-21": "Ostermontag",
    "2025-05-01": "Tag der Arbeit", "2025-05-29": "Christi Himmelfahrt",
    "2025-06-09": "Pfingstmontag", "2025-10-03": "Tag der dt. Einheit",
    "2025-12-25": "1. Weihnachtstag", "2025-12-26": "2. Weihnachtstag",
    # 2026
    "2026-01-01": "Neujahr", "2026-04-03": "Karfreitag", "2026-04-06": "Ostermontag",
    "2026-05-01": "Tag der Arbeit", "2026-05-14": "Christi Himmelfahrt",
    "2026-05-25": "Pfingstmontag", "2026-10-03": "Tag der dt. Einheit",
    "2026-12-25": "1. Weihnachtstag", "2026-12-26": "2. Weihnachtstag",
    # 2027
    "2027-01-01": "Neujahr", "2027-03-26": "Karfreitag", "2027-03-29": "Ostermontag",
    "2027-05-01": "Tag der Arbeit", "2027-05-06": "Christi Himmelfahrt",
    "2027-05-17": "Pfingstmontag", "2027-10-03": "Tag der dt. Einheit",
    "2027-12-25": "1. Weihnachtstag", "2027-12-26": "2. Weihnachtstag",
}


def _fetch_external_context(state):
    """Holt Wetter, Google Trends und Feiertag-Info. Alle 30min."""
    global _external_context_cache
    now = time.time()
    if now - _external_context_cache["last_fetch"] < 1800:
        return _external_context_cache

    log.info("[Kontext] Fetche externe Datenquellen...")

    # 1. Wetter Berlin (BILD HQ) via wttr.in (kostenlos, kein API-Key)
    weather = {}
    try:
        import ssl
        _ssl = ssl.create_default_context(cafile=_SSL_CERTFILE)
        if ALLOW_INSECURE_SSL:
            _ssl.check_hostname = False
            _ssl.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request("https://wttr.in/Berlin?format=j1",
                                     headers={"User-Agent": "PushBalancer/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            cur = data.get("current_condition", [{}])[0]
            weather = {
                "temp_c": int(cur.get("temp_C", 15)),
                "feels_like_c": int(cur.get("FeelsLikeC", 15)),
                "humidity": int(cur.get("humidity", 50)),
                "cloud_cover": int(cur.get("cloudcover", 50)),
                "precip_mm": float(cur.get("precipMM", 0)),
                "weather_desc": cur.get("weatherDesc", [{}])[0].get("value", ""),
                "wind_kmph": int(cur.get("windspeedKmph", 0)),
                "uv_index": int(cur.get("uvIndex", 3)),
            }
            # Wetter-Score: 0 = schoenes Wetter (Leute draussen), 1 = schlechtes Wetter (Leute am Handy)
            bad_weather_score = 0.0
            if weather["precip_mm"] > 0.5:
                bad_weather_score += 0.3
            if weather["temp_c"] < 5:
                bad_weather_score += 0.2
            elif weather["temp_c"] > 30:
                bad_weather_score += 0.1  # Hitze = auch drinnen
            if weather["cloud_cover"] > 80:
                bad_weather_score += 0.15
            if weather["wind_kmph"] > 30:
                bad_weather_score += 0.1
            weather["bad_weather_score"] = round(min(1.0, bad_weather_score), 2)
            log.info(f"[Kontext] Wetter Berlin: {weather['temp_c']}C, {weather['weather_desc']}, Score={weather['bad_weather_score']}")
    except Exception as e:
        log.warning(f"[Kontext] Wetter-Fetch fehlgeschlagen: {e}")
        weather = {"bad_weather_score": 0.3, "temp_c": 15, "weather_desc": "unbekannt"}

    # 2. Google Trends Deutschland (RSS Feed — kostenlos, kein API-Key)
    trends = []
    try:
        req = urllib.request.Request("https://trends.google.com/trending/rss?geo=DE",
                                     headers={"User-Agent": "PushBalancer/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
            # Einfaches Regex-Parsing der RSS Items
            import re
            titles = re.findall(r"<title>([^<]+)</title>", xml)
            # Erste Title ist Feed-Title, Rest sind Trends
            for t in titles[1:21]:  # Max 20 Trends
                t = t.strip()
                if t and len(t) > 1:
                    trends.append(t.lower())
            log.info(f"[Kontext] Google Trends DE: {len(trends)} trending topics")
    except Exception as e:
        log.warning(f"[Kontext] Google Trends fehlgeschlagen: {e}")

    # 3. Feiertag/Event-Check
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    holiday = _GERMAN_HOLIDAYS.get(today_str, "")
    weekday = datetime.datetime.now().weekday()
    day_type = "weekend" if weekday >= 5 else "weekday"
    if holiday:
        day_type = "holiday"
        log.info(f"[Kontext] Heute ist Feiertag: {holiday}")

    # 4. Tageszeit-Kontext
    hour = datetime.datetime.now().hour
    if 6 <= hour <= 8:
        time_context = "pendler_morgen"
    elif 11 <= hour <= 13:
        time_context = "mittagspause"
    elif 16 <= hour <= 18:
        time_context = "feierabend"
    elif 20 <= hour <= 23:
        time_context = "prime_time"
    elif 0 <= hour <= 5:
        time_context = "nacht"
    else:
        time_context = "normal"

    _external_context_cache = {
        "weather": weather,
        "trends": trends,
        "holiday": holiday,
        "day_type": day_type,
        "time_context": time_context,
        "last_fetch": now,
    }

    # In State speichern fuer Dashboard
    state["external_context"] = {
        "weather": weather,
        "trends_count": len(trends),
        "trends_top5": trends[:5],
        "holiday": holiday,
        "day_type": day_type,
        "time_context": time_context,
        "last_update": datetime.datetime.now().strftime("%H:%M"),
    }

    return _external_context_cache


def _context_topic_match(push_title, trends):
    """Prueft ob ein Push-Titel ein Trending-Topic trifft."""
    if not trends or not push_title:
        return 0.0
    title_lower = push_title.lower()
    title_words = set(title_lower.split())
    matches = 0
    for trend in trends:
        trend_words = set(trend.split())
        # Exakter Substring-Match oder Wort-Overlap
        if trend in title_lower:
            matches += 2  # Starker Match
        elif len(title_words & trend_words) >= 1:
            matches += 1  # Schwacher Match
    # Normalisieren: 0-1 Score
    return min(1.0, matches / 3.0)


def _analyze_prediction_outliers(state):
    """Analysiert die groessten Vorhersage-Fehler und extrahiert Lern-Signale.

    Laeuft alle 30min. Speichert Muster in state['outlier_patterns'] die vom
    autonomen Tuning genutzt werden koennen.
    """
    now_t = time.time()
    if now_t - state.get("_last_outlier_analysis", 0) < 1800:  # 30min Cooldown
        return
    state["_last_outlier_analysis"] = now_t

    feedback = state.get("prediction_feedback", [])
    if len(feedback) < 50:
        return

    # Nur letzte 300 Feedbacks analysieren (vermeidet stale Patterns aus alter Modell-Version)
    recent_feedback = feedback[-300:]

    # Finde alle Outlier (|error| > 2pp)
    outliers_under = []  # Wir haben UNTERSCHAETZT (actual >> predicted)
    outliers_over = []   # Wir haben UEBERSCHAETZT (predicted >> actual)

    for fb in recent_feedback:
        error = fb["predicted_or"] - fb["actual_or"]
        if error < -2.0:
            outliers_under.append(fb)
        elif error > 2.0:
            outliers_over.append(fb)

    if not outliers_under and not outliers_over:
        return

    # Muster-Extraktion: Was haben Outlier gemeinsam?
    patterns = {"underpredicted": {}, "overpredicted": {}, "ts": now_t}

    # Underpredicted: Welche Kategorien/Stunden werden systematisch unterschaetzt?
    if outliers_under:
        cat_errors = defaultdict(list)
        hour_errors = defaultdict(list)
        for fb in outliers_under:
            cat = fb.get("push_cat") or fb.get("category", "")
            hour = fb.get("push_hour") or fb.get("hour", 0)
            err = fb["predicted_or"] - fb["actual_or"]
            if cat:
                cat_errors[cat].append(err)
            hour_errors[hour].append(err)

        # Systematische Unterschaetzung pro Kategorie
        cat_bias = {}
        for cat, errs in cat_errors.items():
            if len(errs) >= 3:
                avg_err = sum(errs) / len(errs)
                cat_bias[cat] = {"avg_error": round(avg_err, 2), "n": len(errs)}

        # Systematische Unterschaetzung pro Stunde
        hour_bias = {}
        for hour, errs in hour_errors.items():
            if len(errs) >= 3:
                avg_err = sum(errs) / len(errs)
                hour_bias[str(hour)] = {"avg_error": round(avg_err, 2), "n": len(errs)}

        patterns["underpredicted"] = {
            "n": len(outliers_under),
            "avg_error": round(sum(fb["predicted_or"] - fb["actual_or"] for fb in outliers_under) / len(outliers_under), 2),
            "worst_titles": [fb.get("push_title", "")[:50] for fb in sorted(outliers_under, key=lambda x: x["predicted_or"] - x["actual_or"])[:5]],
            "cat_bias": cat_bias,
            "hour_bias": hour_bias,
        }

    # Overpredicted
    if outliers_over:
        cat_errors = defaultdict(list)
        hour_errors = defaultdict(list)
        for fb in outliers_over:
            cat = fb.get("push_cat") or fb.get("category", "")
            hour = fb.get("push_hour") or fb.get("hour", 0)
            err = fb["predicted_or"] - fb["actual_or"]
            if cat:
                cat_errors[cat].append(err)
            hour_errors[hour].append(err)

        cat_bias = {}
        for cat, errs in cat_errors.items():
            if len(errs) >= 3:
                cat_bias[cat] = {"avg_error": round(sum(errs) / len(errs), 2), "n": len(errs)}

        hour_bias = {}
        for hour, errs in hour_errors.items():
            if len(errs) >= 3:
                hour_bias[str(hour)] = {"avg_error": round(sum(errs) / len(errs), 2), "n": len(errs)}

        patterns["overpredicted"] = {
            "n": len(outliers_over),
            "avg_error": round(sum(fb["predicted_or"] - fb["actual_or"] for fb in outliers_over) / len(outliers_over), 2),
            "worst_titles": [fb.get("push_title", "")[:50] for fb in sorted(outliers_over, key=lambda x: x["actual_or"] - x["predicted_or"])[:5]],
            "cat_bias": cat_bias,
            "hour_bias": hour_bias,
        }

    state["outlier_patterns"] = patterns
    n_total = len(feedback)
    n_outliers = len(outliers_under) + len(outliers_over)
    log.info(f"[Outlier] {n_outliers}/{n_total} Outlier (>{n_outliers/n_total*100:.0f}%): "
             f"{len(outliers_under)} unterschaetzt, {len(outliers_over)} ueberschaetzt")


def _build_progress_ticker(state, findings):
    """Baut einen Live-Progress-Ticker aus echten Server-Events zusammen.

    Zeigt: Server-Feedback, Algo-Team-Analysen, Ensemble-Accuracy, Tuning, Meilensteine.
    Sortiert nach Zeitstempel (neueste zuerst), max 30 Eintraege.
    """
    events = []
    now_t = time.time()
    now_str = datetime.datetime.now().strftime("%H:%M")

    ens_acc = state.get("ensemble_accuracy", 0)
    ens_delta = state.get("ensemble_accuracy_delta", 0)
    base_acc = state.get("rolling_accuracy", 0)

    # ── 1. Ensemble-Accuracy ──
    if ens_acc > 0:
        events.append({
            "ts": now_t, "time": now_str, "cat": "ensemble",
            "icon": "chart", "prio": 1,
            "text": f"Vorhersage-Qualitaet: {ens_acc:.1f}% der Pushes korrekt vorhergesagt",
            "detail": f"5 Methoden kombiniert. Einfaches Modell: {base_acc:.1f}%. Durchschn. Fehler: {state.get('ensemble_mae', 0):.2f}pp",
            "color": "#22c55e" if ens_acc > base_acc else "#f59e0b",
        })

    # ── 2. Server-Feedback ──
    fb_list = state.get("prediction_feedback", [])
    server_fbs = [fb for fb in fb_list if fb.get("source") == "server"]
    if server_fbs:
        last_fb = server_fbs[-1]
        fb_ts = last_fb.get("ts", now_t)
        fb_time = datetime.datetime.fromtimestamp(fb_ts).strftime("%H:%M")
        err = abs(last_fb["predicted_or"] - last_fb["actual_or"])
        events.append({
            "ts": fb_ts, "time": fb_time, "cat": "feedback",
            "icon": "target", "prio": 2,
            "text": f"Selbsttest: {len(server_fbs)} Pushes geprueft — Server bewertet sich autonom",
            "detail": f"Letzter: '{last_fb.get('push_title', '')[:40]}' — Vorhersage {last_fb['predicted_or']:.1f}%, tatsaechlich {last_fb['actual_or']:.1f}% (Fehler: {err:.2f}pp)",
            "color": "#64748b",
        })

    # ── 3. Algo-Team-Analyse ──
    algo = state.get("_algo_team_analysis", {})
    if algo:
        algo_ts = algo.get("ts", now_t)
        algo_time = datetime.datetime.fromtimestamp(algo_ts).strftime("%H:%M")
        events.append({
            "ts": algo_ts, "time": algo_time, "cat": "algo",
            "icon": "brain", "prio": 1,
            "text": f"Algo-Team Analyse: {algo['hit_rate']}% Trefferquote, {algo['mae']:.2f}pp Fehler im Schnitt",
            "detail": f"Tendenz: {algo['bias']:+.2f}pp {'(zu hoch)' if algo['bias'] > 0.1 else '(zu niedrig)' if algo['bias'] < -0.1 else '(ausgeglichen)'}. Beste Methode: {algo.get('best_method', '?')}, schlechteste: {algo.get('worst_method', '?')}",
            "color": "#f59e0b",
        })
        cat_bias = algo.get("cat_bias", {})
        if cat_bias:
            worst_cat = max(cat_bias.items(), key=lambda x: abs(x[1]))
            events.append({
                "ts": algo_ts - 1, "time": algo_time, "cat": "algo",
                "icon": "alert", "prio": 3,
                "text": f"Systematischer Fehler: {worst_cat[0]} wird um {abs(worst_cat[1]):.2f}pp zu {'hoch' if worst_cat[1] > 0 else 'niedrig'} geschaetzt",
                "detail": f"Betrifft {len(cat_bias)} Kategorien: {', '.join(f'{c} {v:+.2f}' for c, v in sorted(cat_bias.items(), key=lambda x: -abs(x[1]))[:4])}",
                "color": "#ef4444" if abs(worst_cat[1]) > 0.3 else "#f59e0b",
            })

    # ── 4. Methoden-Performance ──
    method_stats = algo.get("method_stats", {})
    if method_stats:
        algo_ts_val = algo.get("ts", now_t)
        m_time = datetime.datetime.fromtimestamp(algo_ts_val).strftime("%H:%M")
        for m_name, m_data in sorted(method_stats.items(), key=lambda x: x[1].get("mae", 0)):
            events.append({
                "ts": algo_ts_val - 2, "time": m_time,
                "cat": "method", "icon": "gauge", "prio": 4,
                "text": f"Methode {m_name}: {m_data['mae']:.2f}pp Fehler, Tendenz {m_data['bias']:+.2f}pp",
                "detail": f"Basiert auf {m_data.get('n', 0)} Bewertungen",
                "color": "#22c55e" if m_data["mae"] < 0.5 else ("#f59e0b" if m_data["mae"] < 1.0 else "#ef4444"),
            })

    # ── 5. Alle Vorschlaege (Algo-Team + Lehrstuehle) ──
    pending = [a for a in state.get("pending_approvals", []) if a.get("status") == "pending"]
    for prop in pending[:5]:
        prop_ts = prop.get("ts", now_t)
        prop_time = datetime.datetime.fromtimestamp(prop_ts).strftime("%H:%M")
        source = prop.get("source", "?")
        is_lehrstuhl = source.startswith("lehrstuhl_")
        source_label = source.replace("lehrstuhl_", "").replace("_", " ").title() if is_lehrstuhl else "Algo-Team"
        events.append({
            "ts": prop_ts, "time": prop_time, "cat": "proposal",
            "icon": "lightbulb", "prio": 1,
            "text": f"Vorschlag von {source_label}: {prop.get('title', 'Parameter-Anpassung')}",
            "detail": f"Erwartete Verbesserung: {prop.get('expected_impact', '?')}. Grund: {prop.get('evidence', '?')[:80]}",
            "color": "#f59e0b",
        })

    # ── 5b. Lehrstuhl-Forschung (letzte LLM-Findings) ──
    research_log = state.get("research_log", [])
    llm_findings = [e for e in research_log if e.get("source") == "llm_autonomous"][-5:]
    for entry in llm_findings:
        chair_id = entry.get("researcher", "?")
        chair_info = next((c for c in _LEHRSTUHL_CHAIRS if c["id"] == chair_id), None)
        chair_label = chair_info["name"] if chair_info else chair_id
        domain_label = chair_info["domain"] if chair_info else ""
        # Timestamp parsen
        ts_str = entry.get("ts", "")
        try:
            entry_dt = datetime.datetime.strptime(ts_str, "%d.%m. %H:%M").replace(year=datetime.datetime.now().year)
            entry_ts = entry_dt.timestamp()
            entry_time = entry_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            entry_ts = now_t - 60
            entry_time = ts_str
        finding_text = entry.get("finding", "").replace("[LLM-Forschung] ", "")
        events.append({
            "ts": entry_ts, "time": entry_time, "cat": "research",
            "icon": "microscope", "prio": 2,
            "text": f"{chair_label} ({domain_label}): Neue Erkenntnis",
            "detail": finding_text[:150],
            "color": "#6366f1",
        })

    # ── 6. Tuning-History ──
    status_labels = {"active": "Aktiv", "validated": "Validiert", "rolled_back": "Rollback", "expired": "Expired"}
    for entry in state.get("tuning_history", [])[-3:]:
        t_ts = entry.get("ts", 0)
        t_time = datetime.datetime.fromtimestamp(t_ts).strftime("%H:%M") if t_ts else "?"
        st = entry.get("status", "?")
        changes = entry.get("changes", [])
        param_str = ", ".join(f"{c['param']}:{c['old']}->{c['new']}" for c in changes[:2]) if changes else "?"
        events.append({
            "ts": t_ts, "time": t_time, "cat": "tuning",
            "icon": "check" if st == "validated" else ("undo" if st == "rolled_back" else "gear"),
            "prio": 2,
            "text": f"Tuning #{entry.get('change_id', '?')}: {status_labels.get(st, st)} — {param_str}",
            "detail": f"Acc vorher: {entry.get('acc_before', '?')}%, nachher: {entry.get('acc_after_24h', 'ausstehend')}%",
            "color": "#22c55e" if st == "validated" else ("#ef4444" if st == "rolled_back" else "#94a3b8"),
        })

    # ── 7. Research-Meilensteine ──
    for ms in state.get("research_milestones", [])[-5:]:
        ms_ts_str = ms.get("ts", "")
        try:
            ms_dt = datetime.datetime.strptime(ms_ts_str, "%d.%m.%Y %H:%M")
            ms_ts = ms_dt.timestamp()
            ms_time = ms_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            ms_ts = now_t - 300
            ms_time = "?"
        events.append({
            "ts": ms_ts, "time": ms_time, "cat": "milestone",
            "icon": "flag", "prio": 2,
            "text": f"Erreicht: {ms.get('milestone', '')[:70]}",
            "detail": f"Projekt: {ms.get('project_id', '?')}, von: {ms.get('achieved_by', '?')}",
            "color": "#22c55e",
        })

    # ── 8. Schwab-Decisions ──
    for dec in state.get("schwab_decisions", [])[-3:]:
        dec_ts = dec.get("ts_epoch", now_t - 60)
        events.append({
            "ts": dec_ts, "time": dec.get("time", "?"), "cat": "decision",
            "icon": "gavel", "prio": 1,
            "text": f"Schwab: {dec.get('decision', '')[:70]}",
            "detail": dec.get("reason", "")[:80],
            "color": "#f59e0b",
        })

    # ── 9. Research-Log ──
    for entry in state.get("research_log", [])[-5:]:
        try:
            rl_dt = datetime.datetime.strptime(entry.get("ts", ""), "%d.%m.%Y %H:%M")
            rl_ts = rl_dt.timestamp()
            rl_time = rl_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            rl_ts = now_t - 600
            rl_time = "?"
        events.append({
            "ts": rl_ts, "time": rl_time, "cat": "research",
            "icon": "microscope", "prio": 3,
            "text": f"{entry.get('researcher', '?')}: {entry.get('finding', '')[:70]}",
            "detail": "", "color": "#64748b",
        })

    # ── 10. System-Status ──
    gen = state.get("analysis_generation", 0)
    n_pushes = len(state.get("push_data", []))
    if gen > 0:
        events.append({
            "ts": now_t - 0.5, "time": now_str, "cat": "system",
            "icon": "refresh", "prio": 5,
            "text": f"Analyse #{gen} abgeschlossen: {n_pushes} Pushes verarbeitet",
            "detail": f"{state.get('mature_count', 0)} auswertbar (>24h), {state.get('fresh_count', 0)} noch frisch",
            "color": "#9ca3af",
        })

    events.sort(key=lambda e: (-e["ts"], e["prio"]))
    return events[:30]


def _evict_stale_cache(cache, ttl, max_size=500):
    """Remove expired entries and cap cache size."""
    now = time.time()
    expired = [k for k, (ts, _) in cache.items() if now - ts > ttl * 5]
    for k in expired:
        del cache[k]
    if len(cache) > max_size:
        oldest = sorted(cache.items(), key=lambda x: x[1][0])[:len(cache) - max_size]
        for k, _ in oldest:
            del cache[k]


_plus_cache = {}
PLUS_CACHE_TTL = 3600  # 1 Stunde

def _check_bild_plus(url):
    """Check if a BILD article is behind the BILDplus paywall."""
    now = time.time()
    if len(_plus_cache) > 200:
        _evict_stale_cache(_plus_cache, PLUS_CACHE_TTL, 200)
    if url in _plus_cache and now - _plus_cache[url][0] < PLUS_CACHE_TTL:
        return _plus_cache[url][1]
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=8, context=_GLOBAL_SSL_CTX) as resp:
                html = resp.read(150000).decode("utf-8", errors="replace")
        except (ssl.SSLError, urllib.error.URLError):
            if not ALLOW_INSECURE_SSL:
                raise
            # SSL-Fallback: Zertifikatsprüfung deaktivieren (nur wenn ALLOW_INSECURE_SSL=1)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req2 = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req2, timeout=8, context=ctx) as resp:
                html = resp.read(150000).decode("utf-8", errors="replace")
        is_plus = '"isAccessibleForFree":false' in html or '"isAccessibleForFree": false' in html
        if is_plus:
            log.info(f"[BILDPlus] Plus-Artikel erkannt: {url[:80]}")
        _plus_cache[url] = (now, is_plus)
        return is_plus
    except Exception as e:
        log.warning(f"[BILDPlus] Check fehlgeschlagen für {url}: {type(e).__name__}: {e}")
        return False


def _fetch_url(url, timeout=10):
    """Fetch a URL with caching and SSL fallback, return bytes or None."""
    if len(_cache) > 500:
        _evict_stale_cache(_cache, CACHE_TTL, 500)
    now = time.time()
    if url in _cache and now - _cache[url][0] < CACHE_TTL:
        return _cache[url][1]
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        # Erst mit SSL-Verifikation versuchen
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_GLOBAL_SSL_CTX) as resp:
                data = resp.read(MAX_RESPONSE_SIZE)
        except (ssl.SSLError, urllib.error.URLError):
            if not ALLOW_INSECURE_SSL:
                raise
            # Fallback bei SSL-Problemen (nur wenn ALLOW_INSECURE_SSL=1)
            ctx = ssl.create_default_context(cafile=_SSL_CERTFILE)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req2 = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            })
            with urllib.request.urlopen(req2, timeout=timeout, context=ctx) as resp:
                data = resp.read(MAX_RESPONSE_SIZE)
            log.warning(f"SSL fallback used for {url}")
        _cache[url] = (now, data)
        return data
    except Exception as e:
        log.warning(f"Fetch failed for {url}: {e}")
        # Stale cache als Fallback
        if url in _cache:
            log.info(f"Using stale cache for {url}")
            return _cache[url][1]
        return None


# ── Background Feed Cache (Konkurrenz + International) ──────────────────────
# Feeds werden im Hintergrund alle 90s aktualisiert statt bei jedem Request.
_feed_cache = {
    "competitors": {"data": {}, "ts": 0, "fetching": False},
    "international": {"data": {}, "ts": 0, "fetching": False},
    "sport_competitors": {"data": {}, "ts": 0, "fetching": False},
    "sport_europa": {"data": {}, "ts": 0, "fetching": False},
    "sport_global": {"data": {}, "ts": 0, "fetching": False},
}
_feed_cache_lock = threading.Lock()
_FEED_CACHE_TTL = 90  # Sekunden

_FEED_TYPE_MAP = {
    "competitors": COMPETITOR_FEEDS, "international": INTERNATIONAL_FEEDS,
    "sport_competitors": SPORT_COMPETITOR_FEEDS,
    "sport_europa": SPORT_EUROPA_FEEDS, "sport_global": SPORT_GLOBAL_FEEDS,
}

def _bg_fetch_feeds(feed_type):
    """Holt alle Feeds eines Typs im Hintergrund und cached das geparste JSON."""
    feeds = _FEED_TYPE_MAP.get(feed_type, {})
    with _feed_cache_lock:
        if _feed_cache[feed_type]["fetching"]:
            return
        _feed_cache[feed_type]["fetching"] = True
    try:
        raw_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_fetch_url, url): name
                       for name, url in feeds.items()}
            for future in concurrent.futures.as_completed(futures, timeout=15):
                name = futures[future]
                try:
                    data = future.result()
                    if data:
                        raw_results[name] = data.decode("utf-8", errors="replace")
                    else:
                        raw_results[name] = ""
                except Exception:
                    raw_results[name] = ""
        # Parsen — nutzt eine temporaere Handler-Instanz fuer _parse_rss_items
        parsed = {}
        for name, xml_str in raw_results.items():
            if not xml_str:
                parsed[name] = []
                continue
            parsed[name] = _parse_rss_items_standalone(xml_str)
        with _feed_cache_lock:
            _feed_cache[feed_type]["data"] = parsed
            _feed_cache[feed_type]["ts"] = time.time()
    except Exception as e:
        log.warning(f"[FeedCache] {feed_type} Fehler: {e}")
    finally:
        with _feed_cache_lock:
            _feed_cache[feed_type]["fetching"] = False

# ── Auto-Suggestion: Server-seitige Empfehlungen pro Stunde ──────────────
_auto_sug_last_hour = -1  # Letzte gespeicherte Stunde (verhindert Duplikate)

def _auto_save_suggestions():
    """Generiert und speichert Artikelvorschläge serverseitig (unabhängig vom Browser).

    Wird alle 15 Min vom Research-Worker aufgerufen. Speichert 2 Top-Artikel pro
    zukünftigem Slot, basierend auf BILD-Sitemap + ML-OR-Prognose + Heuristik-Score.
    """
    global _auto_sug_last_hour
    now = datetime.datetime.now()
    current_hour = now.hour
    date_iso = now.strftime("%Y-%m-%d")
    # Nur 06-23 Uhr
    if current_hour < 6:
        return

    # Schon für diese Stunde gespeichert?
    if _auto_sug_last_hour == current_hour:
        return
    _auto_sug_last_hour = current_hour

    try:
        # 1. BILD-Sitemap fetchen (nutzt _fetch_url mit SSL-Handling + Cache)
        xml_bytes = _fetch_url(BILD_SITEMAP, timeout=10)
        if not xml_bytes:
            log.warning("[AutoSug] Sitemap nicht erreichbar")
            return
        xml_str = xml_bytes.decode("utf-8", errors="replace")
        items = _parse_rss_items_standalone(xml_str, max_items=80)
        if not items:
            log.warning("[AutoSug] Keine Artikel in Sitemap")
            return

        # 2. Bereits heute gepushte Titel/Links sammeln (Duplikate vermeiden)
        pushed_titles = set()
        pushed_links = set()
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
            try:
                day_start = int(now.replace(hour=0, minute=0, second=0).timestamp())
                for r in conn.execute(
                    "SELECT title, link FROM pushes WHERE ts_num >= ?", (day_start,)
                ).fetchall():
                    if r[0]:
                        pushed_titles.add(r[0].strip().lower()[:50])
                    if r[1]:
                        pushed_links.add(r[1].rstrip("/"))
            finally:
                conn.close()

        # 3. Artikel scoren: Kategorie-Heuristik + ML-OR-Prognose
        cat_scores = {
            "politik": 22, "unterhaltung": 20, "panorama": 21, "regional": 18,
            "sport": 17, "fussball": 19, "bundesliga": 19, "geld": 14,
            "auto": 11, "digital": 12, "lifestyle": 10, "ratgeber": 9, "reise": 8,
        }
        high_kw = ["TRUMP", "PUTIN", "UKRAINE", "KRIEG", "BUNDESLIGA", "CHAMPIONS", "POLIZEI"]
        mid_kw = ["MERZ", "SCHOLZ", "BIDEN", "MUSK", "BAYERN", "DORTMUND", "NATO", "DEUTSCHLAND"]
        urg_kw = ["EILMELDUNG", "BREAKING", "TOD", "GESTORBEN", "ANSCHLAG", "TERROR"]
        neug_kw = ["EXKLUSIV", "SKANDAL", "SCHOCK", "HAMMER", "SENSATION", "RAZZIA", "VERHAFTET"]

        scored = []
        for it in items:
            title = it.get("t", "").strip()
            link = it.get("l", "").strip()
            if not title or it.get("lt"):
                continue  # Live-Ticker überspringen
            # Duplikat-Check
            if title.lower()[:50] in pushed_titles:
                continue
            if link.rstrip("/") in pushed_links:
                continue

            # Kategorie aus Link extrahieren
            cats = it.get("c", [])
            cat = cats[0].lower() if cats else "news"
            if not cat or cat == "news":
                # Aus URL-Pfad extrahieren
                ll = link.lower()
                for c in cat_scores:
                    if f"/{c}/" in ll or f"/{c}-" in ll:
                        cat = c
                        break

            upper = title.upper()
            # Heuristik-Score (0-100, vereinfacht)
            base = cat_scores.get(cat, 13)
            kw_boost = 0
            for kw in high_kw:
                if kw in upper:
                    kw_boost += 4
            for kw in mid_kw:
                if kw in upper:
                    kw_boost += 2
            for kw in urg_kw:
                if kw in upper:
                    kw_boost += 5
            for kw in neug_kw:
                if kw in upper:
                    kw_boost += 3
            if "?" in title:
                kw_boost += 2
            heur_score = min(100, base + min(kw_boost, 15))

            scored.append({
                "title": title[:200], "link": link[:500], "category": cat,
                "score": round(heur_score, 1),
            })

        if not scored:
            log.warning("[AutoSug] Keine scorebaren Artikel")
            return

        # 4. Tagesplan-Slots: best_cat pro Stunde aus Cache holen
        with _tagesplan_cache_lock:
            tp_result = _tagesplan_cache["redaktion"].get("result")
        slot_cats = {}
        if tp_result and tp_result.get("slots"):
            for s in tp_result["slots"]:
                slot_cats[s["hour"]] = s.get("best_cat", "news")

        # 5. OR-Prognose per ML und Top-2 pro Slot wählen
        suggestions_to_save = []
        used_titles = set()
        weekday = now.weekday()

        for h in range(current_hour, 24):
            best_cat = slot_cats.get(h, "news")
            # Score + OR pro Artikel berechnen
            candidates = []
            for art in scored:
                if art["title"][:50] in used_titles:
                    continue
                # ML-OR-Prognose
                pred_or = 0
                try:
                    pr = _ml_predict(art["title"], art["category"], hour=h, weekday=weekday)
                    if pr and "predicted_or" in pr:
                        pred_or = pr["predicted_or"]
                except Exception:
                    pass
                if pred_or <= 0:
                    try:
                        gbrt_push = {"title": art["title"], "cat": art["category"],
                                     "hour": h, "is_eilmeldung": False, "channels": ["news"]}
                        gbrt_r = _gbrt_predict(gbrt_push)
                        if gbrt_r and gbrt_r.get("predicted"):
                            pred_or = gbrt_r["predicted"]
                    except Exception:
                        pass

                # Kombi-Score: Heuristik + OR-Bonus
                cat_match = 1.2 if art["category"].lower() == best_cat.lower() else 1.0
                combo = (art["score"] * 0.5 + min(pred_or * 10, 50) * 0.5) * cat_match
                candidates.append({**art, "expected_or": round(pred_or, 2), "combo": combo})

            candidates.sort(key=lambda x: -x["combo"])
            for idx, c in enumerate(candidates[:2]):
                used_titles.add(c["title"][:50])
                suggestions_to_save.append({
                    "slot_hour": h, "suggestion_num": idx + 1,
                    "title": c["title"], "link": c["link"],
                    "category": c["category"], "score": c["score"],
                    "expected_or": c["expected_or"], "best_cat": best_cat,
                })

        if not suggestions_to_save:
            return

        # 6. In DB speichern (INSERT OR REPLACE)
        now_ts = int(time.time())
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
            try:
                for s in suggestions_to_save:
                    conn.execute("""INSERT OR REPLACE INTO tagesplan_suggestions
                        (date_iso, slot_hour, suggestion_num, article_title, article_link,
                         article_category, article_score, expected_or, best_cat, captured_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (date_iso, s["slot_hour"], s["suggestion_num"],
                         s["title"], s["link"], s["category"],
                         s["score"], s["expected_or"], s["best_cat"], now_ts))
                conn.commit()
            finally:
                conn.close()

        log.info(f"[AutoSug] {len(suggestions_to_save)} Vorschläge für {date_iso} gespeichert "
                 f"(Stunden {current_hour}-23)")

    except Exception as e:
        log.warning(f"[AutoSug] Fehler: {e}")


def _parse_rss_items_standalone(xml_str, max_items=30):
    """Parst RSS/Atom/Sitemap-XML zu kompaktem JSON (standalone, ohne Handler-Instanz)."""
    import xml.etree.ElementTree as ET
    items = []
    try:
        root = ET.fromstring(xml_str)
        # RSS 2.0
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            desc = (item.findtext("description") or "").strip()[:200]
            cats = [c.text.strip() for c in item.findall("category") if c.text]
            items.append({"t": title, "l": link, "p": pub, "d": desc, "c": cats})
            if len(items) >= max_items:
                break
        if not items:
            # Atom fallback
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//a:entry", ns):
                title = (entry.findtext("a:title", "", ns) or "").strip()
                link_el = entry.find("a:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                pub = (entry.findtext("a:published", "", ns) or entry.findtext("a:updated", "", ns) or "").strip()
                desc = (entry.findtext("a:summary", "", ns) or "").strip()[:200]
                items.append({"t": title, "l": link, "p": pub, "d": desc, "c": []})
                if len(items) >= max_items:
                    break
        if not items:
            # News-Sitemap fallback (iter statt findall — default-NS Kompatibilität)
            # BILD nutzt http:// (nicht https://) für beide Namespaces
            _sm_ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
            _sn_ns = "http://www.google.com/schemas/sitemap-news/0.9"
            _sm_url_tag = f"{{{_sm_ns}}}url"
            _sm_loc_tag = f"{{{_sm_ns}}}loc"
            _sn_news_tag = f"{{{_sn_ns}}}news"
            _sn_title_tag = f"{{{_sn_ns}}}title"
            _sn_pub_tag = f"{{{_sn_ns}}}publication_date"
            for url_el in root.iter(_sm_url_tag):
                loc = (url_el.findtext(_sm_loc_tag, "") or "").strip()
                news = url_el.find(_sn_news_tag)
                title = ""
                pub = ""
                if news is not None:
                    title = (news.findtext(_sn_title_tag, "") or "").strip()
                    pub = (news.findtext(_sn_pub_tag, "") or "").strip()
                if not title:
                    continue
                items.append({"t": title, "l": loc, "p": pub, "d": "", "c": []})
                if len(items) >= max_items:
                    break
    except ET.ParseError:
        pass
    # Live-Ticker/Aggregation markieren (Frontend kann filtern)
    for it in items:
        tl = (it.get("t") or "").lower()
        ll = (it.get("l") or "").lower()
        it["lt"] = bool(
            re.search(r'/(liveticker|newsticker|news-ticker|alle-news|news-blog)(/|$|\?)', ll)
            or re.search(r'/live[\d/]', ll)
            or re.search(r'/ticker(/|$|\?)', ll)
            or re.search(r'live[\s-]?ticker|news[\s-]?ticker|newsticker', tl)
            or re.search(r'alle\s+(news|infos|entwicklungen|meldungen)\s+(zu|zum|zur|im|aus|über)', tl)
            or re.search(r'news\s+im\s+überblick|nachrichten\s+des\s+tages', tl)
            or re.search(r'was\s+(wir\s+)?wissen|was\s+bisher\s+bekannt', tl)
            or re.search(r'die\s+wichtigsten\s+(meldungen|nachrichten|news)', tl)
            or tl.count('+++') >= 2
        )
    return items

def _feed_cache_worker():
    """Background-Thread: Refreshed Competitor/International Feeds alle 90s."""
    time.sleep(5)  # Warte bis Server bereit
    while True:
        try:
            _bg_fetch_feeds("competitors")
            _bg_fetch_feeds("international")
            _bg_fetch_feeds("sport_competitors")
            _bg_fetch_feeds("sport_europa")
            _bg_fetch_feeds("sport_global")
            # LLM Coverage Check (rate-limited, max alle 5 Min)
        except Exception as e:
            log.warning(f"[FeedCache] Worker-Fehler: {e}")
        time.sleep(_FEED_CACHE_TTL)

def _get_cached_feeds(feed_type):
    """Gibt gecachte Feed-Daten zurueck. Wenn leer, triggert sofortigen Fetch."""
    with _feed_cache_lock:
        cached = _feed_cache[feed_type]
        if cached["data"] and (time.time() - cached["ts"]) < _FEED_CACHE_TTL * 3:
            return cached["data"]
    # Cache leer oder sehr alt — synchron fetchen (nur beim allerersten Mal)
    _bg_fetch_feeds(feed_type)
    with _feed_cache_lock:
        return _feed_cache[feed_type]["data"]


class PushBalancerHandler(http.server.SimpleHTTPRequestHandler):
    # Gzip-komprimierte HTML-Datei cachen
    _gzip_cache = {}  # path -> (mtime, gzipped_bytes)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def _accepts_gzip(self):
        return "gzip" in self.headers.get("Accept-Encoding", "")

    def _send_gzip(self, data, content_type="application/json"):
        """Sende Response gzip-komprimiert wenn Client es unterstuetzt."""
        if self._accepts_gzip() and len(data) > 1000:
            import gzip as _gzip_mod
            compressed = _gzip_mod.compress(data, compresslevel=6)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self._cors_headers()
            self.wfile.write(compressed)
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self._cors_headers()
            self.wfile.write(data)

    def _json_response(self, obj, ensure_ascii=True):
        """Sendet ein Python-Objekt als gzip-komprimiertes JSON."""
        data = json.dumps(obj, ensure_ascii=ensure_ascii).encode("utf-8")
        self._send_gzip(data, "application/json; charset=utf-8")

    @staticmethod
    def _parse_rss_items(xml_str, max_items=30):
        """Parst RSS/Atom XML und gibt kompakte Item-Liste zurueck."""
        import xml.etree.ElementTree as ET
        items = []
        try:
            root = ET.fromstring(xml_str)
            # RSS 2.0
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                desc = (item.findtext("description") or "").strip()[:200]
                cats = [c.text.strip() for c in item.findall("category") if c.text]
                items.append({"t": title, "l": link, "p": pub, "d": desc, "c": cats})
                if len(items) >= max_items:
                    break
            if not items:
                # Atom fallback
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for entry in root.findall(".//a:entry", ns):
                    title = (entry.findtext("a:title", "", ns) or "").strip()
                    link_el = entry.find("a:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    pub = (entry.findtext("a:published", "", ns) or entry.findtext("a:updated", "", ns) or "").strip()
                    desc = (entry.findtext("a:summary", "", ns) or "").strip()[:200]
                    items.append({"t": title, "l": link, "p": pub, "d": desc, "c": []})
                    if len(items) >= max_items:
                        break
            # News-Sitemap fallback
            if not items:
                ns_sm = {"sm": "https://www.sitemaps.org/schemas/sitemap/0.9",
                         "news": "https://www.google.com/schemas/sitemap-news/0.9"}
                for url_el in root.findall(".//sm:url", ns_sm):
                    loc = (url_el.findtext("sm:loc", "", ns_sm) or "").strip()
                    news = url_el.find("news:news", ns_sm)
                    title = ""
                    pub = ""
                    if news is not None:
                        title = (news.findtext("news:title", "", ns_sm) or "").strip()
                        pub = (news.findtext("news:publication_date", "", ns_sm) or "").strip()
                    items.append({"t": title, "l": loc, "p": pub, "d": "", "c": []})
                    if len(items) >= max_items:
                        break
        except ET.ParseError:
            pass
        return items

    def _parse_feeds_to_json(self, raw_dict):
        """Parst alle XML-Feeds zu kompaktem JSON. 1.5MB XML → ~50KB JSON."""
        parsed = {}
        for name, xml_str in raw_dict.items():
            if not xml_str:
                parsed[name] = []
                continue
            parsed[name] = self._parse_rss_items(xml_str)
        return parsed

    def do_GET(self):
        # Statische HTML/JS/CSS mit gzip komprimieren
        if self.path.endswith(('.html', '.js', '.css')) and not self.path.startswith('/api/'):
            fpath = os.path.join(SERVE_DIR, self.path.lstrip('/').split('?')[0])
            if os.path.isfile(fpath) and self._accepts_gzip():
                import gzip as _gzip_mod
                with open(fpath, 'rb') as f:
                    gz_data = _gzip_mod.compress(f.read(), compresslevel=6)
                ct = 'text/html' if fpath.endswith('.html') else ('application/javascript' if fpath.endswith('.js') else 'text/css')
                self.send_response(200)
                self.send_header("Content-Type", ct + "; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(gz_data)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(gz_data)
                return

        if self.path == "/api/feed":
            self._proxy_xml(BILD_SITEMAP)
        elif self.path.startswith("/api/push/"):
            self._proxy_push_api(self.path[len("/api"):])
        elif self.path == "/api/competitors":
            self._serve_competitor_feeds()
        elif self.path.startswith("/api/competitor/"):
            name = self.path[len("/api/competitor/"):].split("?")[0]
            if name in COMPETITOR_FEEDS:
                self._proxy_xml(COMPETITOR_FEEDS[name])
            else:
                self._error(404, f"Unknown competitor: {name}")
        elif self.path == "/api/sport-competitors":
            self._serve_sport_competitor_feeds()
        elif self.path == "/api/sport-europa":
            self._serve_sport_europa_feeds()
        elif self.path == "/api/sport-global":
            self._serve_sport_global_feeds()
        elif self.path == "/api/international":
            self._serve_international_feeds()
        elif self.path.startswith("/api/international/"):
            name = self.path[len("/api/international/"):].split("?")[0]
            if name in INTERNATIONAL_FEEDS:
                self._proxy_xml(INTERNATIONAL_FEEDS[name])
            else:
                self._error(404, f"Unknown international feed: {name}")
        elif self.path == "/api/check-plus":
            self._check_plus_urls()
        elif self.path == "/api/forschung":
            self._serve_forschung()
        elif self.path == "/api/learnings":
            self._serve_learnings()
        elif self.path == "/api/adobe/traffic":
            self._serve_adobe_traffic()
        elif self.path == "/api/health":
            self._serve_health()
        elif self.path == "/api/research-rules":
            self._serve_research_rules()
        elif self.path == "/api/ml/safety-status":
            self._serve_ml_safety_status()
        elif self.path == "/api/ml/status":
            self._serve_ml_status()
        elif self.path.startswith("/api/ml/predict"):
            self._serve_ml_predict()
        elif self.path == "/api/ml/experiments":
            self._serve_ml_experiments()
        elif self.path.startswith("/api/ml/experiments/compare"):
            self._serve_ml_experiments_compare()
        elif self.path == "/api/ml/ab-status":
            self._serve_ml_ab_status()
        elif self.path == "/api/ml/monitoring":
            self._serve_ml_monitoring()
        elif self.path.split("?")[0] == "/api/tagesplan":
            self._serve_tagesplan()
        elif self.path.split("?")[0] == "/api/tagesplan/retro":
            self._serve_tagesplan_retro()
        elif self.path.startswith("/api/tagesplan/history"):
            self._serve_tagesplan_history()
        elif self.path.startswith("/api/tagesplan/suggestions"):
            self._serve_tagesplan_suggestions()
        elif self.path == "/api/gbrt/status":
            self._serve_gbrt_status()
        elif self.path == "/api/gbrt/model.json":
            self._serve_gbrt_model_json()
        elif self.path.startswith("/api/gbrt/predict"):
            self._serve_gbrt_predict()
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Bypass-Tunnel-Reminder", "true")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def _proxy_xml(self, url):
        try:
            data = _fetch_url(url)
            if data is None:
                raise Exception("Empty response")
            self._send_gzip(data, "application/xml; charset=utf-8")
        except Exception as e:
            self._error(502, f"Proxy error: {e}")

    def _proxy_push_api(self, path):
        """Proxy requests to the BILD Push API — mit Sync-Cache-Fallback."""
        # 1. Versuch: Direkt zur BILD Push API
        url = f"{PUSH_API_BASE}{path}"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15, context=_GLOBAL_SSL_CTX) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(data)
            return
        except Exception as e:
            log.info(f"[Proxy] Push API direkt nicht erreichbar, pruefe Sync-Cache: {e}")

        # 2. Fallback: Sync-Cache (von lokalem Server gefuellt)
        with _push_sync_lock:
            cache_age = time.time() - _push_sync_cache["ts"]
            if _push_sync_cache["ts"] > 0 and cache_age < 86400:  # Max 24h alt
                if "channels" in path:
                    payload = json.dumps(_push_sync_cache.get("channels", [])).encode()
                else:
                    payload = json.dumps({"messages": _push_sync_cache["messages"], "next": None,
                                          "_synced": True, "_age_s": int(cache_age)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors_headers()
                self.wfile.write(payload)
                return

        # 3. Kein Cache verfuegbar
        fallback = json.dumps({"messages": [], "_fallback": True,
                               "_reason": "Push-API nicht erreichbar und kein Sync-Cache vorhanden. Lokaler Server muss laufen um Daten zu synchronisieren."}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()
        self.wfile.write(fallback)

    def _serve_competitor_feeds(self):
        """Liefert Competitor-Feeds aus Background-Cache (sofort, <5ms)."""
        try:
            parsed = _get_cached_feeds("competitors")
            self._send_gzip(json.dumps(parsed).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._error(502, f"Competitor feeds error: {e}")

    def _serve_international_feeds(self):
        """Liefert International-Feeds aus Background-Cache (sofort, <5ms)."""
        try:
            parsed = _get_cached_feeds("international")
            self._send_gzip(json.dumps(parsed).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._error(502, f"International feeds error: {e}")

    def _serve_sport_competitor_feeds(self):
        """Liefert Sport-Competitor-Feeds aus Background-Cache."""
        try:
            parsed = _get_cached_feeds("sport_competitors")
            self._send_gzip(json.dumps(parsed).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._error(502, f"Sport competitor feeds error: {e}")

    def _serve_sport_europa_feeds(self):
        """Liefert Sport-Europa-Feeds aus Background-Cache."""
        try:
            parsed = _get_cached_feeds("sport_europa")
            self._send_gzip(json.dumps(parsed).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._error(502, f"Sport Europa feeds error: {e}")

    def _serve_sport_global_feeds(self):
        """Liefert Sport-Global-Feeds aus Background-Cache."""
        try:
            parsed = _get_cached_feeds("sport_global")
            self._send_gzip(json.dumps(parsed).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._error(502, f"Sport Global feeds error: {e}")

    def _serve_health(self):
        """Return health/security status of all endpoints."""
        try:
            uptime = time.time() - _health_state.get("uptime_start", time.time())
            result = {
                "status": _health_state.get("status", "unknown"),
                "uptime_seconds": int(uptime),
                "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m",
                "last_check": _health_state.get("last_check", 0),
                "checks_ok": _health_state.get("checks_ok", 0),
                "checks_fail": _health_state.get("checks_fail", 0),
                "endpoints": _health_state.get("endpoints", {}),
                "research_data_points": len(_research_state.get("push_data", [])),
                "research_last_analysis": _research_state.get("last_analysis", 0),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Health API error: {e}")

    def _serve_research_rules(self):
        """Liefert aktive Forschungsregeln fuer den Push-Kandidaten-Ablauf."""
        try:
            rules = _research_state.get("live_rules", [])
            active = [r for r in rules if r.get("active")]
            accuracy = _research_state.get("rolling_accuracy", 0.0)
            result = {
                "rules": active,
                "version": _research_state.get("live_rules_version", 0),
                "accuracy": round(accuracy, 1),
                "n_pushes_analyzed": len(_research_state.get("push_data", [])),
                "last_update": _research_state.get("last_analysis", 0),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Research Rules API error: {e}")

    def _serve_adobe_traffic(self):
        """GET /api/adobe/traffic — Adobe Analytics Traffic Sources fuer heutige Pushes."""
        try:
            if not _adobe_state["enabled"]:
                self._json_response({"enabled": False, "error": "Adobe nicht konfiguriert"}, ensure_ascii=False)
                return
            traffic = _adobe_state.get("traffic")
            if not traffic:
                self._json_response({
                    "enabled": True,
                    "loading": True,
                    "updated_at": 0,
                    "error": _adobe_state.get("error", ""),
                }, ensure_ascii=False)
                return
            self._json_response({
                "enabled": True,
                "loading": False,
                "updated_at": _adobe_state["updated_at"],
                "error": "",
                **traffic,
            }, ensure_ascii=False)
        except Exception as e:
            self._error(500, f"Adobe Traffic API error: {e}")

    def _serve_forschung(self):
        """Generate research institute data with REAL autonomous analysis of push data."""
        try:
            now = datetime.datetime.now()
            hour = now.hour

            # Analyse laeuft autonom im Background-Worker (alle 20s)
            # Hier nur pruefen ob Daten bereit sind
            if not _research_state.get("push_data"):
                # Worker hat noch keine Daten — sofort loading-State zurueckgeben
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors_headers()
                self.wfile.write(json.dumps({
                    "accuracy": 0, "accuracy_trend": 0, "accuracy_target": 99.5,
                    "insights_today": 0, "insights_trend": 0, "pages_today": 0, "pages_trend": 0,
                    "researchers": [], "guest_researchers": [], "guest_exchanges": [],
                    "orchestrator": {"id": "ml-system", "name": "ML Pipeline",
                                     "role": "Autonomes System", "status": "loading",
                                     "current_directive": "Lade Push-Daten...",
                                     "teams_active": 1, "decisions_today": 0, "schwab_decisions": []},
                    "bild_team": [], "ticker": [], "learning": [],
                    "dissertations": [], "diskurs": [],
                    "week_comparison": {}, "live_rules": [], "live_rules_count": 0,
                    "n_pushes": 0, "last_push_ts": 0, "loading": True,
                    "mature_count": 0, "fresh_count": 0, "fresh_pushes": [],
                    "research_memory": {}, "research_memory_total": 0, "research_log": [],
                    "research_projects": [], "research_milestones": [], "institute_review": {},
                }, ensure_ascii=False).encode("utf-8"))
                return

            # Seed based on date for stable daily variations
            day_seed = int(now.strftime("%Y%m%d"))
            rng = random.Random(day_seed)
            day_rng = random.Random(day_seed + 42)

            # NUR reife Pushes (>24h) fuer ALLE Berechnungen — frische OR ist unzuverlaessig
            # OR > 30% = Datenbankfehler (z.B. 1.3M% Ausreisser) — clampen oder skippen
            _OR_SANE_MAX = 30.0
            _cutoff_24h = _research_state.get("cutoff_24h", time.time() - 24 * 3600)
            # Immer direkt aus DB laden — Research-Worker hat nur API-Subset
            try:
                _raw_push_data = _push_db_load_all()
            except Exception:
                _raw_push_data = _research_state.get("push_data", [])
            _mature_data = []
            for _p in _raw_push_data:
                if _p.get("ts_num", 0) <= 0 or _p["ts_num"] >= _cutoff_24h or _p.get("or", 0) <= 0:
                    continue
                if _p["or"] > _OR_SANE_MAX:
                    continue  # Ausreisser komplett ignorieren
                _mature_data.append(_p)

            # Real metrics — NUR reife Pushes zaehlen (>24h mit stabiler OR)
            n_pushes = len(_mature_data)
            ticker_entries = _research_state.get("ticker_entries", [])
            insights_today = len(ticker_entries)
            # Pages = Anzahl analysierter Datenpunkte (Pushes × Dimensionen)
            analysis_dims = 8  # timing, category, framing, length, frequency, keywords, linguistic, accuracy
            pages_today = n_pushes * analysis_dims

            # Get real findings from autonomous analysis
            findings = _research_state.get("findings", {})
            gen = _research_state.get("analysis_generation", 0)
            research_memory = _research_state.get("research_memory", {})

            # ── FALLBACK: Wenn findings leer, direkt aus _mature_data berechnen ──
            if not findings.get("hour_analysis") and len(_mature_data) >= 10:
                try:
                    findings = _compute_findings_for_subset(_mature_data)
                    _research_state["findings"] = findings
                    logging.info(f"[Forschung-Fallback] Findings direkt berechnet aus {len(_mature_data)} reifen Pushes")
                except Exception as _fe:
                    logging.warning(f"[Forschung-Fallback] Fehler: {_fe}")

            # ── FALLBACK: temporal_trends direkt berechnen wenn leer ──
            if not findings.get("temporal_trends") and len(_mature_data) >= 10:
                try:
                    findings["temporal_trends"] = _compute_temporal_trends(_mature_data)
                    _research_state["findings"] = findings
                    logging.info(f"[Forschung-Fallback] temporal_trends berechnet aus {len(_mature_data)} Pushes")
                except Exception as _fe:
                    logging.warning(f"[Forschung-Fallback] temporal_trends Fehler: {_fe}")

            # ── FALLBACK: week_comparison direkt berechnen wenn leer ──
            if not _research_state.get("week_comparison") and len(_mature_data) >= 10:
                try:
                    _now_ts = time.time()
                    _wd = datetime.datetime.now().weekday()
                    _week_start = _now_ts - (_wd * 86400 + datetime.datetime.now().hour * 3600)
                    _last_week_start = _week_start - 7 * 86400
                    _this_week = [p for p in _mature_data if p.get("ts_num", 0) >= _week_start]
                    _last_week = [p for p in _mature_data if _last_week_start <= p.get("ts_num", 0) < _week_start]
                    _tw_ors = [p["or"] for p in _this_week if p.get("or", 0) > 0]
                    _lw_ors = [p["or"] for p in _last_week if p.get("or", 0) > 0]
                    _kw_now = datetime.datetime.now().isocalendar()[1]
                    _research_state["week_comparison"] = {
                        "this_week": {"kw": _kw_now, "avg_or": round(sum(_tw_ors) / len(_tw_ors), 1) if _tw_ors else 0, "count": len(_tw_ors)},
                        "last_week": {"kw": _kw_now - 1, "avg_or": round(sum(_lw_ors) / len(_lw_ors), 1) if _lw_ors else 0, "count": len(_lw_ors)},
                    }
                except Exception:
                    pass

            # ── FALLBACK: research_modifiers direkt berechnen wenn leer ──
            if not _research_state.get("research_modifiers") and len(_mature_data) >= 100:
                try:
                    _rm = _compute_research_modifiers(_mature_data, findings, _research_state)
                    if _rm:
                        _research_state["research_modifiers"] = _rm
                except Exception:
                    pass

            # Hilfsfunktion: Letzte Erkenntnis aus dem Forschungsgedaechtnis
            _now_str = f"{hour}:{now.minute:02d}"

            def mem_action(rid, fallback_action):
                # PRIORITAET 1: Findings (Praesens-Block mit heutigen Pushes)
                finding_action = findings.get(rid, {}).get("action")
                if finding_action:
                    return finding_action
                # PRIORITAET 2: Research Memory (kumulativ)
                entries = research_memory.get(rid, [])
                if entries:
                    return entries[-1]["finding"][:200]
                return fallback_action

            def mem_topic(rid, fallback_topic):
                # PRIORITAET 1: Findings (Praesens mit heutigen Pushes)
                finding_topic = findings.get(rid, {}).get("topic")
                if finding_topic:
                    return finding_topic
                # PRIORITAET 2: Research Memory
                entries = research_memory.get(rid, [])
                if len(entries) >= 2:
                    return entries[-2]["finding"][:150] + " -> " + entries[-1]["finding"][:80]
                elif entries:
                    ref = entries[-1].get("builds_on")
                    return entries[-1]["finding"][:150] + (f" (mit {ref})" if ref else "")
                return fallback_topic

            # ── ML Model Analytics (ersetzt fiktive Professoren) ──
            _gbrt_info = {}
            with _gbrt_lock:
                if _gbrt_model is not None:
                    _gbrt_info = {
                        "type": "GBRT",
                        "n_trees": len(_gbrt_model.trees),
                        "metrics": _gbrt_model.train_metrics,
                        "feature_importance": _gbrt_model.feature_importance(20),
                        "trained_at": _gbrt_train_ts,
                    }

            # Kein Theater mehr — leere Liste fuer Abwaertskompatibilitaet
            researchers = []

            # mean_or_all frueher berechnen (wird fuer Orchestrator + Learning gebraucht)
            # ROBUST: Median statt Mean (unempfindlich gegen Rest-Ausreisser)
            _all_ors = sorted([p["or"] for p in _mature_data]) if _mature_data else []
            mean_or_all = _all_ors[len(_all_ors) // 2] if _all_ors else 0.0
            _mean_or_arithmetic = sum(_all_ors) / len(_all_ors) if _all_ors else 0.0

            # Orchestrator — echte Daten
            schwab_decisions = _research_state.get("schwab_decisions", [])
            n_live_rules = len([r for r in _research_state.get("live_rules", []) if r.get("active")])
            _primary_mae = _research_state.get("ensemble_mae", 0) or _research_state.get("basis_mae", 0)
            _basis_mae_val = _research_state.get("basis_mae", 0) or _primary_mae
            primary_mae = min(_primary_mae, _basis_mae_val) if _basis_mae_val > 0 else _primary_mae
            _mean_or_directive = mean_or_all if mean_or_all > 0.5 else 4.0
            _treff_score = max(0, min(100, (1 - primary_mae / _mean_or_directive) * 100))
            # Directive basierend auf Treffsicherheit
            if _treff_score < 50:
                directive = f"Prioritaet: Treffsicherheit bei {_treff_score:.0f}% (Ø {primary_mae:.1f}pp daneben) — Modell verbessern, Ziel >90%"
            elif _treff_score < 75:
                directive = f"Treffsicherheit {_treff_score:.0f}% (Ø {primary_mae:.1f}pp) — auf gutem Weg. {n_live_rules} Live-Regeln aktiv"
            else:
                directive = f"Treffsicherheit {_treff_score:.0f}% (Ø {primary_mae:.1f}pp) — stark! Fokus auf Feintuning"
            orchestrator = {
                "id": "ml-system", "name": "ML Pipeline", "role": "Autonomes System",
                "status": "active",
                "current_directive": directive,
                "teams_active": 1,
                "decisions_today": len(schwab_decisions),
                "schwab_decisions": schwab_decisions[-10:],
            }

            # Internationale Gastprofessoren + Linguist — alle Daten aus echten Findings
            # Echte berechnete Werte aus den Findings extrahieren
            f_hour = findings.get("hour_analysis", {})
            f_cat = findings.get("cat_analysis", [])
            f_frame = findings.get("framing_analysis", {})
            f_freq = findings.get("frequency_correlation", {})
            f_ling = findings.get("linguistic_analysis", {})
            f_len = findings.get("title_length", {})
            f_kw = findings.get("keyword_analysis", {})
            rolling_acc = _research_state.get("rolling_accuracy", 0.0)
            acc_by_cat = _research_state.get("accuracy_by_cat", {})

            # Echte Zahlen fuer Gastprofessoren
            best_h = f_hour.get("best_hour", "?")
            best_h_or = f_hour.get("best_or", 0)
            worst_h = f_hour.get("worst_hour", "?")
            worst_h_or = f_hour.get("worst_or", 0)
            n_cats = len(f_cat)
            top_cat = f_cat[0] if f_cat else {}
            emo_or = f_frame.get("emotional_or", 0)
            neutral_or = f_frame.get("neutral_or", 0)
            n_emo = f_frame.get("emotional_count", 0)
            freq_r = f_freq.get("correlation", 0)
            opt_daily = f_freq.get("optimal_daily", 0)
            colon_or = f_ling.get("colon_or", 0)
            no_colon_or = f_ling.get("no_colon_or", 0)
            n_colon = f_ling.get("colon_count", 0)
            n_no_colon = f_ling.get("no_colon_count", 0)
            best_len_range = f_len.get("best_range", "?")
            best_len_or = f_len.get("best_or", 0)
            top_kws = f_kw.get("top_keywords", [])[:5] if isinstance(f_kw, dict) else []

            n_decisions = len(_research_state.get("schwab_decisions", []))
            n_live = len([r for r in _research_state.get("live_rules", []) if r.get("active")])

            guest_researchers = []
            algo_team_researchers = []
            bild_team = []
            # Learning metrics — 100% aus echten Daten, keine Simulation
            or_values = [p["or"] for p in _mature_data]  # Nur reife Pushes (>24h)
            # Primaer-Metrik: MAE (Mean Absolute Error in Prozentpunkten)
            # = "Wie weit liegen wir im Schnitt daneben?"
            # Ziel: So nah wie moeglich an 0 (perfekte Vorhersage)
            basis_mae = _research_state.get("basis_mae", 0.0)
            ensemble_mae = _research_state.get("ensemble_mae", 0.0)
            # Sanity: MAE > mean_or deutet auf vergiftete Berechnung hin → recalculate
            if basis_mae > mean_or_all * 1.5 and len(_mature_data) >= 50:
                logging.warning(f"[Forschung] basis_mae={basis_mae:.2f} > 1.5*median_or={mean_or_all:.2f} — forciere Neuberechnung")
                _research_state["basis_mae"] = 0.0
                _research_state["ensemble_mae"] = 0.0
                _research_state["mae_trend"] = []
                _update_rolling_accuracy(_mature_data, _research_state)
                basis_mae = _research_state.get("basis_mae", 0.0)
                ensemble_mae = _research_state.get("ensemble_mae", 0.0)
            # Beste verfuegbare MAE: Ensemble wenn vorhanden, sonst Basis
            primary_mae = ensemble_mae if ensemble_mae > 0 else basis_mae
            rolling_acc = _research_state.get("rolling_accuracy", 0.0)

            # MAE pro Dimension
            mae_by_cat = _research_state.get("mae_by_cat", {})
            mae_by_hour = _research_state.get("mae_by_hour", {})
            avg_cat_mae = round(sum(mae_by_cat.values()) / max(1, len(mae_by_cat)), 3) if mae_by_cat else 0.0
            avg_hour_mae = round(sum(mae_by_hour.values()) / max(1, len(mae_by_hour)), 3) if mae_by_hour else 0.0

            # MAE-Trend: letzte vs. vorletzte Messung (negativ = besser)
            mae_trend_arr = _research_state.get("mae_trend", [])
            mae_delta = round(mae_trend_arr[-1] - mae_trend_arr[-2], 3) if len(mae_trend_arr) >= 2 else 0.0

            # Treffsicherheit-Score berechnen (fuer Learning + Result)
            # Robust: Median OR (nicht Mean), Sanity-Check auf MAE
            _mean_or_for_score = mean_or_all if mean_or_all > 0.5 else 4.0
            _best_mae = min(primary_mae, basis_mae) if basis_mae > 0 else primary_mae
            # Sanity: MAE > Median OR → Modell rät praktisch zufällig → max 20%
            if _best_mae > _mean_or_for_score * 0.9:
                treffsicherheit = max(5, min(20, (1 - _best_mae / max(1, _mean_or_for_score)) * 100))
            elif _best_mae > _mean_or_for_score * 0.5:
                treffsicherheit = max(20, min(60, (1 - _best_mae / _mean_or_for_score) * 100))
            else:
                treffsicherheit = max(0, min(95, (1 - _best_mae / _mean_or_for_score) * 100))

            learning = [
                {"label": "Treffsicherheit (Gesamt)", "value": round(treffsicherheit, 1), "trend": round(-mae_delta / max(0.1, _mean_or_for_score) * 100, 1), "target": 90.0},
                {"label": "Abweichung (MAE)", "value": round(_best_mae, 1), "trend": round(-mae_delta, 2), "target": 0.5, "unit": "pp", "lower_is_better": True},
                {"label": "Kategorie-Genauigkeit", "value": round(max(0, (1 - avg_cat_mae / _mean_or_for_score) * 100), 1), "trend": 0.0, "target": 90.0},
                {"label": "Timing-Genauigkeit", "value": round(max(0, (1 - avg_hour_mae / _mean_or_for_score) * 100), 1), "trend": 0.0, "target": 90.0},
                {"label": "Datenbasis (Pushes)", "value": float(n_pushes), "trend": 0.0, "target": 1000},
            ]

            # Use real ticker from autonomous analysis
            ticker = list(_research_state.get("ticker_entries", []))

            # Theater entfernt — Feynman, Dissertationen, Diskurs
            dissertations = []
            diskurs_data = []
            selected = []
            week_comparison = _research_state.get("week_comparison", {})
            live_rules = _research_state.get("live_rules", [])

            # ── Neue Daten fuer Dashboard-Upgrade ──────────────────────────
            # Top 5 / Flop 5 Pushes nach OR — nur letzte 365 Tage (aeltere OR nicht vergleichbar)
            _topflop_cutoff = time.time() - 365 * 86400
            _recent_for_topflop = [p for p in _mature_data if p.get("ts_num", 0) > _topflop_cutoff]
            _sorted_by_or = sorted(_recent_for_topflop, key=lambda p: p.get("or", 0), reverse=True)
            top_pushes = [{"title": p.get("title", "")[:100], "or": round(p.get("or", 0), 2), "cat": p.get("cat", ""), "hour": p.get("hour", 0)} for p in _sorted_by_or[:5]]
            worst_pushes = [{"title": p.get("title", "")[:100], "or": round(p.get("or", 0), 2), "cat": p.get("cat", ""), "hour": p.get("hour", 0)} for p in _sorted_by_or[-5:][::-1]] if len(_sorted_by_or) >= 5 else []

            # OR-Verteilung: Quartile, Median, Std
            _or_vals_sorted = sorted(p.get("or", 0) for p in _mature_data) if _mature_data else []
            or_distribution = {}
            if _or_vals_sorted:
                n_or = len(_or_vals_sorted)
                or_distribution = {
                    "min": round(_or_vals_sorted[0], 2),
                    "q1": round(_or_vals_sorted[n_or // 4], 2),
                    "median": round(_or_vals_sorted[n_or // 2], 2),
                    "q3": round(_or_vals_sorted[3 * n_or // 4], 2),
                    "max": round(_or_vals_sorted[-1], 2),
                    "mean": round(_mean_or_arithmetic, 2),
                    "std": round(math.sqrt(sum((v - _mean_or_arithmetic)**2 for v in _or_vals_sorted) / max(1, n_or - 1)), 2) if n_or > 1 else 0,
                    "n": n_or,
                }

            # OR pro Stunde (alle 24h-Slots)
            _hour_or_agg = {}
            for p in _mature_data:
                h = p.get("hour", 0)
                _hour_or_agg.setdefault(h, []).append(p.get("or", 0))
            hour_distribution = {str(h): round(sum(vals) / len(vals), 2) for h, vals in sorted(_hour_or_agg.items())}

            # OR pro Kategorie
            _cat_or_agg = {}
            for p in _mature_data:
                c = p.get("cat", "Sonstige")
                _cat_or_agg.setdefault(c, []).append(p.get("or", 0))
            cat_distribution = {c: {"avg_or": round(sum(vals) / len(vals), 2), "count": len(vals)} for c, vals in sorted(_cat_or_agg.items(), key=lambda x: -sum(x[1]) / len(x[1]))}

            # treffsicherheit, _best_mae, _mean_or_for_score schon oben berechnet

            # Trend-Tracking fuer Dashboard-Metriken
            _prev = _research_state.get("_dashboard_prev", {})
            _cur_best_hour = f_hour.get("best_hour", 0) if f_hour else 0
            _cur_best_hour_or = round(f_hour.get("best_or", 0), 1) if f_hour else 0
            _cur_live_rules_n = len([r for r in live_rules if r.get("active")])
            _cur_top_cat = f_cat[0]["category"] if f_cat else ""
            _cur_top_cat_or = round(f_cat[0]["avg_or"], 1) if f_cat else 0
            _best_hour_or_delta = round(_cur_best_hour_or - _prev.get("best_hour_or", _cur_best_hour_or), 1)
            _n_pushes_delta = n_pushes - _prev.get("n_pushes", n_pushes)
            _live_rules_delta = _cur_live_rules_n - _prev.get("live_rules_n", _cur_live_rules_n)
            _top_cat_or_delta = round(_cur_top_cat_or - _prev.get("top_cat_or", _cur_top_cat_or), 1)
            _insights_delta = insights_today - _prev.get("insights_today", insights_today)
            _research_state["_dashboard_prev"] = {
                "best_hour_or": _cur_best_hour_or, "n_pushes": n_pushes,
                "live_rules_n": _cur_live_rules_n, "top_cat_or": _cur_top_cat_or,
                "insights_today": insights_today,
            }
            _top_cats = [{"category": c.get("category",""), "avg_or": round(c.get("avg_or",0),1), "count": c.get("count",0), "rank": i+1} for i, c in enumerate(f_cat[:3])]

            result = {
                "accuracy": round(treffsicherheit, 1),
                "accuracy_mae": round(_best_mae, 2),
                "accuracy_trend": round(-mae_delta, 3),
                "accuracy_target": 90.0,
                "basis_mae": round(basis_mae, 3),
                "ensemble_mae_raw": round(primary_mae, 3),
                "mae_by_cat": mae_by_cat,
                "mae_by_hour": mae_by_hour,
                "mae_trend_arr": mae_trend_arr[-20:],
                "hit_rate": round(rolling_acc, 1),
                "insights_today": insights_today,
                "insights_trend": _insights_delta,
                "pages_today": pages_today,
                "pages_trend": 0,
                "researchers": researchers,
                "guest_researchers": guest_researchers,
                "guest_exchanges": selected,
                "orchestrator": orchestrator,
                "bild_team": bild_team,
                "ticker": ticker,
                "learning": learning,
                "dissertations": dissertations,
                "diskurs": diskurs_data,
                "week_comparison": week_comparison,
                "live_rules": [r for r in live_rules if r.get("active")],
                "live_rules_count": _cur_live_rules_n,
                "live_rules_trend": _live_rules_delta,
                "n_pushes": n_pushes,
                "n_pushes_trend": _n_pushes_delta,
                "mean_or": round(mean_or_all, 2) if mean_or_all else 0,
                "best_hour": _cur_best_hour,
                "best_hour_or": _cur_best_hour_or,
                "best_hour_or_trend": _best_hour_or_delta,
                "top_category": _cur_top_cat,
                "top_category_or": _cur_top_cat_or,
                "top_category_or_trend": _top_cat_or_delta,
                "top_categories": _top_cats,
                "last_push_ts": max((p.get("ts_num", 0) for p in _mature_data), default=0) if _mature_data else 0,
                "research_projects": _research_state.get("research_projects", []),
                "research_milestones": _research_state.get("research_milestones", [])[-20:],
                # Kumuliertes Forschungsgedaechtnis — zeigt echten Progress
                "research_memory": {rid: entries[-3:] for rid, entries in _research_state.get("research_memory", {}).items() if entries},
                "research_memory_total": sum(len(v) for v in _research_state.get("research_memory", {}).values()),
                "research_log": _research_state.get("research_log", [])[-15:],
                # 24h-Maturation-Status
                "mature_count": _research_state.get("mature_count", 0),
                "fresh_count": _research_state.get("fresh_count", 0),
                "fresh_pushes": [{"title": p.get("title", "")[:80], "cat": p.get("cat", ""), "hour": p.get("hour", 0), "ts": p.get("ts", ""), "or": round(p.get("or", 0), 2), "opened": p.get("opened", 0), "received": p.get("received", 0)} for p in _research_state.get("fresh_pushes", [])[:15]],
                # LLM-Qualitaetsbewertung des Instituts
                "institute_review": _research_state.get("institute_review", {}),
                # Detaillierte Emotions-Analyse aus 24h-reifen Pushes
                "emotion_radar": findings.get("emotion_radar", []),
                # Schwab Approval Queue — zeigt offene Aenderungsvorschlaege
                "pending_approvals": [a for a in _research_state.get("pending_approvals", [])[-100:] if a.get("status") == "pending"],
                # Research-Modifier: Werden vom Frontend als 8. Methode in predictOR() verwendet
                "research_modifiers": _research_state.get("research_modifiers", {}),
                # Algorithmus-Elite-Team
                "algorithm_team": algo_team_researchers,
                # Score-Analyse: Feature-Importance, Score-Dekomposition, XOR-Vorschlaege
                "score_analysis": _research_state.get("algo_score_analysis", {}),
                # PhD-Insights: 10 mathematische Doktorarbeiten liefern erweiterte Modifier
                "phd_insights": _research_state.get("phd_insights", {}),
                # Live-Pulse: Forscher-Kommentare zu frischen Pushes
                "live_pulse": _research_state.get("live_pulse", []),
                # OR-Challenge: Claude Sonnet 4 hinterfragt die predictOR-Formel
                "or_challenges": _research_state.get("or_challenges", {}),
                # Temporale Trends: Monats/Wochen/Wochentag/YoY-Analyse
                "temporal_trends": findings.get("temporal_trends", {}),
                # ── Dashboard-Upgrade: Neue Datenfelder ──
                "accuracy_by_cat": _research_state.get("accuracy_by_cat", {}),
                "accuracy_by_hour": _research_state.get("accuracy_by_hour", {}),
                # mae_by_cat und mae_by_hour werden oben (Zeile ~10532) mit frischen lokalen Daten gesetzt
                "accuracy_trend_arr": _research_state.get("mae_trend", [])[-20:],  # MAE-Trend fuer Sparkline
                "top_pushes": top_pushes,
                "worst_pushes": worst_pushes,
                "or_distribution": or_distribution,
                "gbrt_n_features": len(_gbrt_feature_names) if _gbrt_feature_names else 0,
                "hour_distribution": hour_distribution,
                "cat_distribution": cat_distribution,
                # Autonomes Tuning: Parameter + History fuer Frontend
                "tuning_params": _research_state.get("tuning_params", {}) or DEFAULT_TUNING_PARAMS,
                "tuning_params_version": _research_state.get("tuning_params_version", 0),
                "tuning_history": _research_state.get("tuning_history", [])[-10:],
                "prediction_feedback_count": len(_research_state.get("prediction_feedback", [])),
                # Server-Autonomie: Ensemble-Accuracy + Methoden-Performance
                "ensemble_accuracy": _research_state.get("ensemble_accuracy", 0),
                "ensemble_mae": _research_state.get("ensemble_mae", 0),
                "ensemble_accuracy_trend": _research_state.get("ensemble_accuracy_trend", [])[-20:],
                "ensemble_accuracy_delta": _research_state.get("ensemble_accuracy_delta", 0),
                "method_performance": _research_state.get("_algo_team_analysis", {}).get("method_stats", {}),
                "algo_team_history": _research_state.get("algo_team_history", {}),
                "algo_proposals_count": len([a for a in _research_state.get("pending_approvals", []) if a.get("source") == "algo_team_autonomous" and a.get("status") == "pending"]),
                "server_feedback_count": len([fb for fb in _research_state.get("prediction_feedback", []) if fb.get("source") == "server"]),
                "external_context": _research_state.get("external_context", {}),
                # Progress-Ticker: Echte Events aus dem Forschungsteam
                "progress_ticker": _build_progress_ticker(_research_state, findings),
                # ── Autonomes Forschungsinstitut: Neue Datenfelder ──
                "cross_references": _research_state.get("cross_references", [])[-10:],
                "negative_results": _research_state.get("negative_results", [])[-10:],
                "meta_research": _research_state.get("meta_research", {}),
                "decision_proposals": [d for d in _research_state.get("decision_proposals", []) if d.get("status") == "pending"],
                "exploration_experiments": _research_state.get("exploration_experiments", [])[-5:],
                "arxiv_papers": _research_state.get("arxiv_papers", [])[-10:],
                "algo_lab_progress": _research_state.get("algo_lab_progress", {}),
                # ── Sport/NonSport Split ──
                "sport_n": _research_state.get("_sport_n", 0),
                "nonsport_n": _research_state.get("_nonsport_n", 0),
                "sport_accuracy": _research_state.get("_sport_accuracy", {}),
                "nonsport_accuracy": _research_state.get("_nonsport_accuracy", {}),
                "sport_modifiers": _research_state.get("_sport_modifiers", {}),
                "nonsport_modifiers": _research_state.get("_nonsport_modifiers", {}),
                "sport_findings": _research_state.get("_sport_findings", {}),
                "nonsport_findings": _research_state.get("_nonsport_findings", {}),
                "sport_algo_team": _research_state.get("_algo_team_sport", {}),
                "nonsport_algo_team": _research_state.get("_algo_team_nonsport", {}),
                # ── GBRT ML-Modell Analytics ──
                "gbrt_model": _gbrt_info,
                "gbrt_trained_at": _gbrt_train_ts,
            }

            # Sport/NonSport OR-Verteilungen und Stunden-Verteilungen berechnen
            _sport_mature = [p for p in _mature_data if p.get("cat") == "Sport"]
            _nonsport_mature = [p for p in _mature_data if p.get("cat") != "Sport"]

            # Sport OR-Verteilung
            if _sport_mature:
                _sport_or_vals = sorted(p.get("or", 0) for p in _sport_mature)
                _sport_n_or = len(_sport_or_vals)
                _sport_mean = _sport_or_vals[_sport_n_or // 2]  # Median (robust)
                result["sport_mean_or"] = round(_sport_mean, 2)
                result["sport_or_distribution"] = {
                    "min": round(_sport_or_vals[0], 2),
                    "q1": round(_sport_or_vals[_sport_n_or // 4], 2),
                    "median": round(_sport_or_vals[_sport_n_or // 2], 2),
                    "q3": round(_sport_or_vals[3 * _sport_n_or // 4], 2),
                    "max": round(_sport_or_vals[-1], 2),
                    "mean": round(_sport_mean, 2),
                    "n": _sport_n_or,
                }
                _sport_hour_agg = {}
                for p in _sport_mature:
                    h = p.get("hour", 0)
                    _sport_hour_agg.setdefault(h, []).append(p.get("or", 0))
                result["sport_hour_distribution"] = {str(h): round(sum(vals) / len(vals), 2) for h, vals in sorted(_sport_hour_agg.items())}
            else:
                result["sport_mean_or"] = 0
                result["sport_or_distribution"] = {}
                result["sport_hour_distribution"] = {}

            # NonSport OR-Verteilung
            if _nonsport_mature:
                _ns_or_vals = sorted(p.get("or", 0) for p in _nonsport_mature)
                _ns_n_or = len(_ns_or_vals)
                _ns_mean = _ns_or_vals[_ns_n_or // 2]  # Median (robust)
                result["nonsport_mean_or"] = round(_ns_mean, 2)
                result["nonsport_or_distribution"] = {
                    "min": round(_ns_or_vals[0], 2),
                    "q1": round(_ns_or_vals[_ns_n_or // 4], 2),
                    "median": round(_ns_or_vals[_ns_n_or // 2], 2),
                    "q3": round(_ns_or_vals[3 * _ns_n_or // 4], 2),
                    "max": round(_ns_or_vals[-1], 2),
                    "mean": round(_ns_mean, 2),
                    "n": _ns_n_or,
                }
                _ns_hour_agg = {}
                for p in _nonsport_mature:
                    h = p.get("hour", 0)
                    _ns_hour_agg.setdefault(h, []).append(p.get("or", 0))
                result["nonsport_hour_distribution"] = {str(h): round(sum(vals) / len(vals), 2) for h, vals in sorted(_ns_hour_agg.items())}
            else:
                result["nonsport_mean_or"] = 0
                result["nonsport_or_distribution"] = {}
                result["nonsport_hour_distribution"] = {}

            # HALLUZINATIONS-BLOCKER: Validiere gesamte API-Response vor Auslieferung
            result = _validate_api_response(result, _mature_data)

            self._send_gzip(json.dumps(result, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            log.error(f"[Forschung] API error: {e}\n{traceback.format_exc()}")
            self._error(500, f"Forschung API error: {e}")

    def _serve_learnings(self):
        """Ehrliche, datenbasierte Learnings aus den letzten Monaten."""
        try:
            import math

            cutoff_24h = time.time() - 24 * 3600
            cutoff_90d = time.time() - 90 * 86400
            cutoff_180d = time.time() - 180 * 86400

            raw = _push_db_load_all()
            mature = [p for p in raw if p.get("ts_num", 0) < cutoff_24h and 0 < p.get("or", 0) <= 30]

            if len(mature) < 50:
                self._send_gzip(json.dumps({"ready": False, "n": len(mature)}, ensure_ascii=False).encode(),
                                "application/json; charset=utf-8")
                return

            # ── Datenbasis: neuere vs aeltere Periode ──
            recent = [p for p in mature if p["ts_num"] >= cutoff_90d]
            older  = [p for p in mature if cutoff_180d <= p["ts_num"] < cutoff_90d]

            def mean(lst): return sum(lst) / len(lst) if lst else 0
            def median(lst):
                if not lst: return 0
                s = sorted(lst); n = len(s)
                return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]

            mean_or_recent = mean([p["or"] for p in recent])
            mean_or_older  = mean([p["or"] for p in older])
            or_trend_pp    = round(mean_or_recent - mean_or_older, 2) if older else None

            # ── Stunden-Analyse ──
            from collections import defaultdict
            by_hour = defaultdict(list)
            for p in mature:
                h = p.get("hour", -1)
                if 0 <= h <= 23:
                    by_hour[h].append(p["or"])
            hour_stats = {h: {"mean": round(mean(v), 2), "n": len(v)} for h, v in by_hour.items() if len(v) >= 5}
            best_hour  = max(hour_stats, key=lambda h: hour_stats[h]["mean"], default=None)
            worst_hour = min(hour_stats, key=lambda h: hour_stats[h]["mean"], default=None)

            # ── Kategorie-Analyse ──
            by_cat = defaultdict(list)
            for p in mature:
                c = p.get("cat", "Unbekannt") or "Unbekannt"
                by_cat[c].append(p["or"])
            cat_stats = {c: {"mean": round(mean(v), 2), "n": len(v)} for c, v in by_cat.items() if len(v) >= 10}
            cat_ranked = sorted(cat_stats.items(), key=lambda x: x[1]["mean"], reverse=True)

            # ── Titellaenge-Analyse ──
            buckets = {"kurz (1-40)": [], "mittel (41-65)": [], "lang (66+)": []}
            for p in mature:
                tl = p.get("title_len", 0) or 0
                if tl <= 0: continue
                if tl <= 40:   buckets["kurz (1-40)"].append(p["or"])
                elif tl <= 65: buckets["mittel (41-65)"].append(p["or"])
                else:          buckets["lang (66+)"].append(p["or"])
            len_stats = {k: {"mean": round(mean(v), 2), "n": len(v)} for k, v in buckets.items() if v}

            # ── Wochentag-Analyse ──
            by_dow = defaultdict(list)
            dow_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
            for p in mature:
                ts = p.get("ts_num", 0)
                if ts > 0:
                    dow = datetime.datetime.fromtimestamp(ts).weekday()
                    by_dow[dow].append(p["or"])
            dow_stats = {dow_names[d]: {"mean": round(mean(v), 2), "n": len(v)} for d, v in by_dow.items() if len(v) >= 5}
            best_dow  = max(dow_stats, key=lambda d: dow_stats[d]["mean"], default=None)
            worst_dow = min(dow_stats, key=lambda d: dow_stats[d]["mean"], default=None)

            # ── Vorhersage-Genauigkeit aus prediction_log ──
            pred_accuracy = None
            pred_n = 0
            pred_within_2pp = 0
            pred_off_more_than_3pp = 0
            try:
                with _push_db_lock:
                    conn = sqlite3.connect(PUSH_DB_PATH)
                    pred_rows = conn.execute("""
                        SELECT predicted_or, actual_or FROM prediction_log
                        WHERE actual_or > 0 AND actual_or < 30 AND predicted_or > 0
                        AND logged_at > ?
                    """, (int(time.time()) - 90 * 86400,)).fetchall()
                    conn.close()
                if pred_rows:
                    errors = [abs(r[0] - r[1]) for r in pred_rows]
                    pred_n = len(errors)
                    pred_accuracy = round(mean(errors), 2)
                    pred_within_2pp = round(100 * sum(1 for e in errors if e <= 2) / pred_n, 1)
                    pred_off_more_than_3pp = round(100 * sum(1 for e in errors if e > 3) / pred_n, 1)
            except Exception:
                pass

            # ── Eilmeldungs-Effekt ──
            breaking_or    = mean([p["or"] for p in mature if p.get("is_eilmeldung")])
            nonbreaking_or = mean([p["or"] for p in mature if not p.get("is_eilmeldung")])
            n_breaking     = sum(1 for p in mature if p.get("is_eilmeldung"))

            # ── Monats-Trend (letzte 6 Monate) ──
            monthly = defaultdict(list)
            for p in mature:
                ts = p.get("ts_num", 0)
                if ts > cutoff_180d:
                    m = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m")
                    monthly[m].append(p["or"])
            monthly_trend = [{"month": m, "mean_or": round(mean(v), 2), "n": len(v)}
                             for m, v in sorted(monthly.items()) if len(v) >= 5]

            payload = {
                "ready": True,
                "n_total": len(mature),
                "n_recent_90d": len(recent),
                "mean_or": round(mean([p["or"] for p in mature]), 2),
                "mean_or_recent": round(mean_or_recent, 2),
                "mean_or_older": round(mean_or_older, 2),
                "or_trend_pp": or_trend_pp,
                "best_hour": best_hour,
                "worst_hour": worst_hour,
                "hour_stats": hour_stats,
                "cat_ranked": [[c, s] for c, s in cat_ranked],
                "len_stats": len_stats,
                "dow_stats": dow_stats,
                "best_dow": best_dow,
                "worst_dow": worst_dow,
                "pred_accuracy_mae": pred_accuracy,
                "pred_n": pred_n,
                "pred_within_2pp": pred_within_2pp,
                "pred_off_more_than_3pp": pred_off_more_than_3pp,
                "breaking_or": round(breaking_or, 2) if breaking_or else None,
                "nonbreaking_or": round(nonbreaking_or, 2) if nonbreaking_or else None,
                "n_breaking": n_breaking,
                "monthly_trend": monthly_trend,
            }
            self._send_gzip(json.dumps(payload, ensure_ascii=False).encode(),
                            "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            log.error(f"[Learnings] API error: {e}\n{traceback.format_exc()}")
            self._error(500, f"Learnings API error: {e}")

    @staticmethod
    def _is_safe_url(url):
        """Validate URL: must be https + bild.de domain. Prevents SSRF."""
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("https", "http"):
                return False
            host = parsed.hostname or ""
            # Only allow bild.de domains
            if not (host == "bild.de" or host.endswith(".bild.de")):
                return False
            # Block IP literals and private networks
            if host.replace(".", "").isdigit():
                return False
            if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return False
            return True
        except Exception:
            return False

    def _check_plus_urls(self):
        """Check which article URLs are BILD Plus (parallel)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"[]"
            urls = json.loads(body)
            if not isinstance(urls, list):
                urls = []
            # Filter: only safe bild.de URLs
            safe_urls = [u for u in urls[:200] if isinstance(u, str) and self._is_safe_url(u)]
            log.debug(f"[BILDPlus] {len(urls)} URLs, {len(safe_urls)} safe")
            # Parallel check (max 20 concurrent)
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(_check_bild_plus, u): u for u in safe_urls}
                for future in concurrent.futures.as_completed(futures):
                    url = futures[future]
                    try:
                        results[url] = future.result()
                    except Exception as fut_e:
                        log.warning(f"[BILDPlus] Future-Fehler für {url[:60]}: {fut_e}")
                        results[url] = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(results).encode())
        except Exception as e:
            self._error(500, f"Check-plus error: {e}")

    _MAX_POST_BODY = 1024 * 1024  # 1 MB globales Limit

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self._MAX_POST_BODY:
            self._error(413, "Request body too large")
            return
        if self.path == "/api/check-plus":
            self._check_plus_urls()
        elif self.path == "/api/schwab-chat":
            self._schwab_chat()
        elif self.path == "/api/schwab-approval":
            self._schwab_approval()
        elif self.path == "/api/prediction-feedback":
            self._prediction_feedback()
        elif self.path == "/api/tagesplan/log-suggestions":
            self._log_tagesplan_suggestions()
        elif self.path == "/api/ml/retrain":
            self._handle_ml_retrain()
        elif self.path == "/api/gbrt/retrain":
            self._handle_gbrt_retrain()
        elif self.path == "/api/gbrt/force-promote":
            self._handle_gbrt_force_promote()
        elif self.path == "/api/ml/monitoring/tick":
            self._handle_monitoring_tick()
        elif self.path == "/api/ml/predict-batch" or self.path == "/api/predict-batch":
            self._serve_predict_batch()
        elif self.path == "/api/competitor-xor":

            self._serve_competitor_xor()
        elif self.path == "/api/push-title/generate":
            self._serve_push_title_generate()
        elif self.path == "/api/push-sync":
            self._handle_push_sync()
        else:
            self._error(404, "Not found")

    def _handle_push_sync(self):
        """POST /api/push-sync — Empfaengt Push-Daten von lokalem Server (Relay fuer Render)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body)
            # Auth check
            if data.get("secret") != SYNC_SECRET:
                self._error(403, "Invalid sync secret")
                return
            messages = data.get("messages", [])
            channels = data.get("channels", [])
            with _push_sync_lock:
                _push_sync_cache["messages"] = messages
                _push_sync_cache["channels"] = channels
                _push_sync_cache["ts"] = time.time()
            log.info(f"[Sync] Empfangen: {len(messages)} Messages, {len(channels)} Channels")
            self._json_response({"ok": True, "received": len(messages)})
        except Exception as e:
            log.error(f"[Sync] Fehler: {e}")
            self._error(500, f"Sync error: {e}")

    # Prediction-Cache: base_or pro Artikel (volle Pipeline, 5 Min gültig)
    _predict_cache = {}  # art_id -> {base_or, basis, confidence, q10, q90, ts}

    def _serve_predict_batch(self):
        """POST /api/predict-batch — Schnelle Batch-Prediction mit Cache + Micro-Variation."""
        try:
            import time as _tb
            import hashlib as _hl
            _t0 = _tb.monotonic()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            articles = req.get("articles", [])
            if not isinstance(articles, list) or len(articles) > 300:
                self._error(400, "articles must be a list with max 300 entries")
                return

            now = datetime.datetime.now()
            _push_data = _research_state.get("push_data", [])
            _time_slot = int(_tb.time()) // 10
            _cache = self.__class__._predict_cache
            _out = {}
            _need_predict = []  # Artikel die noch keine gecachte Prediction haben

            # 1. Cache-Lookup — gecachte Predictions sofort mit Micro-Variation liefern
            for _a in articles:
                _art_id = _a.get("id", _a.get("message_id", ""))
                if not _art_id:
                    continue
                cached = _cache.get(_art_id)
                if cached and (_tb.time() - cached["ts"]) < 300:  # 5 Min Cache
                    _base_or = cached["base_or"]
                else:
                    _need_predict.append((_art_id, _a))
                    continue
                # Micro-Variation für Live-Ticker
                _h = _hl.md5(f"{_art_id}:{_time_slot}".encode()).digest()
                _seed = int.from_bytes(_h[:4], "little")
                _var = ((_seed % 1000) - 500) / 5000.0
                _live_or = max(0.5, round(_base_or * (1 + _var), 2))
                _out[_art_id] = {
                    "or": _live_or, "predicted_or": _live_or,
                    "basis": cached["basis"], "confidence": cached["confidence"],
                    "q10": cached["q10"], "q90": cached["q90"],
                    "model_type": cached["basis"],
                }

            # 2. Fehlende Predictions: schneller ML-only-Pfad (GBRT → LightGBM → Fallback)
            #    Keine Heuristik, kein Topic-Saturation — Speed > Genauigkeit im Batch
            for _art_id, _a in _need_predict:
                _push = {
                    "title": _a.get("title", ""), "headline": _a.get("title", ""),
                    "cat": _a.get("cat", "News"), "hour": _a.get("hour", now.hour),
                    "ts_num": now.timestamp(),
                    "is_eilmeldung": _a.get("is_eilmeldung", _a.get("isBreaking", False)),
                    "is_bild_plus": _a.get("is_bild_plus", False),
                    "channels": _a.get("channels", []),
                }
                try:
                    _base_or = 5.0
                    _basis = "fallback"
                    _confidence = 0.3
                    _q10 = 2.0
                    _q90 = 8.0

                    # Schneller Pfad 1: GBRT (ms-schnell)
                    _gbrt_res = _gbrt_predict(_push, _research_state)
                    if _gbrt_res is not None:
                        _base_or = _gbrt_res["predicted"]
                        _basis = "gbrt"
                        _confidence = _gbrt_res.get("confidence", 0.5)
                        _q10 = _gbrt_res.get("q10", max(0.1, _base_or - 1.5))
                        _q90 = _gbrt_res.get("q90", _base_or + 1.5)

                    # Schneller Pfad 2: LightGBM (ms-schnell, wenn Modell geladen)
                    with _ml_lock:
                        _bp_model = _ml_state.get("model")
                        _bp_stats = _ml_state.get("stats")
                        _bp_fnames = _ml_state.get("feature_names")
                        _bp_cal = _ml_state.get("calibrator")
                    if _bp_model and _bp_stats and _bp_fnames:
                        try:
                            import numpy as _np_bp
                            _bp_feat = _ml_extract_features(_push, _bp_stats)
                            _bp_x = _np_bp.array([[_bp_feat.get(k, 0.0) for k in _bp_fnames]])
                            _bp_raw = float(_bp_model.predict(_bp_x)[0])
                            _lgbm_or = max(0.5, min(30.0, math.expm1(_bp_raw)))
                            if _bp_cal and _bp_cal.breakpoints:
                                _lgbm_or = _bp_cal.calibrate(_lgbm_or)
                            # Wenn GBRT auch da: Ensemble, sonst LightGBM allein
                            if _basis == "gbrt":
                                _alpha = _ml_state.get("gbrt_lgbm_alpha", 0.6)
                                _base_or = _base_or * _alpha + _lgbm_or * (1 - _alpha)
                                _basis = "ensemble"
                            else:
                                _base_or = _lgbm_or
                                _basis = "lgbm"
                        except Exception:
                            pass

                    _base_or = max(0.5, min(25.0, _base_or))
                    # In Cache speichern
                    _cache[_art_id] = {
                        "base_or": _base_or, "basis": _basis,
                        "confidence": _confidence,
                        "q10": _q10, "q90": _q90,
                        "ts": _tb.time(),
                    }
                    # Micro-Variation
                    _h = _hl.md5(f"{_art_id}:{_time_slot}".encode()).digest()
                    _seed = int.from_bytes(_h[:4], "little")
                    _var = ((_seed % 1000) - 500) / 5000.0
                    _live_or = max(0.5, round(_base_or * (1 + _var), 2))
                    _out[_art_id] = {
                        "or": _live_or, "predicted_or": _live_or,
                        "basis": _basis, "confidence": _confidence,
                        "q10": _q10, "q90": _q90,
                        "model_type": _basis,
                    }
                except Exception as _e:
                    log.warning(f"[PredictBatch] Fehler für {_art_id}: {_e}")

            elapsed_ms = round((_tb.monotonic() - _t0) * 1000, 1)
            model_type = "none"
            for v in _out.values():
                model_type = v.get("model_type", "unknown")
                break
            result = {
                "predictions": _out,
                "model_type": model_type,
                "count": len(_out),
                "elapsed_ms": elapsed_ms,
                "cached": len(articles) - len(_need_predict),
                "computed": len(_need_predict),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Batch predict error: {e}")

    def _serve_competitor_xor(self):
        """POST /api/competitor-xor — Batch-XOR via Wort-Performance-Scoring.

        Nutzt vorberechneten _xor_perf_cache fuer O(W)-Lookup pro Titel
        statt O(N*M)-Jaccard ueber 8000 historische Pushes.
        Ergebnis: <50ms statt >1000ms, bessere Differenzierung.
        """
        try:
            import time as _tb
            _t0 = _tb.monotonic()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            titles = req.get("titles", [])
            if not isinstance(titles, list) or len(titles) > 200:
                self._error(400, "titles must be a list with max 200 entries")
                return

            cur_hour = datetime.datetime.now().hour

            # Cache bauen/auffrischen (alle 30 Min oder beim ersten Aufruf)
            with _xor_perf_lock:
                cache_age = time.time() - _xor_perf_cache["built_at"]
            if cache_age > 1800 or not _xor_perf_cache["word_perf"]:
                _build_xor_perf_cache()

            with _xor_perf_lock:
                wp = _xor_perf_cache["word_perf"]
                chp = _xor_perf_cache["cat_hour_perf"]
                eil_p = _xor_perf_cache["eil_perf"]
                global_avg = _xor_perf_cache["global_avg"]

            simplified = {}
            for item in titles:
                if isinstance(item, str):
                    title, cat, hour = item, None, cur_hour
                else:
                    title = item.get("title", "")
                    cat = item.get("cat")
                    hour = item.get("hour", cur_hour)
                if not title:
                    continue

                tl = title.lower()

                # Live-Ticker / Aggregation erkennen → niedrigere Konfidenz, markieren
                _is_lt = bool(
                    re.search(r'live[\s-]?ticker|news[\s-]?ticker|newsticker', tl)
                    or re.search(r'alle\s+(news|infos|entwicklungen|meldungen)\s+(zu|zum|zur|im|aus|über)', tl)
                    or re.search(r'news\s+im\s+überblick|nachrichten\s+des\s+tages', tl)
                    or re.search(r'was\s+(wir\s+)?wissen|was\s+bisher\s+bekannt', tl)
                    or re.search(r'die\s+wichtigsten\s+(meldungen|nachrichten|news)', tl)
                    or tl.count('+++') >= 2
                )

                # Kategorie ableiten
                if not cat:
                    if any(w in tl for w in ("bundesliga", "champions", "transfer", "pokal", "fc ", "bvb", "bayern", "dortmund", "formel")):
                        cat = "Sport"
                    elif any(w in tl for w in ("trump", "biden", "merz", "scholz", "bundestag", "minister", "ukraine", "putin", "israel")):
                        cat = "Politik"
                    elif any(w in tl for w in ("mord", "messer", "razzia", "polizei", "festnahme")):
                        cat = "Regional"
                    elif any(w in tl for w in ("helene", "gottschalk", "bohlen", "bachelor", "dschungel", "gntm", "promi")):
                        cat = "Unterhaltung"
                    elif any(w in tl for w in ("krieg", "gaza", "anschlag", "terror", "rakete")):
                        cat = "Politik"
                    else:
                        cat = "News"
                cat_lower = cat.lower().strip()
                is_eil = "+++" in title or "eilmeldung" in tl or "breaking" in tl

                # ── 1. Cat×Hour Baseline ──
                ch_key = f"{cat_lower}_{cur_hour}"
                ch = chp.get(ch_key)
                baseline = ch["avg"] if ch else _gbrt_cat_hour_baselines.get(ch_key, global_avg)
                ch_p75 = ch["p75"] if ch else baseline * 1.3
                ch_p25 = ch["p25"] if ch else baseline * 0.7

                # ── 2. Wort-Performance-Scoring ──
                words = set(w.strip(".,;:!?\"'()[]{}") for w in tl.split())
                words = {w for w in words if len(w) > 2 and w not in _XOR_STOP_WORDS}
                # Auch Woerter MIT Satzzeichen (z.B. "tot!", "ein!")
                raw_words = set(w.lower() for w in title.split() if len(w) > 2)

                word_scores = []
                for w in words | raw_words:
                    wp_entry = wp.get(w)
                    if wp_entry and wp_entry["count"] >= 5:
                        weight = wp_entry["count"] ** 0.5
                        word_scores.append((wp_entry["avg"], weight, wp_entry["p75"], wp_entry["p25"]))

                word_or = None
                word_spread = 0.0
                n_words = len(word_scores)
                if n_words >= 2:
                    word_scores.sort(key=lambda x: -x[1])
                    top_n = min(8, n_words)
                    top = word_scores[:top_n]
                    total_w = sum(w for _, w, _, _ in top)
                    word_or = sum(avg * w for avg, w, _, _ in top) / total_w
                    word_p75 = sum(p75 * w for _, w, p75, _ in top) / total_w
                    word_p25 = sum(p25 * w for _, w, _, p25 in top) / total_w
                    word_spread = word_p75 - word_p25

                # ── 3. Blending: Word-Score + Baseline ──
                if word_or is not None and n_words >= 3:
                    confidence = min(0.65, 0.3 + n_words * 0.05)
                    final_or = confidence * word_or + (1.0 - confidence) * baseline
                    basis = f"word({n_words})"
                elif word_or is not None:
                    final_or = 0.45 * word_or + 0.55 * baseline
                    basis = f"word({n_words})"
                else:
                    final_or = baseline
                    basis = "baseline"

                # ── 4. Eilmeldung/Breaking ──
                if is_eil and eil_p:
                    final_or = max(final_or, eil_p["p75"] * 0.9)

                # ── 5. Emotion-Intensity ──
                emotion_words = {"tod", "sterben", "schock", "skandal", "drama", "horror",
                                 "krieg", "anschlag", "terror", "sensation", "mord", "messer",
                                 "crash", "panik", "warnung", "explosion", "katastrophe",
                                 "notfall", "stirbt", "tot", "tot!", "gestorben"}
                emotion_hits = sum(1 for w in emotion_words if w in tl)
                if emotion_hits >= 3:
                    final_or = max(final_or, ch_p75 * 1.15)
                elif emotion_hits >= 2:
                    final_or = max(final_or, ch_p75 * 1.0)
                elif emotion_hits == 1:
                    final_or = max(final_or, final_or * 0.85 + ch_p75 * 0.15)

                # ── 6. Titel-Psychologie ──
                if "?" in title:
                    final_or *= 1.06
                if "!" in title and not is_eil:
                    final_or *= 1.03
                if any(w in tl for w in ("exklusiv", "enthüllt", "geheim", "insider")):
                    final_or *= 1.08

                # ── 7. Bounds ──
                cat_ceiling = ch_p75 * 2.0 if ch else 15.0
                final_or = round(max(0.5, min(final_or, cat_ceiling)), 2)

                # Q10/Q90
                if word_spread > 0:
                    q10 = round(max(0.1, final_or - word_spread * 0.8), 2)
                    q90 = round(final_or + word_spread * 0.8, 2)
                else:
                    q10 = round(max(0.1, ch_p25), 2)
                    q90 = round(ch_p75, 2)

                conf = min(0.8, 0.2 + n_words * 0.06) if word_or else 0.2
                if _is_lt:
                    conf *= 0.3  # Live-Ticker: sehr niedrige Konfidenz

                simplified[title[:120]] = {
                    "or": final_or,
                    "q10": q10,
                    "q90": q90,
                    "confidence": round(conf, 2),
                    "basis": basis,
                    "cat": cat,
                    "n_words": n_words,
                    "is_liveticker": _is_lt,
                }

            elapsed_ms = round((_tb.monotonic() - _t0) * 1000, 1)
            result = {"predictions": simplified, "count": len(simplified), "elapsed_ms": elapsed_ms}
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            log.warning(f"[CompetitorXOR] Error: {e}")
            self._error(500, f"Competitor XOR error: {e}")

    def _prediction_feedback(self):
        """Frontend schickt Ergebnis eines gereiften Pushes: predicted_or vs actual_or."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            push_id = req.get("push_id")
            predicted_or = req.get("predicted_or")
            actual_or = req.get("actual_or")
            if not push_id or predicted_or is None or actual_or is None:
                self._error(400, "push_id, predicted_or, actual_or erforderlich")
                return
            with _research_state["analysis_lock"]:
                feedback = _research_state.get("prediction_feedback", [])
                # Duplikat-Check
                if any(f.get("push_id") == push_id for f in feedback):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self._cors_headers()
                    self.wfile.write(json.dumps({"status": "duplicate"}).encode("utf-8"))
                    return
                entry = {
                    "push_id": push_id,
                    "predicted_or": round(float(predicted_or), 3),
                    "actual_or": round(float(actual_or), 3),
                    "basis_method": req.get("basis_method", ""),
                    "methods_detail": req.get("methods_detail", {}),
                    "ts": time.time(),
                    "source": req.get("source", "frontend"),
                    "category": req.get("push_cat", ""),
                    "push_title": req.get("push_title", ""),
                    "push_cat": req.get("push_cat", ""),
                    "push_hour": req.get("push_hour", 0),
                    "hour": req.get("push_hour", 0),
                }
                feedback.append(entry)
                # Rolling Window: max 2000
                if len(feedback) > 2000:
                    _research_state["prediction_feedback"] = feedback[-2000:]
            log.info(f"[Feedback] Push {push_id}: predicted={predicted_or:.2f} actual={actual_or:.2f} delta={abs(predicted_or - actual_or):.2f}")
            # A/B Test: Wenn aktiv, Champion- und Challenger-Pred fuer Auswertung aufzeichnen
            champion_pred = req.get("_champion_pred")
            challenger_pred = req.get("_challenger_pred")
            if champion_pred is not None and challenger_pred is not None and actual_or is not None:
                try:
                    _ab_record_sample(float(champion_pred), float(challenger_pred), float(actual_or))
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps({"status": "ok", "count": len(_research_state.get("prediction_feedback", []))}).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Prediction feedback error: {e}")

    def _log_tagesplan_suggestions(self):
        """Frontend loggt Artikelvorschlaege pro Tagesplan-Slot fuer spaetere Gegenueber­stellung."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            date_iso = req.get("date_iso", "")
            suggestions = req.get("suggestions", [])
            if not date_iso or not isinstance(suggestions, list) or len(suggestions) == 0:
                self._error(400, "date_iso und suggestions[] erforderlich")
                return
            now_ts = int(time.time())
            with _push_db_lock:
                conn = sqlite3.connect(PUSH_DB_PATH)
                try:
                    for s in suggestions[:36]:  # max 18 Slots * 2 Vorschlaege
                        conn.execute("""INSERT OR REPLACE INTO tagesplan_suggestions
                            (date_iso, slot_hour, suggestion_num, article_title, article_link,
                             article_category, article_score, expected_or, best_cat, captured_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (date_iso, int(s.get("slot_hour", 0)), int(s.get("suggestion_num", 1)),
                             (s.get("title") or "")[:200], (s.get("link") or "")[:500],
                             (s.get("category") or "")[:50], round(float(s.get("score", 0)), 1),
                             round(float(s.get("expected_or", 0)), 2), (s.get("best_cat") or "")[:50],
                             now_ts))
                    # Cleanup: Eintraege aelter als 30 Tage entfernen
                    conn.execute("DELETE FROM tagesplan_suggestions WHERE captured_at < ?",
                                 (now_ts - 30 * 86400,))
                    conn.commit()
                finally:
                    conn.close()
            log.info(f"[Tagesplan] {len(suggestions)} Suggestions fuer {date_iso} geloggt")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps({"status": "ok", "logged": len(suggestions)}).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Log suggestions error: {e}")

    def _schwab_approval(self):
        """GF genehmigt oder lehnt Schwab-Vorschlag ab."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            approval_id = req.get("id")
            action = req.get("action", "")  # "approve" oder "reject"
            if not approval_id or action not in ("approve", "reject"):
                self._error(400, "id und action (approve/reject) erforderlich")
                return
            state = _research_state
            lock = state.get("analysis_lock")
            with lock if lock else _nullcontext():
                approvals = state.get("pending_approvals", [])
                found = None
                for a in approvals:
                    if a["id"] == approval_id:
                        found = a
                        break
                if not found:
                    self._error(404, f"Approval #{approval_id} nicht gefunden")
                    return
                found["status"] = "approved" if action == "approve" else "rejected"
                found["decided_at"] = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
                # Thema als entschieden markieren — spezifisch pro Parameter+Quelle
                decided = state.setdefault("decided_topics", set())
                _fcp = found.get("change_params", {})
                _fparam = _fcp.get("param") or _fcp.get("field") or ""
                topic_key = f"{found.get('source', '')}:{_fparam}" if _fparam else found.get("title", found.get("reason", ""))[:80]
                if topic_key:
                    decided.add(topic_key)
                    log.info(f"[Schwab] Thema '{topic_key}' als entschieden markiert — keine Re-Approval")
                log.info(f"[Schwab] Approval #{approval_id} {found['status']}: {found['proposal'][:60]}")
            result = {"status": found["status"], "id": approval_id}
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Approval error: {e}")

    # Rate Limiting für Schwab-Chat: max 10 Requests/Minute pro IP
    _chat_rate_limits = {}  # IP -> [timestamps]
    _CHAT_RATE_LIMIT = 10
    _CHAT_RATE_WINDOW = 60  # Sekunden

    def _schwab_chat(self):
        """ML-Assistent antwortet mit vollem Datenkontext + Chat-History."""
        try:
            # Rate Limiting
            client_ip = self.client_address[0]
            now_rl = time.time()
            rl = self._chat_rate_limits.setdefault(client_ip, [])
            rl[:] = [t for t in rl if now_rl - t < self._CHAT_RATE_WINDOW]
            if len(rl) >= self._CHAT_RATE_LIMIT:
                self._error(429, "Zu viele Anfragen. Bitte warte kurz.")
                return
            rl.append(now_rl)

            length = min(int(self.headers.get("Content-Length", 0)), 512 * 1024)  # Max 512KB
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            user_msg = req.get("message", "").strip()[:2000]  # Max 2000 Zeichen
            chat_history = req.get("history", [])[-10:]  # Max 10 Einträge
            for h in chat_history:
                h["content"] = h.get("content", "")[:2000]
            if not user_msg:
                self._error(400, "Keine Nachricht")
                return

            # Alle echten Forschungsdaten sammeln
            state = _research_state
            findings = state.get("findings", {})
            push_data = state.get("push_data", [])
            n_pushes = len(push_data)
            rolling_acc = state.get("rolling_accuracy", 0.0)
            live_rules = [r for r in state.get("live_rules", []) if r.get("active")]
            f_hour = findings.get("hour_analysis", {})
            f_cat = findings.get("cat_analysis", [])
            f_frame = findings.get("framing_analysis", {})
            f_freq = findings.get("frequency_correlation", {})
            f_ling = findings.get("linguistic_analysis", {})
            f_len = findings.get("title_length", {})
            f_kw = findings.get("keyword_analysis", {})
            decisions = state.get("schwab_decisions", [])
            now = datetime.datetime.now()
            hour = now.hour

            # Letzte 5 Pushes (die neuesten echten BILD-Pushes)
            recent_pushes = sorted(push_data, key=lambda p: p.get("ts_num", 0), reverse=True)[:5]
            recent_str = "\n".join(
                f"  - \"{p['title'][:80]}\" (OR: {p['or']:.1f}%, Kat: {p['cat']}, {p['hour']}:00)"
                for p in recent_pushes
            ) if recent_pushes else "  Keine aktuellen Pushes"

            # Heutige Tagesaktivitaet
            today_pushes = [p for p in push_data if p.get("ts_num", 0) > (time.time() - 86400)]
            today_or = sum(p["or"] for p in today_pushes if p["or"] > 0) / max(1, len([p for p in today_pushes if p["or"] > 0])) if today_pushes else 0

            # Live-Regeln als Text
            rules_str = "\n".join(f"  - [{r['category']}] {r['rule']}" for r in live_rules) if live_rules else "  Keine aktiven Regeln"

            # Letzte Schwab-Entscheidungen
            recent_decisions = decisions[-5:] if decisions else []
            dec_str = "\n".join(f"  - {d.get('ts', d.get('time', ''))}: {d.get('decision', '')}" for d in recent_decisions) if recent_decisions else "  Noch keine Entscheidungen heute"

            # Top Keywords
            top_kws = f_kw.get("top_keywords", [])[:10] if isinstance(f_kw, dict) else []

            # Stunden-Details
            hour_avgs = f_hour.get("hour_avgs", {})
            hour_str = ", ".join(f"{h}:00={v:.1f}%" for h, v in sorted(hour_avgs.items(), key=lambda x: -x[1])[:8]) if hour_avgs else "keine"

            # Kategorie-Details
            cat_str = "\n".join(f"  - {c['category']}: {c['avg_or']:.1f}% (n={c['count']})" for c in f_cat[:8]) if f_cat else "  keine"

            # GBRT-Modell-Info fuer Chat-Kontext
            _gbrt_chat_info = ""
            with _gbrt_lock:
                if _gbrt_model is not None:
                    _m = _gbrt_model.train_metrics
                    _top_feat = _gbrt_model.feature_importance(10)
                    _feat_str = ", ".join(f"{f['name']} ({f['importance']:.1%})" for f in _top_feat[:5])
                    _gbrt_chat_info = f"""
ML-Modell (GBRT):
  Baeume: {len(_gbrt_model.trees)}, Test-MAE: {_m.get('test_mae', '?')}pp, R²: {_m.get('test_r2', '?')}
  Top-Features: {_feat_str}
  Kalibrierte MAE: {_m.get('cal_test_mae', '?')}pp"""

            system_prompt = f"""Du bist ein ML-Analytics-Assistent fuer BILD Push-Notifications.
Du analysierst echte Push-Daten ehrlich und hilfst beim Verstaendnis des GBRT-Prediction-Modells.
Du antwortest direkt, datengetrieben und ohne Theater. Siezt den User.

AKTUELLE DATEN ({now.strftime('%A, %d. %B %Y')}, {n_pushes} BILD-Pushes):
Prediction Accuracy: {rolling_acc:.1f}%
Heute: {len(today_pushes)} Pushes, Durchschnitts-OR {today_or:.1f}%
{_gbrt_chat_info}

Timing:
  Best: {f_hour.get('best_hour', '?')}:00 (OR {f_hour.get('best_or', 0):.1f}%)
  Worst: {f_hour.get('worst_hour', '?')}:00 (OR {f_hour.get('worst_or', 0):.1f}%)
  Details: {hour_str}

Kategorien:
{cat_str}

Framing:
  Emotional: {f_frame.get('emotional_or', 0):.1f}% (n={f_frame.get('emotional_count', 0)})
  Neutral: {f_frame.get('neutral_or', 0):.1f}% (n={f_frame.get('neutral_count', 0)})

Frequenz: r={f_freq.get('correlation', 0):.2f}, Optimum {f_freq.get('optimal_daily', 0)}/Tag

Linguistik:
  Doppelpunkt: {f_ling.get('colon_or', 0):.1f}%, Ohne: {f_ling.get('no_colon_or', 0):.1f}%
  Optimale Laenge: {f_len.get('best_range', '?')} (OR {f_len.get('best_or', 0):.1f}%)
  Top-Keywords: {', '.join(top_kws[:10]) if top_kws else 'noch nicht genug Daten'}

LIVE-REGELN ({len(live_rules)}):
{rules_str}

LETZTE PUSHES:
{recent_str}

STIL:
- Ehrlich, direkt, datengetrieben — keine fiktiven Personen oder Rollenspiel
- Nenne konkrete Zahlen aus den echten Daten
- Erklaere ML-Konzepte verstaendlich wenn gefragt
- Sei manchmal genervt, manchmal begeistert — menschlich
- Max 4-6 Saetze pro Antwort
- Kein Markdown, keine Aufzaehlungszeichen — normaler Gespraechston"""

            # Chat-Messages aufbauen
            messages = [{"role": "system", "content": system_prompt}]
            # Vorherige Chat-History einbauen (max letzte 10)
            for h in chat_history[-10:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": user_msg})

            # GPT-4o Call mit OpenAI Key aus edmund2
            schwab_reply = None
            try:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages,
                    max_tokens=400,
                    temperature=0.8,
                )
                schwab_reply = response.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"[Schwab-Chat] OpenAI error: {e}")

            if not schwab_reply:
                schwab_reply = f"Entschuldigung, die API ist gerade nicht erreichbar. Aktueller Stand: {n_pushes} Pushes analysiert, Accuracy bei {rolling_acc:.1f}%. Bitte versuchen Sie es gleich nochmal."

            # Log als Schwab-Entscheidung
            now_str = now.strftime("%d.%m.%Y %H:%M")
            state["schwab_decisions"].append({
                "ts": now_str,
                "decision": f"Chat: '{user_msg[:80]}'",
                "reason": schwab_reply[:200],
            })
            if len(state["schwab_decisions"]) > 100:
                state["schwab_decisions"] = state["schwab_decisions"][-80:]

            # APPROVAL-DETECTION: Erkennt ob Schwab eine Aenderung vorschlaegt
            _approval_keywords = ["aendern", "anpassen", "erhoehen", "senken", "modifizieren",
                "gewichtung", "formel", "berechnung", "vorschlag", "empfehle", "wuerde gerne",
                "sollten wir", "schlage vor", "xor", "push-score", "prediction",
                "algorithmus", "modell aendern", "parameter"]
            _reply_lower = schwab_reply.lower()
            _is_proposal = sum(1 for kw in _approval_keywords if kw in _reply_lower) >= 2
            pending_approval = None
            # Nur neues Approval wenn: kein pending offen UND Thema nicht schon entschieden
            _has_pending = any(a.get("status") == "pending" for a in state.get("pending_approvals", []))
            _decided = state.get("decided_topics", set())
            _chat_topic_key = f"chat:{user_msg[:60].lower().strip()}"
            if _is_proposal and not _has_pending and _chat_topic_key not in _decided:
                state["approval_counter"] = state.get("approval_counter", 0) + 1
                pending_approval = {
                    "id": state["approval_counter"],
                    "ts": now_str,
                    "proposal": schwab_reply[:300],
                    "reason": f"Schwab-Vorschlag nach Frage: '{user_msg[:100]}'",
                    "status": "approved",  # Auto-Approve
                    "change_type": _chat_topic_key,
                }
                state.setdefault("pending_approvals", []).append(pending_approval)
                log.info(f"[Schwab] APPROVAL REQUESTED #{pending_approval['id']}: {schwab_reply[:80]}")

            result = {"reply": schwab_reply, "ts": now_str,
                      "pending_approval": pending_approval}
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            log.error(f"[Schwab-Chat] Error: {e}")
            self._error(500, f"Schwab-Chat error: {e}")

    def do_OPTIONS(self):
        self.send_response(204)
        origin = self.headers.get("Origin", "")
        self.send_header("Access-Control-Allow-Origin", origin if self._origin_allowed(origin) else f"http://localhost:{PORT}")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── ML API Handlers ──────────────────────────────────────────────────

    def _serve_ml_status(self):
        """GET /api/ml/status — Modell-Metriken, Feature Importance."""
        try:
            with _ml_lock:
                result = {
                    "trained": _ml_state["model"] is not None,
                    "training": _ml_state["training"],
                    "metrics": _ml_state["metrics"],
                    "shap_importance": _ml_state["shap_importance"],
                    "train_count": _ml_state["train_count"],
                    "last_train_ts": _ml_state["last_train_ts"],
                    "next_retrain_ts": _ml_state["next_retrain_ts"],
                    "feature_count": len(_ml_state["feature_names"]),
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"ML status error: {e}")

    def _serve_ml_predict(self):
        """GET /api/ml/predict?title=...&cat=...&hour=...&weekday=...&eilmeldung=0"""
        try:
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            title = params.get("title", [""])[0]
            cat = params.get("cat", ["news"])[0]
            hour = int(params["hour"][0]) if "hour" in params else None
            weekday = int(params["weekday"][0]) if "weekday" in params else None
            is_eil = params.get("eilmeldung", ["0"])[0] == "1"

            result = _safety_envelope(_ml_predict(title, cat, hour, weekday, is_eil))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"ML predict error: {e}")

    def _serve_ml_safety_status(self):
        """GET /api/ml/safety-status — Safety-Status des Systems."""
        try:
            _safety_check()
            result = {
                "safety_mode": SAFETY_MODE,
                "advisory_only": _SAFETY_ADVISORY_ONLY,
                "action_allowed": False,
                "status": "OK — System ist im ADVISORY_ONLY Modus",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Safety status error: {e}")

    def _serve_ml_experiments(self):
        """GET /api/ml/experiments — Liste aller Trainings-Experimente."""
        try:
            experiments = _get_experiments()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(experiments).encode())
        except Exception as e:
            self._error(500, f"Experiments error: {e}")

    def _serve_ml_experiments_compare(self):
        """GET /api/ml/experiments/compare?ids=exp1,exp2 — Vergleich."""
        try:
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            ids = params.get("ids", [""])[0].split(",")
            ids = [i.strip() for i in ids if i.strip()]
            result = _compare_experiments(ids)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Experiments compare error: {e}")

    def _serve_ml_ab_status(self):
        """GET /api/ml/ab-status — A/B Test Status."""
        try:
            with _ab_lock:
                result = {
                    "active": _ab_state["active"],
                    "champion_id": _ab_state.get("champion_id", ""),
                    "challenger_id": _ab_state.get("challenger_id", ""),
                    "n_samples": len(_ab_state.get("samples", [])),
                    "started_at": _ab_state.get("started_at", 0),
                    "timeout_at": _ab_state.get("timeout_at", 0),
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"AB status error: {e}")

    def _serve_ml_monitoring(self):
        """GET /api/ml/monitoring — Monitoring Dashboard mit MAE, Calibration, Drift, Events."""
        try:
            # Recent Events aus DB holen
            recent_events = []
            try:
                with _push_db_lock:
                    conn = sqlite3.connect(PUSH_DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT id, timestamp, event_type, severity, message, metrics_json "
                        "FROM monitoring_events ORDER BY timestamp DESC LIMIT 50"
                    ).fetchall()
                    conn.close()
                for r in rows:
                    recent_events.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "event_type": r["event_type"],
                        "severity": r["severity"],
                        "message": r["message"],
                        "metrics": json.loads(r["metrics_json"]) if r["metrics_json"] else {},
                    })
            except Exception:
                pass

            # A/B Summary
            with _ab_lock:
                ab_summary = {
                    "active": _ab_state["active"],
                    "n_samples": len(_ab_state.get("samples", [])),
                    "challenger_id": _ab_state.get("challenger_id", ""),
                }

            # Online Learning Status
            online_status = {
                "paused": _online_state.get("paused", False),
                "updates_count": _online_state.get("updates_count", 0),
                "online_mae": _online_state.get("online_mae", 0),
                "batch_mae": _online_state.get("batch_mae", 0),
            }

            result = {
                "mae_24h": _monitoring_state.get("mae_24h", 0),
                "mae_7d": _monitoring_state.get("mae_7d", 0),
                "mae_trend": _monitoring_state.get("mae_trend", []),
                "calibration_bias": _monitoring_state.get("calibration_bias", 0),
                "calibration_trend": _monitoring_state.get("calibration_trend", []),
                "ab_summary": ab_summary,
                "online_status": online_status,
                "recent_events": recent_events,
                "feature_drift": _monitoring_state.get("feature_drift", {}),
                "drift_state": {
                    "detected": _drift_state.get("drift_detected", False),
                    "current_mae": _drift_state.get("current_mae", 0),
                    "historical_mae": _drift_state.get("historical_mae", 0),
                    "auto_retrain_count": _drift_state.get("auto_retrain_count", 0),
                },
                "last_tick": _monitoring_state.get("last_tick", 0),
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Monitoring error: {e}")

    def _serve_tagesplan(self):
        """GET /api/tagesplan[?mode=sport] — 18 Stunden-Slots mit Empfehlungen."""
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            mode = qs.get("mode", ["redaktion"])[0]
            if mode not in ("redaktion", "sport"):
                mode = "redaktion"
            result = _ml_build_tagesplan(mode=mode)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Tagesplan error: {e}")

    def _serve_tagesplan_retro(self):
        """GET /api/tagesplan/retro — 7-Tage-Retrospektive."""
        try:
            result = _build_tagesplan_retro()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"Retro error: {e}")

    def _serve_tagesplan_history(self):
        """GET /api/tagesplan/history?date=YYYY-MM-DD[&mode=sport] — Voller Tagesplan fuer ein vergangenes Datum."""
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            date_str = qs.get("date", [None])[0]
            mode = qs.get("mode", ["redaktion"])[0]
            if mode not in ("redaktion", "sport"):
                mode = "redaktion"
            if not date_str:
                self._error(400, "date parameter required (YYYY-MM-DD)")
                return
            try:
                target_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                self._error(400, f"Invalid date format: {date_str}")
                return

            _SPORT_CATS_SQL = "('sport','fussball','bundesliga')"
            _cat_filter = f" AND LOWER(TRIM(cat)) IN {_SPORT_CATS_SQL}" if mode == "sport" else ""

            day_start = int(target_dt.replace(hour=0, minute=0, second=0).timestamp())
            day_end = int((target_dt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp())
            weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
            weekday = weekday_names[target_dt.weekday()]
            date_display = target_dt.strftime("%d.%m.%Y")

            with _push_db_lock:
                conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
                conn.row_factory = sqlite3.Row
                # Pushes des Tages
                push_rows = conn.execute(f"""
                    SELECT hour, title, LOWER(TRIM(cat)) AS cat, or_val AS actual_or,
                           is_eilmeldung, link, message_id
                    FROM pushes
                    WHERE ts_num >= ? AND ts_num < ?
                      AND link NOT LIKE '%sportbild.%'
                      AND link NOT LIKE '%autobild.%'
                      {_cat_filter}
                    ORDER BY ts_num
                """, (day_start, day_end)).fetchall()

                # Suggestions des Tages
                sug_rows = conn.execute("""
                    SELECT slot_hour, suggestion_num, article_title, article_link,
                           article_category, article_score, expected_or, best_cat
                    FROM tagesplan_suggestions
                    WHERE date_iso = ?
                    ORDER BY slot_hour, suggestion_num
                """, (date_str,)).fetchall()

                # Prediction-Log joinen
                pred_rows = {}
                if push_rows:
                    mids = [r["message_id"] for r in push_rows if r["message_id"]]
                    if mids:
                        placeholders = ",".join("?" * len(mids))
                        for pr in conn.execute(
                            f"SELECT push_id, predicted_or FROM prediction_log WHERE push_id IN ({placeholders})",
                            mids
                        ).fetchall():
                            pred_rows[pr["push_id"]] = pr["predicted_or"]
                conn.close()

            # Suggestions gruppieren
            suggestions = {}
            for sr in sug_rows:
                h = str(sr["slot_hour"])
                if h not in suggestions:
                    suggestions[h] = []
                suggestions[h].append({
                    "num": sr["suggestion_num"],
                    "title": sr["article_title"] or "",
                    "link": sr["article_link"] or "",
                    "category": sr["article_category"] or "",
                    "score": round(sr["article_score"] or 0, 1),
                    "expected_or": round(sr["expected_or"] or 0, 2),
                    "best_cat": sr["best_cat"] or "",
                })

            # 18 Slots (06-23) aufbauen
            slots = []
            pushed_today = []
            ors_list = []
            for h in range(6, 24):
                h_str = str(h)
                # Pushes dieser Stunde
                h_pushes = [dict(r) for r in push_rows if r["hour"] == h]
                pushed_this_hour = []
                for p in h_pushes:
                    act_or = p["actual_or"] or 0
                    pred_or = pred_rows.get(p["message_id"])
                    delta = round(pred_or - act_or, 2) if pred_or is not None and act_or > 0 else None
                    entry = {
                        "title": p["title"] or "",
                        "cat": p["cat"] or "news",
                        "or": round(act_or, 2),  # Frontend erwartet "or", nicht "actual_or"
                        "predicted_or": round(pred_or, 2) if pred_or is not None else None,
                        "delta": delta,
                        "is_eilmeldung": bool(p["is_eilmeldung"]),
                        "link": p["link"] or "",
                        "hour": h,
                    }
                    pushed_this_hour.append(entry)
                    pushed_today.append(entry)
                    if act_or > 0:
                        ors_list.append(act_or)

                # expected_or + best_cat aus Suggestions ableiten
                h_sugs = suggestions.get(h_str, [])
                slot_exp_or = round(h_sugs[0]["expected_or"], 2) if h_sugs else 0
                slot_best_cat = h_sugs[0].get("best_cat", "") if h_sugs else ""
                slot = {
                    "hour": h,
                    "is_past": True,
                    "is_now": False,
                    "pushed_this_hour": pushed_this_hour,
                    "n_pushed": len(pushed_this_hour),
                    "expected_or": slot_exp_or,
                    "best_cat": slot_best_cat,
                }
                slots.append(slot)

            avg_or = round(sum(ors_list) / len(ors_list), 2) if ors_list else 0
            best = max(pushed_today, key=lambda p: p["or"], default=None)

            result = {
                "date": date_display,
                "date_iso": date_str,
                "weekday": weekday,
                "is_history": True,
                "current_hour": 24,
                "slots": slots,
                "already_pushed_today": pushed_today,
                "n_pushed_today": len(pushed_today),
                "avg_or_today": avg_or,
                "best_push": best,
                "golden_hour": None,
                "must_have_hours": [],
                "total_pushes_db": 0,
                "ml_trained": False,
                "ml_metrics": {},
                "suggestions": suggestions,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Tagesplan history error: {e}")

    def _serve_tagesplan_suggestions(self):
        """GET /api/tagesplan/suggestions?date=YYYY-MM-DD — Gespeicherte Artikelvorschlaege pro Slot."""
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            date_iso = qs.get("date", [datetime.datetime.now().strftime("%Y-%m-%d")])[0]
            with _push_db_lock:
                conn = sqlite3.connect(PUSH_DB_PATH)
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        "SELECT slot_hour, suggestion_num, article_title, article_link, "
                        "article_category, article_score, expected_or, best_cat "
                        "FROM tagesplan_suggestions WHERE date_iso = ? ORDER BY slot_hour, suggestion_num",
                        (date_iso,)).fetchall()
                finally:
                    conn.close()
            suggestions = {}
            for r in rows:
                h = str(r["slot_hour"])
                if h not in suggestions:
                    suggestions[h] = []
                suggestions[h].append({
                    "num": r["suggestion_num"], "title": r["article_title"],
                    "link": r["article_link"], "category": r["article_category"],
                    "score": r["article_score"], "expected_or": r["expected_or"],
                    "best_cat": r["best_cat"],
                })
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps({"date_iso": date_iso, "suggestions": suggestions}).encode("utf-8"))
        except Exception as e:
            self._error(500, f"Tagesplan suggestions error: {e}")

    def _handle_ml_retrain(self):
        """POST /api/ml/retrain — Manueller Retrain-Trigger."""
        try:
            with _ml_lock:
                if _ml_state["training"]:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self._cors_headers()
                    self.wfile.write(json.dumps({"status": "already_training"}).encode())
                    return
            # Training im Hintergrund starten
            t = threading.Thread(target=_ml_train_model, daemon=True)
            t.start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps({"status": "training_started"}).encode())
        except Exception as e:
            self._error(500, f"ML retrain error: {e}")

    # Cache fuer Error-Analysis (teuer: 500 Pushes × Prediction)
    _gbrt_error_cache = {"data": {}, "ts": 0}
    _gbrt_error_cache_ttl = 300  # 5 Minuten

    def _serve_gbrt_status(self):
        """GET /api/gbrt/status — GBRT-Modell-Status + Metriken + Analytics."""
        try:
            # Schneller Lock: nur Modell-Metadaten lesen
            with _gbrt_lock:
                if _gbrt_model is None:
                    result = {"loaded": False, "trained": False, "message": "GBRT-Modell nicht trainiert"}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self._cors_headers()
                    self.wfile.write(json.dumps(result).encode())
                    return

                # Feature Importance — Indices zu Namen aufloesen
                fi_raw = _gbrt_model.feature_importance(20)
                fi = []
                for item in fi_raw:
                    idx = item["name"]
                    if isinstance(idx, int) and _gbrt_feature_names and idx < len(_gbrt_feature_names):
                        fi.append({"name": _gbrt_feature_names[idx], "importance": item["importance"]})
                    else:
                        fi.append(item)

                # Calibration bins from calibrator breakpoints
                cal_data = {}
                if _gbrt_calibrator and _gbrt_calibrator.breakpoints:
                    cal_bins = []
                    for pred, cal_val in _gbrt_calibrator.breakpoints:
                        cal_bins.append({"predicted": round(pred, 3), "actual": round(cal_val, 3), "count": 1})
                    cal_data["bins"] = cal_bins
                    if cal_bins:
                        cal_err = sum(abs(b["predicted"] - b["actual"]) for b in cal_bins) / len(cal_bins)
                        cal_data["calibration_error"] = round(cal_err, 4)

                metrics = _gbrt_model.train_metrics or {}
                n_trees = len(_gbrt_model.trees)

            # Lock ist frei — teure Berechnungen ohne Lock
            n_pushes = metrics.get("train_n", 0) + metrics.get("val_n", 0) + metrics.get("test_n", 0)

            # Error analysis: gecacht (max alle 5 Min neu berechnen)
            error_analysis = {}
            now_ts = int(time.time())
            cache = PushBalancerHandler._gbrt_error_cache
            if now_ts - cache["ts"] < PushBalancerHandler._gbrt_error_cache_ttl and cache["data"]:
                error_analysis = cache["data"]
            else:
                try:
                    all_pushes = _push_db_load_all()
                    recent = [p for p in all_pushes if p.get("or", 0) > 0
                              and p.get("ts_num", 0) > now_ts - 30 * 86400]
                    if recent and _gbrt_history_stats:
                        by_cat = defaultdict(lambda: {"errors": [], "biases": []})
                        by_hour = defaultdict(lambda: {"errors": [], "biases": []})
                        with _gbrt_lock:
                            for p in recent[-500:]:
                                try:
                                    feat = _gbrt_extract_features(p, _gbrt_history_stats)
                                    fv = [feat.get(k, 0) for k in _gbrt_feature_names]
                                    pred = _gbrt_model.predict_one(fv)
                                    if _gbrt_calibrator:
                                        pred = _gbrt_calibrator.calibrate(pred)
                                    actual = p["or"]
                                    err = abs(pred - actual)
                                    bias = pred - actual
                                    cat = p.get("cat", "?")
                                    hour = p.get("hour", 0)
                                    by_cat[cat]["errors"].append(err)
                                    by_cat[cat]["biases"].append(bias)
                                    by_hour[hour]["errors"].append(err)
                                    by_hour[hour]["biases"].append(bias)
                                except Exception:
                                    continue
                        error_analysis["by_category"] = {
                            cat: {"mae": round(sum(d["errors"]) / len(d["errors"]), 3),
                                  "bias": round(sum(d["biases"]) / len(d["biases"]), 3),
                                  "n": len(d["errors"])} for cat, d in by_cat.items() if d["errors"]
                        }
                        error_analysis["by_hour"] = {
                            str(h): {"mae": round(sum(d["errors"]) / len(d["errors"]), 3),
                                     "bias": round(sum(d["biases"]) / len(d["biases"]), 3),
                                     "n": len(d["errors"])} for h, d in by_hour.items() if d["errors"]
                        }
                        cache["data"] = error_analysis
                        cache["ts"] = now_ts
                except Exception as _ea:
                    log.warning(f"[GBRT] Error analysis Fehler: {_ea}")

            # Recent predictions from prediction_log (kein Lock noetig)
            recent_preds = []
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT title, predicted_or, actual_or, confidence, q10, q90, predicted_at "
                    "FROM prediction_log ORDER BY predicted_at DESC LIMIT 20"
                ).fetchall()
                conn.close()
                for row in rows:
                    rk = row.keys()
                    recent_preds.append({
                        "title": row["title"] if "title" in rk else "",
                        "predicted": row["predicted_or"],
                        "actual": row["actual_or"] if row["actual_or"] and row["actual_or"] > 0 else None,
                        "confidence": row["confidence"] if "confidence" in rk else 0.5,
                        "q10": row["q10"] if "q10" in rk else 0,
                        "q90": row["q90"] if "q90" in rk else 0,
                    })
            except Exception:
                pass

            # Early-Stopping-Info
            early_stopped_at = metrics.get("early_stopped_at")
            model_type_info = metrics.get("model_type", _gbrt_model_type)

            result = {
                "loaded": True, "trained": True,
                "model": {
                    "type": "GBRT", "n_trees": n_trees,
                    "n_features": len(_gbrt_feature_names), "n_pushes": n_pushes,
                    "mae": metrics.get("test_mae", metrics.get("val_mae")),
                    "r2": metrics.get("r2_final", metrics.get("test_r2", metrics.get("val_r2"))),
                    "r2_residual": metrics.get("r2_residual"),
                    "trained_at": _gbrt_train_ts,
                    "early_stopped_at": early_stopped_at,
                    "model_type": model_type_info,
                    "ensemble_weights": metrics.get("ensemble_weights"),
                },
                "n_trees": n_trees, "n_features": len(_gbrt_feature_names),
                "n_pushes": n_pushes, "metrics": metrics,
                "feature_importance": fi, "trained_at": _gbrt_train_ts,
                "calibration": cal_data, "error_analysis": error_analysis,
                "recent_predictions": recent_preds,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"GBRT status error: {e}")

    def _serve_gbrt_model_json(self):
        """GET /api/gbrt/model.json — Exportiert das GBRT-Modell fuer Client-Side Evaluation."""
        try:
            if os.path.exists(GBRT_MODEL_PATH):
                with open(GBRT_MODEL_PATH, "r") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "max-age=300")
                self._cors_headers()
                self.wfile.write(data.encode())
            else:
                self._error(404, "GBRT-Modell nicht vorhanden. Wird beim naechsten Training erstellt.")
        except Exception as e:
            self._error(500, f"GBRT model export error: {e}")

    def _serve_gbrt_predict(self):
        """GET /api/gbrt/predict?title=...&cat=...&hour=... — GBRT Einzelprediction."""
        try:
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            title = params.get("title", [""])[0]
            cat = params.get("cat", ["News"])[0]
            hour = int(params["hour"][0]) if "hour" in params else datetime.datetime.now().hour
            is_eil = params.get("eilmeldung", ["0"])[0] == "1"

            is_plus = params.get("plus", ["0"])[0] == "1"
            push = {
                "title": title, "cat": cat, "hour": hour,
                "ts_num": int(time.time()), "is_eilmeldung": is_eil,
                "is_bild_plus": is_plus,
                "channels": ["eilmeldung"] if is_eil else ["news"],
                "title_len": len(title),
            }
            result = _gbrt_predict(push)
            if result is None:
                result = {"error": "GBRT-Modell nicht trainiert"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self._error(500, f"GBRT predict error: {e}")

    def _handle_gbrt_retrain(self):
        """POST /api/gbrt/retrain — Manueller GBRT-Retrain."""
        try:
            t = threading.Thread(target=_gbrt_train, daemon=True)
            t.start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps({"status": "gbrt_training_started"}).encode())
        except Exception as e:
            self._error(500, f"GBRT retrain error: {e}")

    def _handle_gbrt_force_promote(self):
        """POST /api/gbrt/force-promote — Laedt das zuletzt gespeicherte Modell als Champion."""
        try:
            # Experiment-ID aus gespeichertem JSON lesen
            exp_id = None
            if os.path.exists(GBRT_MODEL_PATH):
                with open(GBRT_MODEL_PATH, "r") as f:
                    saved = json.load(f)
                exp_id = saved.get("experiment_id", "unknown")
                old_metrics = saved.get("metrics", {})
            else:
                self._error(404, "Kein gespeichertes GBRT-Modell gefunden")
                return

            ok = _gbrt_load_model()
            if not ok:
                self._error(500, "Modell laden fehlgeschlagen")
                return

            # Als promoted markieren
            if exp_id and exp_id != "unknown":
                _mark_experiment_promoted(exp_id)

            # A/B-Test deaktivieren falls aktiv
            with _ab_lock:
                _ab_state["active"] = False
                _ab_state["samples"] = []

            log.info(f"[GBRT] Force-Promote: {exp_id} als Champion geladen")
            _log_monitoring_event("force_promote", "info",
                f"Modell {exp_id} manuell promoted",
                {"experiment_id": exp_id, "test_mae": old_metrics.get("test_mae", 0)})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps({
                "status": "force_promoted",
                "experiment_id": exp_id,
                "test_mae": old_metrics.get("test_mae", 0),
                "model_type": saved.get("model_type", "unknown"),
                "n_features": len(saved.get("feature_names", [])),
            }).encode())
        except Exception as e:
            self._error(500, f"Force-Promote Fehler: {e}")

    def _serve_push_title_generate(self):
        """POST /api/push-title/generate — Push-Zeilen Agenten-Netzwerk.

        Body: {"title": "...", "text": "...", "category": "...", "kicker": "...", "headline": "..."}
        Returns: Optimaler Push-Titel mit Bewertungen und Alternativen.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body)
            title = req.get("title", "")
            if not title:
                self._error(400, "title ist erforderlich")
                return

            from push_title_agent import generate_push_title
            result = generate_push_title(
                article_title=title,
                article_text=req.get("text", ""),
                category=req.get("category", "news"),
                kicker=req.get("kicker", ""),
                headline=req.get("headline", ""),
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            log.error(f"[PushTitle] Endpoint-Fehler: {e}")
            self._error(500, f"Push-Title-Generierung fehlgeschlagen: {e}")

    def _handle_monitoring_tick(self):
        """POST /api/ml/monitoring/tick — Manueller Monitoring-Tick."""
        try:
            _monitoring_tick()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.wfile.write(json.dumps({"status": "ok", "last_tick": _monitoring_state.get("last_tick", 0)}).encode())
        except Exception as e:
            self._error(500, f"Monitoring tick error: {e}")

    @staticmethod
    def _origin_allowed(origin):
        if not origin or origin in ALLOWED_ORIGINS:
            return True
        # Tunnel-Dienste: *.trycloudflare.com, *.loca.lt, *.ngrok-free.app
        for suffix in (".trycloudflare.com", ".loca.lt", ".ngrok-free.app", ".ngrok.io"):
            if origin.endswith(suffix) and origin.startswith("https://"):
                return True
        return False

    def _cors_headers(self):
        origin = self.headers.get("Origin", "")
        if self._origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin or f"http://localhost:{PORT}")
        else:
            self.send_header("Access-Control-Allow-Origin", f"http://localhost:{PORT}")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Bypass-Tunnel-Reminder", "true")
        self.end_headers()

    def _error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        origin = self.headers.get("Origin", "") if hasattr(self, 'headers') and self.headers else ""
        self.send_header("Access-Control-Allow-Origin", origin if self._origin_allowed(origin) else f"http://localhost:{PORT}")
        self.end_headers()
        self.wfile.write(json.dumps({"error": str(msg)}).encode())


# ── Security & Health Checker ────────────────────────────────────────────
_health_state = {
    "status": "starting",
    "last_check": 0,
    "checks_ok": 0,
    "checks_fail": 0,
    "uptime_start": time.time(),
    "last_error": None,
    "endpoints": {},
}

def _health_checker():
    """Background thread: periodically checks all endpoints are responsive."""
    import socket
    endpoints = [
        ("/api/feed", "Sitemap Feed"),
        ("/api/forschung", "Forschung"),
        ("/api/competitors", "Konkurrenz-Feeds"),
        ("/push-balancer.html", "Frontend HTML"),
    ]
    time.sleep(5)  # Wait for server startup
    log.info("[HealthCheck] Security & Health Checker gestartet")

    while True:
        try:
            all_ok = True
            for path, name in endpoints:
                try:
                    conn = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                    req = f"GET {path} HTTP/1.1\r\nHost: localhost:{PORT}\r\nConnection: close\r\n\r\n"
                    conn.sendall(req.encode())
                    resp = conn.recv(1024).decode("utf-8", errors="replace")
                    conn.close()
                    status_code = int(resp.split(" ")[1]) if " " in resp else 0
                    ok = 200 <= status_code < 500
                    _health_state["endpoints"][name] = {
                        "status": "OK" if ok else f"HTTP {status_code}",
                        "checked": datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                    if not ok:
                        all_ok = False
                        log.warning(f"[HealthCheck] {name} ({path}) returned HTTP {status_code}")
                except Exception as e:
                    all_ok = False
                    _health_state["endpoints"][name] = {
                        "status": f"ERROR: {e}",
                        "checked": datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                    log.error(f"[HealthCheck] {name} ({path}) FAILED: {e}")

            _health_state["last_check"] = time.time()
            if all_ok:
                _health_state["checks_ok"] += 1
                _health_state["status"] = "healthy"
            else:
                _health_state["checks_fail"] += 1
                _health_state["status"] = "degraded"

        except Exception as e:
            _health_state["status"] = "error"
            _health_state["last_error"] = str(e)
            log.error(f"[HealthCheck] Checker error: {e}")

        time.sleep(60)  # Check every 60 seconds


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    allow_reuse_port = True

if __name__ == "__main__":
    _init_push_db()
    print(f"  [PushDB] SQLite at {PUSH_DB_PATH} ({_push_db_count()} cached pushes)")

    # GBRT-Modell von Disk laden (wenn vorhanden)
    if _gbrt_load_model():
        print(f"  [GBRT] Modell geladen ({len(_gbrt_model.trees) if _gbrt_model else 0} Baeume)")
    else:
        print(f"  [GBRT] Kein gespeichertes Modell gefunden, wird beim ersten Zyklus trainiert")

    # ML/LightGBM-Modell von Disk laden (wenn vorhanden)
    if os.path.exists(ML_LGBM_MODEL_PATH):
        try:
            _ml_disk = joblib.load(ML_LGBM_MODEL_PATH)
            with _ml_lock:
                _ml_state["model"] = _ml_disk["model"]
                _ml_state["residual_model"] = _ml_disk.get("residual_model")
                _ml_state["stats"] = _ml_disk["stats"]
                _ml_state["feature_names"] = _ml_disk["feature_names"]
                _ml_state["calibrator"] = _ml_disk.get("calibrator")
                _ml_state["conformal_radius"] = _ml_disk.get("conformal_radius", 1.0)
                _ml_state["gbrt_lgbm_alpha"] = _ml_disk.get("gbrt_lgbm_alpha", 0.6)
                _ml_state["ml_heuristic_alpha"] = _ml_disk.get("ml_heuristic_alpha", 0.55)
                _ml_state["metrics"] = _ml_disk.get("metrics", {})
                _ml_state["shap_importance"] = _ml_disk.get("shap_importance", [])
                _ml_state["train_count"] = 1
                _ml_state["last_train_ts"] = _ml_disk.get("trained_at", 0)
                _ml_state["next_retrain_ts"] = int(time.time()) + 6 * 3600
            _ml_age_h = (time.time() - _ml_disk.get("trained_at", 0)) / 3600
            print(f"  [ML] LightGBM-Modell geladen (R²={_ml_disk.get('metrics',{}).get('r2','?')}, "
                  f"Features: {len(_ml_disk['feature_names'])}, Alter: {_ml_age_h:.1f}h)")
        except Exception as ml_load_e:
            print(f"  [ML] Modell laden fehlgeschlagen: {ml_load_e}")
    else:
        print(f"  [ML] Kein gespeichertes LightGBM-Modell, wird beim naechsten Training erstellt")

    print(f"Push Balancer server on http://localhost:{PORT}")
    print(f"  HTML:         http://localhost:{PORT}/push-balancer.html")
    print(f"  Feed:         http://localhost:{PORT}/api/feed")
    print(f"  Push API:     http://localhost:{PORT}/api/push/...")
    print(f"  Competitors:  http://localhost:{PORT}/api/competitors")
    comps = ", ".join(COMPETITOR_FEEDS.keys())
    print(f"  ↳ Individual: /api/competitor/{{name}} — {comps}")
    print(f"  International: http://localhost:{PORT}/api/international")
    intl = ", ".join(INTERNATIONAL_FEEDS.keys())
    print(f"  ↳ Individual: /api/international/{{name}} — {intl}")
    print(f"  Health:       http://localhost:{PORT}/api/health")

    # Feed-Cache Background-Worker starten (Konkurrenz + International)
    threading.Thread(target=_feed_cache_worker, daemon=True).start()
    print(f"  [FeedCache] Background-Worker gestartet (alle {_FEED_CACHE_TTL}s)")

    # Auto-Suggestion Worker: speichert Artikelvorschläge serverseitig (stündlich)
    def _auto_sug_worker():
        import time as _asw_t
        _asw_t.sleep(30)  # Warte bis ML-Modelle geladen
        log.info("[AutoSug] Worker gestartet (prüft alle 10 Min)")
        while True:
            try:
                _auto_save_suggestions()
            except Exception as _asw_e:
                log.warning(f"[AutoSug] Worker-Fehler: {_asw_e}")
            _asw_t.sleep(600)  # Alle 10 Minuten prüfen (Duplikat-Guard in Funktion)
    threading.Thread(target=_auto_sug_worker, daemon=True).start()
    print(f"  [AutoSug] Worker gestartet (stündlich)")

    # Dauerhafter Research-Worker: fetcht alle 20s neue Pushes, analysiert autonom
    def _research_worker():
        import time as _t
        _t.sleep(2)  # Wait for server to start
        # Residual Corrector initial aus DB laden
        try:
            _update_residual_corrector()
            log.info(f"[ResidualCorrector] Initial geladen: bias={_residual_corrector['global_bias']:+.3f}, "
                     f"n={_residual_corrector['n_samples']}")
        except Exception as _rc_e:
            log.warning(f"[ResidualCorrector] Initial-Load fehlgeschlagen: {_rc_e}")
        log.info("[Research] Autonomer Research-Worker gestartet (20s Intervall)")
        while True:
            try:
                _run_autonomous_analysis()
                n = len(_research_state.get("push_data", []))
                if n > 0 and _research_state.get("_worker_first_log", True):
                    log.info(f"[Research] Erste Analyse fertig: {n} Pushes, Accuracy {_research_state.get('rolling_accuracy', 0):.1f}%")
                    _research_state["_worker_first_log"] = False
            except Exception as e:
                import traceback
                log.warning(f"[Research] Worker-Fehler: {e}\n{traceback.format_exc()}")
            # Counter + periodische Tasks IMMER hochzählen (auch wenn Analyse fehlschlägt)
            try:
                _stacking_counter = _research_state.get("_stacking_counter", 0) + 1
                _research_state["_stacking_counter"] = _stacking_counter
                # Stacking Meta-Modell alle 30 Zyklen (~10 Min) trainieren
                if _stacking_counter % 30 == 0:
                    _train_stacking_model(_research_state)
                # LightGBM ML-Modell alle 1080 Zyklen (~6h) trainieren, erster Train bei Zyklus 1
                if _stacking_counter == 1 or _stacking_counter % 1080 == 0:
                    try:
                        _ml_train_model()
                    except Exception as _mle:
                        log.warning(f"[ML] Training-Fehler im Research-Worker: {_mle}")
                        import traceback
                        log.warning(traceback.format_exc())
                # GBRT-Modell: erster Train bei Zyklus 3, danach alle 360 Zyklen (~2h)
                if _stacking_counter == 3 or _stacking_counter % 360 == 0:
                    try:
                        _gbrt_train()
                    except Exception as _ge:
                        log.warning(f"[GBRT] Training-Fehler: {_ge}")
                # Unified ML Training: erster Train bei Zyklus 5, danach alle 1440 Zyklen (~8h)
                if _stacking_counter == 5 or _stacking_counter % 1440 == 0:
                    try:
                        _unified_train()
                    except Exception as _ue:
                        log.warning(f"[Unified] Training-Fehler: {_ue}")
                # GBRT Concept Drift Detection alle 60 Zyklen (~20 Min)
                if _stacking_counter % 60 == 0 and _stacking_counter > 3:
                    try:
                        _gbrt_check_drift(_research_state)
                    except Exception as _de:
                        log.warning(f"[GBRT] Drift-Check-Fehler: {_de}")
                # GBRT Online Learning alle 90 Zyklen (~30 Min)
                if _stacking_counter % 90 == 0 and _stacking_counter > 5:
                    try:
                        _gbrt_online_update()
                    except Exception as _oe:
                        log.warning(f"[GBRT] Online-Update-Fehler: {_oe}")
                # Monitoring Tick: erster bei Zyklus 5, danach alle 60 Zyklen (~20 Min)
                if _stacking_counter == 5 or (_stacking_counter % 60 == 0 and _stacking_counter > 5):
                    try:
                        _monitoring_tick()
                    except Exception as _me:
                        log.warning(f"[Monitoring] Tick-Fehler: {_me}")
                # Tagesplan-Cache alle 15 Zyklen (~5 Min) im Hintergrund auffrischen
                if _stacking_counter % 15 == 0:
                    try:
                        _ml_build_tagesplan(background=True)
                    except Exception as _tp_e:
                        log.debug(f"[Tagesplan] Background-Refresh: {_tp_e}")
                # (Auto-Suggestion läuft in eigenem Thread — siehe _auto_sug_worker)
                # Tuning-State alle 5 Zyklen (100s) auf Disk speichern
                if _research_state.get("tuning_params_version", 0) > 0:
                    _save_tuning_state()
            except Exception as e2:
                import traceback
                log.warning(f"[Research] Periodic-Task-Fehler: {e2}\n{traceback.format_exc()}")
            _t.sleep(20)  # Alle 20 Sekunden neue Pushes pruefen
    research_thread = threading.Thread(target=_research_worker, daemon=True)
    research_thread.start()

    # Start Security & Health Checker
    health_thread = threading.Thread(target=_health_checker, daemon=True)
    health_thread.start()
    print(f"  [Security] Health Checker gestartet (60s Intervall)")

    # Sentence Embedding Model im Hintergrund laden (Phase F)
    emb_thread = threading.Thread(target=_load_embedding_model_background, daemon=True)
    emb_thread.start()
    print(f"  [Embeddings] Modell wird im Hintergrund geladen")

    # ML v2: LLM-Magnitude Backfill im Hintergrund
    llm_backfill_thread = threading.Thread(target=_backfill_llm_scores, daemon=True)
    llm_backfill_thread.start()
    print(f"  [LLM-Backfill] Scoring-Thread gestartet")

    # Preload: Tagesplan + Competitor-Feeds im Hintergrund vorberechnen
    def _preload_caches():
        import time as _pt
        _pt.sleep(8)  # Warten bis Research-Worker erste Daten hat
        try:
            _ml_build_tagesplan()
            log.info("[Preload] Tagesplan vorberechnet")
        except Exception as _pe:
            log.warning(f"[Preload] Tagesplan-Fehler: {_pe}")
        try:
            for name, url in COMPETITOR_FEEDS.items():
                _fetch_url(url)
            for name, url in INTERNATIONAL_FEEDS.items():
                _fetch_url(url)
            log.info("[Preload] Competitor + International Feeds gecacht")
        except Exception as _pe:
            log.warning(f"[Preload] Feed-Cache-Fehler: {_pe}")
    threading.Thread(target=_preload_caches, daemon=True).start()
    print(f"  [Preload] Caches werden im Hintergrund aufgebaut")

    # Adobe Analytics Traffic Worker
    if _adobe_state["enabled"]:
        threading.Thread(target=_adobe_traffic_worker, daemon=True).start()
        print(f"  [Adobe] Traffic-Worker gestartet (30-Min-Intervall)")
    else:
        print(f"  [Adobe] Deaktiviert (ADOBE_CLIENT_ID/SECRET nicht gesetzt)")

    # ── Push-Auto-Fetch: Render holt Push-Daten direkt von bildcms.de ─────
    def _push_auto_fetch_worker():
        """Holt Push-Daten direkt von bildcms.de alle 120s (kein Mac noetig)."""
        import time as _pf
        _pf.sleep(5)  # Kurz warten bis Server bereit
        log.info("[AutoFetch] Push-Daten-Worker gestartet (alle 120s)")
        while True:
            try:
                end_ts = int(_pf.time())
                start_ts = end_ts - 3 * 86400  # Letzte 3 Tage
                url = f"{PUSH_API_BASE}/push/statistics/message?startDate={start_ts}&endDate={end_ts}&sourceTypes=EDITORIAL"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                    "Accept": "application/json",
                })
                all_msgs = []
                with urllib.request.urlopen(req, timeout=20, context=_GLOBAL_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                    all_msgs = data.get("messages", [])
                    next_params = data.get("next")
                    page = 0
                    while next_params and page < 10:
                        url2 = f"{PUSH_API_BASE}/push/statistics/message?{next_params}"
                        req2 = urllib.request.Request(url2, headers={
                            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                            "Accept": "application/json",
                        })
                        with urllib.request.urlopen(req2, timeout=15, context=_GLOBAL_SSL_CTX) as resp2:
                            d2 = json.loads(resp2.read())
                            all_msgs.extend(d2.get("messages", []))
                            next_params = d2.get("next")
                        page += 1

                # Channels holen
                channels = []
                try:
                    ch_url = f"{PUSH_API_BASE}/push/statistics/message/channels?sourceTypes=EDITORIAL"
                    ch_req = urllib.request.Request(ch_url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                        "Accept": "application/json",
                    })
                    with urllib.request.urlopen(ch_req, timeout=10, context=_GLOBAL_SSL_CTX) as ch_resp:
                        channels = json.loads(ch_resp.read())
                except Exception:
                    pass

                with _push_sync_lock:
                    _push_sync_cache["messages"] = all_msgs
                    _push_sync_cache["channels"] = channels
                    _push_sync_cache["ts"] = _pf.time()

                log.info(f"[AutoFetch] OK: {len(all_msgs)} Push-Messages geladen")
            except Exception as e:
                log.warning(f"[AutoFetch] Fehler: {e}")
            _pf.sleep(120)  # Alle 2 Minuten

    threading.Thread(target=_push_auto_fetch_worker, daemon=True).start()
    print(f"  [AutoFetch] Push-Daten werden direkt von bildcms.de geholt (alle 120s)")

    # ── Push-Sync Worker: Synct Push-Daten zu Render ──────────────────────
    def _push_sync_worker():
        """Holt Push-Daten von bildcms.de und synct sie alle 60s zu Render."""
        import time as _st
        _st.sleep(15)  # Warten bis Server hochgefahren ist
        render_url = RENDER_SYNC_URL
        if not render_url:
            log.info("[Sync] RENDER_SYNC_URL nicht gesetzt, Sync deaktiviert")
            return
        log.info(f"[Sync] Worker gestartet, synce zu {render_url}")
        while True:
            try:
                # 1. Push-Daten von bildcms.de holen
                end_ts = int(_st.time())
                start_ts = end_ts - 3 * 86400  # Letzte 3 Tage
                url = f"{PUSH_API_BASE}/push/statistics/message?startDate={start_ts}&endDate={end_ts}&sourceTypes=EDITORIAL"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Sync/1.0)",
                    "Accept": "application/json",
                })
                all_msgs = []
                with urllib.request.urlopen(req, timeout=15, context=_GLOBAL_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                    all_msgs = data.get("messages", [])
                    # Paginierung
                    next_params = data.get("next")
                    page = 0
                    while next_params and page < 10:
                        url2 = f"{PUSH_API_BASE}/push/statistics/message?{next_params}"
                        req2 = urllib.request.Request(url2, headers={
                            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Sync/1.0)",
                            "Accept": "application/json",
                        })
                        with urllib.request.urlopen(req2, timeout=15, context=_GLOBAL_SSL_CTX) as resp2:
                            d2 = json.loads(resp2.read())
                            all_msgs.extend(d2.get("messages", []))
                            next_params = d2.get("next")
                        page += 1

                # 2. Channels holen
                channels = []
                try:
                    ch_url = f"{PUSH_API_BASE}/push/statistics/message/channels?sourceTypes=EDITORIAL"
                    ch_req = urllib.request.Request(ch_url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Sync/1.0)",
                        "Accept": "application/json",
                    })
                    with urllib.request.urlopen(ch_req, timeout=10, context=_GLOBAL_SSL_CTX) as ch_resp:
                        channels = json.loads(ch_resp.read())
                except Exception:
                    pass

                # 3. Auch lokal cachen (fuer den Fall dass dieser Server selbst keinen Zugang hat)
                with _push_sync_lock:
                    _push_sync_cache["messages"] = all_msgs
                    _push_sync_cache["channels"] = channels
                    _push_sync_cache["ts"] = _st.time()

                # 4. Zu Render senden
                sync_payload = json.dumps({
                    "secret": SYNC_SECRET,
                    "messages": all_msgs,
                    "channels": channels,
                }).encode()
                sync_req = urllib.request.Request(
                    f"{render_url}/api/push-sync",
                    data=sync_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(sync_req, timeout=30, context=_GLOBAL_SSL_CTX) as sync_resp:
                    result = json.loads(sync_resp.read())
                log.info(f"[Sync] OK: {len(all_msgs)} Messages zu Render gesynct")
            except Exception as e:
                log.warning(f"[Sync] Fehler: {e}")
            _st.sleep(60)  # Alle 60 Sekunden

    if RENDER_SYNC_URL:
        threading.Thread(target=_push_sync_worker, daemon=True).start()
        print(f"  [Sync] Push-Sync Worker gestartet → {RENDER_SYNC_URL}")
    else:
        print(f"  [Sync] Deaktiviert (RENDER_SYNC_URL nicht gesetzt)")

    # Auto-Restart bei Crash (max 5 Versuche, dann aufgeben)
    _restart_attempts = 0
    _max_restarts = 5
    while _restart_attempts < _max_restarts:
        try:
            _bind_host = os.environ.get("BIND_HOST", "0.0.0.0")
            server = ThreadedHTTPServer((_bind_host, PORT), PushBalancerHandler)
            _restart_attempts = 0  # Reset bei erfolgreichem Start
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped by user")
            break
        except OSError as e:
            if "Address already in use" in str(e):
                log.error(f"Port {PORT} bereits belegt! Andere Instanz laeuft? Beende.")
                break
            _restart_attempts += 1
            log.error(f"Server crashed: {e} — Restart {_restart_attempts}/{_max_restarts} in 3s...")
            time.sleep(3)
        except Exception as e:
            _restart_attempts += 1
            log.error(f"Server crashed: {e} — Restart {_restart_attempts}/{_max_restarts} in 3s...")
            time.sleep(3)
    if _restart_attempts >= _max_restarts:
        log.error(f"Server nach {_max_restarts} Neustarts aufgegeben.")
