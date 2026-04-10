"""app/main.py — FastAPI Application + Route-Registrierungen + Startup-Events.

Ersetzt den frueheren monolithischen HTTP-Handler.

Startup-Sequenz (uebernommen aus dem frueheren Monolithen):
1. DB initialisieren (init_db)
2. GBRT-Modell von Disk laden
3. LightGBM-Modell von Disk laden
4. Push-Snapshot seeden (wenn vorhanden)
5. Feed-Cache Background-Worker starten
6. Research-Worker starten (20s-Intervall)
7. Health-Checker starten
8. Embedding-Modell im Hintergrund laden
9. LLM-Backfill-Thread starten
10. Tagesplan + Feed-Cache vorberechnen
11. Adobe Analytics Traffic-Worker starten (wenn konfiguriert)
12. Push-Auto-Fetch-Worker starten
13. Push-Sync-Worker starten (wenn RENDER_SYNC_URL gesetzt)
14. Auto-Suggestion-Worker starten
"""
import ipaddress
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    ALLOWED_ORIGINS,
    INTERNAL_ACCESS_ALLOWED_CIDRS,
    INTERNAL_ACCESS_ENABLED,
    INTERNAL_ACCESS_EXEMPT_PATHS,
    PORT,
    SERVE_DIR,
    SNAPSHOT_PATH,
)
from app.database import init_db, push_db_count, push_db_upsert
from app.ml.gbrt import gbrt_load_model
from app.routers import (
    feed,
    forschung,
    gbrt,
    health,
    misc,
    ml,
    push,
    tagesplan,
)

log = logging.getLogger("push-balancer")

# ── Logging konfigurieren ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
for _h in logging.root.handlers:
    if hasattr(_h, "stream") and hasattr(_h.stream, "reconfigure"):
        try:
            _h.stream.reconfigure(line_buffering=True)
        except Exception:
            pass

# libomp für LightGBM/XGBoost vorab laden (macOS SIP blockiert DYLD_LIBRARY_PATH)
import ctypes as _ctypes
_omp_lib = os.path.expanduser("~/.local/lib/libomp.dylib")
if os.path.exists(_omp_lib):
    try:
        _ctypes.cdll.LoadLibrary(_omp_lib)
    except OSError:
        pass


# ── Startup / Shutdown ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup und Shutdown Lifecycle Handler."""
    # ── 1. Datenbank initialisieren ──
    init_db()
    log.info("[PushDB] SQLite initialisiert (%d Pushes)", push_db_count())

    # ── 2. Push-Snapshot seeden ──
    _seed_push_snapshot()

    # ── 3. Background-Worker starten ──
    _start_background_workers()

    # ── 4. ML-Modelle im Hintergrund laden (deferred — kein RAM-Spike beim Start) ──
    # Die aktiven app/ml-Modelle (GBRT, LightGBM) laden unabhängig vom Legacy-Referenzcode.

    def _load_ml_models_background():
        import time as _t
        _t.sleep(2)
        try:
            if gbrt_load_model():
                from app.ml.gbrt import _gbrt_model as _m
                n_trees = len(getattr(_m, "trees", []))
                log.info("[GBRT] Modell geladen (%d Bäume)", n_trees)
            else:
                log.info("[GBRT] Kein gespeichertes Modell, wird beim ersten Zyklus trainiert")
        except Exception as e:
            log.warning("[GBRT] Modell-Load fehlgeschlagen: %s", e)
        try:
            _load_lgbm_model_from_disk()
        except Exception as e:
            log.warning("[ML] LightGBM-Load fehlgeschlagen: %s", e)

    threading.Thread(target=_load_ml_models_background, daemon=True).start()

    log.info("Push Balancer FastAPI auf http://0.0.0.0:%d", PORT)

    yield  # Server läuft

    log.info("[Shutdown] Push Balancer beendet")


def _load_lgbm_model_from_disk() -> None:
    """Lädt gespeichertes LightGBM-Modell von Disk (wenn vorhanden)."""
    try:
        import joblib
        from app.ml.lightgbm_model import _ml_lock, _ml_state
    except ImportError:
        return

    # Modellpfad identisch zum frueheren Monolithen
    ml_model_path = os.path.join(SERVE_DIR, ".ml_lgbm_model.pkl")
    if not os.path.exists(ml_model_path):
        log.info("[ML] Kein gespeichertes LightGBM-Modell, wird beim nächsten Training erstellt")
        return

    try:
        ml_disk = joblib.load(ml_model_path)
        with _ml_lock:
            _ml_state["model"] = ml_disk["model"]
            _ml_state["residual_model"] = ml_disk.get("residual_model")
            _ml_state["stats"] = ml_disk.get("stats")
            _ml_state["feature_names"] = ml_disk["feature_names"]
            _ml_state["calibrator"] = ml_disk.get("calibrator")
            _ml_state["conformal_radius"] = ml_disk.get("conformal_radius", 1.0)
            _ml_state["gbrt_lgbm_alpha"] = ml_disk.get("gbrt_lgbm_alpha", 0.6)
            _ml_state["ml_heuristic_alpha"] = ml_disk.get("ml_heuristic_alpha", 0.55)
            _ml_state["metrics"] = ml_disk.get("metrics", {})
            _ml_state["shap_importance"] = ml_disk.get("shap_importance", [])
            _ml_state["train_count"] = 1
            _ml_state["last_train_ts"] = ml_disk.get("trained_at", 0)
            _ml_state["next_retrain_ts"] = int(time.time()) + 6 * 3600

        ml_age_h = (time.time() - ml_disk.get("trained_at", 0)) / 3600
        r2 = ml_disk.get("metrics", {}).get("r2", "?")
        n_feats = len(ml_disk["feature_names"])
        log.info("[ML] LightGBM geladen (R²=%s, Features: %d, Alter: %.1fh)", r2, n_feats, ml_age_h)
    except Exception as e:
        log.warning("[ML] Modell laden fehlgeschlagen: %s", e)


def _problem_response(
    request: Request,
    status_code: int,
    title: str,
    detail: str,
    problem_type: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": problem_type,
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": str(request.url.path),
        },
        media_type="application/problem+json",
    )


_DEPRECATED_COMPATIBILITY_EXACT_PATHS = {
    "/api/competitors",
    "/api/sport-competitors",
    "/api/forschung",
    "/api/learnings",
    "/api/adobe/traffic",
    "/api/ml/status",
    "/api/ml/monitoring",
    "/api/ml/retrain",
    "/api/ml/monitoring/tick",
    "/api/predict-batch",
    "/api/gbrt/status",
    "/api/gbrt/model.json",
    "/api/gbrt/retrain",
    "/api/gbrt/force-promote",
}
_DEPRECATED_COMPATIBILITY_PREFIXES = (
    "/api/push/",
)
_DEPRECATION_SUNSET = "Wed, 31 Dec 2026 23:59:59 GMT"


def _is_deprecated_compatibility_path(path: str) -> bool:
    return path in _DEPRECATED_COMPATIBILITY_EXACT_PATHS or any(
        path.startswith(prefix) for prefix in _DEPRECATED_COMPATIBILITY_PREFIXES
    )


def _apply_runtime_headers(path: str, response: Response) -> Response:
    if _is_deprecated_compatibility_path(path):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = _DEPRECATION_SUNSET
    return response


def _path_is_exempt_from_internal_access(path: str) -> bool:
    for exempt_path in INTERNAL_ACCESS_EXEMPT_PATHS:
        if path == exempt_path or path.startswith(f"{exempt_path}/"):
            return True
    return False


def _extract_client_ip(request: Request) -> str | None:
    for header_name in (
        "cf-connecting-ip",
        "true-client-ip",
        "x-real-ip",
    ):
        header_value = request.headers.get(header_name, "").strip()
        if header_value:
            return header_value

    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        candidate = forwarded_for.split(",", 1)[0].strip()
        if candidate:
            return candidate

    if request.client and request.client.host:
        return request.client.host

    return None


def _client_is_on_allowed_network(client_ip: str | None) -> bool:
    if not client_ip:
        return False

    try:
        parsed_ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for cidr in INTERNAL_ACCESS_ALLOWED_CIDRS:
        try:
            if parsed_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            log.warning("[Access] Ungültige INTERNAL_ACCESS_ALLOWED_CIDRS-Konfiguration: %s", cidr)

    return False


def _frontend_index_path() -> str:
    return os.path.join(SERVE_DIR, "index.html")


def _legacy_frontend_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_push_balancer.html")


def _load_frontend_html() -> str | None:
    index_path = _frontend_index_path()
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as index_file:
            return index_file.read()
    except OSError:
        return None


def _prepare_frontend_html_for_request(html: str, request_path: str) -> str:
    if "/dist-frontend/assets/" not in html:
        return html

    rewritten_html = html.replace("/dist-frontend/assets/", "/assets/")
    if request_path == "/push-balancer.html":
        bootstrap_script = (
            "<script>"
            "window.history.replaceState(window.history.state, '', '/dist-frontend/');"
            "</script>"
        )
        rewritten_html = rewritten_html.replace(
            '<script type="module"',
            f"{bootstrap_script}\n    <script type=\"module\"",
            1,
        )
    return rewritten_html


def _normalize_frontend_path(path: str) -> str:
    return path


def _is_frontend_navigation_request(method: str, path: str) -> bool:
    if method != "GET":
        return False
    if path.startswith("/api"):
        return False
    if path.startswith("/assets/"):
        return False
    return True


def _seed_push_snapshot() -> None:
    """Seedet Push-Snapshot in DB beim Start (für Render: eingebackene Daten als Fallback)."""
    if not os.path.exists(SNAPSHOT_PATH):
        return
    try:
        with open(SNAPSHOT_PATH) as f:
            snap = json.load(f)
        if isinstance(snap, list) and snap:
            n = push_db_upsert(snap)
            log.info("[Snapshot] %d Pushes in DB geseedet", n)
        elif isinstance(snap, dict) and snap.get("messages"):
            from app.routers.push import _push_sync_cache, _push_sync_lock
            with _push_sync_lock:
                _push_sync_cache["messages"] = snap.get("messages", [])
                _push_sync_cache["ts"] = snap.get("_generated", time.time())
            log.info("[Snapshot] %d Pushes aus Snapshot (Dict-Format) geladen",
                     len(_push_sync_cache["messages"]))
    except Exception as e:
        log.warning("[Snapshot] Fehler beim Laden: %s", e)


def _start_background_workers() -> None:
    """Startet alle Background-Worker-Threads (identisch zum Monolith)."""
    from app.research.worker import (
        _research_state,
        monitoring_tick,
        run_autonomous_analysis,
        update_residual_corrector,
    )
    from app.ml.gbrt import gbrt_train, gbrt_online_update, gbrt_check_drift
    from app.ml.lightgbm_model import ml_train_model, unified_train, train_stacking_model
    from app.tagesplan.builder import build_tagesplan

    # 5. Feed-Cache Worker
    def _feed_cache_worker():
        from app.config import COMPETITOR_FEEDS, INTERNATIONAL_FEEDS, SPORT_COMPETITOR_FEEDS, SPORT_EUROPA_FEEDS, SPORT_GLOBAL_FEEDS
        from app.research.worker import _feed_cache, _feed_cache_lock
        from app.routers.feed import _fetch_url, _parse_rss_items
        _FEED_CACHE_TTL = 300

        _FEED_TYPE_MAP = {
            "competitors":       COMPETITOR_FEEDS,
            "international":     INTERNATIONAL_FEEDS,
            "sport_competitors": SPORT_COMPETITOR_FEEDS,
            "sport_europa":      SPORT_EUROPA_FEEDS,
            "sport_global":      SPORT_GLOBAL_FEEDS,
        }

        log.info("[FeedCache] Background-Worker gestartet (alle %ds)", _FEED_CACHE_TTL)
        while True:
            for feed_type, feeds in _FEED_TYPE_MAP.items():
                parsed: dict = {}
                for name, url in feeds.items():
                    try:
                        xml_bytes = _fetch_url(url)
                        parsed[name] = _parse_rss_items(xml_bytes) if xml_bytes else []
                    except Exception as e:
                        log.debug("[FeedCache] %s/%s Fehler: %s", feed_type, name, e)
                        parsed[name] = []
                with _feed_cache_lock:
                    _feed_cache[feed_type]["data"] = parsed
                    _feed_cache[feed_type]["ts"] = time.time()
            log.debug("[FeedCache] Alle Feeds aktualisiert")
            time.sleep(_FEED_CACHE_TTL)

    threading.Thread(target=_feed_cache_worker, daemon=True).start()
    log.info("[FeedCache] Background-Worker gestartet")

    # 6. Research-Worker
    # app/research/worker.py nutzt den modularen app/-Pfad und läuft unabhängig vom Legacy-Referenzcode.
    # Auf Render: erstes Training bei Zyklus 15 (5 Min) statt Zyklus 1 — vermeidet RAM-Spike direkt beim Start.
    _is_render = os.environ.get("RENDER", "").lower() == "true"
    _first_train = 15 if _is_render else 1

    def _research_worker():
        time.sleep(2)
        try:
            from app.research.worker import _residual_corrector, _residual_corrector_lock
            update_residual_corrector()
            with _residual_corrector_lock:
                rc_bias = _residual_corrector["global_bias"]
                rc_n = _residual_corrector["n_samples"]
            log.info("[ResidualCorrector] Initial geladen: bias=%+.3f, n=%d", rc_bias, rc_n)
        except Exception as e:
            log.warning("[ResidualCorrector] Initial-Load fehlgeschlagen: %s", e)

        log.info("[Research] Autonomer Research-Worker gestartet (20s Intervall)")
        while True:
            try:
                run_autonomous_analysis()
                n = len(_research_state.get("push_data", []))
                if n > 0 and _research_state.get("_worker_first_log", True):
                    log.info("[Research] Erste Analyse fertig: %d Pushes, Accuracy %.1f%%",
                             n, _research_state.get("rolling_accuracy", 0))
                    _research_state["_worker_first_log"] = False
            except Exception as e:
                import traceback
                log.warning("[Research] Worker-Fehler: %s\n%s", e, traceback.format_exc())

            # Periodische Tasks
            try:
                counter = _research_state.get("_stacking_counter", 0) + 1
                _research_state["_stacking_counter"] = counter

                if counter % 30 == 0:
                    train_stacking_model(_research_state)
                if counter == _first_train or counter % 1080 == 0:
                    try:
                        ml_train_model()
                    except Exception as e:
                        log.warning("[ML] Training-Fehler im Research-Worker: %s", e)
                if counter == _first_train or counter % 360 == 0:
                    try:
                        gbrt_train()
                    except Exception as e:
                        log.warning("[GBRT] Training-Fehler: %s", e)
                if counter == 5 or counter % 1440 == 0:
                    try:
                        unified_train()
                    except Exception as e:
                        log.warning("[Unified] Training-Fehler: %s", e)
                if counter % 60 == 0 and counter > 3:
                    try:
                        gbrt_check_drift(_research_state)
                    except Exception as e:
                        log.warning("[GBRT] Drift-Check-Fehler: %s", e)
                if counter % 90 == 0 and counter > 5:
                    try:
                        gbrt_online_update()
                    except Exception as e:
                        log.warning("[GBRT] Online-Update-Fehler: %s", e)
                if counter == 5 or (counter % 60 == 0 and counter > 5):
                    try:
                        monitoring_tick()
                    except Exception as e:
                        log.warning("[Monitoring] Tick-Fehler: %s", e)
                if counter % 15 == 0:
                    try:
                        build_tagesplan(background=True)
                    except Exception as e:
                        log.debug("[Tagesplan] Background-Refresh: %s", e)
            except Exception as e:
                import traceback
                log.warning("[Research] Periodic-Task-Fehler: %s\n%s", e, traceback.format_exc())

            time.sleep(20)

    threading.Thread(target=_research_worker, daemon=True).start()
    log.info("[Research] Autonomer Research-Worker gestartet")

    # 7. Health-Checker
    def _health_checker():
        from app.research.worker import _health_state
        import urllib.request

        _health_state["uptime_start"] = time.time()
        time.sleep(5)
        log.info("[Health] Checker gestartet (60s Intervall)")
        while True:
            try:
                endpoints = {}
                for name, url in [("bild_sitemap", "https://www.bild.de/sitemap-news.xml")]:
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
                        import ssl as _ssl
                        try:
                            import certifi
                            ctx = _ssl.create_default_context(cafile=certifi.where())
                        except ImportError:
                            ctx = _ssl.create_default_context()
                        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                            status = resp.status
                        endpoints[name] = {"ok": status == 200, "status": status}
                        _health_state["checks_ok"] = _health_state.get("checks_ok", 0) + 1
                    except Exception as e:
                        endpoints[name] = {"ok": False, "error": str(e)[:100]}
                        _health_state["checks_fail"] = _health_state.get("checks_fail", 0) + 1
                _health_state["endpoints"] = endpoints
                _health_state["last_check"] = time.time()
                _health_state["status"] = "ok" if all(v.get("ok") for v in endpoints.values()) else "degraded"
            except Exception as e:
                log.warning("[Health] Checker-Fehler: %s", e)
            time.sleep(60)

    threading.Thread(target=_health_checker, daemon=True).start()
    log.info("[Health] Checker gestartet")

    # 8. Embedding-Modell im Hintergrund laden
    def _load_embedding_model():
        log.info("[Embeddings] Embedding-Modell-Load noch nicht migriert, übersprungen")

    threading.Thread(target=_load_embedding_model, daemon=True).start()
    log.info("[Embeddings] Modell wird im Hintergrund geladen")

    # 9. LLM-Backfill Thread
    def _llm_backfill():
        from app.config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            log.info("[LLM-Backfill] Kein OPENAI_API_KEY, überspringe Backfill")
            return
        log.info("[LLM-Backfill] LLM-Backfill noch nicht migriert, übersprungen")

    threading.Thread(target=_llm_backfill, daemon=True).start()
    log.info("[LLM-Backfill] Scoring-Thread gestartet")

    # 10. Preload Caches
    def _preload_caches():
        from app.routers.feed import _fetch_url
        from app.config import COMPETITOR_FEEDS, INTERNATIONAL_FEEDS
        # Kurz warten bis ML-Modelle von Disk geladen sind (~2s) + kleiner Puffer
        time.sleep(5)
        try:
            # background=True: blockiert bis der Plan fertig berechnet ist (kein fire-and-forget)
            build_tagesplan(background=True)
            log.info("[Preload] Tagesplan vorberechnet")
        except Exception as e:
            log.warning("[Preload] Tagesplan-Fehler: %s", e)
        try:
            for url in list(COMPETITOR_FEEDS.values()) + list(INTERNATIONAL_FEEDS.values()):
                _fetch_url(url)
            log.info("[Preload] Competitor + International Feeds gecacht")
        except Exception as e:
            log.warning("[Preload] Feed-Cache-Fehler: %s", e)

    threading.Thread(target=_preload_caches, daemon=True).start()
    log.info("[Preload] Caches werden im Hintergrund aufgebaut")

    # 11. Adobe Analytics Traffic Worker
    from app.routers.misc import _adobe_state
    if _adobe_state["enabled"]:
        def _adobe_traffic_worker():
            log.info("[Adobe] Adobe Traffic-Worker noch nicht migriert, übersprungen")

        threading.Thread(target=_adobe_traffic_worker, daemon=True).start()
        log.info("[Adobe] Traffic-Worker gestartet (30-Min-Intervall)")
    else:
        log.info("[Adobe] Deaktiviert (ADOBE_CLIENT_ID/SECRET nicht gesetzt)")

    # 12. Push-Auto-Fetch Worker
    from app.config import push_api_base_candidates
    import ssl as _ssl_mod2
    try:
        import certifi as _certifi2
        _auto_ssl = _ssl_mod2.create_default_context(cafile=_certifi2.where())
    except ImportError:
        _auto_ssl = _ssl_mod2.create_default_context()

    def _push_auto_fetch_worker():
        import urllib.request as _ur
        from app.routers.push import _push_sync_cache, _push_sync_lock
        time.sleep(5)
        log.info("[AutoFetch] Push-Daten-Worker gestartet (alle 120s)")
        while True:
            try:
                end_ts = int(time.time())
                start_ts = end_ts - 3 * 86400
                all_msgs = []
                channels = []
                last_error = None
                for base_url in push_api_base_candidates():
                    try:
                        url = (f"{base_url}/push/statistics/message"
                               f"?startDate={start_ts}&endDate={end_ts}&sourceTypes=EDITORIAL")
                        req = _ur.Request(url, headers={
                            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                            "Accept": "application/json",
                        })
                        with _ur.urlopen(req, timeout=20, context=_auto_ssl) as resp:
                            data = json.loads(resp.read())
                            all_msgs = data.get("messages", [])
                            next_params = data.get("next")
                            page = 0
                            while next_params and page < 10:
                                url2 = f"{base_url}/push/statistics/message?{next_params}"
                                req2 = _ur.Request(url2, headers={
                                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                                    "Accept": "application/json",
                                })
                                with _ur.urlopen(req2, timeout=15, context=_auto_ssl) as resp2:
                                    d2 = json.loads(resp2.read())
                                    all_msgs.extend(d2.get("messages", []))
                                    next_params = d2.get("next")
                                page += 1

                        try:
                            ch_url = f"{base_url}/push/statistics/message/channels?sourceTypes=EDITORIAL"
                            ch_req = _ur.Request(ch_url, headers={
                                "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-AutoFetch/1.0)",
                                "Accept": "application/json",
                            })
                            with _ur.urlopen(ch_req, timeout=10, context=_auto_ssl) as ch_resp:
                                channels = json.loads(ch_resp.read())
                        except Exception:
                            pass
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error and not all_msgs and not channels:
                    raise last_error
                with _push_sync_lock:
                    _push_sync_cache["messages"] = all_msgs
                    _push_sync_cache["channels"] = channels
                    _push_sync_cache["ts"] = time.time()
                log.info("[AutoFetch] OK: %d Push-Messages geladen", len(all_msgs))
            except Exception as e:
                log.warning("[AutoFetch] Fehler: %s", locals().get("last_error", e) or e)
            time.sleep(120)

    threading.Thread(target=_push_auto_fetch_worker, daemon=True).start()
    log.info("[AutoFetch] Push-Daten werden direkt von bildcms.de geholt (alle 120s)")

    # 13. Push-Sync Worker (zu Render)
    from app.config import RENDER_SYNC_URL, SYNC_SECRET

    def _push_sync_worker():
        import urllib.request as _ur2
        from app.routers.push import _push_sync_cache, _push_sync_lock
        time.sleep(15)
        if not RENDER_SYNC_URL:
            log.info("[Sync] RENDER_SYNC_URL nicht gesetzt, Sync deaktiviert")
            return
        log.info("[Sync] Worker gestartet, synce zu %s", RENDER_SYNC_URL)
        while True:
            try:
                with _push_sync_lock:
                    msgs = list(_push_sync_cache["messages"])
                    chs = list(_push_sync_cache["channels"])
                sync_payload = json.dumps({
                    "secret": SYNC_SECRET,
                    "messages": msgs,
                    "channels": chs,
                }).encode()
                req = _ur2.Request(
                    f"{RENDER_SYNC_URL}/api/pushes/sync",
                    data=sync_payload,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with _ur2.urlopen(req, timeout=15, context=_auto_ssl) as resp:
                    resp.read()
                log.info("[Sync] %d Messages zu Render gesendet", len(msgs))
            except Exception as e:
                log.warning("[Sync] Fehler: %s", e)
            time.sleep(60)

    if RENDER_SYNC_URL:
        threading.Thread(target=_push_sync_worker, daemon=True).start()
        log.info("[Sync] Worker gestartet")

    # 14. Auto-Suggestion Worker
    def _auto_sug_worker():
        time.sleep(30)
        log.info("[AutoSug] Worker gestartet (prüft alle 10 Min)")
        while True:
            try:
                from app.tagesplan.builder import _auto_save_suggestions
                _auto_save_suggestions()
            except Exception as e:
                log.warning("[AutoSug] Worker-Fehler: %s", e)
            time.sleep(600)

    threading.Thread(target=_auto_sug_worker, daemon=True).start()
    log.info("[AutoSug] Worker gestartet (stündlich)")

    # 15. Memory-Cleanup Worker (alle 2 Minuten)
    def _memory_cleanup_worker():
        time.sleep(60)
        log.info("[MemCleanup] Worker gestartet (alle 120s)")
        while True:
            try:
                from app.research.worker import trim_state_buffers
                freed = trim_state_buffers()
                if freed > 0:
                    log.info("[MemCleanup] %d Einträge bereinigt", freed)
            except Exception as e:
                log.warning("[MemCleanup] Fehler: %s", e)
            time.sleep(120)

    threading.Thread(target=_memory_cleanup_worker, daemon=True).start()
    log.info("[MemCleanup] Worker gestartet (alle 120s)")


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Push Balancer API",
    description=(
        "Push Balancer is an editorial decision-support API for push notification "
        "planning, research insights, and advisory model outputs."
    ),
    version="3.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    title_map = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        422: "Unprocessable Content",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }
    return _problem_response(
        request=request,
        status_code=exc.status_code,
        title=title_map.get(exc.status_code, "HTTP Error"),
        detail=str(exc.detail),
        problem_type=f"https://api.editorialsuite.io/problems/http-{exc.status_code}",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    detail = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', []))}: {error.get('msg', 'Invalid input')}"
        for error in exc.errors()
    )
    return _problem_response(
        request=request,
        status_code=422,
        title="Unprocessable Content",
        detail=detail or "Request validation failed.",
        problem_type="https://api.editorialsuite.io/problems/validation-error",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("[API] Unhandled error on %s", request.url.path, exc_info=exc)
    return _problem_response(
        request=request,
        status_code=500,
        title="Internal Server Error",
        detail="An unexpected server error occurred.",
        problem_type="https://api.editorialsuite.io/problems/internal-server-error",
    )

# ── Security Headers Middleware ────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    """Fügt Standard-Security-Headers zu allen Antworten hinzu."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    request_path = request.scope.get("path", request.url.path)
    return _apply_runtime_headers(request_path, response)


@app.middleware("http")
async def restrict_internal_access(request: Request, call_next) -> Response:
    """Beschränkt den Zugriff optional auf definierte interne Netze."""
    original_path = request.scope.get("path", request.url.path)
    normalized_path = _normalize_frontend_path(original_path)
    request.scope["path"] = normalized_path
    frontend_navigation = _is_frontend_navigation_request(request.method, normalized_path)

    if not INTERNAL_ACCESS_ENABLED or _path_is_exempt_from_internal_access(request.url.path):
        response = await call_next(request)
        if frontend_navigation and response.status_code == 404:
            index_path = _frontend_index_path()
            if os.path.isfile(index_path):
                return FileResponse(index_path, media_type="text/html")
        return response

    client_ip = _extract_client_ip(request)
    if _client_is_on_allowed_network(client_ip):
        response = await call_next(request)
        if frontend_navigation and response.status_code == 404:
            index_path = _frontend_index_path()
            if os.path.isfile(index_path):
                return FileResponse(index_path, media_type="text/html")
        return response

    log.warning(
        "[Access] Blockiere externen Zugriff auf %s von %s",
        request.url.path,
        client_ip or "<unknown>",
    )
    return _problem_response(
        request=request,
        status_code=404,
        title="Not Found",
        detail="The requested resource was not found.",
        problem_type="about:blank",
    )


# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Encoding", "Content-Length"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(forschung.router, tags=["Forschung"])
app.include_router(tagesplan.router, tags=["Tagesplan"])
app.include_router(ml.router, tags=["ML"])
app.include_router(gbrt.router, tags=["GBRT"])
app.include_router(push.router, tags=["Push"])
app.include_router(feed.router, tags=["Feed"])
app.include_router(misc.router, tags=["Misc"])


def _legacy_frontend_response() -> Response:
    legacy_path = _legacy_frontend_path()
    if not os.path.isfile(legacy_path):
        raise HTTPException(status_code=404, detail="Legacy frontend entrypoint not found.")

    response = FileResponse(legacy_path, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/push-balancer.html", include_in_schema=False)
async def frontend_compat_entrypoint() -> Response:
    """Liefert die historische interne Push-Balancer-Oberflaeche aus."""
    return _legacy_frontend_response()


@app.get("/dist-frontend", include_in_schema=False)
@app.get("/dist-frontend/", include_in_schema=False)
@app.get("/dist-frontend/{asset_path:path}", include_in_schema=False)
async def frontend_dist_entrypoint(asset_path: str = "") -> Response:
    """Kompatibilitaetspfad fuer historische interne Einstiege und alte Asset-Links."""
    normalized_asset_path = asset_path.lstrip("/")
    if normalized_asset_path:
        candidate_path = os.path.normpath(os.path.join(SERVE_DIR, normalized_asset_path))
        if candidate_path.startswith(os.path.normpath(SERVE_DIR) + os.sep) and os.path.isfile(candidate_path):
            return FileResponse(candidate_path)
    return _legacy_frontend_response()

# ── Statische Dateien (HTML, JS, CSS) ─────────────────────────────────────
# Wird nach den API-Routen gemountet, damit /api/* Priorität hat
if os.path.isdir(SERVE_DIR):
    app.mount("/", StaticFiles(directory=SERVE_DIR, html=True), name="static")


# ── Einstiegspunkt für direkten Start ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
