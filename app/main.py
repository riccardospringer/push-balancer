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
10. Push-Schedule + Feed-Cache vorberechnen
11. Adobe Analytics Traffic-Worker starten (wenn konfiguriert)
12. Push-Auto-Fetch-Worker starten
13. Push-Sync-Worker starten (wenn RENDER_SYNC_URL gesetzt)
14. Auto-Suggestion-Worker starten
"""
import ipaddress
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.config import (
    ALLOWED_ORIGINS,
    ARTICLE_PREDICTION_ENRICHMENT_ENABLED,
    BACKGROUND_AUTOMATIONS_ENABLED,
    HEALTH_ACTIVE_CHECKS_ENABLED,
    INTERNAL_ACCESS_ALLOWED_CIDRS,
    INTERNAL_ACCESS_ENABLED,
    INTERNAL_ACCESS_EXEMPT_PATHS,
    IS_RENDER,
    PORT,
    SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS,
    PUSH_TEAMS_ALERTS_ENABLED,
    PUSH_TEAMS_BACKGROUND_SENDER_ENABLED,
    PUSH_TEAMS_CHECK_INTERVAL_SECONDS,
    SERVE_DIR,
    SNAPSHOT_PATH,
    TRUSTED_PROXY_CIDRS,
)
from app.database import init_db, push_db_count, push_db_upsert
from app.ml.gbrt import gbrt_load_model
from app.routers import (
    alarm,
    consumer,
    feed,
    forschung,
    gbrt,
    health,
    misc,
    ml,
    power_automate,
    push,
    push_schedule,
    score_capture,
    score_api,
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
        _ctypes.CDLL(_omp_lib, mode=_ctypes.RTLD_GLOBAL)
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

    # ── 2b. Auto-Seed aus BILD-API wenn DB leer (Background, blockt Startup nicht) ──
    def _bg_auto_seed():
        try:
            from app.routers.push import auto_seed_db_if_empty
            auto_seed_db_if_empty()
        except Exception as e:
            log.warning("[AutoSeed-BG] Fehler: %s", e)
    threading.Thread(target=_bg_auto_seed, daemon=True, name="auto_seed_db").start()

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
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": problem_type,
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": _log_safe_request_path(str(request.url.path)),
        },
        media_type="application/problem+json",
        headers=headers,
    )


def _apply_cache_headers(path: str, response: Response) -> Response:
    if (
        path == "/api/score-capture"
        or path.startswith("/api/score-capture/")
        or path.startswith("/api/v1/power-automate/teams/")
    ):
        response.headers["Cache-Control"] = "no-store"
    if path.startswith("/api/v1/power-automate/teams/"):
        vary = {
            value.strip()
            for value in response.headers.get("Vary", "").split(",")
            if value.strip()
        }
        vary.add("X-Power-Automate-Key")
        response.headers["Vary"] = ", ".join(sorted(vary))
    return response


def _path_is_exempt_from_internal_access(path: str) -> bool:
    for exempt_path in INTERNAL_ACCESS_EXEMPT_PATHS:
        if path == exempt_path or path.startswith(f"{exempt_path}/"):
            return True
    return False


def _ip_is_in_cidrs(client_ip: str | None, cidrs: list[str]) -> bool:
    if not client_ip:
        return False
    try:
        parsed_ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if parsed_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            log.warning("[Access] Ungueltige CIDR-Konfiguration: %s", cidr)
    return False


def _request_comes_from_trusted_proxy(request: Request) -> bool:
    peer_ip = request.client.host if request.client and request.client.host else None
    return _ip_is_in_cidrs(peer_ip, TRUSTED_PROXY_CIDRS)


def _extract_client_ip(request: Request) -> str | None:
    """Return only an ingress-authenticated peer identity for CIDR decisions.

    Auf Render zaehlt ausschliesslich genau ein Cloudflare-Header; weitergereichte
    X-Forwarded-For-Werte werden dort nie als Identitaet akzeptiert. Ausserhalb
    von Render werden Forwarding-Header nur hinter einem vertrauenswuerdigen
    Proxy beruecksichtigt, sonst gilt der Socket-Peer.
    """
    if IS_RENDER:
        cloudflare_values = request.headers.getlist("cf-connecting-ip")
        if len(cloudflare_values) != 1:
            return None
        candidate = cloudflare_values[0].strip()
    elif _request_comes_from_trusted_proxy(request):
        candidate = ""
        for header_name in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
            header_value = request.headers.get(header_name, "").strip()
            if header_value:
                candidate = header_value
                break
        if not candidate:
            forwarded_for = request.headers.get("x-forwarded-for", "")
            candidate = forwarded_for.split(",", 1)[0].strip() if forwarded_for else ""
        if not candidate and request.client and request.client.host:
            candidate = request.client.host.strip()
    else:
        candidate = request.client.host.strip() if request.client else ""
    if not candidate:
        return None
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


_SCORE_CAPTURE_CMS_PATH_PREFIX = "/api/score-capture/by-cms-id/"
_SCORE_CAPTURE_BATCH_PATH = f"{_SCORE_CAPTURE_CMS_PATH_PREFIX}batch"
_SCORE_CAPTURE_HEALTH_PATH = "/api/score-capture/health"


_SCORE_API_PATH_RE = re.compile(r"^/api/v1/scores/[^/]+$")


def _log_safe_request_path(path: str) -> str:
    """Redact CMS identifiers before a request path enters application logs."""
    if path == _SCORE_CAPTURE_BATCH_PATH:
        return path
    if path.startswith(_SCORE_CAPTURE_CMS_PATH_PREFIX):
        return f"{_SCORE_CAPTURE_CMS_PATH_PREFIX}{{cms_id}}"
    if path != "/api/v1/scores/batch" and _SCORE_API_PATH_RE.match(path):
        return "/api/v1/scores/{cms_id}"
    return path


def _is_score_capture_source_path(path: str) -> bool:
    return (
        path == _SCORE_CAPTURE_HEALTH_PATH
        or path == _SCORE_CAPTURE_BATCH_PATH
        or path.startswith(_SCORE_CAPTURE_CMS_PATH_PREFIX)
    )


def _client_is_on_allowed_network(
    client_ip: str | None,
    allowed_cidrs: list[str] | None = None,
) -> bool:
    cidrs = INTERNAL_ACCESS_ALLOWED_CIDRS if allowed_cidrs is None else allowed_cidrs
    return _ip_is_in_cidrs(client_ip, cidrs)


def _is_approved_score_capture_consumer(
    method: str,
    path: str,
    client_ip: str | None,
) -> bool:
    """Allow the approved Next consumer only the minimal score-source methods."""
    if not _client_is_on_allowed_network(
        client_ip,
        SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS,
    ):
        return False
    if method == "POST":
        return path == "/api/score-capture/by-cms-id/batch"
    return method == "GET" and (
        path == "/api/score-capture/health"
        or bool(
            re.fullmatch(r"/api/score-capture/by-cms-id/[0-9a-fA-F]{24}", path)
        )
    )


def _frontend_index_path() -> str:
    return os.path.join(SERVE_DIR, "index.html")


def _frontend_assets_dir() -> str:
    return os.path.join(SERVE_DIR, "assets")


def _find_replacement_asset_name(asset_name: str) -> str | None:
    asset_dir = _frontend_assets_dir()
    if not os.path.isdir(asset_dir):
        return None

    candidate_path = os.path.join(asset_dir, asset_name)
    if os.path.isfile(candidate_path):
        return asset_name

    stem, ext = os.path.splitext(asset_name)
    if "-" not in stem:
        return None

    asset_prefix = stem.split("-", 1)[0]
    matching_candidates: list[tuple[float, str]] = []
    try:
        for entry in os.scandir(asset_dir):
            if not entry.is_file():
                continue
            entry_stem, entry_ext = os.path.splitext(entry.name)
            if entry_ext != ext or "-" not in entry_stem:
                continue
            if entry_stem.split("-", 1)[0] != asset_prefix:
                continue
            matching_candidates.append((entry.stat().st_mtime, entry.name))
    except OSError:
        return None

    if not matching_candidates:
        return None

    matching_candidates.sort(reverse=True)
    return matching_candidates[0][1]


def _repair_frontend_asset_references(html: str) -> str:
    asset_pattern = re.compile(r'((?:/dist-frontend/assets/|/assets/))([^"\'?#]+)')

    def _replace(match: re.Match[str]) -> str:
        prefix, asset_name = match.groups()
        replacement_name = _find_replacement_asset_name(asset_name)
        if not replacement_name:
            return match.group(0)
        return f"{prefix}{replacement_name}"

    return asset_pattern.sub(_replace, html)


def _load_frontend_html() -> str | None:
    index_path = _frontend_index_path()
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as index_file:
            return _repair_frontend_asset_references(index_file.read())
    except OSError:
        return None


def _prepare_frontend_html_for_request(html: str, request_path: str) -> str:
    if "/dist-frontend/assets/" not in html:
        return html

    rewritten_html = html.replace("/dist-frontend/assets/", "/assets/")
    history_target = _frontend_history_target(request_path)
    if history_target:
        bootstrap_script = (
            "<script>"
            f"window.history.replaceState(window.history.state, '', {json.dumps(history_target)});"
            "</script>"
        )
        rewritten_html = rewritten_html.replace(
            '<script type="module"',
            f"{bootstrap_script}\n    <script type=\"module\"",
            1,
        )
    return rewritten_html


def _frontend_history_target(request_path: str) -> str | None:
    if request_path in {"/", "/push-balancer.html"}:
        return "/dist-frontend/"
    if request_path in {"/dist-frontend", "/dist-frontend/"}:
        return None
    if request_path.startswith("/dist-frontend/") or request_path.startswith("/assets/"):
        return None
    if request_path.startswith("/"):
        return f"/dist-frontend{request_path}"
    return f"/dist-frontend/{request_path}"


def _normalize_frontend_path(path: str) -> str:
    return path


def _frontend_html_response(request_path: str) -> Response | None:
    html = _load_frontend_html()
    if html is None:
        return None

    prepared_html = _prepare_frontend_html_for_request(html, request_path)
    response = HTMLResponse(prepared_html)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


_ALWAYS_PUBLIC_PREFIXES = (
    "/assets/",
    "/api/health",
    "/favicon",
    "/robots.txt",
)
_ALWAYS_PUBLIC_PATHS = {
    "/api/feed",
    "/api/push-title/generate",
    "/api/push-title-generations",
}

def _is_always_public(path: str) -> bool:
    return path in _ALWAYS_PUBLIC_PATHS or any(path.startswith(p) for p in _ALWAYS_PUBLIC_PREFIXES)

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
                generated_at = snap.get("_generated")
                _push_sync_cache["ts"] = (
                    float(generated_at)
                    if isinstance(generated_at, (int, float))
                    else 0.0
                )
                _push_sync_cache["source"] = "seed"
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

    # 5. Feed-Cache Worker — läuft immer (günstig: nur RSS-HTTP-Fetches, kein ML)
    if True:
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
    else:
        log.info("[FeedCache] Deaktiviert, Endpunkte nutzen Live-Fallback nur bei Bedarf")

    # 6. Research-Worker
    # app/research/worker.py nutzt den modularen app/-Pfad und läuft unabhängig vom Legacy-Referenzcode.
    # Auf Render: erstes Training bei Zyklus 15 (5 Min) statt Zyklus 1 — vermeidet RAM-Spike direkt beim Start.
    _is_render = os.environ.get("RENDER", "").lower() == "true"
    _first_train = 15 if _is_render else 1

    if BACKGROUND_AUTOMATIONS_ENABLED:
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
                except Exception as e:
                    import traceback
                    log.warning("[Research] Periodic-Task-Fehler: %s\n%s", e, traceback.format_exc())

                time.sleep(20)

        threading.Thread(target=_research_worker, daemon=True).start()
        log.info("[Research] Autonomer Research-Worker gestartet")
    else:
        log.info("[Research] Deaktiviert, Analyse wird nur bei Bedarf berechnet")

    # 6a. Cold-Start: mitgeliefertes Seed-LightGBM-Modell laden, damit sofort
    # belastbare Pro-Artikel-Prognosen verfuegbar sind. Noetig, weil die Render-DB
    # anfangs zu wenig eigene Push-Historie zum Selbst-Trainieren hat. Ein spaeteres
    # erfolgreiches Training (Worker oder Standalone-Trainer) ueberschreibt es.
    if ARTICLE_PREDICTION_ENRICHMENT_ENABLED:
        try:
            from app.ml.lightgbm_model import load_seed_model

            if load_seed_model():
                log.info("[ML] Seed-Modell aktiv (Cold-Start-Prognose)")
            else:
                log.info("[ML] Kein Seed-Modell gefunden")
        except Exception as e:
            log.warning("[ML] Seed-Modell-Load fehlgeschlagen: %s", e)

    # 6b. Eigenstaendiger LightGBM-Trainer — sorgt fuer belastbare Pro-Artikel-OR-
    # Prognosen, auch wenn der schwere Research-Worker aus ist. Ohne trainiertes
    # Modell liefert predict_or nur den globalen Durchschnitt (Fallback), wodurch
    # die Teams-Empfehlungen nur die historische Slot-Prognose zeigen. Bewusst
    # leichtgewichtig: ein LightGBM-Fit aus der Push-Historie, nach Boot verzoegert
    # (RAM), danach periodischer Refresh.
    if ARTICLE_PREDICTION_ENRICHMENT_ENABLED and not BACKGROUND_AUTOMATIONS_ENABLED:
        def _standalone_ml_trainer():
            first_delay = int(os.environ.get("ML_STANDALONE_FIRST_TRAIN_DELAY_S", "150"))
            interval = max(1800, int(os.environ.get("ML_STANDALONE_RETRAIN_INTERVAL_S", "21600")))
            time.sleep(first_delay)
            log.info("[ML] Standalone-LightGBM-Trainer gestartet (Refresh alle %ds)", interval)
            while True:
                try:
                    ml_train_model()
                except Exception as e:
                    log.warning("[ML] Standalone-Training-Fehler: %s", e)
                time.sleep(interval)

        threading.Thread(target=_standalone_ml_trainer, daemon=True, name="standalone_ml_trainer").start()
        log.info("[ML] Standalone-LightGBM-Trainer aktiv (Research-Worker aus, Enrichment an)")

    # 7. Health-Checker
    if HEALTH_ACTIVE_CHECKS_ENABLED:
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
    else:
        log.info("[Health] Aktive Außenchecks deaktiviert")

    # 8. Embedding-Modell im Hintergrund laden
    def _load_embedding_model():
        log.info("[Embeddings] Embedding-Modell-Load noch nicht migriert, übersprungen")

    threading.Thread(target=_load_embedding_model, daemon=True).start()
    log.info("[Embeddings] Modell wird im Hintergrund geladen")

    # 9. LLM-Backfill Thread
    from app.config import OPENAI_BACKFILL_ENABLED, PAID_EXTERNAL_APIS_ENABLED
    if PAID_EXTERNAL_APIS_ENABLED and OPENAI_BACKFILL_ENABLED:
        def _llm_backfill():
            from app.config import OPENAI_API_KEY
            if not OPENAI_API_KEY:
                log.info("[LLM-Backfill] Kein OPENAI_API_KEY, überspringe Backfill")
                return
            log.info("[LLM-Backfill] LLM-Backfill noch nicht migriert, übersprungen")

        threading.Thread(target=_llm_backfill, daemon=True).start()
        log.info("[LLM-Backfill] Scoring-Thread gestartet")
    else:
        log.info("[LLM-Backfill] Deaktiviert (PAID_EXTERNAL_APIS_ENABLED/OPENAI_BACKFILL_ENABLED nicht gesetzt)")

    # 10. Preload Caches
    def _preload_caches():
        from app.routers.feed import _fetch_url
        from app.config import COMPETITOR_FEEDS, INTERNATIONAL_FEEDS
        time.sleep(5)
        if BACKGROUND_AUTOMATIONS_ENABLED:
            try:
                for url in list(COMPETITOR_FEEDS.values()) + list(INTERNATIONAL_FEEDS.values()):
                    _fetch_url(url)
                log.info("[Preload] Competitor + International Feeds gecacht")
            except Exception as e:
                log.warning("[Preload] Feed-Cache-Fehler: %s", e)

    threading.Thread(target=_preload_caches, daemon=True).start()
    log.info("[Preload] Feed-Cache-Vorberechnung gestartet")

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

    from app.config import PUSH_AUTO_FETCH_ENABLED
    if PUSH_AUTO_FETCH_ENABLED:
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
                        _push_sync_cache["source"] = "live"
                    log.info("[AutoFetch] OK: %d Push-Messages geladen", len(all_msgs))
                except Exception as e:
                    log.warning("[AutoFetch] Fehler: %s", locals().get("last_error", e) or e)
                time.sleep(120)

        threading.Thread(target=_push_auto_fetch_worker, daemon=True).start()
        log.info("[AutoFetch] Push-Daten werden direkt von bildcms.de geholt (alle 120s)")
    else:
        log.info("[AutoFetch] Deaktiviert, Live-Refresh nur auf Anfrage")

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
                    cache_source = str(_push_sync_cache.get("source") or "unknown")
                    cache_ts = float(_push_sync_cache.get("ts") or 0.0)
                sync_payload = json.dumps({
                    "secret": SYNC_SECRET,
                    "messages": msgs,
                    "channels": chs,
                    "source": cache_source,
                    "snapshotTs": cache_ts,
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

    from app.config import PUSH_RENDER_SYNC_ENABLED
    if PUSH_RENDER_SYNC_ENABLED and RENDER_SYNC_URL:
        threading.Thread(target=_push_sync_worker, daemon=True).start()
        log.info("[Sync] Worker gestartet")
    elif RENDER_SYNC_URL:
        log.info("[Sync] Deaktiviert, da PUSH_RENDER_SYNC_ENABLED=false")

    # 14. Auto-Suggestion Worker — entfallen (Tagesplan-Suggestions abgeschafft)

    # 15. Microsoft Teams recommendation alert worker
    if PUSH_TEAMS_ALERTS_ENABLED and PUSH_TEAMS_BACKGROUND_SENDER_ENABLED:
        def _teams_alert_worker():
            from app.notifications.teams import (
                TeamsAlertConfig,
                binding_slot_window_open,
                record_worker_start,
                run_teams_alert_cycle,
                seconds_to_defer_cycle_for_binding_slot,
                seconds_until_next_binding_slot,
            )

            record_worker_start()
            config = TeamsAlertConfig()
            try:
                from app.notifications.teams import (
                    log_channel_startup_selfcheck,
                )

                log_channel_startup_selfcheck(config)
            except Exception as exc:
                log.warning("[TeamsAlert] Startup-Selbstcheck uebersprungen: %s", exc)

            interval = max(30, int(PUSH_TEAMS_CHECK_INTERVAL_SECONDS or 120))
            time.sleep(45)
            log.info(
                "[TeamsAlert] Worker gestartet (Takt %ds, sekundengenau auf "
                "Raster-Slots ausgerichtet)",
                interval,
            )
            while True:
                # A normal collection cycle may take longer than a minute when
                # upstream APIs are slow.  Do not start it close enough to cross
                # a mandatory slot; wake just after the exact slot instead.
                try:
                    defer_seconds = seconds_to_defer_cycle_for_binding_slot(
                        time.time(),
                        config,
                        guard_seconds=max(180.0, float(interval) + 60.0),
                    )
                except Exception:
                    defer_seconds = 0.0
                if defer_seconds > 0:
                    time.sleep(defer_seconds)

                result = {}
                try:
                    result = run_teams_alert_cycle()
                    if result.get("sent"):
                        log.info("[TeamsAlert] Empfehlung an Teams gesendet: %s", result.get("candidateId"))
                    else:
                        diagnostics = result.get("diagnostics") or {}
                        log.info(
                            "[TeamsAlert] Kein Versand reason=%s teams_today=%s "
                            "due=%s target=%s score_eligible=%s blockers=%s",
                            result.get("reason") or "unknown",
                            diagnostics.get("teamsAlertsToday"),
                            diagnostics.get("dueOpportunityCount"),
                            diagnostics.get("targetCount"),
                            diagnostics.get("scoreEligibleCandidates"),
                            diagnostics.get("blockerCategories"),
                        )
                except Exception as e:
                    log.warning("[TeamsAlert] Worker-Fehler: %s", e)
                # A transient API or transport failure must not consume the
                # entire five-minute mandatory window.  Slot claims are durable
                # and release failed attempts, so an in-window retry is safe and
                # cannot duplicate a successful delivery.
                try:
                    retry_open_slot = (
                        not result.get("sent")
                        and binding_slot_window_open(time.time(), config)
                    )
                except Exception:
                    retry_open_slot = False
                if retry_open_slot:
                    time.sleep(5)
                    continue
                # Weckzeit exakt auf die naechste Raster-Entscheidung ausrichten:
                # der Zyklus feuert dann im Sekundenbereich nach der Slotzeit
                # statt bis zu einer Minute spaeter. Zwischen den Slots bleibt
                # der normale Takt fuer Live-Push-Spiegelung aktiv.
                try:
                    until_slot = seconds_until_next_binding_slot(time.time())
                except Exception:
                    until_slot = float(interval)
                time.sleep(max(1.0, min(float(interval), until_slot + 0.5)))

        def _teams_alert_supervisor():
            """Watchdog: ein gestorbener Worker-Thread darf den Kanal nicht toeten.

            Startet den Worker neu, wenn der Thread endet (unerwartete Exception
            ausserhalb der Zyklus-Schleife) oder laenger als die Stall-Frist
            keinen Herzschlag mehr geschrieben hat.
            """
            from app.notifications.teams import (
                TeamsAlertConfig,
                channel_health,
                record_worker_start,
            )

            worker = threading.Thread(
                target=_teams_alert_worker, daemon=True, name="teams_alert_worker"
            )
            worker.start()
            while True:
                time.sleep(60)
                try:
                    config = TeamsAlertConfig()
                    health = channel_health(config)
                    stalled = health.get("status") == "stalled"
                    if worker.is_alive() and not stalled:
                        continue
                    log.error(
                        "[TeamsAlert] Watchdog startet Worker neu (alive=%s status=%s reason=%s)",
                        worker.is_alive(),
                        health.get("status"),
                        health.get("reason"),
                    )
                    record_worker_start(restart=True)
                    worker = threading.Thread(
                        target=_teams_alert_worker,
                        daemon=True,
                        name="teams_alert_worker",
                    )
                    worker.start()
                except Exception as exc:
                    log.warning("[TeamsAlert] Watchdog-Fehler: %s", exc)

        threading.Thread(
            target=_teams_alert_supervisor, daemon=True, name="teams_alert_supervisor"
        ).start()
        log.info("[TeamsAlert] Aktiviert (mit Watchdog)")
    elif not PUSH_TEAMS_ALERTS_ENABLED:
        log.info("[TeamsAlert] Deaktiviert (PUSH_TEAMS_ALERTS_ENABLED=false)")
    else:
        log.info(
            "[TeamsAlert] Hintergrund-Sender deaktiviert; "
            "Power Automate besitzt Zeitplan und Transport"
        )

    # 16. Memory-Cleanup Worker (alle 2 Minuten)
    if BACKGROUND_AUTOMATIONS_ENABLED:
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
    else:
        log.info("[MemCleanup] Deaktiviert, da keine Autoloops laufen")

    # ── Push-Alarm Worker ──────────────────────────────────────────────────
    def _push_alarm_worker() -> None:
        from app.push_alarm.logic import check_push_alarm
        from app.routers.alarm import update_alarm_state
        from app.routers.feed import _fetch_url, _extract_sitemap_articles
        from app.push_schedule.weekly_baseline import baseline_for
        from app.config import BILD_SITEMAP, PUSH_DB_PATH
        import datetime

        log.info("[PushAlarm] Worker gestartet (alle 90s)")
        while True:
            try:
                xml = _fetch_url(BILD_SITEMAP)
                articles = _extract_sitemap_articles(xml, max_items=60) if xml else []

                # ML-Prediction anreichern wenn verfügbar
                try:
                    from app.ml.predict import predict_or
                    from app.research.worker import _research_state
                    now = datetime.datetime.now()
                    for a in articles:
                        result = predict_or(
                            {"title": a["title"], "headline": a["title"],
                             "cat": a["category"], "hour": now.hour,
                             "ts_num": int(now.timestamp()),
                             "is_eilmeldung": a["isEilmeldung"],
                             "link": a["url"], "channels": []},
                            _research_state,
                        )
                        por = (result or {}).get("predicted_or")
                        if por is not None:
                            a["predictedOR"] = round(float(por) / 100, 4)
                except Exception:
                    pass

                # Tagesplan-State aus PDF-Wochenmatrix (Golden-Hour = stars==3)
                _wd = datetime.datetime.now().weekday()
                _now_h = datetime.datetime.now().hour
                _now_cell = baseline_for(_now_h, _wd) or {}
                _golden_h = next(
                    (h for h in range(_now_h, 24)
                     if (baseline_for(h, _wd) or {}).get("stars") == 3),
                    None,
                )
                tp_state = {
                    "golden_hour": _golden_h,
                    "slots": [
                        {"hour": h, "expected_or": (baseline_for(h, _wd) or {}).get("avg_or")}
                        for h in range(6, 24)
                    ],
                }

                recommendation = check_push_alarm(articles, PUSH_DB_PATH, tp_state)
                update_alarm_state(recommendation)
            except Exception as exc:
                log.warning("[PushAlarm] Worker-Fehler: %s", exc)
            time.sleep(90)

    threading.Thread(target=_push_alarm_worker, daemon=True).start()


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
        429: "Too Many Requests",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }
    return _problem_response(
        request=request,
        status_code=exc.status_code,
        title=title_map.get(exc.status_code, "HTTP Error"),
        detail=str(exc.detail),
        problem_type=f"https://api.editorialsuite.io/problems/http-{exc.status_code}",
        headers=exc.headers,
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
    request_path = request.scope.get("path", request.url.path)
    safe_path = _log_safe_request_path(request_path)
    if _is_score_capture_source_path(request_path):
        log.error(
            "[API] Unhandled %s on %s",
            type(exc).__name__,
            safe_path,
        )
    else:
        log.exception("[API] Unhandled error on %s", safe_path, exc_info=exc)
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
    if _SCORE_API_PATH_RE.fullmatch(request.url.path):
        response.headers["Cache-Control"] = "no-store"
        vary_values = {
            value.strip()
            for value in response.headers.get("Vary", "").split(",")
            if value.strip()
        }
        vary_values.add("X-Score-Key")
        response.headers["Vary"] = ", ".join(sorted(vary_values))
    request_path = request.scope.get("path", request.url.path)
    return _apply_cache_headers(request_path, response)


@app.middleware("http")
async def restrict_internal_access(request: Request, call_next) -> Response:
    """Beschränkt den Zugriff optional auf definierte interne Netze."""
    original_path = request.scope.get("path", request.url.path)
    normalized_path = _normalize_frontend_path(original_path)
    request.scope["path"] = normalized_path
    frontend_navigation = _is_frontend_navigation_request(request.method, normalized_path)

    if not INTERNAL_ACCESS_ENABLED or _path_is_exempt_from_internal_access(request.url.path) or _is_always_public(normalized_path) or frontend_navigation:
        response = await call_next(request)
        if frontend_navigation and response.status_code == 404:
            frontend_response = _frontend_html_response(normalized_path)
            if frontend_response is not None:
                return frontend_response
        return response

    client_ip = _extract_client_ip(request)
    if _is_approved_score_capture_consumer(request.method, normalized_path, client_ip):
        return await call_next(request)
    if _client_is_on_allowed_network(client_ip):
        response = await call_next(request)
        if frontend_navigation and response.status_code == 404:
            frontend_response = _frontend_html_response(normalized_path)
            if frontend_response is not None:
                return frontend_response
        return response

    safe_path = _log_safe_request_path(normalized_path)
    if _is_score_capture_source_path(normalized_path):
        log.warning("[Access] Blockiere Score-Source-Zugriff auf %s", safe_path)
    else:
        log.warning(
            "[Access] Blockiere externen Zugriff auf %s von %s",
            safe_path,
            client_ip or "<unknown>",
        )
    response = _problem_response(
        request=request,
        status_code=404,
        title="Not Found",
        detail="The requested resource was not found.",
        problem_type="about:blank",
    )
    return _apply_cache_headers(normalized_path, response)


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
app.include_router(alarm.router, tags=["PushAlarm"])
app.include_router(score_capture.router, tags=["ScoreCapture"])
app.include_router(health.router, tags=["Health"])
app.include_router(forschung.router, tags=["Forschung"])
app.include_router(push_schedule.router, tags=["PushSchedule"])
app.include_router(ml.router, tags=["ML"])
app.include_router(gbrt.router, tags=["GBRT"])
app.include_router(push.router, tags=["Push"])
app.include_router(feed.router, tags=["Feed"])
app.include_router(consumer.router, tags=["Consumer"])
app.include_router(power_automate.router, tags=["PowerAutomate"])
app.include_router(score_api.router, tags=["Score"])
app.include_router(tagesplan.router, tags=["Tagesplan"])
app.include_router(misc.router, tags=["Misc"])

def _frontend_file_response(relative_path: str) -> FileResponse | None:
    normalized_relative_path = relative_path.lstrip("/")
    candidate_path = os.path.normpath(os.path.join(SERVE_DIR, normalized_relative_path))
    serve_root = os.path.normpath(SERVE_DIR)
    if not (
        candidate_path == serve_root or candidate_path.startswith(serve_root + os.sep)
    ):
        return None
    if not os.path.isfile(candidate_path):
        return None
    return FileResponse(candidate_path)


@app.get("/", include_in_schema=False)
async def frontend_root_entrypoint() -> Response:
    frontend_response = _frontend_html_response("/")
    if frontend_response is not None:
        return frontend_response
    raise HTTPException(status_code=404, detail="Frontend entrypoint not found.")


def _legacy_capture_frontend_path() -> str:
    # push-balancer.html liegt eine Ebene ueber app/ im Projekt-Root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "push-balancer.html")
    if os.path.isfile(p):
        return p
    # Fallback auf legacy_push_balancer.html im app/-Verzeichnis
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_push_balancer.html")


async def frontend_compat_entrypoint() -> Response:
    """Liefert die Legacy-Capture-UI: einzige Quelle der Browser-Score-Erfassung.

    Die Score-Capture-Pipeline (POST /api/score-capture) laeuft im CvD-Browser
    ueber dieses klassische Frontend; die React-SPA hat keinen Capture-Pfad.
    """
    legacy_path = _legacy_capture_frontend_path()
    if not os.path.isfile(legacy_path):
        frontend_response = _frontend_html_response("/push-balancer.html")
        if frontend_response is not None:
            return frontend_response
        raise HTTPException(status_code=404, detail="Frontend entrypoint not found.")
    response = FileResponse(legacy_path, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/dist-frontend", include_in_schema=False)
@app.get("/dist-frontend/", include_in_schema=False)
@app.get("/dist-frontend/{asset_path:path}", include_in_schema=False)
async def frontend_dist_entrypoint(asset_path: str = "") -> Response:
    """Kompatibilitaetspfad fuer die gebaute SPA und alte Asset-Links."""
    normalized_asset_path = asset_path.lstrip("/")
    if normalized_asset_path:
        file_response = _frontend_file_response(normalized_asset_path)
        if file_response is not None:
            return file_response

        if normalized_asset_path.startswith("assets/"):
            replacement_name = _find_replacement_asset_name(os.path.basename(normalized_asset_path))
            if replacement_name:
                aliased_response = _frontend_file_response(f"assets/{replacement_name}")
                if aliased_response is not None:
                    return aliased_response
            raise HTTPException(status_code=404, detail="Frontend asset not found.")

    frontend_response = _frontend_html_response("/dist-frontend/")
    if frontend_response is not None:
        return frontend_response
    raise HTTPException(status_code=404, detail="Frontend entrypoint not found.")


app.add_api_route("/push-balancer.html", frontend_compat_entrypoint, methods=["GET"], include_in_schema=False)


@app.get("/assets/{asset_path:path}", include_in_schema=False)
async def serve_frontend_asset(asset_path: str) -> Response:
    """Statische Assets — nur noch fuer Legacy-Push-Balancer.html (falls referenziert)."""
    assets_dir = os.path.normpath(_frontend_assets_dir())
    candidate = os.path.normpath(os.path.join(assets_dir, asset_path))
    if candidate.startswith(assets_dir + os.sep) and os.path.isfile(candidate):
        return FileResponse(candidate)
    raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")


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
