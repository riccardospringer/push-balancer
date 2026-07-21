"""app/routers/push.py — Push-API-Proxy und Sync-Endpunkte.

GET  /api/push/{path:path}     — Proxy zur BILD Push-API
POST /api/pushes/sync          — Empfängt Push-Daten von lokalem Server (Relay)
POST /api/predictions/feedback — Prediction-Feedback für ML-Training
"""
import json
import logging
import datetime as dt
import sqlite3
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.config import PUSH_DB_PATH, PUSH_LIVE_FETCH_ENABLED, SYNC_SECRET, push_api_base_candidates
from app.database import push_db_load_all, push_db_log_prediction, push_db_count, push_db_upsert

log = logging.getLogger("push-balancer")
router = APIRouter()

# ── Push-Sync Cache (module-level, thread-safe) ───────────────────────────
_push_sync_cache: dict = {"messages": [], "ts": 0, "channels": []}
_push_sync_lock = threading.Lock()

# ── SSL Context ────────────────────────────────────────────────────────────
import ssl as _ssl_mod
try:
    import certifi as _certifi
    _SSL_CTX = _ssl_mod.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CTX = _ssl_mod.create_default_context()


class PredictionFeedbackRequest(BaseModel):
    # Neues camelCase-Format
    pushId: str = ""
    actualOr: float = 0.0
    title: str | None = None
    predictedOr: float | None = None
    # Altes snake_case-Format (HTML-Frontend)
    push_id: str | None = None
    actual_or: float | None = None
    predicted_or: float | None = None
    push_cat: str | None = None
    push_hour: int | None = None


def _lookup_push_row(push_id: str) -> dict | None:
    """Lädt Push-Daten aus pushes-Tabelle für predicted_or-Berechnung."""
    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=3)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT title, cat, hour, ts_num, is_eilmeldung FROM pushes WHERE message_id = ? LIMIT 1",
            (push_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def _compute_predicted_or(push_id: str, title: str | None, cat: str | None, hour: int | None) -> float:
    """Berechnet predicted_or für einen Push zum Feedback-Zeitpunkt."""
    try:
        push_row = _lookup_push_row(push_id)
        push_data = {
            "title": (push_row or {}).get("title") or title or "",
            "cat": (push_row or {}).get("cat") or cat or "News",
            "hour": (push_row or {}).get("hour") or hour or dt.datetime.now().hour,
            "ts_num": (push_row or {}).get("ts_num") or int(time.time()),
            "is_eilmeldung": bool((push_row or {}).get("is_eilmeldung", False)),
            "channels": [],
        }
        from app.ml.predict import predict_or
        from app.research.worker import _research_state
        result = predict_or(push_data, _research_state)
        return float((result or {}).get("predicted_or") or 0.0)
    except Exception:
        return 0.0


class PushSyncRequest(BaseModel):
    secret: str = ""   # default "" → leerer Body ergibt 403 statt 422
    messages: list = []
    channels: list = []


def _iso_from_ts(ts_value: int | str | float) -> str:
    ts = int(float(ts_value))
    return dt.datetime.fromtimestamp(ts).isoformat()


def _ratio(value: float | int | None) -> float:
    if value is None:
        return 0.0
    numeric = float(value)
    return numeric / 100 if numeric > 1 else numeric


_CATEGORY_SEGMENTS = (
    "regional", "sport", "politik", "unterhaltung", "ratgeber",
    "geld", "auto", "digital", "leben", "spiele", "reise",
    "news", "bild-plus", "lifestyle", "ausland", "video",
)


def _extract_category_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        path = urllib.parse.urlparse(url).path.lower()
    except Exception:
        return ""
    for seg in _CATEGORY_SEGMENTS:
        if f"/{seg}/" in path or path.startswith(f"/{seg}"):
            return seg
    return ""


def _parse_bild_messages(raw_messages: list) -> list:
    """Wandelt BILD Push-API Rohformat in das push_db_upsert-Schema."""
    parsed: list[dict] = []
    for m in raw_messages or []:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("messageId")
        try:
            ts = int(m.get("sendDate") or 0)
        except (TypeError, ValueError):
            ts = 0
        if not mid or ts <= 0:
            continue

        tl = m.get("targetList") or []
        channels = [t.get("channel", "") for t in tl if isinstance(t, dict) and t.get("channel")]
        app_set: set = set()
        for t in tl:
            if isinstance(t, dict):
                for a in (t.get("appList") or []):
                    if a:
                        app_set.add(a)
        app_list = sorted(app_set)
        channel = channels[0] if channels else ""
        is_eil = any("eil" in (c or "").lower() for c in channels)

        try:
            or_raw = float(m.get("openingRate") or 0)
        except (TypeError, ValueError):
            or_raw = 0.0
        or_val = or_raw / 100 if or_raw > 1 else or_raw

        title = m.get("kickerAndHeadline") or m.get("headline") or ""
        link = m.get("url") or m.get("urlId") or ""
        try:
            hour = dt.datetime.fromtimestamp(ts).hour
        except (OSError, ValueError, OverflowError):
            hour = -1

        parsed.append({
            "message_id": str(mid),
            "ts_num": ts,
            "or": or_val,
            "title": title,
            "headline": m.get("headline", "") or "",
            "kicker": m.get("kicker", "") or "",
            "cat": _extract_category_from_url(link) or "news",
            "link": link,
            "type": (m.get("sourceType") or "EDITORIAL").lower(),
            "hour": hour,
            "title_len": len(title),
            "opened": int(m.get("openedCount") or 0),
            "received": int(m.get("receivedCount") or 0),
            "channel": channel,
            "channels": channels,
            "is_eilmeldung": is_eil,
            "target_stats": m.get("targetStatistics") or {},
            "app_list": app_list,
            "n_apps": len(app_list),
            "total_recipients": int(m.get("recipientCount") or 0),
        })
    return parsed


def _fetch_live_push_snapshot(force: bool = False) -> tuple[list, list]:
    if not force and not PUSH_LIVE_FETCH_ENABLED:
        raise RuntimeError("Live push fetch is disabled in economy mode")

    end_ts = int(time.time())
    start_ts = end_ts - 3 * 86400
    all_msgs: list = []
    channels: list = []
    last_error: Exception | None = None

    for base_url in push_api_base_candidates():
        try:
            url = (
                f"{base_url}/push/statistics/message"
                f"?startDate={start_ts}&endDate={end_ts}&sourceTypes=EDITORIAL"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Refresh/1.0)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
                all_msgs = data.get("messages", [])
                next_params = data.get("next")
                page = 0
                while next_params and page < 10:
                    url2 = f"{base_url}/push/statistics/message?{next_params}"
                    req2 = urllib.request.Request(
                        url2,
                        headers={
                            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Refresh/1.0)",
                            "Accept": "application/json",
                        },
                    )
                    with urllib.request.urlopen(req2, timeout=15, context=_SSL_CTX) as resp2:
                        d2 = json.loads(resp2.read())
                        all_msgs.extend(d2.get("messages", []))
                        next_params = d2.get("next")
                    page += 1

            try:
                ch_url = f"{base_url}/push/statistics/message/channels?sourceTypes=EDITORIAL"
                ch_req = urllib.request.Request(
                    ch_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; PushBalancer-Refresh/1.0)",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(ch_req, timeout=10, context=_SSL_CTX) as ch_resp:
                    channels = json.loads(ch_resp.read())
            except Exception:
                pass

            return all_msgs, channels
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error
    return all_msgs, channels


def _load_prediction_map(push_ids: list[str]) -> dict[str, float]:
    if not push_ids:
        return {}

    placeholders = ",".join("?" for _ in push_ids)
    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT push_id, predicted_or FROM prediction_log WHERE push_id IN ({placeholders})",
            push_ids,
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.warning("[pushes] could not load prediction deltas: %s", exc)
        return {}

    return {
        str(row["push_id"]): _ratio(row["predicted_or"])
        for row in rows
        if row["predicted_or"] is not None
    }


def _build_pushes_response(
    limit: int,
    days: int = 30,
    sort: str = "sentAt",
    category: str = "",
) -> dict:
    try:
        rows = push_db_load_all(
            max_days=max(1, min(days, 365)),
            max_rows=max(100, min(limit * 8, 2000)),
        )
    except Exception as exc:
        log.warning("[pushes] could not load push history: %s", exc)
        rows = []

    category_filter = category.strip().lower()
    if category_filter:
        rows = [row for row in rows if str(row.get("cat", "")).strip().lower() == category_filter]

    prediction_map = _load_prediction_map(
        [str(row.get("message_id", "")) for row in rows if row.get("message_id")]
    )

    pushes = []
    today_key = dt.datetime.now().strftime("%Y-%m-%d")
    today_rows = []

    for row in rows:
        sent_at = _iso_from_ts(row.get("ts_num", row.get("ts", 0)))
        if sent_at.startswith(today_key):
            today_rows.append(row)
        predicted_or = prediction_map.get(str(row.get("message_id", "")))
        open_rate = round(_ratio(row.get("or")), 4)
        # Nur echten Score anzeigen (push_score_real=1 = vom Tool erfasst)
        push_score = row.get("push_score") if row.get("push_score_real") else 0
        if not push_score:
            link = row.get("link", "")
            if link:
                from app.routers.score_capture import get_score_for_url
                push_score = get_score_for_url(link) or 0
        pushes.append(
            {
                "id": row.get("message_id", ""),
                "title": row.get("title") or row.get("headline") or "",
                "channel": row.get("channel") or (row.get("channels") or ["main"])[0],
                "category": row.get("cat") or "",
                "type": row.get("type") or "editorial",
                "sentAt": sent_at,
                "recipients": row.get("total_recipients") or row.get("received") or 0,
                "opened": row.get("opened") or 0,
                "openRate": open_rate,
                "predictedOR": round(predicted_or, 4) if predicted_or is not None else None,
                "performanceDelta": (
                    round(open_rate - predicted_or, 4) if predicted_or is not None else None
                ),
                "url": row.get("link") or None,
                "pushScore": push_score,
                "frozenXor": round(float(row.get("frozen_xor") or 0), 2) or None,
            }
        )

    total = len(pushes)

    if sort == "openRate":
        pushes.sort(key=lambda push: (push["openRate"], push["sentAt"]), reverse=True)
    elif sort == "performanceDelta":
        pushes.sort(
            key=lambda push: (push["performanceDelta"] if push["performanceDelta"] is not None else -1, push["sentAt"]),
            reverse=True,
        )
    elif sort == "recipients":
        pushes.sort(key=lambda push: (push["recipients"], push["sentAt"]), reverse=True)
    else:
        pushes.sort(key=lambda push: push["sentAt"], reverse=True)

    pushes = pushes[:limit]
    channels = sorted({push["channel"] for push in pushes if push["channel"]})
    today_rates = [_ratio(row.get("or")) for row in today_rows if row.get("or") is not None]
    today_recipients = [
        int(row.get("total_recipients") or row.get("received") or 0) for row in today_rows
    ]

    return {
        "pushes": pushes,
        "channels": channels,
        "today": {
            "count": len(today_rows),
            "avgOR": round(sum(today_rates) / len(today_rates), 4) if today_rates else 0.0,
            "topOR": round(max(today_rates), 4) if today_rates else 0.0,
            "recipients": sum(today_recipients),
        },
        "total": total,
        "offset": 0,
        "limit": limit,
    }


def _build_refresh_response() -> dict:
    try:
        messages, channels = _fetch_live_push_snapshot(force=True)
        with _push_sync_lock:
            _push_sync_cache["messages"] = messages
            _push_sync_cache["channels"] = channels
            _push_sync_cache["ts"] = time.time()
        try:
            n_written = push_db_upsert(_parse_bild_messages(messages))
        except Exception as upsert_exc:
            log.warning("[refresh] DB-Upsert fehlgeschlagen: %s", upsert_exc)
            n_written = 0
        return {
            "ok": True,
            "synced": len(messages),
            "channels": len(channels),
            "db_written": n_written,
            "source": "live",
            "history_authoritative": True,
            "snapshot_age_seconds": 0,
        }
    except Exception as exc:
        log.warning("[pushes] live refresh failed: %s", exc)
        # Fallback: aus dem Sync-Cache parsen + in DB schreiben (lokales Relay hat ihn ggf. befüllt)
        with _push_sync_lock:
            cache_msgs = list(_push_sync_cache.get("messages") or [])
            cache_ts = float(_push_sync_cache.get("ts") or 0.0)
        if cache_msgs:
            try:
                n_written = push_db_upsert(_parse_bild_messages(cache_msgs))
                cache_age = max(0.0, time.time() - cache_ts) if cache_ts > 0 else None
                history_authoritative = bool(
                    cache_age is not None and cache_age <= 300.0
                )
                return {
                    "ok": True,
                    "synced": len(cache_msgs),
                    "channels": 0,
                    "db_written": n_written,
                    "source": "cache->db",
                    "history_authoritative": history_authoritative,
                    "snapshot_age_seconds": (
                        round(cache_age, 1) if cache_age is not None else None
                    ),
                }
            except Exception as cache_exc:
                log.warning("[pushes] cache→db Fallback fehlgeschlagen: %s", cache_exc)
        try:
            rows = push_db_load_all(max_rows=50)
        except Exception as db_exc:
            log.warning("[pushes] refresh fallback without DB: %s", db_exc)
            rows = []
        return {
            "ok": True,
            "synced": len(rows),
            "channels": 0,
            "source": "db-fallback",
            "history_authoritative": False,
            "snapshot_age_seconds": None,
        }


def auto_seed_db_if_empty() -> int:
    """Seedet die DB einmalig aus BILD-API, wenn sie leer ist.

    Wird beim App-Start in einem Background-Thread aufgerufen, damit der Server
    nicht blockiert wird. Bypassed `PUSH_LIVE_FETCH_ENABLED` (force=True), weil
    eine leere DB den gesamten Tagesplan + CSV-Export blockiert.
    """
    try:
        if push_db_count() > 0:
            return 0
    except Exception as exc:
        log.warning("[AutoSeed] push_db_count fehlgeschlagen: %s", exc)
        return 0
    try:
        messages, channels = _fetch_live_push_snapshot(force=True)
        with _push_sync_lock:
            _push_sync_cache["messages"] = messages
            _push_sync_cache["channels"] = channels
            _push_sync_cache["ts"] = time.time()
        n = push_db_upsert(_parse_bild_messages(messages))
        log.info("[AutoSeed] %d Pushes von BILD-API in leere DB geseedet", n)
        return n
    except Exception as exc:
        with _push_sync_lock:
            cache_msgs = list(_push_sync_cache.get("messages") or [])
        if cache_msgs:
            try:
                n = push_db_upsert(_parse_bild_messages(cache_msgs))
                log.info(
                    "[AutoSeed-Cache] %d Pushes aus Sync-Cache geseedet (Live-Fetch failed: %s)",
                    n, exc,
                )
                return n
            except Exception as cache_exc:
                log.warning("[AutoSeed-Cache] Upsert fehlgeschlagen: %s", cache_exc)
        log.warning("[AutoSeed] Live + Cache fehlgeschlagen, DB bleibt leer: %s", exc)
        return 0


@router.get("/api/push/{path:path}")
async def proxy_push_api(path: str, request: Request) -> Response:
    """Proxy zur BILD Push-API mit Sync-Cache-Fallback.

    1. Versuch: Direkt zur BILD Push-API
    2. Fallback: Sync-Cache (von lokalem Server befüllt)
    3. Kein Cache: Leeres Ergebnis mit Fehlermeldung
    """
    full_path = f"/push/{path}"
    query = str(request.url.query)
    last_error: Exception | None = None
    if PUSH_LIVE_FETCH_ENABLED:
        for base_url in push_api_base_candidates():
            url = f"{base_url}{full_path}"
            if query:
                url = f"{url}?{query}"

            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                    data = resp.read()
                return Response(
                    content=data,
                    media_type="application/json; charset=utf-8",
                )
            except Exception as exc:
                last_error = exc
        log.info("[Proxy] Push-API direkt nicht erreichbar, prüfe Sync-Cache: %s", last_error)
    else:
        log.info("[Proxy] Live Push-API deaktiviert, nutze Cache/Fallback")

    # 2. Sync-Cache
    with _push_sync_lock:
        cache_age = time.time() - _push_sync_cache["ts"]
        if _push_sync_cache["ts"] > 0 and cache_age < 86400:
            if "channels" in path:
                payload = json.dumps(_push_sync_cache.get("channels", [])).encode()
            else:
                payload = json.dumps({
                    "messages": _push_sync_cache["messages"],
                    "next": None,
                    "_synced": True,
                    "_age_s": int(cache_age),
                }).encode()
            return Response(
                content=payload,
                media_type="application/json; charset=utf-8",
            )

    # 3. Kein Cache
    fallback = json.dumps({
        "messages": [],
        "_fallback": True,
        "_reason": "Push-API nicht erreichbar und kein Sync-Cache vorhanden. "
                   "Lokaler Server muss laufen um Daten zu synchronisieren.",
    }).encode()
    return Response(content=fallback, media_type="application/json; charset=utf-8")


@router.get("/api/pushes")
def get_pushes(
    limit: int = 100,
    days: int = 30,
    sort: str = "sentAt",
    category: str = "",
) -> JSONResponse:
    """Return recent push history as a stable JSON collection for the frontend."""
    return JSONResponse(content=_build_pushes_response(limit, days, sort, category))


@router.get("/api/pushes/export.csv")
def export_pushes_csv(
    days: int = 90,
    limit: int = 10000,
    sort: str = "sentAt",
    category: str = "",
) -> Response:
    """CSV-Download des Push-History-Datasets.

    Browser-freundlich: einfach URL aufrufen → Datei wird heruntergeladen,
    Excel/Numbers oeffnet das Format direkt. Kein Tool, kein Auth, kein Skript.
    Default: 90 Tage, bis zu 10000 Pushes.
    """
    import csv
    import io

    payload = _build_pushes_response(limit, days, sort, category)
    pushes = payload.get("pushes", [])

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "id", "sent_at", "title", "channel", "category", "type",
        "recipients", "opened", "open_rate_pct", "predicted_or_pct",
        "performance_delta_pct", "push_score", "frozen_xor", "url",
    ])
    for p in pushes:
        writer.writerow([
            p.get("id", ""),
            p.get("sentAt", ""),
            (p.get("title") or "").replace("\n", " ").replace("\r", " "),
            p.get("channel", ""),
            p.get("category", ""),
            p.get("type", ""),
            p.get("recipients") or 0,
            p.get("opened") or 0,
            f"{(p.get('openRate') or 0) * 100:.2f}",
            f"{(p.get('predictedOR') or 0) * 100:.2f}" if p.get("predictedOR") is not None else "",
            f"{(p.get('performanceDelta') or 0) * 100:.2f}" if p.get("performanceDelta") is not None else "",
            p.get("pushScore") or 0,
            p.get("frozenXor") or "",
            p.get("url") or "",
        ])
    today_str = dt.datetime.now().strftime("%Y-%m-%d")
    filename = f"push-balancer-export_{today_str}_{days}d.csv"
    # UTF-8 BOM, damit Excel Umlaute korrekt erkennt
    body = "\ufeff" + buf.getvalue()
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/pushes/sync")
def post_push_sync(body: PushSyncRequest) -> JSONResponse:
    """Empfängt Push-Daten von lokalem Server (Relay für Render).

    Authentifizierung via PUSH_SYNC_SECRET.
    """
    if not SYNC_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Push sync is disabled because PUSH_SYNC_SECRET is not configured.",
        )
    if body.secret != SYNC_SECRET:
        raise HTTPException(status_code=403, detail="Invalid sync secret")

    with _push_sync_lock:
        _push_sync_cache["messages"] = body.messages
        _push_sync_cache["channels"] = body.channels
        _push_sync_cache["ts"] = time.time()

    log.info("[Sync] Empfangen: %d Messages, %d Channels",
             len(body.messages), len(body.channels))
    return JSONResponse(content={"ok": True, "received": len(body.messages)})


@router.post("/api/pushes/refresh")
def post_push_refresh() -> JSONResponse:
    """Frontend-safe refresh endpoint for the live pushes view."""
    return JSONResponse(content=_build_refresh_response())


@router.post("/api/push-refresh-jobs")
def create_push_refresh_job() -> JSONResponse:
    """Resource-style alias for triggering a live push refresh job."""
    return JSONResponse(content=_build_refresh_response())


def _handle_feedback(body: PredictionFeedbackRequest) -> JSONResponse:
    """Gemeinsame Logik für beide Feedback-Endpunkte."""
    # Normalisiere push_id: camelCase oder snake_case
    push_id = body.pushId or body.push_id or ""
    actual_or = body.actualOr if body.actualOr else (body.actual_or or 0.0)
    if not push_id or actual_or <= 0:
        return JSONResponse(content={"ok": False, "error": "missing push_id or actual_or"})

    # predicted_or: vom Client mitgesendet (altes Frontend) oder frisch berechnen
    predicted_or = (
        body.predicted_or
        or body.predictedOr
        or 0.0
    )
    if predicted_or <= 0:
        predicted_or = _compute_predicted_or(
            push_id,
            title=body.title,
            cat=body.push_cat,
            hour=body.push_hour,
        )

    try:
        push_db_log_prediction(
            push_id=push_id,
            predicted_or=predicted_or,
            actual_or=actual_or,
            title=body.title or "",
        )
        log.info(
            "[Feedback] Push %s: actual_or=%.2f%% predicted_or=%.2f%%",
            push_id, actual_or, predicted_or,
        )
        return JSONResponse(content={"ok": True, "status": "ok"})
    except Exception as exc:
        log.exception("[Feedback] Fehler beim Speichern")
        raise HTTPException(status_code=500, detail="Feedback could not be stored.") from exc


@router.post("/api/predictions/feedback")
def post_prediction_feedback(body: PredictionFeedbackRequest) -> JSONResponse:
    """Speichert tatsächliche Opening Rate für eine frühere Prediction."""
    return _handle_feedback(body)


@router.post("/api/prediction-feedback", include_in_schema=False)
def post_prediction_feedback_legacy(body: PredictionFeedbackRequest) -> JSONResponse:
    """Compat-Endpunkt für altes HTML-Frontend (ohne /s)."""
    return _handle_feedback(body)
