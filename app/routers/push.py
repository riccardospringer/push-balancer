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
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.config import PUSH_DB_PATH, PUSH_LIVE_FETCH_ENABLED, SYNC_SECRET, push_api_base_candidates
from app.database import push_db_load_all, push_db_log_prediction

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
    pushId: str
    actualOr: float
    title: str | None = None


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


def _fetch_live_push_snapshot() -> tuple[list, list]:
    if not PUSH_LIVE_FETCH_ENABLED:
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
        messages, channels = _fetch_live_push_snapshot()
        with _push_sync_lock:
            _push_sync_cache["messages"] = messages
            _push_sync_cache["channels"] = channels
            _push_sync_cache["ts"] = time.time()
        return {"ok": True, "synced": len(messages), "channels": len(channels), "source": "live"}
    except Exception as exc:
        log.warning("[pushes] live refresh failed, falling back to DB summary: %s", exc)
        try:
            rows = push_db_load_all(max_rows=50)
        except Exception as db_exc:
            log.warning("[pushes] refresh fallback without DB: %s", db_exc)
            rows = []
        return {"ok": True, "synced": len(rows), "channels": 0, "source": "db-fallback"}


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


@router.post("/api/predictions/feedback")
def post_prediction_feedback(body: PredictionFeedbackRequest) -> JSONResponse:
    """Speichert tatsächliche Opening Rate für eine frühere Prediction.

    Wird vom Dashboard aufgerufen wenn eine Push-OR bekannt wird.
    Dient als Trainings-Label für das ML-Modell.

    IMPLEMENTIERUNGSHINWEIS:
        Vollstaendige Handler-Logik aus dem frueheren Monolithen:
        _prediction_feedback() hierher migrieren (enthält auch Online-Bias-Update).
    """
    try:
        push_db_log_prediction(
            push_id=body.pushId,
            predicted_or=0.0,
            actual_or=body.actualOr,
            title=body.title or "",
        )
        log.info("[Feedback] Push %s: actual_or=%.2f%%", body.pushId, body.actualOr)
        return JSONResponse(content={"ok": True})
    except Exception as exc:
        log.exception("[Feedback] Fehler in post_prediction_feedback")
        raise HTTPException(
            status_code=500,
            detail="Feedback could not be stored.",
        ) from exc
