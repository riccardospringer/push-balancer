"""app/routers/push.py — Push-API-Proxy und Sync-Endpunkte.

GET  /api/push/{path:path}     — Proxy zur BILD Push-API
POST /api/pushes/sync          — Empfängt Push-Daten von lokalem Server (Relay)
POST /api/predictions/feedback — Prediction-Feedback für ML-Training
"""
import json
import logging
import time
import threading
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.config import PUSH_API_BASE, SYNC_SECRET
from app.database import push_db_log_prediction

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


@router.get("/api/push/{path:path}")
async def proxy_push_api(path: str, request: Request) -> Response:
    """Proxy zur BILD Push-API mit Sync-Cache-Fallback.

    1. Versuch: Direkt zur BILD Push-API
    2. Fallback: Sync-Cache (von lokalem Server befüllt)
    3. Kein Cache: Leeres Ergebnis mit Fehlermeldung
    """
    full_path = f"/push/{path}"
    query = str(request.url.query)
    url = f"{PUSH_API_BASE}{full_path}"
    if query:
        url = f"{url}?{query}"

    # 1. Direkt zur BILD Push-API
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
    except Exception as e:
        log.info("[Proxy] Push-API direkt nicht erreichbar, prüfe Sync-Cache: %s", e)

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


@router.post("/api/pushes/sync")
def post_push_sync(body: PushSyncRequest) -> JSONResponse:
    """Empfängt Push-Daten von lokalem Server (Relay für Render).

    Authentifizierung via PUSH_SYNC_SECRET.
    """
    if body.secret != SYNC_SECRET:
        raise HTTPException(status_code=403, detail="Invalid sync secret")

    with _push_sync_lock:
        _push_sync_cache["messages"] = body.messages
        _push_sync_cache["channels"] = body.channels
        _push_sync_cache["ts"] = time.time()

    log.info("[Sync] Empfangen: %d Messages, %d Channels",
             len(body.messages), len(body.channels))
    return JSONResponse(content={"ok": True, "received": len(body.messages)})


@router.post("/api/predictions/feedback")
def post_prediction_feedback(body: PredictionFeedbackRequest) -> JSONResponse:
    """Speichert tatsächliche Opening Rate für eine frühere Prediction.

    Wird vom Dashboard aufgerufen wenn eine Push-OR bekannt wird.
    Dient als Trainings-Label für das ML-Modell.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py:
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
    except Exception:
        log.exception("[Feedback] Fehler in post_prediction_feedback")
        return JSONResponse(status_code=500, content={"error": "Feedback konnte nicht gespeichert werden"})
