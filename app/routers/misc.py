"""app/routers/misc.py — Verschiedene Endpunkte.

GET  /api/check-plus             — BILD-Plus-Paywall-Check (GET)
POST /api/check-plus             — BILD-Plus-Paywall-Check (POST, Batch)
GET  /api/adobe/traffic          — Adobe Analytics Traffic Sources
POST /api/schwab-chat            — Schwab-Chat (LLM-Dialog)
POST /api/schwab-approval        — Schwab-Genehmigung
POST /api/push-title/generate    — Push-Titel-Generator (LLM)
"""
import concurrent.futures
import logging
import ssl
import urllib.request
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET
from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()

# ── Adobe State (module-level, wird von adobe_traffic_worker befüllt) ─────
# (Referenz auf den globalen State in push-balancer-server.py)
_adobe_state: dict = {
    "access_token": "",
    "token_expires": 0,
    "traffic": None,
    "updated_at": 0,
    "error": "",
    "enabled": bool(ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET),
}

# ── SSL Context ────────────────────────────────────────────────────────────
try:
    import certifi as _certifi
    _SSL_CTX = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


class CheckPlusRequest(BaseModel):
    urls: list[str]


class SchwabChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []


class SchwabApprovalRequest(BaseModel):
    decision: str
    push_id: str | None = None


class PushTitleGenerateRequest(BaseModel):
    headline: str
    kicker: str | None = None
    category: str = "News"
    target_length: int = 80


def _check_bild_plus(url: str) -> bool:
    """Prüft ob eine BILD-URL hinter der Plus-Paywall liegt."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)"},
        )
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            content = resp.read(2048).decode("utf-8", errors="ignore")
        return "bild-plus" in content.lower() or "bildplus" in content.lower()
    except Exception:
        return False


@router.get("/api/check-plus")
@router.post("/api/check-plus")
def check_plus_urls(
    body: CheckPlusRequest | None = None,
    url: str = Query(default=""),
) -> JSONResponse:
    """Prüft BILD-URLs auf BILD-Plus-Paywall (GET: einzelne URL, POST: Batch).

    GET:  ?url=https://...
    POST: {"urls": ["https://...", ...]}

    Prüft max. 20 URLs parallel.
    """
    urls: list[str] = []
    if body is not None:
        urls = body.urls
    elif url:
        urls = [url]

    if not urls:
        return JSONResponse(content={})

    safe_urls = [u for u in urls if u.startswith("https://www.bild.de/")][:20]

    results: dict[str, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_check_bild_plus, u): u for u in safe_urls}
        for future in concurrent.futures.as_completed(futures):
            u = futures[future]
            try:
                results[u] = future.result()
            except Exception:
                results[u] = False

    return JSONResponse(content=results)


@router.get("/api/adobe/traffic")
def get_adobe_traffic() -> JSONResponse:
    """Liefert Adobe Analytics Traffic-Quellen für heutige Pushes.

    Nutzt den Adobe Analytics API (OAuth2 Client Credentials) und
    matcht Traffic-Daten per Fuzzy-Matching mit Push-Titeln.

    Wird vom adobe_traffic_worker alle 30 Min befüllt.
    """
    if not _adobe_state["enabled"]:
        return JSONResponse(content={"enabled": False, "error": "Adobe nicht konfiguriert"})

    traffic = _adobe_state.get("traffic")
    if not traffic:
        return JSONResponse(content={
            "enabled": True,
            "loading": True,
            "updatedAt": 0,
            "error": _adobe_state.get("error", ""),
        })

    return JSONResponse(content={
        "enabled": True,
        "loading": False,
        "updatedAt": _adobe_state["updated_at"],
        "error": "",
        **traffic,
    })


@router.post("/api/schwab-chat")
def post_schwab_chat(body: SchwabChatRequest) -> JSONResponse:
    """Schwab-Chat: LLM-Dialog für Push-Empfehlungen.

    Chat-Funktion noch nicht migriert — gibt Stub zurück.
    """
    return JSONResponse(content={
        "ok": False,
        "message": "Chat-Funktion wird migriert",
    })


@router.post("/api/schwab-approval")
def post_schwab_approval(body: SchwabApprovalRequest) -> JSONResponse:
    """Schwab-Genehmigung: Redakteur bestätigt/lehnt Push-Empfehlung ab.

    SAFETY: Diese Funktion loggt nur die Entscheidung.
    Das System sendet NIEMALS autonom Pushes.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py:
        _schwab_approval() hierher migrieren.
    """
    log.info("[Schwab] Approval: decision=%s push_id=%s", body.decision, body.push_id)
    return JSONResponse(content={
        "ok": True,
        "advisoryOnly": True,
        "actionAllowed": False,
        "message": "Entscheidung geloggt. Push-Versand erfordert manuelle Aktion im CMS.",
    })


@router.post("/api/push-title/generate")
def post_push_title_generate(body: PushTitleGenerateRequest) -> JSONResponse:
    """Generiert Push-Titel-Varianten via LLM (GPT-4o).

    LLM-Titelfunktion noch nicht migriert — gibt Stub zurück.
    """
    return JSONResponse(content={
        "ok": False,
        "message": "LLM-Titelfunktion nicht verfügbar",
        "titles": [],
    })
