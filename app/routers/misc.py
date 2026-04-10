"""app/routers/misc.py — Verschiedene Endpunkte.

GET  /api/check-plus             — BILD-Plus-Paywall-Check (GET)
POST /api/check-plus             — BILD-Plus-Paywall-Check (POST, Batch)
GET  /api/adobe/traffic          — Adobe Analytics Traffic Sources
POST /api/schwab-chat            — Schwab-Chat (LLM-Dialog)
POST /api/schwab-approval        — Schwab-Genehmigung
POST /api/push-title/generate    — Push-Titel-Generator (LLM)
"""
import concurrent.futures
import json
import logging
import ssl
import time
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET

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
    url: str = ""
    title: str = ""
    category: str = "news"


def _build_push_title_response(body: PushTitleGenerateRequest) -> dict:
    from app.config import OPENAI_API_KEY

    if not body.title:
        raise HTTPException(status_code=400, detail="title is required")

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Push title generation is unavailable because OPENAI_API_KEY is not configured.",
        )

    from push_title_agent import generate_push_title

    result = generate_push_title(
        article_title=body.title,
        article_text="",
        category=body.category or "news",
        kicker="",
        headline="",
    )

    gewinner = result.get("gewinner", {})
    alternative = result.get("alternative", {})
    alle = result.get("alle_kandidaten", {})

    winner_title = gewinner.get("titel", body.title)

    alt_titles: list[str] = []
    if alternative.get("titel") and alternative["titel"] != winner_title:
        alt_titles.append(alternative["titel"])
    for gruppe in alle.values():
        for kandidat in gruppe:
            titel = kandidat.get("titel", "")
            if titel and titel != winner_title and titel not in alt_titles:
                alt_titles.append(titel)

    reasoning = gewinner.get("warum_dieser", "")
    if not reasoning:
        analyse = result.get("meta", {}).get("analyse", {})
        reasoning = analyse.get("kern", "")

    return {
        "title": winner_title,
        "alternativeTitles": alt_titles[:5],
        "reasoning": reasoning,
        "advisoryOnly": True,
    }


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


@router.get("/api/analytics/adobe-traffic")
def get_adobe_traffic_analytics() -> JSONResponse:
    """Stable Adobe traffic contract for the frontend and OpenAPI clients."""
    response = get_adobe_traffic()
    payload = json.loads(response.body.decode("utf-8"))
    return JSONResponse(
        content={
            "hourly": payload.get("hourly", []),
            "topArticles": payload.get("topArticles", []),
            "fetchedAt": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(payload.get("updatedAt", 0)))
                if payload.get("updatedAt")
                else ""
            ),
            "enabled": payload.get("enabled", False),
            "loading": payload.get("loading", False),
            "error": payload.get("error", ""),
        }
    )


@router.post("/api/schwab-chat")
def post_schwab_chat(body: SchwabChatRequest) -> JSONResponse:
    """Schwab-Chat: LLM-Dialog für Push-Empfehlungen.

    Die Chat-Funktion ist im FastAPI-Refactor noch nicht migriert.
    """
    raise HTTPException(
        status_code=501,
        detail="Schwab chat is not implemented in the current FastAPI runtime.",
    )


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
    """Generiert Push-Titel via GPT-4o im Editorial-One-Brain-Modus."""
    try:
        return JSONResponse(content=_build_push_title_response(body))
    except RuntimeError as exc:
        log.error("[PushTitle] %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/api/push-title-generations")
def create_push_title_generation(body: PushTitleGenerateRequest) -> JSONResponse:
    """Resource-style alias for advisory push title generation."""
    try:
        return JSONResponse(content=_build_push_title_response(body))
    except HTTPException:
        raise
    except RuntimeError as exc:
        log.error("[PushTitle] %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("[PushTitle] Endpoint-Fehler")
        raise HTTPException(
            status_code=500,
            detail="Push title generation failed.",
        ) from exc
