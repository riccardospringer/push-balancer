"""
app/routers/score_capture.py — Score-Snapshot-Endpoint.

POST /api/score-capture  — empfängt Artikel-Scores vom Kandidaten-Tab (Browser)
GET  /api/score-capture/by-cms-id/{cms_id} — gibt einen gespeicherten Score zurück

Der Browser sendet alle Kandidaten-Scores aus dem unveränderten Standardfilter
alle 30s wenn der Kandidaten-Tab offen ist.
Wenn ein Push rausgeht, wird der zuletzt gespeicherte Score für diese URL genutzt.
Persistenz-TTL: 8 Stunden; der CMS-ID-Endpunkt nutzt dieses Arbeitsfenster,
damit ein bereits angezeigter Original-Score nicht nach drei Minuten verschwindet.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
import threading
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Path, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()

_lock = threading.Lock()
# url_hash → {score, ts, url}
_score_cache: dict[str, dict] = {}
_CACHE_TTL = 8 * 3600  # 8 Stunden (ein Arbeitstag)
_CMS_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_BILD_HOSTS = frozenset({"bild.de", "www.bild.de"})
_MAX_FUTURE_SKEW_SECONDS = 30


def _cleanup():
    cutoff = time.time() - _CACHE_TTL
    dead = [k for k, v in _score_cache.items() if v["ts"] < cutoff]
    for k in dead:
        del _score_cache[k]


def _normalize_url(url: str) -> str:
    """Normalisiert URL für konsistentes Matching zwischen Sitemap und CMS."""
    url = url.strip().rstrip("/").lower()
    # Query-Parameter und Fragment entfernen
    url = url.split("?")[0].split("#")[0]
    # Tracking-Suffixe entfernen (bild.de spezifisch)
    for suffix in (".bild.html", ".html"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _url_matches_cms_id(url: str, cms_id: str) -> bool:
    """Matcht eine CMS-ID ausschließlich im Pfad einer vertrauenswürdigen BILD-URL."""
    if not _CMS_ID_RE.fullmatch(cms_id) or not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _BILD_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    token = re.compile(rf"(?<![0-9a-f]){re.escape(cms_id.lower())}(?![0-9a-f])")
    return bool(token.search(parsed.path.lower()))


def _fresh_minimal_snapshot(
    entry: dict,
    *,
    now: int,
    max_age_seconds: int,
) -> dict[str, float | int] | None:
    """Validiert einen Capture-Eintrag und reduziert ihn auf den Consumer-Vertrag."""
    score_raw = entry.get("score")
    captured_at_raw = entry.get("ts", entry.get("captured_at"))
    if isinstance(score_raw, bool) or isinstance(captured_at_raw, bool):
        return None
    try:
        score = float(score_raw)
        captured_at = int(captured_at_raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(score)
        or not 0 < score <= 100
        or captured_at <= 0
        or captured_at > now + _MAX_FUTURE_SKEW_SECONDS
        or now - captured_at >= max_age_seconds
    ):
        return None
    return {"score": score, "capturedAt": captured_at}


def get_score_snapshot_for_url(
    url: str,
    *,
    max_age_seconds: int = _CACHE_TTL,
    allow_db_fallback: bool = True,
) -> dict[str, float | int | str] | None:
    """Liefert einen frischen Browser-Score mit Alter und technischer Quelle."""
    normalized_url = _normalize_url(url)
    max_age = max(0, int(max_age_seconds))
    if not normalized_url or max_age <= 0:
        return None

    key = hashlib.md5(normalized_url.encode()).hexdigest()
    now = int(time.time())
    # 1. Memory-Cache (schnell)
    with _lock:
        entry = _score_cache.get(key)
        if entry:
            captured_at = int(entry["ts"])
            age_seconds = max(0, now - captured_at)
            if age_seconds < max_age:
                return {
                    "score": float(entry["score"]),
                    "capturedAt": captured_at,
                    "ageSeconds": age_seconds,
                    "source": "memory",
                }
    if not allow_db_fallback:
        return None

    # 2. DB-Fallback (überlebt Restarts)
    try:
        from app.database import get_article_score_snapshot_from_db

        snapshot = get_article_score_snapshot_from_db(
            normalized_url,
            max_age_seconds=max_age,
        )
        if snapshot:
            return {
                "score": float(snapshot["score"]),
                "capturedAt": int(snapshot["captured_at"]),
                "ageSeconds": int(snapshot["age_seconds"]),
                "source": "database",
            }
    except Exception:
        pass
    return None


def get_score_for_url(url: str) -> float | None:
    snapshot = get_score_snapshot_for_url(url)
    return float(snapshot["score"]) if snapshot else None


def get_score_snapshot_for_cms_id(
    cms_id: str,
    *,
    max_age_seconds: int = _CACHE_TTL,
    allow_db_fallback: bool = True,
) -> dict[str, float | int] | None:
    """Liefert den neuesten UI-Capture des laufenden Arbeitstags."""
    if not _CMS_ID_RE.fullmatch(cms_id):
        return None
    max_age = max(0, int(max_age_seconds))
    if max_age <= 0:
        return None

    now = int(time.time())
    memory_matches: list[dict[str, float | int]] = []
    with _lock:
        entries = list(_score_cache.values())
    for entry in entries:
        if not _url_matches_cms_id(entry.get("url", ""), cms_id):
            continue
        snapshot = _fresh_minimal_snapshot(entry, now=now, max_age_seconds=max_age)
        if snapshot is not None:
            memory_matches.append(snapshot)
    if memory_matches:
        return max(memory_matches, key=lambda item: int(item["capturedAt"]))

    if not allow_db_fallback:
        return None
    try:
        from app.database import get_article_score_snapshot_by_cms_id_from_db

        snapshot = get_article_score_snapshot_by_cms_id_from_db(
            cms_id,
            max_age_seconds=max_age,
        )
    except Exception:
        return None
    if snapshot is None:
        return None
    return _fresh_minimal_snapshot(snapshot, now=now, max_age_seconds=max_age)


class ScoreCaptureItem(BaseModel):
    url: str
    score: float
    ts: int


class ScoreCaptureRequest(BaseModel):
    scores: list[ScoreCaptureItem]


class CmsScoreCaptureResponse(BaseModel):
    score: float = Field(gt=0, le=100)
    capturedAt: int = Field(gt=0)


class ScoreCaptureHealthResponse(BaseModel):
    status: str


@router.get(
    "/api/score-capture/health",
    response_model=ScoreCaptureHealthResponse,
    summary="Check score capture reachability",
    description=(
        "Returns a minimal readiness response behind the same internal CIDR gate as the "
        "CMS-ID score lookup."
    ),
)
def get_score_capture_health(response: Response) -> ScoreCaptureHealthResponse:
    response.headers["Cache-Control"] = "no-store"
    return ScoreCaptureHealthResponse(status="ok")


@router.get("/api/score-capture/debug")
def debug_score_capture() -> JSONResponse:
    """Zeigt alle gecachten Scores (für Debugging)."""
    with _lock:
        entries = sorted(_score_cache.values(), key=lambda x: x["ts"], reverse=True)
    return JSONResponse({"count": len(entries), "entries": entries[:30]})


@router.get(
    "/api/score-capture/by-cms-id/{cms_id}",
    response_model=CmsScoreCaptureResponse,
    summary="Get a captured UI score by CMS ID",
    description=(
        "Returns only the latest workday score and capture timestamp already produced by the "
        "unchanged candidate UI. Access remains protected by the service's internal CIDR gate."
    ),
)
def get_score_capture_by_cms_id(
    response: Response,
    cms_id: str = Path(pattern=r"^[0-9a-fA-F]{24}$"),
) -> CmsScoreCaptureResponse:
    snapshot = get_score_snapshot_for_cms_id(cms_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No current score is available for this CMS ID.")
    response.headers["Cache-Control"] = "no-store"
    return CmsScoreCaptureResponse(
        score=float(snapshot["score"]),
        capturedAt=int(snapshot["capturedAt"]),
    )


@router.post("/api/score-capture")
def post_score_capture(body: ScoreCaptureRequest) -> JSONResponse:
    """Empfängt Artikel-Scores vom Kandidaten-Tab — speichert in Memory + DB."""
    from app.database import save_article_score_to_db
    stored = 0
    with _lock:
        for item in body.scores:
            if not item.url or item.score <= 0:
                continue
            key = hashlib.md5(_normalize_url(item.url).encode()).hexdigest()
            existing = _score_cache.get(key)
            if not existing or item.ts >= existing["ts"]:
                _score_cache[key] = {"score": round(item.score, 1), "ts": item.ts, "url": item.url}
                stored += 1
        _cleanup()
    # Persistent in DB schreiben (überlebt Render-Restarts)
    for item in body.scores:
        if item.url and item.score > 0:
            save_article_score_to_db(item.url, round(item.score, 1))
    return JSONResponse({"ok": True, "stored": stored})
