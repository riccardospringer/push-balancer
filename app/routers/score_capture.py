"""
app/routers/score_capture.py — Score-Snapshot-Endpoint.

POST /api/score-capture  — empfängt Artikel-Scores vom Kandidaten-Tab (Browser)
GET  /api/score-capture/{url_hash} — gibt gespeicherten Score für eine URL zurück

Der Browser sendet die Top-20-Artikel-Scores alle 30s wenn der Kandidaten-Tab offen ist.
Wenn ein Push rausgeht, wird der zuletzt gespeicherte Score für diese URL genutzt.
TTL: 4 Stunden (danach irrelevant, da Push-Artikel keine Push-Kandidaten mehr sind).
"""
from __future__ import annotations

import hashlib
import time
import threading
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

_lock = threading.Lock()
# url_hash → {score, ts, url}
_score_cache: dict[str, dict] = {}
_CACHE_TTL = 8 * 3600  # 8 Stunden (ein Arbeitstag)


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


class ScoreCaptureItem(BaseModel):
    url: str
    score: float
    ts: int


class ScoreCaptureRequest(BaseModel):
    scores: list[ScoreCaptureItem]


@router.get("/api/score-capture/debug")
def debug_score_capture() -> JSONResponse:
    """Zeigt alle gecachten Scores (für Debugging)."""
    with _lock:
        entries = sorted(_score_cache.values(), key=lambda x: x["ts"], reverse=True)
    return JSONResponse({"count": len(entries), "entries": entries[:30]})


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
