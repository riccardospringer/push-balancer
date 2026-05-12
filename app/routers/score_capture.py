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
from typing import Any

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


def get_score_for_url(url: str) -> float | None:
    key = hashlib.md5(_normalize_url(url).encode()).hexdigest()
    with _lock:
        entry = _score_cache.get(key)
        if entry and time.time() - entry["ts"] < _CACHE_TTL:
            return entry["score"]
    return None


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
    """Empfängt Artikel-Scores vom Kandidaten-Tab."""
    with _lock:
        for item in body.scores:
            if not item.url or item.score <= 0:
                continue
            key = hashlib.md5(_normalize_url(item.url).encode()).hexdigest()
            # Nur speichern wenn Score neuer oder Score höher (frischere Berechnung)
            existing = _score_cache.get(key)
            if not existing or item.ts >= existing["ts"]:
                _score_cache[key] = {"score": round(item.score, 1), "ts": item.ts, "url": item.url}
        _cleanup()
    return JSONResponse({"ok": True, "stored": len(body.scores)})
