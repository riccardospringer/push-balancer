"""
Push-Alarm HTTP-Endpoints.

GET  /api/push-alarm         — aktueller Alarm-State (Frontend pollt alle 30s)
POST /api/push-alarm/dismiss — Alarm für 10 Min. unterdrücken
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# ── Globaler Alarm-State ──────────────────────────────────────────────────────

_alarm_lock = threading.Lock()
_alarm_state: dict = {
    "active":         False,
    "article":        None,
    "reason":         "",
    "checked_at":     0,
    "dismissed_until": 0,
}


def update_alarm_state(recommendation) -> None:
    """Wird vom Background-Worker aufgerufen."""
    with _alarm_lock:
        now = time.time()
        if recommendation is None:
            _alarm_state["active"]     = False
            _alarm_state["article"]    = None
            _alarm_state["reason"]     = ""
            _alarm_state["checked_at"] = now
        else:
            # Nicht aktivieren wenn gerade dismissed
            if now < _alarm_state["dismissed_until"]:
                _alarm_state["checked_at"] = now
                return
            _alarm_state["active"]     = True
            _alarm_state["article"]    = recommendation.to_dict()
            _alarm_state["reason"]     = recommendation.reason
            _alarm_state["checked_at"] = now


def get_alarm_state() -> dict:
    with _alarm_lock:
        return dict(_alarm_state)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/push-alarm")
def get_push_alarm() -> JSONResponse:
    """Gibt den aktuellen Alarm-State zurück."""
    state = get_alarm_state()
    now = time.time()
    # Dismiss-Ablauf melden
    dismissed_secs = max(0, int(state["dismissed_until"] - now))
    return JSONResponse({
        "active":          state["active"] and now >= state["dismissed_until"],
        "article":         state["article"],
        "reason":          state["reason"],
        "checkedAt":       int(state["checked_at"]),
        "dismissedForSecs": dismissed_secs,
    })


@router.post("/api/push-alarm/dismiss")
def dismiss_push_alarm() -> JSONResponse:
    """Unterdrückt den Alarm für 10 Minuten."""
    with _alarm_lock:
        _alarm_state["active"]          = False
        _alarm_state["dismissed_until"] = time.time() + 600   # 10 Min.
    return JSONResponse({"ok": True, "dismissedForSecs": 600})
