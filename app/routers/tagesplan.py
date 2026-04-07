"""app/routers/tagesplan.py — Tagesplan-Endpunkte.

GET  /api/tagesplan                   — ML-Tagesplan
GET  /api/tagesplan/retro             — Retro-Analyse heutiger Slots
GET  /api/tagesplan/history           — Historische Tagesplan-Daten
GET  /api/tagesplan/suggestions       — Gespeicherte Slot-Vorschläge
POST /api/tagesplan/log-suggestions   — Speichert Slot-Vorschläge
"""
import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.database import load_tagesplan_suggestions, save_tagesplan_suggestions
from app.tagesplan.builder import build_tagesplan, build_tagesplan_retro

log = logging.getLogger("push-balancer")
router = APIRouter()


class LogSuggestionsRequest(BaseModel):
    date_iso: str
    slot_hour: int
    suggestions: list[dict[str, Any]]


@router.get("/api/tagesplan")
def get_tagesplan(mode: str = Query(default="redaktion")) -> JSONResponse:
    """Liefert den ML-gestützten Tagesplan.

    Der Tagesplan wird vom Research-Worker alle 5 Min im Hintergrund
    aufgefrischt. Dieser Endpoint gibt den gecachten Plan zurück.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_tagesplan()
        (Zeile 15014) hierher migrieren.
    """
    try:
        if mode not in ("redaktion", "sport"):
            mode = "redaktion"
        plan = build_tagesplan(background=False, mode=mode)
        if plan is None:
            import datetime
            plan = {"slots": [], "loading": True, "mode": mode,
                    "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                    "ml_trained": False, "total_pushes_db": 0}
        return JSONResponse(content=plan)
    except Exception as e:
        log.exception("[tagesplan] Fehler in get_tagesplan")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/tagesplan/retro")
def get_tagesplan_retro() -> JSONResponse:
    """Liefert die Retro-Analyse vergangener Slots des heutigen Tages.

    Gespeicherte OR-Snapshots werden NIEMALS live überschrieben.
    Past-Slots zeigen nur den DB-Snapshot.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_tagesplan_retro()
        (Zeile 15030) hierher migrieren.
    """
    try:
        retro = build_tagesplan_retro()
        if retro is None:
            return JSONResponse(content={"slots": [], "loading": True})
        return JSONResponse(content=retro)
    except Exception as e:
        log.exception("[tagesplan] Fehler in get_tagesplan_retro")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/tagesplan/history")
def get_tagesplan_history(
    days: int = Query(default=7, ge=1, le=90),
) -> JSONResponse:
    """Liefert historische Tagesplan-Performance-Daten.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_tagesplan_history()
        (Zeile 15041) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _serve_tagesplan_history_data  # type: ignore
        return JSONResponse(content=_serve_tagesplan_history_data(days))
    except ImportError:
        pass
    return JSONResponse(content={"days": days, "slots": [], "loading": False})


@router.get("/api/tagesplan/suggestions")
def get_tagesplan_suggestions(
    date: str | None = Query(default=None),
) -> JSONResponse:
    """Liefert gespeicherte Tagesplan-Slot-Vorschläge.

    Query-Parameter:
        date: ISO-Datumsstring (YYYY-MM-DD), optional
    """
    rows = load_tagesplan_suggestions(date_iso=date)
    # Frontend erwartet {hour_str: [suggestions]} — nach slot_hour gruppieren
    grouped: dict[str, list] = {}
    for row in rows:
        key = str(row.get("slot_hour", ""))
        grouped.setdefault(key, []).append(row)
    return JSONResponse(content={"suggestions": grouped, "count": len(rows)})


@router.post("/api/tagesplan/log-suggestions")
def log_tagesplan_suggestions(body: LogSuggestionsRequest) -> JSONResponse:
    """Speichert Tagesplan-Slot-Vorschläge (serverseitig für Audit-Trail).

    KRITISCH: Gespeicherte ORs werden NIEMALS live überschrieben.
    Beim Speichern: individuelles predictOR(), kein Slot-Level-Fallback.
    """
    try:
        save_tagesplan_suggestions(
            date_iso=body.date_iso,
            slot_hour=body.slot_hour,
            suggestions=body.suggestions,
        )
        return JSONResponse(content={"ok": True, "saved": len(body.suggestions)})
    except Exception as e:
        log.exception("[tagesplan] Fehler in log_tagesplan_suggestions")
        return JSONResponse(status_code=500, content={"error": str(e)})
