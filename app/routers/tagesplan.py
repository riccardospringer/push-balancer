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
    dateIso: str
    slotHour: int
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
                    "mlTrained": False, "totalPushesDb": 0}
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
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> JSONResponse:
    """Liefert historische Tagesplan-Performance-Daten (paginiert).

    Query-Parameter:
        days:   Anzahl vergangener Tage (Standard: 7, max. 90)
        offset: Startindex (Standard: 0)
        limit:  Max. Anzahl Slot-Einträge (Standard: 200, max. 1000)
    """
    import sqlite3
    import datetime
    from app.database import PUSH_DB_PATH

    try:
        cutoff_ts = int(
            (datetime.datetime.now() - datetime.timedelta(days=days)).timestamp()
        )
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        all_rows = conn.execute(
            """
            SELECT
                date(ts_num, 'unixepoch', 'localtime') AS date,
                hour,
                AVG(or_val) AS avg_or,
                COUNT(*) AS n_pushes
            FROM pushes
            WHERE ts_num >= ? AND or_val > 0
            GROUP BY date, hour
            ORDER BY date, hour
            """,
            (cutoff_ts,),
        ).fetchall()
        conn.close()

        all_slots = [
            {
                "date": r["date"],
                "hour": r["hour"],
                "avgOr": round(r["avg_or"], 2),
                "nPushes": r["n_pushes"],
            }
            for r in all_rows
        ]
        total = len(all_slots)
        items = all_slots[offset: offset + limit]
        return JSONResponse(content={
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "days": days,
            "loading": False,
        })
    except Exception as e:
        log.exception("[tagesplan] Fehler in get_tagesplan_history")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/tagesplan/suggestions")
def get_tagesplan_suggestions(
    date: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> JSONResponse:
    """Liefert gespeicherte Tagesplan-Slot-Vorschläge (paginiert).

    Query-Parameter:
        date:   ISO-Datumsstring (YYYY-MM-DD), optional
        offset: Startindex über alle Vorschläge (Standard: 0)
        limit:  Max. Anzahl Einträge (Standard: 20, max. 200)
    """
    all_rows = load_tagesplan_suggestions(date_iso=date)
    total = len(all_rows)
    items = all_rows[offset: offset + limit]
    # Grouped view (nach slot_hour) für Frontend-Kompatibilität
    grouped: dict[str, list] = {}
    for row in items:
        key = str(row.get("slot_hour", ""))
        grouped.setdefault(key, []).append(row)
    return JSONResponse(content={
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "grouped": grouped,
    })


@router.post("/api/tagesplan/log-suggestions")
def log_tagesplan_suggestions(body: LogSuggestionsRequest) -> JSONResponse:
    """Speichert Tagesplan-Slot-Vorschläge (serverseitig für Audit-Trail).

    KRITISCH: Gespeicherte ORs werden NIEMALS live überschrieben.
    Beim Speichern: individuelles predictOR(), kein Slot-Level-Fallback.
    """
    try:
        save_tagesplan_suggestions(
            date_iso=body.dateIso,
            slot_hour=body.slotHour,
            suggestions=body.suggestions,
        )
        return JSONResponse(content={"ok": True, "saved": len(body.suggestions)})
    except Exception as e:
        log.exception("[tagesplan] Fehler in log_tagesplan_suggestions")
        return JSONResponse(status_code=500, content={"error": str(e)})
