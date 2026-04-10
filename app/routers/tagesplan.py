"""app/routers/tagesplan.py — Tagesplan-Endpunkte.

GET  /api/tagesplan                   — ML-Tagesplan
GET  /api/tagesplan/retro             — Retro-Analyse heutiger Slots
GET  /api/tagesplan/history           — Historische Tagesplan-Daten
GET  /api/tagesplan/suggestions       — Gespeicherte Slot-Vorschläge
POST /api/tagesplan/log-suggestions   — Speichert Slot-Vorschläge
"""
import logging
import sqlite3
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import PUSH_DB_PATH
from app.database import load_tagesplan_suggestions, save_tagesplan_suggestions
from app.tagesplan.builder import build_tagesplan, build_tagesplan_retro

log = logging.getLogger("push-balancer")
router = APIRouter()


class LogSuggestionsRequest(BaseModel):
    # Neues Format (React): {dateIso, slotHour, suggestions:[...]}
    dateIso: str | None = None
    slotHour: int | None = None
    # Legacy-Format des frueheren HTML-Clients: {date_iso, suggestions:[{slot_hour,...}]}
    date_iso: str | None = None
    suggestions: list[dict[str, Any]] = []


def _save_suggestions_legacy_flat(date_iso: str, suggestions: list[dict[str, Any]]) -> int:
    """Speichert Legacy-Flat-Format (slot_hour + suggestion_num je Eintrag)."""
    now_ts = int(time.time())
    saved = 0
    with sqlite3.connect(PUSH_DB_PATH, timeout=10) as conn:
        for s in suggestions[:36]:  # max 18 Slots * 2 Vorschläge
            try:
                slot_hour = int(s.get("slot_hour", 0))
                suggestion_num = int(s.get("suggestion_num", 1))
            except (TypeError, ValueError):
                continue
            conn.execute(
                """INSERT OR REPLACE INTO tagesplan_suggestions
                (date_iso, slot_hour, suggestion_num, article_title, article_link,
                 article_category, article_score, expected_or, best_cat, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date_iso,
                    slot_hour,
                    suggestion_num,
                    (s.get("title") or "")[:200],
                    (s.get("link") or "")[:500],
                    (s.get("category") or s.get("cat") or "")[:50],
                    round(float(s.get("score", 0) or 0), 1),
                    round(float(s.get("expected_or", 0) or 0), 2),
                    (s.get("best_cat") or s.get("category") or s.get("cat") or "")[:50],
                    now_ts,
                ),
            )
            saved += 1
        conn.execute(
            "DELETE FROM tagesplan_suggestions WHERE captured_at < ?",
            (now_ts - 30 * 86400,),
        )
        conn.commit()
    return saved


@router.get("/api/tagesplan")
def get_tagesplan(
    date: str | None = Query(default=None),
    mode: str = Query(default="redaktion"),
) -> JSONResponse:
    """Liefert den ML-gestützten Tagesplan.

    Der Tagesplan wird vom Research-Worker alle 5 Min im Hintergrund
    aufgefrischt. Dieser Endpoint gibt den gecachten Plan zurück.

    IMPLEMENTIERUNGSHINWEIS:
        Vollstaendige Handler-Logik aus dem frueheren Monolithen: _serve_tagesplan()
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
        if date:
            plan["requestedDate"] = date
        return JSONResponse(content=plan)
    except Exception as exc:
        log.exception("[tagesplan] Fehler in get_tagesplan")
        raise HTTPException(status_code=500, detail="Daily plan could not be loaded.") from exc


@router.get("/api/tagesplan/retro")
def get_tagesplan_retro(mode: str = Query(default="redaktion")) -> JSONResponse:
    """Liefert die Retro-Analyse vergangener Slots des heutigen Tages.

    Gespeicherte OR-Snapshots werden NIEMALS live überschrieben.
    Past-Slots zeigen nur den DB-Snapshot.

    IMPLEMENTIERUNGSHINWEIS:
        Vollstaendige Handler-Logik aus dem frueheren Monolithen: _serve_tagesplan_retro()
        (Zeile 15030) hierher migrieren.
    """
    try:
        retro = build_tagesplan_retro()
        if retro is None:
            return JSONResponse(content={"slots": [], "loading": True, "mode": mode})
        retro["mode"] = mode
        return JSONResponse(content=retro)
    except Exception as exc:
        log.exception("[tagesplan] Fehler in get_tagesplan_retro")
        raise HTTPException(
            status_code=500,
            detail="Daily plan retrospective data could not be loaded.",
        ) from exc


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
    except Exception as exc:
        log.exception("[tagesplan] Fehler in get_tagesplan_history")
        raise HTTPException(
            status_code=500,
            detail="Daily plan history could not be loaded.",
        ) from exc


@router.get("/api/tagesplan/suggestions")
def get_tagesplan_suggestions(
    date: str | None = Query(default=None),
    mode: str = Query(default="redaktion"),
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
        "mode": mode,
        "grouped": grouped,
    })


@router.post("/api/tagesplan/log-suggestions")
async def log_tagesplan_suggestions(request: Request) -> JSONResponse:
    """Speichert Tagesplan-Slot-Vorschläge (serverseitig für Audit-Trail).

    KRITISCH: Gespeicherte ORs werden NIEMALS live überschrieben.
    Beim Speichern: individuelles predictOR(), kein Slot-Level-Fallback.
    """
    try:
        raw = await request.json()
        body = LogSuggestionsRequest(**raw)

        # Legacy-Format des frueheren HTML-Clients
        if body.date_iso and body.suggestions:
            saved = _save_suggestions_legacy_flat(body.date_iso, body.suggestions)
            return JSONResponse(content={"ok": True, "saved": saved, "format": "legacy"})

        # Neues React-Format
        if body.dateIso and body.slotHour is not None and body.suggestions:
            normalized = []
            for sug in body.suggestions:
                normalized.append(
                    {
                        "title": sug.get("title", ""),
                        "link": sug.get("link", ""),
                        "cat": sug.get("cat", sug.get("category", "")),
                        "score": sug.get("score", 0.0),
                        "expected_or": sug.get("expected_or", sug.get("predicted_or", 0.0)),
                        "best_cat": sug.get("best_cat", sug.get("cat", sug.get("category", ""))),
                    }
                )
            save_tagesplan_suggestions(
                date_iso=body.dateIso,
                slot_hour=body.slotHour,
                suggestions=normalized,
            )
            return JSONResponse(content={"ok": True, "saved": len(normalized), "format": "react"})

        raise HTTPException(
            status_code=400,
            detail="Invalid payload format for tagesplan/log-suggestions.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[tagesplan] Fehler in log_tagesplan_suggestions")
        raise HTTPException(
            status_code=500,
            detail="Daily plan suggestions could not be stored.",
        ) from exc
