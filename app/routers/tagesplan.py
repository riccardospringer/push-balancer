"""app/routers/tagesplan.py — Tagesplan-Endpunkte.

GET  /api/tagesplan                   — ML-Tagesplan
GET  /api/tagesplan/retro             — Retro-Analyse heutiger Slots
GET  /api/tagesplan/history           — Historische Tagesplan-Daten
GET  /api/tagesplan/suggestions       — Gespeicherte Slot-Vorschläge
POST /api/tagesplan/log-suggestions   — Speichert Slot-Vorschläge
"""
import datetime
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


def _fallback_plan(date: str | None = None, mode: str = "redaktion") -> dict[str, Any]:
    """Stabiler Minimal-Tagesplan für Fehlerfälle."""
    today_label = date or datetime.datetime.now().strftime("%d.%m.%Y")
    slots = [
        {
            "hour": hour,
            "best_cat": "sport" if mode == "sport" else "news",
            "expected_or": 0.0,
            "hist_or": None,
            "n_historical": 0,
            "confidence": "niedrig",
            "mood": "",
            "mood_reasons": [],
            "color": "gray",
            "is_now": hour == datetime.datetime.now().hour,
            "is_past": hour < datetime.datetime.now().hour,
            "shap": {},
            "shap_explanation": "",
            "has_ml": False,
            "best_historical": [],
            "pushed_this_hour": [],
            "must_have": False,
        }
        for hour in range(6, 24)
    ]
    return {
        "date": today_label,
        "slots": slots,
        "loading": True,
        "mode": mode,
        "golden_hour": None,
        "golden_cat": "sport" if mode == "sport" else "news",
        "golden_or": 0.0,
        "best_hour": None,
        "best_cat": "sport" if mode == "sport" else "news",
        "best_or": 0.0,
        "ml_metrics": {},
        "ml_trained": False,
        "avg_or_today": None,
        "total_pushes_db": 0,
        "already_pushed_today": [],
        "n_pushed_today": 0,
        "must_have_hours": [],
    }


def _normalize_suggestion_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Bringt DB-Zeilen in eine stabile API-Form für Legacy- und React-Clients."""
    items: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        hour = int(row.get("slot_hour") or row.get("hour") or 0)
        item = {
            "id": row.get("id"),
            "date_iso": row.get("date_iso"),
            "hour": hour,
            "slot_hour": hour,
            "suggestion_num": row.get("suggestion_num", 0),
            "title": row.get("article_title") or row.get("title") or "",
            "url": row.get("article_link") or row.get("url") or "",
            "link": row.get("article_link") or row.get("link") or "",
            "category": row.get("article_category") or row.get("category") or row.get("cat") or "",
            "cat": row.get("article_category") or row.get("category") or row.get("cat") or "",
            "score": float(row.get("article_score") or row.get("score") or 0.0),
            "predictedOR": float(row.get("expected_or") or row.get("predictedOR") or 0.0),
            "expected_or": float(row.get("expected_or") or row.get("predictedOR") or 0.0),
            "best_cat": row.get("best_cat") or "",
            "captured_at": row.get("captured_at"),
        }
        items.append(item)
        grouped.setdefault(str(hour), []).append(item)
    return items, grouped


def _build_history_plan(date_iso: str, mode: str = "redaktion") -> dict[str, Any]:
    """Liefert einen schlanken, stabilen Rückblick-Tagesplan für ein Datum."""
    try:
        suggestions_raw = load_tagesplan_suggestions(date_iso=date_iso)
    except Exception:
        log.exception("[tagesplan] Konnte Verlaufsvorschlaege nicht laden")
        suggestions_raw = []
    suggestion_items, suggestion_map = _normalize_suggestion_rows(suggestions_raw)

    slots_by_hour: dict[int, dict[str, Any]] = {
        hour: {
            "hour": hour,
            "best_cat": "sport" if mode == "sport" else "news",
            "expected_or": 0.0,
            "hist_or": None,
            "n_historical": 0,
            "confidence": "niedrig",
            "mood": "",
            "mood_reasons": [],
            "color": "gray",
            "is_now": False,
            "is_past": True,
            "shap": {},
            "shap_explanation": "",
            "has_ml": False,
            "best_historical": [],
            "pushed_this_hour": [],
            "must_have": False,
        }
        for hour in range(6, 24)
    }

    for hour_str, bucket in suggestion_map.items():
        if not bucket:
            continue
        hour = int(hour_str)
        slot = slots_by_hour.get(hour)
        if not slot:
            continue
        slot["best_cat"] = bucket[0].get("best_cat") or bucket[0].get("category") or slot["best_cat"]
        slot["expected_or"] = max(float(item.get("expected_or") or 0.0) for item in bucket)

    rows: list[sqlite3.Row] = []
    try:
        with sqlite3.connect(PUSH_DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT
                    hour,
                    title,
                    LOWER(TRIM(cat)) AS cat,
                    or_val,
                    link,
                    is_eilmeldung
                FROM pushes
                WHERE date(ts_num, 'unixepoch', 'localtime') = ?
            """
            params: list[Any] = [date_iso]
            if mode == "sport":
                query += """
                  AND LOWER(TRIM(cat)) IN
                    ('sport','fussball','bundesliga','formel1','formel-1','tennis','boxen','motorsport')
                """
            query += " ORDER BY hour, ts_num"
            rows = conn.execute(query, params).fetchall()
    except Exception:
        log.exception("[tagesplan] Konnte Verlaufspushes nicht laden")
        rows = []

    pushed_today: list[dict[str, Any]] = []
    weekday_label = date_iso
    try:
        dt = datetime.datetime.strptime(date_iso, "%Y-%m-%d")
        weekday_label = f"{['Mo','Di','Mi','Do','Fr','Sa','So'][dt.weekday()]}, {dt.strftime('%d.%m.%Y')}"
    except Exception:
        pass

    for row in rows:
        hour = int(row["hour"] if row["hour"] is not None else -1)
        push_item = {
            "hour": hour,
            "title": row["title"] or "",
            "cat": row["cat"] or "news",
            "or": round(float(row["or_val"] or 0.0), 2),
            "actual_or": round(float(row["or_val"] or 0.0), 2),
            "link": row["link"] or "",
            "is_eilmeldung": bool(row["is_eilmeldung"]),
        }
        pushed_today.append(push_item)
        if hour in slots_by_hour:
            slots_by_hour[hour]["pushed_this_hour"].append(push_item)
            actual_or = push_item["actual_or"]
            if actual_or >= 5.5:
                slots_by_hour[hour]["color"] = "green"
            elif actual_or >= 3.0:
                slots_by_hour[hour]["color"] = "yellow"

    return {
        "date": weekday_label,
        "date_iso": date_iso,
        "weekday": weekday_label.split(",")[0],
        "is_history": True,
        "loading": False,
        "mode": mode,
        "slots": [slots_by_hour[hour] for hour in range(6, 24)],
        "already_pushed_today": pushed_today,
        "n_pushed_today": len(pushed_today),
        "golden_hour": None,
        "golden_cat": "sport" if mode == "sport" else "news",
        "golden_or": 0.0,
        "best_hour": None,
        "best_cat": "sport" if mode == "sport" else "news",
        "best_or": 0.0,
        "ml_metrics": {},
        "ml_trained": False,
        "avg_or_today": None,
        "total_pushes_db": len(pushed_today),
        "must_have_hours": [],
        "suggestions": suggestion_map,
        "grouped": suggestion_map,
        "items": suggestion_items,
    }


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
            plan = _fallback_plan(mode=mode)
        if date:
            plan["requestedDate"] = date
        return JSONResponse(content=plan)
    except Exception as exc:
        log.exception("[tagesplan] Fehler in get_tagesplan")
        fallback = _fallback_plan(date=date, mode=mode)
        fallback["error"] = "Daily plan could not be loaded."
        return JSONResponse(content=fallback)


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
        return JSONResponse(
            content={
                "days": [],
                "summary": {
                    "total_pushes": 0,
                    "avg_or_7d": 0,
                    "best_day": None,
                    "worst_day": None,
                    "prediction_mae_7d": None,
                    "top_hour": None,
                    "top_hour_avg_or": 0,
                    "category_breakdown": {},
                },
                "mode": mode,
                "loading": True,
                "error": "Daily plan retrospective data could not be loaded.",
            }
        )


@router.get("/api/tagesplan/history")
def get_tagesplan_history(
    date: str | None = Query(default=None),
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
        if date:
            return JSONResponse(content=_build_history_plan(date, mode="redaktion"))
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
    try:
        all_rows = load_tagesplan_suggestions(date_iso=date)
        normalized_rows, grouped_all = _normalize_suggestion_rows(all_rows)
        total = len(normalized_rows)
        items = normalized_rows[offset: offset + limit]
        grouped: dict[str, list[dict[str, Any]]] = {}
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
            "suggestions": grouped,
            "loading": False,
        })
    except Exception as exc:
        log.exception("[tagesplan] Fehler in get_tagesplan_suggestions")
        return JSONResponse(content={
            "items": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "mode": mode,
            "grouped": {},
            "suggestions": {},
            "loading": True,
            "error": "Daily plan suggestions could not be loaded.",
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
