"""Push-Schedule Service — liefert Tagesplan ausschliesslich aus PDF-Wochenmatrix
plus den heutigen Pushes aus der SQLite-DB.

Keine ML-Calls, kein Caching, kein Background-Refresh: Endpoint ist O(1)
gegenueber der Matrix (im Speicher) + 1 SQLite-Query fuer die heutigen Pushes.
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
from statistics import mean

from app.config import PUSH_DB_PATH
from app.push_schedule.weekly_baseline import (
    PDF_BEST_HOUR,
    PDF_GOLDEN_RULES,
    PDF_HOUR_AVG,
    PDF_KPI,
    PDF_OVERALL_AVG,
    PDF_TOTAL_PUSHES_ANALYZED,
    baseline_for,
    kpi_status,
)

log = logging.getLogger("push-balancer")

WEEKDAY_NAMES_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
SLOT_HOURS = list(range(6, 24))  # 06:00 - 23:00


def _parse_date(date_str: str | None) -> datetime.date:
    if not date_str:
        return datetime.date.today()
    try:
        return datetime.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return datetime.date.today()


def _read_pushes_for_date(target: datetime.date) -> list[dict]:
    """Holt alle Pushes des angegebenen Datums aus der SQLite-DB."""
    day_start = int(datetime.datetime.combine(target, datetime.time.min).timestamp())
    day_end = int(datetime.datetime.combine(target, datetime.time.max).timestamp())
    try:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT ts_num, hour, cat, title, or_val
               FROM pushes
               WHERE ts_num BETWEEN ? AND ?
               ORDER BY ts_num""",
            (day_start, day_end),
        ).fetchall()
        conn.close()
        return [
            {
                "ts_num": int(r["ts_num"] or 0),
                "hour": int(r["hour"]) if r["hour"] is not None and r["hour"] >= 0 else None,
                "cat": (r["cat"] or "").lower(),
                "title": r["title"] or "",
                "or_val": float(r["or_val"] or 0),
            }
            for r in rows
        ]
    except Exception as exc:
        log.warning("[PushSchedule] DB-Lesefehler: %s", exc)
        return []


def _classify_slot(hour: int, is_mandatory: bool, is_avoid: bool, pushed_here: list[dict]) -> str:
    if pushed_here:
        if is_avoid:
            return "avoid_violated"
        return "pushed"
    if is_mandatory:
        return "mandatory_open"
    if is_avoid:
        return "avoid_clean"
    base = baseline_for(hour, 0) or {}
    if base.get("stars", 0) == 3:
        return "gold"
    return "normal"


def build_push_schedule(date: str | None = None) -> dict:
    target = _parse_date(date)
    weekday = target.weekday()
    now = datetime.datetime.now()
    is_today = (target == now.date())

    pushed_today = _read_pushes_for_date(target)
    pushed_hours = sorted({p["hour"] for p in pushed_today if p["hour"] is not None})
    or_vals = [p["or_val"] for p in pushed_today if p["or_val"] > 0]
    pushed_avg_or = round(mean(or_vals), 2) if or_vals else None

    mandatory = PDF_KPI["mandatory_hours"]
    avoid = PDF_KPI["avoid_hours"]

    slots = []
    for h in SLOT_HOURS:
        base = baseline_for(h, weekday) or {}
        pushed_here = [p for p in pushed_today if p["hour"] == h]
        is_mand = h in mandatory
        is_avd = h in avoid
        slots.append({
            "hour": h,
            "weekday": weekday,
            "avg_or": base.get("avg_or"),
            "count": base.get("count"),
            "top_cat": base.get("top_cat"),
            "stars": base.get("stars", 0),
            "is_mandatory": is_mand,
            "is_avoid": is_avd,
            "is_pushed": bool(pushed_here),
            "is_now": is_today and h == now.hour,
            "pushed_in_slot": [
                {"title": p["title"], "cat": p["cat"], "or_val": p["or_val"]}
                for p in pushed_here
            ],
            "status": _classify_slot(h, is_mand, is_avd, pushed_here),
        })

    kpi = kpi_status(
        pushed_hours_today=pushed_hours,
        current_avg_or=pushed_avg_or,
        n_pushed_today=len(pushed_today),
        current_hour=now.hour if is_today else 23,
    )

    return {
        "date": target.isoformat(),
        "weekday": weekday,
        "weekday_name": WEEKDAY_NAMES_DE[weekday],
        "now_hour": now.hour if is_today else None,
        "is_today": is_today,
        "slots": slots,
        "kpi": kpi,
        "baseline_total_pushes": PDF_TOTAL_PUSHES_ANALYZED,
        "baseline_overall_avg_or": PDF_OVERALL_AVG,
        "best_hour": PDF_BEST_HOUR,
        "best_hour_avg_or": PDF_HOUR_AVG.get(PDF_BEST_HOUR),
        "golden_rules": PDF_GOLDEN_RULES,
    }
