"""Push-Schedule Service — kombiniert PDF-Baseline mit Live-Pushes und
echten Artikel-Vorschlaegen aus der BILD-Sitemap.

Daten-Quellen:
- Tatsaechlich gepushte Artikel: SQLite `pushes` (heute, mit gemessener OR)
- Vorschlaege fuer leere Slots: BILD-Sitemap mit Score + ML predictedOR
- Slot-Baseline (Erwartungswerte): PDF-Wochenmatrix aus 14.916 Pushes
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
from statistics import mean
from typing import Any, Optional

from app.config import PUSH_DB_PATH
from app.push_schedule.weekly_baseline import (
    PDF_BEST_HOUR,
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


def _fetch_article_pool(limit: int = 30) -> list[dict]:
    """Holt aktuelle Artikel-Kandidaten aus der BILD-Sitemap (mit Score),
    sortiert nach Score absteigend. Robust gegen Netz-Fehler."""
    try:
        from app.routers.feed import _extract_sitemap_articles, _fetch_url
        from app.config import BILD_SITEMAP
        xml = _fetch_url(BILD_SITEMAP)
        if not xml:
            return []
        articles = _extract_sitemap_articles(xml, max_items=80)
        articles.sort(key=lambda a: (a.get("score", 0), a.get("pubDate", "")), reverse=True)
        return articles[:limit]
    except Exception as exc:
        log.warning("[PushSchedule] Artikel-Pool-Fetch fehlgeschlagen: %s", exc)
        return []


def _enrich_with_predicted_or(articles: list[dict]) -> None:
    """Reichert Artikel-Pool um predictedOR an (in-place). Best-effort, kein Hardfail."""
    if not articles:
        return
    try:
        from app.config import ARTICLE_PREDICTION_ENRICHMENT_ENABLED
        if not ARTICLE_PREDICTION_ENRICHMENT_ENABLED:
            return
        from app.ml.predict import predict_or
        from app.research.worker import _research_state
        now = datetime.datetime.now()
        for a in articles:
            if a.get("predictedOR") is not None:
                continue
            try:
                res = predict_or(
                    {
                        "title": a.get("title", ""),
                        "headline": a.get("title", ""),
                        "cat": a.get("category", ""),
                        "hour": now.hour,
                        "ts_num": int(now.timestamp()),
                        "is_eilmeldung": bool(a.get("isEilmeldung")),
                        "link": a.get("url", ""),
                        "channels": [],
                    },
                    _research_state,
                )
                p = (res or {}).get("predicted_or")
                if p is not None:
                    a["predictedOR"] = round(float(p) / 100.0, 4)
            except Exception:
                continue
    except Exception as exc:
        log.warning("[PushSchedule] predictedOR-Anreicherung fehlgeschlagen: %s", exc)


def _pick_suggestion_for_slot(
    hour: int,
    top_cat: Optional[str],
    pool: list[dict],
    used_urls: set[str],
) -> Optional[dict]:
    """Waehlt einen Artikel-Vorschlag fuer einen Slot.

    Strategie: hoechster Score-Treffer, der nicht bereits in einem frueheren
    Slot verbraucht wurde. Wenn top_cat bekannt → bevorzugt gleiche Kategorie.
    """
    if not pool:
        return None
    candidates = [a for a in pool if a.get("url") not in used_urls and a.get("title")]
    if not candidates:
        return None
    if top_cat:
        matching = [a for a in candidates if (a.get("category") or "").lower() == top_cat.lower()]
        if matching:
            return matching[0]
    return candidates[0]


def _slot_status(is_mand: bool, is_avd: bool, has_push: bool, is_past: bool, stars: int) -> str:
    if has_push:
        return "avoid_violated" if is_avd else "pushed"
    if is_mand and is_past:
        return "mandatory_missed"
    if is_mand:
        return "mandatory_open"
    if is_avd:
        return "avoid_clean"
    if stars == 3:
        return "gold"
    return "normal"


def _format_push(p: dict) -> dict:
    return {
        "title": p.get("title", ""),
        "cat": p.get("cat", ""),
        "or_val": round(p.get("or_val", 0), 2) if p.get("or_val") else None,
    }


def _format_suggestion(a: dict) -> dict:
    pred = a.get("predictedOR")
    pred_pct = round(pred * 100, 2) if isinstance(pred, (int, float)) else None
    return {
        "title": a.get("title", ""),
        "cat": a.get("category", ""),
        "score": int(round(a.get("score", 0))),
        "predicted_or": pred_pct,
        "url": a.get("url", ""),
        "is_eilmeldung": bool(a.get("isEilmeldung")),
    }


def build_push_schedule(date: str | None = None) -> dict:
    target = _parse_date(date)
    weekday = target.weekday()
    now = datetime.datetime.now()
    is_today = (target == now.date())

    pushed_today = _read_pushes_for_date(target)
    pushed_hours = sorted({p["hour"] for p in pushed_today if p["hour"] is not None})
    or_vals = [p["or_val"] for p in pushed_today if p["or_val"] > 0]
    pushed_avg_or = round(mean(or_vals), 2) if or_vals else None

    pool: list[dict] = []
    if is_today:
        pool = _fetch_article_pool(limit=30)
        _enrich_with_predicted_or(pool)

    mandatory = PDF_KPI["mandatory_hours"]
    avoid = PDF_KPI["avoid_hours"]
    used_urls: set[str] = set()

    slots = []
    for h in SLOT_HOURS:
        base = baseline_for(h, weekday) or {}
        stars = int(base.get("stars", 0) or 0)
        pushed_here = [p for p in pushed_today if p["hour"] == h]
        is_mand = h in mandatory
        is_avd = h in avoid
        is_past = is_today and h < now.hour
        is_now = is_today and h == now.hour

        suggestion = None
        if is_today and not pushed_here and not is_past and not is_avd:
            suggestion = _pick_suggestion_for_slot(h, base.get("top_cat"), pool, used_urls)
            if suggestion:
                used_urls.add(suggestion.get("url", ""))

        slots.append({
            "hour": h,
            "weekday": weekday,
            "expected_or": base.get("avg_or"),
            "top_cat": base.get("top_cat"),
            "stars": stars,
            "is_mandatory": is_mand,
            "is_avoid": is_avd,
            "is_pushed": bool(pushed_here),
            "is_now": is_now,
            "is_past": is_past,
            "pushed": _format_push(pushed_here[0]) if pushed_here else None,
            "pushed_extras": [_format_push(p) for p in pushed_here[1:]] if len(pushed_here) > 1 else [],
            "suggestion": _format_suggestion(suggestion) if suggestion else None,
            "status": _slot_status(is_mand, is_avd, bool(pushed_here), is_past, stars),
        })

    kpi = kpi_status(
        pushed_hours_today=pushed_hours,
        current_avg_or=pushed_avg_or,
        n_pushed_today=len(pushed_today),
        current_hour=now.hour if is_today else 23,
    )

    next_mandatory = next(
        (h for h in mandatory if h not in pushed_hours and (not is_today or h >= now.hour)),
        None,
    )
    next_gold = None
    if is_today:
        next_gold = next(
            (h for h in SLOT_HOURS
             if h >= now.hour
             and h not in pushed_hours
             and (baseline_for(h, weekday) or {}).get("stars", 0) == 3),
            None,
        )

    return {
        "date": target.isoformat(),
        "weekday": weekday,
        "weekday_name": WEEKDAY_NAMES_DE[weekday],
        "now_hour": now.hour if is_today else None,
        "is_today": is_today,
        "slots": slots,
        "kpi": kpi,
        "next_mandatory_hour": next_mandatory,
        "next_gold_hour": next_gold,
        "baseline_overall_avg_or": PDF_OVERALL_AVG,
        "best_hour": PDF_BEST_HOUR,
        "best_hour_avg_or": PDF_HOUR_AVG.get(PDF_BEST_HOUR),
        "_meta": {
            "baseline_n": PDF_TOTAL_PUSHES_ANALYZED,
            "pool_size": len(pool),
        },
    }
