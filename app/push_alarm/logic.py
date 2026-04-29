"""
Push-Alarm Kernlogik.

Entscheidet anhand von historischen Push-Daten, Tagesplan-State und
aktuellem Artikel-Feed ob jetzt ein Push empfohlen werden soll.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("push-balancer")

# ── Konfiguration ─────────────────────────────────────────────────────────────

MAX_PUSHES_TODAY = 18
MIN_INTERVAL_SECS = 20 * 60        # 20 Min. globaler Mindestabstand

CAT_COOLDOWN: dict[str, int] = {
    "sport":        45 * 60,
    "politik":      25 * 60,
    "news":         20 * 60,
    "unterhaltung": 35 * 60,
    "wirtschaft":   35 * 60,
    "regional":     40 * 60,
}
_DEFAULT_COOLDOWN = 25 * 60

MIN_SCORE_DEFAULT     = 74.0
MIN_SCORE_GOLDEN_HOUR = 68.0
MIN_SCORE_BREAKING    = 0.0        # Breaking überschreibt Score-Check


# ── Ergebnis-Datenklasse ──────────────────────────────────────────────────────

@dataclass
class AlarmRecommendation:
    title: str
    url: str
    score: float
    predicted_or: float | None
    category: str
    is_breaking: bool
    is_eilmeldung: bool
    reason: str
    pushes_today: int
    mins_since_last_push: int | None
    golden_hour: bool
    expected_or_now: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title":             self.title,
            "url":               self.url,
            "score":             round(self.score, 1),
            "predictedOR":       round(self.predicted_or * 100, 2) if self.predicted_or else None,
            "category":          self.category,
            "isBreaking":        self.is_breaking,
            "isEilmeldung":      self.is_eilmeldung,
            "reason":            self.reason,
            "pushesToday":       self.pushes_today,
            "minsSinceLastPush": self.mins_since_last_push,
            "goldenHour":        self.golden_hour,
            "expectedORNow":     round(self.expected_or_now, 2) if self.expected_or_now else None,
        }


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _today_start_ts() -> int:
    import datetime
    now = datetime.datetime.now()
    return int(datetime.datetime(now.year, now.month, now.day).timestamp())


def _load_today_pushes(db_path: str) -> list[dict]:
    today_start = _today_start_ts()
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT ts_num, LOWER(TRIM(cat)) AS cat, title, link, is_eilmeldung
               FROM pushes
               WHERE ts_num >= ?
                 AND link NOT LIKE '%sportbild.%'
                 AND link NOT LIKE '%autobild.%'
               ORDER BY ts_num""",
            (today_start,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("[PushAlarm] DB-Lesefehler: %s", exc)
        return []


def _is_golden_hour(current_hour: int, tagesplan_state: dict | None) -> tuple[bool, float | None]:
    if not tagesplan_state:
        return False, None
    golden = tagesplan_state.get("golden_hour")
    slots = tagesplan_state.get("slots", [])
    current_slot = next((s for s in slots if s.get("hour") == current_hour), None)
    expected_or = current_slot.get("expected_or") if current_slot else None
    is_golden = (golden == current_hour) if golden is not None else False
    return is_golden, expected_or


def _build_reason(
    article: dict,
    is_golden: bool,
    mins_since_last: int | None,
    pushes_today: int,
    expected_or: float | None,
) -> str:
    parts: list[str] = []

    if article.get("isBreaking") or article.get("isEilmeldung"):
        parts.append("Breaking — sofort pushen!")
    elif is_golden:
        parts.append("Golden Hour")

    score = article.get("score", 0)
    parts.append(f"Score {score:.0f}")

    por = article.get("predictedOR")
    if por:
        parts.append(f"XOR {por * 100:.1f}%")
    elif expected_or:
        parts.append(f"Stunden-OR {expected_or:.1f}%")

    cat = article.get("category", "")
    if cat:
        parts.append(cat.capitalize())

    if mins_since_last is not None:
        parts.append(f"{mins_since_last} Min. seit letztem Push")

    parts.append(f"{pushes_today} Pushes heute")
    return " · ".join(parts)


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def check_push_alarm(
    articles: list[dict],
    db_path: str,
    tagesplan_state: dict | None = None,
) -> AlarmRecommendation | None:
    """
    Gibt eine AlarmRecommendation zurück wenn jetzt gepusht werden sollte,
    sonst None.
    """
    import datetime
    now = time.time()
    current_hour = datetime.datetime.now().hour

    today_pushes = _load_today_pushes(db_path)

    # ── Hard limit: max. Pushes pro Tag ───────────────────────────────────────
    if len(today_pushes) >= MAX_PUSHES_TODAY:
        return None

    # ── Zeitabstand seit letztem Push berechnen ────────────────────────────────
    last_push_ts = max((p["ts_num"] for p in today_pushes), default=None)
    secs_since_last = (now - last_push_ts) if last_push_ts else None
    mins_since_last = int(secs_since_last / 60) if secs_since_last is not None else None

    # ── Kategorie → letzter Push-Zeitstempel ──────────────────────────────────
    last_push_by_cat: dict[str, float] = {}
    for p in today_pushes:
        cat = p.get("cat", "news")
        ts = p.get("ts_num", 0)
        if cat not in last_push_by_cat or ts > last_push_by_cat[cat]:
            last_push_by_cat[cat] = float(ts)

    # ── Golden Hour prüfen ────────────────────────────────────────────────────
    is_golden, expected_or = _is_golden_hour(current_hour, tagesplan_state)
    min_score = MIN_SCORE_GOLDEN_HOUR if is_golden else MIN_SCORE_DEFAULT

    # ── Artikel-Kandidaten prüfen (sortiert nach score desc) ─────────────────
    sorted_articles = sorted(articles, key=lambda a: a.get("score", 0), reverse=True)

    for article in sorted_articles:
        is_breaking  = bool(article.get("isBreaking") or article.get("isEilmeldung"))
        cat          = (article.get("category") or "news").lower().strip()
        score        = article.get("score", 0)

        # Breaking überspringt alle Abstands-Checks
        if is_breaking:
            reason = _build_reason(article, is_golden, mins_since_last,
                                   len(today_pushes), expected_or)
            return AlarmRecommendation(
                title=article.get("title", ""),
                url=article.get("url", ""),
                score=score,
                predicted_or=article.get("predictedOR"),
                category=cat,
                is_breaking=True,
                is_eilmeldung=bool(article.get("isEilmeldung")),
                reason=reason,
                pushes_today=len(today_pushes),
                mins_since_last_push=mins_since_last,
                golden_hour=is_golden,
                expected_or_now=expected_or,
            )

        # Globaler Mindestabstand
        if secs_since_last is not None and secs_since_last < MIN_INTERVAL_SECS:
            continue

        # Kategorie-Cooldown
        cat_last = last_push_by_cat.get(cat)
        if cat_last is not None:
            cooldown = CAT_COOLDOWN.get(cat, _DEFAULT_COOLDOWN)
            if (now - cat_last) < cooldown:
                continue

        # Score-Schwelle
        if score < min_score:
            continue

        reason = _build_reason(article, is_golden, mins_since_last,
                               len(today_pushes), expected_or)
        return AlarmRecommendation(
            title=article.get("title", ""),
            url=article.get("url", ""),
            score=score,
            predicted_or=article.get("predictedOR"),
            category=cat,
            is_breaking=False,
            is_eilmeldung=False,
            reason=reason,
            pushes_today=len(today_pushes),
            mins_since_last_push=mins_since_last,
            golden_hour=is_golden,
            expected_or_now=expected_or,
        )

    return None
