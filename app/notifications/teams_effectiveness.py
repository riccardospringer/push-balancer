"""Wirkungsmessung des Teams-Kanals.

Beantwortet die Frage, die man sonst nur glauben kann: Folgt die Redaktion den
Empfehlungen - und performen befolgte Pushes besser als der Rest?

Datenbasis sind ausschliesslich bereits vorhandene, nicht-personenbezogene
Tabellen: ``teams_recommendations`` (was der Kanal empfohlen hat) und
``pushes`` (was tatsaechlich gesendet wurde).
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from typing import Any
from zoneinfo import ZoneInfo

import app.database as _db

log = logging.getLogger("push-balancer")

# Ein echter Push gilt als Umsetzung einer Empfehlung, wenn er innerhalb
# dieses Fensters nach der Empfehlung rausging.
FOLLOW_WINDOW_SECONDS = 3 * 3600


def _normalize_url(url: str) -> str:
    value = str(url or "").strip().lower()
    if not value:
        return ""
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if value.startswith("www."):
        value = value[4:]
    return value.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def _tokens(text: str) -> set[str]:
    raw = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
    return {token for token in raw.split() if len(token) >= 5}


def _same_story(recommendation: dict[str, Any], push: dict[str, Any]) -> bool:
    """Gleiche Story? Erst exakte URL, sonst deutliche Titelueberlappung."""
    reco_url = _normalize_url(recommendation.get("article_url"))
    push_url = _normalize_url(push.get("link"))
    if reco_url and push_url and reco_url == push_url:
        return True

    reco_tokens = _tokens(recommendation.get("article_title"))
    push_tokens = _tokens(push.get("title") or push.get("headline"))
    if not reco_tokens or not push_tokens:
        return False
    shared = reco_tokens & push_tokens
    if len(shared) < 2:
        return False
    return len(shared) / max(1, min(len(reco_tokens), len(push_tokens))) >= 0.6


def _load_rows(days: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = int(time.time()) - max(1, int(days)) * 86400
    conn = sqlite3.connect(_db.PUSH_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        recommendations = [
            dict(row)
            for row in conn.execute(
                """SELECT article_url, article_title, section, score, predicted_or,
                          sent_at_ts
                     FROM teams_recommendations
                    WHERE recommendation_type = 'teams_alert'
                      AND send_status = 'sent'
                      AND sent_at_ts >= ?
                 ORDER BY sent_at_ts ASC""",
                (cutoff,),
            )
        ]
        pushes = [
            dict(row)
            for row in conn.execute(
                """SELECT message_id, ts_num, or_val, title, headline, link, cat
                     FROM pushes
                    WHERE ts_num >= ?
                 ORDER BY ts_num ASC""",
                (cutoff,),
            )
        ]
    finally:
        conn.close()
    return recommendations, pushes


def _mean(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 3)


def build_effectiveness_report(
    days: int = 14,
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Annahmequote und Opening-Rate-Vergleich fuer die letzten ``days`` Tage."""
    now = int(now_ts or time.time())
    try:
        recommendations, pushes = _load_rows(days)
    except Exception as exc:  # pragma: no cover - reine Diagnose
        log.warning("[TeamsEffectiveness] Datenzugriff fehlgeschlagen: %s", exc)
        return {
            "ok": False,
            "error": type(exc).__name__,
            "days": days,
        }

    matched_push_ids: set[str] = set()
    followed: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    for recommendation in recommendations:
        sent_at = int(recommendation.get("sent_at_ts") or 0)
        if sent_at <= 0:
            continue
        match = None
        for push in pushes:
            push_ts = int(push.get("ts_num") or 0)
            if push_ts < sent_at or push_ts > sent_at + FOLLOW_WINDOW_SECONDS:
                continue
            if str(push.get("message_id")) in matched_push_ids:
                continue
            if _same_story(recommendation, push):
                match = push
                break
        if match is not None:
            matched_push_ids.add(str(match.get("message_id")))
            followed.append(
                {
                    "recommendation": recommendation,
                    "push": match,
                    "delayMinutes": round(
                        (int(match.get("ts_num") or 0) - sent_at) / 60, 1
                    ),
                }
            )
        else:
            ignored.append({"recommendation": recommendation})

    followed_or = [float(item["push"].get("or_val") or 0.0) for item in followed]
    other_or = [
        float(push.get("or_val") or 0.0)
        for push in pushes
        if str(push.get("message_id")) not in matched_push_ids
    ]

    followed_avg = _mean(followed_or)
    other_avg = _mean(other_or)
    uplift = (
        round(followed_avg - other_avg, 3)
        if followed_avg is not None and other_avg is not None
        else None
    )
    uplift_percent = (
        round(100.0 * uplift / other_avg, 1)
        if uplift is not None and other_avg not in (None, 0)
        else None
    )

    total = len(recommendations)
    acceptance_rate = round(100.0 * len(followed) / total, 1) if total else None
    berlin_now = dt.datetime.fromtimestamp(now, ZoneInfo("Europe/Berlin"))

    return {
        "ok": True,
        "days": int(days),
        "generatedAt": berlin_now.strftime("%Y-%m-%d %H:%M"),
        "followWindowMinutes": FOLLOW_WINDOW_SECONDS // 60,
        "recommendationsSent": total,
        "recommendationsFollowed": len(followed),
        "recommendationsIgnored": len(ignored),
        "acceptanceRatePercent": acceptance_rate,
        "medianDelayMinutes": (
            sorted(item["delayMinutes"] for item in followed)[len(followed) // 2]
            if followed
            else None
        ),
        "openingRate": {
            "followedAvg": followed_avg,
            "otherAvg": other_avg,
            "uplift": uplift,
            "upliftPercent": uplift_percent,
            "followedSample": len(followed_or),
            "otherSample": len(other_or),
        },
        "pushesInPeriod": len(pushes),
        "summary": _summary_sentence(
            total, len(followed), acceptance_rate, uplift, uplift_percent
        ),
    }


def _summary_sentence(
    total: int,
    followed: int,
    acceptance_rate: float | None,
    uplift: float | None,
    uplift_percent: float | None,
) -> str:
    if not total:
        return "Noch keine gesendeten Empfehlungen im Zeitraum."
    parts = [
        f"{followed} von {total} Empfehlungen wurden umgesetzt "
        f"({acceptance_rate:.0f} Prozent)."
        if acceptance_rate is not None
        else f"{followed} von {total} Empfehlungen wurden umgesetzt."
    ]
    if uplift is None:
        parts.append("Fuer einen Opening-Rate-Vergleich fehlen noch Daten.")
    elif uplift > 0:
        parts.append(
            f"Befolgte Pushes liegen im Schnitt {uplift:.2f} Punkte "
            f"({uplift_percent:.0f} Prozent) ueber den uebrigen Pushes."
        )
    elif uplift < 0:
        parts.append(
            f"Befolgte Pushes liegen im Schnitt {abs(uplift):.2f} Punkte "
            "unter den uebrigen Pushes - die Auswahl braucht eine Pruefung."
        )
    else:
        parts.append("Befolgte und uebrige Pushes liegen gleichauf.")
    return " ".join(parts)
