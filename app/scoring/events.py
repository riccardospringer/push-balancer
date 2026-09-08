"""Automatic detection of a running major news event ("Grosslage").

Feedback of the election Sunday 2026-08-30: on an event day the politics mix
caps work against the newsroom, and election coverage barely surfaced. Rather
than relying on somebody flipping a switch in time, the balancer reads the
event out of the candidate field itself.

The detector deliberately looks for a *surge*, not for a topic: routine
coverage of a long-running subject (war, politics) must not permanently
disable the mix caps. A group only counts as active when enough candidates
share it, some of them are very fresh, and at least one carries a running-event
signal (live ticker or breaking flag).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterable

from app.scoring.freshness import parse_publication_timestamp

log = logging.getLogger("push-balancer")

# Themengruppen, die eine Grosslage bilden koennen. Sport ist bewusst NICHT
# dabei: Sportereignisse haben eigene Dynamik und waren im Feedback gerade der
# ueberrepraesentierte Teil.
EVENT_TOPIC_GROUPS: dict[str, re.Pattern[str]] = {
    "wahl": re.compile(
        r"(?i)\b(wahlabend|wahlergebnis|wahlergebnisse|wahlsieg|wahlniederlage|"
        r"wahldebakel|wahlkrimi|hochrechnung|hochrechnungen|stimmauszaehlung|"
        r"stimmauszählung|auszaehlung|auszählung|wahllokal|wahllokale|"
        r"bundestagswahl|landtagswahl|europawahl|kommunalwahl|urnengang|"
        r"regierungsbildung|koalitionsverhandlung|koalitionsverhandlungen|"
        r"erste prognose|wahlprognose)\b"
    ),
    "anschlag": re.compile(
        r"(?i)\b(anschlag|attentat|terroranschlag|amoklauf|amokfahrt|geiselnahme|"
        r"terrorverdacht|sprengsatz)\b"
    ),
    "katastrophe": re.compile(
        r"(?i)\b(erdbeben|hochwasser|jahrhundertflut|flutkatastrophe|tsunami|"
        r"grossbrand|großbrand|explosionsungl(ü|ue)ck|zugungl(ü|ue)ck|"
        r"flugzeugabsturz|naturkatastrophe|evakuierung|evakuiert)\b"
    ),
    "krieg": re.compile(
        r"(?i)\b(kriegserkl(ä|ae)rung|angriffswelle|gro(ß|ss)angriff|"
        r"raketenangriff|luftangriff|waffenruhe|feuerpause|invasion|"
        r"mobilmachung|eskalationsstufe)\b"
    ),
}

_LIVE_SIGNAL_RE = re.compile(r"(?i)live-?ticker|liveticker|live-?blog|liveblog|/ticker(/|$|\?)")

# Schwellen: bewusst so gesetzt, dass Routine-Berichterstattung sie nicht
# erreicht. Ueber Umgebungsvariablen justierbar, ohne Deploy.
DEFAULT_MIN_ARTICLES = 5
DEFAULT_MIN_FRESH_ARTICLES = 2
DEFAULT_FRESH_WINDOW_HOURS = 2.0


def _article_text(article: dict[str, Any]) -> str:
    parts = [
        str(article.get("title") or article.get("headline") or ""),
        str(article.get("url") or article.get("link") or ""),
        str(article.get("description") or ""),
    ]
    taxonomy = article.get("taxonomy") or article.get("taxonomyNodes")
    if isinstance(taxonomy, (list, tuple, set)):
        parts.extend(str(node) for node in taxonomy)
    elif isinstance(taxonomy, str):
        parts.append(taxonomy)
    return " ".join(parts)


def article_event_groups(article: dict[str, Any]) -> set[str]:
    """Themengruppen, zu denen dieser Artikel gehoert."""
    text = _article_text(article)
    return {name for name, pattern in EVENT_TOPIC_GROUPS.items() if pattern.search(text)}


def _is_breaking(article: dict[str, Any]) -> bool:
    return bool(
        article.get("isBreaking")
        or article.get("isEilmeldung")
        or article.get("is_eilmeldung")
    )


def _has_running_event_signal(article: dict[str, Any]) -> bool:
    """Laufendes Format (Ticker/Blog) oder Eilmeldung — Beleg fuer eine Lage."""
    return _is_breaking(article) or bool(_LIVE_SIGNAL_RE.search(_article_text(article)))


def _age_hours(article: dict[str, Any], now_ts: float) -> float | None:
    published = parse_publication_timestamp(article.get("pubDate"))
    if published is None:
        return None
    return max(0.0, (now_ts - published) / 3600.0)


def _thresholds() -> tuple[int, int, float]:
    try:
        from app import config

        return (
            int(config.PUSH_BALANCER_EVENT_MIN_ARTICLES),
            int(config.PUSH_BALANCER_EVENT_MIN_FRESH_ARTICLES),
            float(config.PUSH_BALANCER_EVENT_FRESH_WINDOW_HOURS),
        )
    except Exception:
        return DEFAULT_MIN_ARTICLES, DEFAULT_MIN_FRESH_ARTICLES, DEFAULT_FRESH_WINDOW_HOURS


def _forced_mode() -> str:
    """"auto" (Standard), "on" oder "off" — manuelle Uebersteuerung."""
    try:
        from app import config

        return str(config.PUSH_BALANCER_EVENT_MODE or "auto").strip().lower()
    except Exception:
        return "auto"


def detect_active_event_groups(
    candidates: Iterable[dict[str, Any]],
    *,
    now_ts: float | None = None,
) -> set[str]:
    """Themengruppen mit laufender Grosslage im aktuellen Kandidatenfeld.

    Eine Gruppe ist aktiv, wenn genug Artikel sie teilen, davon mehrere sehr
    frisch sind und mindestens einer ein laufendes Format oder eine Eilmeldung
    ist. Damit schlaegt Routine-Berichterstattung nicht an.
    """
    mode = _forced_mode()
    if mode == "off":
        return set()
    if mode == "on":
        return set(EVENT_TOPIC_GROUPS)

    now = time.time() if now_ts is None else now_ts
    min_articles, min_fresh, fresh_window = _thresholds()

    totals: dict[str, int] = {}
    fresh: dict[str, int] = {}
    running: dict[str, bool] = {}

    for article in candidates:
        groups = article_event_groups(article)
        if not groups:
            continue
        age = _age_hours(article, now)
        is_fresh = age is not None and age <= fresh_window
        has_signal = _has_running_event_signal(article)
        for group in groups:
            totals[group] = totals.get(group, 0) + 1
            if is_fresh:
                fresh[group] = fresh.get(group, 0) + 1
            if has_signal:
                running[group] = True

    active = {
        group
        for group, count in totals.items()
        if count >= min_articles
        and fresh.get(group, 0) >= min_fresh
        and running.get(group, False)
    }
    if active:
        log.info(
            "[event-mode] Grosslage erkannt: %s",
            ", ".join(
                f"{group} ({totals[group]} Artikel, {fresh.get(group, 0)} frisch)"
                for group in sorted(active)
            ),
        )
    return active


def article_matches_active_event(
    article: dict[str, Any],
    active_groups: set[str] | None,
) -> bool:
    """True, wenn der Artikel zur laufenden Grosslage gehoert."""
    if not active_groups:
        return False
    return bool(article_event_groups(article) & active_groups)
