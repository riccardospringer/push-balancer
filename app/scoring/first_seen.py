"""Durable first-sighting store for article candidates.

The BILD news sitemap only carries the *current* publication timestamp. When an
article is re-published ("ausgeliehen"), it therefore looks brand new even
though the story is hours or days old. This module remembers when the Push
Balancer first saw an article and exposes the earliest known timestamp, so the
shared freshness policy can treat a re-publish as what it is: an old article.

The store is intentionally conservative — it only ever lowers a publication
timestamp, never raises it.
"""

from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
import threading
import time
from typing import Any

from app.article_identity import canonical_article_url_identity
from app.scoring.freshness import parse_publication_timestamp

log = logging.getLogger("push-balancer")

_MEMORY_LOCK = threading.Lock()
_MEMORY_FIRST_SEEN: dict[str, int] = {}
_SCHEMA_READY = False


def article_key(article: dict[str, Any]) -> str:
    """Stable identity for an article candidate."""
    url = str(article.get("url") or article.get("link") or article.get("id") or "").strip()
    if url:
        return canonical_article_url_identity(url)
    title = str(article.get("title") or article.get("headline") or "").strip().lower()
    return f"title:{title}" if title else ""


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    from app.config import PUSH_DB_PATH

    conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
    if not _SCHEMA_READY:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS article_first_seen (
                article_key TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                first_seen_ts INTEGER NOT NULL,
                first_published_ts INTEGER
            )"""
        )
        conn.commit()
        _SCHEMA_READY = True
    return conn


def _load(key: str) -> int | None:
    with _MEMORY_LOCK:
        cached = _MEMORY_FIRST_SEEN.get(key)
    if cached is not None:
        return cached
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT first_seen_ts, first_published_ts FROM article_first_seen"
                " WHERE article_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log.debug("[first-seen] read failed: %s", exc)
        return None
    if row is None:
        return None
    candidates = [int(value) for value in row if value]
    if not candidates:
        return None
    earliest = min(candidates)
    with _MEMORY_LOCK:
        _MEMORY_FIRST_SEEN[key] = earliest
    return earliest


def _store(key: str, article: dict[str, Any], first_seen_ts: int, published_ts: int | None) -> None:
    with _MEMORY_LOCK:
        _MEMORY_FIRST_SEEN[key] = first_seen_ts
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO article_first_seen"
                " (article_key, url, title, first_seen_ts, first_published_ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    str(article.get("url") or article.get("link") or ""),
                    str(article.get("title") or "")[:300],
                    first_seen_ts,
                    published_ts,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("[first-seen] write failed: %s", exc)


def effective_publication_ts(
    article: dict[str, Any],
    *,
    published_ts: int | None,
    now_ts: int | float | None = None,
) -> int | None:
    """Return the earliest known publication time for this article.

    Records the sighting on first contact. The returned value is never later
    than ``published_ts``: a re-publish can only keep or lower the age, never
    reset it.
    """
    key = article_key(article)
    if not key:
        return published_ts

    now = int(time.time() if now_ts is None else now_ts)
    known = _load(key)

    # A first sighting without a usable publication timestamp still anchors the
    # article at "seen now", so a later re-publish cannot make it look fresher.
    observed = [value for value in (published_ts, known) if value]
    earliest = min(min(observed), now) if observed else now
    if known is None or earliest < known:
        _store(key, article, earliest, published_ts)
    return earliest


def apply_first_seen_publication_floor(
    articles: list[dict[str, Any]],
    *,
    now_ts: int | float | None = None,
) -> list[dict[str, Any]]:
    """Rewrite each article's ``pubDate`` to the earliest known publication time.

    Downstream freshness gates, age multipliers and editorial scoring then all
    see the real age of a re-published article without further changes.
    """
    for article in articles:
        raw_pub = article.get("pubDate")
        published_ts = parse_publication_timestamp(raw_pub)
        earliest = effective_publication_ts(
            article,
            published_ts=published_ts,
            now_ts=now_ts,
        )
        if earliest is None or (published_ts is not None and earliest >= published_ts):
            continue
        article["republishDetected"] = True
        article["sitemapPubDate"] = raw_pub
        article["pubDate"] = (
            _dt.datetime.fromtimestamp(earliest, tz=_dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return articles
