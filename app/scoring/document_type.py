"""Authoritative article document type (article vs. video) from the public page.

The news sitemap does not say whether a document is a video: a BILD video
carries an ordinary editorial URL and ordinary keywords, which is why the
URL/title heuristics miss it. The public page, however, states the CMS
document type verbatim — ``og:type=video`` and a ``VideoObject`` block — and
that is exactly the type behind the editorial.one CMS link
(``/editor/bild/video/...``).

The probe reads only the first bytes of the page (the head), caches the result
durably per article, and fails open: an unknown type never turns an article
into a video, it just leaves the existing heuristics in charge.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from app.article_identity import canonical_article_url_identity

log = logging.getLogger("push-balancer")

# og:type steht im <head>; 48 KB decken es mit Reserve ab.
_HEAD_BYTES = 49152
_OG_TYPE_RE = re.compile(rb'og:type"?\s+content="([a-z.]+)"', re.IGNORECASE)
_VIDEO_OBJECT_RE = re.compile(rb'"@type"\s*:\s*"VideoObject"', re.IGNORECASE)

_MEMORY_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, str] = {}
_MEMORY_MAX_ENTRIES = 5000
_SCHEMA_READY = False

_PROBE_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="doc-type")

# Gleicher Zertifikats-Kontext wie der uebrige Feed-Abruf: ohne certifi
# scheitert die Verbindung auf Systemen ohne eigenes CA-Bundle.
try:
    import certifi as _certifi

    _SSL_CTX = ssl.create_default_context(cafile=_certifi.where())
except ImportError:  # pragma: no cover - certifi ist eine harte Abhaengigkeit
    _SSL_CTX = ssl.create_default_context()


def _remember(key: str, og_type: str) -> None:
    with _MEMORY_LOCK:
        _MEMORY_CACHE[key] = og_type
        overflow = len(_MEMORY_CACHE) - _MEMORY_MAX_ENTRIES
        if overflow > 0:
            for stale in list(_MEMORY_CACHE)[:overflow]:
                _MEMORY_CACHE.pop(stale, None)


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    from app.config import PUSH_DB_PATH

    conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
    if not _SCHEMA_READY:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS article_document_type (
                article_key TEXT PRIMARY KEY,
                url TEXT,
                og_type TEXT NOT NULL,
                checked_at INTEGER NOT NULL
            )"""
        )
        conn.commit()
        _SCHEMA_READY = True
    return conn


def _article_key(article: dict[str, Any]) -> str:
    url = str(article.get("url") or article.get("link") or "").strip()
    return canonical_article_url_identity(url) if url else ""


def get_cached_document_type(article: dict[str, Any]) -> str | None:
    key = _article_key(article)
    if not key:
        return None
    with _MEMORY_LOCK:
        cached = _MEMORY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT og_type FROM article_document_type WHERE article_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log.debug("[doc-type] cache read failed: %s", exc)
        return None
    if row is None:
        return None
    _remember(key, str(row[0]))
    return str(row[0])


def _store(key: str, article: dict[str, Any], og_type: str) -> None:
    _remember(key, og_type)
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO article_document_type"
                " (article_key, url, og_type, checked_at) VALUES (?, ?, ?, ?)",
                (
                    key,
                    str(article.get("url") or article.get("link") or ""),
                    og_type,
                    int(time.time()),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("[doc-type] cache write failed: %s", exc)


def probe_document_type(article: dict[str, Any]) -> str | None:
    """Read the document type off the public page head. ``None`` on failure."""
    url = str(article.get("url") or article.get("link") or "").strip()
    if not url.startswith("https://"):
        return None
    key = _article_key(article)
    if not key:
        return None

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; PushBalancer/2.0)",
            "Range": f"bytes=0-{_HEAD_BYTES - 1}",
            "Accept": "text/html",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10, context=_SSL_CTX) as response:
            head = response.read(_HEAD_BYTES)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.debug("[doc-type] probe failed for %s: %s", url[:70], exc)
        return None

    match = _OG_TYPE_RE.search(head)
    og_type = match.group(1).decode("ascii", "replace").lower() if match else ""
    if not og_type and _VIDEO_OBJECT_RE.search(head):
        og_type = "video"
    if not og_type:
        return None

    _store(key, article, og_type)
    return og_type


def _enabled() -> bool:
    try:
        from app import config

        return bool(config.PUSH_BALANCER_DOCUMENT_TYPE_PROBE_ENABLED)
    except Exception:
        return True


def annotate_document_types(
    articles: list[dict[str, Any]],
    *,
    max_new_probes: int | None = None,
    wait_budget_s: float | None = None,
) -> None:
    """Mark every article whose CMS document type is a video.

    Cached types are applied for free. A bounded number of unknown articles is
    probed now; whatever exceeds the wait budget keeps running in the
    background and lands in the durable cache for the next poll.
    """
    from app import config

    if max_new_probes is None:
        max_new_probes = int(config.PUSH_BALANCER_DOCUMENT_TYPE_MAX_PROBES_PER_REQUEST)
    if wait_budget_s is None:
        wait_budget_s = float(config.PUSH_BALANCER_DOCUMENT_TYPE_WAIT_S)

    pending: list[dict[str, Any]] = []
    for article in articles:
        cached = get_cached_document_type(article)
        if cached is not None:
            _apply(article, cached)
        elif not article.get("isVideo"):
            pending.append(article)

    if not pending or max_new_probes <= 0 or not _enabled():
        return

    batch = pending[:max_new_probes]
    futures = {_PROBE_POOL.submit(probe_document_type, article): article for article in batch}
    deadline = time.monotonic() + max(0.0, wait_budget_s)
    outstanding = set(futures)
    while outstanding:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.info(
                "[doc-type] wait budget reached; %d probe(s) continue in background",
                len(outstanding),
            )
            break
        done, outstanding = wait(outstanding, timeout=remaining, return_when=FIRST_COMPLETED)
        for future in done:
            article = futures[future]
            try:
                og_type = future.result()
            except Exception as exc:
                log.warning("[doc-type] probe worker failed: %s", exc)
                continue
            if og_type:
                _apply(article, og_type)


def _apply(article: dict[str, Any], og_type: str) -> None:
    article["documentType"] = og_type
    if og_type == "video":
        article["isVideo"] = True
        article["type"] = "video"
