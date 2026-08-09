"""Shared, non-content article identities for exact duplicate protection."""

from __future__ import annotations

import re
from urllib.parse import urlsplit


def canonical_article_url_identity(url: object) -> str:
    """Collapse harmless BILD URL aliases without retaining query/fragment data."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").casefold()
    if host == "bild.de" or host.endswith(".bild.de"):
        host = "bild.de"
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/").casefold()
    path = re.sub(r"/(?:amp|amphtml)$", "", path).rstrip("/")
    return f"{host}{path}" if host else path


def canonical_article_id(value: object) -> str:
    """Return a comparable CMS/article ID, rejecting values that are URLs."""
    raw = str(value or "").strip().casefold()
    if not raw or "://" in raw or raw.startswith("//"):
        return ""
    return raw
