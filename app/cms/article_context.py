"""Load minimal public article context for one already-resolved BILD URL."""

from __future__ import annotations

import html
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

try:
    import certifi as _certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_TIMEOUT_SECONDS = 8
_ALLOWED_HOSTS = frozenset({"bild.de", "www.bild.de"})


class _ArticleHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self._in_json_ld = False
        self._json_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {str(key).casefold(): value or "" for key, value in attrs}
        if tag.casefold() == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content", "").strip()
            if key and content:
                self.meta[key] = content
        if tag.casefold() == "script" and values.get("type", "").casefold() == "application/ld+json":
            self._in_json_ld = True
            self._json_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._in_json_ld:
            self.json_ld.append("".join(self._json_parts))
            self._in_json_ld = False
            self._json_parts = []


class _SameSiteRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in _ALLOWED_HOSTS:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _clean(value: Any, *, limit: int) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].strip()


def _iter_json_objects(value: Any):
    if isinstance(value, dict):
        yield value
        graph = value.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _iter_json_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_objects(item)


def fetch_public_article_context(url: str) -> dict[str, str] | None:
    """Return title/text/category from a public BILD article without retaining it."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() not in _ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        return None

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "NextPushBalancer/1.0",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(
        _SameSiteRedirectHandler(),
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
    )
    try:
        with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    if len(raw) > _MAX_RESPONSE_BYTES:
        return None

    parser = _ArticleHtmlParser()
    try:
        parser.feed(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

    title = _clean(parser.meta.get("og:title"), limit=300)
    text = _clean(
        parser.meta.get("description") or parser.meta.get("og:description"),
        limit=4000,
    )
    for block in parser.json_ld:
        try:
            decoded = json.loads(block)
        except (TypeError, json.JSONDecodeError):
            continue
        for item in _iter_json_objects(decoded):
            item_type = str(item.get("@type") or "").casefold()
            if item_type not in {"article", "newsarticle", "reportagenewsarticle"}:
                continue
            title = _clean(item.get("headline") or title, limit=300)
            text = _clean(item.get("articleBody") or item.get("description") or text, limit=4000)
            break
        if title and text:
            break

    title = re.sub(r"\s*[|–-]\s*BILD\s*$", "", title, flags=re.IGNORECASE).strip()
    if not title:
        return None
    category = next((part for part in parsed.path.split("/") if part), "news")
    return {"url": url, "title": title, "text": text, "category": category}
