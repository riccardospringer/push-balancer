"""Resolve one CMS document ID to its canonical public article URL."""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app import config

log = logging.getLogger("push-balancer")

_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 8
_ALLOWED_URL_API_HOSTS = frozenset({"api.stg.editorial.one", "api.editorial.one"})
_URL_API_PATH = "/urlapi/v2"

try:
    import certifi as _certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()


class UrlApiNotConfigured(RuntimeError):
    """Raised when the CMS mapping integration is disabled."""


class UrlApiUnavailable(RuntimeError):
    """Raised when the CMS mapping integration cannot return a valid response."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the UrlServer credential to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_request(request: urllib.request.Request):
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
    )
    return opener.open(request, timeout=_TIMEOUT_SECONDS)


def _normalize_public_url(path: str) -> str | None:
    public_base = urllib.parse.urlsplit(config.PUBLIC_ARTICLE_BASE_URL)
    if (
        public_base.scheme.lower() != "https"
        or not public_base.netloc
        or public_base.username is not None
        or public_base.password is not None
    ):
        return None
    resolved = urllib.parse.urlsplit(
        urllib.parse.urljoin(f"{config.PUBLIC_ARTICLE_BASE_URL}/", path.strip())
    )
    if (
        resolved.scheme.lower() != public_base.scheme.lower()
        or resolved.netloc.lower() != public_base.netloc.lower()
    ):
        return None
    normalized_path = resolved.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit(
        (resolved.scheme.lower(), resolved.netloc.lower(), normalized_path, "", "")
    )


def _canonical_url_from_payload(payload: Any, cms_id: str) -> str | None:
    if not isinstance(payload, list):
        raise UrlApiUnavailable("UrlServer returned an invalid response")

    for item in payload:
        if not isinstance(item, dict) or str(item.get("documentId", "")).strip() != cms_id:
            continue
        urls = item.get("urls")
        if not isinstance(urls, list):
            continue
        for candidate in urls:
            if not isinstance(candidate, dict) or not candidate.get("isCanonicalUrl"):
                continue
            path = candidate.get("path")
            if isinstance(path, str) and path.strip():
                return _normalize_public_url(path)
    return None


def get_canonical_article_url(cms_id: str) -> str | None:
    """Resolve exactly one CMS ID without logging or retaining that identifier."""
    if not config.URL_API_BASE or not config.URL_API_KEY:
        raise UrlApiNotConfigured("UrlServer integration is disabled")

    parsed_base = urllib.parse.urlsplit(config.URL_API_BASE)
    if (
        parsed_base.scheme != "https"
        or parsed_base.hostname not in _ALLOWED_URL_API_HOSTS
        or parsed_base.port not in (None, 443)
        or parsed_base.username is not None
        or parsed_base.password is not None
        or parsed_base.path.rstrip("/") != _URL_API_PATH
        or parsed_base.query
        or parsed_base.fragment
    ):
        raise UrlApiNotConfigured("UrlServer integration is disabled")

    query = urllib.parse.urlencode({"documentIds": cms_id})
    request = urllib.request.Request(
        f"{config.URL_API_BASE}/tenants/bild/document-urls?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "NextPushBalancer/1.0",
            "x-api-key": config.URL_API_KEY,
        },
        method="GET",
    )

    try:
        with _open_request(request) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        log.warning("UrlServer HTTP request failed with status %s", exc.code)
        raise UrlApiUnavailable("UrlServer request failed") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("UrlServer request failed (%s)", type(exc).__name__)
        raise UrlApiUnavailable("UrlServer request failed") from exc

    if len(raw) > _MAX_RESPONSE_BYTES:
        raise UrlApiUnavailable("UrlServer returned an invalid response")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UrlApiUnavailable("UrlServer returned an invalid response") from exc

    return _canonical_url_from_payload(payload, cms_id)
