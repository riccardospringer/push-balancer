"""Fail-closed consumer for the internal Push Balancer score API.

Only CMS IDs are transmitted. The API key stays in the request header and is
never included in URLs, logs, exceptions, diagnostics, or Teams payloads.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


_CMS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_URL_DOCUMENT_ID_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{24})(?![0-9a-fA-F])")
_MAX_RESPONSE_BYTES = 64 * 1024
_CACHE_MAX_ITEMS = 512

try:
    import certifi as _certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=_certifi.where())
except ImportError:  # pragma: no cover - deployment installs certifi
    _SSL_CONTEXT = ssl.create_default_context()


class ScoreApiError(RuntimeError):
    """Base class for sanitized score API failures."""


class ScoreApiConfigurationError(ScoreApiError):
    """Missing/invalid configuration or rejected credentials."""


class ScoreApiUnavailable(ScoreApiError):
    """The API could not provide a trustworthy response."""


@dataclass(frozen=True)
class ArticleScore:
    cms_id: str
    score: float
    scored_at: datetime

    def age_seconds(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        return (reference.astimezone(timezone.utc) - self.scored_at).total_seconds()


@dataclass(frozen=True)
class ScoreLookup:
    status: str
    value: ArticleScore | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the score credential to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


Transport = Callable[[str, dict[str, str], float], tuple[int, bytes]]


def _validated_base_url(raw: str) -> str:
    parsed = urllib.parse.urlsplit((raw or "").strip())
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ScoreApiConfigurationError("Score API base URL is invalid")
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit(("https", parsed.netloc, path, "", ""))


def _parse_scored_at(raw: object) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_cms_id(article: dict) -> str | None:
    """Resolve a canonical field first, then a strict URL-embedded document ID."""
    for field_name in ("cmsId", "documentId", "articleId", "id"):
        value = article.get(field_name)
        if value is None:
            continue
        candidate = str(value).strip()
        if _CMS_ID_RE.fullmatch(candidate):
            return candidate.lower() if len(candidate) == 24 else candidate

    url = str(article.get("url") or article.get("link") or "")
    match = _URL_DOCUMENT_ID_RE.search(urllib.parse.urlsplit(url).path)
    return match.group(1).lower() if match else None


class ScoreApiClient:
    """Small bounded client with strict response validation and short caching."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 2.5,
        cache_ttl_seconds: float = 45.0,
        max_retries: int = 1,
        transport: Transport | None = None,
    ) -> None:
        if not (api_key or "").strip():
            raise ScoreApiConfigurationError("Score API key is missing")
        self._base_url = _validated_base_url(base_url)
        self._api_key = api_key.strip()
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._max_retries = max(0, int(max_retries))
        self._transport = transport or self._urllib_transport
        self._cache: dict[str, tuple[float, ArticleScore | None]] = {}
        self._cache_lock = threading.Lock()

    def get_score(self, cms_id: str) -> ArticleScore | None:
        """Return a validated score; ``None`` has the exact meaning HTTP 404."""
        if not _CMS_ID_RE.fullmatch(cms_id or ""):
            raise ScoreApiUnavailable("CMS ID is invalid")

        cached = self._cache_get(cms_id)
        if cached is not _CACHE_MISS:
            return cached

        safe_id = urllib.parse.quote(cms_id, safe="")
        url = f"{self._base_url}/api/v1/scores/{safe_id}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "PushBalancer-Teams/1.0",
            "X-Score-Key": self._api_key,
        }
        last_error: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                status, body = self._transport(url, headers, self._timeout_seconds)
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                last_error = exc
                continue

            if status == 200:
                result = self._parse_success(cms_id, body)
                self._cache_put(cms_id, result)
                return result
            if status == 404:
                self._cache_put(cms_id, None)
                return None
            if status in (401, 403):
                raise ScoreApiConfigurationError(f"Score API authorization failed (HTTP {status})")
            if 500 <= status <= 599:
                last_error = ScoreApiUnavailable(f"Score API returned HTTP {status}")
                continue
            raise ScoreApiUnavailable(f"Score API returned unexpected HTTP {status}")

        raise ScoreApiUnavailable("Score API is unavailable after bounded retry") from last_error

    def _parse_success(self, cms_id: str, body: bytes) -> ArticleScore:
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ScoreApiUnavailable("Score API response is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoreApiUnavailable("Score API returned invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"cmsId", "score", "scoredAt"}:
            raise ScoreApiUnavailable("Score API response contract is invalid")
        if str(payload.get("cmsId") or "").strip() != cms_id:
            raise ScoreApiUnavailable("Score API returned a different CMS ID")

        raw_score = payload.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ScoreApiUnavailable("Score API score is invalid")
        score = float(raw_score)
        if not 0.0 <= score <= 100.0:
            raise ScoreApiUnavailable("Score API score is outside 0..100")
        try:
            scored_at = _parse_scored_at(payload.get("scoredAt"))
        except (TypeError, ValueError) as exc:
            raise ScoreApiUnavailable("Score API timestamp is invalid") from exc
        return ArticleScore(cms_id=cms_id, score=round(score, 2), scored_at=scored_at)

    @staticmethod
    def _urllib_transport(
        url: str,
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, headers=headers, method="GET")
        opener = urllib.request.build_opener(
            _NoRedirectHandler(),
            urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                return int(response.status), body
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(_MAX_RESPONSE_BYTES + 1)
            except OSError:
                body = b""
            return int(exc.code), body

    def _cache_get(self, cms_id: str):
        if self._cache_ttl_seconds <= 0:
            return _CACHE_MISS
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cms_id)
            if cached and now - cached[0] < self._cache_ttl_seconds:
                return cached[1]
            if cached:
                self._cache.pop(cms_id, None)
        return _CACHE_MISS

    def _cache_put(self, cms_id: str, value: ArticleScore | None) -> None:
        if self._cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            if len(self._cache) >= _CACHE_MAX_ITEMS:
                oldest = min(self._cache, key=lambda key: self._cache[key][0])
                self._cache.pop(oldest, None)
            self._cache[cms_id] = (time.monotonic(), value)


def fetch_score_lookups(
    cms_ids: list[str],
    client: ScoreApiClient,
    *,
    max_concurrency: int = 16,
) -> dict[str, ScoreLookup]:
    """Fetch unique scores concurrently while preserving fail-closed statuses."""
    unique_ids = list(dict.fromkeys(cms_ids))
    if not unique_ids:
        return {}

    workers = max(1, min(int(max_concurrency or 1), 16, len(unique_ids)))
    results: dict[str, ScoreLookup] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_id = {pool.submit(client.get_score, cms_id): cms_id for cms_id in unique_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            cms_id = future_to_id[future]
            try:
                score = future.result()
            except ScoreApiConfigurationError:
                for pending in future_to_id:
                    pending.cancel()
                raise
            except ScoreApiUnavailable:
                results[cms_id] = ScoreLookup(status="unavailable")
            else:
                results[cms_id] = ScoreLookup(
                    status="ok" if score is not None else "not_found",
                    value=score,
                )
    return results


_CACHE_MISS = object()
