"""Fail-closed consumer for the internal Push Balancer score API.

Only CMS IDs are transmitted. The API key stays in the request header and is
never included in URLs, logs, exceptions, diagnostics, or Teams payloads.
"""

from __future__ import annotations

import json
import logging
import math
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable


_CMS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_URL_DOCUMENT_ID_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{24})(?![0-9a-fA-F])")
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_BATCH_RESPONSE_BYTES = 1024 * 1024
_MAX_BATCH_SIZE = 200
_REQUIRED_SCORE_RESPONSE_FIELDS = {"cmsId", "score", "scoredAt"}
_OPTIONAL_SCORE_RESPONSE_FIELDS = {"scoreBreakdown", "orFactor"}
_ALLOWED_SCORE_RESPONSE_FIELDS = (
    _REQUIRED_SCORE_RESPONSE_FIELDS | _OPTIONAL_SCORE_RESPONSE_FIELDS
)
_BATCH_RESPONSE_FIELDS = {
    "requestedCount",
    "uniqueCount",
    "foundCount",
    "notFoundCount",
    "results",
}
_BATCH_FOUND_FIELDS = {
    "status",
    "cmsId",
    "score",
    "scoredAt",
    "scoreBreakdown",
    "orFactor",
}
_BATCH_NOT_FOUND_FIELDS = {"status", "cmsId"}
_ENGAGEMENT_BREAKDOWN_BOUNDS = {
    "relevance": (0.0, 30.0),
    "urgency": (0.0, 25.0),
    "curiosity": (0.0, 25.0),
    "freshness": (0.0, 20.0),
    "timing": (0.0, 15.0),
    "titleBoost": (0.0, 15.0),
    "breaking": (0.0, 15.0),
    "research": (0.0, 12.0),
    "pushHistory": (-4.0, 8.0),
    "topicSaturation": (-30.0, 0.0),
}
_SPORT_BREAKDOWN_BOUNDS = {
    "sportRelevance": (0.0, 35.0),
    "timing": (0.0, 30.0),
    "drama": (0.0, 25.0),
    "freshness": (0.0, 10.0),
}
log = logging.getLogger("push-balancer")
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
    # True, wenn der Score aus dem Ausfall-Puffer stammt (die API war kurz
    # nicht erreichbar). Der Score selbst bleibt derselbe kanonische Wert; die
    # Frische wird weiterhin ausschliesslich ueber ``scored_at`` geprueft.
    served_from_outage_buffer: bool = False

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
BatchTransport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


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


def _is_bounded_number(raw: object, minimum: float, maximum: float) -> bool:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return False
    value = float(raw)
    return math.isfinite(value) and minimum <= value <= maximum


def _validate_optional_score_details(payload: dict) -> None:
    """Validate the complete documented explanation pair when it is present."""
    has_breakdown = "scoreBreakdown" in payload
    has_or_factor = "orFactor" in payload
    if has_breakdown != has_or_factor:
        raise ScoreApiUnavailable("Score API response contract is invalid")
    if not has_breakdown:
        return

    breakdown = payload.get("scoreBreakdown")
    or_factor = payload.get("orFactor")
    if breakdown is None and or_factor is None:
        return
    if breakdown is None or or_factor is None:
        raise ScoreApiUnavailable("Score API response contract is invalid")
    if not _is_bounded_number(or_factor, 0.6, 1.5) or not isinstance(breakdown, dict):
        raise ScoreApiUnavailable("Score API response contract is invalid")

    kind = breakdown.get("kind")
    bounds = (
        _ENGAGEMENT_BREAKDOWN_BOUNDS
        if kind == "engagement"
        else _SPORT_BREAKDOWN_BOUNDS
        if kind == "sport"
        else None
    )
    if bounds is None or set(breakdown) != {"kind", *bounds}:
        raise ScoreApiUnavailable("Score API response contract is invalid")
    if any(
        not _is_bounded_number(breakdown.get(field), minimum, maximum)
        for field, (minimum, maximum) in bounds.items()
    ):
        raise ScoreApiUnavailable("Score API response contract is invalid")


def resolve_cms_id(article: dict) -> str | None:
    """Resolve a canonical field first, then a strict URL-embedded document ID."""
    for field_name in ("cmsId", "documentId", "articleId", "id"):
        value = article.get(field_name)
        if value is None:
            continue
        candidate = str(value).strip()
        if _CMS_ID_RE.fullmatch(candidate):
            return candidate.lower() if len(candidate) == 24 else candidate

    for field_name in ("url", "link", "urlId"):
        raw = str(article.get(field_name) or "").strip()
        if not raw:
            continue
        if re.fullmatch(r"[0-9a-fA-F]{24}", raw):
            return raw.lower()
        match = _URL_DOCUMENT_ID_RE.search(urllib.parse.urlsplit(raw).path)
        if match:
            return match.group(1).lower()
    return None


class ScoreApiClient:
    """Small bounded client with strict response validation and short caching."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 2.5,
        batch_timeout_seconds: float = 35.0,
        cache_ttl_seconds: float = 45.0,
        outage_buffer_seconds: float = 900.0,
        max_retries: int = 1,
        transport: Transport | None = None,
        batch_transport: BatchTransport | None = None,
    ) -> None:
        if not (api_key or "").strip():
            raise ScoreApiConfigurationError("Score API key is missing")
        self._base_url = _validated_base_url(base_url)
        self._api_key = api_key.strip()
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._batch_timeout_seconds = max(0.1, float(batch_timeout_seconds))
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._max_retries = max(0, int(max_retries))
        self._transport = transport or self._urllib_transport
        self._batch_transport = batch_transport or self._urllib_batch_transport
        self._cache: dict[str, tuple[float, ArticleScore | None]] = {}
        self._cache_lock = threading.Lock()
        # Ausfall-Puffer: der zuletzt erfolgreich geholte Score je CMS-ID.
        # Er ueberbrueckt kurze API-Ausfaelle, ohne den Frische-Vertrag
        # aufzuweichen - ``scored_at`` reist mit und wird beim Konsumenten
        # weiterhin gegen die 900-Sekunden-Grenze geprueft.
        self._outage_buffer: dict[str, tuple[float, ArticleScore]] = {}
        self._outage_buffer_seconds = max(0.0, float(outage_buffer_seconds))

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
                self._outage_buffer_put(cms_id, result)
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

        buffered = self._outage_buffer_get(cms_id)
        if buffered is not None:
            # Die API ist kurz nicht erreichbar - der zuletzt gelieferte Score
            # ist aber unveraendert gueltig. Ob er noch frisch genug ist,
            # entscheidet unveraendert der Konsument anhand von ``scored_at``.
            log.warning(
                "[ScoreApi] Ausfall ueberbrueckt: letzter bekannter Score wird "
                "weitergereicht (Alter %.0f s)",
                buffered.age_seconds(),
            )
            return buffered
        raise ScoreApiUnavailable("Score API is unavailable after bounded retry") from last_error

    def get_scores_batch(self, cms_ids: list[str]) -> dict[str, ArticleScore | None]:
        """Fetch one ordered field through the score API's single batch POST."""
        normalized_ids = [str(cms_id or "").strip().lower() for cms_id in cms_ids]
        if (
            not 1 <= len(normalized_ids) <= _MAX_BATCH_SIZE
            or any(not re.fullmatch(r"[0-9a-f]{24}", cms_id) for cms_id in normalized_ids)
        ):
            raise ScoreApiUnavailable("Score API batch request is invalid")

        url = f"{self._base_url}/api/v1/scores/batch"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "PushBalancer-Teams/1.0",
            "X-Score-Key": self._api_key,
        }
        request_body = json.dumps(
            {"cmsIds": normalized_ids},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        last_error: Exception | None = None
        # A field lookup may retry once, regardless of a more permissive legacy
        # single-lookup setting. Two 35-second attempts stay inside the slot window.
        for _attempt in range(min(self._max_retries, 1) + 1):
            try:
                status, body = self._batch_transport(
                    url,
                    headers,
                    request_body,
                    self._batch_timeout_seconds,
                )
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                last_error = exc
                continue

            if status == 200:
                return self._parse_batch_success(normalized_ids, body)
            if status in (401, 403):
                raise ScoreApiConfigurationError(
                    f"Score API authorization failed (HTTP {status})"
                )
            if status in (408, 429) or 500 <= status <= 599:
                last_error = ScoreApiUnavailable(f"Score API returned HTTP {status}")
                continue
            raise ScoreApiUnavailable(f"Score API returned unexpected HTTP {status}")

        raise ScoreApiUnavailable(
            "Score API batch is unavailable after bounded retry"
        ) from last_error

    def _parse_batch_success(
        self,
        requested_ids: list[str],
        body: bytes,
    ) -> dict[str, ArticleScore | None]:
        if len(body) > _MAX_BATCH_RESPONSE_BYTES:
            raise ScoreApiUnavailable("Score API batch response is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoreApiUnavailable("Score API returned invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != _BATCH_RESPONSE_FIELDS:
            raise ScoreApiUnavailable("Score API batch response contract is invalid")

        results = payload.get("results")
        counts = {
            field: payload.get(field)
            for field in (
                "requestedCount",
                "uniqueCount",
                "foundCount",
                "notFoundCount",
            )
        }
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values())
            or not isinstance(results, list)
            or counts["requestedCount"] != len(requested_ids)
            or counts["uniqueCount"] != len(set(requested_ids))
            or len(results) != len(requested_ids)
        ):
            raise ScoreApiUnavailable("Score API batch response contract is invalid")

        parsed: dict[str, ArticleScore | None] = {}
        found_count = 0
        for expected_id, item in zip(requested_ids, results, strict=True):
            if not isinstance(item, dict) or item.get("cmsId") != expected_id:
                raise ScoreApiUnavailable("Score API batch response contract is invalid")
            status = item.get("status")
            if status == "notFound":
                if set(item) != _BATCH_NOT_FOUND_FIELDS:
                    raise ScoreApiUnavailable("Score API batch response contract is invalid")
                parsed[expected_id] = None
                continue
            if status != "found" or set(item) != _BATCH_FOUND_FIELDS:
                raise ScoreApiUnavailable("Score API batch response contract is invalid")
            score_payload = {key: value for key, value in item.items() if key != "status"}
            parsed[expected_id] = self._parse_success(
                expected_id,
                json.dumps(score_payload, separators=(",", ":")).encode("utf-8"),
            )
            found_count += 1

        if (
            counts["foundCount"] != found_count
            or counts["notFoundCount"] != len(requested_ids) - found_count
        ):
            raise ScoreApiUnavailable("Score API batch response contract is invalid")
        return parsed

    def _parse_success(self, cms_id: str, body: bytes) -> ArticleScore:
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ScoreApiUnavailable("Score API response is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoreApiUnavailable("Score API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ScoreApiUnavailable("Score API response contract is invalid")
        fields = set(payload)
        if not _REQUIRED_SCORE_RESPONSE_FIELDS.issubset(fields) or not fields.issubset(
            _ALLOWED_SCORE_RESPONSE_FIELDS
        ):
            raise ScoreApiUnavailable("Score API response contract is invalid")

        _validate_optional_score_details(payload)
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

    @staticmethod
    def _urllib_batch_transport(
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(
            _NoRedirectHandler(),
            urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                response_body = response.read(_MAX_BATCH_RESPONSE_BYTES + 1)
                return int(response.status), response_body
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read(_MAX_BATCH_RESPONSE_BYTES + 1)
            except OSError:
                response_body = b""
            return int(exc.code), response_body

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

    def _outage_buffer_put(self, cms_id: str, value: ArticleScore) -> None:
        if self._outage_buffer_seconds <= 0:
            return
        with self._cache_lock:
            if len(self._outage_buffer) >= _CACHE_MAX_ITEMS:
                oldest = min(
                    self._outage_buffer, key=lambda key: self._outage_buffer[key][0]
                )
                self._outage_buffer.pop(oldest, None)
            self._outage_buffer[cms_id] = (time.monotonic(), value)

    def _outage_buffer_get(self, cms_id: str) -> ArticleScore | None:
        if self._outage_buffer_seconds <= 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            entry = self._outage_buffer.get(cms_id)
            if not entry:
                return None
            if now - entry[0] >= self._outage_buffer_seconds:
                self._outage_buffer.pop(cms_id, None)
                return None
            stored = entry[1]
        return replace(stored, served_from_outage_buffer=True)

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
    """Fetch all valid unique IDs in one batch and fail closed per result."""
    del max_concurrency  # Kept for call-site compatibility; batching replaces fan-out.
    unique_ids = list(dict.fromkeys(cms_ids))
    if not unique_ids:
        return {}

    valid_ids = [cms_id for cms_id in unique_ids if re.fullmatch(r"[0-9a-fA-F]{24}", cms_id)]
    results = {
        cms_id: ScoreLookup(status="unavailable")
        for cms_id in unique_ids
        if cms_id not in valid_ids
    }
    if not valid_ids:
        return results
    try:
        batch = client.get_scores_batch(valid_ids)
    except ScoreApiConfigurationError:
        raise
    except ScoreApiUnavailable:
        results.update(
            {cms_id: ScoreLookup(status="unavailable") for cms_id in valid_ids}
        )
        return results

    for cms_id in valid_ids:
        score = batch.get(cms_id.lower())
        results[cms_id] = ScoreLookup(
            status="ok" if score is not None else "not_found",
            value=score,
        )
    return results


_CACHE_MISS = object()
