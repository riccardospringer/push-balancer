"""Read the score exposed by the Render score source.

Fresh legacy UI captures remain authoritative. If no current capture exists,
the Render source may return its server-side candidate score so consumers are
not dependent on an open POC browser session.
"""

from __future__ import annotations

import datetime as dt
import http.client
import json
import logging
import math
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("push-balancer")

_CAPTURE_BASE_URL = "https://push-balancer.onrender.com/api/score-capture/by-cms-id"
_CAPTURE_BATCH_URL = (
    "https://push-balancer.onrender.com/api/score-capture/by-cms-id/batch"
)
_CAPTURE_HEALTH_URL = "https://push-balancer.onrender.com/api/score-capture/health"
_CAPTURE_HOST = "push-balancer.onrender.com"
_CAPTURE_PATH_PREFIX = "/api/score-capture/by-cms-id/"
_CAPTURE_BATCH_PATH = "/api/score-capture/by-cms-id/batch"
_CAPTURE_QUERY = "includeBreakdown=1"
_CAPTURE_HEALTH_PATH = "/api/score-capture/health"
_CMS_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_MAX_RESPONSE_BYTES = 4 * 1024
_MAX_BATCH_RESPONSE_BYTES = 1024 * 1024
_MAX_BATCH_SIZE = 500
# Match the Render source's existing eight-hour workday cache. This keeps the
# last score actually displayed by the legacy UI available when no browser tab
# happens to be open, without introducing a second scoring implementation.
_MAX_CAPTURE_AGE_SECONDS = 8 * 3600
_TIMEOUT_SECONDS = 25
_HEALTH_TIMEOUT_SECONDS = 2

try:
    import certifi as _certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()


class RenderScoreUnavailable(RuntimeError):
    """Raised when the Render score snapshot cannot be read safely."""


@dataclass(frozen=True)
class EngagementScoreBreakdown:
    """Numeric inputs already displayed by the legacy engagement tooltip."""

    kind: str
    relevance: float
    urgency: float
    curiosity: float
    freshness: float
    timing: float
    title_boost: float
    breaking: float
    research: float
    push_history: float
    topic_saturation: float


@dataclass(frozen=True)
class SportScoreBreakdown:
    """Numeric inputs already displayed by the legacy sport tooltip."""

    kind: str
    sport_relevance: float
    timing: float
    drama: float
    freshness: float


ScoreBreakdown = EngagementScoreBreakdown | SportScoreBreakdown


@dataclass(frozen=True)
class CapturedScore:
    score: float
    captured_at: str
    score_breakdown: ScoreBreakdown | None = None
    or_factor: float | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_request(request: urllib.request.Request, *, timeout: float = _TIMEOUT_SECONDS):
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
    )
    return opener.open(request, timeout=timeout)


def _validated_capture_url(cms_id: str) -> str:
    normalized_cms_id = cms_id.lower()
    if not _CMS_ID_RE.fullmatch(normalized_cms_id):
        raise RenderScoreUnavailable("Render score source is not configured safely")
    url = f"{_CAPTURE_BASE_URL}/{normalized_cms_id}?{_CAPTURE_QUERY}"
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _CAPTURE_HOST
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != f"{_CAPTURE_PATH_PREFIX}{normalized_cms_id}"
        or parsed.query != _CAPTURE_QUERY
        or parsed.fragment
    ):
        raise RenderScoreUnavailable("Render score source is not configured safely")
    return url


def _validated_batch_capture_url() -> str:
    url = f"{_CAPTURE_BATCH_URL}?{_CAPTURE_QUERY}"
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _CAPTURE_HOST
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != _CAPTURE_BATCH_PATH
        or parsed.query != _CAPTURE_QUERY
        or parsed.fragment
    ):
        raise RenderScoreUnavailable("Render score source is not configured safely")
    return url


def _validated_health_url() -> str:
    parsed = urllib.parse.urlsplit(_CAPTURE_HEALTH_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _CAPTURE_HOST
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != _CAPTURE_HEALTH_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise RenderScoreUnavailable("Render score source is not configured safely")
    return _CAPTURE_HEALTH_URL


def _read_capture(cms_id: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        _validated_capture_url(cms_id),
        headers={
            "Accept": "application/json",
            "User-Agent": "NextPushBalancerScoreAdapter/1.0",
        },
        method="GET",
    )
    try:
        with _open_request(request) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        log.warning("Render score source returned status %s", exc.code)
        raise RenderScoreUnavailable("Render score source request failed") from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        log.warning("Render score source request failed (%s)", type(exc).__name__)
        raise RenderScoreUnavailable("Render score source request failed") from exc

    if len(raw) > _MAX_RESPONSE_BYTES:
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise RenderScoreUnavailable("Render score source returned an invalid response") from exc
    if not isinstance(payload, dict):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    return payload


def _read_capture_batch(cms_ids: list[str]) -> dict[str, Any]:
    request_body = json.dumps(
        {"cmsIds": cms_ids},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        _validated_batch_capture_url(),
        data=request_body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "NextPushBalancerScoreAdapter/1.0",
        },
        method="POST",
    )
    try:
        with _open_request(request) as response:
            raw = response.read(_MAX_BATCH_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        log.warning("Render score batch source returned status %s", exc.code)
        raise RenderScoreUnavailable("Render score source request failed") from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        log.warning("Render score batch source request failed (%s)", type(exc).__name__)
        raise RenderScoreUnavailable("Render score source request failed") from exc

    if len(raw) > _MAX_BATCH_RESPONSE_BYTES:
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise RenderScoreUnavailable("Render score source returned an invalid response") from exc
    if not isinstance(payload, dict):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    return payload


def _parse_capture(payload: dict[str, Any], now: float) -> CapturedScore | None:
    payload_keys = set(payload)
    legacy_keys = {"score", "capturedAt"}
    enriched_keys = legacy_keys | {"scoreBreakdown", "orFactor"}
    if payload_keys not in (legacy_keys, enriched_keys):
        raise RenderScoreUnavailable("Render score source returned an invalid response")

    score_raw = payload["score"]
    captured_at_raw = payload["capturedAt"]
    if (
        isinstance(score_raw, bool)
        or not isinstance(score_raw, (int, float))
        or isinstance(captured_at_raw, bool)
        or not isinstance(captured_at_raw, int)
    ):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    try:
        score = float(score_raw)
        timestamp = float(captured_at_raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderScoreUnavailable("Render score source returned an invalid response") from exc
    if (
        not math.isfinite(score)
        or not math.isfinite(timestamp)
        or not 0 < score <= 100
        or timestamp <= 0
        or timestamp > now + 30
    ):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    if now - timestamp >= _MAX_CAPTURE_AGE_SECONDS:
        return None
    captured_at = (
        dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    if payload_keys == legacy_keys:
        return CapturedScore(score=score, captured_at=captured_at)

    score_breakdown = _parse_score_breakdown(payload["scoreBreakdown"])
    or_factor = _strict_number(payload["orFactor"], minimum=0.6, maximum=1.5)
    return CapturedScore(
        score=score,
        captured_at=captured_at,
        score_breakdown=score_breakdown,
        or_factor=or_factor,
    )


def _strict_number(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RenderScoreUnavailable("Render score source returned an invalid response") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    return number


def _parse_score_breakdown(value: Any) -> ScoreBreakdown:
    if not isinstance(value, dict):
        raise RenderScoreUnavailable("Render score source returned an invalid response")

    kind = value.get("kind")
    if kind == "engagement":
        required_keys = {
            "kind",
            "relevance",
            "urgency",
            "curiosity",
            "freshness",
            "timing",
            "titleBoost",
            "breaking",
            "research",
            "pushHistory",
            "topicSaturation",
        }
        if set(value) != required_keys:
            raise RenderScoreUnavailable("Render score source returned an invalid response")
        return EngagementScoreBreakdown(
            kind=kind,
            relevance=_strict_number(value["relevance"], minimum=0, maximum=30),
            urgency=_strict_number(value["urgency"], minimum=0, maximum=25),
            curiosity=_strict_number(value["curiosity"], minimum=0, maximum=25),
            freshness=_strict_number(value["freshness"], minimum=0, maximum=20),
            timing=_strict_number(value["timing"], minimum=0, maximum=15),
            title_boost=_strict_number(value["titleBoost"], minimum=0, maximum=15),
            breaking=_strict_number(value["breaking"], minimum=0, maximum=15),
            research=_strict_number(value["research"], minimum=0, maximum=12),
            push_history=_strict_number(value["pushHistory"], minimum=-4, maximum=8),
            topic_saturation=_strict_number(
                value["topicSaturation"], minimum=-30, maximum=0
            ),
        )
    if kind == "sport":
        required_keys = {"kind", "sportRelevance", "timing", "drama", "freshness"}
        if set(value) != required_keys:
            raise RenderScoreUnavailable("Render score source returned an invalid response")
        return SportScoreBreakdown(
            kind=kind,
            sport_relevance=_strict_number(value["sportRelevance"], minimum=0, maximum=35),
            timing=_strict_number(value["timing"], minimum=0, maximum=30),
            drama=_strict_number(value["drama"], minimum=0, maximum=25),
            freshness=_strict_number(value["freshness"], minimum=0, maximum=10),
        )
    raise RenderScoreUnavailable("Render score source returned an invalid response")


def get_captured_score(cms_id: str, *, now: float | None = None) -> CapturedScore | None:
    """Return the latest workday UI snapshot for one CMS ID."""
    if not _CMS_ID_RE.fullmatch(cms_id):
        return None

    payload = _read_capture(cms_id)
    if payload is None:
        return None
    reference_time = time.time() if now is None else now
    return _parse_capture(payload, reference_time)


def get_captured_scores_batch(
    cms_ids: list[str],
    *,
    now: float | None = None,
) -> tuple[list[str], list[CapturedScore | None]]:
    """Return one strictly validated Render result per unique normalized CMS ID."""
    if not 1 <= len(cms_ids) <= _MAX_BATCH_SIZE:
        raise RenderScoreUnavailable("Render score source is not configured safely")

    normalized_cms_ids: list[str] = []
    seen: set[str] = set()
    for cms_id in cms_ids:
        if not isinstance(cms_id, str) or not _CMS_ID_RE.fullmatch(cms_id):
            raise RenderScoreUnavailable("Render score source is not configured safely")
        normalized = cms_id.lower()
        if normalized not in seen:
            normalized_cms_ids.append(normalized)
            seen.add(normalized)

    payload = _read_capture_batch(normalized_cms_ids)
    if set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise RenderScoreUnavailable("Render score source returned an invalid response")
    source_results = payload["results"]
    if len(source_results) != len(normalized_cms_ids):
        raise RenderScoreUnavailable("Render score source returned an invalid response")

    reference_time = time.time() if now is None else now
    parsed_results: list[CapturedScore | None] = []
    for expected_cms_id, source_item in zip(
        normalized_cms_ids,
        source_results,
        strict=True,
    ):
        if not isinstance(source_item, dict) or source_item.get("cmsId") != expected_cms_id:
            raise RenderScoreUnavailable("Render score source returned an invalid response")
        if source_item.get("status") == "notFound":
            if set(source_item) != {"cmsId", "status"}:
                raise RenderScoreUnavailable("Render score source returned an invalid response")
            parsed_results.append(None)
            continue
        if source_item.get("status") != "found":
            raise RenderScoreUnavailable("Render score source returned an invalid response")

        legacy_keys = {"cmsId", "status", "score", "capturedAt"}
        enriched_keys = legacy_keys | {"scoreBreakdown", "orFactor"}
        if set(source_item) not in (legacy_keys, enriched_keys):
            raise RenderScoreUnavailable("Render score source returned an invalid response")
        capture_payload = {
            key: value
            for key, value in source_item.items()
            if key not in {"cmsId", "status"}
        }
        parsed_results.append(_parse_capture(capture_payload, reference_time))

    return normalized_cms_ids, parsed_results


def require_capture_source_ready() -> None:
    """Fail unless the CIDR-protected Render score source is reachable."""
    request = urllib.request.Request(
        _validated_health_url(),
        headers={
            "Accept": "application/json",
            "User-Agent": "NextPushBalancerScoreAdapter/1.0",
        },
        method="GET",
    )
    try:
        with _open_request(request, timeout=_HEALTH_TIMEOUT_SECONDS) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        log.warning("Render score health source returned status %s", exc.code)
        raise RenderScoreUnavailable("Render score source health request failed") from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        log.warning("Render score health source request failed (%s)", type(exc).__name__)
        raise RenderScoreUnavailable("Render score source health request failed") from exc

    if len(raw) > _MAX_RESPONSE_BYTES:
        raise RenderScoreUnavailable("Render score source health returned an invalid response")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise RenderScoreUnavailable(
            "Render score source health returned an invalid response"
        ) from exc
    if payload != {"status": "ok"}:
        raise RenderScoreUnavailable("Render score source health returned an invalid response")
