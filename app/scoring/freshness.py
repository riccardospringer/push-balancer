"""Shared publication-freshness policy for article scoring and capture storage.

Publication timestamps are deliberately treated as eligibility data rather than
as a best-effort hint: an absent, ambiguous, or implausibly future timestamp
must never make an article look fresh.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Final


MAX_ARTICLE_AGE_HOURS: Final[float] = 12.0
MAX_ARTICLE_AGE_SECONDS: Final[int] = int(MAX_ARTICLE_AGE_HOURS * 3600)

# Piecewise-linear policy requested by Editorial (Score-Umbau 30.08.2026):
# bis 90 min voll frisch, danach deutlich steilerer Abfall als zuvor.  The
# hard exclusion just after 12 hours is intentional; exactly 12 hours still
# uses the final point.
FRESHNESS_SCORE_CURVE: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 1.0),
    (1.5, 1.0),
    (3.0, 0.95),
    (6.0, 0.6),
    (9.0, 0.4),
    (12.0, 0.3),
)


def parse_publication_timestamp(value: object) -> int | None:
    """Return an epoch-second publication timestamp, or ``None`` if invalid.

    ISO-8601 values must include a timezone. Numeric epoch seconds are accepted
    for internal storage callers. Naive datetimes are rejected because their
    interpretation would depend on the host timezone.
    """
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, dt.datetime):
        parsed = value
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        try:
            timestamp = parsed.timestamp()
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, (int, float)):
        timestamp = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            timestamp = float(raw)
        except ValueError:
            try:
                parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return None
            try:
                timestamp = parsed.timestamp()
            except (OverflowError, OSError, ValueError):
                return None
    else:
        return None

    if not math.isfinite(timestamp) or timestamp <= 0:
        return None
    try:
        return int(timestamp)
    except (OverflowError, ValueError):
        return None


def publication_age_hours(
    published_at: object,
    *,
    now_ts: int | float | None = None,
) -> float | None:
    """Return signed publication age in hours, or ``None`` for invalid input."""
    timestamp = parse_publication_timestamp(published_at)
    if timestamp is None:
        return None
    reference = time.time() if now_ts is None else now_ts
    if isinstance(reference, bool):
        return None
    try:
        reference_value = float(reference)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(reference_value) or reference_value <= 0:
        return None
    return (reference_value - timestamp) / 3600.0


def is_publication_eligible(
    published_at: object,
    *,
    now_ts: int | float | None = None,
    max_age_hours: float = MAX_ARTICLE_AGE_HOURS,
    max_future_skew_seconds: int = 0,
) -> bool:
    """Whether a valid publication time is within the configured hard window."""
    age_hours = publication_age_hours(published_at, now_ts=now_ts)
    if age_hours is None:
        return False
    try:
        maximum_age = float(max_age_hours)
        future_skew_hours = max(0, int(max_future_skew_seconds)) / 3600.0
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(maximum_age) or maximum_age < 0:
        return False
    return -future_skew_hours <= age_hours <= maximum_age


def freshness_score_multiplier(age_hours: object) -> float:
    """Return the piecewise-linear multiplier for an article age in hours."""
    if isinstance(age_hours, bool):
        return 0.0
    try:
        age = float(age_hours)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(age):
        return 0.0
    if age <= FRESHNESS_SCORE_CURVE[0][0]:
        return FRESHNESS_SCORE_CURVE[0][1]
    if age > MAX_ARTICLE_AGE_HOURS:
        return 0.0

    for (left_age, left_value), (right_age, right_value) in zip(
        FRESHNESS_SCORE_CURVE,
        FRESHNESS_SCORE_CURVE[1:],
    ):
        if age <= right_age:
            progress = (age - left_age) / (right_age - left_age)
            return left_value + progress * (right_value - left_value)
    return FRESHNESS_SCORE_CURVE[-1][1]


# Short alias for callers that use the policy as part of a scoring pipeline.
freshness_multiplier = freshness_score_multiplier
