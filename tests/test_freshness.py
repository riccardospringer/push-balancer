"""Focused tests for the shared article-publication freshness policy."""

from __future__ import annotations

import pytest

from app.scoring.freshness import (
    MAX_ARTICLE_AGE_SECONDS,
    freshness_score_multiplier,
    is_publication_eligible,
    parse_publication_timestamp,
    publication_age_hours,
)


NOW = 1_800_000_000


@pytest.mark.parametrize(
    ("age_hours", "expected"),
    [
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 0.975),
        (3.0, 0.95),
        (4.5, 0.875),
        (6.0, 0.8),
        (7.5, 0.675),
        (9.0, 0.55),
        (10.5, 0.425),
        (12.0, 0.3),
        (12.0001, 0.0),
    ],
)
def test_freshness_score_multiplier_interpolates_policy_curve(
    age_hours,
    expected,
):
    assert freshness_score_multiplier(age_hours) == pytest.approx(expected)


def test_publication_parser_accepts_timezone_aware_iso_and_epoch_seconds():
    assert parse_publication_timestamp("2027-01-15T09:00:00+01:00") == 1_800_000_000
    assert parse_publication_timestamp(NOW) == NOW
    assert publication_age_hours(NOW - 1800, now_ts=NOW) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value",
    [None, "", "not-a-date", "2027-01-15T09:00:00", True, 0, float("nan")],
)
def test_publication_parser_rejects_missing_or_ambiguous_values(value):
    assert parse_publication_timestamp(value) is None


def test_publication_eligibility_includes_exactly_twelve_hours_only():
    assert is_publication_eligible(
        NOW - MAX_ARTICLE_AGE_SECONDS,
        now_ts=NOW,
    )
    assert not is_publication_eligible(
        NOW - MAX_ARTICLE_AGE_SECONDS - 1,
        now_ts=NOW,
    )


def test_publication_eligibility_allows_only_explicit_future_skew():
    assert is_publication_eligible(
        NOW + 30,
        now_ts=NOW,
        max_future_skew_seconds=30,
    )
    assert not is_publication_eligible(
        NOW + 31,
        now_ts=NOW,
        max_future_skew_seconds=30,
    )
