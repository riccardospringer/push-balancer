from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.routers.feed import (
    _apply_server_score_freshness_weight,
    _fresh_article_candidates,
)
from app.scoring.freshness import (
    MAX_ARTICLE_AGE_HOURS,
    MAX_ARTICLE_AGE_SECONDS,
    freshness_score_multiplier,
    is_publication_eligible,
    parse_publication_timestamp,
)


NOW_TS = 1_800_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iso_timestamp(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


@pytest.mark.parametrize(
    ("age_hours", "expected_multiplier"),
    [
        (0.0, 1.0),
        (1.0, 1.0),
        (3.0, 0.95),
        (4.5, 0.775),
        (6.0, 0.6),
        (9.0, 0.4),
        (12.0, 0.3),
        (12.0 + (1.0 / 3600.0), 0.0),
    ],
)
def test_freshness_curve_and_twelve_hour_boundary(
    age_hours: float,
    expected_multiplier: float,
) -> None:
    assert freshness_score_multiplier(age_hours) == pytest.approx(
        expected_multiplier
    )


def test_publication_eligibility_includes_exact_boundary_and_rejects_older() -> None:
    assert MAX_ARTICLE_AGE_HOURS == 12.0
    assert MAX_ARTICLE_AGE_SECONDS == 12 * 60 * 60

    assert is_publication_eligible(
        NOW_TS - MAX_ARTICLE_AGE_SECONDS,
        now_ts=NOW_TS,
    )
    assert not is_publication_eligible(
        NOW_TS - MAX_ARTICLE_AGE_SECONDS - 1,
        now_ts=NOW_TS,
    )


@pytest.mark.parametrize(
    "published_at",
    [
        None,
        "",
        "not-a-publication-date",
        "2027-01-15T12:00:00",
        True,
        float("nan"),
    ],
)
def test_publication_eligibility_fails_closed_for_missing_or_invalid_timestamp(
    published_at: object,
) -> None:
    assert parse_publication_timestamp(published_at) is None
    assert not is_publication_eligible(published_at, now_ts=NOW_TS)


def test_publication_eligibility_rejects_significantly_future_timestamp() -> None:
    assert is_publication_eligible(
        NOW_TS + 30,
        now_ts=NOW_TS,
        max_future_skew_seconds=30,
    )
    assert not is_publication_eligible(
        NOW_TS + 31,
        now_ts=NOW_TS,
        max_future_skew_seconds=30,
    )
    assert not is_publication_eligible(
        NOW_TS + 60 * 60,
        now_ts=NOW_TS,
        max_future_skew_seconds=30,
    )


def test_feed_candidate_filter_is_a_hard_twelve_hour_cutoff() -> None:
    articles = [
        {
            "id": "exact-boundary",
            "pubDate": _iso_timestamp(NOW_TS - MAX_ARTICLE_AGE_SECONDS),
        },
        {
            "id": "one-second-too-old",
            "pubDate": _iso_timestamp(NOW_TS - MAX_ARTICLE_AGE_SECONDS - 1),
        },
        {"id": "missing-publication"},
        {"id": "invalid-publication", "pubDate": "unknown"},
        {
            "id": "within-future-skew-breaking",
            "pubDate": _iso_timestamp(NOW_TS + 30),
            "isBreaking": True,
        },
        {
            "id": "beyond-future-skew",
            "pubDate": _iso_timestamp(NOW_TS + 31),
        },
        {
            "id": "younger-than-three-minutes",
            "pubDate": _iso_timestamp(NOW_TS - 60),
        },
        {
            "id": "younger-than-three-minutes-breaking",
            "pubDate": _iso_timestamp(NOW_TS - 60),
            "isEilmeldung": True,
        },
        {
            "id": "three-minutes-old",
            "pubDate": _iso_timestamp(NOW_TS - 180),
        },
    ]

    filtered = _fresh_article_candidates(articles, now_ts=NOW_TS)

    assert [article["id"] for article in filtered] == [
        "exact-boundary",
        "within-future-skew-breaking",
        "younger-than-three-minutes-breaking",
        "three-minutes-old",
    ]


def test_server_and_legacy_capture_scores_are_weighted_exactly_once() -> None:
    published_at = NOW_TS - (6 * 60 * 60)
    articles = [
        {
            "id": "server-fallback",
            "pubDate": _iso_timestamp(published_at),
            "score": 80.0,
            "scoreSource": "server_editorial_fallback",
            "scoreReason": "Server-Fallback",
        },
        {
            "id": "legacy-capture",
            "pubDate": _iso_timestamp(published_at),
            "score": 80.0,
            "scoreSource": "captured_push_balancer",
            "pushBalancerScoreArticlePublishedAt": None,
            "scoreReason": "Legacy Capture",
        },
        {
            "id": "freshness-aware-capture",
            "pubDate": _iso_timestamp(published_at),
            "score": 48.0,
            "scoreSource": "captured_push_balancer",
            "pushBalancerScoreArticlePublishedAt": published_at,
            "scoreReason": "Freshness-aware Capture",
        },
    ]

    weighted = _apply_server_score_freshness_weight(articles, now_ts=NOW_TS)
    by_id = {article["id"]: article for article in weighted}

    assert by_id["server-fallback"]["score"] == 48.0
    assert by_id["legacy-capture"]["score"] == 48.0
    assert by_id["freshness-aware-capture"]["score"] == 48.0

    for article in weighted:
        assert article["freshnessEligible"] is True
        assert article["freshnessScoreMultiplier"] == 0.6

    assert "Frischefaktor" in by_id["server-fallback"]["scoreReason"]
    assert "Frischefaktor" in by_id["legacy-capture"]["scoreReason"]
    assert (
        by_id["freshness-aware-capture"]["scoreReason"]
        == "Freshness-aware Capture"
    )


def test_server_weighting_fails_closed_past_cutoff() -> None:
    articles = [
        {
            "pubDate": _iso_timestamp(NOW_TS - MAX_ARTICLE_AGE_SECONDS - 1),
            "score": 100.0,
            "scoreSource": "captured_push_balancer",
            "pushBalancerScoreArticlePublishedAt": NOW_TS,
        }
    ]

    weighted = _apply_server_score_freshness_weight(articles, now_ts=NOW_TS)

    assert weighted[0]["score"] == 0
    assert weighted[0]["freshnessEligible"] is False
    assert weighted[0]["freshnessScoreMultiplier"] == 0
    assert weighted[0]["scoreSource"] == "article_freshness_cutoff"


def test_active_root_html_filters_at_twelve_hours_and_captures_publication_time() -> None:
    source = (PROJECT_ROOT / "push-balancer.html").read_text(encoding="utf-8")

    assert "const MAX_PUSH_ARTICLE_AGE_HOURS = 12;" in source

    filter_section = source.split("function applyFilterAndSort", maxsplit=1)[1]
    filter_section = filter_section.split("// Live-Ticker", maxsplit=1)[0]
    assert (
        "if (!_articleFreshnessPolicy(a.date).eligible) return false;"
        in filter_section
    )

    capture_section = source.split("async function _captureArticleScores", maxsplit=1)[1]
    capture_section = capture_section.split("function ", maxsplit=1)[0]
    assert (
        ".filter(a => a.link && a.score > 0 && "
        "_articleFreshnessPolicy(a.date).eligible)"
        in capture_section
    )
    assert (
        "articlePublishedAt: Math.floor(a.date.getTime() / 1000)"
        in capture_section
    )
