import datetime as dt
import json
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from app.notifications.teams import (
    TeamsAlertConfig,
    _daily_runtime_opportunities,
    build_teams_daily_schedule,
    evaluate_teams_alert_candidates,
)
from app.push_schedule.weekly_baseline import PDF_OR_MATRIX
from app.routers.feed import _apply_internal_score_api_scores
from app.score_api_client import ScoreApiClient


CMS_A = "0123456789abcdef01234567"
CMS_B = "89abcdef0123456701234567"


def _config(**overrides):
    values = {
        "enabled": True,
        "webhook_url": "https://teams.example.invalid/webhook",
        "require_internal_score_api": True,
        "allowed_sections": (),
        "excluded_sections": (),
        "target_pushes_per_day": 15,
        "min_alerts_per_day": 15,
        "max_alerts_per_day": 18,
        "global_cooldown_minutes": 30,
        "min_minutes_since_last_push": 30,
        "slot_deadline_minute": 45,
        "slot_gate_enabled": True,
    }
    values.update(overrides)
    return TeamsAlertConfig(**values)


def _labels(date_iso):
    schedule = build_teams_daily_schedule(date_iso, _config())
    return schedule, {slot["label"] for slot in schedule["slots"]}


def test_monday_uses_all_18_binding_base_and_golden_slots():
    schedule, labels = _labels("2026-07-13")
    expected_hours = {7, 8, 9, 18, 19, 20, 21, 22}
    expected = {"06:15", "06:45"}
    expected.update(f"{hour:02d}:{minute:02d}" for hour in expected_hours for minute in (15, 45))

    assert labels == expected
    assert schedule["count"] == 18
    assert schedule["requiredCount"] == 18
    assert schedule["runtimeOpportunityCount"] == 18
    assert schedule["minimumDoubleCount"] == 9
    assert schedule["optionalDoubleCount"] == 0
    assert schedule["meetsTargetCoverage"] is True


@pytest.mark.parametrize(
    ("date_iso", "weekday"),
    [
        ("2026-07-13", 0),
        ("2026-07-14", 1),
        ("2026-07-15", 2),
        ("2026-07-16", 3),
        ("2026-07-17", 4),
        ("2026-07-18", 5),
        ("2026-07-19", 6),
    ],
)
def test_every_weekday_pairs_each_red_yellow_hour_and_reaches_15_to_18(
    date_iso,
    weekday,
):
    schedule, labels = _labels(date_iso)

    assert {"06:15", "06:45"}.issubset(labels)
    assert 15 <= schedule["runtimeOpportunityCount"] <= 18
    assert len(labels) == schedule["runtimeOpportunityCount"]
    assert not ({"10:45", "11:45"} & labels)

    golden_hours = {
        hour
        for hour in range(6, 24)
        if (PDF_OR_MATRIX.get((hour, weekday)) or {}).get("avg_or", 0.0) >= 6.0
    }
    for hour in golden_hours:
        assert {f"{hour:02d}:15", f"{hour:02d}:45"}.issubset(labels)

    opportunities = _daily_runtime_opportunities(dt.date.fromisoformat(date_iso), _config())
    assert all(
        int(current["ts"]) - int(previous["ts"]) >= 30 * 60
        for previous, current in zip(opportunities, opportunities[1:])
    )


@pytest.mark.parametrize("date_iso", ["2026-01-14", "2026-07-15"])
def test_schedule_keeps_berlin_wall_clock_slots_across_dst_seasons(date_iso):
    opportunities = _daily_runtime_opportunities(dt.date.fromisoformat(date_iso), _config())
    local = [
        dt.datetime.fromtimestamp(item["ts"], ZoneInfo("Europe/Berlin")).strftime("%H:%M")
        for item in opportunities
    ]

    assert local[0] == "06:15"
    assert local[1] == "06:45"


def _mock_decision(candidate):
    return {
        "candidateId": candidate["url"],
        "shouldNotify": True,
        "isBreaking": False,
        "score": candidate["score"],
        "scoreSource": candidate["scoreSource"],
        "selectionScore": candidate["selectionScore"],
        "editorialScore": candidate.get("editorialScore", 90.0),
        "teamsAlertScore": candidate.get("teamsAlertScore", 90.0),
        "expectedOpens": candidate.get("expectedOpens", 1000),
        "minimumPressure": {"active": False},
        "reasons": [],
        "blockingReasons": [],
        "agentReview": {},
    }


def test_higher_api_score_cannot_be_overtaken_by_local_composite():
    higher_api = {
        "id": "higher-api",
        "url": "https://www.bild.de/news/higher-api",
        "title": "Synthetic higher API score",
        "score": 88.0,
        "scoreSource": "internal_score_api",
        "selectionScore": 70.0,
    }
    lower_api = {
        "id": "lower-api",
        "url": "https://www.bild.de/news/lower-api",
        "title": "Synthetic lower API score",
        "score": 87.9,
        "scoreSource": "internal_score_api",
        "selectionScore": 100.0,
    }

    with patch(
        "app.notifications.teams.should_notify_teams",
        side_effect=lambda candidate, *_args: _mock_decision(candidate),
    ):
        result = evaluate_teams_alert_candidates(
            [lower_api, higher_api],
            context={"nowTs": 1_800_000_000},
            config=_config(),
        )

    assert result["selectedCandidate"] == higher_api
    assert result["canonicalApiTop1"] is True
    selected = next(
        item["decision"] for item in result["decisions"] if item["decision"]["shouldNotify"]
    )
    assert selected["competition"]["selectionMetric"] == "internal_push_balancer_score"
    assert selected["competition"]["scoreDelta"] == 0.1


def test_exact_api_score_tie_uses_secondary_score_without_field_veto():
    candidates = [
        {
            "id": "tie-a",
            "url": "https://www.bild.de/news/tie-a",
            "title": "Synthetic tie A",
            "score": 88.0,
            "scoreSource": "internal_score_api",
            "selectionScore": 80.0,
        },
        {
            "id": "tie-b",
            "url": "https://www.bild.de/news/tie-b",
            "title": "Synthetic tie B",
            "score": 88.0,
            "scoreSource": "internal_score_api",
            "selectionScore": 81.0,
        },
    ]

    with patch(
        "app.notifications.teams.should_notify_teams",
        side_effect=lambda candidate, *_args: _mock_decision(candidate),
    ):
        result = evaluate_teams_alert_candidates(
            candidates,
            context={"nowTs": 1_800_000_000},
            config=_config(min_selection_margin=99.0),
        )

    assert result["selectedCandidate"] == candidates[1]
    assert result["fieldUncertain"] is False


def test_0645_refresh_reorders_scores_and_excludes_no_data_fallback():
    score_round = 0

    def transport(url, _headers, _timeout):
        cms_id = url.rsplit("/", 1)[-1]
        scores = ({CMS_A: 90.0, CMS_B: 86.0}, {CMS_A: 89.0, CMS_B: 92.0})[score_round]
        body = {
            "cmsId": cms_id,
            "score": scores[cms_id],
            "scoredAt": "2026-07-20T06:44:30Z" if score_round else "2026-07-20T06:14:30Z",
        }
        return 200, json.dumps(body).encode("utf-8")

    client = ScoreApiClient(
        "https://scores.example.invalid",
        "synthetic-key",
        transport=transport,
        cache_ttl_seconds=0,
    )
    articles = [
        {
            "id": f"https://www.bild.de/news/a-{CMS_A}.html",
            "url": f"https://www.bild.de/news/a-{CMS_A}.html",
            "title": "Synthetic A",
            "score": 99.0,
            "pubDate": "2026-07-20T06:10:00Z",
        },
        {
            "id": f"https://www.bild.de/news/b-{CMS_B}.html",
            "url": f"https://www.bild.de/news/b-{CMS_B}.html",
            "title": "Synthetic B",
            "score": 10.0,
            "pubDate": "2026-07-20T06:11:00Z",
        },
    ]

    first = _apply_internal_score_api_scores(
        [dict(item) for item in articles],
        client=client,
        now=dt.datetime(2026, 7, 20, 6, 15, tzinfo=dt.timezone.utc),
    )
    score_round = 1
    reranked = _apply_internal_score_api_scores(
        [dict(item) for item in articles],
        client=client,
        now=dt.datetime(2026, 7, 20, 6, 45, tzinfo=dt.timezone.utc),
    )

    assert first[0]["cmsId"] == CMS_A
    assert reranked[0]["cmsId"] == CMS_B
    assert reranked[0]["score"] == 92.0
