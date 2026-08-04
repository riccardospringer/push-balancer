import datetime as dt
import json
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from app.notifications.teams import (
    TeamsAlertConfig,
    _daily_runtime_opportunities,
    _mandatory_slot_top1_binding_slot,
    binding_slot_window_open,
    build_teams_daily_schedule,
    evaluate_teams_alert_candidates,
    seconds_to_defer_cycle_for_binding_slot,
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
        "target_pushes_per_day": 11,
        "min_alerts_per_day": 11,
        "max_alerts_per_day": 15,
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


def test_monday_uses_the_deterministic_berlin_layout():
    schedule, labels = _labels("2026-07-13")

    # Morgen-Doppel 06/07/08 (gleichverteilt, mathematisch maximal gespreizt),
    # Mittagsslot 12:30, montags erster Abend-Slot 17:30, danach unveraenderte
    # dynamische Heatmap-Verteilung 18-21.
    assert labels == {
        "06:00", "06:36", "07:12", "07:47", "08:23", "08:59",
        "12:30",
        "17:30", "18:34", "19:08", "19:42", "20:17", "20:51", "21:25", "21:59",
    }
    # Ab 22:00 gilt die Ruhezeit; die Totzone 10/11 bleibt draussen.
    assert {"10:45", "11:45", "22:15", "22:45", "23:45"}.isdisjoint(labels)
    assert schedule["count"] == 15
    assert schedule["requiredCount"] == 15
    assert schedule["runtimeOpportunityCount"] == 15
    assert schedule["meetsTargetCoverage"] is True


def test_date_scoped_delay_moves_only_the_remaining_monday_slots():
    config = _config(
        slot_delay_date="2026-08-03",
        slot_delay_from="19:08",
        slot_delay_minutes=14,
    )
    berlin = ZoneInfo("Europe/Berlin")
    today = [
        dt.datetime.fromtimestamp(slot["ts"], berlin).strftime("%H:%M")
        for slot in _daily_runtime_opportunities(dt.date(2026, 8, 3), config)
    ]
    tomorrow = [
        dt.datetime.fromtimestamp(slot["ts"], berlin).strftime("%H:%M")
        for slot in _daily_runtime_opportunities(dt.date(2026, 8, 4), config)
    ]

    assert today[-6:] == ["19:22", "19:56", "20:31", "21:05", "21:39", "22:13"]
    assert "19:08" not in today
    assert "19:08" in tomorrow
    old_slot = int(dt.datetime(2026, 8, 3, 19, 8, tzinfo=berlin).timestamp())
    new_slot = int(dt.datetime(2026, 8, 3, 19, 22, tzinfo=berlin).timestamp())
    assert _mandatory_slot_top1_binding_slot(old_slot, config) is None
    assert _mandatory_slot_top1_binding_slot(new_slot, config)["ts"] == new_slot


def test_worker_defers_slow_cycle_before_slot_and_retries_inside_window():
    config = _config(
        slot_delay_date="2026-08-03",
        slot_delay_from="19:08",
        slot_delay_minutes=14,
    )
    berlin = ZoneInfo("Europe/Berlin")
    before = dt.datetime(2026, 8, 3, 19, 20, tzinfo=berlin).timestamp()
    inside = dt.datetime(2026, 8, 3, 19, 22, 30, tzinfo=berlin).timestamp()

    assert seconds_to_defer_cycle_for_binding_slot(
        before,
        config,
        guard_seconds=180,
    ) == pytest.approx(120.5)
    assert binding_slot_window_open(before, config) is False
    assert seconds_to_defer_cycle_for_binding_slot(
        inside,
        config,
        guard_seconds=180,
    ) == 0.0
    assert binding_slot_window_open(inside, config) is True


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
def test_every_weekday_uses_the_deterministic_layout_and_reaches_11_to_15(
    date_iso,
    weekday,
):
    schedule, labels = _labels(date_iso)
    slots = sorted(schedule["slots"], key=lambda slot: int(slot["ts"]))

    # Morgen-Doppel und Mittagsslot sind an jedem Wochentag identisch verbindlich.
    assert {"06:00", "06:36", "07:12", "07:47", "08:23", "08:59", "12:30"}.issubset(labels)
    assert 11 <= schedule["runtimeOpportunityCount"] <= 15
    assert len(labels) == schedule["runtimeOpportunityCount"]
    assert not ({"10:45", "11:45"} & labels)

    # Genau zwei Entscheidungen je Morgenstunde 06/07/08.
    for hour in (6, 7, 8):
        assert sum(1 for slot in slots if int(slot["hour"]) == hour) == 2

    # Abend-Hot-Hours (rot/gelb 18-21 laut Heatmap) maximal ausschoepfen:
    # zwei Slots je Hot-Stunde, gleichverteilt ueber den Block.
    hot_hours = [
        hour
        for hour in (18, 19, 20, 21)
        if (PDF_OR_MATRIX.get((hour, weekday)) or {}).get("avg_or", 0.0) >= 6.0
    ]
    evening = [slot for slot in slots if slot.get("slotRole") == "evening_hot"]
    assert len(evening) >= 2 * max(1, len(hot_hours))
    if weekday == 0:
        assert evening[0]["label"] == "17:30"

    # Mindestabstand zwischen allen verbindlichen Entscheidungen: 30 Minuten.
    gaps = [
        (int(later["ts"]) - int(earlier["ts"])) // 60
        for earlier, later in zip(slots, slots[1:])
    ]
    assert min(gaps) >= 30

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

    assert local[0] == "06:00"
    assert local[1] == "06:36"


@pytest.mark.parametrize(
    "date_iso",
    [
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
        "2026-07-18",
        "2026-07-19",
    ],
)
def test_every_calculated_slot_is_a_mandatory_top1_window(date_iso):
    config = _config()
    day = dt.date.fromisoformat(date_iso)
    for slot in _daily_runtime_opportunities(day, config):
        slot_ts = int(slot["ts"])
        assert _mandatory_slot_top1_binding_slot(slot_ts, config)["ts"] == slot_ts
        assert _mandatory_slot_top1_binding_slot(slot_ts + 299, config)["ts"] == slot_ts
        assert _mandatory_slot_top1_binding_slot(slot_ts - 1, config) is None
        assert _mandatory_slot_top1_binding_slot(slot_ts + 300, config) is None
        assert _mandatory_slot_top1_binding_slot(slot_ts + 301, config) is None


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

    def batch_transport(_url, _headers, request_body, _timeout):
        cms_ids = json.loads(request_body)["cmsIds"]
        scores = ({CMS_A: 90.0, CMS_B: 86.0}, {CMS_A: 89.0, CMS_B: 92.0})[score_round]
        results = [
            {
                "status": "found",
                "cmsId": cms_id,
                "score": scores[cms_id],
                "scoredAt": (
                    "2026-07-20T06:44:30Z"
                    if score_round
                    else "2026-07-20T06:14:30Z"
                ),
                "scoreBreakdown": None,
                "orFactor": None,
            }
            for cms_id in cms_ids
        ]
        body = {
            "requestedCount": len(results),
            "uniqueCount": len(set(cms_ids)),
            "foundCount": len(results),
            "notFoundCount": 0,
            "results": results,
        }
        return 200, json.dumps(body).encode("utf-8")

    client = ScoreApiClient(
        "https://scores.example.invalid",
        "synthetic-key",
        batch_transport=batch_transport,
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
