"""Synthetic contract tests for the scheduled Power Automate Teams hand-off."""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app import auth, database
from app.main import app
from app.notifications.teams import candidate_key
from app.teams_slot_claims import (
    teams_recommendation_slot_group_get,
    teams_recommendation_slot_get,
    teams_recommendation_slot_record_group_receipt,
    teams_recommendation_slot_try_claim,
)


POWER_AUTOMATE_KEY = "synthetic-power-automate-key"
HEADERS = {"X-Power-Automate-Key": POWER_AUTOMATE_KEY}
BERLIN = ZoneInfo("Europe/Berlin")
SLOT_TS = int(dt.datetime(2026, 8, 3, 12, 30, tzinfo=BERLIN).timestamp())

client = TestClient(app, raise_server_exceptions=True)


def test_bounded_recovery_env_uses_its_default_only_when_unset(monkeypatch):
    import app.config as app_config

    env_name = "SYNTHETIC_BOUNDED_RECOVERY_SECONDS"
    monkeypatch.delenv(env_name, raising=False)

    assert app_config._env_bounded_int_fail_closed(
        env_name,
        default=600,
        maximum=600,
    ) == (600, True)

    monkeypatch.setenv(env_name, "0")
    assert app_config._env_bounded_int_fail_closed(
        env_name,
        default=600,
        maximum=600,
    ) == (0, True)


@pytest.mark.parametrize("raw_value", ["", "   ", "not-an-integer", "-1", "601"])
def test_bounded_recovery_env_fails_closed_for_invalid_bounds(
    monkeypatch,
    raw_value,
):
    import app.config as app_config

    env_name = "SYNTHETIC_BOUNDED_RECOVERY_SECONDS"
    monkeypatch.setenv(env_name, raw_value)

    assert app_config._env_bounded_int_fail_closed(
        env_name,
        default=600,
        maximum=600,
    ) == (0, False)


def test_weekend_morning_slots_start_two_hours_later():
    import app.routers.power_automate as power_automate

    saturday = dt.date(2026, 8, 8)
    sunday = dt.date(2026, 8, 9)
    expected = (
        "08:00",
        "08:36",
        "09:12",
        "09:47",
        "10:23",
        "10:59",
        "12:30",
        "17:30",
        "18:49",
        "20:08",
        "21:26",
        "22:45",
    )

    assert power_automate.power_automate_slot_labels_for_date(saturday) == expected
    assert power_automate.power_automate_slot_labels_for_date(sunday) == expected
    assert power_automate.power_automate_slot_labels_for_date(dt.date(2026, 8, 10))[0] == "06:00"


@pytest.mark.parametrize("schedule_name", ["weekday", "weekend"])
def test_fixed_slots_are_spaced_beyond_the_maximum_dispatch_window(schedule_name):
    import app.power_automate_schedule as schedule

    labels = (
        schedule.POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS
        if schedule_name == "weekday"
        else schedule.POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS
    )
    minutes = [
        int(label.split(":")[0]) * 60 + int(label.split(":")[1])
        for label in labels
    ]
    maximum_window_minutes = schedule.POWER_AUTOMATE_MAX_DISPATCH_WINDOW_SECONDS // 60

    assert all(
        later - earlier > maximum_window_minutes
        for earlier, later in zip(minutes, minutes[1:])
    )


def test_power_automate_slot_override_applies_only_inside_its_live_window(
    monkeypatch,
):
    import app.notifications.teams as teams

    now_ts = int(dt.datetime(2026, 8, 9, 10, 59, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = {
        "id": "000000000000000000000021",
        "url": "https://www.bild.de/news/synthetic-weekend-override",
        "title": "Synthetische Wochenendmeldung",
        "category": "news",
        "score": 90.0,
        "scoreSource": "internal_score_api",
        "pubDate": "2026-08-09T10:50:00+02:00",
    }
    config = replace(
        teams.TeamsAlertConfig(),
        enabled=True,
        require_internal_score_api=True,
        allow_durable_live_history_fallback=True,
        slot_gate_enabled=True,
    )
    monkeypatch.setattr(
        teams,
        "should_notify_teams",
        lambda item, _context, _config: {
            "candidateId": teams.candidate_key(item),
            "shouldNotify": False,
            "score": item["score"],
            "scoreSource": "internal_score_api",
            "publicationReview": {"status": "valid"},
            "livePushComparison": {"available": False, "authoritative": False},
            "blockingReasons": ["Nur redaktioneller Hinweis"],
            "slotGate": {"enabled": True, "approved": False},
        },
    )
    base_context = {
        "nowTs": now_ts,
        "alertState": {},
        "recentTeamsAlerts": [],
        "contextAvailable": {"alertState": True, "recentTeamsAlerts": True},
    }
    live_context = {
        **base_context,
        "_mandatorySlotOverride": {
            "ts": now_ts - 30,
            "label": "10:59",
            "slotRole": "power_automate_fixed",
        },
    }

    live_evaluation = teams.evaluate_teams_alert_candidates(
        [candidate],
        live_context,
        config,
    )
    live_decision = live_evaluation["decisions"][0]["decision"]

    assert live_evaluation["selectedCandidate"] == candidate
    assert live_decision["shouldNotify"] is True
    assert live_decision["mandatorySlotTop1"] is True
    assert live_decision["slotGate"]["slot"]["ts"] == now_ts - 30

    expired_context = {
        **base_context,
        "_mandatorySlotOverride": {
            "ts": now_ts - 301,
            "label": "10:54",
            "slotRole": "power_automate_fixed",
        },
    }
    expired_evaluation = teams.evaluate_teams_alert_candidates(
        [candidate],
        expired_context,
        config,
    )

    assert expired_evaluation["selectedCandidate"] is None
    assert expired_evaluation["decisions"][0]["decision"].get("mandatorySlotTop1") is not True

    recovery_slot_ts = now_ts - 30
    recovery_evaluation = teams.evaluate_teams_alert_candidates(
        [candidate],
        {
            **base_context,
            "nowTs": recovery_slot_ts + 600,
            "_mandatorySlotOverride": {
                "ts": recovery_slot_ts,
                "label": "10:59",
                "slotRole": "power_automate_fixed",
                "dispatchWindowSeconds": 900,
            },
        },
        config,
    )
    expired_recovery = teams.evaluate_teams_alert_candidates(
        [candidate],
        {
            **base_context,
            "nowTs": recovery_slot_ts + 901,
            "_mandatorySlotOverride": {
                "ts": recovery_slot_ts,
                "label": "10:59",
                "slotRole": "power_automate_fixed",
                "dispatchWindowSeconds": 900,
            },
        },
        config,
    )

    assert recovery_evaluation["selectedCandidate"] == candidate
    assert recovery_evaluation["decisions"][0]["decision"]["mandatorySlotTop1"] is True
    assert expired_recovery["selectedCandidate"] is None

    future_ts = int(dt.datetime(2026, 8, 9, 12, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    invalid_overrides = (
        {"ts": "not-a-timestamp", "label": "10:59", "slotRole": "power_automate_fixed"},
        {"ts": 10**100, "label": "10:59", "slotRole": "power_automate_fixed"},
        {"ts": now_ts - 30, "label": "10:59", "slotRole": "unexpected_role"},
        {"ts": now_ts - 30, "label": "10:23", "slotRole": "power_automate_fixed"},
        {"ts": future_ts, "label": "12:30", "slotRole": "power_automate_fixed"},
    )
    for invalid_override in invalid_overrides:
        invalid_evaluation = teams.evaluate_teams_alert_candidates(
            [candidate],
            {**base_context, "_mandatorySlotOverride": invalid_override},
            config,
        )
        assert invalid_evaluation["selectedCandidate"] is None
        assert invalid_evaluation["decisions"][0]["decision"].get("mandatorySlotTop1") is not True

    readiness_probe = teams.evaluate_teams_alert_candidates(
        [candidate],
        {
            **base_context,
            "_mandatorySlotOverride": {
                "ts": future_ts,
                "label": "12:30",
                "slotRole": "power_automate_fixed",
            },
            "_scheduledReadinessProbe": True,
        },
        config,
    )
    assert readiness_probe["selectedCandidate"] == candidate
    assert readiness_probe["decisions"][0]["decision"]["mandatorySlotTop1"] is True


def test_shared_power_automate_slot_validator_rejects_schedule_mismatches():
    from app.power_automate_schedule import is_power_automate_binding_slot

    sunday_valid = int(
        dt.datetime(2026, 8, 9, 10, 59, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    sunday_weekday_only = int(
        dt.datetime(2026, 8, 9, 6, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    sunday_unscheduled = int(
        dt.datetime(2026, 8, 9, 11, 1, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    assert is_power_automate_binding_slot(sunday_valid, "10:59") is True
    assert is_power_automate_binding_slot(sunday_valid, "10:23") is False
    assert is_power_automate_binding_slot(sunday_weekday_only, "06:00") is False
    assert is_power_automate_binding_slot(sunday_unscheduled, "11:01") is False
    assert is_power_automate_binding_slot("1786258740", "10:59") is False
    assert is_power_automate_binding_slot(10**100, "10:59") is False


def test_power_automate_recovery_window_is_bounded_and_preserves_slot_identity(
    monkeypatch,
):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_RECOVERY_CONFIGURATION_VALID",
        True,
    )
    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_RECOVERY_GRACE_SECONDS",
        600,
    )

    recovered = power_automate._power_automate_binding_slot(SLOT_TS + 600)

    assert recovered == {
        "ts": SLOT_TS,
        "label": "12:30",
        "slotRole": "power_automate_fixed",
        "dispatchWindowSeconds": 900,
        "recovery": True,
    }
    assert power_automate._power_automate_binding_slot(SLOT_TS + 899) == recovered
    assert power_automate._power_automate_binding_slot(SLOT_TS + 900) is None


@pytest.mark.parametrize("raw_grace", [601, 600.0, "600", True])
def test_invalid_recovery_configuration_fails_back_to_primary_window(
    monkeypatch,
    raw_grace,
):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_RECOVERY_CONFIGURATION_VALID",
        True,
    )
    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_RECOVERY_GRACE_SECONDS",
        raw_grace,
    )

    assert power_automate._power_automate_recovery_configuration() == (0, False)
    assert power_automate._power_automate_dispatch_window_seconds() == 300
    assert power_automate._power_automate_binding_slot(SLOT_TS + 301) is None


def test_durable_history_fallback_keeps_mandatory_slot_sendable():
    from app.notifications.teams import (
        TeamsAlertConfig,
        _mandatory_slot_top1_technical_blockers,
    )

    candidate = {
        "id": "synthetic-durable-fallback",
        "url": "https://www.bild.de/news/synthetic-durable-fallback",
        "title": "Synthetische belastbare Meldung",
        "score": 91.4,
    }
    decision = {
        "scoreSource": "internal_score_api",
        "publicationReview": {"status": "valid"},
        "livePushComparison": {
            "available": False,
            "authoritative": False,
            "matchType": "",
        },
    }
    context = {
        "alertState": {},
        "recentTeamsAlerts": [],
        "contextAvailable": {"alertState": True, "recentTeamsAlerts": True},
    }
    config = replace(
        TeamsAlertConfig(),
        enabled=True,
        require_internal_score_api=True,
        allow_durable_live_history_fallback=True,
    )

    assert (
        _mandatory_slot_top1_technical_blockers(
            candidate,
            decision,
            context,
            config,
        )
        == []
    )


def _synthetic_candidates(
    now_ts: int,
    *,
    top_category: str = "news",
    alternative_category: str = "sport",
) -> tuple[dict, dict]:
    published_at = dt.datetime.fromtimestamp(now_ts - 600, dt.timezone.utc).isoformat()
    top = {
        "id": "synthetic-news-top",
        "url": f"https://www.bild.de/{top_category}/synthetic-news-top",
        "title": "Bund beschliesst synthetisches Hilfspaket",
        "category": top_category,
        "score": 91.4,
        "scoreSource": "internal_score_api",
        "predictedOR": 0.061,
        "pubDate": published_at,
    }
    alternative = {
        "id": "synthetic-sport-alternative",
        "url": f"https://www.bild.de/{alternative_category}/synthetic-sport-alternative",
        "title": "Verein bestaetigt synthetischen Transfer",
        "category": alternative_category,
        "score": 88.2,
        "scoreSource": "internal_score_api",
        "predictedOR": 0.058,
        "pubDate": published_at,
    }
    return top, alternative


def _patch_successful_claim(
    monkeypatch,
    *,
    now_ts: int,
    top_category: str = "news",
    alternative_category: str = "sport",
    include_alternative: bool = True,
) -> tuple[dict, dict]:
    import app.routers.power_automate as power_automate

    top, alternative = _synthetic_candidates(
        now_ts,
        top_category=top_category,
        alternative_category=alternative_category,
    )
    config = replace(
        power_automate.TeamsAlertConfig(),
        enabled=True,
        require_internal_score_api=True,
        slot_gate_enabled=True,
    )
    binding_slot = power_automate._power_automate_binding_slot(now_ts)
    binding_slot_ts = int((binding_slot or {}).get("ts") or SLOT_TS)
    decision = {
        "candidateId": top["url"],
        "shouldNotify": True,
        "score": top["score"],
        "scoreSource": "internal_score_api",
        "mandatorySlotTop1Candidate": True,
        "summary": "Verbindlicher Push-Balancer-Top-1 im festen Slot",
    }
    alternative_decision = {
        "candidateId": alternative["url"],
        "shouldNotify": False,
        "score": alternative["score"],
        "scoreSource": "internal_score_api",
        "mandatorySlotTop1Candidate": True,
        "blockingReasons": ["Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"],
    }
    extra_count = 3 if include_alternative else 4
    extras = [
        {
            "id": f"synthetic-extra-{index}",
            "url": f"https://www.bild.de/{top_category}/synthetic-extra-{index}",
            "title": f"Synthetische Zusatzmeldung {index}",
            "category": top_category,
            "score": 84.0 - index,
            "scoreSource": "internal_score_api",
            "predictedOR": 0.05,
            "pubDate": top["pubDate"],
        }
        for index in range(1, extra_count + 1)
    ]
    extra_decisions = [
        {
            "candidate": candidate,
            "decision": {
                "candidateId": candidate["url"],
                "shouldNotify": False,
                "score": candidate["score"],
                "scoreSource": "internal_score_api",
                "mandatorySlotTop1Candidate": True,
                "blockingReasons": ["Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"],
            },
        }
        for candidate in extras
    ]
    message_html = (
        "<h2>🔵 PUSH-EMPFEHLUNG</h2>"
        "<p><strong>Top 1:</strong> Bund beschliesst synthetisches Hilfspaket</p>"
    )
    message = {
        "_dispatchApproved": True,
        "_slotGateApproved": True,
        "payload": {
            "slotId": f"teams-recommendation-{binding_slot_ts}",
            "articleTitle": top["title"],
            "articleUrl": top["url"],
            "category": top["category"],
            "pushScore": top["score"],
            "alternativeRecommendation": (
                {
                    "articleTitle": alternative["title"],
                    "articleUrl": alternative["url"],
                    "category": alternative["category"],
                    "pushScore": alternative["score"],
                }
                if include_alternative
                else {}
            ),
            "messageHtml": message_html,
        },
        "_bindingSlotTs": binding_slot_ts,
    }

    monkeypatch.setattr(power_automate.time, "time", lambda: now_ts)
    monkeypatch.setattr(power_automate, "TeamsAlertConfig", lambda: config)
    monkeypatch.setattr(
        power_automate,
        "build_articles_payload",
        lambda **_kwargs: {
            "articles": [
                top,
                *([alternative] if include_alternative else []),
                *extras,
            ]
        },
    )
    monkeypatch.setattr(
        power_automate,
        "_memory_eligible_candidates",
        lambda candidates, **_kwargs: (candidates, {}),
    )
    monkeypatch.setattr(
        power_automate,
        "build_teams_alert_context",
        lambda candidates, **_kwargs: {"nowTs": now_ts},
    )
    monkeypatch.setattr(
        power_automate,
        "evaluate_teams_alert_candidates",
        lambda *_args, **_kwargs: {
            "selectedCandidate": top,
            "decisions": [
                {"candidate": top, "decision": decision},
                *(
                    [{"candidate": alternative, "decision": alternative_decision}]
                    if include_alternative
                    else []
                ),
                *extra_decisions,
            ],
        },
    )
    monkeypatch.setattr(
        power_automate,
        "build_teams_push_recommendation",
        lambda *_args, **_kwargs: message,
    )
    return top, alternative


def test_claim_requires_dedicated_auth_and_never_allows_caching(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    response = client.post(
        "/api/v1/power-automate/teams/claim",
        json={"requestId": "synthetic-unauthorized-run"},
    )

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["vary"] == "X-Power-Automate-Key"


def test_claim_fails_closed_when_only_ephemeral_storage_is_available(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(power_automate.app_config, "PUSH_DB_DURABLE", False)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-ephemeral-storage-run"},
        )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert "Durable recommendation storage" in response.text
    assert teams_recommendation_slot_get(SLOT_TS) is None


def test_claim_fails_closed_when_legacy_background_sender_is_enabled(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    config = replace(power_automate.TeamsAlertConfig(), enabled=True)
    monkeypatch.setattr(
        power_automate,
        "TeamsAlertConfig",
        lambda: config,
    )
    monkeypatch.setattr(
        power_automate.app_config,
        "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED",
        True,
    )
    monkeypatch.setattr(
        power_automate,
        "build_articles_payload",
        lambda **_kwargs: pytest.fail(
            "a conflicting transport must fail before selection"
        ),
    )

    response = client.post(
        "/api/v1/power-automate/teams/claim",
        headers=HEADERS,
        json={"requestId": "synthetic-conflicting-owner-run"},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"] == auth.config.TEAMS_TRANSPORT_OWNER_CONFLICT


def test_scheduled_message_uses_the_five_highest_valid_push_scores():
    import app.routers.power_automate as power_automate

    decisions = []
    for index, score in enumerate((82.1, 91.4, 76.0, 88.2, 79.5, 84.7), start=1):
        candidate = {
            "title": f"Synthetische Meldung {index}",
            "url": f"https://www.bild.de/news/synthetic-{index}",
            "score": score,
            "scoreSource": "internal_score_api",
            "pubDate": f"2026-08-03T{index + 8:02d}:15:00+02:00",
        }
        decisions.append(
            {
                "candidate": candidate,
                "decision": {
                    "score": score,
                    "scoreSource": "internal_score_api",
                    "mandatorySlotTop1Candidate": True,
                    "shouldNotify": index == 2,
                    "blockingReasons": (
                        []
                        if index == 2
                        else ["Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"]
                    ),
                },
            }
        )

    recommendations = power_automate._scheduled_recommendations(
        {"selectedCandidate": decisions[1]["candidate"], "decisions": decisions}
    )
    message_html = power_automate._scheduled_message_html(recommendations)

    assert [item["pushScore"] for item in recommendations] == [
        91.4,
        88.2,
        84.7,
        82.1,
        79.5,
    ]
    assert message_html.count("<strong>Top ") == 5
    assert "<strong>Top 5:</strong>" in message_html
    assert message_html.count("</p><br><br><p>") == 5
    assert "(03.08.2026, 10:15 Uhr)" in message_html
    assert "(03.08.2026, 14:15 Uhr)" in message_html


def test_scheduled_message_fills_four_safe_display_slots_without_fake_scores():
    import app.routers.power_automate as power_automate

    selected = {
        "id": "000000000000000000000001",
        "title": "Synthetische kanonische Meldung",
        "url": "https://www.bild.de/news/synthetic-canonical",
        "score": 91.4,
        "scoreSource": "internal_score_api",
        "pubDate": "2026-08-03T12:20:00+02:00",
    }
    decisions = [
        {
            "candidate": selected,
            "decision": {
                "candidateId": selected["url"],
                "shouldNotify": True,
                "score": selected["score"],
                "scoreSource": "internal_score_api",
                "mandatorySlotTop1Candidate": True,
                "blockingReasons": [],
            },
        }
    ]
    fallback_specs = [
        ("000000000000000000000002", "fallback-1", 88.0),
        # Same CMS article under a second URL: must not occupy another slot.
        ("000000000000000000000002", "fallback-1-duplicate", 87.0),
        ("000000000000000000000003", "fallback-2", 86.0),
        ("000000000000000000000004", "fallback-3", 84.0),
        ("000000000000000000000005", "fallback-4", 82.0),
        ("000000000000000000000006", "fallback-5", 80.0),
    ]
    for cms_id, slug, fallback_score in fallback_specs:
        candidate = {
            "id": cms_id,
            "title": f"Synthetische Anzeigeempfehlung {slug}",
            "url": f"https://www.bild.de/news/{slug}",
            "score": 0.0,
            "scoreSource": "internal_score_api_missing",
            "scoreBeforeInternalApi": fallback_score,
            "scoreSourceBeforeInternalApi": "server_editorial_fallback",
            "pubDate": "2026-08-03T12:15:00+02:00",
        }
        decisions.append(
            {
                "candidate": candidate,
                "decision": {
                    "candidateId": candidate["url"],
                    "shouldNotify": False,
                    "score": 0.0,
                    "scoreSource": "internal_score_api_missing",
                    "mandatorySlotTop1Candidate": True,
                    "mandatoryTechnicalBlockerCodes": ["missing_canonical_score"],
                    "blockingReasons": [
                        "Synthetischer geaenderter Anzeigetext fuer fehlenden Score"
                    ],
                },
            }
        )

    recommendations = power_automate._scheduled_recommendations(
        {"selectedCandidate": selected, "decisions": decisions}
    )
    message_html = power_automate._scheduled_message_html(recommendations)

    assert len(recommendations) == 5
    assert recommendations[0]["title"] == selected["title"]
    assert recommendations[0]["pushScore"] == 91.4
    assert all(item["pushScore"] is None for item in recommendations[1:])
    assert len({item["url"] for item in recommendations}) == 5
    assert not any(item["url"].endswith("/fallback-1-duplicate") for item in recommendations)
    assert message_html.count("<strong>Top ") == 5
    assert message_html.count("Kanonischer Push Score steht noch aus.") == 4
    assert message_html.count("/100") == 1


def test_scheduled_display_fallback_never_ignores_a_second_hard_blocker():
    import app.routers.power_automate as power_automate

    selected = {
        "id": "000000000000000000000011",
        "title": "Synthetische kanonische Meldung",
        "url": "https://www.bild.de/news/synthetic-selected",
        "score": 90.0,
        "scoreSource": "internal_score_api",
        "pubDate": "2026-08-03T12:20:00+02:00",
    }
    blocked = {
        "id": "000000000000000000000012",
        "title": "Synthetische gesperrte Meldung",
        "url": "https://www.bild.de/news/synthetic-blocked",
        "score": 0.0,
        "scoreSource": "internal_score_api_missing",
        "scoreBeforeInternalApi": 88.0,
        "scoreSourceBeforeInternalApi": "server_editorial_fallback",
        "pubDate": "2026-08-03T12:15:00+02:00",
    }
    recommendations = power_automate._scheduled_recommendations(
        {
            "selectedCandidate": selected,
            "decisions": [
                {
                    "candidate": selected,
                    "decision": {
                        "shouldNotify": True,
                        "score": 90.0,
                        "scoreSource": "internal_score_api",
                        "blockingReasons": [],
                    },
                },
                {
                    "candidate": blocked,
                    "decision": {
                        "shouldNotify": False,
                        "scoreSource": "internal_score_api_missing",
                        "mandatorySlotTop1Candidate": True,
                        "mandatoryTechnicalBlockerCodes": [
                            "missing_canonical_score",
                            "teams_article_duplicate",
                        ],
                        "blockingReasons": [
                            "Kein frischer kanonischer Push-Balancer-Score fuer die Rangfolge",
                            "Identischer Artikel wurde bereits per Teams empfohlen",
                        ],
                    },
                },
            ],
        }
    )

    assert [item["url"] for item in recommendations] == [selected["url"]]


def test_selected_top1_wins_cms_dedup_even_when_duplicate_appears_first():
    import app.routers.power_automate as power_automate

    selected = {
        "id": "000000000000000000000031",
        "title": "Synthetische ausgewaehlte Top-Meldung",
        "url": "https://www.bild.de/news/synthetic-selected-cms",
        "score": 91.0,
        "scoreSource": "internal_score_api",
        "pubDate": "2026-08-03T12:20:00+02:00",
    }
    duplicate = {
        **selected,
        "title": "Synthetischer URL-Doppelgaenger",
        "url": "https://www.bild.de/news/synthetic-duplicate-cms",
        "score": 92.0,
    }
    extras = [
        {
            "id": f"00000000000000000000003{index}",
            "title": f"Synthetische eindeutige Meldung {index}",
            "url": f"https://www.bild.de/news/synthetic-unique-{index}",
            "score": 88.0 - index,
            "scoreSource": "internal_score_api",
            "pubDate": "2026-08-03T12:15:00+02:00",
        }
        for index in range(2, 6)
    ]

    def decision(candidate, *, selected_candidate=False):
        return {
            "candidate": candidate,
            "decision": {
                "candidateId": candidate["url"],
                "shouldNotify": selected_candidate,
                "score": candidate["score"],
                "scoreSource": "internal_score_api",
                "mandatorySlotTop1Candidate": True,
                "blockingReasons": (
                    []
                    if selected_candidate
                    else ["Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"]
                ),
            },
        }

    recommendations = power_automate._scheduled_recommendations(
        {
            "selectedCandidate": selected,
            "decisions": [
                decision(duplicate),
                decision(selected, selected_candidate=True),
                *(decision(candidate) for candidate in extras),
            ],
        }
    )

    assert len(recommendations) == 5
    assert recommendations[0]["url"] == selected["url"]
    assert duplicate["url"] not in {item["url"] for item in recommendations}


def test_scheduled_recommendations_collapse_bild_url_aliases_without_cms_id():
    import app.routers.power_automate as power_automate

    selected = {
        "title": "Synthetische ausgewaehlte URL",
        "url": "https://www.bild.de/news/synthetic-url-alias",
        "score": 91.0,
        "scoreSource": "internal_score_api",
        "pubDate": "2026-08-03T12:20:00+02:00",
    }
    alias = {
        **selected,
        "title": "Synthetischer AMP-Doppelgaenger",
        "url": "https://bild.de/news/synthetic-url-alias/amp?output=1",
        "score": 90.0,
    }
    extras = [
        {
            "title": f"Synthetische eindeutige URL {index}",
            "url": f"https://www.bild.de/news/synthetic-url-unique-{index}",
            "score": 89.0 - index,
            "scoreSource": "internal_score_api",
            "pubDate": "2026-08-03T12:15:00+02:00",
        }
        for index in range(1, 5)
    ]

    def item(candidate, *, winner=False):
        return {
            "candidate": candidate,
            "decision": {
                "shouldNotify": winner,
                "score": candidate["score"],
                "scoreSource": "internal_score_api",
                "mandatorySlotTop1Candidate": True,
                "blockingReasons": (
                    []
                    if winner
                    else ["Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"]
                ),
            },
        }

    recommendations = power_automate._scheduled_recommendations(
        {
            "selectedCandidate": selected,
            "decisions": [
                item(alias),
                item(selected, winner=True),
                *(item(candidate) for candidate in extras),
            ],
        }
    )

    assert len(recommendations) == 5
    assert recommendations[0]["url"] == selected["url"]
    assert alias["url"] not in {item["url"] for item in recommendations}


def test_claim_payload_never_exposes_an_alternative_outside_rendered_five():
    import app.routers.power_automate as power_automate

    selected = {
        "title": "Synthetische Top-Meldung",
        "url": "https://www.bild.de/news/synthetic-top-five-only",
        "category": "news",
        "score": 91.0,
    }
    recommendations = [
        {
            "title": f"Synthetische Empfehlung {index}",
            "url": (
                selected["url"]
                if index == 1
                else f"https://www.bild.de/news/synthetic-five-{index}"
            ),
            "pushScore": 92.0 - index,
            "publicationTs": SLOT_TS - 60,
        }
        for index in range(1, 6)
    ]
    outside_url = "https://www.bild.de/sport/synthetic-sixth-opposite"
    message = {
        "payload": {
            "articleTitle": selected["title"],
            "articleUrl": selected["url"],
            "category": "news",
            "pushScore": 91.0,
            "alternativeRecommendation": {
                "articleTitle": "Synthetische sechste Sportmeldung",
                "articleUrl": outside_url,
                "category": "sport",
                "pushScore": 80.0,
            },
        }
    }

    payload = power_automate._claim_response_payload(
        slot_ts=SLOT_TS,
        selected=selected,
        message=message,
        recommendations=recommendations,
    )

    assert payload["recommendationCount"] == 5
    assert payload["alternative"] is None
    assert outside_url not in str(payload)
    assert payload["messageHtml"].count("<strong>Top ") == 5


def test_preparation_replaces_a_retained_cms_alias_with_the_sixth_candidate(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    candidates = [
        {
            "id": f"synthetic-candidate-{index}",
            "url": f"https://www.bild.de/news/synthetic-prepared-{index}",
            "title": f"Synthetische vorbereitete Meldung {index}",
            "score": 96.0 - index,
            "scoreSource": "internal_score_api",
            "pubDate": "2026-08-03T12:20:00+02:00",
        }
        for index in range(1, 7)
    ]
    blocked = candidates[1]
    database.teams_alert_record(
        article_key="https://m.bild.de/news/synthetic-prepared-old-url",
        article_id=blocked["id"],
        article_url="https://m.bild.de/news/synthetic-prepared-old-url",
        article_title="Synthetische bereits unklare Meldung",
        title_hash="synthetic-retained-title-hash",
        score=90.0,
        predicted_or=0.05,
        candidate_updated_at=now_ts - 120,
        is_breaking=False,
        reason="Synthetische unklare Zustellung",
        status="delivery_uncertain",
        decision_ts=now_ts - 60,
    )
    monkeypatch.setattr(
        power_automate,
        "_memory_eligible_candidates",
        lambda items, **_kwargs: (items, {"skippedCandidates": 0, "reasons": {}}),
    )
    monkeypatch.setattr(
        power_automate,
        "build_teams_alert_context",
        lambda items, **_kwargs: {"nowTs": now_ts},
    )

    def evaluate(items, *_args, **_kwargs):
        selected = items[0]
        return {
            "selectedCandidate": selected,
            "decisions": [
                {
                    "candidate": candidate,
                    "decision": {
                        "candidateId": candidate_key(candidate),
                        "shouldNotify": candidate is selected,
                        "score": candidate["score"],
                        "scoreSource": "internal_score_api",
                        "mandatorySlotTop1Candidate": True,
                        "blockingReasons": (
                            []
                            if candidate is selected
                            else [
                                "Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"
                            ]
                        ),
                    },
                }
                for candidate in items
            ],
        }

    monkeypatch.setattr(power_automate, "evaluate_teams_alert_candidates", evaluate)
    config = replace(
        power_automate.TeamsAlertConfig(),
        enabled=True,
        require_internal_score_api=True,
        slot_gate_enabled=True,
    )

    prepared = power_automate._prepare_scheduled_recommendation_field(
        candidates,
        binding_slot={
            "ts": SLOT_TS,
            "label": "12:30",
            "slotRole": "power_automate_fixed",
        },
        decision_now=now_ts,
        config=config,
        dedup_history=[],
        dedup_history_authoritative=False,
    )

    assert prepared["ready"] is True, prepared
    assert len(prepared["recommendations"]) == 5
    assert candidate_key(blocked) in prepared["identityBlockers"]
    assert blocked["url"] not in {
        item["url"] for item in prepared["recommendations"]
    }
    assert candidates[-1]["url"] in {
        item["url"] for item in prepared["recommendations"]
    }
    assert teams_recommendation_slot_get(SLOT_TS) is None


def test_claim_does_not_prepare_before_the_official_slot(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS - 120
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-early-preparation-run"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["reason"] == "outside_window"


def test_claim_does_not_prepare_before_the_official_slot_by_any_amount(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS - 121
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-too-early-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "outside_window"}


def test_headline_command_returns_three_v14_pairs(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(
        power_automate,
        "_headline_article_context",
        lambda _article_id: {
            "url": "https://www.bild.de/politik/synthetischer-artikel",
            "title": "Bund beschliesst synthetisches Hilfspaket",
            "text": "Das Hilfspaket gilt ab Montag bundesweit.",
            "category": "politik",
        },
    )
    candidates = [
        {
            "titel": "Bund startet neues Hilfspaket",
            "zeile2": "Ab Montag gilt die neue Hilfe",
            "ansatz": "FAKT",
        },
        {
            "titel": "Neue Hilfe erreicht Millionen",
            "zeile2": "Bund setzt Paket am Montag um",
            "ansatz": "BETROFFENHEIT",
        },
        {
            "titel": "Hilfspaket gilt ab Montag",
            "zeile2": "Diese Haushalte profitieren",
            "ansatz": "FOLGE",
        },
    ]
    monkeypatch.setattr(
        "app.routers.misc._build_push_title_response",
        lambda _request: {
            "gewinner": {
                **candidates[0],
                "warum_dieser": "Kern und Folge stehen sofort fest.",
            },
            "alle_kandidaten": {"v1.4": candidates},
            "reasoning": "Kern und Folge stehen sofort fest.",
            "stufe": 2,
            "stufe_begruendung": "Entscheidung hat Zeit",
        },
    )

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "0123456789abcdef01234567"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["ready"] is True
    assert len(payload["suggestions"]) == 3
    assert payload["suggestions"][0] == {
        "type": "FAKT",
        "headline": "Bund startet neues Hilfspaket",
        "line2": "Ab Montag gilt die neue Hilfe",
    }
    assert "Headline-Vorschläge" in payload["messageHtml"]
    assert "bitte vor Versand prüfen" in payload["messageHtml"]


def test_headline_command_requires_auth_and_rejects_invalid_ids(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    unauthorized = client.post(
        "/api/v1/power-automate/teams/headline",
        json={"articleId": "0123456789abcdef01234567"},
    )
    invalid = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "not-an-id"},
    )

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422


def test_headline_command_extracts_one_id_from_teams_html(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    context_lookup = Mock(return_value=None)
    monkeypatch.setattr(power_automate, "_headline_article_context", context_lookup)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={
            "articleId": (
                "<p><span>/headline&nbsp;</span>"
                "0123456789ABCDEF01234567</p>"
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "article_not_found"
    context_lookup.assert_called_once_with("0123456789abcdef01234567")


def test_headline_command_rejects_ambiguous_teams_content(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={
            "articleId": (
                "/headline 0123456789abcdef01234567 "
                "fedcba987654321001234567"
            )
        },
    )

    assert response.status_code == 422


def test_headline_context_falls_back_to_complete_sitemap_lookup(monkeypatch):
    import app.routers.headline as headline
    import app.routers.power_automate as power_automate
    from app.cms.url_api import UrlApiNotConfigured

    monkeypatch.setattr(
        power_automate,
        "build_articles_payload",
        lambda **_kwargs: {"articles": []},
    )

    def unavailable_url_api(_article_id: str):
        raise UrlApiNotConfigured

    monkeypatch.setattr(power_automate, "get_canonical_article_url", unavailable_url_api)
    monkeypatch.setattr(
        headline,
        "resolve_headline_article",
        lambda article_id: {
            "articleId": article_id,
            "url": "https://www.bild.de/politik/synthetischer-artikel",
            "title": "Bund beschließt synthetisches Hilfspaket",
            "category": "politik",
            "contentType": "editorial",
        },
    )

    context = power_automate._headline_article_context(
        "0123456789abcdef01234567"
    )

    assert context == {
        "url": "https://www.bild.de/politik/synthetischer-artikel",
        "title": "Bund beschließt synthetisches Hilfspaket",
        "text": "",
        "category": "politik",
    }


def test_headline_command_returns_no_op_when_article_is_unknown(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(power_automate, "_headline_article_context", lambda _article_id: None)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "0123456789abcdef01234567"},
    )

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["reason"] == "article_not_found"
    assert "Artikel nicht gefunden" in response.json()["messageHtml"]


def test_claim_fails_closed_when_dedicated_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", "")
    monkeypatch.setattr(auth.config, "PUSH_TEAMS_WEBHOOK_URL", "")

    response = client.post(
        "/api/v1/power-automate/teams/claim",
        headers=HEADERS,
        json={"requestId": "synthetic-disabled-run"},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_claim_returns_only_the_minimal_top_opposite_and_html_contract(
    monkeypatch,
    tmp_db,
):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, alternative = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-contract-run"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert set(payload) == {
        "ready",
        "contractVersion",
        "slotId",
        "scheduledAt",
        "scheduledAtUtc",
        "expiresAt",
        "top",
        "alternative",
        "recommendationCount",
        "messageHtml",
    }
    assert payload["ready"] == "yes"
    assert payload["contractVersion"] == 2
    assert payload["recommendationCount"] == 5
    assert payload["slotId"] == f"teams-recommendation-{SLOT_TS}"
    assert payload["scheduledAt"] == "2026-08-03T12:30:00+02:00"
    assert payload["scheduledAtUtc"] == "2026-08-03T10:30:00Z"
    assert payload["expiresAt"] == "2026-08-03T12:45:00+02:00"
    assert payload["top"] == {
        "title": top["title"],
        "url": top["url"],
        "category": "news",
        "pushScore": 91.4,
        "isSport": False,
    }
    assert payload["alternative"] == {
        "title": alternative["title"],
        "url": alternative["url"],
        "category": "sport",
        "pushScore": 88.2,
        "isSport": True,
    }
    assert payload["messageHtml"].startswith("<h2>🔵 JETZT MÜSSEN (!) WIR PUSHEN</h2>")
    assert "</h2><br><br><p>Das sind meine 5 Empfehlungen" in payload["messageHtml"]
    assert "</p><br><br><p><strong>Top 1:</strong>" in payload["messageHtml"]
    assert "</p><br><br><p><strong>Top 2:</strong>" in payload["messageHtml"]
    assert "</p><br><br><p><strong>Top 5:</strong>" in payload["messageHtml"]
    assert payload["messageHtml"].count("<strong>Top ") == 5
    assert "Das sind meine 5 Empfehlungen" in payload["messageHtml"]
    assert (
        '<a href="https://editorial.one/push-balancer/bild/kandidaten">Push Balancer</a>'
        in payload["messageHtml"]
    )
    assert (
        f'<strong>Top 1:</strong> <a href="{top["url"]}">{top["title"]}</a>'
        in payload["messageHtml"]
    )
    assert "(03.08.2026, 12:20 Uhr)" in payload["messageHtml"]
    assert "<strong>Score:</strong> 91,4/100" in payload["messageHtml"]
    assert (
        f'<strong>Top 2:</strong> <a href="{alternative["url"]}">'
        f'{alternative["title"]}</a>' in payload["messageHtml"]
    )
    assert "<strong>Score:</strong> 88,2/100" in payload["messageHtml"]
    assert "webhook" not in response.text.casefold()
    assert "power-automate-key" not in response.text.casefold()


@pytest.mark.parametrize("terminal_status", ["sent", "delivery_uncertain"])
def test_recovery_claim_replays_only_its_owner_and_never_reopens_terminal_group(
    monkeypatch,
    tmp_db,
    terminal_status,
):
    import app.routers.power_automate as power_automate

    recovery_now = SLOT_TS + 600
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=recovery_now)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-recovery-owner"},
        )
        owner_retry = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-recovery-owner"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": first.json()["slotId"],
                "requestId": "synthetic-recovery-owner",
                "status": terminal_status,
            },
        )
        db_patch.setattr(power_automate.time, "time", lambda: SLOT_TS + 700)
        competing_run = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-recovery-competitor"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert first.status_code == 200
    assert first.json()["ready"] == "yes"
    assert first.json()["slotId"] == f"teams-recommendation-{SLOT_TS}"
    assert first.json()["expiresAt"] == "2026-08-03T12:45:00+02:00"
    assert first.json()["recommendationCount"] == 5
    assert first.json()["messageHtml"].count("<strong>Top ") == 5
    assert owner_retry.json() == first.json()
    assert receipt.status_code == 200
    assert competing_run.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert slot is not None
    assert slot["binding_slot_ts"] == SLOT_TS
    assert slot["status"] == terminal_status


def test_stale_concurrent_recovery_run_cannot_recycle_an_unresolved_group(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    first_claim_now = SLOT_TS + 30
    stale_run_now = first_claim_now + 301
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=first_claim_now)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-original-concurrent-run"},
        )
        original_slot = teams_recommendation_slot_get(SLOT_TS)
        original_group = teams_recommendation_slot_group_get(SLOT_TS)

        real_try_claim_group = (
            power_automate.teams_recommendation_slot_try_claim_group
        )
        observed_lease_seconds = []

        def observe_try_claim_group(*args, **kwargs):
            observed_lease_seconds.append(kwargs.get("lease_seconds"))
            return real_try_claim_group(*args, **kwargs)

        # Model a distinct run that passed both read-only checks before the
        # original run acquired its durable claim, then reached the atomic
        # claim after the old five-minute lease would have elapsed.
        db_patch.setattr(power_automate.time, "time", lambda: stale_run_now)
        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_get",
            lambda _slot_ts: None,
        )
        db_patch.setattr(
            power_automate,
            "teams_recommendation_article_identity_block_reasons",
            lambda *_args, **_kwargs: {},
        )
        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_try_claim_group",
            observe_try_claim_group,
        )
        stale_attempt = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-concurrent-run"},
        )
        retained_slot = teams_recommendation_slot_get(SLOT_TS)
        retained_group = teams_recommendation_slot_group_get(SLOT_TS)

    assert first.status_code == 200
    assert first.json()["ready"] == "yes"
    assert stale_run_now < SLOT_TS + 900
    assert stale_attempt.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert observed_lease_seconds == [900]
    assert retained_slot is not None
    assert original_slot is not None
    assert {
        key: retained_slot[key]
        for key in (
            "binding_slot_ts",
            "article_ref",
            "request_ref",
            "status",
            "claimed_at",
            "sent_at",
        )
    } == {
        key: original_slot[key]
        for key in (
            "binding_slot_ts",
            "article_ref",
            "request_ref",
            "status",
            "claimed_at",
            "sent_at",
        )
    }
    assert retained_group == original_group
    assert retained_slot["status"] == "sending"
    assert len(retained_group) == 5
    assert {item["status"] for item in retained_group} == {"sending"}


def test_failed_primary_receipt_can_recover_once_and_then_becomes_terminal(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    primary_now = SLOT_TS + 30
    recovery_now = SLOT_TS + 600
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=primary_now)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        primary_claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-primary-failure"},
        )
        failed_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": primary_claim.json()["slotId"],
                "requestId": "synthetic-primary-failure",
                "status": "failed",
            },
        )

        db_patch.setattr(power_automate.time, "time", lambda: recovery_now)
        recovery_claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-recovery-success"},
        )
        sent_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": recovery_claim.json()["slotId"],
                "requestId": "synthetic-recovery-success",
                "status": "sent",
            },
        )

        db_patch.setattr(power_automate.time, "time", lambda: SLOT_TS + 700)
        after_success = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-after-recovery-success"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)
        group = teams_recommendation_slot_group_get(SLOT_TS)
        sent_message_count = database.teams_alert_sent_count_since(SLOT_TS)

    assert primary_claim.status_code == 200
    assert primary_claim.json()["ready"] == "yes"
    assert failed_receipt.status_code == 200
    assert recovery_claim.status_code == 200
    assert recovery_claim.json()["ready"] == "yes"
    assert recovery_claim.json()["slotId"] == primary_claim.json()["slotId"]
    assert recovery_claim.json()["recommendationCount"] == 5
    assert sent_receipt.status_code == 200
    assert after_success.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert slot is not None
    assert slot["status"] == "sent"
    assert len(group) == 5
    assert {item["status"] for item in group} == {"sent"}
    assert sent_message_count == 1


def test_claim_stays_retryable_until_exactly_five_recommendations_exist(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate,
        "_scheduled_recommendations",
        lambda _evaluation: [
            {
                "title": f"Synthetische Meldung {index}",
                "url": f"https://www.bild.de/news/insufficient-{index}",
                "pushScore": 90.0 - index,
                "publicationTs": now_ts - 60,
            }
            for index in range(4)
        ],
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-insufficient-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)
        alert = database.teams_alert_get(top["url"])

    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "reason": "insufficient_recommendations",
    }
    assert slot is None
    assert alert is None


def test_claim_honors_fail_closed_live_history_requirement(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY",
        True,
    )
    monkeypatch.setattr(
        power_automate,
        "_refresh_push_history_for_dedup",
        lambda: {
            "history": [],
            "history_authoritative": False,
            "source": "synthetic-unavailable",
        },
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-history-unavailable-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "live_history_unavailable"}
    assert slot is None


def test_claim_passes_required_authoritative_history_to_decision_context(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate.app_config,
        "POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY",
        True,
    )
    synthetic_history = [
        {
            "message_id": "synthetic-live-history-entry",
            "title": "Andere synthetische Live-Meldung",
            "link": "https://www.bild.de/news/synthetic-other-live-item",
            "ts_num": now_ts - 60,
        }
    ]
    monkeypatch.setattr(
        power_automate,
        "_refresh_push_history_for_dedup",
        lambda: {
            "history": synthetic_history,
            "history_authoritative": True,
            "source": "synthetic-authoritative",
        },
    )
    observed_context: dict = {}

    def build_context(_candidates, **kwargs):
        observed_context.update(kwargs)
        return {"nowTs": now_ts}

    monkeypatch.setattr(power_automate, "build_teams_alert_context", build_context)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-authoritative-history-run"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] == "yes"
    assert observed_context["history"] == synthetic_history
    assert observed_context["history_authoritative"] is True


def test_claim_supports_sport_top_with_non_sport_alternative(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(
        monkeypatch,
        now_ts=now_ts,
        top_category="sport",
        alternative_category="politik",
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-sport-top-run"},
        )

    assert response.status_code == 200
    assert response.json()["top"]["isSport"] is True
    assert response.json()["alternative"]["isSport"] is False


def test_claim_uses_null_when_no_opposite_section_exists(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(
        monkeypatch,
        now_ts=now_ts,
        top_category="sport",
        include_alternative=False,
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-no-opposite-run"},
        )

    assert response.status_code == 200
    assert response.json()["top"]["isSport"] is True
    assert response.json()["alternative"] is None


def test_claim_is_slot_idempotent_and_receipt_finalizes_article_dedup(
    monkeypatch,
    tmp_db,
):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        duplicate = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        competing_run = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-competing-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": first.json()["slotId"],
                "requestId": "synthetic-idempotent-run",
                "status": "sent",
            },
        )
        replay_after_receipt = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        repeated_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": first.json()["slotId"],
                "requestId": "synthetic-idempotent-run",
                "status": "sent",
            },
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)
        group = teams_recommendation_slot_group_get(SLOT_TS)
        grouped_alerts = database.teams_alert_list_recent(limit=10)
        sent_message_count = database.teams_alert_sent_count_since(SLOT_TS)

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert competing_run.status_code == 200
    assert competing_run.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert receipt.status_code == 200
    assert replay_after_receipt.status_code == 200
    assert replay_after_receipt.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert receipt.headers["cache-control"] == "no-store"
    assert receipt.json() == {
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "status": "sent",
        "recordedAt": "2026-08-03T12:30:30+02:00",
    }
    assert repeated_receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "sent"
    assert alert["alert_count"] == 1
    assert [item["position"] for item in group] == [1, 2, 3, 4, 5]
    assert {item["status"] for item in group} == {"sent"}
    assert len(grouped_alerts) == 5
    assert {item["status"] for item in grouped_alerts} == {"sent"}
    assert {item["alert_count"] for item in grouped_alerts} == {1}
    assert sent_message_count == 1
    assert slot is not None
    assert slot["status"] == "sent"
    assert slot["request_ref"] != "synthetic-idempotent-run"
    assert "synthetic-idempotent-run" not in slot["claim_payload_json"]


def test_failed_receipt_releases_slot_without_recording_a_send(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-failed-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-failed-run",
                "status": "failed",
            },
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "transport_failed"
    assert alert["alert_count"] == 0
    assert slot is not None
    assert slot["status"] == "failed"


def test_uncertain_receipt_is_terminal_and_prevents_a_duplicate(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-uncertain-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-uncertain-run",
                "status": "delivery_uncertain",
            },
        )
        competing = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-after-uncertain-run"},
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert competing.status_code == 200
    assert competing.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert alert is not None
    assert alert["status"] == "delivery_uncertain"
    assert slot is not None
    assert slot["status"] == "delivery_uncertain"


def test_expected_selection_no_ops_stay_http_200(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate,
        "evaluate_teams_alert_candidates",
        lambda *_args, **_kwargs: {"selectedCandidate": None, "decisions": []},
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-no-candidate-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "no_candidate"}


def test_slot_close_during_selection_is_a_no_op(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(power_automate, "_power_automate_slot_open", lambda *_a, **_k: False)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-slot-closed-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "slot_closed"}


def test_fixed_power_automate_slot_ignores_legacy_date_delay(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    delayed_config = replace(
        power_automate.TeamsAlertConfig(),
        slot_delay_date="2026-08-03",
        slot_delay_from="12:30",
        slot_delay_minutes=15,
    )
    observed_dispatch_config: dict = {}

    def evaluate_with_config(candidates, _context, config):
        observed_dispatch_config.update(
            {
                "slot_delay_date": config.slot_delay_date,
                "slot_delay_from": config.slot_delay_from,
                "slot_delay_minutes": config.slot_delay_minutes,
            }
        )
        top = candidates[0]
        return {
            "selectedCandidate": top,
            "decisions": [
                {
                    "candidate": candidate,
                    "decision": {
                        "candidateId": candidate["url"],
                        "shouldNotify": index == 0,
                        "score": candidate["score"],
                        "scoreSource": "internal_score_api",
                        "mandatorySlotTop1Candidate": True,
                        "summary": ("Verbindlicher Push-Balancer-Top-1 im festen Slot"),
                        "blockingReasons": (
                            []
                            if index == 0
                            else ["Staerkerer Kandidat vorhanden: " "vollstaendig geprueftes Feld"]
                        ),
                    },
                }
                for index, candidate in enumerate(candidates)
            ],
        }

    monkeypatch.setattr(power_automate, "TeamsAlertConfig", lambda: delayed_config)
    monkeypatch.setattr(power_automate, "evaluate_teams_alert_candidates", evaluate_with_config)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-fixed-slot-run"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] == "yes"
    assert observed_dispatch_config == {
        "slot_delay_date": "",
        "slot_delay_from": "",
        "slot_delay_minutes": 0,
    }


def test_initial_claim_needs_delivery_budget_before_slot_expiry(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 899
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        nearly_expired = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-nearly-expired-run"},
        )
        db_patch.setattr(power_automate.time, "time", lambda: SLOT_TS + 901)
        after_window = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-nearly-expired-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert nearly_expired.status_code == 200
    assert nearly_expired.json() == {"ready": False, "reason": "slot_closed"}
    assert after_window.status_code == 200
    assert after_window.json() == {"ready": False, "reason": "outside_window"}
    assert slot is None


def test_receipt_is_bound_to_the_claim_run_across_slot_expiry(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    now_ts = SLOT_TS + 30
    late_ts = SLOT_TS + 901
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-owner-run"},
        )
        wrong_owner = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-other-run",
                "status": "sent",
            },
        )
        db_patch.setattr(power_automate.time, "time", lambda: late_ts)
        db_patch.setattr(slot_claims.time, "time", lambda: late_ts)
        correct_owner = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-owner-run",
                "status": "sent",
            },
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert wrong_owner.status_code == 409
    assert correct_owner.status_code == 200
    assert slot is not None
    assert slot["status"] == "sent"
    assert slot["request_ref"]
    assert slot["claim_payload_json"] == ""


def test_late_receipt_lookup_is_not_limited_to_recent_dashboard_rows(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    now_ts = SLOT_TS + 30
    late_ts = SLOT_TS + 901
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-retained-lookup-run"},
        )
        for index in range(101):
            database.teams_alert_record(
                article_key=f"https://www.bild.de/news/synthetic-newer-{index}",
                article_id=f"synthetic-newer-{index}",
                article_url=f"https://www.bild.de/news/synthetic-newer-{index}",
                article_title=f"Synthetische neuere Meldung {index}",
                title_hash=f"synthetic-newer-hash-{index}",
                score=80.0,
                predicted_or=0.05,
                candidate_updated_at=late_ts,
                is_breaking=False,
                reason="synthetic",
                status="failed",
                decision_ts=late_ts + index,
            )
        db_patch.setattr(power_automate.time, "time", lambda: late_ts)
        db_patch.setattr(slot_claims.time, "time", lambda: late_ts)
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-retained-lookup-run",
                "status": "sent",
            },
        )
        alert = database.teams_alert_get(top["url"])

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "sent"


def test_late_receipt_cannot_finalize_a_later_article_claim(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    first_claimed_at = SLOT_TS + 30
    second_slot_ts = SLOT_TS + 600
    second_claimed_at = second_slot_ts + 30
    article_key = "https://www.bild.de/news/synthetic-cross-slot"
    payload = {
        "ready": True,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "top": {
            "title": "Synthetische Cross-Slot-Meldung",
            "url": article_key,
            "category": "news",
            "pushScore": 91.0,
            "isSport": False,
        },
        "alternative": None,
        "messageHtml": "<p>Synthetische Cross-Slot-Meldung</p>",
    }
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    def claim_article(decision_ts: int) -> dict:
        return database.teams_alert_try_claim_send(
            article_key=article_key,
            article_id="synthetic-cross-slot",
            article_url=article_key,
            article_title="Synthetische Cross-Slot-Meldung",
            title_hash="synthetic-title-hash",
            score=91.0,
            predicted_or=0.06,
            candidate_updated_at=decision_ts - 60,
            is_breaking=False,
            reason="synthetic",
            decision_ts=decision_ts,
            alert_cooldown_minutes=0,
            global_cooldown_minutes=0,
            in_progress_cooldown_minutes=5,
            failed_cooldown_minutes=0,
            transport_failure_cooldown_minutes=0,
        )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first_slot = teams_recommendation_slot_try_claim(
            SLOT_TS,
            article_key=article_key,
            request_id="synthetic-old-run",
            claim_payload=payload,
            now_ts=first_claimed_at,
        )
        first_article = claim_article(first_claimed_at)
        second_slot = teams_recommendation_slot_try_claim(
            second_slot_ts,
            article_key=article_key,
            request_id="synthetic-new-run",
            claim_payload={
                **payload,
                "slotId": f"teams-recommendation-{second_slot_ts}",
            },
            now_ts=second_claimed_at,
        )
        second_article = claim_article(second_claimed_at)
        db_patch.setattr(power_automate.time, "time", lambda: second_claimed_at + 1)
        stale_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": f"teams-recommendation-{SLOT_TS}",
                "requestId": "synthetic-old-run",
                "status": "sent",
            },
        )
        first_state = teams_recommendation_slot_get(SLOT_TS)
        second_state = teams_recommendation_slot_get(second_slot_ts)
        alert = database.teams_alert_get(article_key)

    assert first_slot["claimed"] is True
    assert first_article["claimed"] is True
    assert second_slot["claimed"] is True
    assert second_article["claimed"] is True
    assert stale_receipt.status_code == 409
    assert first_state is not None and first_state["status"] == "sending"
    assert second_state is not None and second_state["status"] == "sending"
    assert alert is not None
    assert alert["status"] == "sending"
    assert alert["last_decision_ts"] == second_claimed_at
    assert alert["alert_count"] == 0


def test_parallel_sent_receipts_increment_alert_count_once(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-parallel-receipt-run"},
        )
        receipt_body = {
            "slotId": claim.json()["slotId"],
            "requestId": "synthetic-parallel-receipt-run",
            "status": "sent",
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda _index: client.post(
                        "/api/v1/power-automate/teams/receipt",
                        headers=HEADERS,
                        json=receipt_body,
                    ),
                    range(2),
                )
            )
        alert = database.teams_alert_get(top["url"])

    assert [response.status_code for response in responses] == [200, 200]
    assert alert is not None
    assert alert["status"] == "sent"
    assert alert["alert_count"] == 1


def test_legacy_replay_releases_owned_orphan_before_a_fresh_group_claim(
    monkeypatch,
    tmp_db,
):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, alternative = _patch_successful_claim(monkeypatch, now_ts=now_ts)
    raw_url = top["url"].upper() + "/?wtmc=synthetic"
    article_key = candidate_key({"url": raw_url})
    payload = {
        "ready": "yes",
        "contractVersion": 2,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "scheduledAt": "2026-08-03T12:30:00+02:00",
        "scheduledAtUtc": "2026-08-03T10:30:00Z",
        "expiresAt": "2026-08-03T12:35:00+02:00",
        "top": {
            "title": top["title"],
            "url": raw_url,
            "category": top["category"],
            "pushScore": top["score"],
            "isSport": False,
        },
        "alternative": {
            "title": alternative["title"],
            "url": alternative["url"],
            "category": alternative["category"],
            "pushScore": alternative["score"],
            "isSport": True,
        },
        "recommendationCount": 5,
        "messageHtml": "".join(
            f"<p><strong>Top {index}:</strong> Synthetic replay {index}</p>"
            for index in range(1, 6)
        ),
    }

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        slot_claim = teams_recommendation_slot_try_claim(
            SLOT_TS,
            article_key=article_key,
            request_id="synthetic-repair-run",
            claim_payload=payload,
            now_ts=now_ts,
        )
        article_claim = database.teams_alert_try_claim_send(
            article_key=article_key,
            article_id=article_key,
            article_url=raw_url,
            article_title=top["title"],
            title_hash="synthetic-title-hash",
            score=top["score"],
            predicted_or=0.0,
            candidate_updated_at=now_ts - 60,
            is_breaking=False,
            reason="synthetic legacy claim",
            decision_ts=now_ts,
            alert_cooldown_minutes=0,
            global_cooldown_minutes=0,
            in_progress_cooldown_minutes=5,
            failed_cooldown_minutes=0,
            transport_failure_cooldown_minutes=0,
        )
        stale_replay = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-repair-run"},
        )
        released_slot = teams_recommendation_slot_get(SLOT_TS)
        released_alert = database.teams_alert_get(article_key)
        fresh_claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-fresh-group-run"},
        )

    assert slot_claim["claimed"] is True
    assert article_claim["claimed"] is True
    assert stale_replay.status_code == 200
    assert stale_replay.json() == {
        "ready": False,
        "reason": "article_claim_unavailable",
    }
    assert released_slot is not None
    assert released_slot["status"] == "failed"
    assert released_alert is not None
    assert released_alert["status"] == "claim_released"
    assert fresh_claim.status_code == 200
    assert fresh_claim.json()["ready"] == "yes"
    assert fresh_claim.json()["recommendationCount"] == 5


def test_replay_rejects_and_releases_a_legacy_short_contract(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)
    article_key = candidate_key(top)
    legacy_payload = {
        "ready": True,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "top": {
            "title": top["title"],
            "url": top["url"],
            "category": top["category"],
            "pushScore": top["score"],
            "isSport": False,
        },
        "alternative": None,
        "messageHtml": "<p><strong>Top 1:</strong> Legacy replay</p>",
    }

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        slot_claim = teams_recommendation_slot_try_claim(
            SLOT_TS,
            article_key=article_key,
            request_id="synthetic-legacy-replay-run",
            claim_payload=legacy_payload,
            now_ts=now_ts,
        )
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-legacy-replay-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert slot_claim["claimed"] is True
    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "claim_contract_stale"}
    assert slot is not None
    assert slot["status"] == "failed"
    assert slot["request_ref"] == ""
    assert slot["claim_payload_json"] == ""


def test_replay_contract_validator_rejects_malformed_types_without_raising():
    import app.routers.power_automate as power_automate

    base_payload = {
        "ready": "yes",
        "contractVersion": 2,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "recommendationCount": 5,
        "messageHtml": "".join(
            f"<p><strong>Top {index}:</strong> Synthetic</p>" for index in range(1, 6)
        ),
    }
    invalid_values = ("2", 2.0, True, {}, float("inf"))

    for field in ("contractVersion", "recommendationCount"):
        for value in invalid_values:
            assert (
                power_automate._valid_scheduled_replay_payload(
                    {**base_payload, field: value},
                    slot_ts=SLOT_TS,
                )
                is False
            )


def test_stale_replay_cannot_downgrade_a_sent_slot(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-replay-run"},
        )
        teams_recommendation_slot_record_group_receipt(
            SLOT_TS,
            status="sent",
            request_id="synthetic-stale-replay-run",
            now_ts=now_ts,
        )
        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_replay",
            lambda *_args, **_kwargs: claim.json(),
        )
        replay = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-replay-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert replay.status_code == 200
    assert replay.json() == {"ready": False, "reason": "slot_already_claimed"}
    assert slot is not None
    assert slot["status"] == "sent"


def test_stale_replay_cannot_take_over_another_runs_reclaimed_slot(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        owner_b = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-owner-b-run"},
        )
        original_replay = power_automate.teams_recommendation_slot_replay
        calls = 0

        def stale_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return owner_b.json()
            return original_replay(*args, **kwargs)

        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_replay",
            stale_once,
        )
        stale_a = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-owner-a-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert owner_b.status_code == 200
    assert owner_b.json()["ready"] == "yes"
    assert stale_a.status_code == 200
    assert stale_a.json() == {"ready": False, "reason": "slot_already_claimed"}
    assert slot is not None
    assert slot["status"] == "sending"
