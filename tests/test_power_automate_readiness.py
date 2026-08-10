"""Privacy and auth tests for the scheduled Teams cutover readiness route."""

from __future__ import annotations

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.main import app


POWER_AUTOMATE_KEY = "synthetic-readiness-key"
HEADERS = {"X-Power-Automate-Key": POWER_AUTOMATE_KEY}
READINESS_PATH = "/api/v1/power-automate/teams/readiness"
BERLIN = ZoneInfo("Europe/Berlin")

client = TestClient(app, raise_server_exceptions=True)


def _full_readiness_fixture() -> dict[str, Any]:
    return {
        "ready": True,
        "berlinTime": "2026-08-09 18:50",
        "teamsAlertsEnabled": True,
        "transportMode": "power_automate_scheduled",
        "backgroundSenderEnabled": False,
        "powerAutomateConfigured": True,
        "durableStorage": {
            "required": True,
            "durable": True,
            "mode": "persistent_disk",
            "mountPath": "/data/private",
        },
        "webhookConfigured": False,
        "quietHoursActive": False,
        "quietHoursReason": None,
        "volume": {"min": 12, "max": 12},
        "scoreApi": {
            "ok": True,
            "checkedCandidates": 47,
            "sources": {"internal_score_api": 47},
            "candidate": {
                "title": "must-not-leak",
                "url": "https://example.invalid/private-article",
            },
        },
        "exactFive": {
            "contractOk": True,
            "recommendationCount": 5,
            "top1Canonical": True,
            "reason": "ready",
            "articleIds": ["must-not-leak"],
        },
        "pushHistory": {
            "ok": True,
            "required": False,
            "historyAuthoritative": False,
            "fallbackMode": "durable_slot_and_receipt_dedup",
            "source": "private-source",
            "pushesToday": 23,
            "latestArticleUrl": "https://example.invalid/private-history",
        },
        "slots": {
            "ok": True,
            "plannedToday": 12,
            "labels": [
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
            ],
            "nextSlot": {"label": "20:08", "requestId": "must-not-leak"},
        },
        "runtime": {"lastSendTs": 1786300000, "accountEmail": "private@example.invalid"},
        "configurationProblems": [
            (
                "PUSH_TEAMS_MAX_ALERTS_PER_DAY liegt unter MIN - das Tagesziel ist "
                "widerspruechlich."
            ),
            "dynamic secret: must-not-leak@example.invalid",
        ],
        "secret": "must-not-leak",
        "apiKey": "must-not-leak",
    }


def _assert_no_forbidden_keys(value: Any) -> None:
    forbidden = ("title", "url", "article", "candidate", "secret", "key", "email")
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).casefold()
            assert not any(fragment in normalized for fragment in forbidden), key
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def test_readiness_requires_the_dedicated_power_automate_key(monkeypatch):
    import app.routers.health as health_router

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(
        health_router,
        "build_teams_readiness_payload",
        lambda: pytest.fail("unauthenticated readiness must not be evaluated"),
    )

    missing = client.get(READINESS_PATH)
    wrong = client.get(
        READINESS_PATH,
        headers={"X-Power-Automate-Key": "wrong-synthetic-key"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_readiness_fails_closed_when_the_dedicated_key_is_not_configured(
    monkeypatch,
):
    import app.routers.health as health_router

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", "")
    monkeypatch.setattr(
        health_router,
        "build_teams_readiness_payload",
        lambda: pytest.fail("disabled readiness must not be evaluated"),
    )

    response = client.get(READINESS_PATH, headers=HEADERS)

    assert response.status_code == 503


def test_readiness_returns_only_the_allowlisted_shared_values(monkeypatch):
    import app.routers.health as health_router
    import app.routers.power_automate as power_automate

    full = _full_readiness_fixture()
    latest_slot = {
        "label": "18:49",
        "state": "sent",
        "receiptRecorded": True,
        "timingState": "terminal",
        "recoveryEligible": False,
    }
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(health_router, "build_teams_readiness_payload", lambda: full)
    monkeypatch.setattr(
        power_automate,
        "_latest_due_power_automate_slot",
        lambda: latest_slot,
    )

    response = client.get(READINESS_PATH, headers=HEADERS)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["vary"] == "X-Power-Automate-Key"
    data = response.json()
    assert data == {
        "ready": full["ready"],
        "teamsAlertsEnabled": full["teamsAlertsEnabled"],
        "transportMode": full["transportMode"],
        "backgroundSenderEnabled": full["backgroundSenderEnabled"],
        "powerAutomateConfigured": full["powerAutomateConfigured"],
        "durableStorage": {
            key: full["durableStorage"][key]
            for key in ("required", "durable", "mode")
        },
        "scoreApi": {"ok": full["scoreApi"]["ok"]},
        "exactFive": {
            key: full["exactFive"][key]
            for key in ("contractOk", "recommendationCount", "top1Canonical")
        },
        "pushHistory": {
            key: full["pushHistory"][key]
            for key in ("ok", "required", "historyAuthoritative", "fallbackMode")
        },
        "slots": {
            key: full["slots"][key]
            for key in ("ok", "plannedToday", "labels")
        },
        "recovery": {
            "enabled": True,
            "configurationValid": True,
            "graceSeconds": 600,
        },
        "deliveryHealth": {"ok": True, "attentionRequired": False},
        "configurationProblems": [full["configurationProblems"][0]],
        "latestSlot": latest_slot,
    }
    _assert_no_forbidden_keys(data)
    serialized = response.text.casefold()
    assert "must-not-leak" not in serialized
    assert "example.invalid" not in serialized


def test_readiness_allowlists_the_transport_owner_conflict(monkeypatch):
    import app.routers.health as health_router
    import app.routers.power_automate as power_automate

    full = _full_readiness_fixture()
    full["ready"] = False
    full["configurationProblems"] = [auth.config.TEAMS_TRANSPORT_OWNER_CONFLICT]
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(health_router, "build_teams_readiness_payload", lambda: full)
    monkeypatch.setattr(power_automate, "_latest_due_power_automate_slot", lambda: None)

    response = client.get(READINESS_PATH, headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["configurationProblems"] == [
        auth.config.TEAMS_TRANSPORT_OWNER_CONFLICT
    ]


def test_full_readiness_reports_the_transport_owner_conflict(monkeypatch):
    from dataclasses import replace

    import app.notifications.teams as teams_module
    import app.routers.feed as feed_router
    import app.routers.health as health_router

    config = replace(
        teams_module.TeamsAlertConfig(),
        enabled=True,
        webhook_url="https://example.invalid/synthetic-webhook",
    )
    now_ts = int(dt.datetime(2026, 8, 9, 18, 50, tzinfo=BERLIN).timestamp())
    monkeypatch.setattr(health_router, "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED", True)
    monkeypatch.setattr(health_router, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(health_router, "PUSH_DB_DURABLE", True)
    monkeypatch.setattr(health_router, "durable_db_storage_available", lambda: True)
    monkeypatch.setattr(health_router.time, "time", lambda: now_ts)
    monkeypatch.setattr(teams_module, "TeamsAlertConfig", lambda: config)
    monkeypatch.setattr(
        feed_router,
        "build_articles_payload",
        lambda **_kwargs: {"articles": []},
    )
    monkeypatch.setattr(
        teams_module,
        "_refresh_push_history_for_dedup",
        lambda: {
            "history": [],
            "history_authoritative": True,
            "source": "synthetic",
            "snapshot_age_seconds": 0,
        },
    )
    monkeypatch.setattr(
        teams_module,
        "build_teams_alert_context",
        lambda *_args, **_kwargs: {"lastPushTs": 0},
    )
    monkeypatch.setattr(
        teams_module,
        "_daily_runtime_opportunities",
        lambda *_args, **_kwargs: [
            {"ts": now_ts + index * 60, "label": f"synthetic-{index}"}
            for index in range(12)
        ],
    )
    monkeypatch.setattr(teams_module, "_quiet_hours_reason", lambda *_args: "")
    monkeypatch.setattr(teams_module, "channel_configuration_problems", lambda _config: [])
    monkeypatch.setattr(
        teams_module,
        "channel_health",
        lambda *_args, **_kwargs: {"healthy": True, "status": "synthetic"},
    )

    result = health_router.build_teams_readiness_payload()

    assert result["ready"] is False
    assert result["configurationProblems"] == [
        auth.config.TEAMS_TRANSPORT_OWNER_CONFLICT
    ]


@pytest.mark.parametrize(
    ("delivery", "expected"),
    [
        (
            None,
            {
                "label": "18:49",
                "state": "unclaimed",
                "receiptRecorded": False,
                "timingState": "primary_open",
                "recoveryEligible": False,
            },
        ),
        (
            {"status": "sending", "receiptRecorded": False},
            {
                "label": "18:49",
                "state": "sending",
                "receiptRecorded": False,
                "timingState": "awaiting_receipt",
                "recoveryEligible": False,
            },
        ),
        (
            {"status": "sent", "receiptRecorded": True},
            {
                "label": "18:49",
                "state": "sent",
                "receiptRecorded": True,
                "timingState": "terminal",
                "recoveryEligible": False,
            },
        ),
        (
            {"status": "delivery_uncertain", "receiptRecorded": True},
            {
                "label": "18:49",
                "state": "delivery_uncertain",
                "receiptRecorded": True,
                "timingState": "terminal",
                "recoveryEligible": False,
            },
        ),
        (
            {"status": "failed", "receiptRecorded": False},
            {
                "label": "18:49",
                "state": "failed",
                "receiptRecorded": False,
                "timingState": "primary_open",
                "recoveryEligible": False,
            },
        ),
        (
            {"status": "unexpected", "receiptRecorded": True},
            {
                "label": "18:49",
                "state": "other",
                "timingState": "blocked",
                "recoveryEligible": False,
            },
        ),
    ],
)
def test_latest_slot_uses_only_the_hard_state_enum(monkeypatch, delivery, expected):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    now_ts = int(dt.datetime(2026, 8, 9, 18, 50, tzinfo=BERLIN).timestamp())
    monkeypatch.setattr(
        slot_claims,
        "teams_recommendation_slot_delivery_state_read_only",
        lambda _slot_ts: delivery,
    )

    result = power_automate._latest_due_power_automate_slot(now_ts)

    assert result == expected
    _assert_no_forbidden_keys(result)


@pytest.mark.parametrize(
    ("delivery", "offset_seconds", "timing_state", "recovery_eligible"),
    [
        (None, 300, "recovery_open", True),
        (None, 899, "recovery_open", True),
        (None, 900, "missed", False),
        (
            {"status": "sending", "receiptRecorded": False},
            300,
            "awaiting_receipt",
            False,
        ),
        (
            {"status": "sending", "receiptRecorded": False},
            899,
            "awaiting_receipt",
            False,
        ),
        (
            {"status": "sending", "receiptRecorded": False},
            900,
            "overdue_unresolved",
            False,
        ),
    ],
)
def test_latest_slot_timing_boundaries_are_half_open(
    monkeypatch,
    delivery,
    offset_seconds,
    timing_state,
    recovery_eligible,
):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    slot_ts = int(dt.datetime(2026, 8, 9, 18, 49, tzinfo=BERLIN).timestamp())
    monkeypatch.setattr(
        slot_claims,
        "teams_recommendation_slot_delivery_state_read_only",
        lambda _slot_ts: delivery,
    )
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

    result = power_automate._latest_due_power_automate_slot(
        slot_ts + offset_seconds
    )

    assert result is not None
    assert result["label"] == "18:49"
    assert result["state"] == ("unclaimed" if delivery is None else "sending")
    assert result["timingState"] == timing_state
    assert result["recoveryEligible"] is recovery_eligible
    assert result["receiptRecorded"] is False


@pytest.mark.parametrize(
    ("latest_slot", "expected_ready", "expected_health"),
    [
        (
            {
                "label": "18:49",
                "state": "unclaimed",
                "receiptRecorded": False,
                "timingState": "missed",
                "recoveryEligible": False,
            },
            False,
            {"ok": False, "attentionRequired": True},
        ),
        (
            {
                "label": "18:49",
                "state": "sending",
                "receiptRecorded": False,
                "timingState": "overdue_unresolved",
                "recoveryEligible": False,
            },
            False,
            {"ok": False, "attentionRequired": True},
        ),
        (
            {
                "label": "18:49",
                "state": "delivery_uncertain",
                "receiptRecorded": True,
                "timingState": "terminal",
                "recoveryEligible": False,
            },
            False,
            {"ok": False, "attentionRequired": True},
        ),
        (
            {
                "label": "18:49",
                "state": "sent",
                "receiptRecorded": True,
                "timingState": "terminal",
                "recoveryEligible": False,
            },
            True,
            {"ok": True, "attentionRequired": False},
        ),
        (
            {
                "label": "18:49",
                "state": "unclaimed",
                "receiptRecorded": False,
                "timingState": "recovery_open",
                "recoveryEligible": True,
            },
            True,
            {"ok": True, "attentionRequired": False},
        ),
    ],
    ids=(
        "missed",
        "overdue-unresolved",
        "delivery-uncertain",
        "sent",
        "recoverable",
    ),
)
def test_readiness_delivery_health_controls_top_level_ready(
    monkeypatch,
    latest_slot,
    expected_ready,
    expected_health,
):
    import app.routers.health as health_router
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
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
    monkeypatch.setattr(
        health_router,
        "build_teams_readiness_payload",
        _full_readiness_fixture,
    )
    monkeypatch.setattr(
        power_automate,
        "_latest_due_power_automate_slot",
        lambda: latest_slot,
    )

    response = client.get(READINESS_PATH, headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["ready"] is expected_ready
    assert response.json()["deliveryHealth"] == expected_health
