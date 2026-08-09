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
    latest_slot = {"label": "18:49", "state": "sent", "receiptRecorded": True}
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
        "configurationProblems": [full["configurationProblems"][0]],
        "latestSlot": latest_slot,
    }
    _assert_no_forbidden_keys(data)
    serialized = response.text.casefold()
    assert "must-not-leak" not in serialized
    assert "example.invalid" not in serialized


@pytest.mark.parametrize(
    ("delivery", "expected"),
    [
        (None, {"label": "18:49", "state": "unclaimed", "receiptRecorded": False}),
        (
            {"status": "sending", "receiptRecorded": False},
            {"label": "18:49", "state": "sending", "receiptRecorded": False},
        ),
        (
            {"status": "sent", "receiptRecorded": True},
            {"label": "18:49", "state": "sent", "receiptRecorded": True},
        ),
        (
            {"status": "delivery_uncertain", "receiptRecorded": True},
            {
                "label": "18:49",
                "state": "delivery_uncertain",
                "receiptRecorded": True,
            },
        ),
        ({"status": "failed", "receiptRecorded": False}, {"label": "18:49", "state": "other"}),
        ({"status": "unexpected", "receiptRecorded": True}, {"label": "18:49", "state": "other"}),
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
