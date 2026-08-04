"""End-to-End: ein voller Worker-Zyklus komponiert alle Bausteine korrekt.

Deckt ab, was Unit-Tests je Gate nicht sehen: dass Kandidatenbewertung,
Zustellung, Persistenz und Herzschlag im echten Zykluspfad zusammenspielen -
und dass ein Transportfehler den Zyklus nicht toetet, sondern sichtbar macht.
"""

import datetime as dt
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.notifications.teams import (
    TeamsAlertConfig,
    _CHANNEL_HEALTH,
    _CHANNEL_HEALTH_LOCK,
    channel_health,
    run_teams_alert_cycle,
)

import pytest

from tests.test_teams_notifications import _candidate, _history

# Freitag 21:23 ist eine verbindliche Raster-Entscheidung (+30s = faellig).
_SLOT_TS = int(dt.datetime(2026, 6, 19, 21, 26, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())


@pytest.fixture(autouse=True)
def _reset_send_memory():
    from app.notifications import teams as teams_module

    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()
    yield
    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()


@pytest.fixture(autouse=True)
def _reset_health():
    with _CHANNEL_HEALTH_LOCK:
        snapshot = dict(_CHANNEL_HEALTH)
        for key in _CHANNEL_HEALTH:
            _CHANNEL_HEALTH[key] = 0 if isinstance(_CHANNEL_HEALTH[key], int) else None
        _CHANNEL_HEALTH["lastCycleError"] = ""
    yield
    with _CHANNEL_HEALTH_LOCK:
        _CHANNEL_HEALTH.update(snapshot)


def _strong_candidate():
    return _candidate(
        url="https://www.bild.de/politik/e2e-strong",
        score=94.0,
        scoreSource="internal_score_api",
        predictedOR=0.08,
        pubDate=dt.datetime.fromtimestamp(_SLOT_TS - 6 * 60, ZoneInfo("Europe/Berlin")).isoformat(),
        recommendedText="Rentenpaket beschlossen: Das gilt jetzt fuer Millionen",
    )


def _run_cycle_with(send_result, *, config=None):
    """Zyklus fahren, externe Grenzen gemockt (Feed, Push-Historie, Webhook)."""
    config = config or TeamsAlertConfig(
        enabled=True,
        webhook_url="https://teams.example.test/webhook",
        require_internal_score_api=False,
        agent_review_enabled=False,
        min_alert_score=60.0,
        min_editorial_score=60.0,
        min_or=4.0,
        heartbeat_enabled=False,
        daily_schedule_send_enabled=False,
        live_push_posts_enabled=False,
    )
    payload = {"articles": [_strong_candidate()]}
    refresh = {"history_authoritative": True, "source": "live", "snapshot_age_seconds": 0}
    # Die echte Kontextfunktion VOR dem Patchen festhalten (sonst Rekursion).
    from app.notifications.teams import build_teams_alert_context as real_ctx

    def _ctx(cands, **kwargs):
        kwargs.setdefault("history", _history(minutes_since_last_push=90, now_ts=_SLOT_TS))
        kwargs.setdefault("now_ts", _SLOT_TS)
        kwargs["history_authoritative"] = True
        return real_ctx(cands, **kwargs)

    with (
        patch("app.notifications.teams.time.time", return_value=_SLOT_TS),
        patch("app.notifications.teams.TeamsAlertConfig", return_value=config),
        patch("app.notifications.teams._refresh_push_history_for_dedup", return_value=refresh),
        patch("app.routers.feed.build_articles_payload", return_value=payload),
        patch("app.notifications.teams.build_teams_alert_context", side_effect=_ctx),
        patch("app.notifications.teams.send_teams_notification", return_value=send_result) as send,
    ):
        result = run_teams_alert_cycle()
    return result, send


def test_full_cycle_sends_and_records_heartbeat(tmp_db):
    result, send = _run_cycle_with({"ok": True, "status": 200, "attempts": 1})

    assert result["ok"] is True
    assert result["sent"] is True
    send.assert_called_once()

    health = channel_health(TeamsAlertConfig(enabled=True,
                                             webhook_url="https://x"), now_ts=_SLOT_TS)
    # Der Zyklus muss seinen Herzschlag hinterlassen haben.
    assert health["cycleCount"] == 1
    assert health["lastCycleOk"] is True
    assert health["lastSendTs"] > 0


def test_full_cycle_survives_a_transport_failure(tmp_db):
    """Ein Webhook-Ausfall darf den Zyklus nicht abbrechen - nur sichtbar machen."""
    result, send = _run_cycle_with(
        {"ok": False, "error": "timeout", "transient": True, "attempts": 3}
    )

    # Der Zyklus selbst laeuft sauber durch (ok=True), der Versand schlug fehl.
    assert result["ok"] is True
    assert result["sent"] is False
    send.assert_called_once()

    health = channel_health(TeamsAlertConfig(enabled=True,
                                             webhook_url="https://x"), now_ts=_SLOT_TS)
    assert health["cycleCount"] == 1
    # Herzschlag ist da (Worker lebt), aber es ging nichts raus.
    assert health["lastSendTs"] == 0


def test_cycle_exception_is_recorded_as_a_failed_heartbeat(tmp_db):
    config = TeamsAlertConfig(enabled=True, webhook_url="https://teams.example.test/webhook")
    with (
        patch("app.notifications.teams.time.time", return_value=_SLOT_TS),
        patch("app.notifications.teams.TeamsAlertConfig", return_value=config),
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
            side_effect=RuntimeError("push api down"),
        ),
    ):
        result = run_teams_alert_cycle()

    # Der Zyklus faengt intern ab und meldet den Fehler - er reisst nicht durch.
    assert result["ok"] is False
    health = channel_health(config, now_ts=_SLOT_TS)
    assert health["cycleCount"] == 1
    assert health["lastCycleOk"] is False
