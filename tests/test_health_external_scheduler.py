"""Health status must reflect the external Teams scheduler prerequisites."""

from __future__ import annotations

import json


def test_health_degrades_when_power_automate_transport_is_not_ready(monkeypatch):
    import app.notifications.teams as teams
    import app.routers.health as health

    class _EnabledTeamsConfig:
        enabled = True

    monkeypatch.setattr(teams, "TeamsAlertConfig", _EnabledTeamsConfig)
    monkeypatch.setattr(health, "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED", False)
    monkeypatch.setattr(health, "POWER_AUTOMATE_API_KEY", "")
    monkeypatch.setattr(health, "PUSH_DB_DURABLE", True)
    monkeypatch.setattr(health, "durable_db_storage_available", lambda: True)
    monkeypatch.setitem(health._health_state, "status", "ok")

    payload = json.loads(health.get_health().body)

    assert payload["status"] == "degraded"
    assert payload["teamsChannel"]["status"] == "external_scheduler"
    assert payload["teamsChannel"]["healthy"] is False

