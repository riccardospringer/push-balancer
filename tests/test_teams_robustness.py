"""Robustheit des Teams-Kanals: Zustellung, Watchdog, Sichtbarkeit."""

import time
import urllib.error
from unittest.mock import patch

from app.database import teams_alert_record, teams_alert_try_claim_send
from app.notifications.teams import (
    TeamsAlertConfig,
    _CHANNEL_HEALTH,
    _CHANNEL_HEALTH_LOCK,
    channel_health,
    record_worker_cycle,
    record_worker_start,
    send_teams_notification,
)

import pytest


@pytest.fixture(autouse=True)
def _reset_channel_health():
    with _CHANNEL_HEALTH_LOCK:
        snapshot = dict(_CHANNEL_HEALTH)
        for key in _CHANNEL_HEALTH:
            _CHANNEL_HEALTH[key] = 0 if isinstance(_CHANNEL_HEALTH[key], int) else None
        _CHANNEL_HEALTH["lastCycleError"] = ""
    yield
    with _CHANNEL_HEALTH_LOCK:
        _CHANNEL_HEALTH.update(snapshot)


def _config(**overrides):
    values = {
        "enabled": True,
        "webhook_url": "https://teams.example.test/webhook",
        "webhook_max_attempts": 3,
        "webhook_retry_backoff_seconds": 0.0,
        "transport_failure_cooldown_minutes": 20,
        "worker_stall_seconds": 600,
        "quiet_hours_start": "22:00",
        "quiet_hours_end": "06:00",
    }
    values.update(overrides)
    return TeamsAlertConfig(**values)


def _message():
    return {"payload": {"type": "live_push_sent", "text": "x"}, "text": "x"}


# ── Zustellung ────────────────────────────────────────────────────────────


def test_transient_network_error_is_retried_and_can_succeed():
    """Ein Netzwerk-Blip darf keine Nachricht kosten."""
    attempts = {"n": 0}

    class _Response:
        status = 200

        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _flaky(_req, timeout=None, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return _Response()

    with (
        patch("app.notifications.teams._quiet_hours_reason", return_value=""),
        patch("app.notifications.teams.urllib.request.urlopen", side_effect=_flaky),
    ):
        result = send_teams_notification(_message(), _config())

    assert result["ok"] is True
    assert result["attempts"] == 3
    assert attempts["n"] == 3


def test_permanent_client_error_is_not_retried():
    """Ein echter 400er wird nicht sinnlos wiederholt."""
    attempts = {"n": 0}

    def _bad_request(_req, timeout=None, **_kwargs):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            "https://teams.example.test/webhook", 400, "Bad Request", {}, None
        )

    with (
        patch("app.notifications.teams._quiet_hours_reason", return_value=""),
        patch("app.notifications.teams.urllib.request.urlopen", side_effect=_bad_request),
    ):
        result = send_teams_notification(_message(), _config())

    assert result["ok"] is False
    assert result["transient"] is False
    assert attempts["n"] == 1


def test_server_error_is_treated_as_transient():
    attempts = {"n": 0}

    def _server_error(_req, timeout=None, **_kwargs):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            "https://teams.example.test/webhook", 503, "Unavailable", {}, None
        )

    with (
        patch("app.notifications.teams._quiet_hours_reason", return_value=""),
        patch("app.notifications.teams.urllib.request.urlopen", side_effect=_server_error),
    ):
        result = send_teams_notification(_message(), _config())

    assert result["ok"] is False
    assert result["transient"] is True
    assert attempts["n"] == 3


# ── Transportfehler verbrennt die Story nicht ─────────────────────────────


def test_transport_failure_blocks_only_briefly_not_all_day(tmp_db):
    """Regression: ein Netzwerk-Blip sperrte die Story zuvor 12 Stunden."""
    now = int(time.time())
    common = dict(
        article_key="https://www.bild.de/news/transport-blip",
        article_id="transport-blip",
        article_url="https://www.bild.de/news/transport-blip",
        title_hash="hash",
        article_title="Wichtige Meldung",
        score=90.0,
        predicted_or=6.0,
        candidate_updated_at=now,
        is_breaking=False,
    )
    teams_alert_record(
        **common,
        reason="Push empfohlen",
        status="transport_failed",
        error="timeout",
        decision_ts=now - 25 * 60,
    )

    # Nach 25 Minuten (> 20 Min Transportsperre) ist der erneute Versuch frei.
    claim = teams_alert_try_claim_send(
        **common,
        reason="Push empfohlen",
        decision_ts=now,
        global_cooldown_minutes=0,
        transport_failure_cooldown_minutes=20,
    )
    assert claim["claimed"] is True


def test_transport_failure_is_respected_inside_the_short_cooldown(tmp_db):
    now = int(time.time())
    common = dict(
        article_key="https://www.bild.de/news/transport-blip-2",
        article_id="transport-blip-2",
        article_url="https://www.bild.de/news/transport-blip-2",
        title_hash="hash",
        article_title="Wichtige Meldung",
        score=90.0,
        predicted_or=6.0,
        candidate_updated_at=now,
        is_breaking=False,
    )
    teams_alert_record(
        **common,
        reason="Push empfohlen",
        status="transport_failed",
        error="timeout",
        decision_ts=now - 5 * 60,
    )

    claim = teams_alert_try_claim_send(
        **common,
        reason="Push empfohlen",
        decision_ts=now,
        global_cooldown_minutes=0,
        transport_failure_cooldown_minutes=20,
    )
    assert claim["claimed"] is False
    assert claim["reason"] == "article_transport_cooldown"


def test_hard_rejection_keeps_the_long_suppression(tmp_db):
    """Echte Ablehnungen bleiben lange gesperrt - nur Transport ist kurz."""
    now = int(time.time())
    common = dict(
        article_key="https://www.bild.de/news/hard-fail",
        article_id="hard-fail",
        article_url="https://www.bild.de/news/hard-fail",
        title_hash="hash",
        article_title="Abgelehnte Meldung",
        score=90.0,
        predicted_or=6.0,
        candidate_updated_at=now,
        is_breaking=False,
    )
    teams_alert_record(
        **common,
        reason="Push empfohlen",
        status="failed",
        error="Dispatch approval missing",
        decision_ts=now - 60 * 60,
    )

    claim = teams_alert_try_claim_send(
        **common,
        reason="Push empfohlen",
        decision_ts=now,
        global_cooldown_minutes=0,
        failed_cooldown_minutes=720,
    )
    assert claim["claimed"] is False
    assert claim["reason"] == "article_failure_cooldown"


# ── Watchdog / Sichtbarkeit ───────────────────────────────────────────────


def test_channel_health_starts_neutral_and_turns_ok_after_a_cycle():
    config = _config()
    assert channel_health(config)["status"] == "starting"

    record_worker_start()
    record_worker_cycle(ok=True, sent=True)
    health = channel_health(config)

    assert health["status"] == "ok"
    assert health["healthy"] is True
    assert health["cycleCount"] == 1
    assert health["lastSendTs"] > 0


def test_stalled_worker_is_detected_as_unhealthy():
    """Der wichtigste Fall: der Kanal steht, ohne dass es jemand merkt."""
    config = _config(worker_stall_seconds=300)
    now = time.time()
    record_worker_cycle(ok=True, now_ts=now - 20 * 60)

    health = channel_health(config, now_ts=now)

    assert health["status"] == "stalled"
    assert health["healthy"] is False
    assert "steht" in health["reason"]


def test_repeated_transport_failures_degrade_the_channel():
    config = _config()
    record_worker_cycle(ok=True)
    with _CHANNEL_HEALTH_LOCK:
        _CHANNEL_HEALTH["consecutiveTransportFailures"] = 3

    health = channel_health(config)

    assert health["status"] == "degraded"
    assert health["healthy"] is False
    assert "Zustellungen" in health["reason"]


def test_repeated_cycle_errors_degrade_the_channel():
    config = _config()
    for _ in range(3):
        record_worker_cycle(ok=False, error="ScoreApiUnavailable")

    health = channel_health(config)

    assert health["status"] == "degraded"
    assert health["healthy"] is False
    assert "ScoreApiUnavailable" in health["reason"]


def test_disabled_channel_is_not_reported_as_broken():
    health = channel_health(_config(enabled=False))

    assert health["status"] == "disabled"
    assert health["healthy"] is True


def test_worker_restart_is_counted_for_operations():
    record_worker_start()
    record_worker_start(restart=True)
    record_worker_cycle(ok=True)

    assert channel_health(_config())["workerRestarts"] == 1


# ── Datenquellen-Ausfall wird ueberbrueckt, ohne den Vertrag aufzuweichen ──


def _score_client(transport, **kwargs):
    from app.score_api_client import ScoreApiClient

    return ScoreApiClient(
        "https://score.example.test",
        "key",
        transport=transport,
        max_retries=0,
        cache_ttl_seconds=0.0,
        **kwargs,
    )


def _score_body(cms_id: str, score: float, scored_at: str) -> bytes:
    import json as _json

    return _json.dumps({"cmsId": cms_id, "score": score, "scoredAt": scored_at}).encode()


def test_score_api_outage_is_bridged_with_the_last_known_score():
    """Ein kurzer API-Ausfall darf den Kanal nicht sofort verstummen lassen."""
    from datetime import datetime, timezone

    cms_id = "0123456789abcdef01234567"
    scored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    calls = {"n": 0}

    def _transport(_url, _headers, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, _score_body(cms_id, 87.5, scored_at)
        raise TimeoutError("score api unreachable")

    client = _score_client(_transport)

    first = client.get_score(cms_id)
    assert first.score == 87.5
    assert first.served_from_outage_buffer is False

    bridged = client.get_score(cms_id)
    assert bridged is not None
    assert bridged.score == 87.5
    # Derselbe kanonische Score - nur als ueberbrueckt markiert.
    assert bridged.served_from_outage_buffer is True
    assert bridged.scored_at == first.scored_at


def test_outage_buffer_expires_and_then_fails_closed():
    """Nach dem Pufferfenster gilt wieder fail-closed - keine Altlasten."""
    from datetime import datetime, timezone

    from app.score_api_client import ScoreApiUnavailable

    cms_id = "0123456789abcdef01234567"
    scored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    calls = {"n": 0}

    def _transport(_url, _headers, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, _score_body(cms_id, 91.0, scored_at)
        raise TimeoutError("score api unreachable")

    client = _score_client(_transport, outage_buffer_seconds=0.0)
    assert client.get_score(cms_id).score == 91.0

    with pytest.raises(ScoreApiUnavailable):
        client.get_score(cms_id)


def test_outage_buffer_never_survives_an_authorization_failure():
    """Ein abgelehnter Key ist ein Konfigurationsfehler - nie ueberbruecken."""
    from datetime import datetime, timezone

    from app.score_api_client import ScoreApiConfigurationError

    cms_id = "0123456789abcdef01234567"
    scored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    calls = {"n": 0}

    def _transport(_url, _headers, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, _score_body(cms_id, 80.0, scored_at)
        return 401, b""

    client = _score_client(_transport)
    assert client.get_score(cms_id).score == 80.0

    with pytest.raises(ScoreApiConfigurationError):
        client.get_score(cms_id)


def test_bridged_score_still_fails_the_workday_freshness_contract_when_too_old():
    """Auch der Acht-Stunden-Quellenvertrag bleibt fuer Altwerte fail-closed."""
    import datetime as _dt

    from app.routers.feed import _apply_internal_score_api_scores
    from app.score_api_client import ArticleScore, ScoreLookup

    cms_id = "0123456789abcdef01234567"
    now = _dt.datetime.now(_dt.timezone.utc)
    stale = ArticleScore(
        cms_id=cms_id,
        score=95.0,
        scored_at=now - _dt.timedelta(seconds=8 * 3600),
        served_from_outage_buffer=True,
    )
    articles = [{"url": f"https://www.bild.de/politik/{cms_id}", "score": 50.0}]

    with patch(
        "app.score_api_client.resolve_cms_id",
        return_value=cms_id,
    ), patch(
        "app.score_api_client.fetch_score_lookups",
        return_value={cms_id: ScoreLookup(status="ok", value=stale)},
    ):
        result = _apply_internal_score_api_scores(articles, client=object(), now=now)

    assert result[0]["scoreApiStatus"] == "stale"
    assert result[0]["scoreSource"] == "internal_score_api_stale"
    assert result[0]["score"] == 0.0


# ── Startup-Selbstcheck: Fehlkonfiguration wird laut, nicht still ──────────


def test_selfcheck_passes_for_a_correctly_configured_channel():
    from app.notifications.teams import channel_configuration_problems

    config = _config(webhook_url="https://teams.example.test/webhook",
                     require_internal_score_api=False)
    assert channel_configuration_problems(config) == []


def test_selfcheck_flags_a_missing_webhook():
    from app.notifications.teams import channel_configuration_problems

    config = _config(webhook_url="", require_internal_score_api=False)
    problems = channel_configuration_problems(config)
    assert any("PUSH_TEAMS_WEBHOOK_URL" in p for p in problems)


def test_selfcheck_flags_missing_score_api_key_when_required(monkeypatch):
    import app.config as app_config
    from app.notifications.teams import channel_configuration_problems

    monkeypatch.setattr(app_config, "PUSH_BALANCER_SCORE_API_BASE_URL",
                        "https://score.example.test", raising=False)
    monkeypatch.setattr(app_config, "PUSH_BALANCER_SCORE_API_KEY", "", raising=False)
    config = _config(
        webhook_url="https://teams.example.test/webhook",
        require_internal_score_api=True,
    )
    problems = channel_configuration_problems(config)
    assert any("SCORE_API_KEY" in p for p in problems)


def test_selfcheck_is_silent_for_a_disabled_channel():
    from app.notifications.teams import channel_configuration_problems

    assert channel_configuration_problems(_config(enabled=False, webhook_url="")) == []


def test_selfcheck_flags_contradictory_daily_target():
    from app.notifications.teams import channel_configuration_problems

    config = _config(
        webhook_url="https://teams.example.test/webhook",
        require_internal_score_api=False,
        min_alerts_per_day=15,
        max_alerts_per_day=11,
    )
    problems = channel_configuration_problems(config)
    assert any("MIN" in p for p in problems)
