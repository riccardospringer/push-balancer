"""Timeliness priorisiert brandaktuelle Top-Stories innerhalb des Rasters.

Motiviert durch echtes Redaktions-Feedback ("Zug der Liebe war gut, kam nur
zu spaet"): Im aktuellen Raster-Slot soll ein starkes, brandaktuelles Ereignis
priorisiert werden. Zwischen den verbindlichen Slots bleibt der Versand jedoch
fail-closed. Alle uebrigen Qualitaets-Gates laufen unveraendert weiter.
"""

import datetime as dt
from zoneinfo import ZoneInfo

from app.notifications.teams import build_teams_alert_context, shouldNotifyTeams

from tests.test_teams_notifications import _candidate, _history, _smart_config

# Samstag 14:20 - kein Raster-Slot faellig (Morgen vorbei, Abendblock spaeter).
_OFF_RASTER_TS = int(
    dt.datetime(2026, 7, 18, 14, 20, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
)
_BINDING_TS = int(
    dt.datetime(2026, 7, 18, 12, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
)


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, ZoneInfo("Europe/Berlin")).isoformat()


def _fresh_top_story(score: float, age_minutes: float, *, now_ts: int = _OFF_RASTER_TS, **overrides):
    """Ein starker, editorial-tauglicher Kandidat (besteht die uebrigen Gates)."""
    return _candidate(
        score=score,
        pubDate=_iso(int(now_ts - age_minutes * 60)),
        **overrides,
    )


def _decide(candidate, *, config=None, now_ts=_OFF_RASTER_TS):
    config = config or _smart_config(hot_fresh_enabled=True, hot_fresh_min_score=85.0)
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=90, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=4,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 4
    return shouldNotifyTeams(candidate, context, config)


def test_fresh_top_story_waits_when_no_binding_slot_is_open():
    """Auch gut und brandaktuell darf den verbindlichen Rasterplan nicht umgehen."""
    decision = _decide(_fresh_top_story(score=90.0, age_minutes=6))

    assert decision["shouldNotify"] is False
    assert decision["slotGate"]["mode"] == "wait"
    assert any("nur im 5-Minuten-Fenster" in reason for reason in decision["blockingReasons"])


def test_fresh_top_story_is_prioritized_inside_a_binding_slot():
    candidate = _fresh_top_story(
        score=90.0,
        age_minutes=6,
        now_ts=_BINDING_TS,
    )
    decision = _decide(candidate, now_ts=_BINDING_TS)

    assert decision["shouldNotify"] is True
    assert decision["slotGate"]["mode"] == "hot_fresh_override"
    assert decision["slotGate"]["hotFreshAgeMinutes"] <= 20


def test_stale_top_story_still_waits_for_the_raster():
    """Dieselbe Story 40 Minuten spaeter ist nicht mehr 'frisch' -> Raster."""
    decision = _decide(_fresh_top_story(score=90.0, age_minutes=40))

    assert decision["slotGate"]["mode"] != "hot_fresh_override"
    assert decision["shouldNotify"] is False


def test_weak_but_fresh_story_does_not_escalate():
    """Der Off-Raster-Schutz (11:30-Fix) bleibt: schwache Kandidaten warten."""
    decision = _decide(_fresh_top_story(score=78.0, age_minutes=5))

    assert decision["slotGate"]["mode"] != "hot_fresh_override"
    assert decision["shouldNotify"] is False


def test_hot_fresh_can_be_disabled():
    config = _smart_config(hot_fresh_enabled=False)
    decision = _decide(_fresh_top_story(score=92.0, age_minutes=5), config=config)

    assert decision["slotGate"]["mode"] != "hot_fresh_override"


def test_hot_fresh_still_respects_quiet_hours():
    night_ts = int(
        dt.datetime(2026, 7, 18, 23, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    decision = _decide(
        _fresh_top_story(score=92.0, age_minutes=5, now_ts=night_ts), now_ts=night_ts
    )

    assert decision["shouldNotify"] is False
    assert any("Ruhezeit" in reason for reason in decision["blockingReasons"])


def test_timing_brief_binds_hot_fresh_to_the_current_slot():
    """Timeliness bleibt auf den gerade geöffneten Raster-Slot gebunden."""
    from app.notifications.teams import buildTeamsPushRecommendation

    candidate = _fresh_top_story(
        score=91.0,
        age_minutes=4,
        now_ts=_BINDING_TS,
    )
    config = _smart_config(hot_fresh_enabled=True, hot_fresh_min_score=85.0)
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=90, now_ts=_BINDING_TS),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=4,
        recent_alerts=[],
        now_ts=_BINDING_TS,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 4
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    basis = message["payload"]["decisionBasis"]
    assert basis.startswith("Timeliness-Priorisierung")
    assert message["payload"]["recommendedSendAt"] == "12:30"
