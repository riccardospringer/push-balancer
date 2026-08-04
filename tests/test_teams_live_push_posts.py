"""Nachrichtentyp 2: 🔴 LIVE-PUSH GESENDET - Erkennung, Format und Claim."""

import datetime as dt
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.database import (
    teams_live_push_post_record,
    teams_live_push_post_try_claim,
)
from app.notifications.teams import (
    TeamsAlertConfig,
    announce_new_live_pushes,
    build_teams_alert_context,
    build_teams_live_push_message,
)


def _ts(hour: int, minute: int = 0, day: int = 24) -> int:
    return int(
        dt.datetime(2026, 6, day, hour, minute, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )


def _config(**overrides):
    values = {
        "enabled": True,
        "webhook_url": "https://teams.example.test/webhook",
        "live_push_posts_enabled": True,
        "live_push_post_lookback_minutes": 90,
        "min_alerts_per_day": 15,
        "max_alerts_per_day": 17,
        "min_minutes_since_last_push": 30,
        "sport_min_per_day": 5,
        "sport_max_per_day": 6,
    }
    values.update(overrides)
    return TeamsAlertConfig(**values)


def _live_push(message_id: str, ts: int, *, cat: str = "news", title: str = "", link: str = ""):
    return {
        "message_id": message_id,
        "ts_num": ts,
        "or": 5.2,
        "title": title or f"Live-Push {message_id}",
        "headline": title or f"Live-Push {message_id}",
        "cat": cat,
        "link": link or f"https://www.bild.de/{cat}/{message_id}",
        "is_eilmeldung": False,
    }


def test_live_push_message_contains_all_spec_fields(tmp_db):
    now = _ts(14, 30)
    history = [
        _live_push("lp-1", _ts(9, 0)),
        _live_push("lp-2", _ts(11, 15), cat="sport"),
        _live_push("lp-3", _ts(14, 20), cat="politik", title="Kanzler kuendigt Reform an"),
    ]
    context = build_teams_alert_context(
        [], history=history, alert_state={}, last_teams_alert_ts=0,
        teams_alerts_today=0, recent_alerts=[], now_ts=now,
    )

    message = build_teams_live_push_message(
        history[-1], context=context, config=_config(), now_ts=now
    )
    text = message["text"]
    payload = message["payload"]

    assert text.startswith("🔴 LIVE-PUSH GESENDET")
    assert "Versendet um:\n14:20 Uhr" in text
    assert "Thema:\nKanzler kuendigt Reform an" in text
    assert "Ressort:\nPolitik" in text
    assert "Quelle:\nRedaktion" in text
    assert "Push-Balancer-Score:\nnicht bewertet" in text
    assert "Tagesstand:\n3 von mindestens 15 und maximal 17 Pushes gesendet" in text
    assert "1 von aktuell 3 Pushes sind Sport" in text
    assert "Auswirkung auf den Plan:" in text
    assert "Noch mindestens 12 und maximal 14 Pushes möglich." in text
    assert "frühestens um 14:50 Uhr" in text
    assert "Nächste Empfehlung:" in text
    assert payload["type"] == "live_push_sent"
    assert payload["pushScoreAvailable"] is False
    assert payload["pushesSentToday"] == 3
    assert payload["remainingMinPushes"] == 12
    assert payload["remainingMaxPushes"] == 14


def test_live_push_message_flags_hot_hour_and_sport_source(tmp_db):
    # Montag 21:05 - die 21:00-Zelle ist tiefrot (7,53 % OR).
    now = int(dt.datetime(2026, 7, 13, 21, 5, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    push = _live_push("lp-sport", now - 3 * 60, cat="sport", title="BVB gewinnt Topspiel")
    context = build_teams_alert_context(
        [], history=[push], alert_state={}, last_teams_alert_ts=0,
        teams_alerts_today=0, recent_alerts=[], now_ts=now,
    )

    message = build_teams_live_push_message(push, context=context, config=_config(), now_ts=now)

    assert "Quelle:\nSportredaktion" in message["text"]
    assert "Hot-Hour-Slot 21:00 Uhr ist belegt" in message["text"]
    assert message["payload"]["isSport"] is True


def test_live_push_message_drops_same_topic_recommendation(tmp_db):
    now = _ts(15, 0)
    push = _live_push(
        "lp-topic", now - 5 * 60, title="Bahnstreik legt Fernverkehr bundesweit lahm"
    )
    context = build_teams_alert_context(
        [], history=[push], alert_state={}, last_teams_alert_ts=0, teams_alerts_today=1,
        recent_alerts=[
            {"key": "reco-1", "title": "Bahnstreik: Fernverkehr bundesweit lahmgelegt"}
        ],
        now_ts=now,
    )

    message = build_teams_live_push_message(push, context=context, config=_config(), now_ts=now)

    assert "Die bestehende Empfehlung zum gleichen Thema entfällt" in message["text"]
    assert message["payload"]["sameTopicRecommendationDropped"] is True


def test_announce_posts_each_fresh_live_push_exactly_once(tmp_db):
    now = _ts(15, 0)
    fresh = _live_push("lp-fresh", now - 10 * 60)
    stale = _live_push("lp-stale", now - 4 * 3600)
    history = [fresh, stale]

    with patch(
        "app.notifications.teams.send_teams_notification",
        return_value={"ok": True, "status": 200},
    ) as send:
        first = announce_new_live_pushes(_config(), now_ts=now, history=history)
        second = announce_new_live_pushes(_config(), now_ts=now, history=history)

    assert first["posted"] == 1
    assert send.call_count == 1
    payload = send.call_args[0][0]["payload"]
    assert payload["type"] == "live_push_sent"
    assert payload["messageId"] == "lp-fresh"
    # Der alte Push zaehlt zum Tagesvolumen, wird aber nicht nachgepostet.
    statuses = {item["messageId"]: item["status"] for item in first["results"]}
    assert statuses["lp-stale"] == "skipped_stale"
    assert second["posted"] == 0


def test_announce_skips_quiet_hours(tmp_db):
    now = int(dt.datetime(2026, 6, 24, 23, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    push = _live_push("lp-night", now - 10 * 60)

    with patch("app.notifications.teams.send_teams_notification") as send:
        result = announce_new_live_pushes(_config(), now_ts=now, history=[push])

    assert result["posted"] == 0
    assert result["results"][0]["status"] == "skipped_quiet_hours"
    send.assert_not_called()


def test_live_push_post_claim_is_terminal_after_sent(tmp_db):
    first = teams_live_push_post_try_claim("mid-1", push_ts=1000, now_ts=2000)
    assert first["claimed"] is True

    teams_live_push_post_record("mid-1", push_ts=1000, status="sent", now_ts=2001)
    second = teams_live_push_post_try_claim("mid-1", push_ts=1000, now_ts=9999)
    assert second["claimed"] is False
    assert second["reason"] == "already_sent"


def test_live_push_post_claim_allows_retry_after_failure_cooldown(tmp_db):
    teams_live_push_post_try_claim("mid-2", push_ts=1000, now_ts=2000)
    teams_live_push_post_record("mid-2", push_ts=1000, status="failed", now_ts=2001)

    blocked = teams_live_push_post_try_claim("mid-2", push_ts=1000, now_ts=2001 + 60)
    assert blocked["claimed"] is False
    assert blocked["reason"] == "retry_cooldown"

    retry = teams_live_push_post_try_claim("mid-2", push_ts=1000, now_ts=2001 + 31 * 60)
    assert retry["claimed"] is True
