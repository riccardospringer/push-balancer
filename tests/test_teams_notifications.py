import datetime as dt
import logging
import urllib.error
from unittest.mock import patch

from app.notifications.teams import (
    TeamsAlertConfig,
    buildTeamsPushRecommendation,
    build_teams_alert_context,
    evaluate_and_send_best_candidate,
    evaluate_teams_alert_candidates,
    sendTeamsNotification,
    shouldNotifyTeams,
)
from app.routers.feed import _extract_sitemap_articles


NOW_TS = 1_800_000_000


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat()


def _config(**overrides):
    values = {
        "enabled": True,
        "webhook_url": "https://teams.example.test/webhook",
        "min_score": 70.0,
        "score_only_mode": False,
        "min_or": 5.0,
        "min_minutes_since_last_push": 30,
        "realert_score_delta": 8.0,
        "realert_or_delta": 0.75,
        "alert_cooldown_minutes": 60,
        "global_cooldown_minutes": 30,
        "allowed_sections": (),
        "breaking_override": True,
        "breaking_min_score": 62.0,
        "breaking_min_or": 4.0,
        "breaking_min_minutes_since_last_push": 10,
        "max_article_age_hours": 24,
        "max_pushes_last_6h": 8,
    }
    values.update(overrides)
    return TeamsAlertConfig(**values)


def _candidate(**overrides):
    candidate = {
        "id": "article-1",
        "url": "https://www.bild.de/politik/article-1",
        "title": "Eilmeldung: Regierung beschliesst wichtiges Paket",
        "category": "politik",
        "pubDate": _iso(NOW_TS - 10 * 60),
        "score": 78.4,
        "predictedOR": 0.052,
        "recommendedText": "Eilmeldung: Das bedeutet das neue Paket",
        "isBreaking": False,
        "isEilmeldung": False,
    }
    candidate.update(overrides)
    return candidate


def _history(minutes_since_last_push=42, **overrides):
    item = {
        "message_id": "push-previous",
        "ts_num": NOW_TS - minutes_since_last_push * 60,
        "or": 5.4,
        "title": "Vorheriger Push mit anderem Thema",
        "headline": "Vorheriger Push mit anderem Thema",
        "cat": "news",
        "link": "https://www.bild.de/news/previous",
    }
    item.update(overrides)
    return [item]


def _context(candidate, *, history=None, alert_state=None):
    return build_teams_alert_context(
        [candidate],
        history=history if history is not None else _history(),
        alert_state=alert_state or {},
        last_teams_alert_ts=0,
        now_ts=NOW_TS,
    )


def test_high_score_good_forecast_and_pause_triggers_teams_decision():
    candidate = _candidate()

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is True
    assert decision["status"] == "notify"
    assert any("Push Score" in reason for reason in decision["reasons"])


def test_low_score_does_not_trigger_teams_decision():
    candidate = _candidate(score=63.0)

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is False
    assert any("Score zu niedrig" in reason for reason in decision["blockingReasons"])


def test_too_short_pause_observes_candidate_without_notification():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=_history(minutes_since_last_push=12)),
        _config(),
    )

    assert decision["shouldNotify"] is False
    assert decision["status"] == "observe"
    assert any("Pause seit letztem Push zu kurz" in reason for reason in decision["blockingReasons"])


def test_missing_last_push_timestamp_observes_candidate_without_notification():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(),
    )

    assert decision["shouldNotify"] is False
    assert decision["status"] == "observe"
    assert any("Letzter Push-Zeitpunkt" in reason for reason in decision["blockingReasons"])


def test_bad_forecast_does_not_trigger_teams_decision():
    candidate = _candidate(predictedOR=0.039)

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is False
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])


def test_score_only_mode_triggers_for_score_above_threshold_without_forecast_or_push_time():
    candidate = _candidate(predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True),
    )

    assert decision["shouldNotify"] is True
    assert decision["scoreOnlyMode"] is True
    assert any("Score-Modus aktiv" in reason for reason in decision["reasons"])


def test_score_only_mode_keeps_score_threshold_as_blocker():
    candidate = _candidate(score=69.9, predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True),
    )

    assert decision["shouldNotify"] is False
    assert any("Score zu niedrig" in reason for reason in decision["blockingReasons"])


def test_score_only_mode_does_not_use_lower_breaking_threshold():
    candidate = _candidate(score=79.0, predictedOR=None, isBreaking=True)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True, min_score=80.0, breaking_min_score=62.0),
    )

    assert decision["shouldNotify"] is False
    assert decision["minScore"] == 80.0
    assert any("Score zu niedrig: 79.0 < 80.0" in reason for reason in decision["blockingReasons"])


def test_global_teams_cooldown_blocks_candidate_chain_spam():
    candidate = _candidate(score=92.0)
    context = _context(candidate)
    context["lastTeamsAlertTs"] = NOW_TS - 8 * 60

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(global_cooldown_minutes=30),
    )

    assert decision["shouldNotify"] is False
    assert decision["status"] == "observe"
    assert any("Teams-Cooldown aktiv" in reason for reason in decision["blockingReasons"])


def test_missing_article_link_does_not_trigger_action_recommendation():
    candidate = _candidate(url="")

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is False
    assert any("Artikel-Link" in reason for reason in decision["blockingReasons"])


def test_already_pushed_article_does_not_trigger_teams_decision():
    candidate = _candidate()
    pushed_history = _history(link=candidate["url"], title="Anderer Titel")

    decision = shouldNotifyTeams(candidate, _context(candidate, history=pushed_history), _config())

    assert decision["shouldNotify"] is False
    assert any("Bereits gepushter Artikel" in reason for reason in decision["blockingReasons"])


def test_already_sent_teams_alert_does_not_repeat_without_relevant_change():
    candidate = _candidate()
    alert_state = {
        candidate["id"]: {
            "status": "sent",
            "last_alert_ts": NOW_TS - 90 * 60,
            "last_score": 78.0,
            "last_predicted_or": 5.1,
            "last_candidate_updated_at": NOW_TS - 10 * 60,
            "last_is_breaking": 0,
            "alert_count": 1,
        }
    }

    decision = shouldNotifyTeams(candidate, _context(candidate, alert_state=alert_state), _config())

    assert decision["shouldNotify"] is False
    assert decision["status"] == "sent"
    assert any("Bereits per Teams gemeldet" in reason for reason in decision["blockingReasons"])


def test_realert_requires_relevant_improvement_and_cooldown():
    candidate = _candidate(score=82.0)
    alert_state = {
        candidate["id"]: {
            "status": "sent",
            "last_alert_ts": NOW_TS - 90 * 60,
            "last_score": 78.0,
            "last_predicted_or": 5.1,
            "last_candidate_updated_at": NOW_TS - 10 * 60,
            "last_is_breaking": 0,
            "alert_count": 1,
        }
    }

    no_realert = shouldNotifyTeams(candidate, _context(candidate, alert_state=alert_state), _config())
    improved = shouldNotifyTeams(
        _candidate(score=87.0),
        _context(_candidate(score=87.0), alert_state=alert_state),
        _config(),
    )

    assert no_realert["shouldNotify"] is False
    assert any("Bereits per Teams gemeldet" in reason for reason in no_realert["blockingReasons"])
    assert improved["shouldNotify"] is True
    assert any("Re-Alert wegen relevanter Veraenderung" in reason for reason in improved["reasons"])


def test_multiple_good_candidates_only_notify_best_candidate():
    first = _candidate(id="article-1", url="https://www.bild.de/politik/article-1", score=81.0)
    second = _candidate(
        id="article-2",
        url="https://www.bild.de/sport/article-2",
        title="FC Bayern gewinnt wichtiges Finale",
        category="sport",
        score=74.0,
        predictedOR=0.061,
    )
    context = build_teams_alert_context([first, second], history=_history(), alert_state={}, now_ts=NOW_TS)

    result = evaluate_teams_alert_candidates([first, second], context, _config())
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == "article-1"
    assert decisions["article-1"]["shouldNotify"] is True
    assert decisions["article-2"]["shouldNotify"] is False
    assert any("Staerkerer Kandidat vorhanden" in reason for reason in decisions["article-2"]["blockingReasons"])


def test_teams_message_contains_required_editorial_fields():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert "Push empfohlen" in text
    assert "Was soll ich pushen?" in text
    assert "Welcher Artikel ist gemeint?" in text
    assert "Warum genau jetzt?" in text
    assert candidate["title"] in text
    assert candidate["url"] in text
    assert "Score: 78.4" in text
    assert "Prognose: 5.20 % OR" in text
    assert "Letzter Push: 42 Minuten" in text
    payload = message["payload"]
    assert payload["recommendedAction"] == "Jetzt pushen"
    assert payload["articleTitle"] == candidate["title"]
    assert payload["articleUrl"] == candidate["url"]
    assert payload["recommendedPushText"] == candidate["recommendedText"]
    assert payload["messageText"] == text
    assert "Warum genau jetzt?" in payload["messageHtml"]
    assert "Was soll ich pushen?" in payload["messageHtml"]
    assert isinstance(payload["whyNow"], list)
    assert isinstance(payload["whyPushworthy"], list)


def test_teams_message_hides_global_average_prediction_fallback():
    candidate = _candidate(
        predictedOR=0.0477,
        predictedORBasis="global_avg",
        predictedORConfidence=0.1,
        predictedORIsFallback=True,
    )
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config(score_only_mode=True))

    message = buildTeamsPushRecommendation(candidate, context, decision, _config(score_only_mode=True))
    text = message["text"]

    assert "4.77" not in text
    assert "keine belastbare Prognose" in text
    assert "4.77" not in message["payload"]["messageHtml"]
    assert message["payload"]["predictedOR"] == 0.0
    assert message["payload"]["predictedORAvailable"] is False
    assert message["payload"]["minutesSinceLastPush"] == 42.0


def test_teams_webhook_error_is_logged_and_does_not_crash(caplog):
    caplog.set_level(logging.WARNING, logger="push-balancer")

    with patch(
        "app.notifications.teams.urllib.request.urlopen",
        side_effect=urllib.error.URLError("webhook down"),
    ):
        result = sendTeamsNotification({"payload": {"text": "test"}}, _config())

    assert result["ok"] is False
    assert "webhook down" in result["error"]
    assert "Teams webhook send failed" in caplog.text


def test_send_failure_is_recorded_without_crashing_cycle(tmp_db):
    candidate = _candidate()
    from app.database import push_db_upsert

    push_db_upsert(_history())

    with patch(
        "app.notifications.teams.urllib.request.urlopen",
        side_effect=urllib.error.URLError("webhook down"),
    ):
        result = evaluate_and_send_best_candidate([candidate], config=_config(), now_ts=NOW_TS)

    assert result["ok"] is True
    assert result["sent"] is False
    assert result["sendResult"]["ok"] is False


def test_eil_substring_inside_word_is_not_eilmeldung():
    sitemap = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>https://www.bild.de/news/test</loc>
    <news:news>
      <news:title>Sie teilten Bilder in privaten Chats</news:title>
      <news:publication_date>2026-06-16T14:31:00+02:00</news:publication_date>
    </news:news>
  </url>
</urlset>"""

    article = _extract_sitemap_articles(sitemap, max_items=1)[0]

    assert article["isBreaking"] is False
    assert article["isEilmeldung"] is False
