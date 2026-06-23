import datetime as dt
import logging
import urllib.error
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.notifications.teams import (
    TeamsAlertConfig,
    buildTeamsPushRecommendation,
    build_teams_alert_context,
    evaluate_and_send_best_candidate,
    evaluate_teams_alert_candidates,
    normalize_predicted_or,
    selectTeamsPushRecommendation,
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
        "min_alert_score": 78.0,
        "score_only_mode": False,
        "min_or": 5.0,
        "min_minutes_since_last_push": 30,
        "realert_score_delta": 8.0,
        "realert_or_delta": 0.75,
        "alert_cooldown_minutes": 60,
        "repeat_suppression_hours": 12,
        "global_cooldown_minutes": 30,
        "allowed_sections": (),
        "excluded_sections": ("sport",),
        "breaking_override": True,
        "breaking_min_score": 62.0,
        "breaking_min_or": 4.0,
        "breaking_min_minutes_since_last_push": 10,
        "max_article_age_hours": 24,
        "max_pushes_last_6h": 8,
        # Dynamische Schwelle in den Basistests aus, damit Schwellen deterministisch sind.
        "dynamic_threshold_enabled": False,
        "require_valid_prediction": False,
        "target_pushes_per_day": 11,
        "max_alerts_per_day": 14,
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
        "scoreReason": (
            "stark: hoch wegen aktuelle Entwicklung, BILD-Reiz und klare Zeile. "
            "Risiko: Politik-Dichte heute bereits hoch."
        ),
        "performanceDrivers": [
            "Aktualität: sehr frisch veröffentlicht",
            "BILD-Reiz: große Zielgruppe unmittelbar betroffen",
            "Headline-Stärke: schnell verständlich und zuspitzbar",
        ],
        "risks": [
            "Politik-Dichte: ähnliche Themen heute bereits stark vertreten",
        ],
        "scoreBreakdown": {
            "freshness": 96.0,
            "bildReiz": 84.0,
            "headlineStrength": 78.0,
            "openingRatePotential": 80.0,
            "mixBalance": 72.0,
            "politicsContext": 88.0,
            "videoFit": 68.0,
            "editorialFeedback": 60.0,
            "riskAndFatigue": 75.0,
        },
        "recommendedText": "Eilmeldung: Das bedeutet das neue Paket",
        "isBreaking": False,
        "isEilmeldung": False,
    }
    candidate.update(overrides)
    return candidate


def _history(minutes_since_last_push=42, now_ts=NOW_TS, **overrides):
    item = {
        "message_id": "push-previous",
        "ts_num": now_ts - minutes_since_last_push * 60,
        "or": 5.4,
        "title": "Vorheriger Push mit anderem Thema",
        "headline": "Vorheriger Push mit anderem Thema",
        "cat": "news",
        "link": "https://www.bild.de/news/previous",
    }
    item.update(overrides)
    return [item]


def _context(
    candidate,
    *,
    history=None,
    alert_state=None,
    teams_alerts_today=0,
    recent_alerts=None,
    now_ts=NOW_TS,
):
    return build_teams_alert_context(
        [candidate],
        history=history if history is not None else _history(now_ts=now_ts),
        alert_state=alert_state or {},
        last_teams_alert_ts=0,
        teams_alerts_today=teams_alerts_today,
        recent_alerts=recent_alerts if recent_alerts is not None else [],
        now_ts=now_ts,
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


def test_score_only_mode_blocks_candidate_without_forecast_or_push_time():
    candidate = _candidate(score=82.0, predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True),
    )

    assert decision["shouldNotify"] is False
    assert decision["scoreOnlyMode"] is True
    assert any("Letzter Push-Zeitpunkt" in reason for reason in decision["blockingReasons"])


def test_score_only_mode_keeps_score_threshold_as_blocker():
    candidate = _candidate(score=69.9, predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True),
    )

    assert decision["shouldNotify"] is False
    assert any("Score zu niedrig" in reason for reason in decision["blockingReasons"])


def test_sport_section_is_blocked_even_in_score_only_mode():
    candidate = _candidate(
        score=95.0,
        category="sport",
        title="Bayern-Star vor Wechsel: Entscheidung gefallen",
        url="https://www.bild.de/sport/article-1",
        predictedOR=None,
    )

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(
            score_only_mode=True,
            allowed_sections=("news", "politik", "wirtschaft", "regional", "digital", "unterhaltung"),
        ),
    )

    assert decision["shouldNotify"] is False
    assert any("Ressort sport" in reason for reason in decision["blockingReasons"])


def test_candidate_outside_dashboard_top_limit_is_blocked():
    candidate = _candidate(score=92.0, predictedOR=0.07)
    context = _context(candidate)
    context["dashboardRank"] = 25

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20),
    )

    assert decision["shouldNotify"] is False
    assert any("Nicht im oberen Push-Balancer-Feld" in reason for reason in decision["blockingReasons"])


def test_cvd_gate_blocks_soft_topic_even_with_high_score_and_forecast():
    candidate = _candidate(
        score=95.0,
        predictedOR=0.09,
        category="digital",
        title="Schock fuer Fans: Beliebte Sprachlern-App vor dem Aus",
        url="https://www.bild.de/digital/sprachlern-app",
        isBreaking=False,
        isEilmeldung=False,
    )
    context = _context(candidate)
    context["dashboardRank"] = 3

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(min_alert_score=60.0, min_editorial_score=82.0),
    )

    assert decision["shouldNotify"] is False
    assert decision["editorialReview"]["approved"] is False
    assert any("CvD:" in reason for reason in decision["blockingReasons"])


def test_cvd_gate_blocks_non_breaking_candidate_outside_editorial_top_ten():
    candidate = _candidate(score=96.0, predictedOR=0.08)
    context = _context(candidate)
    context["dashboardRank"] = 11

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20, editorial_top_limit=10),
    )

    assert decision["shouldNotify"] is False
    assert any("Top 10" in reason for reason in decision["blockingReasons"])


def test_cvd_gate_allows_breaking_candidate_beyond_editorial_top_ten():
    candidate = _candidate(
        score=91.0,
        predictedOR=0.065,
        title="Eilmeldung: Iran und Israel einigen sich auf Feuerpause",
        isBreaking=True,
        isEilmeldung=True,
    )
    context = _context(candidate)
    context["dashboardRank"] = 12

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20, editorial_top_limit=10),
    )

    assert decision["shouldNotify"] is True
    assert decision["editorialReview"]["approved"] is True
    assert any("CvD-Freigabe" in reason for reason in decision["reasons"])


def test_score_only_mode_blocks_soft_high_score_when_alert_score_is_too_low():
    candidate = _candidate(
        score=84.0,
        category="unterhaltung",
        title="Sommertrend: Diese Stars feiern neue Rabatt-App",
        predictedOR=None,
        isBreaking=False,
        isEilmeldung=False,
    )

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True, min_score=75.0, min_alert_score=78.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Teams Alert Score zu niedrig" in reason for reason in decision["blockingReasons"])


def test_weighted_model_blocks_breaking_without_timing_and_forecast():
    candidate = _candidate(
        score=78.0,
        predictedOR=None,
        isBreaking=True,
        isEilmeldung=True,
        title="Eilmeldung: Trump und Iran einigen sich auf Feuerpause",
    )

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True, min_score=75.0, breaking_min_score=72.0, min_alert_score=78.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Letzter Push-Zeitpunkt" in reason for reason in decision["blockingReasons"])


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
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_same_story_pushed_under_different_url_is_blocked():
    # Echter Push der gleichen Story unter anderer URL + push-optimiertem Titel.
    candidate = _candidate(
        title="Prozess um Mord an Fabian: Vier Polizisten sagen aus",
        url="https://www.bild.de/regional/rostock/prozess-um-mord-an-fabian-freund-ticker",
        category="news",
    )
    pushed = _history(
        minutes_since_last_push=90,
        link="https://www.bild.de/regional/rostock/prozess-mord-fabian-freund-zeugen-aussage",
        title="🚨 Mordprozess Fabian: Jetzt sagen die Polizisten aus",
    )

    decision = shouldNotifyTeams(candidate, _context(candidate, history=pushed), _config())

    assert decision["shouldNotify"] is False
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_different_story_sharing_one_token_is_not_blocked_as_pushed():
    candidate = _candidate(
        title="Regierung beschließt neues Rentenpaket für Familien",
        url="https://www.bild.de/politik/inland/regierung-rentenpaket-familien",
        category="politik",
    )
    pushed = _history(
        minutes_since_last_push=90,
        link="https://www.bild.de/politik/inland/merz-kritik-opposition-haushalt",
        title="Merz watscht Opposition ab",
    )

    decision = shouldNotifyTeams(candidate, _context(candidate, history=pushed), _config())

    assert not any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_already_sent_teams_alert_does_not_repeat_without_relevant_change():
    candidate = _candidate()
    alert_state = {
        candidate["url"]: {
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


def test_failed_teams_attempt_suppresses_same_candidate_for_repeat_window():
    candidate = _candidate()
    alert_state = {
        candidate["url"]: {
            "status": "failed",
            "last_decision_ts": NOW_TS - 90 * 60,
            "last_score": 78.0,
            "last_predicted_or": 0.0,
            "last_candidate_updated_at": NOW_TS - 10 * 60,
            "last_is_breaking": 0,
            "alert_count": 0,
        }
    }

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, alert_state=alert_state),
        _config(alert_cooldown_minutes=60, repeat_suppression_hours=12),
    )

    assert decision["shouldNotify"] is False
    assert any("Bereits als Teams-Kandidat versucht" in reason for reason in decision["blockingReasons"])


def test_realert_requires_relevant_improvement_and_cooldown():
    candidate = _candidate(score=82.0)
    alert_state = {
        candidate["url"]: {
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


def test_retimestamped_article_does_not_trigger_realert():
    # Gleiche Schlagzeile, nur neuer modDate (BILD-Re-Timestamp) -> KEIN Re-Alert.
    candidate = _candidate(modDate=_iso(NOW_TS))
    alert_state = {
        candidate["url"]: {
            "status": "sent",
            "last_alert_ts": NOW_TS - 180 * 60,
            "last_score": 78.4,
            "last_predicted_or": 5.2,
            "last_candidate_updated_at": NOW_TS - 24 * 3600,
            "last_is_breaking": 0,
            "article_title": candidate["title"],
            "alert_count": 1,
        }
    }

    decision = shouldNotifyTeams(candidate, _context(candidate, alert_state=alert_state), _config())

    assert decision["shouldNotify"] is False
    assert any("Bereits per Teams gemeldet" in reason for reason in decision["blockingReasons"])


def test_substantially_changed_headline_allows_realert():
    candidate = _candidate(
        score=80.0,
        title="Bundestag stoppt ueberraschend das geplante Rentenpaket",
    )
    alert_state = {
        candidate["url"]: {
            "status": "sent",
            "last_alert_ts": NOW_TS - 180 * 60,
            "last_score": 78.0,
            "last_predicted_or": 5.1,
            "last_candidate_updated_at": NOW_TS - 24 * 3600,
            "last_is_breaking": 0,
            "article_title": "Promi zeigt neues Sommer-Outfit im Urlaub",
            "alert_count": 1,
        }
    }

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, alert_state=alert_state),
        _config(min_editorial_score=50.0, min_alert_score=50.0),
    )

    assert decision["shouldNotify"] is True
    assert any("Schlagzeile" in reason for reason in decision["reasons"])


def test_stale_speculative_resignation_is_blocked():
    # "bereitet wohl Ruecktritt vor" + nicht mehr frisch -> wahrscheinlich ueberholt.
    candidate = _candidate(
        score=92.0,
        predictedOR=0.08,
        title="Briten-Premier bereitet wohl Rücktritt vor",
        url="https://www.bild.de/politik/premier-ruecktritt",
        pubDate=_iso(NOW_TS - 6 * 3600),
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=50.0, min_editorial_score=50.0),
    )

    assert decision["shouldNotify"] is False
    assert decision["isSpeculative"] is True
    assert any("ueberholt" in reason for reason in decision["blockingReasons"])


def test_fresh_speculative_item_is_flagged_but_not_blocked():
    candidate = _candidate(
        score=92.0,
        predictedOR=0.08,
        title="Briten-Premier bereitet wohl Rücktritt vor",
        url="https://www.bild.de/politik/premier-ruecktritt",
        pubDate=_iso(NOW_TS - 30 * 60),
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=50.0, min_editorial_score=50.0),
    )

    assert decision["isSpeculative"] is True
    assert not any("ueberholt" in reason for reason in decision["blockingReasons"])
    assert any("pekulativ" in risk for risk in decision["risks"])


def test_soft_quiz_or_service_content_is_blocked():
    for bad_title in (
        "Bundesländer in Deutschland: Erkennen Sie das Gesuchte?",
        "Bester Strand Europas 2026: Lohnt sich Portugal?",
        "Alkohol genießen und Kalorien sparen? Diese Drinks machen es möglich",
    ):
        candidate = _candidate(
            title=bad_title,
            category="news",
            score=85.0,
            predictedOR=0.07,
            url="https://www.bild.de/news/soft",
        )
        context = _context(candidate, history=_history(minutes_since_last_push=120))
        context["dashboardRank"] = 1

        decision = shouldNotifyTeams(
            candidate,
            context,
            _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
        )

        assert decision["shouldNotify"] is False, bad_title
        assert any("Service-/Raetsel" in reason for reason in decision["blockingReasons"]), bad_title


def test_ratgeber_and_gewinnspiel_content_is_blocked():
    for bad_title in (
        "Inflation frisst Zinsen: Kommt man vorzeitig aus einer Festgeldanlage?",
        "Aktuelles Ösi-Urteil: Bietet Ebay-Käuferschutz nur eine Scheinsicherheit?",
        "Geld und Wertsachen verstecken: Wo Einbrecher suchen",
        "LOTTO-Gewinnspiel!: Wer holt sich die 50 Millionen?",
    ):
        candidate = _candidate(
            title=bad_title,
            category="news",
            score=85.0,
            predictedOR=0.07,
            url="https://www.bild.de/news/ratgeber",
        )
        context = _context(candidate, history=_history(minutes_since_last_push=120))
        context["dashboardRank"] = 1

        decision = shouldNotifyTeams(
            candidate,
            context,
            _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
        )

        assert decision["shouldNotify"] is False, bad_title
        assert any("Service-/Raetsel" in reason for reason in decision["blockingReasons"]), bad_title


def test_event_gate_blocks_teaser_without_news_event():
    # Kein Soft-Stichwort, aber auch kein Nachrichten-Ereignis -> strukturell geblockt.
    candidate = _candidate(
        title="Trump hebt neu ab: Präsident zeigt Luxus-Flieger",
        category="news",
        score=85.0,
        predictedOR=0.07,
        url="https://www.bild.de/news/trump-flieger",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=120))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"])


def test_event_gate_allows_real_news_event():
    candidate = _candidate(
        title="Katar: Mindestens 13 Tote nach Explosion in Hafen",
        category="news",
        score=85.0,
        predictedOR=0.07,
        url="https://www.bild.de/news/katar-hafen",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=120))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is True
    assert not any("Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"])


def test_topic_duplicate_against_recent_teams_alert_is_blocked():
    candidate = _candidate(
        title="Große Gasanlage betroffen: 13 Tote bei Explosion in Katar",
        category="news",
        score=88.0,
        predictedOR=0.07,
        url="https://www.bild.de/news/katar-gas",
    )
    recent = [
        {
            "key": "https://www.bild.de/news/katar-hafen",
            "title": "Katar: Mindestens 13 Tote nach Explosion in Hafen",
        }
    ]
    context = _context(
        candidate, history=_history(minutes_since_last_push=120), recent_alerts=recent
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Dublette" in reason for reason in decision["blockingReasons"])


def test_feed_overtaken_blocks_speculative_resignation():
    candidate = _candidate(
        title="Briten-Premier bereitet wohl Rücktritt vor: Tritt Starmer heute wirklich zurück?",
        category="politik",
        score=90.0,
        predictedOR=0.07,
        url="https://www.bild.de/politik/starmer",
        pubDate=_iso(NOW_TS - 20 * 60),  # frisch -> Alters-Guard wuerde NICHT blocken
    )
    context = _context(candidate, history=_history(minutes_since_last_push=120))
    context["dashboardRank"] = 1

    feeds = {"bbc": [{"t": "Keir Starmer resigns as UK prime minister"}]}
    with patch("app.research.worker.get_cached_feeds", return_value=feeds):
        decision = shouldNotifyTeams(
            candidate,
            context,
            _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
        )

    assert decision["shouldNotify"] is False
    assert decision["overtakenByFeed"]
    assert any("vollzogen gemeldet" in reason for reason in decision["blockingReasons"])


def test_non_speculative_headline_is_not_flagged():
    candidate = _candidate(title="Regierung beschliesst neues Rentenpaket")

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["isSpeculative"] is False


def test_candidate_key_normalizes_tracking_query_params():
    first = _candidate(url="https://www.bild.de/politik/article-1?utm_source=x")
    second = _candidate(url="https://www.bild.de/politik/article-1")

    from app.notifications.teams import candidate_key

    assert candidate_key(first) == candidate_key(second)


def test_multiple_good_candidates_only_notify_best_candidate():
    first = _candidate(id="article-1", url="https://www.bild.de/politik/article-1", score=95.0)
    second = _candidate(
        id="article-2",
        url="https://www.bild.de/politik/article-2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket",
        category="politik",
        score=82.0,
        predictedOR=0.061,
    )
    context = build_teams_alert_context([first, second], history=_history(), alert_state={}, now_ts=NOW_TS)

    result = evaluate_teams_alert_candidates([first, second], context, _config())
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == first["url"]
    assert decisions[first["url"]]["shouldNotify"] is True
    assert decisions[second["url"]]["shouldNotify"] is False
    assert any("Staerkerer Kandidat vorhanden" in reason for reason in decisions[second["url"]]["blockingReasons"])


def test_cvd_selection_can_choose_lower_raw_score_when_editorially_stronger():
    high_raw = _candidate(
        id="article-raw",
        url="https://www.bild.de/politik/raw",
        score=96.0,
        predictedOR=0.08,
        title="Regierung beschliesst neues Steuerpaket fuer Familien",
        isBreaking=False,
        isEilmeldung=False,
    )
    stronger_cvd = _candidate(
        id="article-cvd",
        url="https://www.bild.de/politik/cvd",
        score=88.0,
        predictedOR=0.061,
        title="Eilmeldung: Israel und Iran einigen sich auf Feuerpause",
        isBreaking=True,
        isEilmeldung=True,
    )
    candidates = [high_raw, stronger_cvd]
    context = build_teams_alert_context(
        candidates,
        history=_history(minutes_since_last_push=55),
        alert_state={},
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates(
        candidates,
        context,
        _config(min_alert_score=70.0, min_editorial_score=82.0),
    )
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == stronger_cvd["url"]
    assert decisions[stronger_cvd["url"]]["selectionScore"] > decisions[high_raw["url"]]["selectionScore"]
    assert decisions[stronger_cvd["url"]]["shouldNotify"] is True
    assert decisions[high_raw["url"]]["shouldNotify"] is False


def test_auto_push_calibration_allows_public_warning_candidate():
    candidate = _candidate(
        score=80.0,
        predictedOR=None,
        category="news",
        title="Wetterdienst gibt Hitzewarnung fuer Deutschland raus",
        url="https://www.bild.de/news/wetter/hitzewarnung",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45))
    context["dashboardRank"] = 10

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            score_only_mode=True,
            min_alert_score=66.0,
            min_editorial_score=70.0,
            no_forecast_min_alert_score=76.0,
        ),
    )

    assert decision["shouldNotify"] is True
    assert decision["teamsAlertScore"] >= 66.0
    assert decision["editorialScore"] >= 70.0


def test_auto_push_calibration_still_blocks_soft_topic():
    candidate = _candidate(
        score=76.0,
        predictedOR=None,
        category="politik",
        title="Peinliche Momente beim G7-Gipfel: Die grosse Buehne der witzigen Weltpolitik",
        url="https://www.bild.de/politik/g7",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45))
    context["dashboardRank"] = 3

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            score_only_mode=True,
            min_alert_score=66.0,
            min_editorial_score=70.0,
            no_forecast_min_alert_score=76.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert any("CvD:" in reason for reason in decision["blockingReasons"])


def test_cvd_time_fit_blocks_normal_push_at_night():
    night_ts = NOW_TS - 7 * 3600
    candidate = _candidate(
        score=95.0,
        predictedOR=0.08,
        category="news",
        title="Wetterdienst gibt Hitzewarnung fuer Deutschland raus",
        url="https://www.bild.de/news/wetter/hitzewarnung-nacht",
        isBreaking=False,
        isEilmeldung=False,
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45, now_ts=night_ts), now_ts=night_ts)
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(min_alert_score=60.0, min_editorial_score=50.0, min_time_fit_score=4.0),
    )

    assert decision["shouldNotify"] is False
    assert any("unguens" in reason for reason in decision["blockingReasons"])
    assert decision["editorialReview"]["breakdown"]["localHour"] == 2


def test_quiet_hours_block_breaking_push_at_night():
    night_ts = NOW_TS - 7 * 3600
    candidate = _candidate(
        score=95.0,
        predictedOR=0.08,
        category="news",
        title="Eilmeldung: Israel und Iran einigen sich auf Feuerpause",
        url="https://www.bild.de/news/breaking-nacht",
        isBreaking=True,
        isEilmeldung=True,
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45, now_ts=night_ts), now_ts=night_ts)
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(min_alert_score=60.0, min_editorial_score=50.0, min_time_fit_score=4.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Ruhezeit" in reason for reason in decision["blockingReasons"])
    assert decision["editorialReview"]["breakdown"]["timeFit"] >= 4.0


def test_teams_message_contains_required_editorial_fields():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert text.startswith("🚨 Jetzt pushen: Eilmeldung: Das bedeutet das neue Paket")
    assert "Empfohlener Push-Text:" not in text
    assert "Alternativer Push-Titel:\nEilmeldung: Das bedeutet das neue Paket" in text
    assert "Artikel:" in text
    assert "Warum jetzt?" in text
    assert candidate["title"] in text
    assert candidate["url"] in text
    assert "Politik | Score 78,4 | Prognose 5,20 % OR" in text
    assert "Teams-Alert-Score: " not in text
    assert "Push-Balancer-Breakdown:" not in text
    assert "Die Artikel-Prognose liegt aktuell bei 5,20 % OR." in text
    assert "letzter Push vor 42 Minuten" in text
    payload = message["payload"]
    assert payload["recommendedAction"] == "Jetzt pushen"
    assert payload["articleTitle"] == candidate["title"]
    assert payload["articleUrl"] == candidate["url"]
    assert payload["teamsAlertScore"] >= payload["teamsAlertScoreThreshold"]
    assert payload["editorialReview"]["approved"] is True
    assert payload["scoreReason"] == candidate["scoreReason"]
    assert payload["performanceDrivers"] == candidate["performanceDrivers"]
    assert payload["risks"] == candidate["risks"]
    assert payload["scoreBreakdown"]["bildReiz"] == 84.0
    assert "BILD-Reiz: 84.0/100" in payload["scoreBreakdownLabel"]
    assert payload["editorialScore"] >= 82.0
    assert payload["selectionScore"] > 0
    assert payload["timeFitScore"] > 0
    assert payload["timeFitLabel"]
    assert payload["recommendedPushText"] == candidate["recommendedText"]
    assert payload["alternativePushTitle"] == candidate["recommendedText"]
    assert payload["messageText"] == text
    assert "Warum jetzt?" in payload["messageHtml"]
    assert payload["subject"].startswith("🚨 Jetzt pushen:")
    assert "Push-Balancer-Breakdown" not in payload["messageHtml"]
    assert "Alternativer Push-Titel:" in payload["messageHtml"]
    assert isinstance(payload["whyNow"], list)
    assert isinstance(payload["whyPushworthy"], list)


def test_teams_message_uses_llm_generated_title_when_available():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    llm_result = {
        "title": "Eil-Beschluss: So viel mehr Geld gibt es jetzt für Familien",
        "meta": {"llm_call_started": True},
    }
    with patch("push_title_agent._llm_unavailable_reason", return_value=""), patch(
        "push_title_agent.generate_push_title", return_value=llm_result
    ):
        message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert message["payload"]["pushTitleSource"] == "llm"
    assert message["payload"]["alternativePushTitle"] == llm_result["title"]
    assert llm_result["title"] in message["text"]


def test_teams_message_discards_generic_llm_title():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    llm_result = {
        "title": "Regierung beschliesst Paket: Darum geht es jetzt",
        "meta": {"llm_call_started": True},
    }
    with patch("push_title_agent._llm_unavailable_reason", return_value=""), patch(
        "push_title_agent.generate_push_title", return_value=llm_result
    ):
        message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert message["payload"]["pushTitleSource"] != "llm"
    assert "Darum geht es jetzt" not in message["payload"]["alternativePushTitle"]


def test_teams_message_does_not_repeat_identical_push_text_and_article_title():
    candidate = _candidate(recommendedText=_candidate()["title"])
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert text.startswith("🚨 Jetzt pushen:")
    assert "Alternativer Push-Titel:" in text
    assert "Empfohlener Push-Text:" not in text
    assert "Artikel:\n" in text
    assert text.count(candidate["title"]) == 1
    assert "Artikel:</strong>" in message["payload"]["messageHtml"]
    assert message["payload"]["alternativePushTitle"] != candidate["title"]


def test_or_prediction_ratio_is_displayed_as_percent_not_raw_ratio():
    candidate = _candidate(predictedOR=0.0477)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert normalize_predicted_or(0.0477) == 4.77
    assert "Prognose 4,77 % OR" in message["text"]
    assert message["payload"]["predictedOR"] == 4.77
    assert message["payload"]["predictedORLabel"] == "4,77 % OR"
    assert message["payload"]["predictedORSource"] == "article_model"


def test_tiny_double_scaled_or_prediction_is_not_displayed_as_forecast():
    candidate = _candidate(predictedOR=0.0004)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config(score_only_mode=True))

    message = buildTeamsPushRecommendation(candidate, context, decision, _config(score_only_mode=True))

    assert normalize_predicted_or(0.0004) is None
    assert "0,04 % OR" not in message["text"]
    assert "Zeitfenster-Prognose" in message["text"]
    assert message["payload"]["predictedOR"] > 0.0
    assert message["payload"]["predictedORAvailable"] is True
    assert message["payload"]["predictedORSource"] == "historical_slot_baseline"


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
    assert "Zeitfenster-Prognose" in text
    assert "4.77" not in message["payload"]["messageHtml"]
    assert message["payload"]["predictedOR"] > 0.0
    assert message["payload"]["predictedORAvailable"] is True
    assert message["payload"]["predictedORSource"] == "historical_slot_baseline"
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


def test_sport_excluded_by_default_even_without_allow_list():
    candidate = _candidate(
        score=95.0,
        category="sport",
        title="Bayern-Star vor Wechsel: Entscheidung gefallen",
        url="https://www.bild.de/sport/article-1",
        predictedOR=0.07,
    )

    # Allow-Liste leer (= alles erlaubt), Sport muss trotzdem ausgeschlossen sein.
    decision = shouldNotifyTeams(
        candidate,
        _context(candidate),
        _config(allowed_sections=()),
    )

    assert decision["shouldNotify"] is False
    assert any("ausgeschlossen" in reason for reason in decision["blockingReasons"])


def test_sport_allowed_when_explicitly_configured():
    candidate = _candidate(
        score=92.0,
        category="sport",
        title="Eilmeldung: DFB-Team verliert Trainer ueberraschend",
        url="https://www.bild.de/sport/article-2",
        predictedOR=0.07,
    )
    context = _context(candidate)
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(allowed_sections=("sport",), excluded_sections=()),
    )

    assert not any("ausgeschlossen" in reason for reason in decision["blockingReasons"])


def _afternoon_ts() -> int:
    return int(dt.datetime(2026, 6, 19, 15, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())


def test_dynamic_threshold_drops_when_too_few_pushes_today():
    ts = _afternoon_ts()
    candidate = _candidate()

    base = shouldNotifyTeams(
        candidate,
        _context(candidate, now_ts=ts),
        _config(dynamic_threshold_enabled=False),
    )
    low_context = _context(candidate, now_ts=ts)
    low_context["pushesToday"] = 1
    lowered = shouldNotifyTeams(
        candidate,
        low_context,
        _config(dynamic_threshold_enabled=True, target_pushes_per_day=11),
    )

    assert lowered["teamsAlertScoreThreshold"] < base["teamsAlertScoreThreshold"]
    assert "Rueckstand" in lowered["pushBudgetReason"]


def test_dynamic_threshold_rises_when_too_many_pushes_today():
    ts = _afternoon_ts()
    candidate = _candidate()

    base = shouldNotifyTeams(
        candidate,
        _context(candidate, now_ts=ts),
        _config(dynamic_threshold_enabled=False),
    )
    high_context = _context(candidate, now_ts=ts)
    high_context["pushesToday"] = 15
    raised = shouldNotifyTeams(
        candidate,
        high_context,
        _config(dynamic_threshold_enabled=True, target_pushes_per_day=11),
    )

    assert raised["teamsAlertScoreThreshold"] > base["teamsAlertScoreThreshold"]
    assert "budget" in raised["pushBudgetReason"].lower()


def test_max_alerts_per_day_blocks_further_alerts():
    candidate = _candidate(score=95.0, predictedOR=0.08)
    context = _context(candidate, teams_alerts_today=14)

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(max_alerts_per_day=14),
    )

    assert decision["shouldNotify"] is False
    assert any("Tageslimit" in reason for reason in decision["blockingReasons"])


def test_max_alerts_per_day_override_for_breaking():
    candidate = _candidate(
        score=95.0,
        predictedOR=0.08,
        title="Eilmeldung: Israel und Iran einigen sich auf Feuerpause",
        isBreaking=True,
        isEilmeldung=True,
    )
    context = _context(candidate, teams_alerts_today=20)

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(max_alerts_per_day=14, breaking_override=True),
    )

    assert not any("Tageslimit" in reason for reason in decision["blockingReasons"])


def test_require_valid_prediction_blocks_fallback_forecast():
    candidate = _candidate(
        score=90.0,
        predictedOR=None,
        title="Wetterdienst gibt Hitzewarnung fuer Deutschland raus",
        url="https://www.bild.de/news/wetter/hitzewarnung-pred",
        category="news",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=45))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, require_valid_prediction=True, min_alert_score=50.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Belastbare OR-Prognose erforderlich" in reason for reason in decision["blockingReasons"])


def test_constant_field_forecast_is_treated_as_non_belastbar():
    cands = [
        _candidate(
            id=f"const-{i}",
            url=f"https://www.bild.de/news/const-{i}",
            title=f"Wichtige Nachricht Nummer {i} aus der Politik heute Abend",
            predictedOR=0.0477,
        )
        for i in range(4)
    ]
    context = build_teams_alert_context(
        cands,
        history=_history(),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        now_ts=NOW_TS,
    )

    assert 4.77 in context["suspectForecastValues"]

    decision = shouldNotifyTeams(cands[0], context, _config())

    assert decision["forecast"]["source"] != "article_model"
    assert decision["forecastSuspectedDefault"] is True
    assert decision["forecastSuspectValue"] == 4.77


def test_two_known_default_forecasts_are_flagged():
    cands = [
        _candidate(id="kd-1", url="https://www.bild.de/news/kd-1", predictedOR=0.0477),
        _candidate(
            id="kd-2",
            url="https://www.bild.de/news/kd-2",
            title="Ganz anderer Aufmacher mit eigener Schlagzeile heute",
            predictedOR=0.0477,
        ),
    ]
    context = build_teams_alert_context(
        cands,
        history=_history(),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        now_ts=NOW_TS,
    )

    assert 4.77 in context["suspectForecastValues"]


def test_lone_forecast_value_is_not_flagged_as_default():
    candidate = _candidate(predictedOR=0.0477)
    context = _context(candidate)

    assert 4.77 not in context["suspectForecastValues"]

    decision = shouldNotifyTeams(candidate, context, _config())

    assert decision["forecast"]["source"] == "article_model"
    assert decision["forecastSuspectedDefault"] is False


def test_select_teams_push_recommendation_picks_best_and_builds_message():
    first = _candidate(id="article-1", url="https://www.bild.de/politik/article-1", score=95.0)
    second = _candidate(
        id="article-2",
        url="https://www.bild.de/politik/article-2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket",
        score=82.0,
        predictedOR=0.061,
    )
    context = build_teams_alert_context(
        [first, second],
        history=_history(),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        now_ts=NOW_TS,
    )

    result = selectTeamsPushRecommendation([first, second], context, _config())

    assert result["selected"]["url"] == first["url"]
    assert result["decision"]["shouldNotify"] is True
    assert result["recommendation"]["text"].startswith("🚨 Jetzt pushen:")


def test_select_teams_push_recommendation_returns_none_for_weak_field():
    candidate = _candidate(score=50.0)
    context = _context(candidate)

    result = selectTeamsPushRecommendation([candidate], context, _config())

    assert result["selected"] is None
    assert result["recommendation"] is None


def test_uncertain_field_without_clear_winner_sends_no_alert():
    first = _candidate(id="u1", url="https://www.bild.de/politik/u1", score=84.0, predictedOR=0.06)
    second = _candidate(
        id="u2",
        url="https://www.bild.de/politik/u2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket heute Mittag",
        score=83.5,
        predictedOR=0.06,
    )
    context = build_teams_alert_context(
        [first, second],
        history=_history(),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        now_ts=NOW_TS,
    )

    # Hoher Margin-Schwellenwert + hoher Clear-Buffer erzwingen die Unsicherheits-Pruefung.
    result = evaluate_teams_alert_candidates(
        [first, second],
        context,
        _config(min_selection_margin=40.0, selection_clear_editorial_buffer=25.0, min_editorial_score=70.0),
    )

    assert result["selectedCandidateId"] is None
    assert result["fieldUncertain"] is True
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}
    assert all(not d["shouldNotify"] for d in decisions.values())
    assert any("Feld unsicher" in reason for reason in decisions[first["url"]]["blockingReasons"])


def test_clear_strong_winner_still_alerts_despite_margin_rule():
    strong = _candidate(id="w1", url="https://www.bild.de/politik/w1", score=95.0, predictedOR=0.08)
    weak = _candidate(
        id="w2",
        url="https://www.bild.de/unterhaltung/w2",
        title="Sommertrend: Diese Stars feiern neue Rabatt-App",
        category="unterhaltung",
        score=72.0,
        predictedOR=0.052,
    )
    context = build_teams_alert_context(
        [strong, weak],
        history=_history(),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates([strong, weak], context, _config())

    assert result["selectedCandidateId"] == strong["url"]
    assert result["fieldUncertain"] is False


def test_teams_message_is_compact_and_jargon_free():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    # Kein internes Modell-Jargon in der Nachricht.
    assert "von 10" not in text
    assert "von 100" not in text
    assert "Teams-Alert-Modell" not in text
    # Sauberer, nicht doppelter Abschluss (kein wiederholtes "Jetzt pushen" mit vollem Datum).
    assert "Empfehlung um" not in text
    assert "Empfehlung: Jetzt pushen. (Stand " in text
    # "Warum jetzt?" fuehrt mit der inhaltlichen Substanz (Top-Performance-Driver).
    why_block = text.split("Warum jetzt?\n", 1)[1]
    first_bullet = why_block.splitlines()[0]
    assert first_bullet == f"- {candidate['performanceDrivers'][0]}"


def test_time_fit_label_uses_real_umlauts_for_early_window():
    early = int(dt.datetime(2026, 6, 19, 6, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Eilmeldung: Israel und Iran einigen sich auf Feuerpause",
        url="https://www.bild.de/politik/feuerpause-frueh",
        isBreaking=True,
        isEilmeldung=True,
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=45, now_ts=early),
        now_ts=early,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(min_alert_score=50.0, min_editorial_score=50.0, min_time_fit_score=4.0),
    )
    label = decision["editorialReview"]["breakdown"]["timeFitLabel"]

    assert "frühes" in label
    assert "fruehes" not in label
    assert "fuer" not in label


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
