import datetime as dt
import logging
import time
import urllib.error
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from app.notifications.teams import (
    TeamsAlertConfig,
    buildTeamsDailyPushPlan,
    buildTeamsPushRecommendation,
    build_teams_alert_context,
    build_teams_daily_schedule,
    evaluate_and_send_best_candidate,
    evaluate_teams_alert_candidates,
    _daily_runtime_opportunities,
    _format_time,
    _is_breaking,
    _sport_candidate_review,
    normalize_predicted_or,
    selectTeamsPushRecommendation,
    send_teams_daily_schedule_if_due,
    sendTeamsNotification,
    shouldNotifyTeams,
)
from app.notifications.teams_review import REVIEWERS
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
        "min_alerts_per_day": 11,
        "max_alerts_per_day": 14,
        "agent_review_enabled": True,
        # Legacy decision tests exercise editorial gates in isolation. Dedicated
        # slot-gate tests below enable the new :45 production behaviour explicitly.
        "slot_gate_enabled": False,
    }
    values.update(overrides)
    return TeamsAlertConfig(**values)


def _smart_config(**overrides):
    values = {
        "allowed_sections": (
            "news",
            "politik",
            "wirtschaft",
            "geld",
            "regional",
            "digital",
            "unterhaltung",
            "sport",
        ),
        "excluded_sections": (),
        "target_pushes_per_day": 15,
        "min_alerts_per_day": 15,
        "max_alerts_per_day": 18,
        "slot_gate_enabled": True,
        "dynamic_threshold_enabled": True,
    }
    values.update(overrides)
    return _config(**values)


def _candidate(**overrides):
    candidate = {
        "id": "article-1",
        "url": "https://www.bild.de/politik/article-1",
        "title": "Regierung beschliesst Rentenpaket für Millionen Beschäftigte",
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
        "recommendedText": "Rentenpaket: Was der Beschluss für Millionen Beschäftigte bedeutet",
        "isBreaking": False,
        "isEilmeldung": False,
    }
    candidate.update(overrides)
    return candidate


def _history(minutes_since_last_push=50, now_ts=NOW_TS, **overrides):
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
    if now_ts != NOW_TS and candidate.get("pubDate") == _iso(NOW_TS - 10 * 60):
        candidate["pubDate"] = _iso(now_ts - 10 * 60)
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


def test_recent_live_push_does_not_block_independent_teams_recommendation():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=_history(minutes_since_last_push=12)),
        _config(),
    )

    assert decision["shouldNotify"] is True
    assert decision["pacingBasis"] == "teams_alerts"
    assert decision["recommendationsIndependentFromLivePushes"] is True
    assert not any("Pause seit letztem Push" in reason for reason in decision["blockingReasons"])


def test_missing_last_live_push_timestamp_does_not_block_independent_channel():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(min_alerts_per_day=0),
    )

    assert decision["shouldNotify"] is True
    assert decision["recommendationsIndependentFromLivePushes"] is True
    assert not any("Letzter Push-Zeitpunkt" in reason for reason in decision["blockingReasons"])


def test_bad_forecast_does_not_trigger_teams_decision():
    candidate = _candidate(predictedOR=0.039)
    context = _context(candidate, teams_alerts_today=11)
    context["pushesToday"] = 11

    decision = shouldNotifyTeams(candidate, context, _config())

    assert decision["shouldNotify"] is False
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])


def test_score_only_mode_does_not_override_bad_article_forecast():
    candidate = _candidate(score=88.0, predictedOR=0.0377)
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])


def test_historical_slot_forecast_alone_does_not_allow_normal_alert():
    candidate = _candidate(
        score=90.0,
        predictedOR=None,
        category="news",
        title="Große Gasanlage betroffen: Details nach Explosion in Katar",
        url="https://www.bild.de/news/katar-gas-details",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1
    context["pushesToday"] = 3
    context["teamsAlertsToday"] = 3

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Artikel-Prognose fehlt" in reason for reason in decision["blockingReasons"])


def test_minimum_pacing_allows_real_event_with_slot_forecast_when_day_is_behind():
    evening_ts = NOW_TS + 11 * 3600
    candidate = _candidate(
        score=80.0,
        predictedOR=None,
        category="news",
        title="Polizei nimmt mutmaßlichen Täter nach Angriff fest",
        url="https://www.bild.de/news/polizei-festnahme-angriff",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=evening_ts),
        now_ts=evening_ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 4
    context["teamsAlertsToday"] = 4

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=78.0,
            min_editorial_score=74.0,
            max_alerts_per_day=11,
        ),
    )

    assert decision["shouldNotify"] is True
    assert decision["minimumPressure"]["active"] is True
    assert any("Mindest-Pacing" in reason for reason in decision["reasons"])


def test_minimum_pacing_uses_teams_count_even_when_actual_push_count_is_available():
    noon_ts = int(dt.datetime(2026, 6, 24, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        score=84.0,
        predictedOR=0.061,
        category="news",
        title="Streik legt Bahnverkehr in mehreren Bundeslaendern lahm",
        url="https://www.bild.de/news/bahn-streik-verkehr",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=noon_ts),
        now_ts=noon_ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 5
    context["teamsAlertsToday"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=78.0,
            min_editorial_score=68.0,
            slot_gate_enabled=False,
            target_pushes_per_day=15,
            min_alerts_per_day=15,
            max_alerts_per_day=18,
        ),
    )

    assert decision["minimumPressure"]["active"] is True
    assert decision["minimumPressure"]["basis"] == "teams_alerts"
    assert decision["minimumPressure"]["current"] == 1
    assert decision["minimumPressure"]["actualPushesToday"] == 5
    assert decision["minimumPressure"]["actualPushesToday"] == 5
    assert decision["minimumPressure"]["teamsAlertsToday"] == 1


def test_minimum_pacing_never_waives_soft_or_or_wait_gate():
    afternoon_ts = int(
        dt.datetime(2026, 6, 24, 14, 36, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        score=77.1,
        predictedOR=0.0475,
        category="politik",
        title="Regierungsbefragung im Bundestag: Kanzler im Kreuzfeuer",
        url="https://www.bild.de/politik/regierungsbefragung-kanzler-kreuzfeuer",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=45, now_ts=afternoon_ts),
        teams_alerts_today=0,
        now_ts=afternoon_ts,
    )
    context["dashboardRank"] = 5
    context["pushesToday"] = 3

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=78.0,
            min_editorial_score=74.0,
            min_or=5.0,
            target_pushes_per_day=15,
            min_alerts_per_day=15,
            max_alerts_per_day=18,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["minimumPressure"]["active"] is True
    assert decision["minimumPressure"]["basis"] == "teams_alerts"
    assert decision["minimumPressure"]["thresholdDrop"] == 0.0
    assert decision["teamsAlertScoreThreshold"] == 78.0
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])
    assert any("Teams-Mindest-Pacing aktiv" in reason for reason in decision["reasons"])


def test_minimum_pacing_does_not_rescue_crime_below_quality_floors():
    afternoon_ts = int(
        dt.datetime(2026, 6, 30, 14, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        id="crime-hard-news",
        url="https://www.bild.de/crime/stade-schuesse",
        category="crime",
        title="6 Tote nach Schüssen in Stade: Polizei nimmt Verdächtigen fest",
        score=68.8,
        predictedOR=0.039,
        pubDate=_iso(afternoon_ts - 20 * 60),
    )
    context = _context(
        candidate,
        history=[],
        teams_alerts_today=0,
        now_ts=afternoon_ts,
    )
    context["dashboardRank"] = 9
    context["pushesToday"] = 0

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            allowed_sections=(
                "news",
                "politik",
                "wirtschaft",
                "regional",
                "digital",
                "unterhaltung",
            ),
            dynamic_threshold_enabled=True,
            target_pushes_per_day=15,
            min_alerts_per_day=15,
            max_alerts_per_day=18,
            min_alert_score=78.0,
            min_editorial_score=74.0,
            min_editorial_news_value=24.0,
            min_or=5.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["minimumPressure"]["active"] is True
    assert decision["minimumPressure"]["basis"] == "teams_alerts"
    assert decision["minimumPressure"]["thresholdDrop"] == 0.0
    assert decision["teamsAlertScoreThreshold"] == 78.0
    assert any("Score zu niedrig" in reason for reason in decision["blockingReasons"])
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])


def test_minimum_pacing_does_not_allow_curiosity_story():
    evening_ts = NOW_TS + 11 * 3600
    candidate = _candidate(
        score=82.0,
        predictedOR=None,
        category="news",
        title="Schock auf dem Highway: Millionen Bienen entkommen nach Lkw-Unfall",
        url="https://www.bild.de/news/highway-lkw-unfall-minimum",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=evening_ts),
        now_ts=evening_ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 4
    context["teamsAlertsToday"] = 4

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=78.0,
            min_editorial_score=74.0,
            max_alerts_per_day=11,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["minimumPressure"]["active"] is True
    assert any("Kurios-/Click-Reiz" in reason for reason in decision["blockingReasons"])


def test_live_ticker_without_real_new_development_is_blocked():
    candidate = _candidate(
        score=88.5,
        predictedOR=0.0555,
        category="regional",
        title=(
            "Live-Ticker zum Prozess um Fabian: Vier Polizisten sagen heute aus! "
            "Wie verhielt sich Gina H.?"
        ),
        url="https://www.bild.de/regional/fabian-prozess-live",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Live-Ticker ohne neue" in reason for reason in decision["blockingReasons"])


def test_process_schedule_without_new_development_is_blocked_even_without_live_ticker():
    candidate = _candidate(
        score=88.5,
        predictedOR=0.0555,
        category="regional",
        title="Prozess um Fabian: Vier Polizisten heute im Zeugenstand",
        url="https://www.bild.de/regional/fabian-prozess-polizisten",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Termin-/Prozesslage" in reason for reason in decision["blockingReasons"])


def test_live_ticker_with_decisive_update_can_pass_cvd_gate():
    candidate = _candidate(
        score=91.0,
        predictedOR=0.061,
        category="regional",
        title="Live-Ticker: Gericht verurteilt Angeklagten im Fabian-Prozess",
        url="https://www.bild.de/regional/fabian-urteil-live",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is True
    assert not any("Live-Ticker ohne neue" in reason for reason in decision["blockingReasons"])


def test_explainer_question_without_new_development_is_blocked():
    candidate = _candidate(
        score=88.0,
        predictedOR=0.0522,
        category="news",
        title="E-Autos brennen häufiger? Vorurteile auf dem Prüfstand",
        url="https://www.bild.de/news/e-autos-brand-vorurteile",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Erklär-/Debattenstück" in reason for reason in decision["blockingReasons"])


def test_nonessential_curiosity_story_is_blocked_despite_high_forecast():
    candidate = _candidate(
        score=78.8,
        predictedOR=0.06,
        category="news",
        title="Schock auf dem Highway: Millionen Bienen entkommen nach Lkw-Unfall",
        url="https://www.bild.de/news/highway-lkw-unfall",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Kurios-/Click-Reiz" in reason for reason in decision["blockingReasons"])


def test_low_civic_impact_accident_story_does_not_win_on_or_alone():
    candidate = _candidate(
        score=85.5,
        predictedOR=0.0611,
        category="news",
        title="Unfall mit Folgen: Bienenstich-Alarm auf Autobahn",
        url="https://www.bild.de/news/unfall-mit-folgen-bienenstich-alarm-auf-autobahn",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("enger Kurios-/Click-Reiz" in reason for reason in decision["blockingReasons"])
    assert decision["visitPotential"]["audienceFactor"] < 1.0


def test_fabian_topic_variant_is_blocked_after_recent_teams_alert():
    candidate = _candidate(
        score=88.5,
        predictedOR=0.0555,
        category="regional",
        title="Prozess um Fabian: Vier Polizisten heute im Zeugenstand",
        url="https://www.bild.de/regional/fabian-prozess-polizisten",
    )
    recent = [
        {
            "key": "https://www.bild.de/regional/fabian-prozess-live",
            "title": (
                "Live-Ticker zum Prozess um Fabian: Vier Polizisten sagen heute aus! "
                "Wie verhielt sich Gina H.?"
            ),
        }
    ]
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90),
        recent_alerts=recent,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Dublette" in reason for reason in decision["blockingReasons"])


def test_non_breaking_push_title_does_not_add_false_eil_prefix():
    from app.notifications.teams import _teams_push_title_recommendation

    candidate = _candidate(
        isBreaking=False,
        isEilmeldung=False,
        title="Elektroauto-Vorurteile: Brand bei E-Autos häufiger als bei Verbrenner?",
        recommendedText="EIL: Brand bei E-Autos häufiger als bei Verbrenner",
    )

    title, source = _teams_push_title_recommendation(
        candidate,
        candidate["title"],
        "news",
        candidate["url"],
        _config(llm_title_enabled=False),
    )

    assert source == "editorial"
    assert not title.startswith("EIL:")
    assert title == "Brand bei E-Autos häufiger als bei Verbrenner"


def test_score_only_mode_does_not_require_a_live_push_timestamp():
    candidate = _candidate(score=82.0, predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True, min_alerts_per_day=0),
    )

    assert decision["shouldNotify"] is True
    assert decision["scoreOnlyMode"] is True
    assert decision["recommendationsIndependentFromLivePushes"] is True


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
            allowed_sections=(
                "news",
                "politik",
                "wirtschaft",
                "regional",
                "digital",
                "unterhaltung",
            ),
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
    assert any(
        "Nicht im oberen Push-Balancer-Feld" in reason for reason in decision["blockingReasons"]
    )


def test_strong_visit_pattern_outside_dashboard_top_limit_can_notify():
    candidate = _candidate(
        id="public-fraud-raid",
        url="https://www.bild.de/news/grossrazzia-leistungsbetrueger",
        title="200 Polizisten im Einsatz: Grossrazzia gegen Leistungsbetrueger",
        category="news",
        score=87.5,
        predictedOR=0.094,
    )
    context = _context(candidate)
    context["dashboardRank"] = 35

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20, editorial_top_limit=10, candidate_limit=80),
    )

    assert decision["shouldNotify"] is True
    assert decision["expandedFieldCandidate"] is True
    assert any("Expanded Field" in reason for reason in decision["reasons"])


def test_soft_candidate_outside_dashboard_top_limit_stays_blocked():
    candidate = _candidate(
        id="soft-app",
        url="https://www.bild.de/digital/sprachlern-app",
        title="Schock fuer Fans: Beliebte Sprachlern-App vor dem Aus",
        category="digital",
        score=95.0,
        predictedOR=0.09,
    )
    context = _context(candidate)
    context["dashboardRank"] = 35

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20, editorial_top_limit=10, candidate_limit=80),
    )

    assert decision["shouldNotify"] is False
    assert decision["expandedFieldCandidate"] is False
    assert any(
        "Nicht im oberen Push-Balancer-Feld" in reason for reason in decision["blockingReasons"]
    )


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


def test_weighted_model_allows_breaking_without_live_push_timing():
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
        _config(
            score_only_mode=True, min_score=75.0, breaking_min_score=72.0, min_alert_score=78.0
        ),
    )

    assert decision["shouldNotify"] is True
    assert decision["recommendationsIndependentFromLivePushes"] is True


def test_score_only_mode_does_not_use_lower_breaking_threshold():
    candidate = _candidate(
        score=79.0,
        predictedOR=None,
        isBreaking=True,
        title="Eilmeldung: Regierung beschliesst Rentenpaket für Millionen Beschäftigte",
    )

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


def test_live_pushed_article_remains_an_independent_teams_recommendation():
    candidate = _candidate()
    pushed_history = _history(link=candidate["url"], title="Anderer Titel")

    decision = shouldNotifyTeams(candidate, _context(candidate, history=pushed_history), _config())

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert not any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_same_live_story_under_different_url_is_comparison_only():
    # Echter Push der gleichen Story unter anderer URL + push-optimiertem Titel.
    candidate = _candidate(
        title="Bund beschliesst Strombonus fuer Millionen Haushalte",
        url="https://www.bild.de/news/inland/bund-strombonus-millionen-haushalte",
        category="news",
    )
    pushed = _history(
        minutes_since_last_push=90,
        link="https://www.bild.de/news/inland/strombonus-millionen-haushalte-beschlossen",
        title="Strombonus beschlossen: Millionen Haushalte profitieren",
    )

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=pushed),
        _config(min_alert_score=70.0, min_editorial_score=70.0),
    )

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] in {
        "same_story_slug",
        "same_story_title",
    }


def test_live_push_comparison_canonicalizes_bild_host_scheme_query_and_amp():
    candidate = _candidate(
        url="https://www.bild.de/news/eindeutige-artikel-identitaet?utm_source=test",
        title="Regierung beschliesst neue Hilfe fuer Familien",
        category="news",
    )
    history = _history(
        minutes_since_last_push=60,
        link="http://m.bild.de/news/eindeutige-artikel-identitaet/amp#top",
        title="Andere historische Headline",
        headline="Andere historische Headline",
    )

    decision = shouldNotifyTeams(candidate, _context(candidate, history=history), _config())

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"


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


def test_breaking_cannot_repeat_an_article_already_recommended_in_teams():
    candidate = _candidate(
        title="Eilmeldung: Regierung beschliesst sofort neue Hilfen",
        score=95.0,
        predictedOR=0.08,
        isBreaking=True,
        isEilmeldung=True,
    )
    alert_state = {
        candidate["url"]: {
            "status": "sent",
            "last_alert_ts": NOW_TS - 24 * 3600,
            "last_decision_ts": NOW_TS - 24 * 3600,
            "last_score": 90.0,
            "last_predicted_or": 7.0,
            "last_candidate_updated_at": NOW_TS - 24 * 3600,
            "last_is_breaking": 0,
            "alert_count": 1,
        }
    }

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, alert_state=alert_state),
        _config(),
    )

    assert decision["shouldNotify"] is False
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
    assert any(
        "Bereits als Teams-Kandidat versucht" in reason for reason in decision["blockingReasons"]
    )


def test_same_teams_article_never_realerts_even_after_score_improvement():
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

    no_realert = shouldNotifyTeams(
        candidate, _context(candidate, alert_state=alert_state), _config()
    )
    improved = shouldNotifyTeams(
        _candidate(score=87.0),
        _context(_candidate(score=87.0), alert_state=alert_state),
        _config(),
    )

    assert no_realert["shouldNotify"] is False
    assert any("Bereits per Teams gemeldet" in reason for reason in no_realert["blockingReasons"])
    assert improved["shouldNotify"] is False
    assert any("Bereits per Teams gemeldet" in reason for reason in improved["blockingReasons"])


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


def test_same_teams_article_key_stays_blocked_after_headline_change():
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

    assert decision["shouldNotify"] is False
    assert any("Bereits per Teams gemeldet" in reason for reason in decision["blockingReasons"])


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
        assert any(
            "Service-/Raetsel" in reason for reason in decision["blockingReasons"]
        ), bad_title


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
        assert any(
            "Service-/Raetsel" in reason for reason in decision["blockingReasons"]
        ), bad_title


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


def test_push_score_dominates_a_higher_response_forecast_between_eligible_candidates():
    first = _candidate(id="article-1", url="https://www.bild.de/politik/article-1", score=95.0)
    second = _candidate(
        id="article-2",
        url="https://www.bild.de/politik/article-2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket",
        category="politik",
        score=82.0,
        predictedOR=0.061,
    )
    context = build_teams_alert_context(
        [first, second],
        history=_history(),
        alert_state={},
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates([first, second], context, _config())
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == first["url"]
    assert decisions[second["url"]]["expectedVisits"] > decisions[first["url"]]["expectedVisits"]
    assert decisions[first["url"]]["shouldNotify"] is True
    assert decisions[first["url"]]["selectionScore"] > decisions[second["url"]]["selectionScore"]
    assert decisions[second["url"]]["shouldNotify"] is False
    assert any(
        "Staerkerer Kandidat vorhanden" in reason
        for reason in decisions[second["url"]]["blockingReasons"]
    )


def test_fresh_dashboard_rating_beats_a_more_negative_server_preference():
    useful = _candidate(
        id="useful",
        url="https://www.bild.de/news/inland/strombonus",
        title="Bund beschliesst Strombonus fuer Millionen Haushalte",
        category="news",
        score=86.0,
        scoreSource="captured_push_balancer",
        pushBalancerScore=86.0,
        serverEditorialScore=76.0,
        predictedOR=0.056,
    )
    crime = _candidate(
        id="crime",
        url="https://www.bild.de/news/inland/einbruchserie",
        title="Polizei ermittelt nach bundesweiter Einbruchserie",
        category="news",
        score=84.0,
        scoreSource="captured_push_balancer",
        pushBalancerScore=84.0,
        serverEditorialScore=96.0,
        predictedOR=0.085,
    )
    context = build_teams_alert_context(
        [useful, crime],
        history=_history(),
        alert_state={},
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates(
        [useful, crime],
        context,
        _config(min_alert_score=60.0, min_editorial_score=60.0, min_or=4.0),
    )
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == useful["url"]
    assert decisions[useful["url"]]["scoreSource"] == "captured_push_balancer"
    assert decisions[crime["url"]]["shouldNotify"] is False

    message = buildTeamsPushRecommendation(
        useful,
        context,
        decisions[useful["url"]],
        _config(min_alert_score=60.0, min_editorial_score=60.0, min_or=4.0),
    )
    assert message["payload"]["pushScoreSource"] == "captured_push_balancer"
    assert "pushBalancerScoreCapturedAt" not in message["payload"]
    assert "serverEditorialScore" not in message["payload"]
    assert "Quelle: frisches Push-Balancer-Rating" in message["text"]


def test_pure_us_domestic_people_story_is_blocked_even_with_high_push_score():
    candidate = _candidate(
        id="us-people",
        url="https://www.bild.de/news/ausland/us-mutter-zwillinge",
        title="Todesstrafe droht: US-Mutter soll ihre Zwillinge erstickt haben",
        category="news",
        score=96.0,
        predictedOR=0.09,
    )

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate),
        _config(min_alert_score=60.0, min_editorial_score=60.0),
    )

    assert decision["shouldNotify"] is False
    assert decision["germanyRelevance"]["level"] == "usa_domestic"
    assert any("rein US-inlaendische" in reason for reason in decision["blockingReasons"])


def test_german_candidate_wins_inside_three_point_push_score_band():
    international = _candidate(
        id="international",
        url="https://www.bild.de/politik/ausland-und-internationales/iran-krieg",
        title="Trump warnt vor weiterer Eskalation im Iran-Krieg",
        category="politik",
        score=86.0,
        predictedOR=0.064,
    )
    german = _candidate(
        id="german",
        url="https://www.bild.de/politik/rentenpaket",
        title="Regierung beschliesst Rentenpaket fuer Millionen Beschaeftigte",
        category="politik",
        score=84.0,
        predictedOR=0.058,
    )
    candidates = [international, german]
    context = build_teams_alert_context(
        candidates,
        history=_history(),
        alert_state={},
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates(
        candidates,
        context,
        _config(
            min_alert_score=60.0,
            min_editorial_score=60.0,
            visit_optimization_enabled=False,
        ),
    )

    assert result["selectedCandidateId"] == german["url"]
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}
    assert decisions[german["url"]]["germanyRelevance"]["level"] == "germany_broad"
    assert decisions[international["url"]]["germanyRelevance"]["level"] == "international"


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
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates(
        candidates,
        context,
        _config(min_alert_score=70.0, min_editorial_score=82.0, visit_optimization_enabled=False),
    )
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == stronger_cvd["url"]
    assert decisions[stronger_cvd["url"]]["isBreaking"] is True
    assert (
        decisions[high_raw["url"]]["selectionScore"]
        > decisions[stronger_cvd["url"]]["selectionScore"]
    )
    assert decisions[stronger_cvd["url"]]["shouldNotify"] is True
    assert decisions[high_raw["url"]]["shouldNotify"] is False


def test_visit_potential_cannot_override_a_seven_point_push_score_lead():
    now = NOW_TS
    slot_hour = dt.datetime.fromtimestamp(now, ZoneInfo("Europe/Berlin")).hour
    history = _history(minutes_since_last_push=55, now_ts=now)
    for idx in range(8):
        history.append(
            {
                "message_id": f"politics-reach-{idx}",
                "ts_num": now - (2 * 86400) - idx * 3600,
                "or": 6.8,
                "title": f"Politik-Historie {idx}",
                "headline": f"Politik-Historie {idx}",
                "cat": "politik",
                "link": f"https://www.bild.de/politik/history-{idx}",
                "hour": slot_hour,
                "total_recipients": 80000,
            }
        )
        history.append(
            {
                "message_id": f"news-reach-{idx}",
                "ts_num": now - (3 * 86400) - idx * 3600,
                "or": 5.2,
                "title": f"News-Historie {idx}",
                "headline": f"News-Historie {idx}",
                "cat": "news",
                "link": f"https://www.bild.de/news/history-{idx}",
                "hour": slot_hour,
                "total_recipients": 520000,
            }
        )

    narrow_high_or = _candidate(
        id="narrow-or",
        url="https://www.bild.de/politik/narrow-or",
        title="Regierung beschliesst neues Sicherheitspaket",
        category="politik",
        score=91.0,
        predictedOR=0.071,
    )
    broader_news = _candidate(
        id="broad-visits",
        url="https://www.bild.de/news/bahn-ausfall-visits",
        title="Warnung: Deutsche Bahn meldet bundesweiten Totalausfall",
        category="news",
        score=84.0,
        predictedOR=0.052,
    )
    context = build_teams_alert_context(
        [narrow_high_or, broader_news],
        history=history,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now,
    )

    result = evaluate_teams_alert_candidates(
        [narrow_high_or, broader_news],
        context,
        _config(min_alert_score=60.0, min_editorial_score=60.0, min_or=4.0),
    )
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}

    assert result["selectedCandidateId"] == narrow_high_or["url"]
    assert (
        decisions[broader_news["url"]]["expectedVisits"]
        > decisions[narrow_high_or["url"]]["expectedVisits"]
    )
    assert decisions[narrow_high_or["url"]]["shouldNotify"] is True
    assert any(
        "Response-Potenzial" in reason for reason in decisions[broader_news["url"]]["reasons"]
    )
    assert decisions[broader_news["url"]]["shouldNotify"] is False
    assert any(
        "Staerkerer Kandidat vorhanden" in reason
        for reason in decisions[broader_news["url"]]["blockingReasons"]
    )


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


def test_public_money_fraud_razzia_can_pass_near_or_threshold():
    early_morning = int(
        dt.datetime(2026, 6, 25, 6, 31, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        score=75.5,
        predictedOR=0.049,
        category="news",
        title="200 Polizisten im Einsatz: Großrazzia gegen Leistungsbetrüger",
        url="https://www.bild.de/news/grossrazzia-leistungsbetrueger",
        pubDate=_iso(early_morning - 15 * 60),
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=480, now_ts=early_morning),
        now_ts=early_morning,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 0
    context["teamsAlertsToday"] = 0

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dynamic_threshold_enabled=True),
    )

    assert decision["shouldNotify"] is True
    assert decision["teamsAlertScore"] >= decision["teamsAlertScoreThreshold"]
    assert decision["predictedOR"] == 4.9
    assert any("OR knapp unter Schwelle" in reason for reason in decision["reasons"])


def test_evening_celebrity_relationship_money_conflict_can_pass_near_or_threshold():
    evening = int(dt.datetime(2026, 6, 24, 20, 1, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        score=78.0,
        predictedOR=0.049,
        category="unterhaltung",
        title="Wie bei so vielen Paaren – es geht ums Geld | Scheidungszoff bei WM-Held Schweini",
        url="https://www.bild.de/unterhaltung/schweini-scheidungszoff",
        pubDate=_iso(evening - 30 * 60),
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=evening),
        now_ts=evening,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 7
    context["teamsAlertsToday"] = 7

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dynamic_threshold_enabled=True),
    )

    assert decision["shouldNotify"] is True
    assert decision["teamsAlertScore"] >= decision["teamsAlertScoreThreshold"]
    assert decision["editorialReview"]["newsValue"] >= 30
    assert any("Promi-/Beziehungs-/Geldkonflikt" in reason for reason in decision["reasons"])


def test_confirmed_german_public_figure_parenthood_can_pass_people_gate_at_feierabend():
    evening = int(dt.datetime(2026, 7, 15, 17, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        id="people-parenthood",
        score=87.0,
        predictedOR=0.0475,
        predictedORBasis="lightgbm",
        predictedORConfidence=0.70,
        category="unterhaltung",
        title="CDU-Politiker Max Beispiel und sein Partner sind Papas geworden",
        recommendedText="CDU-Politiker Max Beispiel und sein Partner sind Papas geworden",
        url="https://example.invalid/unterhaltung/stars-und-leute/beispiel",
        pubDate=_iso(evening - 20 * 60),
        performanceDrivers=[
            "BILD-Reiz: bestaetigte Elternschaft einer benannten deutschen oeffentlichen Person"
        ],
        risks=[],
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=evening),
        now_ts=evening,
    )
    context["dashboardRank"] = 1
    context["suspectForecastValues"] = [4.75]

    decision = shouldNotifyTeams(
        candidate,
        context,
        _smart_config(
            slot_gate_enabled=False,
            min_score=75.0,
            min_alert_score=66.0,
            min_editorial_score=66.0,
            min_editorial_news_value=24.0,
            require_article_forecast=True,
            no_forecast_min_alert_score=76.0,
        ),
    )

    assert decision["shouldNotify"] is True
    assert decision["germanyRelevance"]["level"] == "germany_people"
    assert decision["forecast"]["source"] == "historical_slot_baseline"
    assert decision["teamsAlertScore"] >= 80.0
    assert decision["editorialReview"]["newsValue"] >= 28.0
    assert any("People-Ereignis" in reason for reason in decision["reasons"])
    assert not any(
        "kein konkretes Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"]
    )
    assert not any("Artikel-Prognose fehlt" in reason for reason in decision["blockingReasons"])


def test_evening_celebrity_money_conflict_does_not_reopen_sport_section():
    evening = int(dt.datetime(2026, 6, 24, 20, 1, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        score=78.0,
        predictedOR=0.049,
        category="sport",
        title="Wie bei so vielen Paaren – es geht ums Geld | Scheidungszoff bei WM-Held Schweini",
        url="https://www.bild.de/sport/schweini-scheidungszoff",
        pubDate=_iso(evening - 30 * 60),
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=evening),
        now_ts=evening,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dynamic_threshold_enabled=True),
    )

    assert decision["shouldNotify"] is False
    assert any(
        "sport" in reason.lower() and "ausgeschlossen" in reason.lower()
        for reason in decision["blockingReasons"]
    )


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


def test_daily_strategy_blocks_normal_candidate_when_push_count_is_ahead():
    candidate = _candidate(
        score=82.0,
        predictedOR=0.053,
        category="news",
        title="Regierung kündigt neue Regel für Verbraucher an",
        url="https://www.bild.de/news/verbraucher-regel",
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1
    context["pushesToday"] = 8
    context["teamsAlertsToday"] = 11

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=40.0,
            min_editorial_score=40.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert any("Tagesstrategie" in reason for reason in decision["blockingReasons"])


def test_daily_strategy_allows_breaking_candidate_when_push_count_is_ahead():
    candidate = _candidate(
        score=88.0,
        predictedOR=0.052,
        category="politik",
        title="Eilmeldung: Israel und Iran einigen sich auf Feuerpause",
        url="https://www.bild.de/politik/feuerpause-breaking",
        isBreaking=True,
        isEilmeldung=True,
    )
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1
    context["pushesToday"] = 8

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=40.0,
            min_editorial_score=40.0,
        ),
    )

    assert decision["shouldNotify"] is True
    assert not any("Tagesstrategie" in reason for reason in decision["blockingReasons"])


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
    context = _context(
        candidate, history=_history(minutes_since_last_push=45, now_ts=night_ts), now_ts=night_ts
    )
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
    context = _context(
        candidate, history=_history(minutes_since_last_push=45, now_ts=night_ts), now_ts=night_ts
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(min_alert_score=60.0, min_editorial_score=50.0, min_time_fit_score=4.0),
    )

    assert decision["shouldNotify"] is False
    assert any("Ruhezeit" in reason for reason in decision["blockingReasons"])
    assert decision["editorialReview"]["breakdown"]["timeFit"] >= 4.0


def test_generic_teams_sender_blocks_every_payload_during_quiet_hours():
    night_ts = int(dt.datetime(2026, 7, 15, 2, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    message = {
        "payload": {
            "type": "push_daily_schedule",
            "messageHtml": "<p>Synthetic schedule</p>",
        }
    }

    with (
        patch("app.notifications.teams.time.time", return_value=night_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, _config())

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "quiet_hours"
    urlopen.assert_not_called()


def test_mandatory_quiet_hours_cannot_be_disabled_by_runtime_config():
    night_ts = int(dt.datetime(2026, 7, 15, 4, 45, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    config = _config(quiet_hours_start="00:00", quiet_hours_end="00:00")

    with (
        patch("app.notifications.teams.time.time", return_value=night_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification({"payload": {"text": "Synthetic message"}}, config)

    assert result["blocked"] is True
    assert result["reason"] == "quiet_hours"
    urlopen.assert_not_called()


def test_transport_rejects_recommendation_below_mandatory_raw_score_floor():
    now_ts = _gold_slot_ts()
    candidate = _candidate(score=82.0)
    config = _config()
    context = _context(candidate, now_ts=now_ts)
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    message["payload"]["pushScore"] = 68.1

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["blocked"] is True
    assert "score" in result["error"].lower()
    urlopen.assert_not_called()


def test_transport_rejects_live_push_dependent_recommendation_payload():
    now_ts = _gold_slot_ts()
    candidate = _candidate(score=82.0)
    config = _config()
    context = _context(candidate, now_ts=now_ts)
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    message["payload"]["recommendationsIndependentFromLivePushes"] = False

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["blocked"] is True
    assert "independent" in result["error"].lower()
    urlopen.assert_not_called()


def test_teams_message_contains_required_editorial_fields():
    candidate = _candidate()
    context = _context(candidate, now_ts=_gold_slot_ts())
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert text.startswith(f"🚨 Jetzt pushen: {candidate['recommendedText']}")
    assert "Empfohlener Push-Text:" not in text
    assert f"Empfohlener Push-Titel:\n{candidate['recommendedText']}" in text
    assert "Artikel:" in text
    assert "Versandfenster: Jetzt senden, ideal bis" in text
    assert "Qualitätsurteil:" in text
    assert "Warum dieser Push?" in text
    assert "Deutschland-Relevanz:" in text
    assert "Warum jetzt?" in text
    assert "Gegencheck:" in text
    assert candidate["title"] in text
    assert candidate["url"] in text
    assert "Politik | OR-Prognose 5,20 % OR" in text
    assert "Teams-Alert-Score: " not in text
    assert "Push-Balancer-Breakdown:" not in text
    assert "Die Artikel-Prognose liegt aktuell bei 5,20 % OR." in text
    assert "letzter Teams-Hinweis noch keiner bekannt" in text
    assert "unabhängig von echten Live-Pushes" in text
    assert "Live-Vergleich:" in text
    payload = message["payload"]
    assert payload["recommendedAction"] == "Jetzt pushen"
    assert payload["recommendationPolicyVersion"] == "internal-score-golden-slots-v4"
    assert payload["recommendationsIndependentFromLivePushes"] is True
    assert payload["livePushComparison"] == {
        "available": True,
        "matched": False,
        "matchType": "",
    }
    assert payload["minimumPushScore"] >= 75.0
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
    assert payload["expectedVisits"] > 0
    assert payload["estimatedReach"] > 0
    assert payload["visitPotentialScore"] > 0
    assert payload["responseMetric"] == "expected_opens"
    assert payload["expectedOpens"] > 0
    assert "erwartete Öffnungen" in payload["messageText"]
    assert payload["timeFitScore"] > 0
    assert payload["timeFitLabel"]
    assert payload["recommendedPushText"] == candidate["recommendedText"]
    assert payload["alternativePushTitle"] == candidate["recommendedText"]
    assert payload["pushTitleReview"]["approved"] is True
    assert payload["pushTitleReview"]["clickReason"]
    assert payload["recommendationQuality"]["score"] > 0
    assert payload["recommendationQuality"]["confidence"] in {"hoch", "mittel", "niedrig"}
    assert payload["recommendationQuality"]["window"]["sendBy"]
    assert payload["recommendedSendWindow"].startswith("Jetzt senden")
    assert payload["messageText"] == text
    assert "Warum dieser Push?" in payload["messageHtml"]
    assert "Warum jetzt?" in payload["messageHtml"]
    assert "Gegencheck:" in payload["messageHtml"]
    assert payload["subject"].startswith("🚨 Jetzt pushen:")
    assert "Push-Balancer-Breakdown" not in payload["messageHtml"]
    assert "Empfohlener Push-Titel:" in payload["messageHtml"]
    assert "Titelqualität:" in payload["messageHtml"]
    assert isinstance(payload["whyNow"], list)
    assert isinstance(payload["whyPushworthy"], list)


def test_agent_disabled_message_still_exposes_local_quality_and_decision_basis():
    candidate = _candidate()
    context = _context(candidate, now_ts=_gold_slot_ts())
    config = _config(agent_review_enabled=False)
    decision = shouldNotifyTeams(candidate, context, config)

    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    payload = message["payload"]

    assert payload["type"] == "push_recommendation"
    assert payload["dispatchApproved"] is True
    assert payload["pushTitleReview"]["approved"] is True
    assert payload["decisionBasis"].startswith("Reguläre Vollfreigabe")
    assert "Entscheidungsbasis:" in message["text"]
    assert "Empfehlungsstärke " in message["text"]
    assert "Schätzung, keine Garantie" in message["text"]
    assert "Sicherheit " not in message["text"]
    assert "Reichweitenbasis unsicher" in message["text"]
    review = message["_recommendationReview"]
    dimensions = review["dimensions"]
    assert dimensions["agentConsensus"] is None
    expected_score = round(
        (
            dimensions["pushScore"] * 0.50
            + dimensions["articleStrength"] * 0.10
            + dimensions["orForecast"] * 0.08
            + dimensions["timing"] * 0.09
            + dimensions["title"] * 0.06
            + dimensions["candidateField"] * 0.02
            + dimensions["germanyRelevance"] * 0.05
        )
        / 0.90,
        1,
    )
    assert review["score"] == expected_score


def test_unapproved_preview_cannot_reach_webhook_without_agent_network():
    candidate = _candidate()
    context = _context(candidate)
    config = _config(agent_review_enabled=False)
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert decision["shouldNotify"] is True
    assert message["payload"]["type"] == "push_recommendation_preview"
    assert message["payload"]["dispatchApproved"] is False
    assert message["payload"]["recommendedAction"] == ""
    assert message["text"].startswith("Nicht senden:")

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["blocked"] is True
    urlopen.assert_not_called()


def test_stale_live_history_does_not_block_independent_recommendation():
    candidate = _candidate()
    context = _context(candidate, now_ts=_gold_slot_ts())
    context["historyAuthoritative"] = False
    config = _config(agent_review_enabled=False)
    decision = shouldNotifyTeams(candidate, context, config)

    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["available"] is False
    assert message["payload"]["type"] == "push_recommendation"
    assert message["payload"]["dispatchApproved"] is True
    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = sendTeamsNotification(message, config)
    assert result["ok"] is True
    urlopen.assert_called_once()


def test_teams_message_uses_llm_generated_title_when_available():
    candidate = _candidate(
        title="Regierung beschließt Rentenpaket für Millionen Familien",
        recommendedText="Regierung beschließt Rentenpaket für Millionen Familien",
    )
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    llm_result = {
        "title": "Rentenpaket: Was der Beschluss für Millionen Familien bedeutet",
        "meta": {"llm_call_started": True},
    }
    with (
        patch("push_title_agent._llm_unavailable_reason", return_value=""),
        patch("push_title_agent.generate_push_title", return_value=llm_result),
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
    with (
        patch("push_title_agent._llm_unavailable_reason", return_value=""),
        patch("push_title_agent.generate_push_title", return_value=llm_result),
    ):
        message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert message["payload"]["pushTitleSource"] != "llm"
    assert "Darum geht es jetzt" not in message["payload"]["alternativePushTitle"]


def test_teams_message_rejects_llm_title_with_unsupported_fact():
    candidate = _candidate()
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    llm_result = {
        "title": "Rentenpaket: 500 Euro mehr für alle Beschäftigten",
        "meta": {"llm_call_started": True},
    }
    with (
        patch("push_title_agent._llm_unavailable_reason", return_value=""),
        patch("push_title_agent.generate_push_title", return_value=llm_result),
    ):
        message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert message["payload"]["pushTitleSource"] != "llm"
    assert "500 Euro" not in message["payload"]["recommendedPushText"]
    assert message["payload"]["pushTitleReview"]["approved"] is True


def test_teams_message_does_not_repeat_identical_push_text_and_article_title():
    candidate = _candidate(recommendedText=_candidate()["title"])
    context = _context(candidate, now_ts=_gold_slot_ts())
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert text.startswith("🚨 Jetzt pushen:")
    assert "Empfohlener Push-Titel:" in text
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

    message = buildTeamsPushRecommendation(
        candidate, context, decision, _config(score_only_mode=True)
    )

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

    message = buildTeamsPushRecommendation(
        candidate, context, decision, _config(score_only_mode=True)
    )
    text = message["text"]

    assert "4.77" not in text
    assert "Zeitfenster-Prognose" in text
    assert "4.77" not in message["payload"]["messageHtml"]
    assert message["payload"]["predictedOR"] > 0.0
    assert message["payload"]["predictedORAvailable"] is True
    assert message["payload"]["predictedORSource"] == "historical_slot_baseline"
    assert message["payload"]["minutesSinceLastPush"] == 50.0


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
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        url="https://www.bild.de/politik/send-failure-recorded",
        pubDate=_iso(now_ts - 10 * 60),
    )
    from app.database import push_db_upsert, teams_recommendation_list_recent

    push_db_upsert(_history(now_ts=now_ts))

    with patch(
        "app.notifications.teams.urllib.request.urlopen",
        side_effect=urllib.error.URLError("webhook down"),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is False
    assert result["sendResult"]["ok"] is False
    rows = teams_recommendation_list_recent(limit=5)
    assert rows
    assert rows[0]["article_url"] == candidate["url"]
    assert rows[0]["recommendation_type"] == "teams_alert"
    assert rows[0]["status"] == "failed"
    assert rows[0]["send_status"] == "failed"
    assert rows[0]["send_error"]


def test_send_cycle_continues_when_live_comparison_is_stale(tmp_db):
    now_ts = _gold_slot_ts()
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    from app.database import push_db_upsert

    push_db_upsert(_history(now_ts=now_ts))

    with (
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=False,
        )

    assert result["ok"] is True
    assert result["sent"] is True
    send.assert_called_once()
    assert send.call_args.args[0]["payload"]["livePushComparison"]["available"] is False


@pytest.mark.parametrize(
    ("cache_age_seconds", "expected_authoritative"),
    [(60, True), (600, False)],
)
def test_push_refresh_only_trusts_fresh_relay_cache(
    cache_age_seconds,
    expected_authoritative,
):
    import app.routers.push as push_router

    with (
        patch(
            "app.routers.push._fetch_live_push_snapshot",
            side_effect=RuntimeError("synthetic direct-fetch outage"),
        ),
        patch("app.routers.push._parse_bild_messages", return_value=[]),
        patch("app.routers.push.push_db_upsert", return_value=1),
        patch.dict(
            push_router._push_sync_cache,
            {
                "messages": [{"synthetic": True}],
                "channels": [],
                "ts": time.time() - cache_age_seconds,
            },
            clear=True,
        ),
    ):
        result = push_router._build_refresh_response()

    assert result["source"] == "cache->db"
    assert result["history_authoritative"] is expected_authoritative
    assert result["snapshot_age_seconds"] >= cache_age_seconds


def test_title_jury_blocks_vague_candidate_before_webhook(tmp_db):
    candidate = _candidate(
        url="https://www.bild.de/politik/vague-package",
        title="Eilmeldung: Regierung beschliesst wichtiges Paket",
        recommendedText="Das bedeutet das neue Paket",
    )
    from app.database import push_db_upsert

    push_db_upsert(_history())

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(),
            now_ts=NOW_TS,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is False
    assert result["reason"] == "title_review_blocked"
    assert result["titleReview"]["approved"] is False
    urlopen.assert_not_called()


def test_title_jury_blocks_vague_candidate_when_agent_network_is_disabled(tmp_db):
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        url="https://www.bild.de/politik/vague-package-no-agents",
        title="Eilmeldung: Regierung beschliesst wichtiges Paket",
        recommendedText="Das bedeutet das neue Paket",
        pubDate=_iso(now_ts - 10 * 60),
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(now_ts=now_ts))

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is False
    assert result["reason"] == "title_review_blocked"
    assert result["titleReview"]["approved"] is False
    urlopen.assert_not_called()


def test_final_recommendation_jury_approves_strong_monday_morning_candidate():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["pushesToday"] = 1
    evaluation = evaluate_teams_alert_candidates([candidate], context, config)
    selected = evaluation["decisions"][0]["decision"]

    assert selected["shouldNotify"] is True
    message = buildTeamsPushRecommendation(candidate, context, selected, config)
    quality = message["_recommendationReview"]

    assert quality["enforced"] is True
    assert quality["approved"] is True
    assert quality["score"] >= quality["threshold"]
    assert quality["dimensions"]["timing"] >= 76.0
    assert quality["dimensions"]["pushScore"] == candidate["score"]
    assert "Push-Score:" in message["text"]
    assert "08:51 Uhr" in message["text"]


def test_final_recommendation_jury_fails_closed_below_the_push_score_floor():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60), score=78.4)
    config = _smart_config(agent_review_enabled=False, min_score=75.0)
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["pushesToday"] = 1
    decision = shouldNotifyTeams(candidate, context, config)
    assert decision["shouldNotify"] is True

    weak_candidate = {**candidate, "score": 68.1}
    bypassed_decision = {
        **decision,
        "score": 68.1,
        "minScore": 75.0,
        "shouldNotify": True,
    }
    message = buildTeamsPushRecommendation(
        weak_candidate,
        context,
        bypassed_decision,
        config,
    )
    quality = message["_recommendationReview"]

    assert quality["approved"] is False
    assert quality["dimensions"]["pushScore"] == 68.1
    assert any("harten Freigabeschwelle 75.0" in reason for reason in quality["blockers"])


def test_deadline_fallback_is_labeled_honestly_without_agent_network():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    config = _smart_config(agent_review_enabled=False)
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["pushesToday"] = 1
    decision = shouldNotifyTeams(candidate, context, config)

    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert message["payload"]["dispatchApproved"] is True
    assert message["payload"]["decisionBasis"].startswith("Mindestfenster-Auswahl")
    assert "Mindestfenster-Auswahl" in message["text"]
    assert "harte Fakten-, Aktualitäts-, Titel-, Ruhezeit- und Dublettengates" in message["text"]


def test_final_recommendation_jury_uses_three_minute_window_for_breaking():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 15, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        category="news",
        title="Eilmeldung: Israel und Iran vereinbaren sofortige Feuerpause",
        recommendedText="Israel und Iran: Sofortige Feuerpause vereinbart",
        pubDate=_iso(now_ts - 3 * 60),
        score=94.0,
        predictedOR=0.082,
        isBreaking=True,
        isEilmeldung=True,
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["pushesToday"] = 1
    evaluation = evaluate_teams_alert_candidates([candidate], context, config)
    selected = evaluation["decisions"][0]["decision"]

    assert selected["shouldNotify"] is True
    message = buildTeamsPushRecommendation(candidate, context, selected, config)
    timing = message["_recommendationReview"]["timing"]

    assert timing["mode"] == "breaking_override"
    assert timing["windowMinutes"] == 3
    assert timing["sendByLabel"] == "08:18"
    assert "Sofort senden, ideal bis 08:18 Uhr" in message["text"]


def test_final_recommendation_jury_blocks_before_webhook_and_send_claim(tmp_db):
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        url="https://www.bild.de/politik/final-quality-block",
        pubDate=_iso(now_ts - 10 * 60),
    )
    config = _smart_config(min_recommendation_quality=99.0)
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=60, now_ts=now_ts))

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=config,
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is False
    assert result["reason"] == "recommendation_quality_blocked"
    assert result["recommendationQuality"]["approved"] is False
    urlopen.assert_not_called()


def test_send_cycle_considers_expanded_candidate_beyond_dashboard_top_limit(tmp_db):
    from app.database import push_db_upsert

    now_ts = _gold_slot_ts()
    push_db_upsert(_history(minutes_since_last_push=65, now_ts=now_ts))
    weak = [
        _candidate(
            id=f"weak-{index}",
            url=f"https://www.bild.de/news/weak-{index}",
            title=f"Weicher Kandidat {index}: kein konkreter Push-Anlass",
            score=61.0,
            predictedOR=0.035,
        )
        for index in range(24)
    ]
    strong = _candidate(
        id="rank-25-raid",
        url="https://www.bild.de/news/grossrazzia-leistungsbetrueger-rang-25",
        title="200 Polizisten im Einsatz: Grossrazzia gegen Leistungsbetrueger",
        category="news",
        score=88.0,
        predictedOR=0.094,
        pubDate=_iso(now_ts - 10 * 60),
    )

    with patch(
        "app.notifications.teams.send_teams_notification",
        return_value={"ok": True, "status": 200},
    ):
        result = evaluate_and_send_best_candidate(
            [*weak, strong],
            config=_config(
                dashboard_top_limit=20,
                editorial_top_limit=10,
                candidate_limit=80,
                global_cooldown_minutes=0,
            ),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is True
    assert result["candidateId"] == strong["url"]
    selected = next(
        item["decision"]
        for item in result["evaluation"]["decisions"]
        if item["decision"]["candidateId"] == strong["url"]
    )
    assert selected["dashboardRank"] == 25
    assert selected["expandedFieldCandidate"] is True


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


def _dead_zone_ts() -> int:
    return int(dt.datetime(2026, 6, 19, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())


def _gold_slot_ts() -> int:
    return int(dt.datetime(2026, 6, 19, 21, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())


def test_dynamic_threshold_does_not_drop_when_too_few_pushes_today():
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

    assert lowered["teamsAlertScoreThreshold"] == base["teamsAlertScoreThreshold"]
    assert lowered["minimumPressure"]["thresholdDrop"] == 0.0
    assert "Rueckstand" in lowered["pushBudgetReason"]


def test_dead_zone_waits_when_day_is_not_behind_push_pace():
    ts = _dead_zone_ts()
    candidate = _candidate(
        score=84.0,
        predictedOR=0.061,
        title="Eilmeldung: Regierung beschliesst wichtiges Paket",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=50, now_ts=ts),
        now_ts=ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 2
    context["teamsAlertsToday"] = 11

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=66.0,
            slot_gate_enabled=True,
            target_pushes_per_day=15,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["editorialReview"]["breakdown"]["timeFit"] == 4.0
    assert "historische Totzone" in decision["editorialReview"]["breakdown"]["timeFitLabel"]
    assert any("Totzone" in reason for reason in decision["blockingReasons"])


def test_dead_zone_allows_score_floor_recovery_when_daily_minimum_is_impossible():
    ts = _dead_zone_ts()
    candidate = _candidate(
        score=90.0,
        predictedOR=0.07,
        title="Eilmeldung: Regierung beschliesst wichtiges Paket",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=50, now_ts=ts),
        now_ts=ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 0

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=66.0,
            slot_gate_enabled=True,
            target_pushes_per_day=15,
        ),
    )

    assert decision["shouldNotify"] is True
    assert decision["pushPacing"]["deficit"] >= 1.5
    assert "Rueckstand" in decision["pushPacing"]["label"]
    assert decision["slotGate"]["mode"] == "projected_shortfall_catchup"
    assert decision["slotGate"]["projectedShortfall"] == 5


def test_friday_lunch_recovers_when_waiting_would_make_15_impossible():
    friday_noon = int(dt.datetime(2026, 6, 19, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        score=86.0,
        predictedOR=0.061,
        category="news",
        title="Regierung beschliesst neue Regel fuer Verbraucher",
        url="https://www.bild.de/news/verbraucher-regel-freitag",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=friday_noon),
        now_ts=friday_noon,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 3
    context["teamsAlertsToday"] = 3

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_alert_score=66.0,
            slot_gate_enabled=True,
            target_pushes_per_day=15,
            min_alerts_per_day=15,
        ),
    )

    breakdown = decision["editorialReview"]["breakdown"]
    assert decision["shouldNotify"] is True
    assert breakdown["timeFit"] == 4.0
    assert "historische Totzone" in breakdown["timeFitLabel"]
    assert decision["slotGate"]["mode"] == "projected_shortfall_catchup"
    assert decision["deadlineFallback"]["approved"] is True


def test_thursday_lunch_uses_shortfall_recovery_when_15_is_at_risk():
    thursday_lunch = int(
        dt.datetime(2026, 6, 25, 12, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        score=80.0,
        predictedOR=0.052,
        category="news",
        title="Warnung: Deutsche Bahn meldet bundesweiten Ausfall",
        url="https://www.bild.de/news/bahn-ausfall-mittag",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=180, now_ts=thursday_lunch),
        now_ts=thursday_lunch,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 1
    context["teamsAlertsToday"] = 1

    config = _config(
        dynamic_threshold_enabled=True,
        min_alert_score=74.0,
        min_editorial_score=72.0,
        slot_gate_enabled=True,
        target_pushes_per_day=15,
        min_alerts_per_day=15,
    )
    decision = shouldNotifyTeams(candidate, context, config)

    breakdown = decision["editorialReview"]["breakdown"]
    assert decision["shouldNotify"] is True
    assert decision["pushPacing"]["deficit"] >= 2.0
    assert breakdown["timeFit"] == 4.0
    assert decision["slotGate"]["mode"] == "projected_shortfall_catchup"
    assert decision["slotGate"]["projectedShortfall"] == 4
    assert decision["deadlineFallback"]["approved"] is True
    assert decision["blockingReasons"] == []


def test_lunch_prime_catchup_does_not_lower_score_floor():
    thursday_lunch = int(
        dt.datetime(2026, 6, 25, 12, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        score=66.0,
        predictedOR=0.054,
        category="news",
        title="Warnung: Deutsche Bahn meldet bundesweiten Ausfall",
        url="https://www.bild.de/news/bahn-ausfall-lunch-floor",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=180, now_ts=thursday_lunch),
        now_ts=thursday_lunch,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 1
    context["teamsAlertsToday"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            dynamic_threshold_enabled=True,
            min_score=70.0,
            min_alert_score=66.0,
            min_editorial_score=66.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["pushPacing"]["deficit"] >= 2.0
    assert decision["minimumPressure"]["thresholdDrop"] == 0.0
    assert any("Score zu niedrig" in reason for reason in decision["blockingReasons"])


def test_gold_slot_uses_historical_baseline_in_time_fit():
    ts = _gold_slot_ts()
    candidate = _candidate(
        score=82.0,
        predictedOR=0.058,
        title="Eilmeldung: Regierung beschliesst wichtiges Paket",
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=55, now_ts=ts),
        now_ts=ts,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dynamic_threshold_enabled=True, min_alert_score=66.0),
    )

    breakdown = decision["editorialReview"]["breakdown"]
    assert decision["shouldNotify"] is True
    assert breakdown["timeFit"] >= 8.0
    assert breakdown["slotAvgOR"] >= 7.0
    assert "Pflicht-/Goldfenster" in breakdown["timeFitLabel"]


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
    high_context["teamsAlertsToday"] = 11
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


def test_verified_breaking_is_recommended_immediately_despite_slot_and_teams_cooldown():
    now_ts = int(dt.datetime(2026, 7, 13, 10, 12, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        id="verified-breaking-now",
        url="https://www.bild.de/news/verified-breaking-now",
        title="Eilmeldung: Bundesregierung ordnet sofortige Evakuierung an",
        category="news",
        score=95.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 3 * 60),
        isBreaking=True,
        isEilmeldung=True,
        breakingProvenance="editorial_verified",
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=2, now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=now_ts - 5 * 60,
        teams_alerts_today=4,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is True
    assert decision["isBreaking"] is True
    assert decision["slotGate"]["mode"] == "breaking_override"
    assert not any("Teams-Cooldown aktiv" in reason for reason in decision["blockingReasons"])
    assert any("Sofortprüfung" in reason for reason in decision["reasons"])


def test_verified_breaking_wins_candidate_selection_over_higher_scoring_normal_story():
    now_ts = _gold_slot_ts()
    normal = _candidate(
        id="normal-higher-score",
        url="https://www.bild.de/news/normal-higher-score",
        title="Netzbetreiber melden neue Stromausfaelle in mehreren Grossstaedten",
        category="news",
        score=98.0,
        predictedOR=0.09,
        pubDate=_iso(now_ts - 5 * 60),
    )
    breaking = _candidate(
        id="breaking-selection-priority",
        url="https://www.bild.de/news/breaking-selection-priority",
        title="Eilmeldung: Bundesregierung ordnet sofortige Evakuierung an",
        category="news",
        score=88.0,
        predictedOR=0.075,
        pubDate=_iso(now_ts - 3 * 60),
        isBreaking=True,
        isEilmeldung=True,
        breakingProvenance="editorial_verified",
    )
    config = _config(agent_review_enabled=False)
    context = build_teams_alert_context(
        [normal, breaking],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=5,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )

    result = evaluate_teams_alert_candidates([normal, breaking], context, config)

    assert result["selectedCandidateId"] == breaking["url"]


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
    assert any(
        "Belastbare OR-Prognose erforderlich" in reason for reason in decision["blockingReasons"]
    )


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
        teams_alerts_today=11,
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
        recent_alerts=[],
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
    now_ts = _gold_slot_ts()
    first = _candidate(
        id="article-1",
        url="https://www.bild.de/politik/article-1",
        score=95.0,
        pubDate=_iso(now_ts - 10 * 60),
    )
    second = _candidate(
        id="article-2",
        url="https://www.bild.de/politik/article-2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket",
        score=82.0,
        predictedOR=0.061,
        pubDate=_iso(now_ts - 10 * 60),
    )
    context = build_teams_alert_context(
        [first, second],
        history=_history(now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
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
    strong_slot_ts = int(
        dt.datetime(2027, 1, 15, 21, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    first = _candidate(
        id="u1",
        url="https://www.bild.de/politik/u1",
        score=84.0,
        predictedOR=0.06,
        pubDate=_iso(strong_slot_ts - 10 * 60),
    )
    second = _candidate(
        id="u2",
        url="https://www.bild.de/politik/u2",
        title="Eilmeldung: Regierung beschliesst weiteres Paket heute Mittag",
        score=83.5,
        predictedOR=0.06,
        pubDate=_iso(strong_slot_ts - 10 * 60),
    )
    context = build_teams_alert_context(
        [first, second],
        history=_history(now_ts=strong_slot_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=11,
        recent_alerts=[],
        now_ts=strong_slot_ts,
    )
    context["pushesToday"] = 11

    # Hoher Margin-Schwellenwert + hoher Clear-Buffer erzwingen die Unsicherheits-Pruefung.
    result = evaluate_teams_alert_candidates(
        [first, second],
        context,
        _config(
            min_selection_margin=40.0,
            selection_clear_editorial_buffer=25.0,
            min_editorial_score=70.0,
        ),
    )

    assert result["selectedCandidateId"] is None
    assert result["fieldUncertain"] is True
    decisions = {item["decision"]["candidateId"]: item["decision"] for item in result["decisions"]}
    assert all(not d["shouldNotify"] for d in decisions.values())
    assert any("Feld unsicher" in reason for reason in decisions[first["url"]]["blockingReasons"])


def test_minimum_pacing_chooses_best_candidate_even_when_field_is_close():
    first = _candidate(
        id="mu1", url="https://www.bild.de/politik/mu1", score=84.0, predictedOR=0.06
    )
    second = _candidate(
        id="mu2",
        url="https://www.bild.de/politik/mu2",
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
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates(
        [first, second],
        context,
        _config(
            min_selection_margin=40.0,
            selection_clear_editorial_buffer=25.0,
            min_editorial_score=70.0,
        ),
    )

    assert result["selectedCandidateId"] is not None
    assert result["fieldUncertain"] is False
    selected = next(
        item["decision"]
        for item in result["decisions"]
        if item["decision"]["candidateId"] == result["selectedCandidateId"]
    )
    assert selected["shouldNotify"] is True
    assert selected["minimumPressure"]["active"] is True


def test_minimum_pacing_allows_urgent_public_service_disruption():
    candidate = _candidate(
        id="bahn-service",
        url="https://www.bild.de/leben-wissen/deutsche-bahn-blackout-totalausfall-geld-zurueck",
        title="Nach Deutsche Bahn-Totalausfall: So bekommen Sie ihr Geld zurück!",
        category="news",
        score=78.6,
        predictedOR=0.0515,
    )
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=51),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    decision = shouldNotifyTeams(candidate, context, _config())

    assert decision["shouldNotify"] is True
    assert decision["minimumPressure"]["active"] is True
    assert not any("Service-/Raetsel-/Ratgeber" in reason for reason in decision["blockingReasons"])
    assert any("Push Score" in reason for reason in decision["reasons"])


def test_minimum_pacing_still_blocks_soft_service_without_public_disruption():
    candidate = _candidate(
        id="soft-service",
        url="https://www.bild.de/service/digital/livestream-kaufberater-prueft-prime-days",
        title="Livestream: Der Kaufberater prüft die Prime Days",
        category="digital",
        score=79.0,
        predictedOR=0.052,
    )
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=51),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    decision = shouldNotifyTeams(candidate, context, _config())

    assert decision["shouldNotify"] is False
    assert any(
        "kein konkretes Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"]
    )


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
        recent_alerts=[],
        now_ts=NOW_TS,
    )

    result = evaluate_teams_alert_candidates([strong, weak], context, _config())

    assert result["selectedCandidateId"] == strong["url"]
    assert result["fieldUncertain"] is False


def test_teams_message_is_compact_and_jargon_free():
    candidate = _candidate()
    context = _context(candidate, now_ts=_gold_slot_ts())
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
    # Story- und Timing-Begründung sind getrennt und sofort scanbar.
    story_block = text.split("Warum dieser Push?\n", 1)[1]
    story_first_bullet = story_block.splitlines()[0]
    assert story_first_bullet.startswith("- Deutschland-Relevanz:")
    assert f"- {candidate['performanceDrivers'][0]}" in story_block
    timing_block = text.split("Warum jetzt?\n", 1)[1]
    timing_first_bullet = timing_block.splitlines()[0]
    assert "erreicht historisch" in timing_first_bullet


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


def _daily_plan_candidates(count=18):
    topics = [
        ("politik", "Regierung beschließt neues Rentenpaket für Familien"),
        ("news", "Polizei nimmt Tatverdächtigen nach Angriff fest"),
        ("news", "Bahn meldet Funkstörung im Fernverkehr"),
        ("politik", "Ukraine-Krieg: Versorgungskrise auf der Krim eskaliert"),
        ("regional", "Gericht verurteilt Angeklagten nach Messerattacke"),
        ("wirtschaft", "Autobauer kündigt Stellenabbau in Deutschland an"),
        ("news", "Warnung vor Unwetter in mehreren Bundesländern"),
        ("digital", "Regierung verbietet riskante China-App"),
        ("wirtschaft", "Krankenkassen erhöhen Beiträge ab Juli"),
        ("news", "Flughafenstreik legt Verkehr in Deutschland lahm"),
        ("news", "Explosion in Chemiewerk: Verletzte gemeldet"),
        ("politik", "EU beschließt neue Sanktionen gegen Russland"),
        ("regional", "Polizei findet vermisstes Kind nach großer Suche"),
        ("wirtschaft", "Rente steigt: Was sich für Millionen ändert"),
        ("digital", "Festnahme nach Cyberangriff auf Klinik"),
        ("news", "Urteil im Betrugsprozess gegen Unternehmer gefallen"),
        ("regional", "Hochwasserwarnung: Städte bereiten Evakuierung vor"),
        ("politik", "Bundesregierung stoppt umstrittenes Gesetz"),
    ]
    candidates = []
    for index, (section, title) in enumerate(topics[:count], start=1):
        candidates.append(
            _candidate(
                id=f"daily-{index}",
                url=f"https://www.bild.de/{section}/daily-{index}",
                title=title,
                category=section,
                score=88.0 - index * 0.45,
                predictedOR=0.067 - index * 0.0007,
                pubDate=_iso(NOW_TS - index * 8 * 60),
                recommendedText=title,
                performanceDrivers=[
                    "Aktualität: klare neue Lage",
                    "Nutzwert/Relevanz: breites Publikum betroffen",
                ],
            )
        )
    return candidates


def _daily_plan_context(candidates, *, history=None):
    return build_teams_alert_context(
        candidates,
        history=history if history is not None else _history(minutes_since_last_push=120),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
        config=_config(),
    )


def _smart_slot_decision(
    *,
    hour,
    minute,
    candidate=None,
    pushes_today=1,
    history=None,
    config=None,
):
    now_ts = int(
        dt.datetime(2026, 7, 13, hour, minute, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    article = candidate or _candidate(
        pubDate=_iso(now_ts - 10 * 60),
        title="Regierung beschliesst sofort neue Entlastung fuer Millionen",
        score=84.0,
        predictedOR=0.065,
    )
    article["pubDate"] = _iso(now_ts - 10 * 60)
    smart_config = config or _smart_config()
    actual_history = history or _history(minutes_since_last_push=60, now_ts=now_ts)
    context = build_teams_alert_context(
        [article],
        history=actual_history,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=pushes_today,
        recent_alerts=[],
        now_ts=now_ts,
        config=smart_config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = pushes_today
    return shouldNotifyTeams(article, context, smart_config)


def test_smart_schedule_uses_all_monday_golden_pairs_and_morning_base():
    schedule = build_teams_daily_schedule("2026-07-13", _smart_config())
    labels = [slot["label"] for slot in schedule["slots"]]

    assert schedule["weekday"] == "Montag"
    assert schedule["count"] == 18
    assert {"06:15", "06:45", "07:15", "07:45", "08:15", "08:45"}.issubset(labels)
    assert {"10:45", "11:45", "23:45"}.isdisjoint(labels)
    assert all(slot["required"] is True for slot in schedule["slots"])
    assert "08:15 + 08:45" in {
        opportunity["label"] for opportunity in schedule["doubleOpportunities"]
    }
    assert "Heute bewusst nachrangig" in schedule["messageHtml"]
    assert len(schedule["messageHtml"].encode("utf-8")) < 28_000


def test_smart_schedule_is_truly_weekday_specific():
    monday = build_teams_daily_schedule("2026-07-13", _smart_config())
    tuesday = build_teams_daily_schedule("2026-07-14", _smart_config())
    monday_labels = {slot["label"] for slot in monday["slots"]}
    tuesday_labels = {slot["label"] for slot in tuesday["slots"]}

    assert monday_labels != tuesday_labels
    assert {"06:15", "06:45"}.issubset(monday_labels)
    assert "10:45" not in tuesday_labels
    assert {"06:15", "06:45"}.issubset(tuesday_labels)
    assert "17:45" in tuesday_labels
    assert "23:45" not in tuesday_labels
    assert tuesday["requiredCount"] == 15
    assert tuesday["minimumRecoveryCount"] == 0
    assert tuesday["qualityOpportunityCount"] == 7
    assert tuesday["count"] == 15
    assert tuesday["meetsTargetCoverage"] is True


@pytest.mark.parametrize(
    "date_iso",
    [
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
        "2026-07-18",
        "2026-07-19",
    ],
)
def test_every_weekday_has_15_to_18_binding_runtime_opportunities(date_iso):
    schedule = build_teams_daily_schedule(date_iso, _smart_config())
    required_doubles = [
        item for item in schedule["doubleOpportunities"] if item["requiredForMinimum"]
    ]

    expected_count = 18 if date_iso == "2026-07-13" else 15
    assert schedule["runtimeOpportunityCount"] == expected_count
    assert schedule["requiredCount"] == expected_count
    assert schedule["minimumDoubleCount"] == len(required_doubles)
    assert required_doubles
    assert {"06:15", "06:45"}.issubset({slot["label"] for slot in schedule["slots"]})
    assert schedule["meetsTargetCoverage"] is True


def test_week_plan_can_recommend_15_strong_editorial_events_each_day():
    config = _smart_config()

    for day_number in range(13, 20):
        target_date = dt.date(2026, 7, day_number)
        opportunities = _daily_runtime_opportunities(target_date, config)
        sent = 0
        last_sent_ts = 0

        assert 15 <= len(opportunities) <= 18
        assert all(
            int(current["ts"]) - int(previous["ts"]) >= 30 * 60
            for previous, current in zip(opportunities, opportunities[1:])
        )

        for index, opportunity in enumerate(opportunities):
            now_ts = int(opportunity["ts"]) + 5
            candidate = _candidate(
                id=f"week-simulation-{day_number}-{index}",
                url=f"https://www.bild.de/news/week-simulation-{day_number}-{index}",
                title=(
                    "Bundesregierung beschliesst Soforthilfe fuer Millionen "
                    f"Haushalte Paket {index}"
                ),
                category="news",
                score=92.0,
                predictedOR=0.085,
                pubDate=_iso(now_ts - 5 * 60),
                recommendedText=(
                    "Soforthilfe beschlossen: Das gilt jetzt fuer Millionen " f"Paket {index}"
                ),
            )
            context = build_teams_alert_context(
                [candidate],
                history=[],
                history_authoritative=True,
                alert_state={},
                last_teams_alert_ts=last_sent_ts,
                teams_alerts_today=sent,
                recent_alerts=[],
                now_ts=now_ts,
                config=config,
            )
            context["dashboardRank"] = 1

            decision = shouldNotifyTeams(candidate, context, config)

            assert decision["shouldNotify"] is True, (
                target_date,
                opportunity["label"],
                decision["blockingReasons"],
            )
            message = buildTeamsPushRecommendation(
                candidate,
                context,
                decision,
                config,
            )
            assert message["_pushTitleReview"]["approved"] is True
            assert message["_recommendationReview"]["approved"] is True
            assert message["_dispatchApproved"] is True
            sent += 1
            last_sent_ts = now_ts

        assert sent == len(opportunities)


def test_midday_restart_with_one_alert_can_still_reach_daily_minimum():
    config = _smart_config()
    berlin = ZoneInfo("Europe/Berlin")
    current = dt.datetime(2026, 7, 15, 12, 50, tzinfo=berlin)
    end = dt.datetime(2026, 7, 15, 23, 59, tzinfo=berlin)
    teams_alerts_today = 1
    last_sent_ts = int((current - dt.timedelta(minutes=45)).timestamp())
    send_modes: list[str] = []
    index = 0

    while current <= end and teams_alerts_today < 15:
        now_ts = int(current.timestamp())
        candidate = _candidate(
            id=f"restart-recovery-{index}",
            url=f"https://www.bild.de/news/restart-recovery-{index}",
            title=(
                "Bundesregierung beschliesst Soforthilfe fuer Millionen " f"Haushalte Paket {index}"
            ),
            category="news",
            score=92.0,
            predictedOR=0.085,
            pubDate=_iso(now_ts - 5 * 60),
            recommendedText=(
                "Soforthilfe beschlossen: Das gilt jetzt fuer Millionen " f"Paket {index}"
            ),
        )
        context = build_teams_alert_context(
            [candidate],
            history=[],
            history_authoritative=True,
            alert_state={},
            last_teams_alert_ts=last_sent_ts,
            teams_alerts_today=teams_alerts_today,
            recent_alerts=[],
            now_ts=now_ts,
            config=config,
        )
        context["dashboardRank"] = 1
        decision = shouldNotifyTeams(candidate, context, config)

        if decision["shouldNotify"]:
            teams_alerts_today += 1
            last_sent_ts = now_ts
            send_modes.append(decision["slotGate"]["mode"])

        current += dt.timedelta(minutes=5)
        index += 1

    assert teams_alerts_today == 15
    assert "projected_shortfall_catchup" in send_modes


def test_wednesday_first_binding_deadline_is_due_at_0645_and_releases_candidate():
    now_ts = int(dt.datetime(2026, 7, 15, 6, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=84.0,
        predictedOR=0.065,
        pubDate=_iso(now_ts - 5 * 60),
    )
    config = _smart_config(agent_review_enabled=False)
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is True
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["slotGate"]["slot"]["label"] == "06:45"
    assert decision["slotGate"]["minimumDouble"] is True
    assert decision["slotGate"]["minimumCommitment"] is True
    assert decision["slotGate"]["dueCount"] == 2
    assert decision["slotGate"]["plannedOpportunityCount"] == 15


def test_due_minimum_slot_uses_raw_push_score_over_secondary_model_floors():
    now_ts = int(dt.datetime(2026, 7, 15, 6, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=75.0,
        predictedOR=0.052,
        pubDate=_iso(now_ts - 5 * 60),
    )
    config = _smart_config(
        agent_review_enabled=False,
        min_alert_score=95.0,
        min_editorial_score=95.0,
    )
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert decision["shouldNotify"] is True
    assert decision["deadlineFallback"]["approved"] is True
    assert len(decision["deadlineFallback"]["secondaryCautions"]) == 2
    assert message["payload"]["dispatchApproved"] is True
    assert message["payload"]["recommendationQuality"]["approved"] is True


def test_double_opportunities_report_incompatible_cooldown_configuration():
    schedule = build_teams_daily_schedule(
        "2026-07-14",
        _smart_config(
            global_cooldown_minutes=60,
            min_minutes_since_last_push=60,
        ),
    )

    assert schedule["requiredCount"] == 15
    assert schedule["doubleOpportunities"]
    assert all(not item["cooldownCompatible"] for item in schedule["doubleOpportunities"])
    assert schedule["qualityOpportunityCount"] == 7
    assert schedule["count"] == 15
    assert schedule["meetsTargetCoverage"] is True


def test_tuesday_0645_does_not_recommend_the_isolated_baby_death_story():
    now_ts = int(dt.datetime(2026, 7, 14, 6, 45, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        id="baby-first-aid",
        url="https://www.bild.de/news/inland/baby-erste-hilfe",
        title="Baby stirbt, weil Mutter Erste Hilfe verweigert",
        recommendedText="Mutter verweigert Erste Hilfe - Baby stirbt tragisch",
        category="news",
        score=78.1,
        predictedOR=0.0612,
        pubDate=_iso(now_ts - 20 * 60),
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=686, now_ts=now_ts),
        alert_state={},
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 0

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is False
    assert decision["morningReview"]["approved"] is False
    assert decision["slotGate"]["slot"]["label"] == "06:45"
    assert any("Morgenfit" in reason for reason in decision["blockingReasons"])


def test_tuesday_0746_does_not_recommend_the_isolated_lake_death_story():
    now_ts = int(dt.datetime(2026, 7, 14, 7, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        id="munich-lake-death",
        url="https://www.bild.de/regional/muenchen/see-badegaeste",
        title="Mann ertrinkt in Muenchner See - Badegaeste schauen nur zu",
        recommendedText="Mann ertrinkt in Muenchner See - Badegaeste schauen nur zu",
        category="regional",
        score=65.1,
        predictedOR=0.0614,
        pubDate=_iso(now_ts - 15 * 60),
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=747, now_ts=now_ts),
        alert_state={},
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 0

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is False
    assert decision["morningReview"]["approved"] is False
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["deadlineFallback"]["approved"] is False
    assert any("Morgenfit" in reason for reason in decision["blockingReasons"])


def test_morning_gate_keeps_actionable_major_public_safety_news_eligible():
    now_ts = int(dt.datetime(2026, 7, 14, 7, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Explosion im Chemiewerk: Warnung fuer Anwohner, 6 Tote",
        category="news",
        score=96.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 5 * 60),
    )
    context = _context(
        candidate,
        history=_history(minutes_since_last_push=90, now_ts=now_ts),
        now_ts=now_ts,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 0

    decision = shouldNotifyTeams(candidate, context, _smart_config())

    assert decision["morningReview"]["approved"] is True
    assert any("uebergeordnete Relevanz" in reason for reason in decision["reasons"])


def test_teams_clock_is_always_formatted_in_berlin_time():
    now_ts = int(dt.datetime(2026, 7, 14, 6, 45, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())

    assert _format_time(now_ts) == "06:45"


def test_smart_schedule_carries_the_historically_best_ressort():
    wednesday = build_teams_daily_schedule("2026-07-15", _smart_config())
    morning = next(slot for slot in wednesday["slots"] if slot["label"] == "07:45")

    assert morning["topCategory"] == "geld"
    assert "geld" in morning["preferredSections"]


def test_slot_gate_waits_before_15_and_recovers_a_missed_binding_slot():
    before = _smart_slot_decision(hour=8, minute=10, pushes_today=4)
    missed = _smart_slot_decision(hour=8, minute=30, pushes_today=4)
    after = _smart_slot_decision(hour=8, minute=46, pushes_today=5)

    assert before["shouldNotify"] is False
    assert before["slotGate"]["mode"] == "wait"
    assert any("bis 08:15" in reason for reason in before["blockingReasons"])
    assert missed["shouldNotify"] is True
    assert missed["slotGate"]["mode"] == "deadline_fallback"
    assert after["shouldNotify"] is True
    assert after["slotGate"]["mode"] == "deadline_fallback"
    assert after["slotGate"]["dueCount"] == 6
    assert after["deadlineFallback"]["approved"] is True


def test_slot_gate_uses_teams_count_even_when_actual_push_count_is_already_high():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=84.0,
        predictedOR=0.065,
        pubDate=_iso(now_ts - 10 * 60),
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=5, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 15

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is True
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["slotGate"]["countBasis"] == "teams_alerts"
    assert decision["slotGate"]["currentCount"] == 0
    assert decision["pushesToday"] == 15


def test_deadline_fallback_selects_best_available_but_keeps_absolute_floor():
    best_available = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=75.0,
        predictedOR=0.052,
    )
    too_weak = _candidate(
        id="too-weak",
        url="https://www.bild.de/politik/too-weak",
        title="Das ist heute ebenfalls wichtig",
        score=68.1,
        predictedOR=0.09,
    )

    fallback = _smart_slot_decision(hour=8, minute=46, candidate=best_available)
    rejected = _smart_slot_decision(hour=8, minute=46, candidate=too_weak)

    assert fallback["shouldNotify"] is True
    assert fallback["deadlineFallback"]["approved"] is True
    assert fallback["deadlineFallback"]["remainingBlockers"] == []
    assert rejected["shouldNotify"] is False
    assert any(
        "absolute Untergrenze" in reason
        for reason in rejected["deadlineFallback"]["remainingBlockers"]
    )


def test_deadline_fallback_never_lets_high_or_rescue_a_sub_75_push_score():
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=68.1,
        predictedOR=0.095,
    )
    decision = _smart_slot_decision(
        hour=8,
        minute=46,
        candidate=candidate,
        config=_smart_config(
            min_score=75.0,
            deadline_fallback_min_score=75.0,
            min_alert_score=55.0,
            min_editorial_score=55.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["deadlineFallback"]["approved"] is False
    assert any(
        "Push Score 68.1 < absolute Untergrenze 75.0" in reason
        for reason in decision["deadlineFallback"]["remainingBlockers"]
    )


def test_deadline_fallback_selects_highest_push_score_not_first_feed_item():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    first = _candidate(
        id="fallback-first",
        url="https://www.bild.de/politik/fallback-first",
        title="Regierung beschliesst neue Hilfe fuer Gruppe eins",
        category="news",
        pubDate=_iso(now_ts - 10 * 60),
        score=75.0,
        predictedOR=0.052,
    )
    better = _candidate(
        id="fallback-better",
        url="https://www.bild.de/politik/fallback-better",
        title="Regierung beschliesst neue Hilfe fuer Gruppe zwei",
        category="news",
        pubDate=_iso(now_ts - 10 * 60),
        score=84.0,
        predictedOR=0.058,
    )
    config = _smart_config(min_selection_margin=0)
    context = build_teams_alert_context(
        [first, better],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["pushesToday"] = 1

    result = evaluate_teams_alert_candidates([first, better], context, config)

    assert result["selectedCandidateId"] == better["url"]
    selected = next(
        item["decision"] for item in result["decisions"] if item["candidate"]["id"] == better["id"]
    )
    assert selected["shouldNotify"] is True
    assert selected["expectedVisits"] > 0
    assert selected["deadlineFallback"]["approved"] is True
    assert selected["competition"]["eligibleCompetitors"] == 1
    assert selected["competition"]["selectionMargin"] is None
    assert selected["competition"]["selectionMarginPercent"] is None
    assert selected["competition"]["selectionConfidence"] in {"hoch", "mittel", "niedrig"}


def test_peak_0815_is_a_binding_top_one_decision():
    exceptional = _candidate(
        title="Regierung beschliesst sofort neue Entlastung fuer Millionen",
        score=96.0,
        predictedOR=0.08,
        performanceDrivers=[
            "Aktualitaet: neue Entscheidung",
            "Relevanz: Millionen unmittelbar betroffen",
        ],
    )

    decision = _smart_slot_decision(hour=8, minute=15, candidate=exceptional)

    assert decision["shouldNotify"] is True
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["slotGate"]["slot"]["label"] == "08:15"


def test_shortfall_recovery_does_not_wait_for_an_exact_double_slot_minute():
    early = _smart_slot_decision(hour=20, minute=2, pushes_today=10)
    mid_slot = _smart_slot_decision(hour=20, minute=20, pushes_today=10)
    on_plan = _smart_slot_decision(hour=20, minute=20, pushes_today=13)

    assert early["shouldNotify"] is True
    assert early["slotGate"]["mode"] == "projected_shortfall_catchup"
    assert early["slotGate"]["deficit"] == 2
    assert early["slotGate"]["projectedShortfall"] == 2
    assert mid_slot["shouldNotify"] is True
    assert mid_slot["slotGate"]["mode"] == "deadline_fallback"
    assert mid_slot["deadlineFallback"]["approved"] is True
    assert on_plan["shouldNotify"] is False
    assert on_plan["slotGate"]["projectedShortfall"] == 0
    assert on_plan["slotGate"]["mode"] == "wait"


def test_deadline_fallback_keeps_live_pushed_article_as_independent_comparison():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    pushed_history = _history(
        minutes_since_last_push=60,
        now_ts=now_ts,
        title=candidate["title"],
        headline=candidate["title"],
        link=candidate["url"],
    )

    decision = _smart_slot_decision(
        hour=8,
        minute=46,
        candidate=candidate,
        history=pushed_history,
    )

    assert decision["shouldNotify"] is True
    assert decision["deadlineFallback"]["approved"] is True
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert not any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_context_reads_90_days_for_non_blocking_live_push_comparison():
    now_ts = int(dt.datetime(2026, 7, 13, 18, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    history = [
        {
            "message_id": "old-exact-push",
            "ts_num": now_ts - 40 * 24 * 3600,
            "title": "Fruehere Zeile fuer denselben Artikel",
            "headline": "Fruehere Zeile fuer denselben Artikel",
            "cat": "politik",
            "link": candidate["url"],
        },
        *_history(minutes_since_last_push=60, now_ts=now_ts),
    ]
    config = _smart_config()

    with patch("app.notifications.teams.push_db_load_all", return_value=history) as load_history:
        context = build_teams_alert_context(
            [candidate],
            alert_state={},
            last_teams_alert_ts=0,
            teams_alerts_today=9,
            recent_alerts=[],
            now_ts=now_ts,
            config=config,
            history_authoritative=True,
        )

    context["dashboardRank"] = 1
    context["pushesToday"] = 9
    decision = shouldNotifyTeams(candidate, context, config)

    load_history.assert_called_once_with(max_days=90, max_rows=3000)
    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"] == {
        "available": True,
        "matched": True,
        "matchType": "exact_article",
        "reason": "Bereits live gepusht (gleiche Artikel-URL)",
    }


def test_confirmed_sport_event_can_pass_but_routine_sport_cannot():
    confirmed = _candidate(
        id="sport-transfer",
        url="https://www.bild.de/sport/bayern-transfer",
        title="Bayern bestaetigt: Star wechselt ueberraschend nach England",
        category="sport",
        score=95.0,
        predictedOR=0.08,
    )
    routine = _candidate(
        id="sport-training",
        url="https://www.bild.de/sport/bayern-training",
        title="Bayern-Stars starten heute ins Training",
        category="sport",
        score=95.0,
        predictedOR=0.08,
    )

    confirmed_decision = _smart_slot_decision(
        hour=18,
        minute=46,
        candidate=confirmed,
        pushes_today=9,
    )
    routine_decision = _smart_slot_decision(
        hour=18,
        minute=46,
        candidate=routine,
        pushes_today=9,
    )

    sport_review = confirmed_decision["editorialReview"]["breakdown"]["sportReview"]
    assert confirmed_decision["shouldNotify"] is True
    assert sport_review["eventful"] is True
    assert "bestaetigter Transfer" in sport_review["context"]
    assert routine_decision["shouldNotify"] is False
    assert any(
        "Sport ohne frische bestaetigte" in reason for reason in routine_decision["blockingReasons"]
    )


def test_daily_schedule_is_sent_only_once_per_berlin_day(tmp_db):
    now_ts = int(dt.datetime(2026, 7, 13, 6, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    config = _smart_config(
        daily_schedule_send_enabled=True,
        daily_schedule_send_time="05:45",
    )

    with patch(
        "app.notifications.teams.send_teams_notification",
        return_value={"ok": True, "status": 200},
    ) as send:
        first = send_teams_daily_schedule_if_due(config, now_ts=now_ts)
        second = send_teams_daily_schedule_if_due(config, now_ts=now_ts + 60)

    assert first["sent"] is True
    assert first["count"] == 18
    assert second["sent"] is False
    assert second["reason"] == "already_sent"
    assert send.call_count == 1
    payload = send.call_args.args[0]["payload"]
    assert payload["type"] == "push_daily_schedule"
    assert len(payload["slots"]) == 18


def test_daily_schedule_never_claims_or_sends_during_quiet_hours():
    night_ts = int(dt.datetime(2026, 7, 13, 2, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    config = _smart_config(
        daily_schedule_send_enabled=True,
        daily_schedule_send_time="00:30",
    )

    with (
        patch("app.notifications.teams.teams_daily_schedule_try_claim") as claim,
        patch("app.notifications.teams.send_teams_notification") as send,
    ):
        result = send_teams_daily_schedule_if_due(config, now_ts=night_ts)

    assert result["sent"] is False
    assert result["reason"] == "quiet_hours"
    claim.assert_not_called()
    send.assert_not_called()


def test_daily_push_plan_returns_minimum_15_teams_ready_items():
    candidates = _daily_plan_candidates(18)
    context = _daily_plan_context(candidates)

    plan = buildTeamsDailyPushPlan(
        candidates,
        context,
        _config(),
        target_date="2026-06-24",
        min_items=15,
        max_items=15,
        now_ts=NOW_TS,
    )

    assert plan["count"] == 15
    assert plan["meetsMinimum"] is True
    assert plan["requiredSlotCount"] == 15
    assert plan["qualityOpportunityCount"] == 3
    assert len(plan["items"]) == 15
    quality_chances = [item for item in plan["items"] if item["qualityOnly"]]
    assert quality_chances == []
    assert "Fenster: regulaere :45-Entscheidung" in plan["messageText"]
    assert len(plan["top5"]) == 5
    assert "qualitySummary" in plan
    assert "Tagesplan Pushes für 2026-06-24, Mittwoch" in plan["messageText"]
    assert "Qualität:" in plan["messageText"]
    assert "Top 5 Pushes des Tages" in plan["messageText"]
    assert "Bewusst nicht pushen" in plan["messageText"]
    for item in plan["items"]:
        assert item["pushText"]
        assert item["articleUrl"]
        assert item["priority"] in {"A", "B", "C"}
        assert item["confidence"] in {"hoch", "mittel", "niedrig"}
        assert item["status"] in {"fix", "optional", "nur bei ruhiger Nachrichtenlage"}
        assert 1.0 <= float(item["visitPotential"]) <= 10.0
        assert item["alternativeTime"]


def test_daily_push_plan_excludes_sport_from_teams_plan():
    sport = _candidate(
        id="sport-plan",
        url="https://www.bild.de/sport/top-transfer",
        title="Bayern-Star wechselt überraschend nach England",
        category="sport",
        score=99.0,
        predictedOR=0.09,
    )
    candidates = [sport, *_daily_plan_candidates(18)]
    context = _daily_plan_context(candidates)

    plan = buildTeamsDailyPushPlan(
        candidates,
        context,
        _config(),
        target_date="2026-06-24",
        min_items=15,
        max_items=15,
        now_ts=NOW_TS,
    )

    assert all(item["sectionLabel"] != "Sport" for item in plan["items"])
    assert any(item["section"] == "Sport" for item in plan["notRecommended"])


def test_daily_push_plan_excludes_author_profile_pages():
    author_page = _candidate(
        id="author-profile",
        url="https://www.bild.de/autor/michaela-steuer",
        title="Michaela Steuer",
        category="news",
        score=96.0,
        predictedOR=0.09,
        pubDate=_iso(NOW_TS - 10 * 60),
    )
    candidates = [author_page, *_daily_plan_candidates(18)]
    context = _daily_plan_context(candidates)

    plan = buildTeamsDailyPushPlan(
        candidates,
        context,
        _config(),
        target_date="2026-06-24",
        min_items=15,
        max_items=15,
        now_ts=NOW_TS,
    )

    assert all(item["articleUrl"] != author_page["url"] for item in plan["items"])
    assert any("Autor-/Meta-Seite" in item["reason"] for item in plan["notRecommended"])


def test_daily_push_plan_keeps_live_pushed_article_and_marks_comparison():
    pushed = _candidate(
        id="already-pushed-plan",
        url="https://www.bild.de/news/already-pushed-plan",
        title="Warnung vor Unwetter in mehreren Bundesländern",
        category="news",
        score=96.0,
        predictedOR=0.08,
    )
    candidates = [pushed, *_daily_plan_candidates(18)]
    history = [
        *_history(minutes_since_last_push=120),
        {
            "message_id": "pushed-plan",
            "ts_num": NOW_TS - 6 * 3600,
            "title": pushed["title"],
            "headline": pushed["title"],
            "cat": "news",
            "link": pushed["url"],
        },
    ]
    context = _daily_plan_context(candidates, history=history)

    plan = buildTeamsDailyPushPlan(
        candidates,
        context,
        _config(),
        target_date="2026-06-24",
        min_items=15,
        max_items=15,
        now_ts=NOW_TS,
    )

    item = next(item for item in plan["items"] if item["articleUrl"] == pushed["url"])
    assert item["livePushComparison"]["matched"] is True
    assert item["livePushComparison"]["matchType"] == "exact_article"


def test_daily_push_plan_keeps_only_best_duplicate_topic():
    first = _candidate(
        id="bahn-1",
        url="https://www.bild.de/news/bahn-funkstoerung-fernverkehr",
        title="Bahn meldet Funkstörung im Fernverkehr",
        category="news",
        score=94.0,
        predictedOR=0.075,
    )
    duplicate = _candidate(
        id="bahn-2",
        url="https://www.bild.de/news/funkstoerung-bahn-fernverkehr",
        title="Funkstörung bei der Bahn legt Fernverkehr lahm",
        category="news",
        score=91.0,
        predictedOR=0.071,
    )
    candidates = [first, duplicate, *_daily_plan_candidates(17)]
    context = _daily_plan_context(candidates)

    plan = buildTeamsDailyPushPlan(
        candidates,
        context,
        _config(),
        target_date="2026-06-24",
        min_items=15,
        max_items=15,
        now_ts=NOW_TS,
    )

    planned_urls = {item["articleUrl"] for item in plan["items"]}
    assert first["url"] in planned_urls
    assert duplicate["url"] not in planned_urls
    assert any("Dublette im Tagesplan" in item["reason"] for item in plan["notRecommended"])


def test_daily_push_plan_does_not_mass_generate_llm_titles():
    candidates = _daily_plan_candidates(16)
    context = _daily_plan_context(candidates)

    with patch("app.notifications.teams._llm_push_title") as llm_title:
        llm_title.side_effect = AssertionError("daily plan must not call LLM title generation")
        plan = buildTeamsDailyPushPlan(
            candidates,
            context,
            _config(llm_title_enabled=True),
            target_date="2026-06-24",
            min_items=15,
            max_items=15,
            now_ts=NOW_TS,
        )

    assert plan["count"] == 15
    assert llm_title.call_count == 0


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


def test_regional_kinder_story_is_not_misclassified_as_ki_digital():
    sitemap = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>https://www.bild.de/regional/bochum/ex-trainer-missbraucht-kinder</loc>
    <news:news>
      <news:title>Ex-Fussballtrainer soll Kinder missbraucht haben</news:title>
      <news:publication_date>2026-07-15T12:31:00+02:00</news:publication_date>
    </news:news>
  </url>
</urlset>"""

    article = _extract_sitemap_articles(sitemap, max_items=1)[0]

    assert article["category"] == "regional"


def test_local_agent_network_uses_all_specialists_for_strong_candidate():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    review = decision["agentReview"]
    assert decision["shouldNotify"] is True
    assert len(REVIEWERS) == 17
    assert review["agentCount"] == 17
    assert review["approved"] is True
    assert review["hardVetoCount"] == 0
    assert review["reviewerSetVersion"] == "teams-review-v2"
    assert review["evidenceApprovalCount"] >= review["requiredEvidenceApprovals"]
    assert review["latencyBreached"] is False
    assert review["latencyMs"] < review["latencyBudgetMs"]


def test_agent_network_keeps_deadline_fallback_but_reports_every_caution():
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        score=75.0,
        predictedOR=0.052,
    )

    decision = _smart_slot_decision(hour=8, minute=46, candidate=candidate)
    review = decision["agentReview"]

    assert decision["shouldNotify"] is True
    assert review["approved"] is True
    assert review["hardVetoCount"] == 0
    assert review["cautionCount"] >= 1
    assert review["evidenceApprovalCount"] >= review["requiredEvidenceApprovals"]
    assert review["mainCounterargument"]


def test_agent_network_marks_real_push_match_without_veto_at_deadline():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(pubDate=_iso(now_ts - 10 * 60))
    history = _history(
        minutes_since_last_push=60,
        now_ts=now_ts,
        title=candidate["title"],
        headline=candidate["title"],
        link=candidate["url"],
    )

    decision = _smart_slot_decision(
        hour=8,
        minute=46,
        candidate=candidate,
        history=history,
    )
    review = decision["agentReview"]

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["matched"] is True
    assert review["approved"] is True
    assert review["hardVetoCount"] == 0
    assert any(
        item["agent"] == "Live-Push-Vergleich"
        and item["verdict"] == "approve"
        and not item["hardVeto"]
        for item in review["verdicts"]
    )


def test_agent_network_hard_vetoes_routine_sport_without_event():
    candidate = _candidate(
        id="sport-routine-review",
        url="https://www.bild.de/sport/training-review",
        title="Bayern-Stars starten heute ins Training",
        category="sport",
        score=95.0,
        predictedOR=0.08,
    )

    decision = _smart_slot_decision(
        hour=18,
        minute=46,
        candidate=candidate,
        pushes_today=10,
    )

    assert decision["shouldNotify"] is False
    assert any(
        item["agent"] == "Sport-Ereignis" and item["hardVeto"]
        for item in decision["agentReview"]["verdicts"]
    )


def test_agent_network_continues_when_live_push_comparison_cannot_load():
    candidate = _candidate(score=92.0, predictedOR=0.075)
    config = _config()
    with patch(
        "app.notifications.teams.push_db_load_all",
        side_effect=RuntimeError("synthetic history outage"),
    ):
        context = build_teams_alert_context(
            [candidate],
            alert_state={},
            last_teams_alert_ts=0,
            teams_alerts_today=0,
            recent_alerts=[],
            now_ts=NOW_TS,
            config=config,
        )
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is True
    assert decision["livePushComparison"]["available"] is False
    assert decision["agentReview"]["approved"] is True
    assert any(
        item["agent"] == "Kontext-Integritaet"
        and item["verdict"] == "caution"
        and not item["hardVeto"]
        for item in decision["agentReview"]["verdicts"]
    )


def test_push_dispatch_rejects_missing_agent_approval_before_webhook():
    message = {
        "_agentReview": {"approved": False},
        "payload": {
            "type": "push_recommendation",
            "messageHtml": "<p>Synthetic recommendation</p>",
        },
    }

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = sendTeamsNotification(message, _config())

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "approval" in result["error"].lower()
    urlopen.assert_not_called()


@pytest.mark.parametrize("agent_review_enabled", [True, False])
def test_dispatch_rechecks_real_push_history_as_non_blocking_comparison(
    tmp_db,
    agent_review_enabled,
):
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        id="dispatch-race",
        url="https://www.bild.de/news/dispatch-race",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    initial_history = _history(minutes_since_last_push=60, now_ts=now_ts)
    newly_pushed = [
        *initial_history,
        {
            "message_id": "new-real-push",
            "ts_num": now_ts - 50 * 60,
            "title": candidate["title"],
            "headline": candidate["title"],
            "cat": "news",
            "link": candidate["url"],
        },
    ]

    with (
        patch(
            "app.notifications.teams.push_db_load_all",
            side_effect=[initial_history, newly_pushed],
        ),
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(agent_review_enabled=agent_review_enabled),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["sent"] is True
    send.assert_called_once()
    sent_message = send.call_args.args[0]
    assert sent_message["payload"]["livePushComparison"] == {
        "available": True,
        "matched": True,
        "matchType": "exact_article",
    }


def test_dispatch_ignores_recent_unrelated_live_push_for_recommendation_timing(tmp_db):
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        id="independent-dispatch",
        url="https://www.bild.de/news/independent-dispatch",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    recent_unrelated_push = _history(minutes_since_last_push=5, now_ts=now_ts)

    with (
        patch(
            "app.notifications.teams.push_db_load_all",
            side_effect=[recent_unrelated_push, recent_unrelated_push],
        ),
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["sent"] is True
    send.assert_called_once()


def test_empty_cycle_reports_aggregate_minimum_pacing_diagnostics(tmp_db):
    now_ts = int(dt.datetime(2026, 7, 15, 6, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    weak = _candidate(
        id="diagnostic-weak",
        url="https://www.bild.de/news/diagnostic-weak",
        title="Regierung diskutiert heute allgemein ueber neue Plaene",
        category="news",
        score=60.0,
        predictedOR=0.04,
        pubDate=_iso(now_ts - 5 * 60),
    )

    result = evaluate_and_send_best_candidate(
        [weak],
        config=_smart_config(agent_review_enabled=False),
        now_ts=now_ts,
        history_authoritative=False,
    )

    diagnostics = result["diagnostics"]
    assert result["sent"] is False
    assert result["reason"] == "no_candidate"
    assert diagnostics["plannedOpportunityCount"] == 15
    assert diagnostics["dueOpportunityCount"] == 2
    assert diagnostics["teamsAlertsToday"] == 0
    assert diagnostics["scoreEligibleCandidates"] == 0
    assert diagnostics["projectedShortfall"] == 2
    assert diagnostics["blockerCategories"]["score"] == 1


def test_teams_payload_shows_consensus_and_counterargument_without_raw_context():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    payload = message["payload"]

    review = message["_agentReview"]
    assert review["approved"] is True
    assert f"{review['agentCount']} lokale Checks" in payload["messageText"]
    assert (
        f"Evidenz {review['evidenceApprovalCount']}/{review['evidenceReviewerCount']}"
        in payload["messageText"]
    )
    assert "Prüfstatus:" in payload["messageHtml"]
    assert "agentReview" not in payload
    assert "verdicts" in review
    assert all("history" not in item for item in review["verdicts"])
    assert all("url" not in item for item in review["verdicts"])


def test_agent_network_is_deterministic_and_stays_inside_ten_milliseconds():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    context = _context(candidate)
    config = _config(agent_review_max_latency_ms=10)

    reviews = [shouldNotifyTeams(candidate, context, config)["agentReview"] for _ in range(50)]

    expected_verdicts = reviews[0]["verdicts"]
    assert all(review["verdicts"] == expected_verdicts for review in reviews)
    assert all(review["latencyBreached"] is False for review in reviews)
    assert max(review["latencyMs"] for review in reviews) < 10


def test_agent_failure_is_fail_closed_and_does_not_reach_teams():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    context = _context(candidate)

    def broken_reviewer(_snapshot):
        raise RuntimeError("synthetic reviewer failure")

    with patch("app.notifications.teams_review.REVIEWERS", (broken_reviewer,)):
        decision = shouldNotifyTeams(
            candidate,
            context,
            _config(agent_review_min_evidence_approvals=1),
        )

    assert decision["shouldNotify"] is False
    assert decision["agentReview"]["hardVetoCount"] == 1
    assert "Prueferfehler" in decision["agentReview"]["blockingReason"]


def test_agent_latency_overrun_is_fail_closed():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    context = _context(candidate)

    def slow_context_reviewer(snapshot):
        time.sleep(0.005)
        return REVIEWERS[0](snapshot)

    with patch(
        "app.notifications.teams_review.REVIEWERS",
        (slow_context_reviewer, *REVIEWERS[1:]),
    ):
        decision = shouldNotifyTeams(
            candidate,
            context,
            _config(agent_review_max_latency_ms=1),
        )

    assert decision["shouldNotify"] is False
    assert decision["agentReview"]["latencyBreached"] is True
    assert any(
        item["agent"] == "Pruef-Latenz" and item["hardVeto"]
        for item in decision["agentReview"]["verdicts"]
    )


@pytest.mark.parametrize("marketing_word", ["LIVE", "EXKLUSIV", "SCHOCK", "WARNUNG"])
def test_marketing_words_do_not_create_breaking_privileges(marketing_word):
    candidate = _candidate(
        title=f"{marketing_word}: Das muessen Fans jetzt wissen",
        isBreaking=True,
        isEilmeldung=False,
    )

    assert _is_breaking(candidate) is False


@pytest.mark.parametrize(
    ("publication_fields", "expected_status"),
    [
        ({"pubDate": ""}, "missing"),
        ({"pubDate": "not-a-date"}, "invalid"),
        ({"pubDate": _iso(NOW_TS + 10 * 60)}, "future"),
        ({"pubDate": _iso(NOW_TS - 25 * 3600)}, "stale"),
    ],
)
def test_publication_time_is_an_absolute_agent_gate(publication_fields, expected_status):
    candidate = _candidate(**publication_fields, score=96.0, predictedOR=0.09)
    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is False
    assert decision["publicationReview"]["status"] == expected_status
    assert any(
        item["agent"] == "Aktualitaet" and item["hardVeto"]
        for item in decision["agentReview"]["verdicts"]
    )


def test_deadline_cannot_waive_missing_publication_time():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 46, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        title="Regierung beschliesst neue Soforthilfe fuer Millionen",
        category="news",
        pubDate="",
        score=84.0,
        predictedOR=0.065,
    )
    config = _smart_config()
    context = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60, now_ts=now_ts),
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=1,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )
    context["dashboardRank"] = 1
    context["pushesToday"] = 1

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["shouldNotify"] is False
    assert any(
        item["agent"] == "Aktualitaet" and item["hardVeto"]
        for item in decision["agentReview"]["verdicts"]
    )


def test_speculation_requires_structured_confirmation_signal():
    unconfirmed = _candidate(
        title="Minister soll wohl noch heute zuruecktreten",
        score=93.0,
        predictedOR=0.08,
    )
    confirmed = dict(unconfirmed, id="confirmed", url="https://www.bild.de/politik/confirmed")
    confirmed["confirmationStatus"] = "confirmed"

    blocked = shouldNotifyTeams(unconfirmed, _context(unconfirmed), _config())
    approved = shouldNotifyTeams(confirmed, _context(confirmed), _config())

    assert blocked["shouldNotify"] is False
    assert any(
        item["agent"] == "Faktenrisiko" and item["hardVeto"]
        for item in blocked["agentReview"]["verdicts"]
    )
    assert not any(
        item["agent"] == "Faktenrisiko" and item["hardVeto"]
        for item in approved["agentReview"]["verdicts"]
    )


def test_sport_state_machine_separates_prematch_live_final_and_transfer():
    now_ts = int(dt.datetime(2026, 7, 13, 20, 10, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    cases = {
        "prematch": _candidate(
            title="Anpfiff um 20:30 Uhr: Bayern gegen Dortmund",
            category="sport",
            eventUpdatedAt=_iso(now_ts - 2 * 60),
        ),
        "live": _candidate(
            title="LIVE: Tor! Bayern fuehrt jetzt 2:1",
            category="sport",
            eventUpdatedAt=_iso(now_ts - 5 * 60),
        ),
        "final": _candidate(
            title="Bayern gewinnt 2:1 nach dramatischem Schlusspfiff",
            category="sport",
            eventUpdatedAt=_iso(now_ts - 30 * 60),
        ),
        "transfer": _candidate(
            title="Bayern bestaetigt: Star wechselt nach England",
            category="sport",
            eventUpdatedAt=_iso(now_ts - 120 * 60),
        ),
        "stale_live": _candidate(
            title="LIVE: Tor! Bayern fuehrt jetzt 2:1",
            category="sport",
            eventUpdatedAt=_iso(now_ts - 11 * 60),
        ),
    }
    reviews = {
        key: _sport_candidate_review(item["title"], now_ts, item) for key, item in cases.items()
    }

    assert reviews["prematch"]["state"] == "PREMATCH"
    assert reviews["prematch"]["eventful"] is False
    assert reviews["live"]["state"] == "LIVE_MATERIAL"
    assert reviews["live"]["bypassSlotWait"] is True
    assert reviews["final"]["state"] == "FINAL"
    assert reviews["final"]["bypassSlotWait"] is True
    assert reviews["transfer"]["state"] == "TRANSFER"
    assert reviews["transfer"]["eventful"] is True
    assert reviews["stale_live"]["eventful"] is False


def test_live_push_pause_is_ignored_but_teams_cooldown_uses_the_45_minute_edge():
    candidate = _candidate(score=92.0, predictedOR=0.075)
    config = _config(
        min_minutes_since_last_push=45,
        global_cooldown_minutes=45,
        breaking_min_minutes_since_last_push=45,
    )

    actual_44 = _context(candidate, history=_history(minutes_since_last_push=44))
    actual_45 = _context(candidate, history=_history(minutes_since_last_push=45))
    teams_44 = build_teams_alert_context(
        [candidate],
        history=_history(minutes_since_last_push=60),
        alert_state={},
        last_teams_alert_ts=NOW_TS - 44 * 60,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
        config=config,
    )
    teams_45 = dict(teams_44, lastTeamsAlertTs=NOW_TS - 45 * 60)

    assert shouldNotifyTeams(candidate, actual_44, config)["shouldNotify"] is True
    assert shouldNotifyTeams(candidate, actual_45, config)["shouldNotify"] is True
    assert shouldNotifyTeams(candidate, teams_44, config)["shouldNotify"] is False
    assert shouldNotifyTeams(candidate, teams_45, config)["shouldNotify"] is True


def test_runtime_double_slots_equal_the_advertised_monday_slots():
    schedule = build_teams_daily_schedule("2026-07-13", _smart_config())
    advertised = {int(item["hour"]) for item in schedule["doubleOpportunities"]}
    unadvertised = _smart_slot_decision(hour=17, minute=2, pushes_today=10)
    advertised_catchup = _smart_slot_decision(hour=20, minute=2, pushes_today=10)

    assert advertised == {6, 7, 8, 9, 18, 19, 20, 21, 22}
    assert unadvertised["slotGate"]["doubleOpportunity"] is False
    assert unadvertised["shouldNotify"] is False
    assert advertised_catchup["slotGate"]["doubleOpportunity"] is True
    assert advertised_catchup["slotGate"]["mode"] == "projected_shortfall_catchup"


def test_internal_score_mode_never_waives_a_missing_api_score_at_deadline():
    decision = _smart_slot_decision(
        hour=6,
        minute=16,
        pushes_today=0,
        config=_smart_config(
            require_internal_score_api=True,
            agent_review_enabled=False,
        ),
    )

    assert decision["shouldNotify"] is False
    assert decision["deadlineFallback"]["approved"] is False
    assert any(
        "Kein gueltiger interner Push-Balancer-Score" in reason
        for reason in decision["blockingReasons"]
    )


def test_sender_rejects_local_score_payload_when_internal_api_mode_is_required():
    candidate = _candidate(score=92.0, predictedOR=0.075)
    build_config = _config(agent_review_enabled=False)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, build_config)
    message = buildTeamsPushRecommendation(candidate, context, decision, build_config)
    message["_dispatchApproved"] = True
    message["_teamsDedupApproved"] = True
    message["payload"]["type"] = "push_recommendation"
    message["payload"]["dispatchApproved"] = True

    with patch("app.notifications.teams.urllib.request.urlopen") as post:
        result = sendTeamsNotification(
            message,
            _config(
                agent_review_enabled=False,
                require_internal_score_api=True,
            ),
        )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] == "Canonical internal Push Balancer score is missing"
    post.assert_not_called()


def test_large_agent_field_stays_fast_with_long_real_push_history():
    candidates = [
        _candidate(
            id=f"perf-{index}",
            url=f"https://www.bild.de/news/perf-{index}",
            title=f"Regierung beschliesst Hilfspaket Nummer {index} fuer Region {index}",
            category="news",
            score=90.0 - index * 0.05,
            predictedOR=0.07 - index * 0.0001,
        )
        for index in range(80)
    ]
    history = [
        {
            "message_id": f"history-{index}",
            "ts_num": NOW_TS - 10 * 86400 - index,
            "title": f"Historische Meldung Nummer {index}",
            "headline": f"Historische Meldung Nummer {index}",
            "cat": "news",
            "link": f"https://www.bild.de/news/history-perf-{index}",
            "total_recipients": 250000,
        }
        for index in range(3000)
    ]
    config = _smart_config(slot_gate_enabled=False, min_selection_margin=0)

    started = time.perf_counter()
    context = build_teams_alert_context(
        candidates,
        history=history,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
        config=config,
    )
    result = evaluate_teams_alert_candidates(candidates, context, config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    reviewer_ms = sum(
        float(item["decision"]["agentReview"]["latencyMs"]) for item in result["decisions"]
    )

    assert len(result["decisions"]) == 80
    assert reviewer_ms < 20
    assert elapsed_ms < 250
