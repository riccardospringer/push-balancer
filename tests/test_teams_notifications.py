import datetime as dt
import json
import logging
import time
import urllib.error
from dataclasses import replace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from app.notifications.teams import (
    TeamsAlertConfig,
    buildTeamsDailyPushPlan,
    buildTeamsPushRecommendation,
    build_teams_alert_context,
    build_teams_daily_schedule,
    _has_news_event,
    _llm_slot_fit_review,
    _maybe_send_heartbeat,
    _realert_blocker_or_reason,
    _daily_plan_already_covered,
    _HEARTBEAT_ALERT_REASON,
    evaluate_and_send_best_candidate,
    evaluate_teams_alert_candidates,
    _daily_runtime_opportunities,
    _format_time,
    _is_breaking,
    _sport_candidate_review,
    normalize_predicted_or,
    selectTeamsPushRecommendation,
    send_teams_daily_schedule_if_due,
    send_teams_test_notification,
    sendTeamsNotification,
    shouldNotifyTeams,
)
from app.notifications.teams_review import REVIEWERS
from app.routers.feed import _extract_sitemap_articles


NOW_TS = 1_800_000_000


@pytest.fixture(autouse=True)
def _reset_process_local_teams_send_memory():
    from app.notifications import teams as teams_module

    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()
    yield
    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()


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
        "target_pushes_per_day": 11,
        "min_alerts_per_day": 11,
        "max_alerts_per_day": 15,
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


def test_recent_live_push_blocks_recommendation_until_min_pause():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=_history(minutes_since_last_push=12)),
        _config(),
    )

    assert decision["shouldNotify"] is False
    assert decision["pacingBasis"] == "actual_pushes"
    assert decision["recommendationsIndependentFromLivePushes"] is False
    assert any("Pause seit letztem Push" in reason for reason in decision["blockingReasons"])


def test_live_push_pause_satisfied_allows_recommendation():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=_history(minutes_since_last_push=50)),
        _config(),
    )

    assert decision["shouldNotify"] is True
    assert decision["pacingBasis"] == "actual_pushes"
    assert not any("Pause seit letztem Push" in reason for reason in decision["blockingReasons"])


def test_missing_last_live_push_timestamp_blocks_live_aware_channel():
    candidate = _candidate()

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(min_alerts_per_day=0),
    )

    assert decision["shouldNotify"] is False
    assert decision["recommendationsIndependentFromLivePushes"] is False
    assert any("Letzter Push-Zeitpunkt" in reason for reason in decision["blockingReasons"])


def test_mandatory_quiet_hours_block_sends_between_23_and_6():
    from app.notifications.teams import _quiet_hours_reason

    config = _config()
    # Aktives Fenster 06:00-23:00; Ruhezeit 23:00-06:00.
    for hour, minute, expected in (
        (22, 59, False),
        (23, 0, True),
        (23, 30, True),
        (2, 0, True),
        (5, 59, True),
        (6, 0, False),
        (12, 0, False),
    ):
        ts = int(
            dt.datetime(2026, 6, 24, hour, minute, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        )
        reason = _quiet_hours_reason(ts, config)
        assert bool(reason) is expected, (hour, minute, reason)
        if reason:
            assert "23:00 bis 06:00" in reason


def test_sport_deficit_prefers_near_equal_sport_candidate():
    from app.notifications.teams import _sport_balance_review

    news = _candidate(
        id="news-top",
        url="https://www.bild.de/news/news-top",
        title="Regierung beschliesst neue Entlastung fuer Millionen Haushalte",
        category="news",
        score=86.0,
        predictedOR=0.06,
    )
    sport = _candidate(
        id="sport-close",
        url="https://www.bild.de/sport/bayern-sieg",
        title="Bayern gewinnt Spitzenspiel gegen Leverkusen mit 3:1",
        category="sport",
        score=84.5,
        predictedOR=0.06,
    )
    # 6 Live-Pushes heute, keiner Sport -> Sportanteil klar unter dem Korridor.
    history = [
        dict(
            _history(now_ts=NOW_TS)[0],
            message_id=f"lp-{index}",
            ts_num=NOW_TS - (60 + index * 40) * 60,
        )
        for index in range(6)
    ]
    config = _config(
        require_internal_score_api=True,
        excluded_sections=(),
        min_alert_score=50.0,
        min_editorial_score=50.0,
        min_or=4.0,
    )
    context = build_teams_alert_context(
        [news, sport],
        history=history,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
        config=config,
    )

    balance = _sport_balance_review(context, config)
    assert balance["sportStatus"] == "unter"
    assert balance["selectionPreference"] == "sport"

    for candidate in (news, sport):
        candidate["scoreSource"] = "internal_score_api"

    with patch(
        "app.notifications.teams._sport_candidate_review",
        return_value={
            "eventful": True,
            "state": "final_result",
            "bypassSlotWait": False,
            "label": "Endergebnis bestaetigt",
            "timingDelta": 1.0,
        },
    ):
        result = evaluate_teams_alert_candidates([news, sport], context, config)

    # Sport liegt nur 1,5 Punkte hinter dem News-Kandidaten -> innerhalb der
    # Praeferenz-Bandbreite gewinnt Sport bei Unterdeckung.
    if result["selectedCandidateId"] is not None:
        assert result["sportBalance"]["selectionPreference"] == "sport"
        if result["sportPreferenceApplied"]:
            assert result["selectedCandidateId"] == sport["url"]


def test_sport_preference_never_overrides_clearly_stronger_news_push():
    news = _candidate(
        id="news-strong",
        url="https://www.bild.de/news/news-strong",
        title="Regierung beschliesst neue Entlastung fuer Millionen Haushalte",
        category="news",
        score=92.0,
        predictedOR=0.07,
        scoreSource="internal_score_api",
    )
    weak_sport = _candidate(
        id="sport-weak",
        url="https://www.bild.de/sport/regionalliga",
        title="Regionalliga: Aufsteiger holt spaeten Punkt im Kellerduell",
        category="sport",
        score=78.0,
        predictedOR=0.05,
        scoreSource="internal_score_api",
    )
    history = [
        dict(
            _history(now_ts=NOW_TS)[0],
            message_id=f"lp-strong-{index}",
            ts_num=NOW_TS - (60 + index * 40) * 60,
        )
        for index in range(6)
    ]
    config = _config(
        require_internal_score_api=True,
        excluded_sections=(),
        min_alert_score=50.0,
        min_editorial_score=50.0,
        min_or=4.0,
    )
    context = build_teams_alert_context(
        [news, weak_sport],
        history=history,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
        config=config,
    )

    result = evaluate_teams_alert_candidates([news, weak_sport], context, config)

    # 14 Punkte Abstand liegen weit ausserhalb der Bandbreite: der deutlich
    # staerkere News-Push wird nie von einem schwachen Sport-Push verdraengt.
    assert result["sportPreferenceApplied"] is False
    if result["selectedCandidateId"] is not None:
        assert result["selectedCandidateId"] == news["url"]


def test_daily_maximum_of_sent_live_pushes_blocks_further_recommendations():
    candidate = _candidate()
    context = _context(
        candidate,
        history=[
            dict(
                _history(now_ts=NOW_TS)[0],
                message_id=f"push-{index}",
                ts_num=NOW_TS - (40 + index) * 60,
            )
            for index in range(17)
        ],
    )

    decision = shouldNotifyTeams(candidate, context, _config(max_alerts_per_day=17))

    assert decision["shouldNotify"] is False
    assert any("Tagesmaximum erreicht" in reason for reason in decision["blockingReasons"])


def test_bad_forecast_does_not_trigger_teams_decision():
    candidate = _candidate(predictedOR=0.039)
    context = _context(candidate, teams_alerts_today=11)
    context["pushesToday"] = 11

    decision = shouldNotifyTeams(candidate, context, _config())

    assert decision["shouldNotify"] is False
    assert any("Prognose zu niedrig" in reason for reason in decision["blockingReasons"])


def test_push_score_over_80_overrides_bad_article_forecast():
    candidate = _candidate(score=88.0, predictedOR=0.0377)
    context = _context(candidate, history=_history(minutes_since_last_push=90))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(score_only_mode=True, min_alert_score=40.0, min_editorial_score=40.0),
    )

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Prognose zu niedrig" in reason
        for reason in decision["highScoreOverride"]["waivedBlockers"]
    )


def test_push_score_over_80_can_use_historical_slot_forecast():
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

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Artikel-Prognose fehlt" in reason
        for reason in decision["highScoreOverride"]["waivedBlockers"]
    )


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


def test_minimum_pacing_uses_actual_live_push_count():
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

    assert decision["minimumPressure"]["basis"] == "actual_pushes"
    assert decision["minimumPressure"]["current"] == 5
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
    assert decision["minimumPressure"]["basis"] == "actual_pushes"
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
    assert decision["minimumPressure"]["basis"] == "actual_pushes"
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
    # Unterhalb der High-Score-Schwelle (80) bleibt das Erklärstück-Gate hart.
    # Oberhalb darf der kanonische Push-Balancer-Score die reine FORMAT-
    # Einstufung überstimmen (siehe test_high_score_waives_format_gates).
    candidate = _candidate(
        score=79.5,
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


def test_score_only_mode_still_requires_live_push_timestamp():
    candidate = _candidate(score=82.0, predictedOR=None)

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=[]),
        _config(score_only_mode=True, min_alerts_per_day=0),
    )

    assert decision["shouldNotify"] is False
    assert decision["scoreOnlyMode"] is True
    assert decision["recommendationsIndependentFromLivePushes"] is False
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


def test_push_score_over_80_overrides_dashboard_rank_gate():
    candidate = _candidate(score=92.0, predictedOR=0.07)
    context = _context(candidate)
    context["dashboardRank"] = 25

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20),
    )

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Nicht im oberen Push-Balancer-Feld" in reason
        for reason in decision["highScoreOverride"]["waivedBlockers"]
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
    # Score unterhalb der High-Score-Schwelle: das Ereignis-Gate bleibt hart.
    candidate = _candidate(
        id="soft-app",
        url="https://www.bild.de/digital/sprachlern-app",
        title="Schock fuer Fans: Beliebte Sprachlern-App vor dem Aus",
        category="digital",
        score=79.5,
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
    assert decision["highScoreOverride"]["approved"] is False
    assert any("Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"])


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


def test_push_score_over_80_overrides_editorial_rank_gate():
    candidate = _candidate(score=96.0, predictedOR=0.08)
    context = _context(candidate)
    context["dashboardRank"] = 11

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(dashboard_top_limit=20, editorial_top_limit=10),
    )

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Top 10" in reason for reason in decision["highScoreOverride"]["waivedBlockers"]
    )


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


def test_push_score_at_or_below_80_does_not_override_missing_news_event():
    # Bis einschliesslich 80 bleibt das Ereignis-Gate hart; erst strikt darüber
    # überstimmt der kanonische Score die FORMAT-Einstufung (neue Policy).
    candidate = _candidate(
        score=80.0,
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
    assert decision["highScoreOverride"]["approved"] is False
    assert any("Nachrichten-Ereignis" in reason for reason in decision["blockingReasons"])


def test_high_score_waives_format_gates_but_never_quiz():
    """Neue Policy: kanonischer Score > 80 überstimmt die reinen CvD-FORMAT-
    Einstufungen (Erklärstück / kein konkretes Ereignis) — hochbewertete
    Verbraucher-News sind pushwürdig. Rätsel-/Ratgeber-/Gewinnspiel-Formate
    bleiben auch bei Höchstscore hart geblockt."""
    import app.notifications.teams as _t

    cfg = _t.TeamsAlertConfig()
    waived = _t._high_score_override_review(
        [
            "CvD: Erklär-/Debattenstück ohne neue aktuelle Lage",
            "CvD: kein konkretes Nachrichten-Ereignis erkennbar (Service/Teaser)",
        ],
        score=83.2,
        score_source="internal_score_api",
        config=cfg,
    )
    assert waived["approved"] is True
    assert len(waived["waivedBlockers"]) == 2

    quiz = _t._high_score_override_review(
        ["CvD: Service-/Raetsel-/Ratgeber-Format, nicht pushwuerdig"],
        score=95.0,
        score_source="internal_score_api",
        config=cfg,
    )
    assert quiz["approved"] is False
    assert quiz["hardBlockers"]


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
    assert decision["recommendationsIndependentFromLivePushes"] is False


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


def test_post_send_score_threshold_peaks_then_decays_to_baseline():
    candidate = _candidate(score=79.0)
    config = _config(agent_review_enabled=False, min_score=75.0)
    cases = (
        (30, 80.0, "peak", False),
        (45, 78.75, "decay", True),
        (60, 77.5, "decay", True),
        (75, 76.25, "decay", True),
        (90, 75.0, "baseline", True),
    )

    for minutes, threshold, phase, should_notify in cases:
        context = _context(candidate)
        context["lastTeamsAlertTs"] = NOW_TS - minutes * 60

        decision = shouldNotifyTeams(candidate, context, config)

        assert decision["minScore"] == threshold
        assert decision["postSendScoreThreshold"]["phase"] == phase
        assert decision["shouldNotify"] is should_notify


def test_post_send_threshold_uses_teams_send_while_live_pause_blocks():
    candidate = _candidate(score=76.0)
    context = _context(candidate, history=_history(minutes_since_last_push=2))

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(agent_review_enabled=False, min_score=75.0),
    )

    # Die adaptive Schwelle haengt weiter am letzten Teams-Hinweis, aber der
    # frische Live-Push blockiert die Empfehlung ueber den Mindestabstand.
    assert decision["minScore"] == 75.0
    assert decision["postSendScoreThreshold"]["phase"] == "baseline"
    assert decision["shouldNotify"] is False
    assert any("Pause seit letztem Push" in reason for reason in decision["blockingReasons"])


def test_score_exactly_80_does_not_activate_high_score_override():
    candidate = _candidate(
        score=80.0,
        scoreSource="internal_score_api",
        predictedOR=0.035,
        title="Bundesregierung beschliesst neue Entlastung fuer Millionen Haushalte",
    )
    config = _config(
        require_internal_score_api=True,
        agent_review_enabled=False,
        min_score=75.0,
        min_alert_score=99.0,
        min_editorial_score=99.0,
        require_article_forecast=False,
    )

    decision = shouldNotifyTeams(candidate, _context(candidate), config)

    assert decision["shouldNotify"] is False
    assert decision["highScoreOverride"]["active"] is False
    assert any("Teams Alert Score zu niedrig" in item for item in decision["blockingReasons"])


def test_canonical_score_over_80_waives_only_soft_quality_gates():
    candidate = _candidate(
        score=80.1,
        scoreSource="internal_score_api",
        predictedOR=0.035,
        title="Bundesregierung beschliesst neue Entlastung fuer Millionen Haushalte",
    )
    config = _config(
        require_internal_score_api=True,
        agent_review_enabled=False,
        min_score=75.0,
        min_alert_score=99.0,
        min_editorial_score=99.0,
        require_article_forecast=False,
    )
    context = _context(candidate)
    context["lastTeamsAlertTs"] = NOW_TS - 30 * 60

    decision = shouldNotifyTeams(candidate, context, config)

    assert decision["shouldNotify"] is True
    assert decision["minScore"] == 80.0
    assert decision["highScoreOverride"]["approved"] is True
    assert not decision["highScoreOverride"]["hardBlockers"]
    assert any(
        "Teams Alert Score zu niedrig" in item
        for item in decision["highScoreOverride"]["waivedBlockers"]
    )
    assert any(
        "Prognose zu niedrig" in item
        for item in decision["highScoreOverride"]["waivedBlockers"]
    )


def test_score_over_80_still_respects_teams_cooldown():
    candidate = _candidate(score=91.0, scoreSource="internal_score_api")
    context = _context(candidate)
    context["lastTeamsAlertTs"] = NOW_TS - 29 * 60

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(require_internal_score_api=True, agent_review_enabled=False),
    )

    assert decision["shouldNotify"] is False
    assert decision["highScoreOverride"]["active"] is True
    assert decision["highScoreOverride"]["approved"] is False
    assert any("Teams-Cooldown aktiv" in item for item in decision["blockingReasons"])


def test_score_over_80_still_respects_quiet_hours():
    quiet_ts = int(
        dt.datetime(2026, 7, 20, 4, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(score=91.0, scoreSource="internal_score_api")

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, now_ts=quiet_ts),
        _config(require_internal_score_api=True, agent_review_enabled=False),
    )

    assert decision["shouldNotify"] is False
    assert decision["highScoreOverride"]["approved"] is False
    assert any("Ruhezeit aktiv" in item for item in decision["blockingReasons"])


def test_score_over_80_requires_canonical_api_score_when_configured():
    candidate = _candidate(score=91.0, scoreSource="server_editorial_fallback")

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate),
        _config(require_internal_score_api=True, agent_review_enabled=False),
    )

    assert decision["shouldNotify"] is False
    assert decision["highScoreOverride"]["active"] is False
    assert any("Kein gueltiger interner" in item for item in decision["blockingReasons"])


def test_missing_article_link_does_not_trigger_action_recommendation():
    candidate = _candidate(url="")

    decision = shouldNotifyTeams(candidate, _context(candidate), _config())

    assert decision["shouldNotify"] is False
    assert any("Artikel-Link" in reason for reason in decision["blockingReasons"])


def test_live_pushed_article_is_blocked_from_teams_recommendation():
    candidate = _candidate(score=95.0)
    pushed_history = _history(link=candidate["url"], title="Anderer Titel")

    decision = shouldNotifyTeams(candidate, _context(candidate, history=pushed_history), _config())

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])
    assert decision["highScoreOverride"]["approved"] is False


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

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_live_push_dedup_matches_cms_id_when_history_contains_only_url_id():
    cms_id = "6a57392b664e99bc41e93660"
    candidate = _candidate(
        cmsId=cms_id,
        url=(
            "https://www.bild.de/unterhaltung/stars-und-leute/"
            f"beispiel-artikel-{cms_id}"
        ),
        score=96.0,
    )
    history = _history(
        link=cms_id,
        title="Abweichende Live-Push-Headline",
    )

    decision = shouldNotifyTeams(candidate, _context(candidate, history=history), _config())

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"] == {
        "available": True,
        "authoritative": True,
        "matched": True,
        "matchType": "exact_article",
        "reason": "Bereits live gepusht (gleiche CMS-ID)",
    }
    assert decision["highScoreOverride"]["approved"] is False


def test_raw_live_push_preserves_cms_identity_counts_sport_and_blocks_recommendation():
    from app.routers.push import _parse_bild_messages

    cms_id = "6a57392b664e99bc41e93660"
    article_url = f"https://www.bild.de/sport/fussball/topspiel-{cms_id}.html"
    parsed = _parse_bild_messages(
        [
            {
                "id": "raw-live-sport-push",
                "sendDate": NOW_TS - 60 * 60,
                "headline": "Topspiel ist entschieden",
                "url": "https://www.bild.de/news",
                "urlId": article_url,
                "sourceType": "EDITORIAL",
            }
        ]
    )

    assert len(parsed) == 1
    assert parsed[0]["link"] == article_url
    assert parsed[0]["cmsId"] == cms_id
    assert parsed[0]["cat"] == "sport"

    candidate = _candidate(
        cmsId=cms_id,
        url=f"https://www.bild.de/sport/fussball/anderer-slug-{cms_id}.html",
        category="news",
        score=99.0,
    )
    context = build_teams_alert_context(
        [candidate],
        history=parsed,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=NOW_TS,
    )
    context["dashboardRank"] = 1
    decision = shouldNotifyTeams(candidate, context, _config(excluded_sections=()))

    assert context["sportPushesToday"] == 1
    assert decision["section"] == "sport"
    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_breaking_cannot_recommend_an_article_already_pushed_live():
    candidate = _candidate(
        title="Eilmeldung: Regierung beschliesst sofort neue Hilfen",
        score=99.0,
        predictedOR=0.09,
        isBreaking=True,
        isEilmeldung=True,
    )
    pushed_history = _history(link=candidate["url"], title="Fruehere Live-Push-Zeile")

    decision = shouldNotifyTeams(
        candidate,
        _context(candidate, history=pushed_history),
        _config(agent_review_enabled=False),
    )

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["matchType"] == "exact_article"
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


def test_promotional_giveaway_is_hard_blocked_even_at_maximum_score():
    candidate = _candidate(
        title="TECH-HIGHLIGHT: einen von 15 Dæly® Familienkalender gewinnen!",
        category="news",
        type="editorial",
        score=99.0,
        predictedOR=0.09,
        url=(
            "https://www.bild.de/sonstiges/bildplus-gewinnspiele-aktionen/"
            "tech-highlight-einen-von-15-dly-familienkalender-gewinnen-"
            "6a671914cbfb33efe26fa782"
        ),
    )
    context = _context(candidate, history=_history(minutes_since_last_push=120))
    context["dashboardRank"] = 1

    decision = shouldNotifyTeams(
        candidate,
        context,
        _config(
            agent_review_enabled=False,
            score_only_mode=True,
            min_alert_score=40.0,
            min_editorial_score=40.0,
        ),
    )

    assert decision["shouldNotify"] is False
    assert any(
        "Gewinnspiel/Promo/Advertorial" in reason
        for reason in decision["blockingReasons"]
    )
    assert decision["highScoreOverride"]["approved"] is False
    assert any(
        "Gewinnspiel/Promo/Advertorial" in reason
        for reason in decision["highScoreOverride"]["hardBlockers"]
    )


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
    assert message["payload"]["pushScoreSourceLabel"] == "frisches Push-Balancer-Rating"


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


def test_highest_push_balancer_score_wins_inside_the_selection_band():
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

    # Top-1-Garantie: der hoechste rohe Push-Balancer-Score gewinnt; die
    # Deutschland-Relevanz bleibt Metadatum und Gate, kein Auswahl-Override.
    assert result["selectedCandidateId"] == international["url"]
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


def test_push_score_over_80_overrides_daily_fatigue_gate():
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

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Tagesstrategie" in reason
        for reason in decision["highScoreOverride"]["waivedBlockers"]
    )


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
    assert any("Ruhezeit aktiv" in reason for reason in decision["blockingReasons"])
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


def test_transport_rejects_legacy_heartbeat_payload_outside_quiet_hours():
    now_ts = _gold_slot_ts()
    message = {
        "payload": {
            "type": "teams_heartbeat",
            "text": "Synthetic off-schedule recommendation",
        }
    }

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, _config())

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "outside_schedule_blocked"
    urlopen.assert_not_called()


def _fully_approved_transport_message(*, slot_gate_enabled: bool):
    if slot_gate_enabled:
        decision_ts = int(
            dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        )
        candidate = _candidate(score=84.0, pubDate=_iso(decision_ts - 10 * 60))
        config = _smart_config()
        context = build_teams_alert_context(
            [candidate],
            history=_history(minutes_since_last_push=60, now_ts=decision_ts),
            alert_state={},
            last_teams_alert_ts=0,
            teams_alerts_today=2,
            recent_alerts=[],
            now_ts=decision_ts,
            config=config,
        )
        context["pushesToday"] = 5
        context["dashboardRank"] = 1
    else:
        decision_ts = int(
            dt.datetime(2026, 7, 18, 20, 10, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        )
        candidate = _candidate(
            score=95.0,
            predictedOR=0.09,
            pubDate=_iso(decision_ts - 5 * 60),
        )
        config = _config(slot_gate_enabled=False, excluded_sections=())
        context = _context(candidate, now_ts=decision_ts)
    decision = shouldNotifyTeams(candidate, context, config)
    assert decision["shouldNotify"] is True
    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    assert message["payload"]["type"] == "push_recommendation"
    return message, config


def test_transport_revalidates_real_slot_even_when_decision_gate_is_disabled():
    message, config = _fully_approved_transport_message(slot_gate_enabled=False)
    off_slot = int(
        dt.datetime(2026, 7, 13, 17, 20, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    with (
        patch("app.notifications.teams.time.time", return_value=off_slot),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "outside_schedule_blocked"
    urlopen.assert_not_called()


def test_transport_allows_approved_recommendation_in_binding_slot_window():
    message, config = _fully_approved_transport_message(slot_gate_enabled=True)
    binding_slot = int(
        dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    with (
        patch("app.notifications.teams.time.time", return_value=binding_slot),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is True
    urlopen.assert_called_once()


@pytest.mark.parametrize(("seconds_after_slot", "allowed"), [(299, True), (300, False)])
def test_transport_enforces_binding_slot_grace_boundary(seconds_after_slot, allowed):
    message, config = _fully_approved_transport_message(slot_gate_enabled=True)
    binding_slot = int(
        dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    with (
        patch(
            "app.notifications.teams.time.time",
            return_value=binding_slot + seconds_after_slot,
        ),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is allowed
    if allowed:
        urlopen.assert_called_once()
    else:
        assert result["reason"] == "outside_schedule_blocked"
        urlopen.assert_not_called()


def test_mandatory_transport_never_reposts_an_ambiguous_timeout():
    message, config = _fully_approved_transport_message(slot_gate_enabled=True)
    binding_slot = int(
        dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    message["_mandatorySlotTop1"] = True
    message["payload"]["mandatorySlotTop1"] = True
    message["payload"]["rankingPosition"] = 1
    config = replace(config, webhook_max_attempts=3, webhook_retry_backoff_seconds=0)

    with (
        patch("app.notifications.teams.time.time", return_value=binding_slot),
        patch(
            "app.notifications.teams.urllib.request.urlopen",
            side_effect=urllib.error.URLError("ambiguous timeout"),
        ) as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["deliveryUncertain"] is True
    assert result["attempts"] == 1
    urlopen.assert_called_once()


def test_transport_aborts_when_retry_would_cross_slot_window():
    message, config = _fully_approved_transport_message(slot_gate_enabled=True)
    binding_slot = int(
        dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    with (
        patch(
            "app.notifications.teams.time.time",
            side_effect=[binding_slot, binding_slot, binding_slot + 301],
        ),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["reason"] == "outside_schedule_blocked"
    assert result["attempts"] == 0
    urlopen.assert_not_called()


def test_transport_rejects_unknown_payload_type_fail_closed():
    message, config = _fully_approved_transport_message(slot_gate_enabled=False)
    message["payload"]["type"] = "teams_recommendation"

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["reason"] == "unsupported_payload_type"
    urlopen.assert_not_called()


def test_transport_rejects_stale_message_during_a_later_binding_slot():
    message, config = _fully_approved_transport_message(slot_gate_enabled=True)
    later_slot = int(
        dt.datetime(2026, 7, 13, 18, 34, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )

    with (
        patch("app.notifications.teams.time.time", return_value=later_slot),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["reason"] == "outside_schedule_blocked"
    urlopen.assert_not_called()


def test_live_push_mirror_remains_sendable_outside_recommendation_slots():
    off_slot = int(
        dt.datetime(2026, 7, 13, 17, 20, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    message = {"payload": {"type": "live_push_sent", "text": "Synthetic live push"}}

    with (
        patch("app.notifications.teams.time.time", return_value=off_slot),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, _config())

    assert result["ok"] is True
    urlopen.assert_called_once()


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


def test_transport_rejects_payload_without_live_push_volume_context():
    now_ts = _gold_slot_ts()
    candidate = _candidate(score=82.0)
    config = _config()
    context = _context(candidate, now_ts=now_ts)
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    message["payload"]["livePushVolumeConsidered"] = False

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["blocked"] is True
    assert "live push volume" in result["error"].lower()
    urlopen.assert_not_called()


def test_transport_rejects_missing_exact_live_push_dedup_approval():
    now_ts = _gold_slot_ts()
    candidate = _candidate(score=82.0)
    config = _config()
    context = _context(candidate, now_ts=now_ts)
    decision = shouldNotifyTeams(candidate, context, config)
    message = buildTeamsPushRecommendation(candidate, context, decision, config)
    message["_livePushDedupApproved"] = False
    message["payload"]["livePushDedupApproved"] = False

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.urllib.request.urlopen") as urlopen,
    ):
        result = sendTeamsNotification(message, config)

    assert result["blocked"] is True
    assert result["error"] == "Exact live-push duplicate protection approval missing"
    urlopen.assert_not_called()


def test_teams_message_contains_required_editorial_fields():
    candidate = _candidate()
    runner_up = _candidate(
        id="article-2",
        url="https://www.bild.de/unterhaltung/article-2",
        title="Ariana Grande kündigt eine längere Auszeit an",
        category="unterhaltung",
        score=76.2,
    )
    context = _context(candidate, now_ts=_gold_slot_ts())
    decision = shouldNotifyTeams(candidate, context, _config())
    decision["competition"] = {
        "eligibleCompetitors": 1,
        "runnerUpScore": runner_up["score"],
        "runnerUp": {
            "articleTitle": runner_up["title"],
            "articleUrl": runner_up["url"],
            "category": runner_up["category"],
            "pushScore": runner_up["score"],
            "rankingPosition": 2,
        },
    }

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    text = message["text"]

    assert text.startswith("🔵 PUSH-EMPFEHLUNG")
    assert text.count("Top 1:") == 1
    assert text.count("Alternative (Platz 2):") == 1
    assert "Score: 78,4/100" in text
    assert "Warum:" in text
    assert candidate["url"] in text
    assert candidate["recommendedText"] in text
    assert runner_up["title"] in text
    assert runner_up["url"] in text
    assert "Score: 76,2/100" in text
    assert len(text) < 1_000
    for removed_label in (
        "Alternatives Zeitfenster:",
        "Spätester sinnvoller Versand:",
        "Warum dieser Zeitpunkt:",
        "Tagesstand:",
        "Sportstand:",
        "Letzter gesendeter Push:",
        "Auswirkung auf den Tagesplan:",
    ):
        assert removed_label not in text
    assert "Teams-Alert-Score: " not in text
    assert "Push-Balancer-Breakdown:" not in text
    payload = message["payload"]
    assert payload["recommendedAction"] == "Jetzt pushen"
    assert payload["recommendationPolicyVersion"] == "live-aware-dual-message-v11"
    assert payload["recommendationsIndependentFromLivePushes"] is False
    assert payload["livePushVolumeConsidered"] is True
    assert payload["statusLabel"] in {"Sofort senden", "Nach Live-Push neu terminiert"}
    assert payload["rankingPosition"] == 1
    assert payload["dailyStatusLabel"].endswith("Pushes gesendet")
    assert "Sportanteil" in payload["sportStatusLabel"]
    assert payload["lastLivePushLabel"]
    assert payload["planImpactLabel"]
    assert payload["livePushDedupApproved"] is True
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
    assert payload["messageHtml"].count("<strong>Top 1:</strong>") == 1
    assert payload["messageHtml"].count("<strong>Alternative (Platz 2):</strong>") == 1
    assert f'href="{candidate["url"]}"' in payload["messageHtml"]
    assert f'href="{runner_up["url"]}"' in payload["messageHtml"]
    assert len(payload["messageHtml"].encode()) < 2_000
    assert payload["subject"].startswith("🔵 PUSH-EMPFEHLUNG:")
    assert "Push-Balancer-Breakdown" not in payload["messageHtml"]
    assert payload["alternativeRecommendation"] == {
        "articleTitle": runner_up["title"],
        "articleUrl": runner_up["url"],
        "category": runner_up["category"],
        "pushScore": runner_up["score"],
        "rankingPosition": 2,
    }
    assert isinstance(payload["whyNow"], list)
    assert isinstance(payload["whyPushworthy"], list)


def test_score_over_80_reaches_final_payload_despite_soft_quality_threshold():
    candidate = _candidate(
        score=80.1,
        scoreSource="internal_score_api",
        predictedOR=0.035,
        title="Bundesregierung beschliesst neue Entlastung fuer Millionen Haushalte",
    )
    context = _context(candidate, now_ts=_gold_slot_ts())
    config = _config(
        require_internal_score_api=True,
        agent_review_enabled=False,
        min_score=75.0,
        min_alert_score=99.0,
        min_editorial_score=99.0,
        min_recommendation_quality=99.0,
        require_article_forecast=False,
    )
    decision = shouldNotifyTeams(candidate, context, config)

    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert message["payload"]["dispatchApproved"] is True
    assert message["payload"]["highScoreOverride"] == {
        "active": True,
        "approved": True,
        "threshold": 80.0,
        "waivedGateCount": 3,
        "hardBlockerCount": 0,
    }
    assert message["_recommendationReview"]["highScoreOverrideApplied"] is True
    assert message["payload"]["highScoreOverride"]["approved"] is True


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
    assert payload["recommendationConfidence"] in {"hoch", "mittel", "niedrig"}
    assert "Sicherheit " not in message["text"]
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
    assert message["text"].startswith("🔵 PUSH-EMPFEHLUNG (nicht freigegeben)")
    assert message["payload"]["subject"].startswith("Nicht senden:")

    with patch("app.notifications.teams.urllib.request.urlopen") as urlopen:
        result = sendTeamsNotification(message, config)

    assert result["ok"] is False
    assert result["blocked"] is True
    urlopen.assert_not_called()


def test_stale_live_history_blocks_recommendation_fail_closed():
    """Ohne autoritative Live-Historie darf kein Vorschlag versendet werden."""
    candidate = _candidate()
    context = _context(candidate, now_ts=_gold_slot_ts())
    context["historyAuthoritative"] = False
    config = _config(agent_review_enabled=False)
    decision = shouldNotifyTeams(candidate, context, config)

    message = buildTeamsPushRecommendation(candidate, context, decision, config)

    assert decision["livePushComparison"]["available"] is False
    assert decision["livePushComparison"]["authoritative"] is False
    assert decision["shouldNotify"] is False
    assert any(
        "sicherheitshalber gestoppt" in item for item in decision["blockingReasons"]
    )
    assert message["payload"]["type"] == "push_recommendation_preview"
    assert message["payload"]["dispatchApproved"] is False


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

    assert text.startswith("🔵 PUSH-EMPFEHLUNG")
    assert "Top 1:" in text
    assert text.count(message["payload"]["recommendedPushText"]) == 1
    assert text.count(candidate["title"]) <= 1
    assert message["payload"]["alternativePushTitle"] != candidate["title"]


def test_or_prediction_ratio_is_displayed_as_percent_not_raw_ratio():
    candidate = _candidate(predictedOR=0.0477)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())

    assert normalize_predicted_or(0.0477) == 4.77
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
    assert "Slot-Prognose" in message["payload"]["predictedORExplanation"]
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
    assert "Slot-Prognose" in message["payload"]["predictedORExplanation"]
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
        result = sendTeamsNotification(
            {"payload": {"type": "live_push_sent", "text": "test"}},
            _config(),
        )

    assert result["ok"] is False
    assert "webhook down" in result["error"]
    assert "Teams webhook send failed" in caplog.text


def test_send_failure_is_recorded_without_crashing_cycle(tmp_db):
    now_ts = int(
        dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        url="https://www.bild.de/politik/send-failure-recorded",
        pubDate=_iso(now_ts - 10 * 60),
    )
    from app.database import push_db_upsert, teams_recommendation_list_recent

    push_db_upsert(_history(now_ts=now_ts))

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch(
            "app.notifications.teams.urllib.request.urlopen",
            side_effect=urllib.error.URLError("webhook down"),
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(),
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
    # Nach einem mehrdeutigen Timeout wird nicht erneut gepostet: Teams koennte
    # den ersten Request bereits angenommen haben. Der Status bleibt sichtbar.
    assert rows[0]["status"] == "delivery_uncertain"
    assert rows[0]["send_status"] == "delivery_uncertain"
    assert rows[0]["send_error"]
    from app.notifications import teams as teams_module

    with teams_module._RECENT_SEND_LOCK:
        assert candidate["url"] not in teams_module._RECENT_SEND_MEMORY


def test_mandatory_slot_retries_transport_failure_but_records_only_one_success(tmp_db):
    now_ts = int(
        dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        url="https://www.bild.de/politik/mandatory-transport-retry",
        pubDate=_iso(now_ts - 10 * 60),
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(now_ts=now_ts))
    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch(
            "app.notifications.teams.send_teams_notification",
            side_effect=[
                RuntimeError("temporary transport crash"),
                {"ok": True, "status": 200},
            ],
        ) as send,
    ):
        first = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )
        second = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )
        third = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert first["sent"] is False
    assert second["sent"] is True
    assert third["sent"] is False
    assert third["reason"] == "no_candidate"
    assert send.call_count == 2


def test_expired_slot_is_blocked_before_memory_and_database_claim(tmp_db):
    from app.database import push_db_upsert

    decision_ts = int(
        dt.datetime(2026, 7, 13, 17, 30, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        score=90.0,
        predictedOR=0.08,
        pubDate=_iso(decision_ts - 5 * 60),
    )
    push_db_upsert(_history(minutes_since_last_push=90, now_ts=decision_ts))
    config = _smart_config(agent_review_enabled=False)

    with (
        patch("app.notifications.teams.time.time", return_value=decision_ts + 301),
        patch("app.notifications.teams._memory_send_blocker_or_reserve") as memory_claim,
        patch("app.notifications.teams.teams_alert_try_claim_send") as database_claim,
        patch("app.notifications.teams.send_teams_notification") as send,
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=config,
            now_ts=decision_ts,
            history_authoritative=True,
        )

    assert result["sent"] is False
    assert result["reason"] == "outside_schedule_blocked"
    memory_claim.assert_not_called()
    database_claim.assert_not_called()
    send.assert_not_called()


def test_memory_blocked_top_candidate_does_not_starve_runner_up(tmp_db):
    from app.database import push_db_upsert
    from app.notifications import teams as teams_module

    now_ts = int(
        dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    config = _smart_config(
        agent_review_enabled=False,
        min_score=75.0,
        global_cooldown_minutes=30,
        alert_cooldown_minutes=60,
    )
    top = _candidate(
        id="blocked-top",
        url="https://www.bild.de/news/blocked-top",
        title="Bundesweite Unwetterwarnung: Schwere Gewitter ziehen auf",
        recommendedText="Schwere Gewitter: Bundesweite Unwetterwarnung gilt ab sofort",
        score=96.0,
        predictedOR=0.085,
        pubDate=_iso(now_ts - 8 * 60),
    )
    runner_up = _candidate(
        id="eligible-runner-up",
        url="https://www.bild.de/geld/eligible-runner-up",
        title="Rentenplus beschlossen: Millionen Beschaeftigte bekommen mehr Geld",
        recommendedText="Rentenplus beschlossen: So viel Geld bekommen Beschaeftigte",
        category="geld",
        score=90.0,
        predictedOR=0.075,
        pubDate=_iso(now_ts - 10 * 60),
    )
    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))

    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()
    teams_module._memory_send_blocker_or_reserve(
        article_key=top["url"],
        title=top["title"],
        now_ts=now_ts - 45 * 60,
        config=config,
    )
    teams_module._memory_record_send_result(top["url"], ok=True, now_ts=now_ts - 45 * 60)

    try:
        with (
            patch("app.notifications.teams.time.time", return_value=now_ts),
            patch(
                "app.notifications.teams.send_teams_notification",
                return_value={"ok": True, "status": 200},
            ) as send,
        ):
            result = evaluate_and_send_best_candidate(
                [top, runner_up],
                config=config,
                now_ts=now_ts,
                history_authoritative=True,
            )

        assert result["sent"] is True
        assert result["candidateId"] == runner_up["url"]
        assert result["evaluation"]["memoryGuard"] == {
            "skippedCandidates": 1,
            "reasons": {"memory_article_alert_cooldown": 1},
        }
        send.assert_called_once()
    finally:
        with teams_module._RECENT_SEND_LOCK:
            teams_module._RECENT_SEND_MEMORY.clear()


def test_database_claim_rejection_releases_process_reservation(tmp_db):
    from app.database import push_db_upsert
    from app.notifications import teams as teams_module

    now_ts = int(
        dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        id="claim-race",
        url="https://www.bild.de/news/claim-race",
        title="Bundesrat beschliesst Rentenplus fuer Millionen Beschaeftigte",
        recommendedText="Rentenplus beschlossen: So viel Geld bekommen Beschaeftigte",
        score=91.0,
        predictedOR=0.076,
        pubDate=_iso(now_ts - 10 * 60),
    )
    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))
    with teams_module._RECENT_SEND_LOCK:
        teams_module._RECENT_SEND_MEMORY.clear()

    try:
        with (
            patch(
                "app.notifications.teams.teams_alert_try_claim_send",
                return_value={"claimed": False, "reason": "article_already_sent"},
            ),
            patch("app.notifications.teams.time.time", return_value=now_ts),
            patch("app.notifications.teams.send_teams_notification") as send,
        ):
            result = evaluate_and_send_best_candidate(
                [candidate],
                config=_smart_config(agent_review_enabled=False, min_score=75.0),
                now_ts=now_ts,
                history_authoritative=True,
            )

        assert result["sent"] is False
        assert result["reason"] == "send_claim_blocked"
        send.assert_not_called()
        with teams_module._RECENT_SEND_LOCK:
            assert candidate["url"] not in teams_module._RECENT_SEND_MEMORY
    finally:
        with teams_module._RECENT_SEND_LOCK:
            teams_module._RECENT_SEND_MEMORY.clear()


def test_send_cycle_blocks_when_live_push_dedup_is_stale(tmp_db):
    """Der Zyklus stoppt ohne autoritative Live-Dedup-Quelle."""
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        id="stale-dedup-sends",
        url="https://www.bild.de/news/stale-dedup-sends",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
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
    assert result["sent"] is False
    assert result["reason"] == "no_candidate"
    send.assert_not_called()


def test_heartbeat_never_dispatches_even_when_runtime_flag_is_enabled(tmp_db):
    """A stale env override must not restore off-raster recommendations."""
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        id="hb-silent",
        url="https://www.bild.de/politik/inland/hb-silent",
        title="Neue Details im Fall X: Ermittler praesentieren Zwischenstand",
        category="politik",
        score=72.0,
        predictedOR=0.05,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Fall X: Ermittler praesentieren neuen Zwischenstand",
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))
    with patch(
        "app.notifications.teams.send_teams_notification",
        return_value={"ok": True, "status": 200},
    ) as send:
        result = _maybe_send_heartbeat(
            [candidate],
            config=_config(agent_review_enabled=False, heartbeat_enabled=True),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["fired"] is False
    assert result["reason"] == "outside_schedule_blocked"
    send.assert_not_called()


def test_heartbeat_runtime_override_stays_fail_closed_with_recent_post(tmp_db):
    from app.database import teams_alert_record

    now_ts = _gold_slot_ts()
    teams_alert_record(
        article_key="prev",
        article_id="prev",
        article_url="https://www.bild.de/x",
        title_hash="h",
        article_title="Frueherer Push",
        score=80.0,
        predicted_or=0.05,
        candidate_updated_at=now_ts,
        is_breaking=False,
        reason="r",
        status="sent",
        error="",
        decision_ts=now_ts - 10 * 60,
    )
    candidate = _candidate(
        id="hb-recent",
        url="https://www.bild.de/politik/inland/hb-recent",
        title="Sachmeldung mit konkretem Ereignis",
        category="politik",
        score=72.0,
        pubDate=_iso(now_ts - 10 * 60),
    )
    with patch("app.notifications.teams.send_teams_notification") as send:
        result = _maybe_send_heartbeat(
            [candidate],
            config=_config(agent_review_enabled=False, heartbeat_enabled=True),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["fired"] is False
    assert result["reason"] == "outside_schedule_blocked"
    send.assert_not_called()


@pytest.mark.parametrize(
    "headline",
    [
        "BILD exklusiv: Merz wirft Verkehrsminister Patrick Schnieder raus!",
        "Paukenschlag: Kanzler feuert Wirtschaftsminister",
        "Nach Skandal: Minister muss gehen",
        "Regierungsbeben! Habeck schmeißt hin",
        "Ampel platzt: Koalition am Ende",
        "Minister wackelt: Rücktrittsforderungen werden lauter",
        "Preis-Schock an der Zapfsäule",
        "Aldi ruft Hackfleisch zurück",
        "Traditionsbäcker meldet Insolvenz an",
    ],
)
def test_has_news_event_recognizes_boulevard_event_idioms(headline):
    """Boulevard-Ereignis-Idiome (Rauswurf, Ruecktritt, Krise, Rueckruf, Insolvenz)
    muessen als konkretes Nachrichten-Ereignis erkannt werden."""
    assert _has_news_event(headline) is True


@pytest.mark.parametrize(
    "headline",
    [
        "So sparen Sie beim Tanken: 5 Tipps",
        "Die 10 schönsten Strände Europas",
        "Horoskop heute: Das erwartet die Sternzeichen",
        "Testen Sie Ihr Wissen im großen Sommer-Quiz",
    ],
)
def test_has_news_event_ignores_service_teaser(headline):
    """Reine Service-/Ratgeber-/Raetsel-Teaser sind kein Nachrichten-Ereignis."""
    assert _has_news_event(headline) is False


def test_slot_fit_review_failsafe_when_llm_unavailable():
    """Ohne LLM-Key wird nicht zurueckgehalten: fitsNow=True, available=False."""
    review = _llm_slot_fit_review(
        _candidate(title="Ratgeber: So pflegen Sie Zimmerpflanzen", category="leben-wissen"),
        now_ts=_gold_slot_ts(),
        config=_config(agent_review_enabled=False),
    )
    assert review["available"] is False
    assert review["fitsNow"] is True


def _slot_fit_llm_patch(payload_json):
    return (
        patch("push_title_agent._llm_unavailable_reason", return_value=""),
        patch("push_title_agent._llm_call", return_value=json.dumps(payload_json)),
    )


def test_mandatory_slot_never_defers_soft_story_to_better_hour(tmp_db):
    """Ein fester Pflichtslot darf nicht durch die optionale Slot-Fit-Jury ausfallen."""
    now_ts = int(dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    now_hour = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin")).hour
    better = min(21, now_hour + 1)
    candidate = _candidate(
        id="slot-defer",
        url="https://www.bild.de/news/stromausfall-x",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))
    unavail, call = _slot_fit_llm_patch(
        {"fitsNow": False, "confidence": 0.9, "betterSlotHour": better, "reason": "weich"}
    )
    with (
        unavail,
        call,
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.send_teams_notification", return_value={"ok": True}) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["sent"] is True
    send.assert_called_once()


def test_slot_fit_never_defers_breaking(tmp_db):
    """Breaking wird nie fuer einen besseren Slot zurueckgehalten."""
    now_ts = int(dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    now_hour = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin")).hour
    better = min(21, now_hour + 1)
    candidate = _candidate(
        id="slot-breaking",
        url="https://www.bild.de/news/eil-x",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        isBreaking=True,
        isEilmeldung=True,
        breakingProvenance="cms_verified",
        pubDate=_iso(now_ts - 5 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))
    unavail, call = _slot_fit_llm_patch(
        {"fitsNow": False, "confidence": 0.9, "betterSlotHour": better, "reason": "warten"}
    )
    with (
        unavail,
        call,
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.send_teams_notification", return_value={"ok": True}) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_smart_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result.get("reason") != "slot_deferred"
    assert result["sent"] is True
    send.assert_called_once()


def test_slot_fit_does_not_defer_when_article_ages_out_before_slot(tmp_db):
    """Kein Zurueckhalten, wenn der Artikel den besseren Slot nicht mehr erlebt.

    Sonst laeuft er vor der Ziel-Hot-Hour in die harte Publikations-Altersgrenze
    und wuerde dort geblockt -> lautloser Drop statt Push.
    """
    now_ts = int(dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    now_hour = dt.datetime.fromtimestamp(now_ts, ZoneInfo("Europe/Berlin")).hour
    better = min(21, now_hour + 2)
    candidate = _candidate(
        id="slot-ageout",
        url="https://www.bild.de/news/stromausfall-ageout",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 90 * 60),  # 1,5h alt
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=90, now_ts=now_ts))
    # Sendbar jetzt (1,5h < 2h), aber 1,5h + 2h Wartezeit > 2h Altersgrenze.
    cfg = _smart_config(
        agent_review_enabled=False,
        max_article_age_hours=2,
        slot_fit_max_article_age_hours=4,
        slot_fit_max_defer_hours=3,
    )
    unavail, call = _slot_fit_llm_patch(
        {"fitsNow": False, "confidence": 0.9, "betterSlotHour": better, "reason": "weich"}
    )
    with (
        unavail,
        call,
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch("app.notifications.teams.send_teams_notification", return_value={"ok": True}) as send,
        patch(
            "app.notifications.teams._memory_send_blocker_or_reserve",
            return_value={"blocked": False, "reserved": True},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=cfg,
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result.get("reason") != "slot_deferred"
    assert result["sent"] is True
    send.assert_called_once()


def test_heartbeat_story_can_escalate_to_hard_alert():
    """Eine nur per Heartbeat gemeldete Story darf bei klarer Score-Eskalation
    regulaer als Hard-Alert erneut in den Channel; ohne Eskalation bleibt sie
    gesperrt. Ein echter Alert (kein Heartbeat) sperrt weiterhin dauerhaft."""
    cfg = _config(dynamic_threshold_enabled=False)
    hb_state = {
        "status": "sent",
        "last_reason": _HEARTBEAT_ALERT_REASON,
        "last_score": 72.0,
        "last_decision_ts": NOW_TS - 3600,
    }
    # Kein deutlicher Anstieg -> weiter gesperrt (aber mit Eskalations-Hinweis).
    low = _realert_blocker_or_reason({}, hb_state, 74.0, 6.0, False, NOW_TS, cfg)
    assert "blocker" in low and "Heartbeat" in low["blocker"]
    # Klar eskaliert (>= last_score + margin und >= min_score) -> erlaubt.
    high = _realert_blocker_or_reason({}, hb_state, 85.0, 6.0, False, NOW_TS, cfg)
    assert "blocker" not in high
    assert "positive" in high
    # Breaking auf Heartbeat-Story -> erlaubt.
    brk = _realert_blocker_or_reason({}, hb_state, 60.0, 6.0, True, NOW_TS, cfg)
    assert "blocker" not in brk
    # Echter Alert (kein Heartbeat) -> weiterhin harter Dauerblock.
    real_state = {
        "status": "sent",
        "last_reason": "Score/OR ok",
        "last_score": 88.0,
        "last_decision_ts": NOW_TS - 3600,
    }
    real = _realert_blocker_or_reason({}, real_state, 90.0, 6.0, False, NOW_TS, cfg)
    assert "blocker" in real and "Heartbeat" not in real["blocker"]


def test_daily_plan_already_covered_surfaces_pushed_stories():
    """Bereits live/per Teams abgedeckte Top-Themen werden sichtbar gemacht."""
    entries = [
        {
            "candidateId": "a",
            "title": "Terror-Experte Neumann zur Lage",
            "section": "news",
            "score": 94.1,
            "hardBlockers": ["Bereits per Teams gemeldet; derselbe Artikel ..."],
        },
        {
            "candidateId": "b",
            "title": "Porsche baut 5000 Stellen ab",
            "section": "wirtschaft",
            "score": 81.0,
            "hardBlockers": ["Bereits live gepusht (gleiche Artikel-URL oder CMS-ID); ..."],
        },
        {
            "candidateId": "c",
            "title": "Schwache Service-Story",
            "section": "news",
            "score": 61.0,
            "hardBlockers": [],
        },
    ]
    covered = _daily_plan_already_covered(entries)
    assert [c["title"] for c in covered] == [
        "Terror-Experte Neumann zur Lage",
        "Porsche baut 5000 Stellen ab",
    ]
    assert covered[0]["via"] == "Teams-Hinweis"
    assert covered[1]["via"] == "Live-Push"
    # Sortierung nach Score absteigend.
    assert covered[0]["score"] >= covered[1]["score"]


def test_heartbeat_excludes_fiction_tv_teaser(tmp_db):
    """Auch als Fallback darf kein Fiktions-/TV-Programm-Teaser (GZSZ) gepostet werden."""
    now_ts = int(
        dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    gzsz = _candidate(
        id="hb-gzsz",
        url="https://www.bild.de/unterhaltung/tv-fernsehformate/gzsz",
        title="GZSZ heute auf RTL+: Ninas Festnahme erschuettert den ganzen Kiez",
        category="unterhaltung",
        score=83.0,
        pubDate=_iso(now_ts - 10 * 60),
    )
    with patch("app.notifications.teams.send_teams_notification") as send:
        result = _maybe_send_heartbeat(
            [gzsz],
            config=_config(agent_review_enabled=False, heartbeat_enabled=True),
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["fired"] is False
    assert result["reason"] == "outside_schedule_blocked"
    send.assert_not_called()


def test_heartbeat_policy_blocks_before_message_build_or_live_refresh(tmp_db):
    now_ts = int(
        dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
    candidate = _candidate(
        id="heartbeat-race",
        url="https://www.bild.de/news/heartbeat-race",
        title="Regierung beschliesst neues Hilfspaket fuer Millionen Familien",
        category="news",
        score=95.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 5 * 60),
    )
    with (
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
        ) as refresh,
        patch(
            "app.notifications.teams.build_teams_heartbeat_message",
        ) as build,
        patch("app.notifications.teams.send_teams_notification") as send,
    ):
        result = _maybe_send_heartbeat(
            [candidate],
            config=_config(
                agent_review_enabled=False,
                heartbeat_enabled=True,
                excluded_sections=(),
            ),
            now_ts=now_ts,
            history=_history(minutes_since_last_push=60, now_ts=now_ts),
            history_authoritative=True,
            refresh_live_history_before_dispatch=True,
        )

    refresh.assert_not_called()
    build.assert_not_called()
    send.assert_not_called()
    assert result["fired"] is False
    assert result["reason"] == "outside_schedule_blocked"


@pytest.mark.parametrize(
    "dedup_reason",
    ["live_push_duplicate_blocked", "live_push_dedup_unavailable"],
)
def test_cycle_never_bypasses_dispatch_dedup_with_heartbeat(dedup_reason):
    """Ein finaler Live-Dedup-Stopp gilt auch fuer den Heartbeat-Pfad."""
    import app.notifications.teams as teams_module

    config = _config(
        agent_review_enabled=False,
        heartbeat_enabled=True,
        daily_schedule_send_enabled=False,
        live_push_posts_enabled=False,
        require_internal_score_api=False,
    )
    refresh = {"history": [], "history_authoritative": True}
    with (
        patch("app.notifications.teams.TeamsAlertConfig", return_value=config),
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
            return_value=refresh,
        ),
        patch("app.notifications.teams.announce_new_live_pushes", return_value={}),
        patch("app.notifications.teams.send_teams_daily_schedule_if_due", return_value={}),
        patch(
            "app.routers.feed.build_articles_payload",
            return_value={"articles": [_candidate()]},
        ),
        patch(
            "app.notifications.teams.evaluate_and_send_best_candidate",
            return_value={"ok": True, "sent": False, "reason": dedup_reason},
        ),
        patch("app.notifications.teams._maybe_send_heartbeat") as heartbeat,
    ):
        result = teams_module._run_teams_alert_cycle_inner()

    heartbeat.assert_not_called()
    assert result["heartbeat"] == {"fired": False, "reason": dedup_reason}


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
        patch(
            "app.routers.push._parse_bild_messages",
            return_value=[{"message_id": "cache-1", "ts_num": int(time.time())}],
        ),
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


def test_push_refresh_exposes_fresh_snapshot_privately_but_fails_closed_on_db_error():
    """Frische Parse-Daten bleiben intern nutzbar; die oeffentliche Antwort ist sauber."""
    import app.routers.push as push_router

    raw = [{"id": "live-1", "sendDate": 1_800_000_000}]
    parsed = [{"message_id": "live-1", "ts_num": 1_800_000_000, "cat": "sport"}]
    with (
        patch("app.routers.push._fetch_live_push_snapshot", return_value=(raw, [])),
        patch("app.routers.push._parse_bild_messages", return_value=parsed),
        patch("app.routers.push.push_db_upsert", side_effect=RuntimeError("synthetic db error")),
    ):
        internal = push_router._build_refresh_response(include_history=True)
        public = push_router._build_refresh_response()

    assert internal["history_authoritative"] is False
    assert internal["_snapshot_authoritative"] is True
    assert internal["_parsed_history"] == parsed
    assert "_snapshot_authoritative" not in public
    assert "_parsed_history" not in public
    assert public["history_authoritative"] is False


def test_empty_live_snapshot_is_never_authoritative():
    import app.routers.push as push_router

    with (
        patch("app.routers.push._fetch_live_push_snapshot", return_value=([], [])),
        patch("app.routers.push.push_db_upsert") as upsert,
    ):
        result = push_router._build_refresh_response(include_history=True)

    assert result["source"] == "live"
    assert result["history_authoritative"] is False
    assert result["_snapshot_authoritative"] is False
    assert result["_parsed_history"] == []
    upsert.assert_not_called()


def test_live_snapshot_with_unfinished_pagination_fails_instead_of_truncating():
    import app.routers.push as push_router

    class _JsonResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    responses = [
        _JsonResponse(
            {
                "messages": [{"id": f"page-{index}"}],
                "next": f"page={index + 1}",
            }
        )
        for index in range(11)
    ]
    with (
        patch("app.routers.push.push_api_base_candidates", return_value=["https://push.test"]),
        patch("app.routers.push.urllib.request.urlopen", side_effect=responses) as urlopen,
    ):
        with pytest.raises(RuntimeError, match="pagination exceeded"):
            push_router._fetch_live_push_snapshot(force=True)

    assert urlopen.call_count == 11


def test_raw_url_id_does_not_replace_clickable_article_link():
    from app.routers.push import _parse_bild_messages

    cms_id = "6a57392b664e99bc41e93660"
    clickable = "https://www.bild.de/sport"
    parsed = _parse_bild_messages(
        [
            {
                "id": "raw-id-link",
                "sendDate": NOW_TS,
                "headline": "Topspiel",
                "url": clickable,
                "urlId": cms_id,
            }
        ]
    )

    assert parsed[0]["link"] == clickable
    assert parsed[0]["cmsId"] == cms_id


def test_fresh_snapshot_overrides_stale_persisted_sport_and_article_identity():
    import app.notifications.teams as teams_module

    now_ts = _gold_slot_ts()
    cms_id = "0123456789abcdef01234567"
    stale = {
        "message_id": "live-sport-1",
        "ts_num": now_ts - 30 * 60,
        "title": "Finale entschieden",
        "cat": "news",
        "link": "https://www.bild.de/sport/",
    }
    fresh = {
        **stale,
        "cat": "sport",
        "link": f"https://www.bild.de/sport/fussball/finale-{cms_id}.bild.html",
        "cmsId": cms_id,
    }

    merged = teams_module._merge_live_push_history([stale], [fresh])
    assert merged == [fresh]

    candidate = _candidate(
        id=cms_id,
        url=f"https://www.bild.de/sport/fussball/anderer-slug-{cms_id}.bild.html",
        title="Finale entschieden",
        category="news",
    )
    context = build_teams_alert_context(
        [candidate],
        history=merged,
        history_authoritative=True,
        now_ts=now_ts,
        config=_config(excluded_sections=()),
    )
    decision = shouldNotifyTeams(candidate, context, _config(excluded_sections=()))

    assert context["pushesToday"] == 1
    assert context["sportPushesToday"] == 1
    assert decision["section"] == "sport"
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert "gleiche CMS-ID" in decision["livePushComparison"]["reason"]


def test_worker_refresh_uses_complete_snapshot_even_when_upsert_failed():
    import app.notifications.teams as teams_module

    stale = {
        "message_id": "same-push",
        "ts_num": NOW_TS,
        "cat": "news",
        "link": "https://www.bild.de/sport/",
    }
    fresh = {
        **stale,
        "cat": "sport",
        "link": "https://www.bild.de/sport/fussball/artikel-0123456789abcdef01234567.html",
        "cmsId": "0123456789abcdef01234567",
    }
    internal_refresh = {
        "ok": True,
        "source": "live",
        "history_authoritative": False,
        "_snapshot_authoritative": True,
        "_parsed_history": [fresh],
        "synced": 1,
        "db_written": 0,
        "snapshot_age_seconds": 0,
    }
    with (
        patch(
            "app.routers.push._build_refresh_response",
            return_value=internal_refresh,
        ) as refresh,
        patch("app.notifications.teams.push_db_load_all", return_value=[stale]),
    ):
        result = teams_module._refresh_push_history_for_dedup()

    refresh.assert_called_once_with(include_history=True)
    assert result["history_authoritative"] is True
    assert result["history"] == [fresh]
    assert "_snapshot_authoritative" not in result
    assert "_parsed_history" not in result


def test_worker_never_treats_relay_cache_as_final_live_authority():
    import app.notifications.teams as teams_module

    cached = {"message_id": "cache-only", "ts_num": NOW_TS, "link": "https://bild.de/news"}
    refresh_payload = {
        "ok": True,
        "source": "cache->db",
        "history_authoritative": True,
        "_snapshot_authoritative": True,
        "_parsed_history": [cached],
        "synced": 1,
        "db_written": 1,
        "snapshot_age_seconds": 30,
    }
    with (
        patch("app.routers.push._build_refresh_response", return_value=refresh_payload),
        patch("app.notifications.teams.push_db_load_all", return_value=[]),
    ):
        result = teams_module._refresh_push_history_for_dedup()

    assert result["history"] == [cached]
    assert result["history_authoritative"] is False


def test_fresh_snapshot_keeps_other_bild_brands_out_of_live_counts():
    import app.notifications.teams as teams_module

    main = {
        "message_id": "main-bild",
        "ts_num": NOW_TS,
        "link": "https://www.bild.de/news/main-bild",
    }
    other_brands = [
        {
            "message_id": "sportbild",
            "ts_num": NOW_TS,
            "link": "https://www.sportbild.de/fussball/test",
        },
        {
            "message_id": "autobild",
            "ts_num": NOW_TS,
            "link": "https://www.autobild.de/artikel/test",
        },
    ]

    assert teams_module._merge_live_push_history([], [main, *other_brands]) == [main]


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
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert "Score: 78,4/100" in message["text"]
    assert "Empfohlener Versand:" not in message["text"]
    assert message["payload"]["recommendedSendAt"] == "08:23"


def test_final_recommendation_jury_fails_closed_below_the_push_score_floor():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert "harte Fakten-, Aktualitäts-, Titel-, Ruhezeit- und Dublettengates" in (
        message["payload"]["decisionBasis"]
    )


def test_final_recommendation_jury_uses_three_minute_window_for_breaking():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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

    assert timing["mode"] == "mandatory_slot_top1"
    assert timing["windowMinutes"] == 3
    assert timing["sendByLabel"] == "08:27"
    assert message["payload"]["recommendedSendWindow"] == "Sofort senden, ideal bis 08:27 Uhr"


def test_final_recommendation_jury_is_advisory_in_mandatory_slot(tmp_db):
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
    candidate = _candidate(
        url="https://www.bild.de/politik/final-quality-block",
        pubDate=_iso(now_ts - 10 * 60),
    )
    config = _smart_config(min_recommendation_quality=99.0)
    from app.database import push_db_upsert

    push_db_upsert(_history(minutes_since_last_push=60, now_ts=now_ts))

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True},
        ) as send,
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=config,
            now_ts=now_ts,
            history_authoritative=True,
        )

    assert result["ok"] is True
    assert result["sent"] is True
    send.assert_called_once()


def test_send_cycle_considers_expanded_candidate_beyond_dashboard_top_limit(tmp_db):
    from app.database import push_db_upsert

    now_ts = int(
        dt.datetime(2026, 6, 19, 19, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    )
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

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ),
    ):
        result = evaluate_and_send_best_candidate(
            [*weak, strong],
            config=_smart_config(
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
    context["pushesToday"] = 11
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
    assert any("Tagesplan:" in reason for reason in decision["blockingReasons"])


def test_dead_zone_never_fires_and_recovers_on_the_raster():
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

    # Raster-Treue: Auch bei grossem Rueckstand feuert die Totzone nie frei -
    # der Plan wird auf dem Raster verdichtet und beim naechsten Slot aufgeholt.
    assert decision["shouldNotify"] is False
    assert decision["pushPacing"]["deficit"] >= 1.5
    assert "Rueckstand" in decision["pushPacing"]["label"]
    assert decision["slotGate"]["mode"] == "wait"
    assert decision["slotGate"]["recoveryBoosted"] is True
    assert any("Raster" in reason for reason in decision["blockingReasons"])


def test_friday_noon_waits_for_the_1230_raster_decision():
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
    # Raster-Treue: um 12:00 wird bis zur verbindlichen 12:30-Entscheidung
    # gesammelt; der Rueckstand wird dort und an den Folgeslots aufgeholt.
    assert decision["shouldNotify"] is False
    assert breakdown["timeFit"] == 4.0
    assert decision["slotGate"]["mode"] == "wait"
    assert decision["slotGate"]["slot"]["label"] == "12:30"
    assert any("12:30" in reason for reason in decision["blockingReasons"])


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
    # Die verbindliche Mittagspausen-Entscheidung (12:30) ist faellig.
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["slotGate"]["slot"]["label"] == "12:30"
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


def test_verified_breaking_waits_for_the_next_binding_slot():
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
        history=_history(minutes_since_last_push=12, now_ts=now_ts),
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

    assert decision["shouldNotify"] is False
    assert decision["isBreaking"] is True
    assert decision["slotGate"]["mode"] == "wait"
    assert any(
        "nur im 5-Minuten-Fenster" in reason
        for reason in decision["blockingReasons"]
    )
    assert any("Teams-Cooldown aktiv" in reason for reason in decision["blockingReasons"])


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


def test_push_score_over_80_overrides_required_prediction_soft_gate():
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

    assert decision["shouldNotify"] is True
    assert decision["highScoreOverride"]["approved"] is True
    assert any(
        "Belastbare OR-Prognose erforderlich" in reason
        for reason in decision["highScoreOverride"]["waivedBlockers"]
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
    assert result["recommendation"]["text"].startswith("🔵 PUSH-EMPFEHLUNG")


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
            selection_clear_editorial_buffer=40.0,
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
    assert "Teams-Alert-Modell" not in text
    assert "Qualitätsurteil" not in text
    assert "Entscheidungsbasis" not in text
    assert text.count("Warum:") == 1
    assert text.count("Alternative (Platz 2):") == 1
    assert "Keine weitere gültige Alternative verfügbar." in text
    assert "Warum dieser Zeitpunkt:" not in text
    assert len(text) < 1_000


def test_teams_test_message_uses_compact_top1_and_one_alternative():
    now_ts = _gold_slot_ts()

    with patch(
        "app.notifications.teams.send_teams_notification",
        return_value={"ok": True, "status": 202, "attempts": 1},
    ) as send:
        result = send_teams_test_notification(_config(), now_ts=now_ts)

    assert result["ok"] is True
    send.assert_called_once()
    message = send.call_args.args[0]
    text = message["text"]
    assert text.startswith("TESTNACHRICHT – bitte ignorieren")
    assert text.count("Top 1:") == 1
    assert text.count("Alternative (Platz 2):") == 1
    assert "TEST: Alternative zu Top 1" in text
    assert len(text) < 1_000
    assert message["payload"]["alternativeRecommendation"]["rankingPosition"] == 2


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


def test_morning_double_slots_are_mathematically_maximally_spread():
    from app.notifications.teams import _morning_double_minutes

    minutes = _morning_double_minutes()

    # Endpunkte voll ausgereizt: erste Entscheidung 06:00, letzte 08:59.
    assert minutes[0] == 6 * 60
    assert minutes[-1] == 8 * 60 + 59
    # Gleichverteilung von 6 Slots ueber 179 Minuten -> 35/36er-Raster;
    # das maximiert den minimalen Abstand (mathematisches Optimum).
    gaps = [later - earlier for earlier, later in zip(minutes, minutes[1:])]
    assert min(gaps) >= 35
    assert max(gaps) <= 36
    # Genau zwei Entscheidungen je Morgenstunde.
    assert [minute // 60 for minute in minutes] == [6, 6, 7, 7, 8, 8]


def test_smart_schedule_uses_deterministic_berlin_layout_on_monday():
    schedule = build_teams_daily_schedule("2026-07-13", _smart_config())
    labels = [slot["label"] for slot in schedule["slots"]]

    assert schedule["weekday"] == "Montag"
    assert schedule["count"] == 15
    # Morgen-Doppel 06/07/08 (6 Slots gleichverteilt, mathematisch maximal
    # gespreizt), Mittagsslot 12:30, montags Abendstart 17:30 und danach die
    # unveraenderte dynamische Heatmap-Verteilung.
    assert labels == [
        "06:00", "06:36", "07:12", "07:47", "08:23", "08:59",
        "12:30",
        "17:30", "18:34", "19:08", "19:42", "20:17", "20:51", "21:25", "21:59",
    ]
    assert {"10:45", "11:45", "22:45", "23:45"}.isdisjoint(labels)
    assert all(slot["required"] is True for slot in schedule["slots"])
    assert "06:00 + 06:36" in {
        opportunity["label"] for opportunity in schedule["doubleOpportunities"]
    }
    assert "Heute bewusst nachrangig" in schedule["messageHtml"]
    assert len(schedule["messageHtml"].encode("utf-8")) < 28_000


def test_smart_schedule_is_truly_weekday_specific():
    monday = build_teams_daily_schedule("2026-07-13", _smart_config())
    wednesday = build_teams_daily_schedule("2026-07-15", _smart_config())
    monday_labels = {slot["label"] for slot in monday["slots"]}
    wednesday_labels = {slot["label"] for slot in wednesday["slots"]}

    # Der Abend-Hot-Block folgt der Heatmap: Montag 18-21 komplett rot/gelb,
    # Mittwoch nur 20-21 -> andere Slotzeiten plus Reserve-Entscheidungen.
    assert monday_labels != wednesday_labels
    assert {"06:00", "06:36", "07:12", "07:47", "08:23", "08:59", "12:30"}.issubset(monday_labels)
    assert {"06:00", "06:36", "12:30"}.issubset(wednesday_labels)
    assert "10:45" not in wednesday_labels
    assert "23:45" not in wednesday_labels
    assert wednesday["requiredCount"] == 11
    assert wednesday["count"] == 11
    assert wednesday["meetsTargetCoverage"] is True


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
def test_every_weekday_has_11_to_15_binding_runtime_opportunities(date_iso):
    schedule = build_teams_daily_schedule(date_iso, _smart_config())
    required_doubles = [
        item for item in schedule["doubleOpportunities"] if item["requiredForMinimum"]
    ]

    assert 11 <= schedule["runtimeOpportunityCount"] <= 15
    assert schedule["requiredCount"] == schedule["runtimeOpportunityCount"]
    assert schedule["minimumDoubleCount"] == len(required_doubles)
    assert required_doubles
    assert {"06:00", "06:36", "07:12", "07:47", "08:23", "08:59", "12:30"}.issubset(
        {slot["label"] for slot in schedule["slots"]}
    )
    assert schedule["meetsTargetCoverage"] is True


def test_week_plan_can_recommend_15_strong_editorial_events_each_day():
    config = _smart_config()

    for day_number in range(13, 20):
        target_date = dt.date(2026, 7, day_number)
        opportunities = _daily_runtime_opportunities(target_date, config)
        sent = 0
        last_sent_ts = 0
        live_history = _history(
            minutes_since_last_push=8 * 60,
            now_ts=int(opportunities[0]["ts"]),
        )

        assert 11 <= len(opportunities) <= 15
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
                history=list(live_history),
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
            live_history.append(
                {
                    **live_history[0],
                    "message_id": f"live-{day_number}-{index}",
                    "ts_num": now_ts,
                    "title": f"Anderes Live-Thema Nummer {index}",
                    "headline": f"Anderes Live-Thema Nummer {index}",
                    "link": f"https://www.bild.de/news/live-{day_number}-{index}",
                }
            )

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

    while current <= end and teams_alerts_today < 11:
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
            history=_history(minutes_since_last_push=60, now_ts=now_ts),
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

    assert teams_alerts_today == 11
    assert "deadline_fallback" in send_modes


def test_wednesday_first_binding_deadline_is_due_at_0645_and_releases_candidate():
    now_ts = int(dt.datetime(2026, 7, 15, 6, 37, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert decision["slotGate"]["slot"]["label"] == "06:36"
    assert decision["slotGate"]["minimumDouble"] is True
    assert decision["slotGate"]["minimumCommitment"] is True
    assert decision["slotGate"]["dueCount"] == 2
    assert decision["slotGate"]["plannedOpportunityCount"] == 11


def test_due_minimum_slot_uses_raw_push_score_over_secondary_model_floors():
    now_ts = int(dt.datetime(2026, 7, 15, 6, 37, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert len(decision["deadlineFallback"]["secondaryCautions"]) == 1
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
    assert schedule["qualityOpportunityCount"] >= 1
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
    assert decision["slotGate"]["slot"]["label"] == "06:36"
    assert any("Morgenfit" in reason for reason in decision["blockingReasons"])


def test_tuesday_0746_does_not_recommend_the_isolated_lake_death_story():
    now_ts = int(dt.datetime(2026, 7, 14, 7, 48, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    now_ts = int(dt.datetime(2026, 7, 14, 7, 48, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    morning = next(slot for slot in wednesday["slots"] if slot["label"] == "07:47")

    assert morning["topCategory"] == "geld"
    assert "geld" in morning["preferredSections"]


def test_slot_gate_waits_before_the_binding_slot_and_does_not_recover_a_missed_one():
    before = _smart_slot_decision(hour=8, minute=10, pushes_today=4)
    on_plan = _smart_slot_decision(hour=8, minute=24, pushes_today=5)
    missed = _smart_slot_decision(hour=8, minute=46, pushes_today=3)

    assert before["shouldNotify"] is False
    assert before["slotGate"]["mode"] == "wait"
    assert any("Raster-Entscheidung 08:23" in reason for reason in before["blockingReasons"])
    assert on_plan["shouldNotify"] is False
    assert on_plan["slotGate"]["mode"] == "wait"
    assert missed["shouldNotify"] is False
    assert missed["slotGate"]["mode"] == "wait"
    assert missed["slotGate"]["dueCount"] == 5
    assert missed["deadlineFallback"]["approved"] is False
    assert any(
        "nur im 5-Minuten-Fenster" in reason
        for reason in missed["blockingReasons"]
    )


def test_slot_gate_counts_actual_live_pushes_and_respects_min_pause():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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

    assert decision["shouldNotify"] is False
    assert decision["slotGate"]["countBasis"] == "actual_pushes"
    assert decision["slotGate"]["currentCount"] == 15
    assert decision["pushesToday"] == 15
    assert any("Pause seit letztem Push" in reason for reason in decision["blockingReasons"])


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

    fallback = _smart_slot_decision(hour=8, minute=24, candidate=best_available)
    rejected = _smart_slot_decision(hour=8, minute=24, candidate=too_weak)

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
        minute=24,
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
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert selected["competition"]["selectionMargin"] == 9.0
    assert selected["competition"]["selectionMarginPercent"] == 10.7
    assert selected["competition"]["selectionConfidence"] in {"hoch", "mittel", "niedrig"}


def test_morning_double_start_0712_is_a_binding_top_one_decision():
    exceptional = _candidate(
        title="Regierung beschliesst sofort neue Entlastung fuer Millionen",
        score=96.0,
        predictedOR=0.08,
        performanceDrivers=[
            "Aktualitaet: neue Entscheidung",
            "Relevanz: Millionen unmittelbar betroffen",
        ],
    )

    decision = _smart_slot_decision(hour=7, minute=16, candidate=exceptional)

    assert decision["shouldNotify"] is True
    assert decision["slotGate"]["mode"] == "deadline_fallback"
    assert decision["slotGate"]["slot"]["label"] == "07:12"


def test_shortfall_recovery_fires_only_at_raster_times():
    early = _smart_slot_decision(hour=20, minute=2, pushes_today=10)
    at_slot = _smart_slot_decision(hour=20, minute=18, pushes_today=10)
    on_plan = _smart_slot_decision(hour=20, minute=20, pushes_today=13)

    # Vor der Raster-Zeit 20:17 wird trotz Rueckstand gesammelt ...
    assert early["shouldNotify"] is False
    assert early["slotGate"]["mode"] == "wait"
    assert early["slotGate"]["slot"]["label"] == "20:17"
    # ... ab der Raster-Zeit holt die faellige Entscheidung auf ...
    assert at_slot["shouldNotify"] is True
    assert at_slot["slotGate"]["mode"] == "deadline_fallback"
    # ... und im Soll wird bis zum naechsten Slot gewartet.
    assert on_plan["shouldNotify"] is False
    assert on_plan["slotGate"]["projectedShortfall"] == 0
    assert on_plan["slotGate"]["mode"] == "wait"


def test_deadline_fallback_cannot_recommend_live_pushed_article():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
        minute=24,
        candidate=candidate,
        history=pushed_history,
    )

    assert decision["shouldNotify"] is False
    assert decision["deadlineFallback"]["approved"] is False
    assert decision["livePushComparison"]["matched"] is True
    assert decision["livePushComparison"]["matchType"] == "exact_article"
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


def test_context_reads_90_days_for_exact_live_push_deduplication():
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
    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"] == {
        "available": True,
        "authoritative": True,
        "matched": True,
        "matchType": "exact_article",
        "reason": "Bereits live gepusht (gleiche Artikel-URL)",
    }
    assert any("Bereits live gepusht" in reason for reason in decision["blockingReasons"])


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
        hour=19,
        minute=9,
        candidate=confirmed,
        pushes_today=9,
    )
    routine_decision = _smart_slot_decision(
        hour=19,
        minute=9,
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


def test_sport_url_cannot_bypass_event_gate_with_wrong_news_category():
    candidate = _candidate(
        id="miscategorized-sport",
        url="https://www.bild.de/sport/fussball/bayern-training",
        title="Bayern-Stars starten heute ins Training",
        category="news",
        score=95.0,
        predictedOR=0.08,
    )

    decision = _smart_slot_decision(
        hour=19,
        minute=9,
        candidate=candidate,
        pushes_today=9,
    )

    assert decision["section"] == "sport"
    assert decision["shouldNotify"] is False
    assert any(
        "Sport ohne frische bestaetigte" in reason
        for reason in decision["blockingReasons"]
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
    assert first["count"] == 15
    assert second["sent"] is False
    assert second["reason"] == "already_sent"
    assert send.call_count == 1
    payload = send.call_args.args[0]["payload"]
    assert payload["type"] == "push_daily_schedule"
    assert len(payload["slots"]) == 15


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
    assert plan["qualityOpportunityCount"] == 5
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


def test_daily_push_plan_excludes_article_already_pushed_live():
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

    assert all(item["articleUrl"] != pushed["url"] for item in plan["items"])
    assert any(
        item["title"] == pushed["title"] and "Bereits live gepusht" in item["reason"]
        for item in plan["notRecommended"]
    )


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
    assert review["reviewerSetVersion"] == "teams-review-v3"
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

    decision = _smart_slot_decision(hour=8, minute=24, candidate=candidate)
    review = decision["agentReview"]

    assert decision["shouldNotify"] is True
    assert review["approved"] is True
    assert review["hardVetoCount"] == 0
    assert review["cautionCount"] >= 1
    assert review["evidenceApprovalCount"] >= review["requiredEvidenceApprovals"]
    assert review["mainCounterargument"]


def test_agent_network_hard_vetoes_exact_live_push_duplicate_at_deadline():
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
        minute=24,
        candidate=candidate,
        history=history,
    )
    review = decision["agentReview"]

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["matched"] is True
    assert review["approved"] is False
    assert review["hardVetoCount"] >= 1
    assert any(
        item["agent"] == "Live-Push-Vergleich"
        and item["verdict"] == "veto"
        and item["hardVeto"]
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


def test_agent_network_fails_closed_when_live_push_dedup_cannot_load():
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

    assert decision["shouldNotify"] is False
    assert decision["livePushComparison"]["available"] is False
    assert decision["agentReview"]["approved"] is False
    assert any(
        item["agent"] == "Kontext-Integritaet"
        and item["verdict"] == "veto"
        and item["hardVeto"]
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
def test_dispatch_blocks_article_that_was_live_pushed_after_selection(
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
    call_order: list[str] = []

    def _slot_review(*_args, **_kwargs):
        call_order.append("slot_review")
        return {"available": False, "fitsNow": True, "reason": ""}

    def _final_refresh():
        call_order.append("final_refresh")
        return {"history_authoritative": True, "history": newly_pushed}

    with (
        patch(
            "app.notifications.teams.push_db_load_all",
            return_value=initial_history,
        ),
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
            side_effect=_final_refresh,
        ) as refresh,
        patch("app.notifications.teams._llm_slot_fit_review", side_effect=_slot_review),
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
            refresh_live_history_before_dispatch=True,
        )

    refresh.assert_called_once_with()
    assert call_order == ["slot_review", "final_refresh"]
    assert result["sent"] is False
    assert result["reason"] == "live_push_duplicate_blocked"
    send.assert_not_called()
    assert result["livePushDedup"]["livePushComparison"] == {
        "available": True,
        "authoritative": True,
        "matched": True,
        "matchType": "exact_article",
        "reason": "Bereits live gepusht (gleiche Artikel-URL)",
    }


def test_mandatory_slot_reranks_to_runner_up_after_final_live_duplicate_race(tmp_db):
    config = _smart_config(agent_review_enabled=False)
    now_ts = int(
        _daily_runtime_opportunities(dt.date(2026, 6, 19), config)[-1]["ts"]
    )
    top = _candidate(
        id="race-top",
        url="https://www.bild.de/news/race-top",
        title="Topmeldung wird parallel live gepusht",
        score=96.0,
        pubDate=_iso(now_ts - 5 * 60),
    )
    runner_up = _candidate(
        id="race-runner-up",
        url="https://www.bild.de/news/race-runner-up",
        title="Zweitstaerkste Meldung bleibt ungepusht",
        score=91.0,
        pubDate=_iso(now_ts - 5 * 60),
    )
    initial_history = _history(minutes_since_last_push=60, now_ts=now_ts)
    refreshed_history = [
        *initial_history,
        {
            "message_id": "parallel-live-push",
            "ts_num": now_ts,
            "title": top["title"],
            "headline": top["title"],
            "cat": "news",
            "link": top["url"],
        },
    ]

    with (
        patch("app.notifications.teams.time.time", return_value=now_ts),
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
            return_value={"history_authoritative": True, "history": refreshed_history},
        ) as refresh,
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ) as send,
    ):
        result = evaluate_and_send_best_candidate(
            [top, runner_up],
            config=config,
            now_ts=now_ts,
            history=initial_history,
            history_authoritative=True,
            refresh_live_history_before_dispatch=True,
        )

    assert result["sent"] is True
    assert result["candidateId"] == runner_up["url"]
    assert result["raceFallbackFromCandidateId"] == top["url"]
    assert refresh.call_count == 2
    assert send.call_count == 1
    assert send.call_args.args[0]["payload"]["articleUrl"] == runner_up["url"]


def test_dispatch_blocks_when_fresh_live_push_history_cannot_be_reloaded(tmp_db):
    """Der finale Dispatch stoppt, wenn die Live-Historie nicht ladbar ist."""
    now_ts = _gold_slot_ts()
    candidate = _candidate(
        id="dispatch-history-outage",
        url="https://www.bild.de/news/dispatch-history-outage",
        title="Netzbetreiber melden Stoerung: Stromausfall trifft fuenf Grossstaedte",
        category="news",
        score=94.0,
        predictedOR=0.08,
        pubDate=_iso(now_ts - 10 * 60),
        recommendedText="Stromausfall: Was die Stoerung fuer fuenf Grossstaedte bedeutet",
    )
    initial_history = _history(minutes_since_last_push=60, now_ts=now_ts)

    with (
        patch(
            "app.notifications.teams.push_db_load_all",
            return_value=initial_history,
        ),
        patch(
            "app.notifications.teams._refresh_push_history_for_dedup",
            return_value={"history_authoritative": False, "history": []},
        ),
        patch(
            "app.notifications.teams.send_teams_notification",
            return_value={"ok": True, "status": 200},
        ) as send,
    ):
        result = evaluate_and_send_best_candidate(
            [candidate],
            config=_config(agent_review_enabled=False),
            now_ts=now_ts,
            history_authoritative=True,
            refresh_live_history_before_dispatch=True,
        )

    assert result["sent"] is False
    assert result["reason"] == "live_push_dedup_unavailable"
    send.assert_not_called()


def test_dispatch_waits_after_recent_unrelated_live_push(tmp_db):
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

    assert result["sent"] is False
    assert result["reason"] == "no_candidate"
    send.assert_not_called()


def test_empty_cycle_reports_aggregate_minimum_pacing_diagnostics(tmp_db):
    now_ts = int(dt.datetime(2026, 7, 15, 6, 37, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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
    assert diagnostics["plannedOpportunityCount"] == 11
    assert diagnostics["dueOpportunityCount"] == 2
    assert diagnostics["teamsAlertsToday"] == 0
    assert diagnostics["scoreEligibleCandidates"] == 0
    assert diagnostics["projectedShortfall"] == 2
    assert diagnostics["blockerCategories"]["live_push_duplicate"] == 1


def test_teams_payload_shows_consensus_and_counterargument_without_raw_context():
    candidate = _candidate(score=91.0, predictedOR=0.072)
    context = _context(candidate)
    decision = shouldNotifyTeams(candidate, context, _config())

    message = buildTeamsPushRecommendation(candidate, context, decision, _config())
    payload = message["payload"]

    review = message["_agentReview"]
    assert review["approved"] is True
    assert f"{review['agentCount']} lokale Checks" in payload["agentSummary"]
    assert (
        f"Evidenz {review['evidenceApprovalCount']}/{review['evidenceReviewerCount']}"
        in payload["agentSummary"]
    )
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
    now_ts = int(dt.datetime(2026, 7, 13, 8, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
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


def test_live_push_pause_and_teams_cooldown_share_the_45_minute_edge():
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

    decision_44 = shouldNotifyTeams(candidate, actual_44, config)
    assert decision_44["shouldNotify"] is False
    assert any("Pause seit letztem Push" in r for r in decision_44["blockingReasons"])
    assert shouldNotifyTeams(candidate, actual_45, config)["shouldNotify"] is True
    assert shouldNotifyTeams(candidate, teams_44, config)["shouldNotify"] is False
    assert shouldNotifyTeams(candidate, teams_45, config)["shouldNotify"] is True


def test_monday_evening_gate_opens_at_1730_not_before():
    before = _smart_slot_decision(hour=17, minute=29, pushes_today=7)
    due = _smart_slot_decision(hour=17, minute=30, pushes_today=7)

    assert before["shouldNotify"] is False
    assert before["slotGate"]["slot"]["label"] == "17:30"
    assert due["shouldNotify"] is True
    assert due["slotGate"]["mode"] == "deadline_fallback"


def test_runtime_double_slots_equal_the_advertised_monday_slots():
    schedule = build_teams_daily_schedule("2026-07-13", _smart_config())
    advertised = {int(item["hour"]) for item in schedule["doubleOpportunities"]}
    unadvertised = _smart_slot_decision(hour=17, minute=2, pushes_today=10)
    advertised_catchup = _smart_slot_decision(hour=20, minute=18, pushes_today=10)

    # Morgen-Doppel 6/7/8 plus die Abendstunden, in denen nach dem 17:30-Lead-in
    # weiterhin zwei Entscheidungen liegen.
    assert advertised == {6, 7, 8, 19, 20, 21}
    assert unadvertised["slotGate"]["doubleOpportunity"] is False
    assert unadvertised["shouldNotify"] is False
    assert advertised_catchup["shouldNotify"] is True
    assert advertised_catchup["slotGate"]["mode"] == "deadline_fallback"


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


def test_mandatory_slot_selects_raw_top1_without_score_section_or_quality_gates():
    config = _smart_config(
        require_internal_score_api=True,
        allowed_sections=("politik",),
        excluded_sections=("ratgeber", "sport"),
        min_score=99.0,
        min_alert_score=99.0,
    )
    now_ts = int(
        _daily_runtime_opportunities(dt.date(2026, 6, 19), config)[-1]["ts"]
    )
    top1 = _candidate(
        id="mandatory-top1",
        url="https://www.bild.de/ratgeber/mandatory-top1",
        title="Ariana Grande legt eine Auszeit ein – Fans sorgen sich nach Video",
        category="ratgeber",
        score=41.0,
        scoreSource="internal_score_api",
        predictedOR=0.01,
        pubDate=_iso(now_ts - 5 * 60),
    )
    runner_up = _candidate(
        id="mandatory-runner-up",
        url="https://www.bild.de/sport/mandatory-runner-up",
        title="FC Bayern gewinnt das Abendspiel",
        category="sport",
        score=40.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    context = build_teams_alert_context(
        [runner_up, top1],
        history=_history(now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=now_ts - 60,
        teams_alerts_today=15,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )

    result = evaluate_teams_alert_candidates([runner_up, top1], context, config)

    assert result["selectedCandidateId"] == top1["url"]
    selected = next(
        item["decision"]
        for item in result["decisions"]
        if item["candidate"]["id"] == top1["id"]
    )
    assert selected["mandatorySlotTop1"] is True
    assert selected["shouldNotify"] is True
    assert selected["competition"]["selectionMetric"] == "internal_push_balancer_score"
    assert selected["competition"]["runnerUp"] == {
        "articleTitle": runner_up["title"],
        "articleUrl": runner_up["url"],
        "category": runner_up["category"],
        "pushScore": runner_up["score"],
        "rankingPosition": 2,
    }

    message = buildTeamsPushRecommendation(top1, context, selected, config)
    assert top1["url"] in message["text"]
    assert runner_up["url"] in message["text"]
    assert message["text"].count("Alternative (Platz 2):") == 1
    assert message["payload"]["alternativeRecommendation"]["articleUrl"] == runner_up["url"]


def test_mandatory_slot_forces_best_sport_only_when_daily_quota_requires_it():
    config = _smart_config(
        require_internal_score_api=True,
        excluded_sections=("sport",),
    )
    slots = _daily_runtime_opportunities(dt.date(2026, 8, 3), config)
    now_ts = int(slots[-6]["ts"])
    news = _candidate(
        id="quota-news-top",
        url="https://www.bild.de/politik/quota-news-top",
        title="Bundestag beschliesst neues Entlastungspaket",
        category="politik",
        score=92.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    sport_high = _candidate(
        id="quota-sport-high",
        url="https://www.bild.de/sport/quota-sport-high",
        title="FC Bayern gewinnt das Spitzenspiel",
        category="sport",
        score=78.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    sport_low = _candidate(
        id="quota-sport-low",
        url="https://www.bild.de/sport/quota-sport-low",
        title="Borussia Dortmund gewinnt das Abendspiel",
        category="sport",
        score=75.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )

    def evaluate(mix):
        context = build_teams_alert_context(
            [sport_low, news, sport_high],
            history=_history(now_ts=now_ts),
            history_authoritative=True,
            alert_state={},
            last_teams_alert_ts=0,
            teams_alerts_today=mix["sent"],
            teams_recommendation_mix_today=mix,
            recent_alerts=[],
            now_ts=now_ts,
            config=config,
        )
        return evaluate_teams_alert_candidates([sport_low, news, sport_high], context, config)

    required = evaluate({"available": True, "sent": 11, "sport": 0})
    assert required["selectedCandidateId"] == sport_high["url"]
    assert required["mandatorySportQuota"]["required"] is True
    assert required["mandatorySportQuota"]["applied"] is True
    selected = next(
        item["decision"]
        for item in required["decisions"]
        if item["candidate"]["id"] == sport_high["id"]
    )
    assert selected["competition"]["selectionMetric"] == (
        "mandatory_sport_quota_then_internal_score"
    )
    assert selected["competition"]["runnerUp"]["articleUrl"] == news["url"]

    not_yet_required = evaluate({"available": True, "sent": 11, "sport": 1})
    assert not_yet_required["selectedCandidateId"] == news["url"]
    assert not_yet_required["mandatorySportQuota"]["required"] is False

    unavailable = evaluate({"available": False, "sent": 0, "sport": 0})
    assert unavailable["selectedCandidateId"] == news["url"]
    assert unavailable["mandatorySportQuota"]["applied"] is False


def test_mandatory_slot_shows_best_opposite_ressort_as_alternative():
    """User-Vorgabe: Top Nicht-Sport -> Alternative Sport (und umgekehrt).

    Der echte Platz 2 (Unterhaltung) bleibt Margin-Basis, aber als Alternative
    erscheint der beste gegenlaeufige Kandidat (Sport, Rang 3)."""
    config = _smart_config(require_internal_score_api=True)
    now_ts = int(
        _daily_runtime_opportunities(dt.date(2026, 6, 19), config)[-1]["ts"]
    )
    top1 = _candidate(
        id="api-rank-1",
        url="https://www.bild.de/news/api-rank-1",
        title="Topmeldung des Abends",
        score=93.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    runner_up = _candidate(
        id="api-rank-2",
        url="https://www.bild.de/unterhaltung/api-rank-2",
        title="Zweitstärkste Meldung des Abends",
        category="unterhaltung",
        score=89.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    third = _candidate(
        id="api-rank-3",
        url="https://www.bild.de/sport/api-rank-3",
        title="Drittstärkste Meldung des Abends",
        category="sport",
        score=87.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    context = build_teams_alert_context(
        [third, top1, runner_up],
        history=_history(now_ts=now_ts),
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=now_ts - 60,
        teams_alerts_today=10,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )

    result = evaluate_teams_alert_candidates([third, top1, runner_up], context, config)
    selected = next(
        item["decision"]
        for item in result["decisions"]
        if item["candidate"]["id"] == top1["id"]
    )
    message = buildTeamsPushRecommendation(top1, context, selected, config)

    assert result["selectedCandidateId"] == top1["url"]
    # Alternative = bester Sport-Kandidat, weil die Top-Meldung Nicht-Sport ist.
    assert selected["competition"]["runnerUp"]["articleUrl"] == third["url"]
    # Margin-Basis bleibt der echte Platz 2.
    assert selected["competition"]["runnerUpScore"] == runner_up["score"]
    assert third["url"] in message["text"]
    assert runner_up["url"] not in message["text"]
    assert third["title"] in message["payload"]["messageHtml"]
    assert runner_up["title"] not in message["payload"]["messageHtml"]
    assert message["text"].count("Alternative (Platz 2):") == 1


def test_sport_top_gets_non_sport_alternative_and_vice_versa():
    """Sport-Top -> Nicht-Sport-Alternative; ohne gegenlaeufigen Kandidaten
    bleibt die Alternative leer statt die Ressort-Vorgabe zu verletzen."""
    config = _smart_config(require_internal_score_api=True)
    now_ts = int(
        _daily_runtime_opportunities(dt.date(2026, 6, 19), config)[-1]["ts"]
    )

    def _mk(id_, section, score):
        return _candidate(
            id=id_,
            url=f"https://www.bild.de/{section}/{id_}",
            title=f"Kandidat {id_}",
            category=section,
            score=score,
            scoreSource="internal_score_api",
            pubDate=_iso(now_ts - 5 * 60),
        )

    def _context(candidates):
        return build_teams_alert_context(
            candidates,
            history=_history(now_ts=now_ts),
            history_authoritative=True,
            alert_state={},
            last_teams_alert_ts=now_ts - 60,
            teams_alerts_today=10,
            recent_alerts=[],
            now_ts=now_ts,
            config=config,
        )

    def _evaluate(candidates):
        result = evaluate_teams_alert_candidates(
            candidates, _context(candidates), config
        )
        selected = next(
            item["decision"]
            for item in result["decisions"]
            if item["decision"].get("shouldNotify")
        )
        return result, selected

    # Sport-Top: die Alternative muss Nicht-Sport sein (Rang 3 statt Rang 2).
    sport_top = _mk("opp-sport-top", "sport", 93.0)
    sport_second = _mk("opp-sport-2", "sport", 90.0)
    news_third = _mk("opp-news-3", "politik", 87.0)
    result, selected = _evaluate([news_third, sport_top, sport_second])
    assert result["selectedCandidateId"] == sport_top["url"]
    assert selected["competition"]["runnerUp"]["articleUrl"] == news_third["url"]

    # Nur Sport-Kandidaten: keine gegenlaeufige Alternative verfuegbar -> leer.
    result, selected = _evaluate([sport_top, sport_second])
    assert result["selectedCandidateId"] == sport_top["url"]
    assert selected["competition"]["runnerUp"] == {}
    message = buildTeamsPushRecommendation(
        sport_top, _context([sport_top, sport_second]), selected, config
    )
    assert (
        "Keine weitere gültige Alternative verfügbar"
        in message["payload"]["messageHtml"]
    )


def test_mandatory_slot_skips_exact_live_duplicate_and_promo_then_uses_next_rank():
    config = _smart_config(require_internal_score_api=True)
    now_ts = int(
        _daily_runtime_opportunities(dt.date(2026, 6, 19), config)[-1]["ts"]
    )
    already_live = _candidate(
        id="already-live",
        url="https://www.bild.de/news/already-live",
        title="Topmeldung ist schon live gepusht",
        score=99.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    promo = _candidate(
        id="promo",
        url="https://www.bild.de/sonstiges/bildplus-gewinnspiele-aktionen/promo",
        title="Tech-Highlight: einen von 15 Kalendern gewinnen!",
        score=98.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    valid = _candidate(
        id="valid-next",
        url="https://www.bild.de/unterhaltung/valid-next",
        title="Ariana Grande legt eine Auszeit ein – Fans sorgen sich nach Video",
        score=45.0,
        scoreSource="internal_score_api",
        pubDate=_iso(now_ts - 5 * 60),
    )
    context = build_teams_alert_context(
        [already_live, promo, valid],
        history=[
            {
                "message_id": "live-top1",
                "ts_num": now_ts - 60,
                "title": already_live["title"],
                "cat": "news",
                "link": already_live["url"],
            }
        ],
        history_authoritative=True,
        alert_state={},
        last_teams_alert_ts=0,
        teams_alerts_today=0,
        recent_alerts=[],
        now_ts=now_ts,
        config=config,
    )

    result = evaluate_teams_alert_candidates([already_live, promo, valid], context, config)

    assert result["selectedCandidateId"] == valid["url"]


def test_recommendation_slot_claim_allows_exactly_one_success(tmp_db):
    from app.teams_slot_claims import (
        teams_recommendation_slot_record,
        teams_recommendation_slot_try_claim,
    )

    slot_ts = _gold_slot_ts()
    first = teams_recommendation_slot_try_claim(
        slot_ts,
        article_key="https://www.bild.de/news/first",
        now_ts=slot_ts,
    )
    concurrent = teams_recommendation_slot_try_claim(
        slot_ts,
        article_key="https://www.bild.de/news/second",
        now_ts=slot_ts + 1,
    )
    teams_recommendation_slot_record(
        slot_ts,
        article_key="https://www.bild.de/news/first",
        status="sent",
        now_ts=slot_ts + 2,
    )
    repeated = teams_recommendation_slot_try_claim(
        slot_ts,
        article_key="https://www.bild.de/news/second",
        now_ts=slot_ts + 400,
    )

    assert first["claimed"] is True
    assert concurrent == {"claimed": False, "reason": "slot_send_in_progress"}
    assert repeated == {"claimed": False, "reason": "slot_already_sent"}
