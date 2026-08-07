"""Synthetic contract tests for the scheduled Power Automate Teams hand-off."""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app import auth, database
from app.main import app
from app.notifications.teams import candidate_key
from app.teams_slot_claims import (
    teams_recommendation_slot_get,
    teams_recommendation_slot_record,
    teams_recommendation_slot_try_claim,
)


POWER_AUTOMATE_KEY = "synthetic-power-automate-key"
HEADERS = {"X-Power-Automate-Key": POWER_AUTOMATE_KEY}
BERLIN = ZoneInfo("Europe/Berlin")
SLOT_TS = int(dt.datetime(2026, 8, 3, 12, 30, tzinfo=BERLIN).timestamp())

client = TestClient(app, raise_server_exceptions=True)


def test_weekend_morning_slots_start_two_hours_later():
    import app.routers.power_automate as power_automate

    saturday = dt.date(2026, 8, 8)
    sunday = dt.date(2026, 8, 9)
    expected = (
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
    )

    assert power_automate.power_automate_slot_labels_for_date(saturday) == expected
    assert power_automate.power_automate_slot_labels_for_date(sunday) == expected
    assert power_automate.power_automate_slot_labels_for_date(
        dt.date(2026, 8, 10)
    )[0] == "06:00"


def _synthetic_candidates(
    now_ts: int,
    *,
    top_category: str = "news",
    alternative_category: str = "sport",
) -> tuple[dict, dict]:
    published_at = dt.datetime.fromtimestamp(now_ts - 600, dt.timezone.utc).isoformat()
    top = {
        "id": "synthetic-news-top",
        "url": f"https://www.bild.de/{top_category}/synthetic-news-top",
        "title": "Bund beschliesst synthetisches Hilfspaket",
        "category": top_category,
        "score": 91.4,
        "scoreSource": "internal_score_api",
        "predictedOR": 0.061,
        "pubDate": published_at,
    }
    alternative = {
        "id": "synthetic-sport-alternative",
        "url": f"https://www.bild.de/{alternative_category}/synthetic-sport-alternative",
        "title": "Verein bestaetigt synthetischen Transfer",
        "category": alternative_category,
        "score": 88.2,
        "scoreSource": "internal_score_api",
        "predictedOR": 0.058,
        "pubDate": published_at,
    }
    return top, alternative


def _patch_successful_claim(
    monkeypatch,
    *,
    now_ts: int,
    top_category: str = "news",
    alternative_category: str = "sport",
    include_alternative: bool = True,
) -> tuple[dict, dict]:
    import app.routers.power_automate as power_automate

    top, alternative = _synthetic_candidates(
        now_ts,
        top_category=top_category,
        alternative_category=alternative_category,
    )
    config = replace(
        power_automate.TeamsAlertConfig(),
        enabled=True,
        require_internal_score_api=True,
        slot_gate_enabled=True,
    )
    decision = {
        "candidateId": top["url"],
        "shouldNotify": True,
        "score": top["score"],
        "summary": "Verbindlicher Push-Balancer-Top-1 im festen Slot",
    }
    alternative_decision = {
        "candidateId": alternative["url"],
        "shouldNotify": False,
        "score": alternative["score"],
        "blockingReasons": [
            "Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"
        ],
    }
    message_html = (
        "<h2>🔵 PUSH-EMPFEHLUNG</h2>"
        "<p><strong>Top 1:</strong> Bund beschliesst synthetisches Hilfspaket</p>"
    )
    message = {
        "_dispatchApproved": True,
        "_slotGateApproved": True,
        "payload": {
            "articleTitle": top["title"],
            "articleUrl": top["url"],
            "category": top["category"],
            "pushScore": top["score"],
            "alternativeRecommendation": (
                {
                    "articleTitle": alternative["title"],
                    "articleUrl": alternative["url"],
                    "category": alternative["category"],
                    "pushScore": alternative["score"],
                }
                if include_alternative
                else {}
            ),
            "messageHtml": message_html,
        },
    }

    monkeypatch.setattr(power_automate.time, "time", lambda: now_ts)
    monkeypatch.setattr(power_automate, "TeamsAlertConfig", lambda: config)
    monkeypatch.setattr(
        power_automate,
        "build_articles_payload",
        lambda **_kwargs: {
            "articles": [top, alternative] if include_alternative else [top]
        },
    )
    monkeypatch.setattr(
        power_automate,
        "_refresh_push_history_for_dedup",
        lambda: {"history": [], "history_authoritative": True},
    )
    monkeypatch.setattr(
        power_automate,
        "_dispatch_live_push_comparison",
        lambda *_args, **_kwargs: {"blocked": False},
    )
    monkeypatch.setattr(
        power_automate,
        "_memory_eligible_candidates",
        lambda candidates, **_kwargs: (candidates, {}),
    )
    monkeypatch.setattr(
        power_automate,
        "build_teams_alert_context",
        lambda candidates, **_kwargs: {"nowTs": now_ts},
    )
    monkeypatch.setattr(
        power_automate,
        "evaluate_teams_alert_candidates",
        lambda *_args, **_kwargs: {
            "selectedCandidate": top,
            "decisions": [
                {"candidate": top, "decision": decision},
                *(
                    [{"candidate": alternative, "decision": alternative_decision}]
                    if include_alternative
                    else []
                ),
            ],
        },
    )
    monkeypatch.setattr(
        power_automate,
        "build_teams_push_recommendation",
        lambda *_args, **_kwargs: message,
    )
    return top, alternative


def test_claim_requires_dedicated_auth_and_never_allows_caching(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    response = client.post(
        "/api/v1/power-automate/teams/claim",
        json={"requestId": "synthetic-unauthorized-run"},
    )

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["vary"] == "X-Power-Automate-Key"


def test_scheduled_message_uses_the_five_highest_valid_push_scores():
    import app.routers.power_automate as power_automate

    decisions = []
    for index, score in enumerate((82.1, 91.4, 76.0, 88.2, 79.5, 84.7), start=1):
        candidate = {
            "title": f"Synthetische Meldung {index}",
            "url": f"https://www.bild.de/news/synthetic-{index}",
            "score": score,
            "pubDate": f"2026-08-03T{index + 8:02d}:15:00+02:00",
        }
        decisions.append(
            {
                "candidate": candidate,
                "decision": {
                    "score": score,
                    "blockingReasons": (
                        []
                        if index == 2
                        else [
                            "Staerkerer Kandidat vorhanden: vollstaendig geprueftes Feld"
                        ]
                    ),
                },
            }
        )

    recommendations = power_automate._scheduled_recommendations(
        {"decisions": decisions}
    )
    message_html = power_automate._scheduled_message_html(recommendations)

    assert [item["pushScore"] for item in recommendations] == [
        91.4,
        88.2,
        84.7,
        82.1,
        79.5,
    ]
    assert message_html.count("<strong>Top ") == 5
    assert "<strong>Top 5:</strong>" in message_html
    assert message_html.count("</p><br><p>") == 5
    assert "(03.08.2026, 10:15 Uhr)" in message_html
    assert "(03.08.2026, 14:15 Uhr)" in message_html


def test_scheduled_candidates_exclude_articles_already_live_pushed():
    import app.routers.power_automate as power_automate

    top, alternative = _synthetic_candidates(SLOT_TS)
    same_title_new_article = {
        **top,
        "id": "synthetic-news-follow-up",
        "url": "https://www.bild.de/news/synthetic-news-follow-up",
    }
    eligible = power_automate._exclude_already_live_pushed_articles(
        [top, alternative, same_title_new_article],
        history=[
            {
                "url": top["url"],
                "title": top["title"],
                "ts_num": SLOT_TS - 60,
            }
        ],
        now_ts=SLOT_TS,
        config=power_automate.TeamsAlertConfig(),
    )

    assert [candidate["url"] for candidate in eligible] == [
        alternative["url"],
        same_title_new_article["url"],
    ]


def test_claim_can_prepare_two_minutes_before_the_official_slot(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS - 120
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-early-preparation-run"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["scheduledAt"] == "2026-08-03T12:30:00+02:00"
    assert response.json()["scheduledAtUtc"] == "2026-08-03T10:30:00Z"


def test_claim_does_not_prepare_more_than_two_minutes_early(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS - 121
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-too-early-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "outside_window"}


def test_headline_command_returns_three_v14_pairs(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(
        power_automate,
        "_headline_article_context",
        lambda _article_id: {
            "url": "https://www.bild.de/politik/synthetischer-artikel",
            "title": "Bund beschliesst synthetisches Hilfspaket",
            "text": "Das Hilfspaket gilt ab Montag bundesweit.",
            "category": "politik",
        },
    )
    candidates = [
        {
            "titel": "Bund startet neues Hilfspaket",
            "zeile2": "Ab Montag gilt die neue Hilfe",
            "ansatz": "FAKT",
        },
        {
            "titel": "Neue Hilfe erreicht Millionen",
            "zeile2": "Bund setzt Paket am Montag um",
            "ansatz": "BETROFFENHEIT",
        },
        {
            "titel": "Hilfspaket gilt ab Montag",
            "zeile2": "Diese Haushalte profitieren",
            "ansatz": "FOLGE",
        },
    ]
    monkeypatch.setattr(
        "app.routers.misc._build_push_title_response",
        lambda _request: {
            "gewinner": {
                **candidates[0],
                "warum_dieser": "Kern und Folge stehen sofort fest.",
            },
            "alle_kandidaten": {"v1.4": candidates},
            "reasoning": "Kern und Folge stehen sofort fest.",
            "stufe": 2,
            "stufe_begruendung": "Entscheidung hat Zeit",
        },
    )

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "0123456789abcdef01234567"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["ready"] is True
    assert len(payload["suggestions"]) == 3
    assert payload["suggestions"][0] == {
        "type": "FAKT",
        "headline": "Bund startet neues Hilfspaket",
        "line2": "Ab Montag gilt die neue Hilfe",
    }
    assert "Headline-Vorschläge" in payload["messageHtml"]
    assert "bitte vor Versand prüfen" in payload["messageHtml"]


def test_headline_command_requires_auth_and_rejects_invalid_ids(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    unauthorized = client.post(
        "/api/v1/power-automate/teams/headline",
        json={"articleId": "0123456789abcdef01234567"},
    )
    invalid = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "not-an-id"},
    )

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422


def test_headline_command_extracts_one_id_from_teams_html(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    context_lookup = Mock(return_value=None)
    monkeypatch.setattr(power_automate, "_headline_article_context", context_lookup)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={
            "articleId": (
                "<p><span>/headline&nbsp;</span>"
                "0123456789ABCDEF01234567</p>"
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "article_not_found"
    context_lookup.assert_called_once_with("0123456789abcdef01234567")


def test_headline_command_rejects_ambiguous_teams_content(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={
            "articleId": (
                "/headline 0123456789abcdef01234567 "
                "fedcba987654321001234567"
            )
        },
    )

    assert response.status_code == 422


def test_headline_context_falls_back_to_complete_sitemap_lookup(monkeypatch):
    import app.routers.headline as headline
    import app.routers.power_automate as power_automate
    from app.cms.url_api import UrlApiNotConfigured

    monkeypatch.setattr(
        power_automate,
        "build_articles_payload",
        lambda **_kwargs: {"articles": []},
    )

    def unavailable_url_api(_article_id: str):
        raise UrlApiNotConfigured

    monkeypatch.setattr(power_automate, "get_canonical_article_url", unavailable_url_api)
    monkeypatch.setattr(
        headline,
        "resolve_headline_article",
        lambda article_id: {
            "articleId": article_id,
            "url": "https://www.bild.de/politik/synthetischer-artikel",
            "title": "Bund beschließt synthetisches Hilfspaket",
            "category": "politik",
            "contentType": "editorial",
        },
    )

    context = power_automate._headline_article_context(
        "0123456789abcdef01234567"
    )

    assert context == {
        "url": "https://www.bild.de/politik/synthetischer-artikel",
        "title": "Bund beschließt synthetisches Hilfspaket",
        "text": "",
        "category": "politik",
    }


def test_headline_command_returns_no_op_when_article_is_unknown(monkeypatch):
    import app.routers.power_automate as power_automate

    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    monkeypatch.setattr(power_automate, "_headline_article_context", lambda _article_id: None)

    response = client.post(
        "/api/v1/power-automate/teams/headline",
        headers=HEADERS,
        json={"articleId": "0123456789abcdef01234567"},
    )

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["reason"] == "article_not_found"
    assert "Artikel nicht gefunden" in response.json()["messageHtml"]


def test_claim_fails_closed_when_dedicated_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", "")
    monkeypatch.setattr(auth.config, "PUSH_TEAMS_WEBHOOK_URL", "")

    response = client.post(
        "/api/v1/power-automate/teams/claim",
        headers=HEADERS,
        json={"requestId": "synthetic-disabled-run"},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_claim_returns_only_the_minimal_top_opposite_and_html_contract(
    monkeypatch,
    tmp_db,
):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, alternative = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-contract-run"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert set(payload) == {
        "ready",
        "slotId",
        "scheduledAt",
        "scheduledAtUtc",
        "expiresAt",
        "top",
        "alternative",
        "messageHtml",
    }
    assert payload["ready"] is True
    assert payload["slotId"] == f"teams-recommendation-{SLOT_TS}"
    assert payload["scheduledAt"] == "2026-08-03T12:30:00+02:00"
    assert payload["scheduledAtUtc"] == "2026-08-03T10:30:00Z"
    assert payload["expiresAt"] == "2026-08-03T12:35:00+02:00"
    assert payload["top"] == {
        "title": top["title"],
        "url": top["url"],
        "category": "news",
        "pushScore": 91.4,
        "isSport": False,
    }
    assert payload["alternative"] == {
        "title": alternative["title"],
        "url": alternative["url"],
        "category": "sport",
        "pushScore": 88.2,
        "isSport": True,
    }
    assert payload["messageHtml"].startswith(
        "<h2>🔵 JETZT MÜSSEN (!) WIR PUSHEN</h2>"
    )
    assert "Das sind meine 5 Empfehlungen" in payload["messageHtml"]
    assert (
        '<a href="https://editorial.one/push-balancer/bild/kandidaten">Push Balancer</a>'
        in payload["messageHtml"]
    )
    assert (
        f'<strong>Top 1:</strong> <a href="{top["url"]}">{top["title"]}</a>'
        in payload["messageHtml"]
    )
    assert "(03.08.2026, 12:20 Uhr)" in payload["messageHtml"]
    assert "<strong>Score:</strong> 91,4/100" in payload["messageHtml"]
    assert (
        f'<strong>Top 2:</strong> <a href="{alternative["url"]}">'
        f'{alternative["title"]}</a>' in payload["messageHtml"]
    )
    assert "<strong>Score:</strong> 88,2/100" in payload["messageHtml"]
    assert "webhook" not in response.text.casefold()
    assert "power-automate-key" not in response.text.casefold()


def test_claim_reuses_its_authoritative_snapshot_for_final_dedup(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    history = [{"message_id": "synthetic-live-push", "ts_num": now_ts - 60}]
    refresh_calls = 0
    comparison_kwargs: dict = {}

    def refresh_once():
        nonlocal refresh_calls
        refresh_calls += 1
        return {"history": history, "history_authoritative": True}

    def compare_once(*_args, **kwargs):
        comparison_kwargs.update(kwargs)
        return {"blocked": False}

    monkeypatch.setattr(power_automate, "_refresh_push_history_for_dedup", refresh_once)
    monkeypatch.setattr(power_automate, "_dispatch_live_push_comparison", compare_once)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-single-refresh-run"},
        )

    assert response.status_code == 200
    assert refresh_calls == 1
    assert comparison_kwargs["history"] is history
    assert comparison_kwargs["comparison_authoritative"] is True
    assert comparison_kwargs["refresh_live_history"] is False


def test_claim_supports_sport_top_with_non_sport_alternative(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(
        monkeypatch,
        now_ts=now_ts,
        top_category="sport",
        alternative_category="politik",
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-sport-top-run"},
        )

    assert response.status_code == 200
    assert response.json()["top"]["isSport"] is True
    assert response.json()["alternative"]["isSport"] is False


def test_claim_uses_null_when_no_opposite_section_exists(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(
        monkeypatch,
        now_ts=now_ts,
        top_category="sport",
        include_alternative=False,
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-no-opposite-run"},
        )

    assert response.status_code == 200
    assert response.json()["top"]["isSport"] is True
    assert response.json()["alternative"] is None


def test_claim_is_slot_idempotent_and_receipt_finalizes_article_dedup(
    monkeypatch,
    tmp_db,
):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        duplicate = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        competing_run = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-competing-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": first.json()["slotId"],
                "requestId": "synthetic-idempotent-run",
                "status": "sent",
            },
        )
        replay_after_receipt = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-idempotent-run"},
        )
        repeated_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": first.json()["slotId"],
                "requestId": "synthetic-idempotent-run",
                "status": "sent",
            },
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert competing_run.status_code == 200
    assert competing_run.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert receipt.status_code == 200
    assert replay_after_receipt.status_code == 200
    assert replay_after_receipt.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert receipt.headers["cache-control"] == "no-store"
    assert receipt.json() == {
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "status": "sent",
        "recordedAt": "2026-08-03T12:30:30+02:00",
    }
    assert repeated_receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "sent"
    assert alert["alert_count"] == 1
    assert slot is not None
    assert slot["status"] == "sent"
    assert slot["request_ref"] != "synthetic-idempotent-run"
    assert "synthetic-idempotent-run" not in slot["claim_payload_json"]


def test_failed_receipt_releases_slot_without_recording_a_send(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-failed-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-failed-run",
                "status": "failed",
            },
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "transport_failed"
    assert alert["alert_count"] == 0
    assert slot is not None
    assert slot["status"] == "failed"


def test_uncertain_receipt_is_terminal_and_prevents_a_duplicate(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-uncertain-run"},
        )
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-uncertain-run",
                "status": "delivery_uncertain",
            },
        )
        competing = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-after-uncertain-run"},
        )
        alert = database.teams_alert_get(top["url"])
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert competing.status_code == 200
    assert competing.json() == {
        "ready": False,
        "reason": "slot_already_claimed",
    }
    assert alert is not None
    assert alert["status"] == "delivery_uncertain"
    assert slot is not None
    assert slot["status"] == "delivery_uncertain"


def test_claim_fails_closed_when_final_live_dedup_refresh_is_unavailable(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate,
        "_dispatch_live_push_comparison",
        lambda *_args, **_kwargs: {
            "blocked": True,
            "code": "live_push_dedup_unavailable_failclosed",
        },
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-dedup-outage-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert slot is None


def test_claim_fails_closed_without_authoritative_live_push_history(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate,
        "_refresh_push_history_for_dedup",
        lambda: {"history": [], "history_authoritative": False},
    )
    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-cloud-only-run"},
        )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_expected_selection_no_ops_stay_http_200(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(
        power_automate,
        "evaluate_teams_alert_candidates",
        lambda *_args, **_kwargs: {"selectedCandidate": None, "decisions": []},
    )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-no-candidate-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "no_candidate"}


def test_slot_close_during_selection_is_a_no_op(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    monkeypatch.setattr(power_automate, "_power_automate_slot_open", lambda *_a, **_k: False)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-slot-closed-run"},
        )

    assert response.status_code == 200
    assert response.json() == {"ready": False, "reason": "slot_closed"}


def test_fixed_power_automate_slot_ignores_legacy_date_delay(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)
    delayed_config = replace(
        power_automate.TeamsAlertConfig(),
        slot_delay_date="2026-08-03",
        slot_delay_from="12:30",
        slot_delay_minutes=15,
    )
    observed_dispatch_config: dict = {}

    def evaluate_with_config(_candidates, _context, config):
        observed_dispatch_config.update(
            {
                "slot_delay_date": config.slot_delay_date,
                "slot_delay_from": config.slot_delay_from,
                "slot_delay_minutes": config.slot_delay_minutes,
            }
        )
        top, _ = _synthetic_candidates(now_ts)
        decision = {
            "candidateId": top["url"],
            "shouldNotify": True,
            "summary": "Verbindlicher Push-Balancer-Top-1 im festen Slot",
        }
        return {
            "selectedCandidate": top,
            "decisions": [{"candidate": top, "decision": decision}],
        }

    monkeypatch.setattr(power_automate, "TeamsAlertConfig", lambda: delayed_config)
    monkeypatch.setattr(power_automate, "evaluate_teams_alert_candidates", evaluate_with_config)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-fixed-slot-run"},
        )

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert observed_dispatch_config == {
        "slot_delay_date": "",
        "slot_delay_from": "",
        "slot_delay_minutes": 0,
    }


def test_initial_claim_needs_delivery_budget_before_slot_expiry(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 299
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        nearly_expired = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-nearly-expired-run"},
        )
        db_patch.setattr(power_automate.time, "time", lambda: SLOT_TS + 301)
        after_window = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-nearly-expired-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert nearly_expired.status_code == 200
    assert nearly_expired.json() == {"ready": False, "reason": "slot_closed"}
    assert after_window.status_code == 200
    assert after_window.json() == {"ready": False, "reason": "outside_window"}
    assert slot is None


def test_receipt_is_bound_to_the_claim_run_across_slot_expiry(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    now_ts = SLOT_TS + 30
    late_ts = SLOT_TS + 301
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-owner-run"},
        )
        wrong_owner = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-other-run",
                "status": "sent",
            },
        )
        db_patch.setattr(power_automate.time, "time", lambda: late_ts)
        db_patch.setattr(slot_claims.time, "time", lambda: late_ts)
        correct_owner = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-owner-run",
                "status": "sent",
            },
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert wrong_owner.status_code == 409
    assert correct_owner.status_code == 200
    assert slot is not None
    assert slot["status"] == "sent"
    assert slot["request_ref"]
    assert slot["claim_payload_json"] == ""


def test_late_receipt_lookup_is_not_limited_to_recent_dashboard_rows(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate
    import app.teams_slot_claims as slot_claims

    now_ts = SLOT_TS + 30
    late_ts = SLOT_TS + 301
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-retained-lookup-run"},
        )
        for index in range(101):
            database.teams_alert_record(
                article_key=f"https://www.bild.de/news/synthetic-newer-{index}",
                article_id=f"synthetic-newer-{index}",
                article_url=f"https://www.bild.de/news/synthetic-newer-{index}",
                article_title=f"Synthetische neuere Meldung {index}",
                title_hash=f"synthetic-newer-hash-{index}",
                score=80.0,
                predicted_or=0.05,
                candidate_updated_at=late_ts,
                is_breaking=False,
                reason="synthetic",
                status="failed",
                decision_ts=late_ts + index,
            )
        db_patch.setattr(power_automate.time, "time", lambda: late_ts)
        db_patch.setattr(slot_claims.time, "time", lambda: late_ts)
        receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": claim.json()["slotId"],
                "requestId": "synthetic-retained-lookup-run",
                "status": "sent",
            },
        )
        alert = database.teams_alert_get(top["url"])

    assert claim.status_code == 200
    assert receipt.status_code == 200
    assert alert is not None
    assert alert["status"] == "sent"


def test_late_receipt_cannot_finalize_a_later_article_claim(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    first_claimed_at = SLOT_TS + 30
    second_slot_ts = SLOT_TS + 600
    second_claimed_at = second_slot_ts + 30
    article_key = "https://www.bild.de/news/synthetic-cross-slot"
    payload = {
        "ready": True,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "top": {
            "title": "Synthetische Cross-Slot-Meldung",
            "url": article_key,
            "category": "news",
            "pushScore": 91.0,
            "isSport": False,
        },
        "alternative": None,
        "messageHtml": "<p>Synthetische Cross-Slot-Meldung</p>",
    }
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)

    def claim_article(decision_ts: int) -> dict:
        return database.teams_alert_try_claim_send(
            article_key=article_key,
            article_id="synthetic-cross-slot",
            article_url=article_key,
            article_title="Synthetische Cross-Slot-Meldung",
            title_hash="synthetic-title-hash",
            score=91.0,
            predicted_or=0.06,
            candidate_updated_at=decision_ts - 60,
            is_breaking=False,
            reason="synthetic",
            decision_ts=decision_ts,
            alert_cooldown_minutes=0,
            global_cooldown_minutes=0,
            in_progress_cooldown_minutes=5,
            failed_cooldown_minutes=0,
            transport_failure_cooldown_minutes=0,
        )

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        first_slot = teams_recommendation_slot_try_claim(
            SLOT_TS,
            article_key=article_key,
            request_id="synthetic-old-run",
            claim_payload=payload,
            now_ts=first_claimed_at,
        )
        first_article = claim_article(first_claimed_at)
        second_slot = teams_recommendation_slot_try_claim(
            second_slot_ts,
            article_key=article_key,
            request_id="synthetic-new-run",
            claim_payload={
                **payload,
                "slotId": f"teams-recommendation-{second_slot_ts}",
            },
            now_ts=second_claimed_at,
        )
        second_article = claim_article(second_claimed_at)
        db_patch.setattr(power_automate.time, "time", lambda: second_claimed_at + 1)
        stale_receipt = client.post(
            "/api/v1/power-automate/teams/receipt",
            headers=HEADERS,
            json={
                "slotId": f"teams-recommendation-{SLOT_TS}",
                "requestId": "synthetic-old-run",
                "status": "sent",
            },
        )
        first_state = teams_recommendation_slot_get(SLOT_TS)
        second_state = teams_recommendation_slot_get(second_slot_ts)
        alert = database.teams_alert_get(article_key)

    assert first_slot["claimed"] is True
    assert first_article["claimed"] is True
    assert second_slot["claimed"] is True
    assert second_article["claimed"] is True
    assert stale_receipt.status_code == 409
    assert first_state is not None and first_state["status"] == "sending"
    assert second_state is not None and second_state["status"] == "sending"
    assert alert is not None
    assert alert["status"] == "sending"
    assert alert["last_decision_ts"] == second_claimed_at
    assert alert["alert_count"] == 0


def test_parallel_sent_receipts_increment_alert_count_once(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-parallel-receipt-run"},
        )
        receipt_body = {
            "slotId": claim.json()["slotId"],
            "requestId": "synthetic-parallel-receipt-run",
            "status": "sent",
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda _index: client.post(
                        "/api/v1/power-automate/teams/receipt",
                        headers=HEADERS,
                        json=receipt_body,
                    ),
                    range(2),
                )
            )
        alert = database.teams_alert_get(top["url"])

    assert [response.status_code for response in responses] == [200, 200]
    assert alert is not None
    assert alert["status"] == "sent"
    assert alert["alert_count"] == 1


def test_replay_repairs_missing_article_claim_with_canonical_url(monkeypatch, tmp_db):
    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, alternative = _patch_successful_claim(monkeypatch, now_ts=now_ts)
    raw_url = top["url"].upper() + "/?wtmc=synthetic"
    article_key = candidate_key({"url": raw_url})
    payload = {
        "ready": True,
        "slotId": f"teams-recommendation-{SLOT_TS}",
        "scheduledAt": "2026-08-03T12:30:00+02:00",
        "expiresAt": "2026-08-03T12:35:00+02:00",
        "top": {
            "title": top["title"],
            "url": raw_url,
            "category": top["category"],
            "pushScore": top["score"],
            "isSport": False,
        },
        "alternative": {
            "title": alternative["title"],
            "url": alternative["url"],
            "category": alternative["category"],
            "pushScore": alternative["score"],
            "isSport": True,
        },
        "messageHtml": "<p>Synthetic replay</p>",
    }

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        slot_claim = teams_recommendation_slot_try_claim(
            SLOT_TS,
            article_key=article_key,
            request_id="synthetic-repair-run",
            claim_payload=payload,
            now_ts=now_ts,
        )
        response = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-repair-run"},
        )
        alert = database.teams_alert_get(article_key)

    assert slot_claim["claimed"] is True
    assert response.status_code == 200
    assert response.json() == payload
    assert alert is not None
    assert alert["status"] == "sending"


def test_stale_replay_cannot_downgrade_a_sent_slot(monkeypatch, tmp_db):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    top, _ = _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        claim = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-replay-run"},
        )
        teams_recommendation_slot_record(
            SLOT_TS,
            article_key=candidate_key(top),
            status="sent",
            now_ts=now_ts,
        )
        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_replay",
            lambda *_args, **_kwargs: claim.json(),
        )
        replay = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-replay-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert replay.status_code == 200
    assert replay.json() == {"ready": False, "reason": "slot_already_claimed"}
    assert slot is not None
    assert slot["status"] == "sent"


def test_stale_replay_cannot_take_over_another_runs_reclaimed_slot(
    monkeypatch,
    tmp_db,
):
    import app.routers.power_automate as power_automate

    now_ts = SLOT_TS + 30
    monkeypatch.setattr(auth.config, "POWER_AUTOMATE_API_KEY", POWER_AUTOMATE_KEY)
    _patch_successful_claim(monkeypatch, now_ts=now_ts)

    with monkeypatch.context() as db_patch:
        db_patch.setattr(database, "PUSH_DB_PATH", tmp_db)
        owner_b = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-owner-b-run"},
        )
        original_replay = power_automate.teams_recommendation_slot_replay
        calls = 0

        def stale_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return owner_b.json()
            return original_replay(*args, **kwargs)

        db_patch.setattr(
            power_automate,
            "teams_recommendation_slot_replay",
            stale_once,
        )
        stale_a = client.post(
            "/api/v1/power-automate/teams/claim",
            headers=HEADERS,
            json={"requestId": "synthetic-stale-owner-a-run"},
        )
        slot = teams_recommendation_slot_get(SLOT_TS)

    assert owner_b.status_code == 200
    assert owner_b.json()["ready"] is True
    assert stale_a.status_code == 200
    assert stale_a.json() == {"ready": False, "reason": "slot_already_claimed"}
    assert slot is not None
    assert slot["status"] == "sending"
