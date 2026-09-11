"""Contract and parity tests for the CMS-ID score adapter."""

from __future__ import annotations

import json
import logging
import urllib.error
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import auth, config
from app.cms import url_api
from app import main as main_module
from app.main import app
from app.render_score_capture import (
    CapturedScore,
    EditorialScoreBreakdown,
    EngagementScoreBreakdown,
    RenderScoreUnavailable,
    SportScoreBreakdown,
)
from app.routers import score_api

client = TestClient(app, raise_server_exceptions=True)

CMS_ID = "0123456789abcdef01234567"
OTHER_CMS_ID = "fedcba987654321001234567"
ARTICLE_URL = "https://www.bild.de/politik/synthetic-score-article"
SCORE_KEY = "synthetic-score-key"
SCORE_HEADERS = {"X-Score-Key": SCORE_KEY}
SCORED_AT = "2026-07-15T12:00:00Z"


@pytest.fixture(autouse=True)
def configured_score_key(monkeypatch):
    monkeypatch.setattr(auth, "SCORE_API_KEY", SCORE_KEY)


def test_score_lookup_projects_exact_captured_ui_score(monkeypatch):
    lookup = Mock(return_value=CapturedScore(score=54.3, captured_at=SCORED_AT))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "cmsId": CMS_ID,
        "score": 54.3,
        "scoredAt": SCORED_AT,
        "scoreBreakdown": None,
        "orFactor": None,
    }
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    lookup.assert_called_once_with(CMS_ID, include_editorial=False)


def test_score_lookup_projects_exact_engagement_breakdown_and_factor(monkeypatch):
    captured = CapturedScore(
        score=55,
        captured_at=SCORED_AT,
        score_breakdown=EngagementScoreBreakdown(
            kind="engagement",
            relevance=30,
            urgency=0,
            curiosity=7.6,
            freshness=11.7,
            timing=6,
            title_boost=3,
            breaking=0,
            research=0,
            push_history=0,
            topic_saturation=0,
        ),
        or_factor=1.06,
    )
    monkeypatch.setattr(score_api, "get_captured_score", lambda _cms_id, **_kwargs: captured)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "cmsId": CMS_ID,
        "score": 55.0,
        "scoredAt": SCORED_AT,
        "scoreBreakdown": {
            "kind": "engagement",
            "relevance": 30.0,
            "urgency": 0.0,
            "curiosity": 7.6,
            "freshness": 11.7,
            "timing": 6.0,
            "titleBoost": 3.0,
            "breaking": 0.0,
            "research": 0.0,
            "pushHistory": 0.0,
            "topicSaturation": 0.0,
        },
        "orFactor": 1.06,
    }


def test_score_lookup_projects_exact_sport_breakdown(monkeypatch):
    captured = CapturedScore(
        score=70,
        captured_at=SCORED_AT,
        score_breakdown=SportScoreBreakdown(
            kind="sport",
            sport_relevance=32,
            timing=18,
            drama=12,
            freshness=8,
        ),
        or_factor=0.94,
    )
    monkeypatch.setattr(score_api, "get_captured_score", lambda _cms_id, **_kwargs: captured)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 200
    assert response.json()["scoreBreakdown"] == {
        "kind": "sport",
        "sportRelevance": 32.0,
        "timing": 18.0,
        "drama": 12.0,
        "freshness": 8.0,
    }
    assert response.json()["orFactor"] == 0.94


def test_batch_projects_one_result_per_position_with_one_deduplicated_source_call(
    monkeypatch,
):
    captured = CapturedScore(
        score=55,
        captured_at=SCORED_AT,
        score_breakdown=EngagementScoreBreakdown(
            kind="engagement",
            relevance=30,
            urgency=0,
            curiosity=7.6,
            freshness=11.7,
            timing=6,
            title_boost=3,
            breaking=0,
            research=0,
            push_history=0,
            topic_saturation=0,
        ),
        or_factor=1.06,
    )
    source = Mock(return_value=([CMS_ID, OTHER_CMS_ID], [captured, None]))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID.upper(), OTHER_CMS_ID, CMS_ID]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "requestedCount": 3,
        "uniqueCount": 2,
        "foundCount": 2,
        "notFoundCount": 1,
        "results": [
            {
                "status": "found",
                "cmsId": CMS_ID.upper(),
                "score": 55.0,
                "scoredAt": SCORED_AT,
                "scoreBreakdown": {
                    "kind": "engagement",
                    "relevance": 30.0,
                    "urgency": 0.0,
                    "curiosity": 7.6,
                    "freshness": 11.7,
                    "timing": 6.0,
                    "titleBoost": 3.0,
                    "breaking": 0.0,
                    "research": 0.0,
                    "pushHistory": 0.0,
                    "topicSaturation": 0.0,
                },
                "orFactor": 1.06,
            },
            {
                "status": "notFound",
                "cmsId": OTHER_CMS_ID,
            },
            {
                "status": "found",
                "cmsId": CMS_ID,
                "score": 55.0,
                "scoredAt": SCORED_AT,
                "scoreBreakdown": {
                    "kind": "engagement",
                    "relevance": 30.0,
                    "urgency": 0.0,
                    "curiosity": 7.6,
                    "freshness": 11.7,
                    "timing": 6.0,
                    "titleBoost": 3.0,
                    "breaking": 0.0,
                    "research": 0.0,
                    "pushHistory": 0.0,
                    "topicSaturation": 0.0,
                },
                "orFactor": 1.06,
            },
        ],
    }
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    source.assert_called_once_with([CMS_ID, OTHER_CMS_ID], include_editorial=False)


def test_batch_projects_legacy_found_with_explicit_null_pair(monkeypatch):
    source = Mock(
        return_value=(
            [CMS_ID],
            [CapturedScore(score=54.3, captured_at=SCORED_AT)],
        )
    )
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "status": "found",
            "cmsId": CMS_ID,
            "score": 54.3,
            "scoredAt": SCORED_AT,
            "scoreBreakdown": None,
            "orFactor": None,
        }
    ]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"cmsIds": []},
        {"cmsIds": [CMS_ID], "extra": True},
        {"cmsIds": ["not-a-cms-id"]},
        {"cmsIds": [CMS_ID] * 501},
    ],
)
def test_batch_rejects_invalid_body_before_source(monkeypatch, body):
    source = Mock(side_effect=AssertionError("source must not run"))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json=body,
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    assert response.json()["instance"] == "/api/v1/scores/batch"
    assert CMS_ID not in response.text
    source.assert_not_called()


@pytest.mark.parametrize("headers", [{}, {"X-Score-Key": "wrong"}])
def test_batch_rejects_missing_or_wrong_key_before_source(monkeypatch, headers):
    source = Mock(side_effect=AssertionError("source must not run"))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    response = client.post(
        "/api/v1/scores/batch",
        headers=headers,
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["instance"] == "/api/v1/scores/batch"
    source.assert_not_called()


def test_batch_maps_whole_source_failure_to_redacted_bad_gateway(monkeypatch):
    source = Mock(side_effect=RenderScoreUnavailable("synthetic private body"))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID, OTHER_CMS_ID]},
    )

    assert response.status_code == 502
    assert response.json()["instance"] == "/api/v1/scores/batch"
    assert CMS_ID not in response.text
    assert OTHER_CMS_ID not in response.text
    assert "synthetic private body" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_batch_unexpected_failure_does_not_log_ids_or_exception_body(
    monkeypatch,
    caplog,
):
    private_body = f"synthetic private body {CMS_ID}"
    monkeypatch.setattr(
        score_api,
        "get_captured_scores_batch",
        Mock(side_effect=RuntimeError(private_body)),
    )

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/api/v1/scores/batch",
            headers=SCORE_HEADERS,
            json={"cmsIds": [CMS_ID]},
        )

    assert response.status_code == 503
    assert CMS_ID not in response.text
    assert private_body not in response.text
    assert CMS_ID not in caplog.text
    assert private_body not in caplog.text
    assert "RuntimeError" in caplog.text


def test_batch_rejects_inconsistent_source_mapping_as_bad_gateway(monkeypatch):
    monkeypatch.setattr(
        score_api,
        "get_captured_scores_batch",
        lambda _cms_ids, **_kwargs: ([OTHER_CMS_ID], [None]),
    )

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 502
    assert CMS_ID not in response.text


def test_third_batch_is_rejected_immediately_without_source_call(monkeypatch):
    source = Mock(side_effect=AssertionError("source must not run"))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)
    assert score_api._BATCH_SOURCE_SLOTS.acquire(blocking=False)
    assert score_api._BATCH_SOURCE_SLOTS.acquire(blocking=False)
    try:
        response = client.post(
            "/api/v1/scores/batch",
            headers=SCORE_HEADERS,
            json={"cmsIds": [CMS_ID]},
        )
    finally:
        score_api._BATCH_SOURCE_SLOTS.release()
        score_api._BATCH_SOURCE_SLOTS.release()

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["instance"] == "/api/v1/scores/batch"
    source.assert_not_called()


def test_score_lookup_without_fresh_ui_capture_returns_not_found(monkeypatch):
    monkeypatch.setattr(score_api, "get_captured_score", lambda _cms_id, **_kwargs: None)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 404


@pytest.mark.parametrize("headers", [{}, {"X-Score-Key": "wrong"}])
def test_score_lookup_rejects_missing_or_wrong_key(monkeypatch, headers):
    lookup = Mock(side_effect=AssertionError("lookup must not run"))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=headers)

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
    lookup.assert_not_called()


def test_score_lookup_is_disabled_without_server_key(monkeypatch):
    monkeypatch.setattr(auth, "SCORE_API_KEY", "")
    lookup = Mock(side_effect=AssertionError("lookup must not run"))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 503
    lookup.assert_not_called()


def test_score_lookup_rejects_invalid_id_before_mapping(monkeypatch):
    lookup = Mock(side_effect=AssertionError("lookup must not run"))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)

    response = client.get("/api/v1/scores/not.a.cms.id", headers=SCORE_HEADERS)

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
    lookup.assert_not_called()


def test_score_lookup_redacts_overlong_id(monkeypatch):
    lookup = Mock(side_effect=AssertionError("lookup must not run"))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)

    response = client.get(f"/api/v1/scores/{'a' * 129}", headers=SCORE_HEADERS)

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
    lookup.assert_not_called()


def test_score_route_remains_behind_internal_network_gate(monkeypatch):
    lookup = Mock(side_effect=AssertionError("lookup must not run"))
    monkeypatch.setattr(score_api, "get_captured_score", lookup)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", ["192.0.2.0/24"])

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 404
    lookup.assert_not_called()


def test_score_lookup_maps_render_failures_without_identifier_leak(monkeypatch):
    monkeypatch.setattr(score_api, "get_captured_score", Mock(
        side_effect=RenderScoreUnavailable("synthetic upstream body")
    ))

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 502
    assert CMS_ID not in response.text
    assert "synthetic upstream body" not in response.text
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
def test_score_lookup_maps_unexpected_failure_without_identifier_leak(monkeypatch):
    monkeypatch.setattr(score_api, "get_captured_score", Mock(
        side_effect=RuntimeError("synthetic internal body")
    ))

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 503
    assert CMS_ID not in response.text
    assert "synthetic internal body" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_url_api_requests_only_one_encoded_id(monkeypatch):
    requested_urls: list[str] = []
    requested_headers: list[dict[str, str]] = []

    class SyntheticResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(
                [
                    {
                        "documentId": CMS_ID,
                        "urls": [
                            {
                                "path": "/politik/synthetic-score-article",
                                "isCanonicalUrl": True,
                            }
                        ],
                    }
                ]
            ).encode()

    def fake_open_request(request):
        requested_urls.append(request.full_url)
        requested_headers.append(dict(request.header_items()))
        return SyntheticResponse()

    monkeypatch.setattr(config, "URL_API_BASE", "https://api.stg.editorial.one/urlapi/v2")
    monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
    monkeypatch.setattr(url_api, "_open_request", fake_open_request)

    result = url_api.get_canonical_article_url(CMS_ID)

    assert result == ARTICLE_URL
    assert requested_urls == [
        f"https://api.stg.editorial.one/urlapi/v2/tenants/bild/document-urls?documentIds={CMS_ID}"
    ]
    assert requested_headers == [
        {
            "Accept": "application/json",
            "User-agent": "NextPushBalancer/1.0",
            "X-api-key": "synthetic-api-key",
        }
    ]


def test_url_api_failure_log_does_not_include_id_or_upstream_body(monkeypatch, caplog):
    upstream_body = "synthetic-private-upstream-body"

    def fail_open_request(request):
        raise urllib.error.HTTPError(
            request.full_url,
            502,
            upstream_body,
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(config, "URL_API_BASE", "https://api.stg.editorial.one/urlapi/v2")
    monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
    monkeypatch.setattr(url_api, "_open_request", fail_open_request)

    with caplog.at_level(logging.WARNING), pytest.raises(url_api.UrlApiUnavailable):
        url_api.get_canonical_article_url(CMS_ID)

    assert CMS_ID not in caplog.text
    assert upstream_body not in caplog.text
    assert "status 502" in caplog.text


def test_url_api_rejects_canonical_url_from_another_host(monkeypatch):
    payload = [
        {
            "documentId": CMS_ID,
            "urls": [
                {
                    "path": "https://unapproved.example.invalid/article",
                    "isCanonicalUrl": True,
                }
            ],
        }
    ]

    assert url_api._canonical_url_from_payload(payload, CMS_ID) is None


def test_url_api_requires_https_before_sending_credential(monkeypatch):
    monkeypatch.setattr(config, "URL_API_BASE", "http://api.stg.editorial.one/urlapi/v2")
    monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
    open_request = Mock(side_effect=AssertionError("HTTP request must not be sent"))
    monkeypatch.setattr(url_api, "_open_request", open_request)

    with pytest.raises(url_api.UrlApiNotConfigured):
        url_api.get_canonical_article_url(CMS_ID)

    open_request.assert_not_called()


@pytest.mark.parametrize(
    "invalid_base",
    [
        "https://unapproved.example.invalid/urlapi/v2",
        "https://api.stg.editorial.one/other-api",
        "https://api.stg.editorial.one:8443/urlapi/v2",
        "https://api.stg.editorial.one/urlapi/v2?unexpected=true",
    ],
)
def test_url_api_rejects_unapproved_base_before_sending_key(monkeypatch, invalid_base):
    monkeypatch.setattr(config, "URL_API_BASE", invalid_base)
    monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
    open_request = Mock(side_effect=AssertionError("request must not be sent"))
    monkeypatch.setattr(url_api, "_open_request", open_request)

    with pytest.raises(url_api.UrlApiNotConfigured):
        url_api.get_canonical_article_url(CMS_ID)

    open_request.assert_not_called()


def test_url_api_redirect_handler_never_forwards_request_headers():
    request = url_api.urllib.request.Request(
        "https://api.stg.editorial.one/urlapi/v2/source",
        headers={"x-api-key": "synthetic-api-key"},
    )

    redirected = url_api._NoRedirectHandler().redirect_request(
        request,
        fp=None,
        code=302,
        msg="Found",
        headers={},
        newurl="https://unapproved.example.invalid/target",
    )

    assert redirected is None


def test_url_api_rejects_oversized_response(monkeypatch):
    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"x" * (url_api._MAX_RESPONSE_BYTES + 1)

    monkeypatch.setattr(config, "URL_API_BASE", "https://api.stg.editorial.one/urlapi/v2")
    monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
    monkeypatch.setattr(url_api, "_open_request", lambda _request: OversizedResponse())

    with pytest.raises(url_api.UrlApiUnavailable):
        url_api.get_canonical_article_url(CMS_ID)


def test_url_api_requires_a_service_credential(monkeypatch):
    monkeypatch.setattr(config, "URL_API_BASE", "https://api.stg.editorial.one/urlapi/v2")
    monkeypatch.setattr(config, "URL_API_KEY", "")
    open_request = Mock(side_effect=AssertionError("request must not be sent"))
    monkeypatch.setattr(url_api, "_open_request", open_request)

    with pytest.raises(url_api.UrlApiNotConfigured):
        url_api.get_canonical_article_url(CMS_ID)

    open_request.assert_not_called()


def test_url_api_requires_https_for_the_public_article_base(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_ARTICLE_BASE_URL", "http://www.bild.de")

    assert url_api._normalize_public_url("/synthetic-article") is None


def test_score_openapi_contract_is_minimal_and_key_protected():
    operation = app.openapi()["paths"]["/api/v1/scores/{cms_id}"]["get"]

    assert operation["operationId"] == "getScoreByCmsId"
    assert {tuple(item) for item in operation["security"]} == {("scoreApiKey",)}
    schema_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    schema_name = schema_ref.rsplit("/", 1)[-1]
    schema = app.openapi()["components"]["schemas"][schema_name]
    assert set(schema["properties"]) == {
        "cmsId",
        "score",
        "scoredAt",
        "scoreBreakdown",
        "orFactor",
    }


EDITORIAL_BREAKDOWN = EditorialScoreBreakdown(
    kind="editorial",
    bild_reiz=80.0,
    bild_reiz_points=32.0,
    bild_reiz_source="llm_reader_score",
    reader_score=80.0,
    reader_score_reasoning="Das trifft viele Autofahrer direkt und ich will sofort wissen, was los ist.",
    opening_rate_potential=70.0,
    opening_rate_potential_points=14.0,
    freshness=90.0,
    freshness_points=13.5,
    mix_balance=55.0,
    mix_balance_points=5.5,
    historical_timing=60.0,
    historical_timing_points=6.0,
    headline_strength=65.0,
    headline_strength_points=1.95,
    risk_and_fatigue=50.0,
    risk_and_fatigue_points=1.0,
    editorial_feedback=60.0,
    editorial_feedback_points=0.0,
    other_adjustments=3.95,
    base_score=77.9,
    freshness_multiplier=1.0,
)
EXPECTED_EDITORIAL_JSON = {
    "kind": "editorial",
    "bildReiz": 80.0,
    "bildReizPoints": 32.0,
    "bildReizSource": "llm_reader_score",
    "readerScore": 80.0,
    "readerScoreReasoning": "Das trifft viele Autofahrer direkt und ich will sofort wissen, was los ist.",
    "openingRatePotential": 70.0,
    "openingRatePotentialPoints": 14.0,
    "freshness": 90.0,
    "freshnessPoints": 13.5,
    "mixBalance": 55.0,
    "mixBalancePoints": 5.5,
    "historicalTiming": 60.0,
    "historicalTimingPoints": 6.0,
    "headlineStrength": 65.0,
    "headlineStrengthPoints": 1.95,
    "riskAndFatigue": 50.0,
    "riskAndFatiguePoints": 1.0,
    "editorialFeedback": 60.0,
    "editorialFeedbackPoints": 0.0,
    "otherAdjustments": 3.95,
    "baseScore": 77.9,
    "freshnessMultiplier": 1.0,
}


def test_score_lookup_projects_the_weighted_editorial_breakdown(monkeypatch):
    captured = CapturedScore(
        score=77.9,
        captured_at=SCORED_AT,
        score_breakdown=EDITORIAL_BREAKDOWN,
        or_factor=1.21,
    )
    monkeypatch.setattr(score_api, "get_captured_score", Mock(return_value=captured))

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "cmsId": CMS_ID,
        "score": 77.9,
        "scoredAt": SCORED_AT,
        "scoreBreakdown": EXPECTED_EDITORIAL_JSON,
        "orFactor": 1.21,
    }


def test_batch_projects_the_weighted_editorial_breakdown(monkeypatch):
    captured = CapturedScore(
        score=77.9,
        captured_at=SCORED_AT,
        score_breakdown=EDITORIAL_BREAKDOWN,
        or_factor=1.21,
    )
    monkeypatch.setattr(
        score_api,
        "get_captured_scores_batch",
        Mock(return_value=([CMS_ID], [captured])),
    )

    response = client.post(
        "/api/v1/scores/batch",
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "status": "found",
            "cmsId": CMS_ID,
            "score": 77.9,
            "scoredAt": SCORED_AT,
            "scoreBreakdown": EXPECTED_EDITORIAL_JSON,
            "orFactor": 1.21,
        }
    ]


def test_editorial_breakdown_reconciles_with_the_delivered_score():
    points = sum(
        value
        for key, value in EXPECTED_EDITORIAL_JSON.items()
        if key.endswith("Points")
    )
    base = EXPECTED_EDITORIAL_JSON["baseScore"]
    assert round(points + EXPECTED_EDITORIAL_JSON["otherAdjustments"], 2) == base
    assert round(base * EXPECTED_EDITORIAL_JSON["freshnessMultiplier"], 1) == 77.9


def test_old_consumer_keeps_its_exact_contract_without_the_new_parameter(monkeypatch):
    """Ohne includeEditorialBreakdown=1 wird die serverseitige Form nicht geliefert."""
    captured = CapturedScore(
        score=77.9,
        captured_at=SCORED_AT,
        score_breakdown=EDITORIAL_BREAKDOWN,
        or_factor=1.21,
    )
    source = Mock(return_value=captured)
    monkeypatch.setattr(score_api, "get_captured_score", source)

    response = client.get(f"/api/v1/scores/{CMS_ID}", headers=SCORE_HEADERS)

    assert response.status_code == 200
    source.assert_called_once_with(CMS_ID, include_editorial=False)


def test_opt_in_parameter_reaches_the_source(monkeypatch):
    source = Mock(return_value=None)
    monkeypatch.setattr(score_api, "get_captured_score", source)

    client.get(
        f"/api/v1/scores/{CMS_ID}",
        params={"includeEditorialBreakdown": 1},
        headers=SCORE_HEADERS,
    )

    source.assert_called_once_with(CMS_ID, include_editorial=True)


def test_batch_opt_in_parameter_reaches_the_source(monkeypatch):
    source = Mock(return_value=([CMS_ID], [None]))
    monkeypatch.setattr(score_api, "get_captured_scores_batch", source)

    client.post(
        "/api/v1/scores/batch",
        params={"includeEditorialBreakdown": 1},
        headers=SCORE_HEADERS,
        json={"cmsIds": [CMS_ID]},
    )

    source.assert_called_once_with([CMS_ID], include_editorial=True)
