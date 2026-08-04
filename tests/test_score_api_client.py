import json
from datetime import datetime, timedelta, timezone

import pytest

from app.routers.feed import _apply_internal_score_api_scores
from app.routers.score_api import ArticleScoreResponse
from app.score_api_client import (
    ScoreApiClient,
    ScoreApiConfigurationError,
    ScoreApiUnavailable,
    fetch_score_lookups,
    resolve_cms_id,
)


CMS_A = "0123456789abcdef01234567"
CMS_B = "89abcdef0123456701234567"
BASE_URL = "https://scores.example.invalid"
API_KEY = "synthetic-test-key"
ENGAGEMENT_BREAKDOWN = {
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
}


def _body(cms_id=CMS_A, score=87.4, scored_at="2026-07-20T06:12:00Z"):
    return json.dumps({"cmsId": cms_id, "score": score, "scoredAt": scored_at}).encode("utf-8")


def _found(cms_id=CMS_A, score=87.4, scored_at="2026-07-20T06:12:00Z"):
    return {
        "status": "found",
        "cmsId": cms_id,
        "score": score,
        "scoredAt": scored_at,
        "scoreBreakdown": None,
        "orFactor": None,
    }


def _batch_body(results):
    found = sum(item.get("status") == "found" for item in results)
    return json.dumps(
        {
            "requestedCount": len(results),
            "uniqueCount": len({item["cmsId"].lower() for item in results}),
            "foundCount": found,
            "notFoundCount": len(results) - found,
            "results": results,
        }
    ).encode("utf-8")


def test_score_client_validates_response_and_uses_header_only():
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers, timeout))
        return 200, _body()

    client = ScoreApiClient(BASE_URL, API_KEY, transport=transport)

    first = client.get_score(CMS_A)
    second = client.get_score(CMS_A)

    assert first == second
    assert first.score == 87.4
    assert first.scored_at == datetime(2026, 7, 20, 6, 12, tzinfo=timezone.utc)
    assert len(calls) == 1
    assert calls[0][0] == f"{BASE_URL}/api/v1/scores/{CMS_A}"
    assert API_KEY not in calls[0][0]
    assert calls[0][1]["X-Score-Key"] == API_KEY
    assert calls[0][2] == 2.5


def test_score_client_accepts_the_deployed_score_api_response_contract():
    body = ArticleScoreResponse(
        cmsId=CMS_A,
        score=87.4,
        scoredAt="2026-07-20T06:12:00Z",
        scoreBreakdown=None,
        orFactor=None,
    ).model_dump_json().encode("utf-8")
    client = ScoreApiClient(BASE_URL, API_KEY, transport=lambda *_args: (200, body))

    assert client.get_score(CMS_A).score == 87.4


def test_score_client_accepts_documented_enriched_response_fields():
    payload = {
        "cmsId": CMS_A,
        "score": 87.4,
        "scoredAt": "2026-07-20T06:12:00Z",
        "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
        "orFactor": 1.06,
    }
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        transport=lambda *_args: (200, json.dumps(payload).encode("utf-8")),
    )

    assert client.get_score(CMS_A).score == 87.4


def test_score_client_maps_404_to_no_score_without_retry():
    calls = 0

    def transport(_url, _headers, _timeout):
        nonlocal calls
        calls += 1
        return 404, b"{}"

    client = ScoreApiClient(BASE_URL, API_KEY, transport=transport)

    assert client.get_score(CMS_A) is None
    assert client.get_score(CMS_A) is None
    assert calls == 1


def test_score_client_retries_one_timeout_then_succeeds():
    calls = 0

    def transport(_url, _headers, _timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("synthetic timeout")
        return 200, _body()

    client = ScoreApiClient(BASE_URL, API_KEY, transport=transport, max_retries=1)

    assert client.get_score(CMS_A).score == 87.4
    assert calls == 2


@pytest.mark.parametrize("status", [401, 403])
def test_score_client_auth_failure_is_fatal_and_does_not_expose_key(status):
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        transport=lambda *_args: (status, b"{}"),
    )

    with pytest.raises(ScoreApiConfigurationError) as error:
        client.get_score(CMS_A)

    assert API_KEY not in str(error.value)
    assert CMS_A not in str(error.value)


def test_score_client_rejects_missing_key_and_insecure_base_url():
    with pytest.raises(ScoreApiConfigurationError):
        ScoreApiClient(BASE_URL, "")
    with pytest.raises(ScoreApiConfigurationError):
        ScoreApiClient("http://scores.example.invalid", API_KEY)


def test_score_client_accepts_loopback_http_base_url():
    """Selbstkonsum (Render): http ist nur fuer Loopback-Hosts erlaubt."""
    client = ScoreApiClient(
        "http://127.0.0.1:8050",
        API_KEY,
        transport=lambda url, headers, timeout: (
            (_ for _ in ()).throw(AssertionError("no request expected"))
        ),
    )
    assert client._base_url == "http://127.0.0.1:8050"
    with pytest.raises(ScoreApiConfigurationError):
        ScoreApiClient("http://127.0.0.1.evil.example", API_KEY)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        _body(cms_id=CMS_B),
        _body(score=101),
        _body(scored_at="not-a-time"),
        json.dumps(
            {
                "cmsId": CMS_A,
                "score": 87.4,
                "scoredAt": "2026-07-20T06:12:00Z",
                "unexpected": True,
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "cmsId": CMS_A,
                "score": 87.4,
                "scoredAt": "2026-07-20T06:12:00Z",
                "scoreBreakdown": None,
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "cmsId": CMS_A,
                "score": 87.4,
                "scoredAt": "2026-07-20T06:12:00Z",
                "scoreBreakdown": None,
                "orFactor": 1.06,
            }
        ).encode("utf-8"),
    ],
)
def test_score_client_rejects_untrustworthy_success_payload(body):
    client = ScoreApiClient(BASE_URL, API_KEY, transport=lambda *_args: (200, body))

    with pytest.raises(ScoreApiUnavailable):
        client.get_score(CMS_A)


def test_field_lookup_uses_one_ordered_batch_and_keeps_not_found_closed():
    calls = []

    def batch_transport(url, headers, body, timeout):
        calls.append((url, headers, json.loads(body), timeout))
        return 200, _batch_body(
            [_found(CMS_A), {"status": "notFound", "cmsId": CMS_B}]
        )

    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        batch_transport=batch_transport,
    )

    results = fetch_score_lookups([CMS_A, CMS_B, CMS_A], client, max_concurrency=16)

    assert len(calls) == 1
    assert calls[0][0] == f"{BASE_URL}/api/v1/scores/batch"
    assert calls[0][1]["X-Score-Key"] == API_KEY
    assert calls[0][1]["Content-Type"] == "application/json"
    assert calls[0][2] == {"cmsIds": [CMS_A, CMS_B]}
    assert calls[0][3] == 35.0
    assert results[CMS_A].status == "ok"
    assert results[CMS_A].value.score == 87.4
    assert results[CMS_B].status == "not_found"
    assert results[CMS_B].value is None


def test_batch_timeout_retries_once_then_succeeds():
    calls = 0

    def batch_transport(_url, _headers, _body_bytes, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 35.0
        if calls == 1:
            raise TimeoutError("synthetic timeout")
        return 200, _batch_body([_found(CMS_A)])

    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        max_retries=9,
        batch_transport=batch_transport,
    )

    assert fetch_score_lookups([CMS_A], client)[CMS_A].status == "ok"
    assert calls == 2


def test_batch_timeout_after_one_retry_fails_closed():
    calls = 0

    def batch_transport(*_args):
        nonlocal calls
        calls += 1
        raise TimeoutError("synthetic timeout")

    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        max_retries=9,
        batch_transport=batch_transport,
    )

    assert fetch_score_lookups([CMS_A], client)[CMS_A].status == "unavailable"
    assert calls == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"requestedCount": 1},
        json.loads(_batch_body([_found(CMS_B)])),
        {
            **json.loads(_batch_body([_found(CMS_A)])),
            "foundCount": 0,
        },
        json.loads(_batch_body([{**_found(CMS_A), "unexpected": True}])),
        json.loads(
            _batch_body(
                [
                    {
                        **_found(CMS_A),
                        "scoreBreakdown": {"kind": "engagement", "relevance": 30.0},
                        "orFactor": 1.06,
                    }
                ]
            )
        ),
    ],
)
def test_malformed_batch_response_fails_the_whole_field_closed(payload):
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        batch_transport=lambda *_args: (200, json.dumps(payload).encode("utf-8")),
    )

    assert fetch_score_lookups([CMS_A], client)[CMS_A].status == "unavailable"


def test_one_malformed_batch_item_discards_every_partial_result():
    malformed_second = {**_found(CMS_B), "score": "not-numeric"}
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        batch_transport=lambda *_args: (
            200,
            _batch_body([_found(CMS_A), malformed_second]),
        ),
    )

    results = fetch_score_lookups([CMS_A, CMS_B], client)

    assert results[CMS_A].status == "unavailable"
    assert results[CMS_B].status == "unavailable"


def test_field_of_200_ids_is_one_batch_not_single_fan_out():
    cms_ids = [f"{index:024x}" for index in range(200)]
    calls = 0

    def batch_transport(_url, _headers, body, _timeout):
        nonlocal calls
        calls += 1
        requested = json.loads(body)["cmsIds"]
        return 200, _batch_body([_found(cms_id) for cms_id in requested])

    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        batch_transport=batch_transport,
    )

    results = fetch_score_lookups(cms_ids, client)

    assert calls == 1
    assert len(results) == 200
    assert all(lookup.status == "ok" for lookup in results.values())


def test_cms_id_resolution_prefers_field_and_uses_strict_url_fallback():
    assert resolve_cms_id({"cmsId": CMS_A.upper(), "url": f"https://x/{CMS_B}"}) == CMS_A
    assert resolve_cms_id({"url": f"https://www.bild.de/news/thema-{CMS_B}.html"}) == CMS_B
    assert resolve_cms_id({"url": "https://www.bild.de/sport", "urlId": CMS_B}) == CMS_B
    assert (
        resolve_cms_id(
            {
                "url": "https://www.bild.de/sport",
                "urlId": f"https://www.bild.de/sport/thema-{CMS_B}.html",
            }
        )
        == CMS_B
    )
    assert resolve_cms_id({"id": "https://www.bild.de/news/no-document-id"}) is None


def test_internal_overlay_keeps_only_fresh_api_scores_and_never_falls_back():
    now = datetime(2026, 7, 20, 6, 15, tzinfo=timezone.utc)

    def batch_transport(_url, _headers, _body_bytes, _timeout):
        return 200, _batch_body(
            [
                _found(CMS_A, 87.4, "2026-07-20T06:12:00Z"),
                _found(CMS_B, 99.0, "2026-07-20T05:00:00Z"),
            ]
        )

    client = ScoreApiClient(BASE_URL, API_KEY, batch_transport=batch_transport)
    articles = [
        {
            "id": f"https://www.bild.de/news/a-{CMS_A}.html",
            "url": f"https://www.bild.de/news/a-{CMS_A}.html",
            "title": "Synthetic A",
            "score": 55.0,
            "pubDate": "2026-07-20T06:10:00Z",
        },
        {
            "id": f"https://www.bild.de/news/b-{CMS_B}.html",
            "url": f"https://www.bild.de/news/b-{CMS_B}.html",
            "title": "Synthetic B",
            "score": 98.0,
            "pubDate": "2026-07-20T06:11:00Z",
        },
        {
            "id": "https://www.bild.de/news/no-id.html",
            "url": "https://www.bild.de/news/no-id.html",
            "title": "Synthetic missing",
            "score": 100.0,
            "pubDate": "2026-07-20T06:14:00Z",
        },
    ]

    ranked = _apply_internal_score_api_scores(
        articles,
        client=client,
        now=now,
        max_age_seconds=900,
    )

    assert ranked[0]["cmsId"] == CMS_A
    assert ranked[0]["score"] == 87.4
    assert ranked[0]["scoreSource"] == "internal_score_api"
    stale = next(item for item in ranked if item.get("cmsId") == CMS_B)
    missing = next(item for item in ranked if item.get("cmsId") is None)
    assert stale["score"] == 0.0
    assert stale["scoreSource"] == "internal_score_api_stale"
    assert missing["score"] == 0.0
    assert missing["scoreBeforeInternalApi"] == 100.0


def test_internal_overlay_uses_source_eight_hour_freshness_boundary():
    now = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)

    def batch_transport(_url, _headers, _body_bytes, _timeout):
        return 200, _batch_body(
            [
                _found(CMS_A, 87.4, "2026-07-20T06:00:01Z"),
                _found(CMS_B, 99.0, "2026-07-20T06:00:00Z"),
            ]
        )

    client = ScoreApiClient(BASE_URL, API_KEY, batch_transport=batch_transport)
    articles = [
        {"url": f"https://www.bild.de/news/a-{CMS_A}", "score": 10.0},
        {"url": f"https://www.bild.de/news/b-{CMS_B}", "score": 10.0},
    ]

    ranked = _apply_internal_score_api_scores(articles, client=client, now=now)

    accepted = next(item for item in ranked if item["cmsId"] == CMS_A)
    boundary = next(item for item in ranked if item["cmsId"] == CMS_B)
    assert accepted["scoreSource"] == "internal_score_api"
    assert accepted["pushBalancerScoreAgeSeconds"] == 28_799
    assert boundary["scoreSource"] == "internal_score_api_stale"
    assert boundary["pushBalancerScoreAgeSeconds"] == 28_800


def test_article_score_age_is_timezone_safe():
    scored_at = datetime(2026, 7, 20, 6, 12, tzinfo=timezone.utc)
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        transport=lambda *_args: (200, _body(scored_at=scored_at.isoformat())),
    )
    result = client.get_score(CMS_A)

    assert result.age_seconds(scored_at + timedelta(seconds=30)) == 30
