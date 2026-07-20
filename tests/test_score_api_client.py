import json
from datetime import datetime, timedelta, timezone

import pytest

from app.routers.feed import _apply_internal_score_api_scores
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


def _body(cms_id=CMS_A, score=87.4, scored_at="2026-07-20T06:12:00Z"):
    return json.dumps({"cmsId": cms_id, "score": score, "scoredAt": scored_at}).encode("utf-8")


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
    ],
)
def test_score_client_rejects_untrustworthy_success_payload(body):
    client = ScoreApiClient(BASE_URL, API_KEY, transport=lambda *_args: (200, body))

    with pytest.raises(ScoreApiUnavailable):
        client.get_score(CMS_A)


def test_bounded_batch_distinguishes_not_found_and_unavailable():
    def transport(url, _headers, _timeout):
        if url.endswith(CMS_A):
            return 404, b"{}"
        raise TimeoutError("synthetic timeout")

    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        transport=transport,
        max_retries=0,
        cache_ttl_seconds=0,
    )

    results = fetch_score_lookups([CMS_A, CMS_B, CMS_A], client, max_concurrency=16)

    assert results[CMS_A].status == "not_found"
    assert results[CMS_B].status == "unavailable"


def test_cms_id_resolution_prefers_field_and_uses_strict_url_fallback():
    assert resolve_cms_id({"cmsId": CMS_A.upper(), "url": f"https://x/{CMS_B}"}) == CMS_A
    assert resolve_cms_id({"url": f"https://www.bild.de/news/thema-{CMS_B}.html"}) == CMS_B
    assert resolve_cms_id({"id": "https://www.bild.de/news/no-document-id"}) is None


def test_internal_overlay_keeps_only_fresh_api_scores_and_never_falls_back():
    now = datetime(2026, 7, 20, 6, 15, tzinfo=timezone.utc)

    def transport(url, _headers, _timeout):
        if url.endswith(CMS_A):
            return 200, _body(CMS_A, 87.4, "2026-07-20T06:12:00Z")
        return 200, _body(CMS_B, 99.0, "2026-07-20T05:00:00Z")

    client = ScoreApiClient(BASE_URL, API_KEY, transport=transport, cache_ttl_seconds=0)
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


def test_article_score_age_is_timezone_safe():
    scored_at = datetime(2026, 7, 20, 6, 12, tzinfo=timezone.utc)
    client = ScoreApiClient(
        BASE_URL,
        API_KEY,
        transport=lambda *_args: (200, _body(scored_at=scored_at.isoformat())),
    )
    result = client.get_score(CMS_A)

    assert result.age_seconds(scored_at + timedelta(seconds=30)) == 30
