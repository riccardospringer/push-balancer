"""Security and parity tests for the legacy Render UI score capture client."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import Mock

import pytest

from app import render_score_capture as capture

CMS_ID = "0123456789abcdef01234567"
OTHER_CMS_ID = "fedcba987654321001234567"
NOW = 1_800_000_000.0
CAPTURED_AT_TIMESTAMP = int(NOW - 30)
CAPTURED_AT = "2027-01-15T07:59:30Z"
ENGAGEMENT_BREAKDOWN = {
    "kind": "engagement",
    "relevance": 30,
    "urgency": 0,
    "curiosity": 7.6,
    "freshness": 11.7,
    "timing": 6,
    "titleBoost": 3,
    "breaking": 0,
    "research": 0,
    "pushHistory": 0,
    "topicSaturation": 0,
}
SPORT_BREAKDOWN = {
    "kind": "sport",
    "sportRelevance": 32,
    "timing": 18,
    "drama": 12,
    "freshness": 8,
}


def _payload(
    *,
    score=54.3,
    captured_at=CAPTURED_AT_TIMESTAMP,
    score_breakdown=None,
    or_factor=None,
):
    payload = {"score": score, "capturedAt": captured_at}
    if score_breakdown is not None or or_factor is not None:
        payload["scoreBreakdown"] = score_breakdown
        payload["orFactor"] = or_factor
    return payload


def test_returns_fresh_ui_capture(monkeypatch):
    read_capture = Mock(return_value=_payload())
    monkeypatch.setattr(capture, "_read_capture", read_capture)

    result = capture.get_captured_score(CMS_ID, now=NOW)

    assert result is not None
    assert result.score == 54.3
    assert result.captured_at == CAPTURED_AT
    assert result.score_breakdown is None
    assert result.or_factor is None
    read_capture.assert_called_once_with(CMS_ID)


def test_returns_exact_engagement_breakdown_without_recomputing_total(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_read_capture",
        lambda _cms_id: _payload(
            score=58.3,
            score_breakdown=ENGAGEMENT_BREAKDOWN,
            or_factor=1.06,
        ),
    )

    result = capture.get_captured_score(CMS_ID, now=NOW)

    assert result is not None
    assert result.score == 58.3
    assert result.score_breakdown == capture.EngagementScoreBreakdown(
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
    )
    assert result.or_factor == 1.06


def test_returns_exact_sport_breakdown(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_read_capture",
        lambda _cms_id: _payload(
            score=70,
            score_breakdown=SPORT_BREAKDOWN,
            or_factor=0.94,
        ),
    )

    result = capture.get_captured_score(CMS_ID, now=NOW)

    assert result is not None
    assert result.score_breakdown == capture.SportScoreBreakdown(
        kind="sport",
        sport_relevance=32,
        timing=18,
        drama=12,
        freshness=8,
    )
    assert result.or_factor == 0.94


def test_capture_remains_available_after_previous_three_minute_cutoff(monkeypatch):
    captured_at = int(NOW - 181)
    monkeypatch.setattr(
        capture,
        "_read_capture",
        lambda _cms_id: _payload(captured_at=captured_at),
    )

    result = capture.get_captured_score(CMS_ID, now=NOW)

    assert result is not None
    assert result.score == 54.3
    assert result.captured_at == "2027-01-15T07:56:59Z"


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 54.3},
        {
            "score": 54.3,
            "capturedAt": CAPTURED_AT_TIMESTAMP,
            "url": "https://example.invalid",
        },
        {
            "score": 54.3,
            "capturedAt": CAPTURED_AT_TIMESTAMP,
            "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
        },
        {
            "score": 54.3,
            "capturedAt": CAPTURED_AT_TIMESTAMP,
            "scoreBreakdown": None,
            "orFactor": None,
        },
        _payload(score=True),
        _payload(score="54.3"),
        _payload(score=10**400),
        _payload(score=0),
        _payload(score=101),
        _payload(captured_at=True),
        _payload(captured_at="1799999970"),
        _payload(captured_at=NOW - 30.5),
        _payload(captured_at=10**400),
        _payload(captured_at=int(NOW + 31)),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN),
        _payload(or_factor=1.0),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN, or_factor=True),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN, or_factor="1.0"),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN, or_factor=10**400),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN, or_factor=0.59),
        _payload(score_breakdown=ENGAGEMENT_BREAKDOWN, or_factor=1.51),
        _payload(score_breakdown=None, or_factor=1.0),
    ],
)
def test_rejects_invalid_capture_payload(monkeypatch, payload):
    monkeypatch.setattr(capture, "_read_capture", lambda _cms_id: payload)

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("relevance", -0.1),
        ("relevance", 30.1),
        ("relevance", 10**400),
        ("urgency", 25.1),
        ("curiosity", True),
        ("freshness", "11.7"),
        ("timing", 15.1),
        ("titleBoost", 15.1),
        ("breaking", 15.1),
        ("research", 12.1),
        ("pushHistory", -4.1),
        ("pushHistory", 8.1),
        ("topicSaturation", -30.1),
        ("topicSaturation", 0.1),
    ],
)
def test_rejects_invalid_engagement_breakdown_number(monkeypatch, field, invalid_value):
    invalid_breakdown = ENGAGEMENT_BREAKDOWN | {field: invalid_value}
    monkeypatch.setattr(
        capture,
        "_read_capture",
        lambda _cms_id: _payload(
            score_breakdown=invalid_breakdown,
            or_factor=1.0,
        ),
    )

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)


@pytest.mark.parametrize(
    "invalid_breakdown",
    [
        {},
        {"kind": "unknown"},
        ENGAGEMENT_BREAKDOWN | {"extra": 1},
        {key: value for key, value in ENGAGEMENT_BREAKDOWN.items() if key != "timing"},
        SPORT_BREAKDOWN | {"sportRelevance": 35.1},
        SPORT_BREAKDOWN | {"timing": 30.1},
        SPORT_BREAKDOWN | {"drama": 25.1},
        SPORT_BREAKDOWN | {"freshness": 10.1},
        SPORT_BREAKDOWN | {"drama": False},
        SPORT_BREAKDOWN | {"extra": 1},
        "not-an-object",
    ],
)
def test_rejects_invalid_score_breakdown_shape(monkeypatch, invalid_breakdown):
    monkeypatch.setattr(
        capture,
        "_read_capture",
        lambda _cms_id: _payload(
            score_breakdown=invalid_breakdown,
            or_factor=1.0,
        ),
    )

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)


def test_stale_capture_returns_none(monkeypatch):
    stale_payload = _payload(
        captured_at=int(NOW - capture._MAX_CAPTURE_AGE_SECONDS)
    )
    monkeypatch.setattr(capture, "_read_capture", lambda _cms_id: stale_payload)

    assert capture.get_captured_score(CMS_ID, now=NOW) is None


def test_404_returns_none(monkeypatch):
    def not_found(_request):
        raise urllib.error.HTTPError(
            f"{capture._CAPTURE_BASE_URL}/{CMS_ID}",
            404,
            "synthetic not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(capture, "_open_request", not_found)

    assert capture.get_captured_score(CMS_ID, now=NOW) is None


def test_rejects_invalid_cms_id_without_network_call(monkeypatch):
    read_capture = Mock(side_effect=AssertionError("network must not run"))
    monkeypatch.setattr(capture, "_read_capture", read_capture)

    assert capture.get_captured_score("not-a-cms-id", now=NOW) is None
    read_capture.assert_not_called()


def test_capture_request_is_fixed_https_and_credential_free(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(_payload()).encode()

    def fake_open(request):
        requests.append(request)
        return Response()

    monkeypatch.setattr(capture, "_open_request", fake_open)

    assert capture._read_capture(CMS_ID.upper()) == _payload()
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == (
        f"{capture._CAPTURE_BASE_URL}/{CMS_ID}?{capture._CAPTURE_QUERY}"
    )
    assert request.full_url.endswith("?includeBreakdown=1")
    assert request.method == "GET"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert set(headers) == {"accept", "user-agent"}


def test_batch_uses_one_fixed_post_and_returns_strict_ordered_results(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(
                {
                    "results": [
                        {
                            "cmsId": CMS_ID,
                            "status": "found",
                            **_payload(
                                score=58.3,
                                score_breakdown=ENGAGEMENT_BREAKDOWN,
                                or_factor=1.06,
                            ),
                        },
                        {
                            "cmsId": OTHER_CMS_ID,
                            "status": "notFound",
                        },
                    ]
                }
            ).encode()

    def fake_open(request):
        requests.append(request)
        return Response()

    monkeypatch.setattr(capture, "_open_request", fake_open)

    source_ids, results = capture.get_captured_scores_batch(
        [CMS_ID.upper(), CMS_ID, OTHER_CMS_ID],
        now=NOW,
    )

    assert source_ids == [CMS_ID, OTHER_CMS_ID]
    assert results[0] is not None
    assert results[0].score == 58.3
    assert results[0].or_factor == 1.06
    assert results[1] is None
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == (
        f"{capture._CAPTURE_BATCH_URL}?{capture._CAPTURE_QUERY}"
    )
    assert request.method == "POST"
    assert json.loads(request.data) == {"cmsIds": [CMS_ID, OTHER_CMS_ID]}
    assert request.data == (
        f'{{"cmsIds":["{CMS_ID}","{OTHER_CMS_ID}"]}}'.encode()
    )
    headers = {key.lower(): value for key, value in request.header_items()}
    assert set(headers) == {"accept", "content-type", "user-agent"}
    assert headers["content-type"] == "application/json"


def test_batch_accepts_exact_legacy_found_item(monkeypatch):
    monkeypatch.setattr(
        capture,
        "_read_capture_batch",
        lambda _cms_ids: {
            "results": [
                {
                    "cmsId": CMS_ID,
                    "status": "found",
                    **_payload(),
                }
            ]
        },
    )

    source_ids, results = capture.get_captured_scores_batch([CMS_ID], now=NOW)

    assert source_ids == [CMS_ID]
    assert results == [
        capture.CapturedScore(
            score=54.3,
            captured_at=CAPTURED_AT,
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"results": [], "extra": True},
        {"results": []},
        {"results": [{"cmsId": OTHER_CMS_ID, "status": "notFound"}]},
        {"results": [{"cmsId": CMS_ID, "status": "missing"}]},
        {"results": [{"cmsId": CMS_ID, "status": "notFound", "extra": True}]},
        {
            "results": [
                {
                    "cmsId": CMS_ID,
                    "status": "found",
                    **_payload(),
                    "extra": True,
                }
            ]
        },
        {
            "results": [
                {
                    "cmsId": CMS_ID,
                    "status": "found",
                    **_payload(score=0),
                }
            ]
        },
        {
            "results": [
                {
                    "cmsId": CMS_ID,
                    "status": "found",
                    **_payload(captured_at=CAPTURED_AT_TIMESTAMP + 0.5),
                }
            ]
        },
        {
            "results": [
                {
                    "cmsId": CMS_ID,
                    "status": "found",
                    "score": 54.3,
                }
            ]
        },
    ],
)
def test_batch_rejects_malformed_whole_source_response(monkeypatch, payload):
    monkeypatch.setattr(capture, "_read_capture_batch", lambda _cms_ids: payload)

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch([CMS_ID], now=NOW)


@pytest.mark.parametrize("cms_ids", [[], [CMS_ID] * 501, ["invalid"], [123]])
def test_batch_rejects_invalid_request_before_network(monkeypatch, cms_ids):
    read_batch = Mock(side_effect=AssertionError("network must not run"))
    monkeypatch.setattr(capture, "_read_capture_batch", read_batch)

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch(cms_ids, now=NOW)

    read_batch.assert_not_called()


def test_batch_rejects_http_errors_and_oversized_response(monkeypatch):
    def fail(_request):
        raise urllib.error.HTTPError(
            capture._CAPTURE_BATCH_URL,
            503,
            "synthetic error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(capture, "_open_request", fail)
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch([CMS_ID], now=NOW)

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"x" * (capture._MAX_BATCH_RESPONSE_BYTES + 1)

    monkeypatch.setattr(capture, "_open_request", lambda _request: OversizedResponse())
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch([CMS_ID], now=NOW)


def test_single_and_batch_normalize_truncated_protocol_reads(monkeypatch):
    class TruncatedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            raise capture.http.client.IncompleteRead(b'{"partial":', 20)

    monkeypatch.setattr(capture, "_open_request", lambda _request: TruncatedResponse())

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch([CMS_ID], now=NOW)


def test_single_and_batch_normalize_pathological_json_parser_failure(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"synthetic":"bounded"}'

    monkeypatch.setattr(capture, "_open_request", lambda _request: Response())
    monkeypatch.setattr(
        capture.json,
        "loads",
        Mock(side_effect=RecursionError("synthetic pathological nesting")),
    )

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_scores_batch([CMS_ID], now=NOW)


@pytest.mark.parametrize("status", [302, 500])
def test_capture_rejects_non_404_error_status(monkeypatch, status):
    def fail(_request):
        raise urllib.error.HTTPError(
            f"{capture._CAPTURE_BASE_URL}/{CMS_ID}",
            status,
            "synthetic error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(capture, "_open_request", fail)

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)


def test_capture_rejects_oversized_or_invalid_payload(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return self.body

    monkeypatch.setattr(
        capture,
        "_open_request",
        lambda _request: Response(b"x" * (capture._MAX_RESPONSE_BYTES + 1)),
    )
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)

    monkeypatch.setattr(
        capture,
        "_open_request",
        lambda _request: Response(json.dumps([]).encode()),
    )
    with pytest.raises(capture.RenderScoreUnavailable):
        capture.get_captured_score(CMS_ID, now=NOW)


def test_health_probe_uses_fixed_https_endpoint_without_credentials(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"status":"ok"}'

    def fake_open(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(capture, "_open_request", fake_open)

    capture.require_capture_source_ready()

    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == capture._CAPTURE_HEALTH_URL
    assert request.method == "GET"
    assert timeout == capture._HEALTH_TIMEOUT_SECONDS
    headers = {key.lower(): value for key, value in request.header_items()}
    assert set(headers) == {"accept", "user-agent"}


@pytest.mark.parametrize("body", [b"{}", b'{"status":"down"}', b"[]", b"not-json"])
def test_health_probe_rejects_invalid_payload(monkeypatch, body):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return body

    monkeypatch.setattr(
        capture,
        "_open_request",
        lambda _request, *, timeout: Response(),
    )

    with pytest.raises(capture.RenderScoreUnavailable):
        capture.require_capture_source_ready()
