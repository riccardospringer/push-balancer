"""Synthetic HTTP contract fixture executed inside the release container."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from fastapi.testclient import TestClient

from app import auth, config
from app.render_score_capture import CapturedScore, EngagementScoreBreakdown
from app.routers import score_api
from app.score_main import app

CMS_ID = "0123456789abcdef01234567"
UNKNOWN_CMS_ID = "fedcba987654321001234567"
API_KEY = "synthetic-contract-key"
SCORED_AT = "2026-01-01T12:00:00Z"

auth.SCORE_API_KEY = API_KEY
config.INTERNAL_ACCESS_ENABLED = False
config.ARTICLE_PREDICTION_ENRICHMENT_ENABLED = False


def _captured_score(cms_id: str) -> CapturedScore | None:
    if cms_id != CMS_ID:
        return None
    return CapturedScore(
        score=75.6,
        captured_at=SCORED_AT,
        score_breakdown=EngagementScoreBreakdown(
            kind="engagement",
            relevance=30,
            urgency=10,
            curiosity=12,
            freshness=10,
            timing=8,
            title_boost=3,
            breaking=0,
            research=0,
            push_history=0,
            topic_saturation=0,
        ),
        or_factor=1.04,
    )


score_api.get_captured_score = _captured_score


def _captured_scores_batch(
    cms_ids: list[str],
) -> tuple[list[str], list[CapturedScore | None]]:
    return cms_ids, [_captured_score(cms_id) for cms_id in cms_ids]


score_api.get_captured_scores_batch = _captured_scores_batch

client = TestClient(app)

health = client.get("/api/health")
assert health.status_code == 200
assert health.json() == {"status": "healthy"}

unauthorised = client.get(f"/api/v1/scores/{CMS_ID}")
assert unauthorised.status_code == 401
assert unauthorised.headers["cache-control"] == "no-store"
assert CMS_ID not in unauthorised.text

success = client.get(
    f"/api/v1/scores/{CMS_ID}",
    headers={"X-Score-Key": API_KEY},
)
assert success.status_code == 200
assert success.json() == {
    "cmsId": CMS_ID,
    "score": 75.6,
    "scoredAt": SCORED_AT,
    "scoreBreakdown": {
        "kind": "engagement",
        "relevance": 30.0,
        "urgency": 10.0,
        "curiosity": 12.0,
        "freshness": 10.0,
        "timing": 8.0,
        "titleBoost": 3.0,
        "breaking": 0.0,
        "research": 0.0,
        "pushHistory": 0.0,
        "topicSaturation": 0.0,
    },
    "orFactor": 1.04,
}
assert success.headers["cache-control"] == "no-store"
assert "X-Score-Key" in success.headers["vary"]

batch = client.post(
    "/api/v1/scores/batch",
    headers={"X-Score-Key": API_KEY},
    json={"cmsIds": [CMS_ID, UNKNOWN_CMS_ID, CMS_ID]},
)
assert batch.status_code == 200
assert batch.json()["requestedCount"] == 3
assert batch.json()["uniqueCount"] == 2
assert batch.json()["foundCount"] == 2
assert batch.json()["notFoundCount"] == 1
assert [result["status"] for result in batch.json()["results"]] == [
    "found",
    "notFound",
    "found",
]
assert [result["cmsId"] for result in batch.json()["results"]] == [
    CMS_ID,
    UNKNOWN_CMS_ID,
    CMS_ID,
]
assert batch.headers["cache-control"] == "no-store"

not_found = client.get(
    f"/api/v1/scores/{UNKNOWN_CMS_ID}",
    headers={"X-Score-Key": API_KEY},
)
assert not_found.status_code == 404
assert not_found.json()["instance"] == "/api/v1/scores/{cms_id}"
assert UNKNOWN_CMS_ID not in not_found.text

invalid = client.get(
    "/api/v1/scores/invalid!cms-id",
    headers={"X-Score-Key": API_KEY},
)
assert invalid.status_code == 422
assert "invalid!cms-id" not in invalid.text

for unavailable_path in (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/articles",
):
    assert client.get(unavailable_path).status_code == 404

print(json.dumps({"contract": "ok", "score": 75.6}, separators=(",", ":")))
