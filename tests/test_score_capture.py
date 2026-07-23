"""Focused tests for the browser score capture projection."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import database
from app import main as main_module
from app.main import app
from app.routers import score_capture


CMS_ID = "0123456789abcdef01234567"
ARTICLE_URL = f"https://www.bild.de/politik/synthetic-score-{CMS_ID}"
CMS_ID_2 = "89abcdef0123456701234567"
ARTICLE_URL_2 = f"https://www.bild.de/sport/synthetic-score-{CMS_ID_2}"
MISSING_CMS_ID = "aaaaaaaaaaaaaaaaaaaaaaaa"
NOW = 1_800_000_000
client = TestClient(app, raise_server_exceptions=True)
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
SPORT_BREAKDOWN = {
    "kind": "sport",
    "sportRelevance": 30.0,
    "timing": 28.4,
    "drama": 8.0,
    "freshness": 9.2,
}


@pytest.fixture(autouse=True)
def isolated_score_cache(tmp_db):
    del tmp_db
    with score_capture._lock:
        previous = dict(score_capture._score_cache)
        score_capture._score_cache.clear()
    try:
        yield
    finally:
        with score_capture._lock:
            score_capture._score_cache.clear()
            score_capture._score_cache.update(previous)


def _cache_score(
    *,
    score: float,
    ts: int,
    url: str = ARTICLE_URL,
    score_breakdown: dict | None = None,
    or_factor: float | None = None,
) -> None:
    key = score_capture.hashlib.md5(score_capture._normalize_url(url).encode()).hexdigest()
    entry = {"score": score, "ts": ts, "url": url}
    if score_breakdown is not None and or_factor is not None:
        entry["scoreBreakdown"] = score_breakdown
        entry["orFactor"] = or_factor
    with score_capture._lock:
        score_capture._score_cache[key] = entry


def test_endpoint_returns_only_newest_fresh_memory_score(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=53.2, ts=NOW - 30)
    _cache_score(
        score=54.3,
        ts=NOW - 5,
        url=f"https://www.bild.de/news/newer-synthetic-score-{CMS_ID}",
    )

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 200
    assert response.json() == {"score": 54.3, "capturedAt": NOW - 5}
    assert response.headers["cache-control"] == "no-store"


def test_default_endpoint_remains_exact_legacy_shape_with_enriched_capture(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(
        score=58.3,
        ts=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 200
    assert response.json() == {"score": 58.3, "capturedAt": NOW - 5}


def test_legacy_single_lookup_keeps_accepting_uppercase_cms_id(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=55.1, ts=NOW - 5)

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID.upper()}")

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW - 5}


def test_opt_in_endpoint_returns_exact_engagement_tooltip_values(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(
        score=58.3,
        ts=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json() == {
        "score": 58.3,
        "capturedAt": NOW - 5,
        "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
        "orFactor": 1.06,
    }
    assert response.headers["cache-control"] == "no-store"


def test_opt_in_endpoint_returns_sport_breakdown(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(
        score=75.6,
        ts=NOW - 5,
        score_breakdown=SPORT_BREAKDOWN,
        or_factor=1.2,
    )

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json()["scoreBreakdown"] == SPORT_BREAKDOWN
    assert response.json()["orFactor"] == 1.2


def test_opt_in_legacy_capture_keeps_exact_legacy_shape(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=55.1, ts=NOW - 5)

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW - 5}


def test_batch_returns_found_and_not_found_in_input_order(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(database.time, "time", lambda: NOW)
    _cache_score(
        score=58.3,
        ts=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )
    database.save_article_score_to_db(
        ARTICLE_URL_2,
        75.6,
        captured_at=NOW - 10,
    )

    response = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        json={"cmsIds": [CMS_ID_2, MISSING_CMS_ID, CMS_ID]},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "results": [
            {
                "cmsId": CMS_ID_2,
                "status": "found",
                "score": 75.6,
                "capturedAt": NOW - 10,
            },
            {"cmsId": MISSING_CMS_ID, "status": "notFound"},
            {
                "cmsId": CMS_ID,
                "status": "found",
                "score": 58.3,
                "capturedAt": NOW - 5,
                "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
                "orFactor": 1.06,
            },
        ]
    }


def test_batch_uses_one_memory_scan_and_one_database_scan(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=55.1, ts=NOW - 5)
    _cache_score(score=60.2, ts=NOW - 6, url=ARTICLE_URL_2)
    database_calls = 0
    url_scans = 0
    original_url_scanner = score_capture._cms_ids_in_trusted_url

    def counted_database_scan(cms_ids, *, max_age_seconds):
        nonlocal database_calls
        database_calls += 1
        assert cms_ids == [CMS_ID, CMS_ID_2]
        assert max_age_seconds == score_capture._CACHE_TTL
        return {}

    def counted_url_scan(url):
        nonlocal url_scans
        url_scans += 1
        return original_url_scanner(url)

    monkeypatch.setattr(
        database,
        "get_article_score_snapshots_by_cms_ids_from_db",
        counted_database_scan,
    )
    monkeypatch.setattr(
        score_capture,
        "_cms_ids_in_trusted_url",
        counted_url_scan,
    )

    response = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        json={"cmsIds": [CMS_ID, CMS_ID_2]},
    )

    assert response.status_code == 200
    assert database_calls == 1
    assert url_scans == 2


def test_batch_newest_snapshot_wins_without_recalculation(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(database.time, "time", lambda: NOW)
    _cache_score(score=40.0, ts=NOW - 20)
    database.save_article_score_to_db(
        ARTICLE_URL,
        58.3,
        captured_at=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )

    response = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["score"] == 58.3
    assert response.json()["results"][0]["scoreBreakdown"] == ENGAGEMENT_BREAKDOWN


@pytest.mark.parametrize(
    "body",
    [
        {"cmsIds": []},
        {"cmsIds": [CMS_ID, CMS_ID]},
        {"cmsIds": [CMS_ID.upper()]},
        {"cmsIds": ["not-a-cms-id"]},
        {"cmsIds": [CMS_ID], "unexpected": True},
        {"cmsIds": [f"{value:024x}" for value in range(501)]},
    ],
)
def test_batch_rejects_non_exact_request_contract(body):
    response = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        json=body,
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?includeBreakdown=0",
        "?includeBreakdown=1&unexpected=1",
        "?includeBreakdown=1&includeBreakdown=1",
    ],
)
def test_batch_requires_exact_breakdown_query(query):
    response = client.post(
        f"/api/score-capture/by-cms-id/batch{query}",
        json={"cmsIds": [CMS_ID]},
    )

    assert response.status_code == 422


def test_score_source_read_failure_is_503_not_not_found(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise database.ArticleScoreReadError("synthetic")

    monkeypatch.setattr(
        database,
        "get_article_score_snapshots_by_cms_ids_from_db",
        unavailable,
    )

    batch = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        json={"cmsIds": [MISSING_CMS_ID]},
    )
    single = client.get(f"/api/score-capture/by-cms-id/{MISSING_CMS_ID}")

    assert batch.status_code == 503
    assert single.status_code == 503
    assert batch.json()["detail"] == "Score capture storage is unavailable."
    assert MISSING_CMS_ID not in batch.text


def test_only_literal_one_enables_breakdown(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=55.1, ts=NOW - 5)

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=0"
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


def test_missing_score_response_is_not_cacheable(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


def test_score_capture_health_is_minimal_and_not_cacheable():
    response = client.get("/api/score-capture/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_endpoint_uses_persistent_db_fallback(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(database.time, "time", lambda: NOW)
    database.save_article_score_to_db(ARTICLE_URL, 55.1)

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW}


def test_endpoint_uses_enriched_persistent_db_fallback(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    database.save_article_score_to_db(
        ARTICLE_URL,
        58.3,
        captured_at=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json() == {
        "score": 58.3,
        "capturedAt": NOW - 5,
        "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
        "orFactor": 1.06,
    }


def test_db_never_exposes_a_partial_enrichment_pair(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    database.save_article_score_to_db(ARTICLE_URL, 55.1, captured_at=NOW - 5)
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "UPDATE article_score_log SET score_breakdown_json = ?",
        ('{"kind":"engagement"}',),
    )
    conn.commit()
    conn.close()

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW - 5}


def test_database_newest_capture_wins_atomically(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    database.save_article_score_to_db(
        ARTICLE_URL,
        58.3,
        captured_at=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )
    database.save_article_score_to_db(
        ARTICLE_URL,
        40.0,
        captured_at=NOW - 10,
        score_breakdown=SPORT_BREAKDOWN,
        or_factor=0.9,
    )

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json()["score"] == 58.3
    assert response.json()["scoreBreakdown"] == ENGAGEMENT_BREAKDOWN
    assert response.json()["orFactor"] == 1.06


def test_newer_database_capture_wins_over_stale_memory_replay(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    database.save_article_score_to_db(
        ARTICLE_URL,
        58.3,
        captured_at=NOW - 5,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )
    _cache_score(
        score=40.0,
        ts=NOW - 10,
        score_breakdown=SPORT_BREAKDOWN,
        or_factor=0.9,
    )

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert response.status_code == 200
    assert response.json()["score"] == 58.3
    assert response.json()["scoreBreakdown"] == ENGAGEMENT_BREAKDOWN
    assert response.json()["orFactor"] == 1.06


def test_database_migrates_legacy_score_table(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE article_score_log (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            score REAL NOT NULL,
            captured_at INTEGER NOT NULL
        )"""
    )
    conn.commit()
    conn.close()

    with patch.object(database, "PUSH_DB_PATH", db_path):
        database.init_db()

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(article_score_log)")}
    conn.close()
    assert {"score_breakdown_json", "or_factor", "enrichment_captured_at"} <= columns


def test_legacy_upsert_cannot_attach_stale_enrichment_to_a_newer_score(
    tmp_db,
    monkeypatch,
):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    database.save_article_score_to_db(
        ARTICLE_URL,
        58.3,
        captured_at=NOW - 10,
        score_breakdown=ENGAGEMENT_BREAKDOWN,
        or_factor=1.06,
    )

    # Simulate a rollback to the legacy binary. Its UPSERT knows only the four
    # original columns, so the enrichment columns remain in the migrated table.
    normalized_url = score_capture._normalize_url(ARTICLE_URL)
    url_hash = score_capture.hashlib.md5(normalized_url.encode()).hexdigest()
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        """INSERT INTO article_score_log (url_hash, url, score, captured_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(url_hash) DO UPDATE SET
               score = excluded.score,
               captured_at = excluded.captured_at""",
        (url_hash, ARTICLE_URL, 61.2, NOW - 5),
    )
    conn.commit()
    stale_row = conn.execute(
        """SELECT score_breakdown_json, or_factor, enrichment_captured_at, captured_at
           FROM article_score_log WHERE url_hash = ?""",
        (url_hash,),
    ).fetchone()
    conn.close()
    assert stale_row[0] is not None
    assert stale_row[1] == 1.06
    assert stale_row[2] == NOW - 10
    assert stale_row[3] == NOW - 5

    rolled_forward = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert rolled_forward.status_code == 200
    assert rolled_forward.json() == {"score": 61.2, "capturedAt": NOW - 5}

    database.save_article_score_to_db(
        ARTICLE_URL,
        62.4,
        captured_at=NOW,
        score_breakdown=SPORT_BREAKDOWN,
        or_factor=1.2,
    )
    recaptured = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )
    assert recaptured.status_code == 200
    assert recaptured.json() == {
        "score": 62.4,
        "capturedAt": NOW,
        "scoreBreakdown": SPORT_BREAKDOWN,
        "orFactor": 1.2,
    }


def test_post_persists_complete_enrichment_pair(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)

    posted = client.post(
        "/api/score-capture",
        json={
            "scores": [
                {
                    "url": ARTICLE_URL,
                    "score": 58.3,
                    "ts": NOW - 5,
                    "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
                    "orFactor": 1.06,
                }
            ]
        },
    )
    with score_capture._lock:
        score_capture._score_cache.clear()
    read = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}?includeBreakdown=1"
    )

    assert posted.status_code == 200
    assert posted.json() == {"ok": True, "stored": 1}
    assert posted.headers["cache-control"] == "no-store"
    assert read.status_code == 200
    assert read.json()["scoreBreakdown"] == ENGAGEMENT_BREAKDOWN
    assert read.json()["orFactor"] == 1.06


def test_post_rejects_future_timestamp_before_memory_or_database_write(
    tmp_db,
    monkeypatch,
):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)

    response = client.post(
        "/api/score-capture",
        json={
            "scores": [
                {
                    "url": ARTICLE_URL,
                    "score": 58.3,
                    "ts": NOW + score_capture._MAX_FUTURE_SKEW_SECONDS + 1,
                }
            ]
        },
    )

    assert response.status_code == 422
    with score_capture._lock:
        assert score_capture._score_cache == {}
    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM article_score_log").fetchone()[0]
    conn.close()
    assert count == 0


def test_post_reports_storage_failure_so_browser_can_retry(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)

    def unavailable(*_args, **_kwargs):
        raise database.ArticleScoreWriteError("synthetic")

    monkeypatch.setattr(database, "save_article_score_to_db", unavailable)

    response = client.post(
        "/api/score-capture",
        json={
            "scores": [
                {
                    "url": ARTICLE_URL,
                    "score": 58.3,
                    "ts": NOW - 5,
                }
            ]
        },
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "patch_data",
    [
        {"orFactor": None},
        {"scoreBreakdown": None},
        {"orFactor": "1.06"},
        {"orFactor": 1.51},
        {"scoreBreakdown": {**ENGAGEMENT_BREAKDOWN, "relevance": 30.1}},
        {"scoreBreakdown": {**ENGAGEMENT_BREAKDOWN, "reason": "not allowed"}},
    ],
)
def test_post_rejects_incomplete_or_invalid_enrichment_pair(tmp_db, patch_data):
    item = {
        "url": ARTICLE_URL,
        "score": 58.3,
        "ts": NOW - 5,
        "scoreBreakdown": ENGAGEMENT_BREAKDOWN,
        "orFactor": 1.06,
    }
    item.update(patch_data)

    response = client.post("/api/score-capture", json={"scores": [item]})

    assert response.status_code == 422


def test_endpoint_keeps_original_ui_score_after_previous_three_minute_cutoff(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    _cache_score(score=55.1, ts=NOW - 181)

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW - 181}


def test_db_fallback_skips_newer_untrusted_url(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(database.time, "time", lambda: NOW)
    database.save_article_score_to_db(ARTICLE_URL, 55.1)
    database.save_article_score_to_db(f"https://evil.example/news/{CMS_ID}", 99.0)

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 200
    assert response.json() == {"score": 55.1, "capturedAt": NOW}


def test_endpoint_rejects_stale_db_score(tmp_db, monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(database.time, "time", lambda: NOW)
    database.save_article_score_to_db(ARTICLE_URL, 55.1)
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "UPDATE article_score_log SET captured_at = ?",
        (NOW - score_capture._CACHE_TTL - 1,),
    )
    conn.commit()
    conn.close()

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 404


def test_endpoint_remains_behind_internal_cidr_gate(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(main_module, "IS_RENDER", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
    monkeypatch.setattr(main_module, "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS", [])
    _cache_score(score=54.3, ts=NOW - 5)

    blocked = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}",
        headers={"CF-Connecting-IP": "192.0.2.10"},
    )
    allowed = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}",
        headers={"CF-Connecting-IP": "145.243.163.23"},
    )
    blocked_health = client.get(
        "/api/score-capture/health",
        headers={"CF-Connecting-IP": "192.0.2.10"},
    )
    allowed_health = client.get(
        "/api/score-capture/health",
        headers={"CF-Connecting-IP": "145.243.163.23"},
    )

    assert blocked.status_code == 404
    assert blocked.headers["cache-control"] == "no-store"
    assert allowed.status_code == 200
    assert blocked_health.status_code == 404
    assert blocked_health.headers["cache-control"] == "no-store"
    assert allowed_health.status_code == 200


def test_approved_next_egress_can_read_only_minimal_score_routes(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(main_module, "IS_RENDER", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
    monkeypatch.setattr(
        main_module,
        "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS",
        ["198.51.100.10/32"],
    )
    headers = {"CF-Connecting-IP": "198.51.100.10"}
    _cache_score(score=54.3, ts=NOW - 5)

    health = client.get("/api/score-capture/health", headers=headers)
    score = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}", headers=headers)
    batch = client.post(
        "/api/score-capture/by-cms-id/batch?includeBreakdown=1",
        headers=headers,
        json={"cmsIds": [CMS_ID]},
    )
    debug = client.get("/api/score-capture/debug", headers=headers)
    write = client.post("/api/score-capture", headers=headers, json={"scores": []})
    wrong_batch_method = client.get(
        "/api/score-capture/by-cms-id/batch",
        headers=headers,
    )
    unrelated = client.get("/api/pushes", headers=headers)

    assert health.status_code == 200
    assert score.status_code == 200
    assert score.json() == {"score": 54.3, "capturedAt": NOW - 5}
    assert batch.status_code == 200
    assert batch.json()["results"][0]["status"] == "found"
    assert debug.status_code == 404
    assert write.status_code == 404
    assert wrong_batch_method.status_code == 404
    assert unrelated.status_code == 404


@pytest.mark.parametrize(
    "spoofed_header",
    ["True-Client-IP", "X-Real-IP", "X-Forwarded-For"],
)
def test_render_score_gate_never_trusts_fallback_ip_headers(
    monkeypatch,
    spoofed_header,
):
    monkeypatch.setattr(main_module, "IS_RENDER", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", [])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
    monkeypatch.setattr(
        main_module,
        "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS",
        ["198.51.100.10/32"],
    )

    response = client.get(
        "/api/score-capture/health",
        headers={spoofed_header: "198.51.100.10"},
    )

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "cf_connecting_ip",
    [None, "", "not-an-ip", "198.51.100.10, 203.0.113.4"],
)
def test_render_score_gate_fails_closed_without_one_valid_cloudflare_ip(
    monkeypatch,
    cf_connecting_ip,
):
    monkeypatch.setattr(main_module, "IS_RENDER", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", [])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
    monkeypatch.setattr(
        main_module,
        "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS",
        ["198.51.100.10/32"],
    )
    headers = {
        "X-Forwarded-For": "198.51.100.10",
        "X-Real-IP": "198.51.100.10",
        "True-Client-IP": "198.51.100.10",
    }
    if cf_connecting_ip is not None:
        headers["CF-Connecting-IP"] = cf_connecting_ip

    response = client.get("/api/score-capture/health", headers=headers)

    assert response.status_code == 404


def test_blocked_score_source_log_redacts_cms_id_and_client_ip(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(main_module, "IS_RENDER", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", [])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
    monkeypatch.setattr(main_module, "SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS", [])
    caplog.set_level("WARNING", logger="push-balancer")
    caplog.clear()

    response = client.get(
        f"/api/score-capture/by-cms-id/{CMS_ID}",
        headers={"CF-Connecting-IP": "192.0.2.44"},
    )
    application_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "push-balancer"
    )

    assert response.status_code == 404
    assert "/api/score-capture/by-cms-id/{cms_id}" in application_logs
    assert CMS_ID not in application_logs
    assert "192.0.2.44" not in application_logs


def test_unhandled_score_source_log_redacts_cms_id_and_exception_message(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", False)

    def fail_with_identifier(_cms_id):
        raise RuntimeError(f"synthetic failure for {CMS_ID}")

    monkeypatch.setattr(
        score_capture,
        "get_score_snapshot_for_cms_id",
        fail_with_identifier,
    )
    caplog.set_level("ERROR", logger="push-balancer")
    caplog.clear()

    non_raising_client = TestClient(app, raise_server_exceptions=False)
    response = non_raising_client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")
    application_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "push-balancer"
    )

    assert response.status_code == 500
    assert "Unhandled RuntimeError" in application_logs
    assert "/api/score-capture/by-cms-id/{cms_id}" in application_logs
    assert CMS_ID not in application_logs
    assert "synthetic failure" not in application_logs


@pytest.mark.parametrize(
    "url",
    [
        f"https://evil.example/politik/{CMS_ID}",
        f"https://www.bild.de/politik/{CMS_ID}f",
        f"https://www.bild.de:invalid/politik/{CMS_ID}",
        f"https://www.bild.de/politik/article?cmsId={CMS_ID}",
    ],
)
def test_cms_id_match_accepts_only_exact_bild_path_token(url):
    assert score_capture._url_matches_cms_id(url, CMS_ID) is False


def test_invalid_cms_id_is_rejected_before_lookup(monkeypatch):
    monkeypatch.setattr(
        score_capture,
        "get_score_snapshot_for_cms_id",
        lambda _cms_id: pytest.fail("lookup must not run"),
    )

    response = client.get("/api/score-capture/by-cms-id/not-a-cms-id")

    assert response.status_code == 422


def test_frontend_captures_every_candidate_only_in_standard_filter():
    html = (Path(__file__).resolve().parents[1] / "push-balancer.html").read_text(
        encoding="utf-8"
    )

    assert "if (currentFilter === 'all') {\n    _captureArticleScores(filteredArticles);\n  }" in html
    assert "_captureArticleScores(filteredArticles.slice(0, 20))" not in html


def test_frontend_recaptures_immediately_when_visible_scores_change():
    html = (Path(__file__).resolve().parents[1] / "push-balancer.html").read_text(
        encoding="utf-8"
    )

    assert "let _lastScoreCaptureFingerprint = '';" in html
    assert "fingerprint === _lastScoreCaptureFingerprint" in html
    assert "scoreBreakdown: item.scoreBreakdown || null" in html
    assert "orFactor: item.orFactor == null ? null : item.orFactor" in html
    fingerprint = html.split("const fingerprint = payload", 1)[1].split(
        ".sort()", 1
    )[0]
    assert "ts:" not in fingerprint
    assert "if (now - _captureThrottle < 30000) return;" not in html


def test_frontend_capture_retries_failures_and_uses_bounded_chunks():
    html = (Path(__file__).resolve().parents[1] / "push-balancer.html").read_text(
        encoding="utf-8"
    )
    capture = html.split("async function _captureArticleScores", 1)[1].split(
        "// Server-Batch-Prediction", 1
    )[0]

    assert "const _SCORE_CAPTURE_CHUNK_SIZE = 100;" in html
    assert "payload.slice(offset, offset + _SCORE_CAPTURE_CHUNK_SIZE)" in capture
    assert "if (!response.ok) return;" in capture
    assert capture.index("if (!response.ok) return;") < capture.index(
        "_captureThrottle = now;"
    )
    assert capture.index("if (!response.ok) return;") < capture.index(
        "_lastScoreCaptureFingerprint = fingerprint;"
    )
    assert "_scoreCaptureInFlight = false;" in capture


def test_frontend_drops_invalid_enrichment_but_keeps_legacy_total():
    html = (Path(__file__).resolve().parents[1] / "push-balancer.html").read_text(
        encoding="utf-8"
    )
    projection = html.split(
        "function _captureScoreEnrichment(article) {", 1
    )[1].split("async function _captureArticleScores", 1)[0]
    capture = html.split("async function _captureArticleScores", 1)[1].split(
        "// Server-Batch-Prediction", 1
    )[0]

    assert "article._xorFactor < 0.6 || article._xorFactor > 1.5" in projection
    assert "if (!Object.entries(bounds).every" in projection
    assert "))) return null;" in projection
    assert "return enrichment ? { ...item, ...enrichment } : item;" in capture


def test_frontend_capture_is_numeric_allowlist_matching_tooltip_fields():
    html = (Path(__file__).resolve().parents[1] / "push-balancer.html").read_text(
        encoding="utf-8"
    )
    capture_projection = html.split(
        "function _captureScoreEnrichment(article) {", 1
    )[1].split("async function _captureArticleScores", 1)[0]

    assert "isSportArticle(article) && Number.isFinite(b.sportRelevanz)" in capture_projection
    for mapping in [
        "relevance: numericOrZero(b.relevanz)",
        "urgency: numericOrZero(b.dringlichkeit)",
        "curiosity: numericOrZero(b.neugier)",
        "freshness: numericOrZero(b['aktualit\\u00e4t'])",
        "timing: numericOrZero(b.timing)",
        "titleBoost: numericOrZero(b.individual)",
        "breaking: numericOrZero(b.breaking)",
        "research: numericOrZero(b.forschung)",
        "pushHistory: numericOrZero(b.pushHistorie)",
        "topicSaturation: numericOrZero(b._topicPenalty)",
        "sportRelevance: numericOrZero(b.sportRelevanz)",
        "drama: numericOrZero(b.dramatik)",
        "freshness: numericOrZero(b.frische)",
        "orFactor: article._xorFactor",
    ]:
        assert mapping in capture_projection
    for excluded in ["title:", "description:", "topicType", "timingDetail", "_cachedOR"]:
        assert excluded not in capture_projection
