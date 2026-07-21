"""Focused tests for the browser score capture projection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import database
from app import main as main_module
from app.main import app
from app.routers import score_capture


CMS_ID = "0123456789abcdef01234567"
ARTICLE_URL = f"https://www.bild.de/politik/synthetic-score-{CMS_ID}"
NOW = 1_800_000_000
client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def isolated_score_cache():
    with score_capture._lock:
        previous = dict(score_capture._score_cache)
        score_capture._score_cache.clear()
    try:
        yield
    finally:
        with score_capture._lock:
            score_capture._score_cache.clear()
            score_capture._score_cache.update(previous)


def _cache_score(*, score: float, ts: int, url: str = ARTICLE_URL) -> None:
    key = score_capture.hashlib.md5(score_capture._normalize_url(url).encode()).hexdigest()
    with score_capture._lock:
        score_capture._score_cache[key] = {"score": score, "ts": ts, "url": url}


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
    conn.execute("UPDATE article_score_log SET captured_at = ?", (NOW - 181,))
    conn.commit()
    conn.close()

    response = client.get(f"/api/score-capture/by-cms-id/{CMS_ID}")

    assert response.status_code == 404


def test_endpoint_remains_behind_internal_cidr_gate(monkeypatch):
    monkeypatch.setattr(score_capture.time, "time", lambda: NOW)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
    monkeypatch.setattr(main_module, "INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])
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
    assert allowed.status_code == 200
    assert blocked_health.status_code == 404
    assert allowed_health.status_code == 200


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
    assert "item.url + '=' + item.score.toFixed(1)" in html
    assert "if (now - _captureThrottle < 30000) return;" not in html
