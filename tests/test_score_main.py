"""Surface and fail-closed tests for the score-only ASGI entrypoint."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import auth, config, score_main
from app.ml import lightgbm_model
from app.routers import score_api

CMS_ID = "0123456789abcdef01234567"
SCORE_KEY = "synthetic-score-key"


def test_score_only_app_exposes_no_ui_docs_or_mutation_routes():
    direct_paths = {
        route.path
        for route in score_main.app.routes
        if isinstance(getattr(route, "path", None), str)
    }
    documented_paths = set(score_main.app.openapi()["paths"])

    assert direct_paths | documented_paths == {
        "/api/health",
        "/api/ready",
        "/api/v1/scores/batch",
        "/api/v1/scores/{cms_id}",
    }


def test_score_only_health_is_minimal_after_successful_startup(monkeypatch):
    monkeypatch.setattr(config, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", False)
    with TestClient(score_main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_score_only_readiness_checks_render_source(monkeypatch):
    ready = Mock()
    monkeypatch.setattr(score_main, "require_capture_source_ready", ready)

    response = TestClient(score_main.app).get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    ready.assert_called_once_with()


def test_score_only_readiness_fails_closed_when_render_source_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        score_main,
        "require_capture_source_ready",
        Mock(side_effect=score_main.RenderScoreUnavailable("synthetic unavailable")),
    )

    response = TestClient(score_main.app).get("/api/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Score source is unavailable."


def test_score_only_startup_fails_closed_without_frozen_seed(monkeypatch):
    monkeypatch.setattr(config, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(lightgbm_model, "load_seed_model", lambda: False)

    with pytest.raises(RuntimeError, match="seed model is unavailable"):
        with TestClient(score_main.app):
            pass


def test_score_only_startup_fails_closed_on_seed_integrity_mismatch(monkeypatch):
    monkeypatch.setattr(config, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(score_main, "_FROZEN_SEED_SHA256", "0" * 64)
    loader = Mock(side_effect=AssertionError("untrusted seed must not be loaded"))
    monkeypatch.setattr(lightgbm_model, "load_seed_model", loader)

    with pytest.raises(RuntimeError, match="integrity validation"):
        with TestClient(score_main.app):
            pass

    loader.assert_not_called()


def test_score_only_route_keeps_identifier_out_of_errors(monkeypatch):
    monkeypatch.setattr(auth, "SCORE_API_KEY", SCORE_KEY)
    monkeypatch.setattr(
        score_api,
        "get_captured_score",
        lambda _cms_id: None,
    )
    monkeypatch.setattr(config, "INTERNAL_ACCESS_ENABLED", False)
    client = TestClient(score_main.app)

    response = client.get(
        f"/api/v1/scores/{CMS_ID}",
        headers={"X-Score-Key": SCORE_KEY},
    )

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert "X-Score-Key" in response.headers["vary"]
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
    assert CMS_ID not in response.text


def test_score_only_network_denial_is_non_cacheable_and_redacted(monkeypatch):
    monkeypatch.setattr(config, "INTERNAL_ACCESS_ENABLED", True)
    monkeypatch.setattr(config, "INTERNAL_ACCESS_ALLOWED_CIDRS", ["192.0.2.0/24"])
    client = TestClient(score_main.app)

    response = client.get(
        f"/api/v1/scores/{CMS_ID}",
        headers={"X-Score-Key": SCORE_KEY},
    )

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["instance"] == "/api/v1/scores/{cms_id}"
    assert CMS_ID not in response.text
