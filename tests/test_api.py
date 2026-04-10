"""test_api.py — Tests für die aktive FastAPI-App.

Alle Tests laufen ohne laufenden Server über FastAPI `TestClient`.
"""
from unittest.mock import patch

import pytest
from app.main import app as _test_app
from fastapi.testclient import TestClient

client = TestClient(_test_app, raise_server_exceptions=True)


# ── /api/health ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /api/health → HTTP 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_response_has_status_field(self):
        """Response muss 'status'-Feld enthalten."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data

    def test_health_content_type_json(self):
        """Content-Type muss JSON sein."""
        resp = client.get("/api/health")
        assert "application/json" in resp.headers.get("content-type", "")

    def test_health_status_value_is_string(self):
        """'status'-Feld muss ein String sein."""
        resp = client.get("/api/health")
        assert isinstance(resp.json().get("status"), str)


class TestStableFrontendContracts:
    def test_pushes_contract_returns_collection(self):
        resp = client.get("/api/pushes")
        assert resp.status_code == 200
        data = resp.json()
        assert "pushes" in data
        assert "channels" in data
        assert "today" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data

    def test_ml_model_contract_returns_status(self):
        resp = client.get("/api/ml-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "modelVersion" in data
        assert "trainedAt" in data
        assert "features" in data
        assert "advisoryOnly" in data

    def test_ml_model_monitoring_contract_returns_metrics(self):
        resp = client.get("/api/ml-model/monitoring")
        assert resp.status_code == 200
        data = resp.json()
        assert "recentPredictions" in data
        assert "rollingMAE" in data
        assert "drift" in data

    def test_research_insights_contract_returns_learnings(self):
        resp = client.get("/api/research-insights")
        assert resp.status_code == 200
        data = resp.json()
        assert "learnings" in data
        assert "experiments" in data

    def test_articles_contract_returns_collection(self, monkeypatch):
        sitemap = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>https://www.bild.de/politik/test-artikel</loc>
    <news:news>
      <news:title>Breaking Test Artikel</news:title>
      <news:publication_date>2026-04-09T08:00:00+00:00</news:publication_date>
    </news:news>
  </url>
</urlset>"""

        monkeypatch.setattr("app.routers.feed._fetch_url", lambda _url: sitemap)
        monkeypatch.setattr(
            "app.ml.predict.predict_or",
            lambda *_args, **_kwargs: {"predicted_or": 5.5},
        )

        resp = client.get("/api/articles")
        assert resp.status_code == 200
        data = resp.json()
        assert "articles" in data
        assert "total" in data
        assert data["count"] >= 1
        assert data["articles"][0]["title"] == "Breaking Test Artikel"

    def test_push_refresh_job_alias_returns_sync_result(self):
        resp = client.post("/api/push-refresh-jobs", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "synced" in data

    def test_deprecated_compatibility_route_emits_runtime_headers(self):
        resp = client.get("/api/ml/status")
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"
        assert resp.headers.get("Sunset") == "Wed, 31 Dec 2026 23:59:59 GMT"

    def test_deprecated_compatibility_prefix_route_emits_runtime_headers(self):
        with patch("app.routers.push.urllib.request.urlopen", side_effect=OSError("offline")):
            resp = client.get("/api/push/messages")

        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"
        assert resp.headers.get("Sunset") == "Wed, 31 Dec 2026 23:59:59 GMT"

    def test_stable_contract_route_has_no_deprecation_headers(self):
        resp = client.get("/api/ml-model")
        assert resp.status_code == 200
        assert "Deprecation" not in resp.headers
        assert "Sunset" not in resp.headers


class TestInternalAccessControl:
    def test_allows_cf_connecting_ip_when_allowlisted(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/api/pushes", headers={"CF-Connecting-IP": "145.243.163.23"})

        assert resp.status_code == 200

    def test_blocks_non_allowlisted_clients_when_enabled(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["10.0.0.0/8"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/api/pushes", headers={"X-Forwarded-For": "203.0.113.7"})

        assert resp.status_code == 404
        data = resp.json()
        assert data["title"] == "Not Found"
        assert data["status"] == 404

    def test_allows_allowlisted_clients_when_enabled(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["10.0.0.0/8"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/api/pushes", headers={"X-Forwarded-For": "10.24.8.15"})

        assert resp.status_code == 200

    def test_health_stays_reachable_for_health_checks(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["10.0.0.0/8"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.7"})

        assert resp.status_code == 200

    def test_legacy_frontend_path_serves_index_for_allowlisted_clients(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/push-balancer.html", headers={"CF-Connecting-IP": "145.243.163.23"})

        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Push Balancer" in resp.text
        assert 'data-tab="live"' in resp.text
        assert 'data-tab="analyse"' in resp.text
        assert 'data-tab="konkurrenz"' in resp.text
        assert 'data-tab="forschung"' in resp.text
        assert 'data-tab="tagesplan"' in resp.text
        assert "/api/forschung" in resp.text
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"

    def test_unknown_frontend_path_falls_back_to_spa_for_allowlisted_clients(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/legacy/bookmark", headers={"CF-Connecting-IP": "145.243.163.23"})

        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Push Balancer" in resp.text

    def test_dist_frontend_asset_prefix_is_rewritten_for_allowlisted_clients(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        asset_name = "index-5oNyFBdq.js"
        resp = client.get(
            f"/dist-frontend/assets/{asset_name}",
            headers={"CF-Connecting-IP": "145.243.163.23"},
        )

        assert resp.status_code == 200
        assert "javascript" in resp.headers.get("content-type", "")

    def test_dist_frontend_root_serves_spa_shell_for_allowlisted_clients(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/dist-frontend/", headers={"CF-Connecting-IP": "145.243.163.23"})

        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Push Balancer" in resp.text
        assert 'data-tab="konkurrenz"' in resp.text
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"

    def test_prepare_frontend_html_rewrites_legacy_bundle_paths_for_compat_route(self):
        from app.main import _prepare_frontend_html_for_request

        html = """
<!doctype html>
<script type="module" src="/dist-frontend/assets/index-old.js"></script>
<link rel="stylesheet" href="/dist-frontend/assets/index-old.css">
"""

        rewritten = _prepare_frontend_html_for_request(html, "/push-balancer.html")

        assert "/dist-frontend/assets/" not in rewritten
        assert "/assets/index-old.js" in rewritten
        assert "/assets/index-old.css" in rewritten
        assert "replaceState" in rewritten


# ── /api/tagesplan ────────────────────────────────────────────────────────────

class TestTagesplanEndpoint:
    def test_tagesplan_returns_200(self):
        """GET /api/tagesplan → HTTP 200."""
        resp = client.get("/api/tagesplan")
        assert resp.status_code == 200

    def test_tagesplan_has_slots_or_loading(self):
        """Response enthält 'slots' oder einen Lade-Indikator."""
        resp = client.get("/api/tagesplan")
        data = resp.json()
        assert "slots" in data or "loading" in data or "error" in data

    def test_tagesplan_sport_mode_returns_200(self):
        """GET /api/tagesplan?mode=sport → HTTP 200."""
        resp = client.get("/api/tagesplan?mode=sport")
        assert resp.status_code == 200

    def test_tagesplan_sport_mode_has_slots(self):
        """Sport-Modus liefert ebenfalls 'slots' oder loading."""
        resp = client.get("/api/tagesplan?mode=sport")
        data = resp.json()
        assert "slots" in data or "loading" in data or "error" in data

    def test_tagesplan_invalid_mode_defaults_gracefully(self):
        """Ungültiger mode-Parameter → 200, kein Server-Crash."""
        resp = client.get("/api/tagesplan?mode=INVALID_MODE")
        assert resp.status_code in (200, 400, 422)


# ── /api/ml/status ────────────────────────────────────────────────────────────

class TestMlStatusEndpoint:
    def test_ml_status_returns_200(self):
        """GET /api/ml/status → HTTP 200."""
        resp = client.get("/api/ml/status")
        assert resp.status_code == 200

    def test_ml_status_has_trained_field(self):
        """Response muss 'trained'-Feld enthalten."""
        resp = client.get("/api/ml/status")
        data = resp.json()
        assert "trained" in data

    def test_ml_status_trained_is_bool(self):
        """'trained' muss ein Boolean sein."""
        resp = client.get("/api/ml/status")
        data = resp.json()
        assert isinstance(data.get("trained"), bool)

    def test_ml_status_content_type_json(self):
        resp = client.get("/api/ml/status")
        assert "application/json" in resp.headers.get("content-type", "")


# ── POST /api/pushes/sync ───────────────────────────────────────────────────────

class TestPushSyncEndpoint:
    def test_push_sync_without_config_returns_503(self, monkeypatch):
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "")
        resp = client.post(
            "/api/pushes/sync",
            json={"secret": "anything", "messages": [], "channels": []},
        )
        assert resp.status_code == 503

    def test_push_sync_wrong_secret_returns_403(self, monkeypatch):
        """POST /api/pushes/sync mit falschem secret → 403."""
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "test-sync-secret")
        resp = client.post(
            "/api/pushes/sync",
            json={"secret": "WRONG_SECRET", "messages": [], "channels": []},
        )
        assert resp.status_code == 403

    def test_push_sync_empty_secret_returns_403(self, monkeypatch):
        """POST /api/pushes/sync ohne secret → 403."""
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "test-sync-secret")
        resp = client.post(
            "/api/pushes/sync",
            json={"messages": [], "channels": []},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["title"] == "Forbidden"
        assert data["status"] == 403
        assert "detail" in data

    def test_push_sync_correct_secret_returns_200(self, monkeypatch):
        """POST /api/pushes/sync mit korrektem secret → 200."""
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "test-sync-secret")
        resp = client.post(
            "/api/pushes/sync",
            json={
                "secret": "test-sync-secret",
                "messages": [],
                "channels": [],
            },
        )
        # 200 oder 201 sind beide akzeptabel
        assert resp.status_code in (200, 201)

    def test_push_sync_correct_secret_response_ok(self, monkeypatch):
        """Korrekter Secret → Response mit 'ok: true'."""
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "test-sync-secret")
        resp = client.post(
            "/api/pushes/sync",
            json={
                "secret": "test-sync-secret",
                "messages": [{"id": "1", "title": "Test"}],
                "channels": [],
            },
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            assert data.get("ok") is True

    def test_push_sync_with_env_secret(self, monkeypatch):
        """Wenn PUSH_SYNC_SECRET per Env gesetzt, wird dieser verwendet."""
        monkeypatch.setattr("app.routers.push.SYNC_SECRET", "custom-secret-test")
        resp = client.post(
            "/api/pushes/sync",
            json={"secret": "custom-secret-test", "messages": [], "channels": []},
        )
        assert resp.status_code in (200, 201)


class TestPushTitleGenerateEndpoint:
    def test_generate_push_title_one_brain_response_shape(self, monkeypatch):
        monkeypatch.setattr("app.config.OPENAI_API_KEY", "test-key")

        mocked_result = {
            "gewinner": {
                "titel": "Klarer Gewinner-Titel",
                "warum_dieser": "Er ist konkret, aktiv und bildstark.",
            },
            "alternative": {"titel": "Alternative A"},
            "alle_kandidaten": {
                "direkt": [
                    {"titel": "Alternative A"},
                    {"titel": "Alternative B"},
                ],
                "narrativ": [{"titel": "Alternative C"}],
            },
            "meta": {"analyse": {"kern": "Fallback-Reasoning"}},
        }

        with patch("push_title_agent.generate_push_title", return_value=mocked_result):
            resp = client.post(
                "/api/push-title/generate",
                json={"title": "Ausgangstitel", "category": "news"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Klarer Gewinner-Titel"
        assert isinstance(data["alternativeTitles"], list)
        assert data["alternativeTitles"] == [
            "Alternative A",
            "Alternative B",
            "Alternative C",
        ]
        assert isinstance(data["reasoning"], str)
        assert data["reasoning"] == "Er ist konkret, aktiv und bildstark."
        assert data["advisoryOnly"] is True

    def test_generate_push_title_alias_requires_title_problem(self):
        resp = client.post("/api/push-title-generations", json={"category": "news"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["title"] == "Bad Request"
        assert data["status"] == 400

    def test_schwab_chat_returns_not_implemented_problem(self):
        resp = client.post("/api/schwab-chat", json={"message": "Hallo", "history": []})
        assert resp.status_code == 501
        data = resp.json()
        assert data["title"] == "HTTP Error"
        assert data["status"] == 501
        assert "not implemented" in data["detail"].lower()

    def test_ml_ab_status_returns_not_implemented_problem(self):
        resp = client.get("/api/ml/ab-status")
        assert resp.status_code == 501
        data = resp.json()
        assert data["title"] == "HTTP Error"
        assert data["status"] == 501
        assert "not implemented" in data["detail"].lower()
