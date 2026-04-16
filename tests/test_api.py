"""test_api.py — Tests für die aktive FastAPI-App.

Alle Tests laufen ohne laufenden Server über FastAPI `TestClient`.
"""
from pathlib import Path
import sqlite3
import time
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

    def test_health_exposes_cost_controls(self):
        resp = client.get("/api/health")
        data = resp.json()
        assert "costControls" in data
        assert "paidExternalApisEnabled" in data["costControls"]
        assert "backgroundAutomationsEnabled" in data["costControls"]

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

    def test_articles_contract_marks_video_items(self, monkeypatch):
        sitemap = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>https://www.bild.de/sport/video/test-artikel</loc>
    <news:news>
      <news:title>Das sind die Szenen im Video</news:title>
      <news:publication_date>2026-04-13T08:00:00+00:00</news:publication_date>
    </news:news>
  </url>
</urlset>"""

        monkeypatch.setattr("app.routers.feed._fetch_url", lambda _url: sitemap)
        monkeypatch.setattr(
            "app.ml.predict.predict_or",
            lambda *_args, **_kwargs: {"predicted_or": 4.8},
        )

        resp = client.get("/api/articles")
        assert resp.status_code == 200
        article = resp.json()["articles"][0]
        assert article["isVideo"] is True
        assert article["type"] == "video"
        assert "video" in article["scoreReason"]

    def test_international_feed_contract_falls_back_to_live_fetch_without_background_cache(self, monkeypatch):
        monkeypatch.setattr("app.routers.feed.get_cached_feeds", lambda _name: {})
        monkeypatch.setattr(
            "app.routers.feed._fetch_feeds_live",
            lambda _feeds: {"bbc": [{"t": "Titel", "l": "https://example.com"}]},
        )

        resp = client.get("/api/international")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] >= 1
        assert payload["items"][0]["name"] == "bbc"

    def test_international_feed_contract_stays_empty_when_live_fallback_disabled(self, monkeypatch):
        import app.routers.feed as feed_router

        monkeypatch.setattr(feed_router, "LIVE_FEED_FALLBACK_ENABLED", False)
        monkeypatch.setattr("app.routers.feed.get_cached_feeds", lambda _name: {})

        resp = client.get("/api/international")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] == 0

    def test_push_refresh_job_alias_returns_sync_result(self):
        resp = client.post("/api/push-refresh-jobs", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "synced" in data

    def test_push_refresh_job_fetches_live_snapshot_on_demand(self, monkeypatch):
        import app.routers.push as push_router

        monkeypatch.setattr(
            push_router,
            "_fetch_live_push_snapshot",
            lambda: ([{"id": "abc"}], [{"name": "main"}]),
        )

        resp = client.post("/api/push-refresh-jobs", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["synced"] == 1
        assert data["source"] == "live"

    def test_push_proxy_skips_live_fetch_when_disabled(self, monkeypatch):
        import app.routers.push as push_router

        monkeypatch.setattr(push_router, "PUSH_LIVE_FETCH_ENABLED", False)
        with push_router._push_sync_lock:
            push_router._push_sync_cache["messages"] = [{"id": "cached"}]
            push_router._push_sync_cache["channels"] = [{"name": "main"}]
            push_router._push_sync_cache["ts"] = time.time()

        resp = client.get("/api/push/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert data["_synced"] is True

    def test_articles_skip_prediction_enrichment_when_disabled(self, monkeypatch):
        import app.routers.feed as feed_router

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

        monkeypatch.setattr(feed_router, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", False)
        monkeypatch.setattr("app.routers.feed._fetch_url", lambda _url: sitemap)
        monkeypatch.setattr(
            "app.ml.predict.predict_or",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("predict_or should not run")),
        )

        resp = client.get("/api/articles")

        assert resp.status_code == 200
        assert resp.json()["articles"][0]["predictedOR"] is None

    def test_articles_prediction_enrichment_uses_runtime_cache(self, monkeypatch):
        import app.routers.feed as feed_router

        sitemap = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>https://www.bild.de/politik/cache-artikel</loc>
    <news:news>
      <news:title>Cache Test Artikel</news:title>
      <news:publication_date>2026-04-10T08:00:00+00:00</news:publication_date>
    </news:news>
  </url>
</urlset>"""

        calls = {"count": 0}

        def _predict(*_args, **_kwargs):
            calls["count"] += 1
            return {"predicted_or": 5.2}

        feed_router._article_prediction_cache.clear()
        monkeypatch.setattr(feed_router, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", True)
        monkeypatch.setattr("app.routers.feed._fetch_url", lambda _url: sitemap)
        monkeypatch.setattr("app.ml.predict.predict_or", _predict)

        first = client.get("/api/articles")
        second = client.get("/api/articles")

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["articles"][0]["predictedOR"] == 0.052
        assert second.json()["articles"][0]["predictedOR"] == 0.052
        assert calls["count"] == 1

    def test_tagesplan_returns_lightweight_payload_when_on_demand_build_disabled(self, monkeypatch):
        import app.routers.tagesplan as tagesplan_router

        monkeypatch.setattr(tagesplan_router, "TAGESPLAN_ON_DEMAND_BUILD_ENABLED", False)

        resp = client.get("/api/tagesplan")

        assert resp.status_code == 200
        data = resp.json()
        assert data["loading"] is True
        assert data["economyMode"] is True

    def test_research_endpoint_warms_state_on_demand(self, monkeypatch):
        import app.routers.forschung as research_router

        original_state = dict(research_router._research_state)
        try:
            research_router._research_state.clear()
            research_router._research_state.update({"push_data": [], "last_analysis": 0})

            def _warm():
                research_router._research_state["push_data"] = [{"message_id": "p1", "or": 5.0, "ts_num": 1}]
                research_router._research_state["mature_count"] = 1
                research_router._research_state["fresh_count"] = 0
                research_router._research_state["findings"] = {}
                research_router._research_state["ticker_entries"] = []
                research_router._research_state["schwab_decisions"] = []
                research_router._research_state["live_rules"] = []
                research_router._research_state["rolling_accuracy"] = 0.6
                research_router._research_state["accuracy_history"] = []
                research_router._research_state["last_analysis"] = 1

            monkeypatch.setattr("app.research.worker.run_autonomous_analysis", _warm)

            resp = client.get("/api/research-insights")

            assert resp.status_code == 200
            data = resp.json()
            assert "learnings" in data
        finally:
            research_router._research_state.clear()
            research_router._research_state.update(original_state)

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


class TestAdobeTrafficEndpoint:
    def test_adobe_traffic_stays_disabled_without_explicit_opt_in(self, monkeypatch):
        import app.routers.misc as misc_router

        monkeypatch.setitem(misc_router._adobe_state, "enabled", False)
        monkeypatch.setitem(
            misc_router._adobe_state,
            "traffic",
            {"hourly": [{"hour": 12, "pageviews": 10, "visitors": 8}]},
        )

        resp = client.get("/api/adobe/traffic")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

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

        assets_dir = Path(__file__).resolve().parents[1] / "dist-frontend" / "assets"
        asset_name = next(path.name for path in assets_dir.glob("index-*.js"))
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

    def test_load_frontend_html_repairs_missing_hashed_assets(self, monkeypatch, tmp_path):
        from app.main import _load_frontend_html

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "index-new.js").write_text("console.log('ok');", encoding="utf-8")
        (assets_dir / "index-new.css").write_text("body{}", encoding="utf-8")
        (tmp_path / "index.html").write_text(
            """
<!doctype html>
<script type="module" src="/dist-frontend/assets/index-old.js"></script>
<link rel="stylesheet" href="/dist-frontend/assets/index-old.css">
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setattr("app.main.SERVE_DIR", str(tmp_path))

        repaired_html = _load_frontend_html()

        assert repaired_html is not None
        assert "/dist-frontend/assets/index-new.js" in repaired_html
        assert "/dist-frontend/assets/index-new.css" in repaired_html

    def test_root_serves_frontend_html_with_no_cache_headers(self, monkeypatch):
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ENABLED", True)
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_ALLOWED_CIDRS", ["145.243.0.0/16"])
        monkeypatch.setattr("app.main.INTERNAL_ACCESS_EXEMPT_PATHS", ["/api/health"])

        resp = client.get("/", headers={"CF-Connecting-IP": "145.243.163.23"})

        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
        assert "Push Balancer" in resp.text


class TestPushApiBaseCandidates:
    def test_prefers_https_and_keeps_http_fallback_for_bildcms(self, monkeypatch):
        import app.config as config

        monkeypatch.setattr(config, "PUSH_API_BASE", "http://push-frontend.bildcms.de")
        candidates = config.push_api_base_candidates()

        assert candidates == [
            "http://push-frontend.bildcms.de",
            "https://push-frontend.bildcms.de",
        ]


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

    def test_tagesplan_db_outage_returns_loading_fallback(self, monkeypatch):
        def _raise_db_outage(*_args, **_kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr("app.routers.tagesplan.build_tagesplan", _raise_db_outage)
        resp = client.get("/api/tagesplan")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("loading") is True
        assert isinstance(data.get("slots"), list)

    def test_tagesplan_suggestions_normalizes_items_for_frontends(self, monkeypatch):
        monkeypatch.setattr(
            "app.routers.tagesplan.load_tagesplan_suggestions",
            lambda date_iso=None: [
                {
                    "date_iso": "2026-04-13",
                    "slot_hour": 8,
                    "suggestion_num": 1,
                    "article_title": "Test Titel",
                    "article_link": "https://www.bild.de/test",
                    "article_category": "politik",
                    "article_score": 91.3,
                    "expected_or": 5.4,
                    "best_cat": "politik",
                    "captured_at": 12345,
                }
            ],
        )
        resp = client.get("/api/tagesplan/suggestions?date=2026-04-13")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"][0]["title"] == "Test Titel"
        assert data["items"][0]["url"] == "https://www.bild.de/test"
        assert data["items"][0]["predictedOR"] == 5.4
        assert "suggestions" in data
        assert "8" in data["suggestions"]

    def test_tagesplan_suggestions_db_outage_returns_empty_fallback(self, monkeypatch):
        def _raise_db_outage(*_args, **_kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(
            "app.routers.tagesplan.load_tagesplan_suggestions",
            _raise_db_outage,
        )
        resp = client.get("/api/tagesplan/suggestions?date=2026-04-13")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["suggestions"] == {}
        assert data["loading"] is True


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
    def test_generate_push_title_returns_stable_local_response_shape(self):
        resp = client.post(
            "/api/push-title/generate",
            json={"title": "Ausgangstitel", "category": "news"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["title"]
        assert isinstance(data["alternativeTitles"], list)
        assert isinstance(data["reasoning"], str)
        assert data["advisoryOnly"] is True
        assert data["contentType"] == "editorial"
        assert data["gewinner"]["titel"] == data["title"]
        assert isinstance(data["alle_kandidaten"], dict)
        assert data["meta"]["modus"] == "local-fallback"

    def test_generate_push_title_marks_video_context(self):
        resp = client.post(
            "/api/push-title/generate",
            json={
                "title": "Ausgangstitel",
                "category": "sport",
                "url": "https://www.bild.de/sport/video/test",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["contentType"] == "video"
        assert "video" in data["title"].lower() or any(
            "video" in title.lower() for title in data["alternativeTitles"]
        )
        assert isinstance(data["alternativeTitles"], list)
        assert data["advisoryOnly"] is True

    def test_generate_push_title_falls_back_cleanly_when_generator_crashes(self):
        with patch("app.routers.misc.build_push_title_suggestions", side_effect=Exception("boom")):
            resp = client.post(
                "/api/push-title/generate",
                json={"title": "Ausgangstitel", "category": "news"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["title"]
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


class TestPushHistoryEnhancements:
    def test_pushes_support_sorting_filtering_and_delta(self, monkeypatch):
        monkeypatch.setattr(
            "app.routers.push.push_db_load_all",
            lambda **_kwargs: [
                {
                    "message_id": "sport-1",
                    "title": "Sport stark",
                    "headline": "",
                    "cat": "sport",
                    "type": "editorial",
                    "channel": "main",
                    "channels": ["main"],
                    "ts_num": 1710000000,
                    "or": 7.4,
                    "total_recipients": 120000,
                    "opened": 8880,
                    "link": "https://www.bild.de/sport/test",
                },
                {
                    "message_id": "politik-1",
                    "title": "Politik solide",
                    "headline": "",
                    "cat": "politik",
                    "type": "video",
                    "channel": "main",
                    "channels": ["main"],
                    "ts_num": 1710003600,
                    "or": 5.1,
                    "total_recipients": 110000,
                    "opened": 5610,
                    "link": "https://www.bild.de/politik/video/test",
                },
            ],
        )
        monkeypatch.setattr(
            "app.routers.push._load_prediction_map",
            lambda push_ids: {"sport-1": 0.061, "politik-1": 0.055},
        )

        resp = client.get("/api/pushes?sort=performanceDelta&category=sport&days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["pushes"][0]["category"] == "sport"
        assert abs(data["pushes"][0]["performanceDelta"] - 0.013) < 1e-6
