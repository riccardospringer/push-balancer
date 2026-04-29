"""HTTP adapter for Story Radar using the repo's existing stdlib server."""

from __future__ import annotations

import json
import urllib.parse

from .service import StoryRadarService


class StoryRadarHTTPAPI:
    def __init__(self, service: StoryRadarService | None = None):
        self.service = service or StoryRadarService()

    @staticmethod
    def _query_bool(query: dict[str, list[str]], keys: tuple[str, ...], default: bool = False) -> bool:
        for key in keys:
            if key in query and query[key]:
                return query[key][0].lower() == "true"
        return default

    @staticmethod
    def _ranked_view_metadata(query: dict[str, list[str]], model_variant: str, include_suppressed: bool) -> dict[str, object]:
        ignored = [
            key
            for key in (
                "team_first",
                "teamFirst",
                "only_open_gaps",
                "open_gaps_only",
                "nur_offene_luecken",
                "gap_only",
            )
            if key in query
        ]
        return {
            "applied_view": {
                "model_variant": model_variant,
                "team_first": False,
                "include_suppressed": include_suppressed,
                "open_gaps_filter": None,
            },
            "ignored_params": ignored,
        }

    def handle_get(self, handler) -> bool:
        parsed = urllib.parse.urlparse(handler.path)
        path = parsed.path
        if not path.startswith("/api/story-radar/"):
            return False

        query = urllib.parse.parse_qs(parsed.query)
        if path == "/api/story-radar/clusters":
            handler._json_response({"items": self.service.list_clusters()}, ensure_ascii=False)
            return True
        if path == "/api/story-radar/ranked":
            variant = query.get("model_variant", ["hybrid"])[0]
            include_suppressed = self._query_bool(query, ("include_suppressed",), default=False)
            response = self.service.get_ranked(variant, include_suppressed)
            response.update(self._ranked_view_metadata(query, variant, include_suppressed))
            handler._json_response(response, ensure_ascii=False)
            return True
        if path == "/api/story-radar/evaluation":
            handler._json_response(self.service.get_evaluation(), ensure_ascii=False)
            return True
        if path == "/api/story-radar/debug/coverage":
            handler._json_response(self.service.get_debug_coverage(), ensure_ascii=False)
            return True
        if path == "/api/story-radar/debug/suppressed":
            handler._json_response(self.service.get_suppressed(), ensure_ascii=False)
            return True
        if path.startswith("/api/story-radar/clusters/"):
            cluster_id = path.split("/api/story-radar/clusters/", 1)[1]
            cluster = self.service.get_cluster(cluster_id)
            if cluster is None:
                handler._error(404, f"Unknown cluster_id: {cluster_id}")
                return True
            handler._json_response(cluster, ensure_ascii=False)
            return True
        if path.startswith("/api/story-radar/explanations/"):
            cluster_id = path.split("/api/story-radar/explanations/", 1)[1]
            explanation = self.service.get_explanation(cluster_id)
            if explanation is None:
                handler._error(404, f"Unknown cluster_id: {cluster_id}")
                return True
            handler._json_response(explanation, ensure_ascii=False)
            return True
        handler._error(404, f"Unknown Story Radar endpoint: {path}")
        return True

    def handle_post(self, handler) -> bool:
        parsed = urllib.parse.urlparse(handler.path)
        path = parsed.path
        if not path.startswith("/api/story-radar/"):
            return False

        length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(length) if length else b"{}"
        payload = json.loads(body or b"{}")

        if path == "/api/story-radar/train":
            result = self.service.train()
            handler._json_response(result, ensure_ascii=False)
            return True
        if path == "/api/story-radar/rescore":
            clusters = payload.get("clusters")
            response = self.service.rescore(clusters)
            handler._json_response(response, ensure_ascii=False)
            return True
        if path == "/api/story-radar/feedback":
            response = self.service.submit_feedback(payload)
            handler._json_response({"ok": True, "event": response}, ensure_ascii=False)
            return True
        handler._error(404, f"Unknown Story Radar endpoint: {path}")
        return True
