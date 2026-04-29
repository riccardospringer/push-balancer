import unittest

from story_radar.api import StoryRadarHTTPAPI
from story_radar.service import StoryRadarService


class _DummyHandler:
    def __init__(self, path):
        self.path = path
        self.payload = None
        self.error = None

    def _json_response(self, obj, ensure_ascii=True):
        self.payload = obj

    def _error(self, code, message):
        self.error = (code, message)


class StoryRadarAPIViewTest(unittest.TestCase):
    def test_ranked_endpoint_ignores_legacy_ui_params(self):
        api = StoryRadarHTTPAPI(StoryRadarService())
        handler = _DummyHandler(
            "/api/story-radar/ranked?model_variant=hybrid&team_first=true&only_open_gaps=true"
        )

        handled = api.handle_get(handler)

        self.assertTrue(handled)
        self.assertIsNone(handler.error)
        self.assertIsNotNone(handler.payload)
        self.assertFalse(handler.payload["applied_view"]["team_first"])
        self.assertIsNone(handler.payload["applied_view"]["open_gaps_filter"])
        self.assertIn("team_first", handler.payload["ignored_params"])
        self.assertIn("only_open_gaps", handler.payload["ignored_params"])
        self.assertGreaterEqual(handler.payload["count"], 1)


if __name__ == "__main__":
    unittest.main()
