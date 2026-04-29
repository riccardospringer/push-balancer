import json
import unittest

from story_radar.llm_scorer import LLM_OUTPUT_SCHEMA, StoryRadarLLMScorer
from story_radar.repositories import InMemoryStoryRadarRepository
from story_radar.service import StoryRadarService


class LLMContractTest(unittest.TestCase):
    def test_request_uses_strict_json_schema(self):
        service = StoryRadarService(repository=InMemoryStoryRadarRepository.seeded())
        card = service.list_clusters()[0]
        scorer = StoryRadarLLMScorer(client=None)

        request = scorer.build_request(
            cluster=service._last_scorecards[0].cluster,
            gap=service._last_scorecards[0].gap,
            features=service._last_scorecards[0].features,
            performance=service.repository.get_performance_snapshot(),
        )

        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(request["text"]["format"]["schema"], LLM_OUTPUT_SCHEMA)
        self.assertIn("cluster", json.loads(request["input"][1]["content"]))
        self.assertIn("cluster", card)


if __name__ == "__main__":
    unittest.main()
