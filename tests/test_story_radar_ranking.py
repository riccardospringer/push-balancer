import unittest

from story_radar.repositories import InMemoryStoryRadarRepository
from story_radar.service import StoryRadarService


class RankingTest(unittest.TestCase):
    def setUp(self):
        self.service = StoryRadarService(repository=InMemoryStoryRadarRepository.seeded())

    def test_hybrid_top_story_and_suppression(self):
        ranked = self.service.get_ranked("hybrid", include_suppressed=True)["items"]
        ordered = sorted(ranked, key=lambda item: item["ranking"]["final_rank"])

        self.assertEqual(ordered[0]["cluster"]["cluster_id"], "crime-berlin-knife-01")
        suppressed = {item["cluster"]["cluster_id"]: item["ranking"]["suppression_reason"] for item in ordered if item["ranking"]["suppressed"]}
        self.assertEqual(suppressed["politik-merz-energie-01"], "already_covered")
        self.assertEqual(suppressed["noise-eu-committee-01"], "standard_noise")


if __name__ == "__main__":
    unittest.main()
