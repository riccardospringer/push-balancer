import unittest

from story_radar.repositories import InMemoryStoryRadarRepository
from story_radar.service import StoryRadarService


class CoverageMatcherTest(unittest.TestCase):
    def setUp(self):
        self.service = StoryRadarService(repository=InMemoryStoryRadarRepository.seeded())

    def test_expected_gap_statuses(self):
        clusters = {item["cluster"]["cluster_id"]: item for item in self.service.get_ranked("hybrid", include_suppressed=True)["items"]}
        self.assertEqual(clusters["politik-merz-energie-01"]["gap_assessment"]["coverage_status"], "already_covered")
        self.assertEqual(clusters["consumer-bahn-strike-01"]["gap_assessment"]["coverage_status"], "angle_gap")
        self.assertEqual(clusters["crime-berlin-knife-01"]["gap_assessment"]["coverage_status"], "not_covered")


if __name__ == "__main__":
    unittest.main()
