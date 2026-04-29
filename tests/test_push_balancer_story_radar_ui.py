import unittest
from pathlib import Path


HTML_PATH = Path(__file__).resolve().parents[1] / "push-balancer.html"


class PushBalancerStoryRadarUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML_PATH.read_text(encoding="utf-8")

    def test_story_radar_tab_is_present(self):
        self.assertIn('data-tab="storyradar"', self.html)
        self.assertIn('id="tabStoryRadar"', self.html)

    def test_team_first_default_is_not_selected(self):
        self.assertIn("teamFirst: false", self.html)

    def test_gap_only_buttons_are_removed(self):
        self.assertNotIn("Nur Lücken", self.html)
        self.assertNotIn("Nur Gaps", self.html)

    def test_story_radar_uses_editorial_labels(self):
        self.assertIn("BILD status", self.html)
        self.assertIn("Update recommended", self.html)
        self.assertIn("Not on BILD", self.html)


if __name__ == "__main__":
    unittest.main()
