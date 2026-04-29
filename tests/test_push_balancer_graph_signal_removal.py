import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "push-balancer-server.py"
HTML = ROOT / "push-balancer.html"


class PushBalancerGraphSignalRemovalTest(unittest.TestCase):
    def test_graph_signal_is_neutralized_in_scoring(self):
        source = SERVER.read_text()
        self.assertIn('feat["phd_entity_context"] = 0.0', source)
        self.assertIn('"phd_entity_ctx_damp": 0.00', source)
        self.assertIn("graphD / Entity-Context wurde aus dem produktiven Score entfernt.", source)

    def test_combined_sort_option_removed_from_ui(self):
        html = HTML.read_text()
        self.assertNotIn('<option value="combined">', html)
        self.assertIn("currentSort = (sort === 'combined') ? 'score' : sort;", html)
        self.assertIn("if (currentSort === 'combined') currentSort = 'score';", html)


if __name__ == "__main__":
    unittest.main()
