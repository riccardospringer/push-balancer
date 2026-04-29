import unittest

from story_radar.graphd_removal import assert_graphd_absent, find_graphd_fields, strip_graphd_fields
from story_radar.ingestion import ClusterIngestionService


class GraphDRemovalTest(unittest.TestCase):
    def test_strip_graphd_fields_recursively(self):
        payload = {
            "cluster_id": "abc",
            "title": "Test",
            "graphD_score": 0.88,
            "metadata": {
                "graphdScore": 12,
                "nested": [{"graphd_rank": 1, "keep": True}],
            },
        }

        cleaned = strip_graphd_fields(payload)
        self.assertEqual(find_graphd_fields(cleaned), [])
        assert_graphd_absent(cleaned)

    def test_ingestion_accepts_legacy_graphd_payload_but_drops_field(self):
        ingestion = ClusterIngestionService()
        clusters = ingestion.ingest(
            [
                {
                    "cluster_id": "legacy-1",
                    "title": "Legacy",
                    "summary": "Legacy summary",
                    "graphDScore": 99,
                    "topics": ["crime"],
                    "entities": ["Berlin"],
                }
            ]
        )
        self.assertEqual(len(clusters), 1)
        self.assertNotIn("graphDScore", clusters[0].metadata)


if __name__ == "__main__":
    unittest.main()
