import unittest

from mypic_bot.reranker import (
    RerankResult,
    credible_rerank_results,
    parse_rerank_results,
    reranker_query,
)


class RerankerTests(unittest.TestCase):
    def test_credible_band_removes_large_score_gap(self):
        credible, minimum = credible_rerank_results(
            [
                RerankResult(0, 0.7663),
                RerankResult(1, 0.7233),
                RerankResult(2, 0.2805),
            ]
        )
        self.assertEqual([item.original_index for item in credible], [0, 1])
        self.assertAlmostEqual(minimum, 0.6163, places=4)

    def test_parses_sorts_and_filters_results(self):
        results = parse_rerank_results(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 9, "relevance_score": 1.0},
                    {"index": 0, "relevance_score": 0.8},
                ]
            },
            candidate_count=2,
        )
        self.assertEqual(
            [(result.original_index, result.score) for result in results],
            [(0, 0.9), (1, 0.2)],
        )

    def test_query_contains_task_and_user_message(self):
        query = reranker_query(
            "朋友說五分鐘到",
            "Andy: 他每次都說快到了",
        )
        self.assertIn("third friend", query)
        self.assertIn("does not need to answer", query)
        self.assertIn("朋友說五分鐘到", query)
        self.assertIn("每次都說快到了", query)


if __name__ == "__main__":
    unittest.main()
