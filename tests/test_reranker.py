import unittest

from mypic_bot.reranker import parse_rerank_results, reranker_query


class RerankerTests(unittest.TestCase):
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
        query = reranker_query("朋友說五分鐘到")
        self.assertIn("natural, witty reaction-meme", query)
        self.assertIn("朋友說五分鐘到", query)


if __name__ == "__main__":
    unittest.main()
