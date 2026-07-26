import unittest

try:
    from mypic_bot.embeddings import meme_retrieval_queries
except ImportError:
    meme_retrieval_queries = None


@unittest.skipIf(
    meme_retrieval_queries is None,
    "Embedding runtime dependencies are not installed",
)
class RetrievalQueryTests(unittest.TestCase):
    def test_without_goal_keeps_four_broad_queries(self):
        queries = meme_retrieval_queries("今天好累")
        self.assertEqual(len(queries), 4)

    def test_goal_adds_a_direct_caption_query(self):
        queries = meme_retrieval_queries(
            "主管說今天大家再撐一下",
            "小明: 我看很難準時下班",
            "接住主管要求大家再撐一下的壓迫感",
        )
        self.assertEqual(len(queries), 5)
        self.assertIn("指定群聊反應", queries[0])
        self.assertIn("接住主管要求大家再撐一下的壓迫感", queries[0])
        self.assertIn("不得只靠圖片背景或硬拗", queries[0])


if __name__ == "__main__":
    unittest.main()
