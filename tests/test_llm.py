import unittest

from mypic_bot.llm import parse_candidate_choice


class CandidateChoiceTests(unittest.TestCase):
    def test_parses_structured_choice(self):
        choice = parse_candidate_choice(
            """
            {
              "action": "post",
              "index": 2,
              "reason": "用誇張反應吐槽加班。",
              "meme_role": "exaggeration",
              "confidence": 0.82
            }
            """,
            candidate_count=4,
        )
        self.assertEqual(choice.index, 2)
        self.assertEqual(choice.action, "post")
        self.assertEqual(choice.humor_style, "exaggeration")
        self.assertEqual(choice.confidence, 0.82)
        self.assertIn("加班", choice.reason)

    def test_out_of_range_index_falls_back_to_first_candidate(self):
        choice = parse_candidate_choice(
            '{"index":99,"reason":"x","humor_style":"other","confidence":2}',
            candidate_count=3,
        )
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.confidence, 1.0)

    def test_parses_stay_silent_choice(self):
        choice = parse_candidate_choice(
            """
            {
              "action": "stay_silent",
              "index": 0,
              "reason": "普通問候沒有適合插話的時機。",
              "meme_role": "other",
              "confidence": 0.91
            }
            """,
            candidate_count=3,
        )
        self.assertEqual(choice.action, "stay_silent")
        self.assertIn("普通問候", choice.reason)

    def test_invalid_truncated_json_raises(self):
        with self.assertRaises(RuntimeError):
            parse_candidate_choice(
                '{"action":"post","index":1,"confidence":0',
                candidate_count=3,
            )


if __name__ == "__main__":
    unittest.main()
