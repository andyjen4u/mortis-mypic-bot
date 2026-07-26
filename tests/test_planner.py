import unittest

from mypic_bot.planner import parse_interjection_plan


class InterjectionPlanTests(unittest.TestCase):
    def test_parses_structured_plan(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "慶祝朋友考試滿分",
              "search_terms": ["恭喜", "厲害"],
              "meme_role": "celebration",
              "reason": "成就值得群友一起慶祝。",
              "confidence": 0.88
            }
            """
        )
        self.assertEqual(plan.action, "post")
        self.assertEqual(plan.meme_role, "celebration")
        self.assertIn("滿分", plan.reaction_goal)
        self.assertEqual(plan.search_terms, ("恭喜", "厲害"))
        self.assertEqual(plan.confidence, 0.88)

    def test_invalid_action_safely_stays_silent(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "maybe",
              "reaction_goal": "",
              "search_terms": "不是陣列",
              "meme_role": "other",
              "reason": "",
              "confidence": 2
            }
            """
        )
        self.assertEqual(plan.action, "stay_silent")
        self.assertEqual(plan.confidence, 1.0)

    def test_long_reaction_goal_is_clamped(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "這是一個刻意寫得非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常長的反應目標",
              "search_terms": ["這是一個超過十二個字的搜尋詞會被截短", "", "慘了"],
              "meme_role": "other",
              "reason": "",
              "confidence": 0.5
            }
            """
        )
        self.assertLessEqual(len(plan.reaction_goal), 60)
        self.assertEqual(len(plan.search_terms), 2)
        self.assertTrue(all(len(term) <= 12 for term in plan.search_terms))

    def test_search_term_removes_generic_action_suffix(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "接住加班壓力",
              "search_terms": ["撐一下", "加班"],
              "meme_role": "commiseration",
              "reason": "",
              "confidence": 0.9
            }
            """
        )
        self.assertEqual(plan.search_terms, ("撐", "加班"))

    def test_search_terms_are_deduplicated_after_normalization(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "回應對方道謝",
              "search_terms": ["謝謝", "謝謝", "好的", "好的"],
              "meme_role": "greeting",
              "reason": "",
              "confidence": 0.9
            }
            """
        )
        self.assertEqual(plan.search_terms, ("謝謝", "好的"))


if __name__ == "__main__":
    unittest.main()
