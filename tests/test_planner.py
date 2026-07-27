import unittest

from mypic_bot.planner import (
    build_interjection_prompt,
    enforce_plan_consistency,
    expanded_search_terms,
    parse_interjection_plan,
)


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
        self.assertEqual(plan.speaker_perspective, "observer")
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

    def test_parses_self_perspective(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "用「抱歉／不是故意」心虛承認忽視對方",
              "search_terms": ["抱歉", "不是故意", "被發現了"],
              "meme_role": "awkwardness",
              "speaker_perspective": "self",
              "reason": "對方在抱怨機器人本人。",
              "confidence": 0.95
            }
            """
        )
        self.assertEqual(plan.speaker_perspective, "self")
        self.assertEqual(plan.meme_role, "awkwardness")

    def test_prompt_routes_bot_accusations_to_self_perspective(self):
        prompt = build_interjection_prompt(
            "我被motis選擇性忽視了",
            "Andy: motis怎麼都沒回",
            bot_aliases=("Mortis", "motis"),
        )
        self.assertIn("名稱或別名包括：Mortis、motis", prompt)
        self.assertIn("必須選 self", prompt)
        self.assertIn("不能站在旁邊安慰對方", prompt)
        self.assertIn("不得複製對方對你的指控", prompt)

    def test_prompt_routes_bot_behavior_questions_to_direct_answers(self):
        prompt = build_interjection_prompt(
            "只傳貼圖他是不是就掛了",
            "Andy: 這機器人看不懂貼圖嗎",
            bot_aliases=("Mortis",),
        )
        self.assertIn("會不會故障", prompt)
        self.assertIn("不知道／可能吧／不會吧／我沒問題", prompt)
        self.assertIn("不要只把「掛了／完蛋了」", prompt)

    def test_self_accusation_repairs_contradictory_search_terms(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "解釋被忽視的指控",
              "search_terms": ["沒看到", "沒在理你"],
              "meme_role": "commiseration",
              "speaker_perspective": "self",
              "reason": "對方在抱怨我忽視他",
              "confidence": 0.9
            }
            """
        )
        repaired = enforce_plan_consistency(
            plan,
            "我被 motis 選擇性忽視了",
        )
        self.assertEqual(
            repaired.search_terms,
            ("抱歉", "不是故意", "被發現了", "我錯了"),
        )
        self.assertEqual(repaired.meme_role, "awkwardness")

    def test_self_status_question_keeps_direct_answer_terms(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "回答自己是否正常",
              "search_terms": ["沒問題", "還在"],
              "meme_role": "other",
              "speaker_perspective": "self",
              "reason": "對方在詢問機器人狀態",
              "confidence": 0.9
            }
            """
        )
        repaired = enforce_plan_consistency(plan, "他是不是掛了")
        self.assertEqual(repaired.search_terms, ("沒問題", "還在"))
        self.assertEqual(repaired.meme_role, "other")

    def test_unaddressed_group_you_is_not_forced_into_bot_perspective(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "用「被抓包」承認自己又買了一張顯卡",
              "search_terms": ["又買一張", "被發現了", "顯卡"],
              "meme_role": "teasing",
              "speaker_perspective": "self",
              "reason": "對方問我是不是又弄了一張卡",
              "confidence": 0.95
            }
            """
        )
        repaired = enforce_plan_consistency(
            plan,
            "你又搞一張卡？",
            "Kiwinamimimi: 960比我想的大張欸\nKiwinamimimi: 比1060大張",
            ("Mortis", "Motis"),
            False,
        )
        self.assertEqual(repaired.speaker_perspective, "observer")
        self.assertEqual(
            repaired.search_terms,
            ("又買一張", "被發現了", "顯卡"),
        )

    def test_context_can_resolve_third_person_reference_to_bot(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "回答自己是否正常",
              "search_terms": ["沒問題", "還在"],
              "meme_role": "other",
              "speaker_perspective": "self",
              "reason": "前文中的他是 Mortis",
              "confidence": 0.9
            }
            """
        )
        repaired = enforce_plan_consistency(
            plan,
            "只傳貼圖他是不是就掛了",
            "Andy: Mortis 好像看不懂貼圖",
            ("Mortis", "Motis"),
            False,
        )
        self.assertEqual(repaired.speaker_perspective, "self")

    def test_prompt_explains_platform_address_signal_and_context_weight(self):
        prompt = build_interjection_prompt(
            "你又搞一張卡？",
            "Kiwi: 960比我想的大張\nKiwi: 比1060大張",
            bot_aliases=("Mortis", "Motis"),
            latest_author="Andy",
            bot_addressed=False,
        )
        self.assertIn("本次最新訊息作者：Andy", prompt)
        self.assertIn("沒有提及你，也沒有回覆你", prompt)
        self.assertIn("其中「你」指其他群友", prompt)
        self.assertIn("最近群聊是重要判斷依據", prompt)
        self.assertIn("延續中的笑點", prompt)
        self.assertIn("speaker_perspective 必須是 observer", prompt)
        self.assertIn("不能代替對方", prompt)
        self.assertIn("又來了／不愧是你", prompt)

    def test_observer_role_adds_general_retrieval_terms(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "吐槽群友又買東西",
              "search_terms": ["又搞一張", "真有錢"],
              "meme_role": "teasing",
              "speaker_perspective": "observer",
              "reason": "第三人補刀",
              "confidence": 0.9
            }
            """
        )
        terms = expanded_search_terms(plan)
        self.assertEqual(terms[:2], ("又搞一張", "真有錢"))
        self.assertIn("又來了", terms)
        self.assertIn("真的假的", terms)
        self.assertLessEqual(len(terms), 8)

    def test_self_perspective_does_not_add_observer_role_terms(self):
        plan = parse_interjection_plan(
            """
            {
              "action": "post",
              "reaction_goal": "心虛承認被抓包",
              "search_terms": ["抱歉", "不是故意"],
              "meme_role": "teasing",
              "speaker_perspective": "self",
              "reason": "本人回答",
              "confidence": 0.9
            }
            """
        )
        self.assertEqual(
            expanded_search_terms(plan),
            ("抱歉", "不是故意"),
        )


if __name__ == "__main__":
    unittest.main()
