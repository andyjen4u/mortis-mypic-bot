import unittest

from mypic_bot.reranker import (
    FusedRerankResult,
    RerankResult,
    candidate_text_key,
    contextual_reranker_query,
    fuse_rerank_results,
    include_anchor_candidates,
    parse_rerank_results,
    reaction_reranker_query,
    reranker_query,
    select_fused_candidate,
)


class RerankerTests(unittest.TestCase):
    def test_candidate_text_key_normalizes_punctuation_and_pronouns(self):
        self.assertEqual(
            candidate_text_key("妳好，請問怎麼了嗎？"),
            candidate_text_key("你好請問怎麼了嗎"),
        )

    def test_fuses_two_perspectives_without_hard_pruning(self):
        fused = fuse_rerank_results(
            [
                RerankResult(0, 0.7663),
                RerankResult(1, 0.7233),
            ],
            [
                RerankResult(2, 0.91),
                RerankResult(1, 0.80),
            ],
        )
        self.assertEqual([item.original_index for item in fused], [0, 2, 1])
        self.assertEqual(fused[2].contextual_score, 0.7233)
        self.assertEqual(fused[2].reaction_score, 0.80)

    def test_role_routes_to_expected_perspective(self):
        fused = fuse_rerank_results(
            [RerankResult(3, 0.9), RerankResult(1, 0.5)],
            [RerankResult(7, 0.8), RerankResult(3, 0.4)],
        )
        self.assertEqual(select_fused_candidate(fused, "greeting"), (0, "contextual"))
        self.assertEqual(select_fused_candidate(fused, "celebration"), (1, "reaction"))

    def test_anchor_candidate_is_preserved_without_overriding_reranker(self):
        fused = fuse_rerank_results([], [RerankResult(0, 0.99)])
        fused = include_anchor_candidates(
            fused,
            "放假",
            ["太好了，妳終於來了", "終於放假了"],
        )
        selected = select_fused_candidate(
            fused,
            "celebration",
            "終於放假啦",
            ["太好了，妳終於來了", "終於放假了"],
            anchor_term="放假",
        )
        self.assertEqual(selected, (0, "reaction"))

    def test_greeting_uses_normalized_lexical_match_inside_fused_pool(self):
        fused = fuse_rerank_results(
            [RerankResult(3, 0.9), RerankResult(1, 0.5)],
            [RerankResult(7, 0.8), RerankResult(3, 0.4)],
        )
        selected = select_fused_candidate(
            fused,
            "greeting",
            "你好",
            ["別的字幕", "妳好，請問怎麼了嗎？", "仍然無關"],
        )
        self.assertEqual(selected, (1, "greeting_lexical"))

    def test_preserves_one_candidate_for_each_reaction_term(self):
        fused = include_anchor_candidates(
            [FusedRerankResult(0, None, 0.9, None, 0)],
            ["等", "等等", "沒問題", "好的"],
            ["別的字幕", "等等", "沒問題啦", "好的"],
            limit=3,
        )
        self.assertEqual(
            [item.original_index for item in fused],
            [0, 1, 2, 3],
        )

    def test_skips_ungrounded_pronoun_before_selecting_top_rank(self):
        fused = fuse_rerank_results(
            [],
            [RerankResult(0, 0.99), RerankResult(1, 0.98)],
        )
        selected = select_fused_candidate(
            fused,
            "commiseration",
            "主管說今天大家再撐一下",
            [
                "不過，聽說她已經算撐得很久了",
                "人生這麼漫長會撐不住的喔",
            ],
        )
        self.assertEqual(selected, (1, "reaction"))

    def test_short_message_prefers_concise_near_tie(self):
        fused = fuse_rerank_results(
            [],
            [
                RerankResult(0, 0.9950),
                RerankResult(1, 0.9850),
                RerankResult(2, 0.9782),
                RerankResult(3, 0.9775),
            ],
        )
        selected = select_fused_candidate(
            fused,
            "commiseration",
            "怕",
            [
                "悲傷呢? 恐懼呢? 遺忘呢?",
                "就說了我會怕啦,小祥",
                "妳這樣她會怕的",
                '"恐懼"呢',
            ],
        )
        self.assertEqual(selected, (3, "reaction_brevity"))

    def test_skips_leading_name_that_is_absent_from_conversation(self):
        fused = fuse_rerank_results(
            [],
            [RerankResult(0, 0.99), RerankResult(1, 0.98)],
        )
        selected = select_fused_candidate(
            fused,
            "teasing",
            "他已經遲到半小時了",
            ["初華就只會騙人", "一定是騙人的"],
        )
        self.assertEqual(selected, (1, "reaction"))

    def test_skips_future_and_first_person_lateness_for_completed_third_person_event(self):
        fused = fuse_rerank_results(
            [],
            [
                RerankResult(0, 0.99),
                RerankResult(1, 0.98),
                RerankResult(2, 0.80),
            ],
        )
        selected = select_fused_candidate(
            fused,
            "teasing",
            "他說五分鐘就到，現在已經過半小時了",
            ["要遲到了喔", "抱歉，我們遲到了", "遲到十分鐘了"],
            "",
            "吐槽對方說謊且遲到",
        )
        self.assertEqual(selected, (2, "reaction"))

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
        self.assertIn("第三位朋友", query)
        self.assertIn("原封不動", query)
        self.assertIn("點名對話外人物", query)
        self.assertIn("朋友說五分鐘到", query)
        self.assertIn("每次都說快到了", query)

    def test_contextual_query_emphasizes_grounded_relationship(self):
        query = contextual_reranker_query("你好", "Andy: 剛上線")
        self.assertIn("自然且直接相關", query)
        self.assertIn("唯一要回應", query)
        self.assertIn("Andy: 剛上線", query)
        self.assertIn("你好", query)

    def test_reaction_query_includes_specific_planner_goal(self):
        query = reaction_reranker_query(
            "主管說今天大家再撐一下",
            "小明: 我看很難準時下班",
            "接住主管要求大家再撐一下的壓迫感",
            "commiseration",
        )
        self.assertIn("接住主管要求大家再撐一下的壓迫感", query)
        self.assertIn("同病相憐", query)
        self.assertIn("前一句", query)


if __name__ == "__main__":
    unittest.main()
