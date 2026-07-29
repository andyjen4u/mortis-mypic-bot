import json
from pathlib import Path
import tempfile
import unittest

from mypic_bot.audit import read_decisions, summarize_decision


class DecisionAuditTests(unittest.TestCase):
    def test_summarizes_shadow_trace_and_flags_low_fit(self):
        record = {
            "selection_id": "abc",
            "query": "Teto 的聲庫太強",
            "conversation": "Andy: 不用翻唱性能就好",
            "context": {
                "trigger": "shadow_evaluation",
                "shadow": True,
                "configured_mode": "off",
                "evaluated_mode": "auto",
            },
            "planner": {
                "raw": {"search_terms": ["太強"]},
                "adjustments": [],
                "reaction_goal": "附和很厲害",
                "search_terms": ["太強", "厲害"],
                "meme_role": "agreement",
                "speaker_perspective": "observer",
                "confidence": 0.95,
            },
            "candidates": [
                {
                    "segment_id": 1,
                    "text": "我好像也有點太強硬了",
                    "contextual_reranker_score": 0.08,
                    "reaction_reranker_score": None,
                }
            ],
            "cross_encoder": {
                "baseline": {
                    "index": 0,
                    "text": "我好像也有點太強硬了",
                }
            },
            "llm_reranker": {"action": "stay_silent"},
            "result": {
                "status": "selected",
                "segment_id": 1,
                "delivery": "suppressed_shadow",
            },
        }
        summary = summarize_decision(record)
        self.assertIn("low_reranker_fit", summary["risks"])
        self.assertIn(
            "planner_confidence_not_grounded_in_fit",
            summary["risks"],
        )
        self.assertIn("shadow_only", summary["risks"])
        self.assertEqual(summary["baseline"]["index"], 0)

    def test_reads_only_shadow_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"selection_id": "live", "context": {"shadow": False}},
                        {"selection_id": "shadow", "context": {"shadow": True}},
                    )
                ),
                encoding="utf-8",
            )
            records = read_decisions(path, shadow_only=True)
        self.assertEqual(
            [record["selection_id"] for record in records],
            ["shadow"],
        )


if __name__ == "__main__":
    unittest.main()
