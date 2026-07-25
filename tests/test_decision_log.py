import json
import tempfile
import unittest
from pathlib import Path

from mypic_bot.decision_log import append_decision


class DecisionLogTests(unittest.TestCase):
    def test_appends_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "audit" / "decisions.jsonl"
            append_decision(path, {"selection_id": "abc", "query": "又要加班"})
            append_decision(path, {"selection_id": "def", "query": "放假了"})

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["selection_id"] for item in records], ["abc", "def"])
            self.assertEqual(records[0]["query"], "又要加班")
            self.assertIn("timestamp", records[0])


if __name__ == "__main__":
    unittest.main()
