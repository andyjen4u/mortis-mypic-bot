import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import httpx  # noqa: F401
except ImportError:
    np = None

from mypic_bot.database import (
    connect,
    get_reply_policy,
    import_metadata,
    search,
    search_by_terms,
    set_reply_policy,
)
from mypic_bot.database import store_embeddings

if np is not None:
    from mypic_bot.embeddings import SemanticIndex


class DatabaseTests(unittest.TestCase):
    def test_reply_policy_is_persistent_per_scope(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "test.sqlite3"
            connection = connect(path)
            self.assertIsNone(get_reply_policy(connection, "channel", 123))
            set_reply_policy(connection, "channel", 123, "auto", "high")
            policy = get_reply_policy(connection, "channel", 123)
            self.assertEqual((policy["mode"], policy["activity"]), ("auto", "high"))
            set_reply_policy(connection, "channel", 123, "off", "low")
            connection.close()

            connection = connect(path)
            policy = get_reply_policy(connection, "channel", 123)
            self.assertEqual((policy["mode"], policy["activity"]), ("off", "low"))
            connection.close()

    def test_import_and_chinese_substring_search(self):
        sample = [
            {
                "text": "你們要我什麼也不帶就這麼過去?",
                "season": 2,
                "episode": 12,
                "frame_start": 439,
                "frame_prefer": 462,
                "frame_end": 485,
                "segment_id": 6369,
                "character": 32768,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / "data.json"
            metadata.write_text(
                json.dumps(sample, ensure_ascii=False), encoding="utf-8"
            )
            connection = connect(root / "test.sqlite3")
            self.assertEqual(import_metadata(connection, metadata), 1)
            results = search(connection, "什麼也不帶", 5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["segment_id"], 6369)
            connection.close()

    def test_search_by_terms_prioritizes_multiple_matches(self):
        sample = [
            {
                "text": "辛苦了，先休息一下",
                "season": 1,
                "episode": 1,
                "frame_prefer": 10,
                "segment_id": 1,
                "character": 0,
            },
            {
                "text": "今天真的很辛苦",
                "season": 1,
                "episode": 1,
                "frame_prefer": 20,
                "segment_id": 2,
                "character": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / "data.json"
            metadata.write_text(
                json.dumps(sample, ensure_ascii=False), encoding="utf-8"
            )
            connection = connect(root / "test.sqlite3")
            import_metadata(connection, metadata)
            results = search_by_terms(
                connection,
                ["辛苦", "休息", "真的"],
                limit=5,
            )
            self.assertEqual([row["segment_id"] for row in results], [1, 2])
            self.assertEqual(results[0]["term_matches"], 2)
            connection.close()

    def test_search_by_terms_filters_noisy_single_characters_and_prefers_short_text(self):
        sample = [
            {
                "text": "今天真的很謝謝妳願意臨時過來幫忙",
                "season": 1,
                "episode": 1,
                "frame_prefer": 10,
                "segment_id": 1,
                "character": 0,
            },
            {
                "text": "謝謝大家",
                "season": 1,
                "episode": 1,
                "frame_prefer": 20,
                "segment_id": 2,
                "character": 0,
            },
            {
                "text": "好",
                "season": 1,
                "episode": 1,
                "frame_prefer": 30,
                "segment_id": 3,
                "character": 0,
            },
            {
                "text": "謝謝大家",
                "season": 1,
                "episode": 2,
                "frame_prefer": 40,
                "segment_id": 4,
                "character": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / "data.json"
            metadata.write_text(
                json.dumps(sample, ensure_ascii=False), encoding="utf-8"
            )
            connection = connect(root / "test.sqlite3")
            import_metadata(connection, metadata)
            results = search_by_terms(
                connection,
                ["謝", "好", "謝謝"],
                limit=5,
            )
            self.assertEqual(
                [row["segment_id"] for row in results],
                [4, 1],
            )
            connection.close()

    @unittest.skipIf(np is None, "NumPy is not installed in the local test runtime")
    def test_semantic_index_uses_cosine_similarity(self):
        sample = [
            {
                "text": "我今天非常開心",
                "season": 1,
                "episode": 1,
                "frame_prefer": 10,
                "segment_id": 1,
                "character": 0,
            },
            {
                "text": "累死了，完全不想工作",
                "season": 1,
                "episode": 1,
                "frame_prefer": 20,
                "segment_id": 2,
                "character": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / "data.json"
            metadata.write_text(
                json.dumps(sample, ensure_ascii=False), encoding="utf-8"
            )
            connection = connect(root / "test.sqlite3")
            import_metadata(connection, metadata)
            vectors = [
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ]
            store_embeddings(
                connection,
                [
                    (item["segment_id"], 2, vector.tobytes())
                    for item, vector in zip(sample, vectors)
                ],
            )
            index = SemanticIndex.from_database(connection)
            self.assertIsNotNone(index)
            results = index.search_entries(
                connection,
                np.asarray([0.1, 0.9], dtype=np.float32),
                limit=1,
            )
            self.assertEqual(results[0][0]["segment_id"], 2)
            detailed = index.diverse_search_entries_detailed(
                connection,
                np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32),
                per_query_limit=1,
                total_limit=2,
            )
            self.assertEqual(
                [
                    (entry["segment_id"], query_index, query_rank)
                    for entry, _score, query_index, query_rank in detailed
                ],
                [(2, 0, 0), (1, 1, 0)],
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
