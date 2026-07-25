import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mypic_bot.config import Settings


class SettingsTests(unittest.TestCase):
    def test_auto_reply_defaults_are_safe_for_guilds(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.auto_reply_enabled)
        self.assertTrue(settings.auto_reply_dms)
        self.assertTrue(settings.auto_reply_guild_mentions_only)
        self.assertEqual(settings.auto_reply_channel_ids, frozenset())
        self.assertEqual(settings.auto_reply_cooldown_seconds, 10.0)
        self.assertEqual(settings.auto_reply_mode, "auto")
        self.assertEqual(settings.auto_reply_activity, "medium")
        self.assertEqual(settings.context_message_limit, 5)
        self.assertTrue(settings.decision_log_enabled)
        self.assertEqual(
            settings.decision_log_path,
            settings.data_dir / "decisions.jsonl",
        )
        self.assertEqual(settings.reranker_base_url, "")
        self.assertEqual(settings.retrieval_pool_size, 48)
        self.assertEqual(settings.reranker_top_n, 5)

    def test_auto_reply_channel_ids_and_boolean_values(self):
        environment = {
            "AUTO_REPLY_ENABLED": "yes",
            "AUTO_REPLY_DMS": "off",
            "AUTO_REPLY_GUILD_MENTIONS_ONLY": "false",
            "AUTO_REPLY_CHANNEL_IDS": "123, 456",
            "AUTO_REPLY_COOLDOWN_SECONDS": "2.5",
            "AUTO_REPLY_MODE": "always",
            "AUTO_REPLY_ACTIVITY": "high",
            "CONTEXT_MESSAGE_LIMIT": "4",
            "DECISION_LOG_ENABLED": "false",
            "DECISION_LOG_PATH": "/tmp/mortis-decisions.jsonl",
            "RERANKER_BASE_URL": "http://localhost:8083/v1/",
            "RERANKER_MODEL": "qwen3-reranker",
            "RETRIEVAL_POOL_SIZE": "64",
            "RERANKER_TOP_N": "8",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.auto_reply_enabled)
        self.assertFalse(settings.auto_reply_dms)
        self.assertFalse(settings.auto_reply_guild_mentions_only)
        self.assertEqual(settings.auto_reply_channel_ids, frozenset({123, 456}))
        self.assertEqual(settings.auto_reply_cooldown_seconds, 2.5)
        self.assertEqual(settings.auto_reply_mode, "always")
        self.assertEqual(settings.auto_reply_activity, "high")
        self.assertEqual(settings.context_message_limit, 4)
        self.assertFalse(settings.decision_log_enabled)
        self.assertEqual(
            settings.decision_log_path,
            Path("/tmp/mortis-decisions.jsonl"),
        )
        self.assertEqual(settings.reranker_base_url, "http://localhost:8083/v1")
        self.assertEqual(settings.reranker_model, "qwen3-reranker")
        self.assertEqual(settings.retrieval_pool_size, 64)
        self.assertEqual(settings.reranker_top_n, 8)


if __name__ == "__main__":
    unittest.main()
