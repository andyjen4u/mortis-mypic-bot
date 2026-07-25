import os
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

    def test_auto_reply_channel_ids_and_boolean_values(self):
        environment = {
            "AUTO_REPLY_ENABLED": "yes",
            "AUTO_REPLY_DMS": "off",
            "AUTO_REPLY_GUILD_MENTIONS_ONLY": "false",
            "AUTO_REPLY_CHANNEL_IDS": "123, 456",
            "AUTO_REPLY_COOLDOWN_SECONDS": "2.5",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.auto_reply_enabled)
        self.assertFalse(settings.auto_reply_dms)
        self.assertFalse(settings.auto_reply_guild_mentions_only)
        self.assertEqual(settings.auto_reply_channel_ids, frozenset({123, 456}))
        self.assertEqual(settings.auto_reply_cooldown_seconds, 2.5)


if __name__ == "__main__":
    unittest.main()
