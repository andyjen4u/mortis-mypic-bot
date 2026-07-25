import unittest

from mypic_bot.policy import (
    activity_threshold,
    is_low_signal_message,
    policy_summary,
    should_post_choice,
)


class ReplyPolicyTests(unittest.TestCase):
    def test_greeting_is_low_signal_but_longer_message_is_not(self):
        self.assertTrue(is_low_signal_message("你好！"))
        self.assertTrue(is_low_signal_message("  安安  "))
        self.assertFalse(is_low_signal_message("你好，我今天又遲到了"))

    def test_mentions_always_override_off_and_silence(self):
        post, reason = should_post_choice(
            mode="off",
            activity="low",
            mentioned=True,
            action="stay_silent",
            confidence=0.0,
        )
        self.assertTrue(post)
        self.assertEqual(reason, "mention_override")

    def test_auto_mode_uses_activity_threshold(self):
        self.assertGreater(activity_threshold("low"), activity_threshold("high"))
        low_post, _ = should_post_choice(
            mode="auto",
            activity="low",
            mentioned=False,
            action="post",
            confidence=0.70,
        )
        high_post, _ = should_post_choice(
            mode="auto",
            activity="high",
            mentioned=False,
            action="post",
            confidence=0.70,
        )
        self.assertFalse(low_post)
        self.assertTrue(high_post)

    def test_always_ignores_model_silence(self):
        post, reason = should_post_choice(
            mode="always",
            activity="medium",
            mentioned=False,
            action="stay_silent",
            confidence=0.0,
        )
        self.assertTrue(post)
        self.assertEqual(reason, "mode_always")

    def test_only_auto_summary_includes_activity(self):
        self.assertIn("積極度「高」", policy_summary("這個頻道", "auto", "high"))
        self.assertNotIn("積極度", policy_summary("這個頻道", "always", "low"))
        self.assertNotIn("積極度", policy_summary("這個頻道", "off", "high"))


if __name__ == "__main__":
    unittest.main()
