import unittest

from mypic_bot.policy import (
    activity_threshold,
    is_low_signal_message,
    is_reply_feature_feedback,
    policy_summary,
    resolve_evaluation_mode,
    should_post_choice,
)


class ReplyPolicyTests(unittest.TestCase):
    def test_greeting_is_low_signal_but_longer_message_is_not(self):
        self.assertTrue(is_low_signal_message("你好！"))
        self.assertTrue(is_low_signal_message("  安安  "))
        self.assertFalse(is_low_signal_message("你好，我今天又遲到了"))
        self.assertTrue(is_low_signal_message("有聽過"))
        self.assertTrue(is_low_signal_message("OK"))

    def test_reply_feature_feedback_stays_out_of_auto_conversation(self):
        self.assertTrue(
            is_reply_feature_feedback(
                "這樣每一句話都要一張圖其實蠻占版面的",
                "上一句: 不如調成指令發圖",
            )
        )
        self.assertTrue(
            is_reply_feature_feedback("感覺模型的判斷要多調一下")
        )
        self.assertFalse(
            is_reply_feature_feedback("這張顯卡比我想像的大")
        )

    def test_off_mode_can_run_as_suppressed_shadow_auto(self):
        mode, shadow = resolve_evaluation_mode(
            "off",
            mentioned=False,
            shadow_enabled=True,
        )
        self.assertEqual(mode, "auto")
        self.assertTrue(shadow)

    def test_off_without_shadow_does_not_evaluate(self):
        mode, shadow = resolve_evaluation_mode(
            "off",
            mentioned=False,
            shadow_enabled=False,
        )
        self.assertIsNone(mode)
        self.assertFalse(shadow)

    def test_mention_still_uses_configured_off_override(self):
        mode, shadow = resolve_evaluation_mode(
            "off",
            mentioned=True,
            shadow_enabled=True,
        )
        self.assertEqual(mode, "off")
        self.assertFalse(shadow)

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
