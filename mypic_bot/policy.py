from __future__ import annotations

import re


MODES = ("always", "auto", "off")
ACTIVITIES = ("low", "medium", "high")
ACTIVITY_THRESHOLDS = {
    "low": 0.82,
    "medium": 0.68,
    "high": 0.52,
}

_LOW_SIGNAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "嗨",
    "哈囉",
    "哈啰",
    "你好",
    "大家好",
    "早",
    "早安",
    "午安",
    "晚安",
    "安安",
    "在嗎",
    "在么",
    "有人嗎",
    "有人在嗎",
    "有聽過",
    "沒聽過",
    "好",
    "好喔",
    "嗯",
    "喔",
    "哦",
    "ok",
    "okay",
}

_REPLY_FEATURE_FEEDBACK = (
    "指令發圖",
    "指令回圖",
    "關閉回圖",
    "不要回圖",
    "不要發圖",
    "每一句話都要一張圖",
    "每句話都要一張圖",
    "太常回圖",
    "一直回圖",
    "很占版面",
    "蠻占版面",
    "很佔版面",
    "蠻佔版面",
    "回圖功能",
    "發圖功能",
    "模型的判斷要多調",
    "判斷要多調",
)


def normalize_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    return normalized


def normalize_activity(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in ACTIVITIES:
        raise ValueError(f"activity must be one of: {', '.join(ACTIVITIES)}")
    return normalized


def activity_threshold(activity: str) -> float:
    return ACTIVITY_THRESHOLDS[normalize_activity(activity)]


def is_low_signal_message(text: str) -> bool:
    normalized = re.sub(r"[\s!?！？。,.，～~]+", "", text).lower()
    return normalized in _LOW_SIGNAL_MESSAGES


def is_reply_feature_feedback(text: str, conversation: str = "") -> bool:
    combined = f"{conversation}\n{text}".lower()
    return any(marker in combined for marker in _REPLY_FEATURE_FEEDBACK)


def resolve_evaluation_mode(
    configured_mode: str,
    mentioned: bool,
    shadow_enabled: bool,
) -> tuple[str | None, bool]:
    mode = normalize_mode(configured_mode)
    shadow = (
        mode == "off"
        and not mentioned
        and shadow_enabled
    )
    if mode == "off" and not mentioned and not shadow:
        return None, False
    return ("auto" if shadow else mode), shadow


def should_post_choice(
    *,
    mode: str,
    activity: str,
    mentioned: bool,
    action: str,
    confidence: float,
) -> tuple[bool, str]:
    mode = normalize_mode(mode)
    if mentioned:
        return True, "mention_override"
    if mode == "off":
        return False, "mode_off"
    if mode == "always":
        return True, "mode_always"
    if action != "post":
        return False, "model_stay_silent"
    threshold = activity_threshold(activity)
    if confidence < threshold:
        return False, f"confidence_below_{threshold:.2f}"
    return True, "auto_threshold_passed"


def policy_summary(scope_label: str, mode: str, activity: str) -> str:
    mode = normalize_mode(mode)
    mode_labels = {
        "always": "每則都回圖",
        "auto": "智慧判斷",
        "off": "關閉自動回圖",
    }
    if mode == "auto":
        activity_labels = {
            "low": "低",
            "medium": "中",
            "high": "高",
        }
        activity = normalize_activity(activity)
        detail = f"，積極度「{activity_labels[activity]}」"
    else:
        detail = ""
    return (
        f"{scope_label}：{mode_labels[mode]}{detail}。"
        "被提及及 `/mypic` 仍一定選圖。"
    )
