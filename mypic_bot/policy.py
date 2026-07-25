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
}


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
