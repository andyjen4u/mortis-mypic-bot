from dataclasses import dataclass
import json
import re

from .config import Settings


@dataclass(frozen=True)
class CandidateChoice:
    action: str
    index: int
    reason: str
    meme_role: str
    confidence: float

    @property
    def humor_style(self) -> str:
        return self.meme_role


def parse_candidate_choice(content: str, candidate_count: int) -> CandidateChoice:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError(
                f"LLM did not return a choice JSON object; content={content!r}"
            )
        result = json.loads(match.group(0))

    index = int(result["index"])
    if not 0 <= index < candidate_count:
        index = 0
    return CandidateChoice(
        action=(
            str(result.get("action", "post")).strip().lower()
            if str(result.get("action", "post")).strip().lower()
            in {"post", "stay_silent"}
            else "stay_silent"
        ),
        index=index,
        reason=str(result.get("reason", "")).strip(),
        meme_role=str(
            result.get("meme_role", result.get("humor_style", "other"))
        ).strip()
        or "other",
        confidence=min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
    )


async def choose_candidate(
    settings: Settings,
    query: str,
    candidates,
    conversation: str = "",
    allow_silence: bool = False,
) -> CandidateChoice:
    import httpx

    if not settings.llm_base_url or not settings.llm_model:
        return CandidateChoice(
            action="post" if not allow_silence else "stay_silent",
            index=0,
            reason="未設定重排模型，使用檢索排序第一名。",
            meme_role="retrieval_fallback",
            confidence=0.0,
        )

    candidate_text = "\n".join(
        f"{index}: {row['text']}" for index, row in enumerate(candidates)
    )
    conversation_text = conversation or "（沒有更早的對話）"
    silence_instruction = (
        "你可以選擇 stay_silent。只有當第三位朋友此刻丟出候選反應圖，"
        "真的會自然、有梗且不搶話時才選 post。普通問候、資訊不足、沒有"
        "明顯情緒或所有候選都很牽強時必須 stay_silent。"
        if allow_silence
        else "這是使用者主動要求選圖，action 必須是 post。"
    )
    prompt = (
        "你是熟悉中文網路文化的群聊梗圖選手。你不是對話助理，也不是在"
        "回答問題；你是一位旁觀群聊的第三個朋友，判斷現在丟哪張 reaction "
        "meme 插話最有時機感。候選字幕可扮演附和、補刀、起鬨、吐槽、"
        "難以置信、慶祝、同病相憐、尷尬或誇張反應。不要因為字面關鍵詞"
        "相似就選圖，也不要硬把無關台詞解釋成荒謬幽默。"
        "避免仇恨、歧視或惡意人身攻擊；若訊息涉及真實危機，選擇較溫和的"
        "幽默。reason 必須限制在 40 個中文字以內，只說明時機與梗圖作用，"
        "不要重述對話、輸出逐步思考或冗長分析。信心值代表『此刻插入這張"
        "圖是否自然且好笑』，不是"
        "語意相似度。"
        f"{silence_instruction}"
        "只輸出 JSON，包含 action、index、reason、meme_role、confidence。"
        "\n\n"
        f"最近群聊：\n{conversation_text}\n"
        f"最新訊息：{query}\n\n候選：\n{candidate_text}"
    )
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 768,
        "thinking_budget_tokens": 384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "candidate_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["post", "stay_silent"],
                        },
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": len(candidates) - 1,
                        },
                        "reason": {
                            "type": "string",
                            "maxLength": 80,
                        },
                        "meme_role": {
                            "type": "string",
                            "enum": [
                                "agreement",
                                "pile_on",
                                "teasing",
                                "disbelief",
                                "celebration",
                                "commiseration",
                                "awkwardness",
                                "exaggeration",
                                "gentle",
                                "other",
                            ],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "action",
                        "index",
                        "reason",
                        "meme_role",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    }
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        response = await client.post(
            f"{settings.llm_base_url}/chat/completions", json=payload
        )
        response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    return parse_candidate_choice(content, len(candidates))
