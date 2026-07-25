from dataclasses import dataclass
import json
import re

from .config import Settings


@dataclass(frozen=True)
class CandidateChoice:
    index: int
    reason: str
    humor_style: str
    confidence: float


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
        index=index,
        reason=str(result.get("reason", "")).strip(),
        humor_style=str(result.get("humor_style", "unknown")).strip() or "unknown",
        confidence=min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
    )


async def choose_candidate(
    settings: Settings,
    query: str,
    candidates,
) -> CandidateChoice:
    import httpx

    if not settings.llm_base_url or not settings.llm_model:
        return CandidateChoice(
            index=0,
            reason="未設定重排模型，使用檢索排序第一名。",
            humor_style="retrieval_fallback",
            confidence=0.0,
        )

    candidate_text = "\n".join(
        f"{index}: {row['text']}" for index, row in enumerate(candidates)
    )
    prompt = (
        "你是熟悉中文網路文化的 reaction meme 選圖手。你的任務不是找"
        "語意最相似的字幕，也不是提供正常、認真或溫馨的回答；而是從候選"
        "字幕中，選出一句傳出去最像網路梗圖回覆、最有戲謔效果的台詞。"
        "優先考慮吐槽、反諷、荒謬反差、誇張反應、冷面笑匠或朋友間欠揍的"
        "幽默。台詞必須能作為對使用者訊息的回應，而不是單純重述它。"
        "避免仇恨、歧視或惡意人身攻擊；若訊息涉及真實危機，選擇較溫和的"
        "幽默。請用一個短句提供可供事後稽核的選擇理由，不要輸出逐步思考"
        "或冗長分析。只輸出 JSON，包含 index、reason、humor_style、"
        "confidence。\n\n"
        f"使用者訊息：{query}\n\n候選：\n{candidate_text}"
    )
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "thinking_budget_tokens": 384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "candidate_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": len(candidates) - 1,
                        },
                        "reason": {
                            "type": "string",
                            "maxLength": 200,
                        },
                        "humor_style": {
                            "type": "string",
                            "enum": [
                                "sarcasm",
                                "absurd_contrast",
                                "exaggeration",
                                "deadpan",
                                "schadenfreude",
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
                        "index",
                        "reason",
                        "humor_style",
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
