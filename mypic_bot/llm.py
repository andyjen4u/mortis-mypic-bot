import json
import re

import httpx

from .config import Settings


async def choose_candidate(settings: Settings, query: str, candidates) -> int:
    if not settings.llm_base_url or not settings.llm_model:
        return 0

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
        "幽默。只輸出 JSON，格式為"
        '{"index": 整數}。\n\n'
        f"使用者訊息：{query}\n\n候選：\n{candidate_text}"
    )
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 768,
        "thinking_budget_tokens": 512,
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
                        }
                    },
                    "required": ["index"],
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
    match = re.search(r"\{[^{}]*\"index\"\s*:\s*-?\d+[^{}]*\}", content)
    if not match:
        raise RuntimeError(
            f"LLM did not return an index JSON object; content={content!r}"
        )
    index = int(json.loads(match.group(0))["index"])
    return index if 0 <= index < len(candidates) else 0
