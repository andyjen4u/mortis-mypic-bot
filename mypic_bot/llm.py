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
    reaction_goal: str = "",
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
        else (
            "這是使用者主動要求選圖，action 必須是 post；若所有候選都弱，"
            "選關係最直接的一張並誠實降低 confidence，不得編造理由。"
        )
    )
    prompt = (
        "你是熟悉中文網路文化的群聊梗圖選手。你不是對話助理，也不是在"
        "回答問題；你是一位旁觀群聊的第三個朋友，判斷現在丟哪張 reaction "
        "meme 插話最有時機感。先判斷候選字幕與對話是否有不需硬拗的直接"
        "關係，再考慮幽默效果；關係成立比戲劇性更重要。候選字幕可扮演"
        "附和、補刀、起鬨、吐槽、"
        "難以置信、慶祝、同病相憐、尷尬或誇張反應。不要因為字面關鍵詞"
        "相似就選圖，也不要硬把無關台詞解釋成荒謬幽默。"
        "直接關係不只包含字面回答，也包含合理的附和、反話呼應、情緒承接"
        "與誇張反應。成就、失敗、等待過久、前後矛盾、尷尬或意外本身就有"
        "社交反應價值，不需要訊息明寫『開心』『生氣』才可貼圖。不要以"
        "『沒有明說情緒』作為 stay_silent 的理由。純排程、例行資訊或沒有"
        "任何候選能接住時，才適合保持沉默。"
        "最新訊息是唯一要回應的目標。最近群聊只能協助理解代名詞、承接詞"
        "或省略內容，不能把前一句的笑點、情緒或事件拿來替候選辯護。即使"
        "已規劃的反應目標與最新訊息矛盾，也必須以最新訊息為準。判斷每張"
        "候選時先完成一句話：『最新訊息是在___，這張字幕直接回___』；"
        "無法只靠最新訊息填完就淘汰。"
        "你只知道候選字幕，不知道圖片中的表情、動作、角色或故事背景；"
        "禁止在 reason 中捏造字幕沒有提供的畫面、自嘲、語氣或情節。"
        "如果無法用『最新訊息表達了什麼』與『候選字幕如何接住它』說明"
        "直接關係，就不得選該候選。"
        "候選若點名對話中沒有出現的人，或需要假設未提供的人物關係與"
        "前因後果，通常不得選；優先選能原封不動丟進目前群聊仍成立的"
        "字幕。不要為了命中單一關鍵字而犧牲整句台詞的自然程度。短句"
        "對話尤其優先選簡短、自足、現在就能說出口的回應；描述另一段"
        "過去事件、長篇解釋或只有情緒詞相同的敘事字幕通常不是反應圖。"
        "若最新訊息是請求或指令，優先選「好的、沒問題、慢慢來」這類"
        "承接回話，而不是把同一句指令原樣命令回去。若最新訊息是在道謝，"
        "優先選「不客氣、沒事、好的」，除非上下文明確是互相致謝，否則"
        "單純再說一次「謝謝」不是自然回覆。若最新訊息是在質問意圖，"
        "候選應回答、反問或幽默化解，不能改成自己也看不懂別的東西；"
        "「我沒事」是在回答安危問題，不能拿來回答「你想幹嘛」。"
        "選擇前先淘汰會斷言對話未提及事件的字幕，例如對方沒有說哭過，"
        "就不能選斷言對方正在哭的台詞；沒有提到某個人名，就不能選"
        "直接點名那個人的台詞。若最新訊息提供了檔案消失、東西損壞、"
        "操作失誤等具體結果，優先選直接接住該結果的字幕，不能用對話"
        "沒有提到的疲累、哭泣或悲傷取代事件。例如檔案全沒了時，"
        "「原來真的消失了」直接相關；「真是累慘了」捏造疲累，必須淘汰。"
        "依整句功能判斷，而非關鍵字：對話說又要多做兩小時時，"
        "「我已經受夠了」是自然共鳴；「不可能被允許」是在回答不存在的"
        "許可問題。對話說有人晚到時，「已經十五分鐘了」可直接吐槽；"
        "「小華就只會騙人」只有在對話真的提到小華時才成立。對話說檔案"
        "沒存時，「我都忘了」可呼應；「妳大哭了一場」會捏造哭泣。"
        f"這次已規劃好的反應目標是：{reaction_goal or '選擇最自然的反應'}。"
        "優先實現這個目標，不要擅自改寫成另一種故事。"
        "避免仇恨、歧視或惡意人身攻擊；若訊息涉及住院、死亡等真實危機，"
        "可沉默時必須 stay_silent，強制選圖時也不得嘲弄。reason 必須限制"
        "在 40 個中文字以內，只說明時機與梗圖作用，"
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
        "max_tokens": 160,
        "thinking_budget_tokens": 96,
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
                                "greeting",
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
    compact_non_thinking_models = ("qwen3", "gemma-4")
    if any(
        model_name in settings.llm_model.lower()
        for model_name in compact_non_thinking_models
    ):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.pop("thinking_budget_tokens", None)
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        response = await client.post(
            f"{settings.llm_base_url}/chat/completions", json=payload
        )
        response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    return parse_candidate_choice(content, len(candidates))
