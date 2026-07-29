from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re

from .config import Settings


@dataclass(frozen=True)
class InterjectionPlan:
    action: str
    reaction_goal: str
    search_terms: tuple[str, ...]
    meme_role: str
    speaker_perspective: str
    reason: str
    confidence: float
    raw_reaction_goal: str = ""
    raw_search_terms: tuple[str, ...] = ()
    raw_meme_role: str = ""
    raw_speaker_perspective: str = ""
    adjustments: tuple[str, ...] = ()


SELF_ACCOUNTABILITY_CUES = (
    "忽視",
    "無視",
    "不理",
    "沒理",
    "沒回",
    "故意",
    "裝死",
    "忘了",
    "忘記",
    "騙",
    "害我",
    "指責",
    "抱怨",
    "道歉",
    "心虛",
    "抓包",
    "被發現",
    "做錯",
)
SELF_ACCOUNTABILITY_TERMS = ("抱歉", "不是故意", "被發現了", "我錯了")
OBSERVER_ROLE_SEARCH_TERMS = {
    "greeting": ("你好", "早安", "晚安"),
    "agreement": ("沒錯", "就是這樣", "對啊"),
    "commiseration": ("好慘", "辛苦了", "受不了"),
    "celebration": ("恭喜", "好厲害", "太好了"),
    "teasing": ("又來了", "真的假的", "太扯了", "不愧是"),
    "disbelief": ("真的假的", "不會吧", "怎麼可能"),
    "pile_on": ("又來了", "太扯了", "不愧是"),
    "awkwardness": ("好尷尬", "糟糕", "被發現了"),
    "exaggeration": ("完蛋了", "太誇張", "不得了"),
    "gentle": ("沒事", "加油", "辛苦了"),
}


def _normalize_search_term(value) -> str:
    term = str(value).strip()[:12]
    if term.endswith("一下") and len(term) > 2:
        term = term[:-2]
    return term


def parse_interjection_plan(content: str) -> InterjectionPlan:
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
                f"LLM did not return a plan JSON object; content={content!r}"
            )
        result = json.loads(match.group(0))
    action = str(result.get("action", "stay_silent")).strip().lower()
    if action not in {"post", "stay_silent"}:
        action = "stay_silent"
    raw_search_terms = result.get("search_terms", [])
    if not isinstance(raw_search_terms, list):
        raw_search_terms = []
    search_terms = tuple(
        dict.fromkeys(
            term
            for term in (
                _normalize_search_term(value)
                for value in raw_search_terms[:4]
            )
            if term
        )
    )
    reaction_goal = str(result.get("reaction_goal", "")).strip()[:60]
    meme_role = str(result.get("meme_role", "other")).strip() or "other"
    speaker_perspective = (
        str(result.get("speaker_perspective", "observer")).strip().lower()
        if str(result.get("speaker_perspective", "observer")).strip().lower()
        in {"observer", "self"}
        else "observer"
    )
    return InterjectionPlan(
        action=action,
        reaction_goal=reaction_goal,
        search_terms=search_terms,
        meme_role=meme_role,
        speaker_perspective=speaker_perspective,
        reason=str(result.get("reason", "")).strip(),
        confidence=min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
        raw_reaction_goal=reaction_goal,
        raw_search_terms=search_terms,
        raw_meme_role=meme_role,
        raw_speaker_perspective=speaker_perspective,
    )


def enforce_plan_consistency(
    plan: InterjectionPlan,
    query: str,
    conversation: str = "",
    bot_aliases: tuple[str, ...] = (),
    bot_addressed: bool | None = None,
) -> InterjectionPlan:
    adjustments = list(plan.adjustments)
    if (
        plan.speaker_perspective == "self"
        and bot_addressed is False
        and not _text_refers_to_bot(query, conversation, bot_aliases)
    ):
        plan = replace(
            plan,
            speaker_perspective="observer",
        )
        adjustments.append("forced_observer_from_platform_context")
    if plan.speaker_perspective != "self":
        return replace(plan, adjustments=tuple(adjustments))
    if not any(cue in query for cue in SELF_ACCOUNTABILITY_CUES):
        return replace(plan, adjustments=tuple(adjustments))
    adjustments.append("self_accountability_terms")
    return replace(
        plan,
        search_terms=SELF_ACCOUNTABILITY_TERMS,
        meme_role=(
            plan.meme_role
            if plan.meme_role in {"awkwardness", "teasing"}
            else "awkwardness"
        ),
        adjustments=tuple(adjustments),
    )


def expanded_search_terms(plan: InterjectionPlan) -> tuple[str, ...]:
    if plan.speaker_perspective != "observer":
        return plan.search_terms
    return tuple(
        dict.fromkeys(
            (
                *plan.search_terms,
                *OBSERVER_ROLE_SEARCH_TERMS.get(plan.meme_role, ()),
            )
        )
    )[:8]


def _normalize_identity_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _text_refers_to_bot(
    query: str,
    conversation: str,
    bot_aliases: tuple[str, ...],
) -> bool:
    normalized_query = _normalize_identity_text(query)
    normalized_conversation = _normalize_identity_text(conversation)
    aliases = tuple(
        normalized
        for normalized in (
            _normalize_identity_text(alias) for alias in bot_aliases
        )
        if len(normalized) >= 3
    )
    if any(alias in normalized_query for alias in aliases):
        return True
    bot_nouns = ("機器人", "bot", "機器狗")
    if any(noun in query.lower() for noun in bot_nouns):
        return True
    has_identity_in_context = any(
        alias in normalized_conversation for alias in aliases
    ) or any(noun in conversation.lower() for noun in bot_nouns)
    return has_identity_in_context and any(
        pronoun in query for pronoun in ("他", "它", "你")
    )


def build_interjection_prompt(
    query: str,
    conversation: str = "",
    allow_silence: bool = True,
    bot_aliases: tuple[str, ...] = (),
    latest_author: str = "",
    bot_addressed: bool | None = None,
) -> str:
    action_rule = (
        "先嚴格判斷 action。下列情況必須 stay_silent："
        "（1）只有時間、行程、地點、餐點、天氣等例行資訊，且對話沒有"
        "明說情緒、衝突或意外；不得因為是早會、工作或可表示贊同，就"
        "自行腦補疲累、壓力或插話價值。"
        "（2）住院、急診、死亡、受傷或家人安危等真實個人危機，此時"
        "不要用梗圖插話。"
        "（3）群友正在抱怨回圖太頻繁、占版面、判斷很差，或建議改成"
        "指令觸發、降低頻率與關閉功能；此時再丟圖只會加重問題。"
        "（4）只是兩位群友間的普通問答、簡短事實確認、一般推薦，或只有"
        "網址而沒有可反應的文字內容；第三位朋友不必每次都加入。"
        "只有訊息明確包含成就或興奮、可戲謔的失敗或技術事故、遲到與"
        "前後矛盾、尷尬、驚喜、荒謬反差或強烈情緒時才選 post。"
        if allow_silence
        else "使用者要求一定回圖，action 必須是 post。"
    )
    aliases = "、".join(bot_aliases) or "這個機器人"
    forced_observer = (
        bot_addressed is False
        and not _text_refers_to_bot(query, conversation, bot_aliases)
    )
    address_signal = {
        True: "平台確認這則訊息在私訊、提及你，或直接回覆你的訊息。",
        False: "平台確認這是群組的一般訊息，沒有提及你，也沒有回覆你。",
        None: "平台無法確認這則訊息是否在對你說。",
    }[bot_addressed]
    perspective_rules = (
        "硬限制：平台與前文已確定最新訊息不是對你說，其中「你」指其他"
        "群友；speaker_perspective 必須是 observer。只能旁觀附和、補刀"
        "或起鬨，不能代替對方用「抱歉／沒關係／不是故意」回答；可用"
        "「又來了／不愧是你／真有錢」等第三方反應。"
        if forced_observer
        else (
            "先判定 speaker_perspective：若最新訊息直接使用你的名稱、"
            "別名、Discord 提及，或其中「你／他／機器人」依最近群聊明顯"
            "是指你，必須選 self；只有訊息在談其他人或事件時才選 observer。"
            "若平台確認群組訊息沒有提及或回覆你，單獨出現的第二人稱「你」"
            "通常是在接前一位群友，不能因此判成 self；除非最新訊息明說"
            "你的名稱、機器人身分，或前文清楚建立該代名詞就是你。"
            "speaker_perspective=self 時，圖片字幕就是你本人說的話：對方"
            "抱怨你忽視他、沒回覆、做錯事或被抓包時，應從被指控者角度用"
            "「抱歉／不是故意／被發現了／我錯了」等承認、心虛或幽默化解，"
            "不能站在旁邊安慰對方，更不能用「沒關係」替對方決定不介意。"
            "對方詢問你會不會故障、能不能理解某種訊息或是否正常時，應從"
            "本人角度直接承接，用「不知道／可能吧／不會吧／我沒問題」等"
            "回答、自嘲或含糊化解；不要只把「掛了／完蛋了」當成已發生的"
            "災難再複述一次。只要對方在指責或抱怨你忽視、不理、沒回、"
            "忘記、騙人或做錯事，reaction_goal 必須是承認、道歉或心虛"
            "化解，至少兩個 search_terms 必須是「抱歉、對不起、不是故意、"
            "我錯了、被發現了」這類本人會說的回話；禁止用「沒在理你、"
            "不想理你」等會坐實惡意的辯解。"
        )
    )
    return (
        f"你是群聊梗圖機器人本人，名稱或別名包括：{aliases}。"
        f"本次最新訊息作者：{latest_author or '未知'}。{address_signal}"
        f"{perspective_rules}"
        "你通常像群聊中的第三位朋友插話，但不是永遠的旁觀者。現在只規劃"
        "插話意圖，不選圖片、不回答"
        "問題。根據最近群聊與最新訊息，決定朋友此刻最自然的一個社交反應"
        "目標，例如附和、慶祝、吐槽等待、同病相憐、驚訝或簡單打招呼。"
        "「最新訊息」是本次要接的話，但最近群聊是重要判斷依據：要用前"
        "幾則辨認正在談的主題、人物、代名詞、對話方向、情緒與延續中的"
        "笑點。若最新訊息是省略主詞的追問、承接句或吐槽，反應必須放回"
        "整段對話才成立。前文可以補足最新訊息省略的資訊，但不能把已經"
        "轉換掉的舊話題或無關情緒硬搬成本次目標。即使最新訊息很短、是"
        "口語、錯字或注音諧音，也要結合前文理解它本身正在接什麼。輸出"
        "前必須檢查 reaction_goal 與 search_terms 是否確實回應最新訊息，"
        "並且與最近群聊建立的主題、人物關係和語氣一致。"
        "reaction_goal 必須是 30 個中文字內、可直接拿去搜尋字幕的一個"
        "目標，而且包含 1 至 3 個可能出現在字幕中的短語概念。格式為"
        "「用『短語／短語』接住哪個事實」，例如「用『厲害／恭喜』"
        "慶祝對方完成難事」；短語只代表語意概念，不必捏造完整台詞。"
        "reaction_goal 描述的是朋友要如何回話，不是把最新訊息原樣再說"
        "一次。若最新訊息是請求或指令，目標應是答應、婉拒或幽默承接，"
        "例如用「好的／沒問題」答應對方，而不是把「等一下」命令回去。"
        "不要描述群聊氣氛、後續效果或抽象目的，不得捏造對話沒有提供的"
        "事件或人物關係。系統只會檢索字幕，因此不得使用表情、姿勢、"
        "動作、畫面或角色外觀作為反應目標；目標必須能由一句字幕本身"
        "完成。"
        "search_terms 必須是 2 至 4 個可直接在中文字幕中出現的短詞，"
        "每個不超過 6 個中文字；要保留最新訊息的具體情緒或反應語意，"
        "並包含自然近義詞。第一個必須是直接複製最新訊息中連續出現的核心"
        "短語，或只修正其中的錯字後再複製；不得增加否定詞、改成預想回覆、"
        "前文情緒或更抽象的概念；但 speaker_perspective=self 是例外，"
        "第一個詞必須是你本人自然會說的回答、道歉或辯解，不得複製對方"
        "對你的指控。例如「你想幹嘛」的第一詞是「想幹嘛」，"
        "不是「沒幹嘛」；至少一個是"
        "朋友真的會說出口的完整反應短語。搜尋的是『圖片上會出現的回話』，"
        "不能只把使用者原句拆成單字。除「撐、累、哭、慘」外避免單字搜尋"
        "詞；使用「謝謝」而不是「謝」，使用「等一下／等等」而不是「等」，"
        "使用「搞不懂／這什麼」而不是只用「什麼」。例如工作延長可用"
        "「加班、撐不住、辛苦、受夠」，檔案消失可用「沒了、慘了、消失、"
        "完蛋了」，等待過久可用「遲到、等太久、騙人」；短句表示絕望時，"
        "反應詞可用「不行了、完蛋了、受不了」；道謝時可用「謝謝、不客氣、"
        "沒事、好的」；詢問對方意圖時保留「幹嘛／想做什麼」，並加入"
        "「沒幹嘛／怎麼了」等可直接回話，不要擅自改成『搞不懂』。"
        "若最新訊息包含檔案消失、刪除、遲到、加班、考試等具體事件，"
        "第二個搜尋詞必須是同一事件的另一個具體字幕說法，不能四個詞都"
        "只寫情緒。例如「全沒了」後要有「消失／沒了」，「刪掉資料庫」"
        "後要有「消失／刪除」，「過半小時」後要有「遲到」，其餘位置"
        "才放「慘了／騙人／受不了」等反應詞。"
        "禁止使用「心情、反應、事情、真的、可以、沒辦法、"
        "這樣、那樣」等抽象或低資訊詞，也禁止重複同一個詞。"
        "若 action 是 stay_silent，search_terms 必須是空陣列。"
        "meme_role 必須依下列固定定義選擇：greeting=回應招呼；"
        "agreement=直接附和觀點；commiseration=接住失敗、損失或壓力；"
        "celebration=慶祝成就或好消息；teasing=不惡意吐槽；"
        "disbelief=難以置信；pile_on=跟著補刀；awkwardness=尷尬反應；"
        "exaggeration=誇張放大；gentle=溫和支持；other=以上皆非。"
        "若 speaker_perspective=observer 且最新訊息描述發話者自己造成、"
        "後果荒謬或災難性的技術失誤，優先用 disbelief 或 exaggeration；"
        "若 speaker_perspective=self 且對方在指控你，優先用 awkwardness "
        "或 teasing 表達心虛、被抓包或道歉，不要用 commiseration。"
        f"{action_rule}"
        "reason 限 40 個中文字。只輸出 JSON：action、reaction_goal、"
        "search_terms、meme_role、speaker_perspective、reason、confidence。"
        "\n\n"
        f"最近群聊：\n{conversation or '（沒有更早的對話）'}\n"
        f"最新訊息：{query}"
    )


async def plan_interjection(
    settings: Settings,
    query: str,
    conversation: str = "",
    allow_silence: bool = True,
    bot_aliases: tuple[str, ...] = (),
    latest_author: str = "",
    bot_addressed: bool | None = None,
) -> InterjectionPlan:
    import httpx

    if not settings.llm_base_url or not settings.llm_model:
        return InterjectionPlan(
            action="stay_silent" if allow_silence else "post",
            reaction_goal="對最新訊息做最直接、自然且不冒犯的群聊反應",
            search_terms=(),
            meme_role="other",
            speaker_perspective="observer",
            reason="未設定規劃模型。",
            confidence=0.0,
        )
    prompt = build_interjection_prompt(
        query,
        conversation,
        allow_silence,
        bot_aliases or settings.bot_aliases,
        latest_author,
        bot_addressed,
    )
    roles = [
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
    ]
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "thinking_budget_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "interjection_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["post", "stay_silent"],
                        },
                        "reaction_goal": {
                            "type": "string",
                            "maxLength": 60,
                        },
                        "search_terms": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "maxLength": 12,
                            },
                            "maxItems": 4,
                        },
                        "meme_role": {
                            "type": "string",
                            "enum": roles,
                        },
                        "speaker_perspective": {
                            "type": "string",
                            "enum": ["observer", "self"],
                        },
                        "reason": {
                            "type": "string",
                            "maxLength": 80,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "action",
                        "reaction_goal",
                        "search_terms",
                        "meme_role",
                        "speaker_perspective",
                        "reason",
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
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        response = await client.post(
            f"{settings.llm_base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
    content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    return enforce_plan_consistency(
        parse_interjection_plan(content),
        query,
        conversation,
        bot_aliases or settings.bot_aliases,
        bot_addressed,
    )
