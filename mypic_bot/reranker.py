from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .config import Settings


@dataclass(frozen=True)
class RerankResult:
    original_index: int
    score: float


@dataclass(frozen=True)
class FusedRerankResult:
    original_index: int
    contextual_score: float | None
    reaction_score: float | None
    contextual_rank: int | None
    reaction_rank: int | None


CONTEXTUAL_ROLES = {"greeting", "agreement", "gentle", "other"}
ROLE_GOALS = {
    "greeting": "用一句簡短招呼，自然回應對方打招呼",
    "agreement": "直接附和或用反話呼應最新訊息所暗示的處境",
    "commiseration": "對對方的失敗、損失或壓力表示同病相憐",
    "celebration": "對對方的成就或好消息表示慶祝或佩服",
    "teasing": "針對等待、矛盾或失誤做不惡意的吐槽",
    "disbelief": "對意外或誇張事件表達難以置信",
    "pile_on": "順著群聊已經出現的吐槽繼續補刀",
    "awkwardness": "對尷尬或冷場做自然反應",
    "exaggeration": "用誇張反應放大最新訊息的效果",
    "gentle": "以溫和且不嘲弄的方式表示支持",
    "other": "選擇與對話關係最直接的自然反應",
}


def select_fused_candidate(
    results: Sequence[FusedRerankResult],
    meme_role: str,
    query: str = "",
    candidate_texts: Sequence[str] = (),
    conversation: str = "",
    reaction_goal: str = "",
    anchor_term: str = "",
    speaker_perspective: str = "observer",
) -> tuple[int, str] | None:
    if not results:
        return None
    if meme_role == "greeting" and query and candidate_texts:
        normalized_query = _normalize_lexical(query)
        for index, text in enumerate(candidate_texts):
            if normalized_query and normalized_query in _normalize_lexical(text):
                return index, "greeting_lexical"
    perspective = (
        "reaction"
        if speaker_perspective == "self"
        else "contextual" if meme_role in CONTEXTUAL_ROLES else "reaction"
    )
    rank_field = f"{perspective}_rank"
    score_field = f"{perspective}_score"
    ranked_indexes = sorted(
        range(len(results)),
        key=lambda index: (
            getattr(results[index], rank_field) is None,
            getattr(results[index], rank_field)
            if getattr(results[index], rank_field) is not None
            else len(results),
        ),
    )
    context = f"{conversation}\n{query}\n{reaction_goal}".strip()
    grounded_indexes = [
        index
        for index in ranked_indexes
        if (
            index >= len(candidate_texts)
            or not _has_ungrounded_reference(candidate_texts[index], context)
        )
    ]
    if grounded_indexes:
        top_index = grounded_indexes[0]
        top_score = getattr(results[top_index], score_field)
        if (
            top_score is not None
            and len(_normalize_lexical(query)) <= 6
            and candidate_texts
        ):
            near_top_indexes = [
                index
                for index in grounded_indexes
                if (
                    getattr(results[index], score_field) is not None
                    and top_score - getattr(results[index], score_field) <= 0.02
                )
            ]
            if near_top_indexes:
                concise_index = min(
                    near_top_indexes,
                    key=lambda index: (
                        len(_normalize_lexical(candidate_texts[index])),
                        getattr(results[index], rank_field),
                    ),
                )
                if concise_index != top_index:
                    return concise_index, f"{perspective}_brevity"
        return top_index, perspective
    return 0, "fused_fallback"


def _normalize_lexical(text: str) -> str:
    return "".join(
        character
        for character in text.lower().replace("妳", "你")
        if character.isalnum()
    )


def candidate_text_key(text: str) -> str:
    return _normalize_lexical(text)


def mentions_speaker_alias(text: str, aliases: Sequence[str]) -> bool:
    normalized_text = _normalize_lexical(text)
    return any(
        normalized_alias in normalized_text
        for normalized_alias in (
            _normalize_lexical(alias) for alias in aliases
        )
        if len(normalized_alias) >= 3
    )


def _has_ungrounded_reference(text: str, context: str) -> bool:
    normalized_context = " ".join(context.split())
    if "已經" in normalized_context and any(
        future_phrase in text
        for future_phrase in ("要遲到", "快遲到", "會遲到")
    ):
        return True
    if (
        re.search(r"[他她].*(?:遲到|還沒到|沒到)", normalized_context)
        and re.search(r"(?:我|我們).*(?:遲到|還沒到|沒到)", text)
    ):
        return True
    for pronoun in ("他", "她"):
        if pronoun in text and pronoun not in normalized_context:
            return True
    for mentioned_name in re.findall(r"@([^,\s，)）]+)", text):
        if mentioned_name not in normalized_context:
            return True
    for named_person in re.findall(
        r"([\u4e00-\u9fff]{2,4})(?:同學|小姐|先生)",
        text,
    ):
        if named_person not in normalized_context:
            return True
    leading_subject = re.match(r"^([\u4e00-\u9fff]{2,4})就", text)
    if (
        leading_subject
        and leading_subject.group(1)
        not in {
            "大家",
            "我們",
            "你們",
            "妳們",
            "他們",
            "她們",
            "人家",
            "這件事",
            "那件事",
        }
        and leading_subject.group(1) not in normalized_context
    ):
        return True
    return False


def parse_rerank_results(payload: dict, candidate_count: int) -> list[RerankResult]:
    parsed = []
    seen = set()
    for item in payload.get("results", []):
        index = int(item["index"])
        if index in seen or not 0 <= index < candidate_count:
            continue
        seen.add(index)
        parsed.append(
            RerankResult(
                original_index=index,
                score=float(item["relevance_score"]),
            )
        )
    return sorted(parsed, key=lambda result: result.score, reverse=True)


def contextual_reranker_query(
    query: str,
    conversation: str = "",
    reaction_goal: str = "",
    speaker_perspective: str = "observer",
) -> str:
    perspective = (
        "候選字幕是被談論的機器人本人所說，必須像當事人的回答；"
        if speaker_perspective == "self"
        else "候選字幕是群聊朋友對其他人或事件的反應；"
    )
    return (
        f"判斷候選字幕能否原封不動成為目前群聊插話者的反應圖；{perspective}"
        "整句需"
        "自然且直接相關。最新訊息是唯一要回應的目標；前一句只能消歧。"
        "若需腦補未提事件或情緒、點名對話外人物、回答未問問題，判為無關。"
        f"\n最近對話：{conversation or '無'}"
        f"\n最新訊息：{query}"
    )


def reaction_reranker_query(
    query: str,
    conversation: str = "",
    reaction_goal: str = "",
    meme_role: str = "other",
    speaker_perspective: str = "observer",
) -> str:
    perspective = (
        "候選字幕是被談論的機器人本人所說，必須像當事人的回答；"
        if speaker_perspective == "self"
        else "候選字幕是群聊朋友對其他人或事件的反應；"
    )
    return (
        f"判斷候選字幕能否原封不動成為目前群聊插話者的 reaction meme；"
        f"{perspective}"
        "最新訊息是唯一要回應的目標；前一句只能消歧。允許附和、反話"
        "與誇張，但整句要直接自然。若需腦補未提事件或情緒、點名對話"
        "外人物、回答未問問題，判為無關。"
        f"\n反應類型：{ROLE_GOALS.get(meme_role, ROLE_GOALS['other'])}"
        f"\n本次具體目標：{reaction_goal or ROLE_GOALS.get(meme_role, ROLE_GOALS['other'])}"
        f"\n最近對話：{conversation or '無'}"
        f"\n最新訊息：{query}"
    )


def reranker_query(query: str, conversation: str = "") -> str:
    return reaction_reranker_query(query, conversation)


def fuse_rerank_results(
    contextual: Sequence[RerankResult],
    reaction: Sequence[RerankResult],
) -> list[FusedRerankResult]:
    by_index: dict[int, dict[str, float | int | None]] = {}
    for rank, result in enumerate(contextual):
        by_index.setdefault(
            result.original_index,
            {
                "contextual_score": None,
                "reaction_score": None,
                "contextual_rank": None,
                "reaction_rank": None,
            },
        ).update(
            contextual_score=result.score,
            contextual_rank=rank,
        )
    for rank, result in enumerate(reaction):
        by_index.setdefault(
            result.original_index,
            {
                "contextual_score": None,
                "reaction_score": None,
                "contextual_rank": None,
                "reaction_rank": None,
            },
        ).update(
            reaction_score=result.score,
            reaction_rank=rank,
        )

    ordered_indexes = []
    seen = set()
    for rank in range(max(len(contextual), len(reaction))):
        for results in (contextual, reaction):
            if rank >= len(results):
                continue
            index = results[rank].original_index
            if index in seen:
                continue
            seen.add(index)
            ordered_indexes.append(index)

    return [
        FusedRerankResult(original_index=index, **by_index[index])
        for index in ordered_indexes
    ]


def include_anchor_candidates(
    results: Sequence[FusedRerankResult],
    anchor_terms: str | Sequence[str],
    candidate_texts: Sequence[str],
    limit: int = 4,
    per_term_limit: int = 2,
) -> list[FusedRerankResult]:
    included = list(results)
    if isinstance(anchor_terms, str):
        anchor_terms = (anchor_terms,)
    normalized_anchors = tuple(
        dict.fromkeys(
            normalized
            for normalized in (
                _normalize_lexical(term) for term in anchor_terms
            )
            if len(normalized) >= 2
        )
    )
    if not normalized_anchors:
        return included
    existing_indexes = {result.original_index for result in included}
    added = 0
    for normalized_anchor in normalized_anchors:
        term_added = 0
        for index, text in enumerate(candidate_texts):
            if index in existing_indexes:
                continue
            if normalized_anchor not in _normalize_lexical(text):
                continue
            included.append(
                FusedRerankResult(
                    original_index=index,
                    contextual_score=None,
                    reaction_score=None,
                    contextual_rank=None,
                    reaction_rank=None,
                )
            )
            existing_indexes.add(index)
            added += 1
            term_added += 1
            if added >= limit:
                return included
            if term_added >= per_term_limit:
                break
    return included


async def _request_rerank(
    settings: Settings,
    task_query: str,
    candidates: Sequence,
) -> list[RerankResult]:
    import httpx

    headers = {"Content-Type": "application/json"}
    if settings.reranker_api_key:
        headers["Authorization"] = f"Bearer {settings.reranker_api_key}"
    payload = {
        "model": settings.reranker_model,
        "query": task_query,
        "documents": [candidate["text"] for candidate in candidates],
        "top_n": min(settings.reranker_top_n, len(candidates)),
    }
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        response = await client.post(
            f"{settings.reranker_base_url}/rerank",
            json=payload,
        )
        if response.is_error:
            raise RuntimeError(
                f"reranker returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
    return parse_rerank_results(response.json(), len(candidates))


async def rerank_candidates(
    settings: Settings,
    query: str,
    candidates: Sequence,
    conversation: str = "",
) -> list[RerankResult]:
    if not settings.reranker_base_url:
        return []
    return await _request_rerank(
        settings,
        reaction_reranker_query(query, conversation),
        candidates,
    )


async def rerank_candidate_perspectives(
    settings: Settings,
    query: str,
    candidates: Sequence,
    conversation: str = "",
    reaction_goal: str = "",
    meme_role: str = "other",
    anchor_terms: str | Sequence[str] = (),
    speaker_perspective: str = "observer",
) -> list[FusedRerankResult]:
    if not settings.reranker_base_url:
        return []
    if meme_role == "greeting":
        direct_greetings = include_anchor_candidates(
            [],
            anchor_terms,
            [candidate["text"] for candidate in candidates],
        )
        if direct_greetings:
            return direct_greetings
    if meme_role in CONTEXTUAL_ROLES and speaker_perspective != "self":
        contextual = await _request_rerank(
            settings,
            contextual_reranker_query(
                query,
                conversation,
                reaction_goal,
                speaker_perspective,
            ),
            candidates,
        )
        reaction = []
    else:
        contextual = []
        reaction = await _request_rerank(
            settings,
            reaction_reranker_query(
                query,
                conversation,
                reaction_goal,
                meme_role,
                speaker_perspective,
            ),
            candidates,
        )
    fused = fuse_rerank_results(contextual, reaction)
    return include_anchor_candidates(
        fused,
        anchor_terms,
        [candidate["text"] for candidate in candidates],
    )
