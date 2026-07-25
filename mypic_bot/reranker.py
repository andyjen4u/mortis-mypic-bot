from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import Settings


@dataclass(frozen=True)
class RerankResult:
    original_index: int
    score: float


def credible_rerank_results(
    results: Sequence[RerankResult],
) -> tuple[list[RerankResult], float | None]:
    if not results:
        return [], None
    top_score = results[0].score
    minimum_score = max(top_score * 0.70, top_score - 0.15)
    return [
        result for result in results if result.score >= minimum_score
    ], minimum_score


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


def reranker_query(query: str, conversation: str = "") -> str:
    context = f"\nRecent group-chat context:\n{conversation}" if conversation else ""
    return (
        "Judge whether a third friend who is observing the conversation could drop "
        "the candidate into the group chat as a timely, witty reaction image. It "
        "does not need to answer the message directly. Prefer agreement, piling on, "
        "playful teasing, disbelief, celebration, commiseration, awkwardness, "
        "absurd contrast, and exaggeration. A keyword-only, random, or forced "
        "candidate is irrelevant. "
        "For distress or crisis messages, prefer gentle and supportive replies. "
        f"Latest Discord message: {query}{context}"
    )


async def rerank_candidates(
    settings: Settings,
    query: str,
    candidates: Sequence,
    conversation: str = "",
) -> list[RerankResult]:
    import httpx

    if not settings.reranker_base_url:
        return []
    headers = {"Content-Type": "application/json"}
    if settings.reranker_api_key:
        headers["Authorization"] = f"Bearer {settings.reranker_api_key}"
    payload = {
        "model": settings.reranker_model,
        "query": reranker_query(query, conversation),
        "documents": [candidate["text"] for candidate in candidates],
        "top_n": min(settings.reranker_top_n, len(candidates)),
    }
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        response = await client.post(
            f"{settings.reranker_base_url}/rerank",
            json=payload,
        )
        response.raise_for_status()
    return parse_rerank_results(response.json(), len(candidates))
