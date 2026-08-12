from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import httpx
import numpy as np

from .config import Settings
from .database import (
    embedding_counts,
    entries_missing_embeddings,
    get_entries_by_ids,
    load_embedding_rows,
    store_embeddings,
)


def _headers(api_key: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def meme_retrieval_queries(
    text: str,
    conversation: str = "",
    reaction_goal: str = "",
    speaker_perspective: str = "observer",
) -> list[str]:
    context = f"\n最近群聊：\n{conversation}" if conversation else ""
    goal = f"\n這次希望的反應：{reaction_goal}" if reaction_goal else ""
    perspective = (
        "圖片字幕必須是被談論的機器人本人所說的話，從當事人角度承接；"
        if speaker_perspective == "self"
        else "圖片字幕是群聊朋友對其他人或事件的自然反應；"
    )
    broad_queries = [
        (
            f"尋找一張能丟進群聊附和或補刀的 reaction meme；{perspective}"
            f"針對最新訊息做輕微吐槽或反諷：{text}{context}{goal}"
        ),
        (
            f"尋找一張能製造荒謬反差的網路梗圖；{perspective}"
            f"不要把它當成問答，針對最新訊息插話：{text}{context}{goal}"
        ),
        (
            f"尋找一張適合群組聊天的誇張反應圖；{perspective}"
            f"以戲劇化方式附和、驚訝或同情最新訊息：{text}{context}{goal}"
        ),
        (
            f"尋找一張能自然插入群聊的迷因；{perspective}"
            f"可以補刀、起鬨、幸災樂禍或冷面吐槽但不可惡意攻擊：{text}{context}{goal}"
        ),
    ]
    if not reaction_goal:
        return broad_queries
    direct_goal_query = (
        "尋找一句能直接完成指定群聊反應的圖片字幕；字幕本身必須自然"
        "表達這個反應，不得只靠圖片背景或硬拗。"
        f"\n指定群聊反應：{reaction_goal}"
        f"\n說話視角：{'機器人本人' if speaker_perspective == 'self' else '群聊朋友'}"
        f"\n最新訊息：{text}{context}"
    )
    return [direct_goal_query, *broad_queries]


async def request_embeddings(
    settings: Settings,
    texts: Sequence[str],
    client: Optional[httpx.AsyncClient] = None,
    input_kind: Literal["query", "passage"] = "query",
) -> np.ndarray:
    if not settings.embedding_base_url:
        raise RuntimeError("EMBEDDING_BASE_URL is not configured.")
    payload = {"input": [f"{input_kind}: {text}" for text in texts]}
    if settings.embedding_model:
        payload["model"] = settings.embedding_model

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=120,
            headers=_headers(settings.embedding_api_key),
        )
    try:
        response = await client.post(
            f"{settings.embedding_base_url}/embeddings",
            json=payload,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = np.asarray(
            [item["embedding"] for item in data],
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"unexpected embedding shape {vectors.shape} for {len(texts)} texts"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("embedding endpoint returned a zero vector")
        return vectors / norms
    finally:
        if owns_client:
            await client.aclose()


async def build_missing_embeddings(
    settings: Settings,
    connection,
    batch_size: int = 16,
) -> tuple[int, int]:
    total, before = embedding_counts(connection)
    processed = 0
    async with httpx.AsyncClient(
        timeout=120,
        headers=_headers(settings.embedding_api_key),
    ) as client:
        while True:
            rows = entries_missing_embeddings(connection, batch_size)
            if not rows:
                break
            vectors = await request_embeddings(
                settings,
                [row["text"] for row in rows],
                client,
                input_kind="passage",
            )
            store_embeddings(
                connection,
                [
                    (
                        row["segment_id"],
                        int(vector.shape[0]),
                        vector.astype(np.float32, copy=False).tobytes(),
                    )
                    for row, vector in zip(rows, vectors)
                ],
            )
            processed += len(rows)
            current = before + processed
            if current % 160 == 0 or current == total:
                print(f"Embedded {current}/{total}.", flush=True)
    return processed, total


@dataclass
class SemanticIndex:
    segment_ids: np.ndarray
    vectors: np.ndarray

    @classmethod
    def from_database(cls, connection) -> Optional["SemanticIndex"]:
        total, embedded = embedding_counts(connection)
        if not total or embedded != total:
            return None
        rows = load_embedding_rows(connection)
        if not rows:
            return None
        dimensions = {row["dimensions"] for row in rows}
        if len(dimensions) != 1:
            raise RuntimeError(f"mixed embedding dimensions: {sorted(dimensions)}")
        dimension = dimensions.pop()
        vectors = np.vstack(
            [
                np.frombuffer(row["embedding"], dtype=np.float32, count=dimension)
                for row in rows
            ]
        )
        return cls(
            segment_ids=np.asarray(
                [row["segment_id"] for row in rows],
                dtype=np.int64,
            ),
            vectors=vectors,
        )

    def search_ids(self, query_vector: np.ndarray, limit: int = 20):
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.vectors.shape[1]:
            raise RuntimeError(
                f"query dimension {vector.shape[0]} != index {self.vectors.shape[1]}"
            )
        scores = self.vectors @ vector
        count = min(max(1, limit), scores.shape[0])
        candidate_indices = np.argpartition(scores, -count)[-count:]
        ordered = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return [
            (int(self.segment_ids[index]), float(scores[index]))
            for index in ordered
        ]

    def search_entries(self, connection, query_vector: np.ndarray, limit: int = 20):
        scored_ids = self.search_ids(query_vector, limit)
        entries = get_entries_by_ids(
            connection,
            [segment_id for segment_id, _ in scored_ids],
        )
        scores = dict(scored_ids)
        return [(entry, scores[entry["segment_id"]]) for entry in entries]

    def diverse_search_entries(
        self,
        connection,
        query_vectors: np.ndarray,
        per_query_limit: int = 4,
        total_limit: int = 12,
    ):
        return [
            (entry, score)
            for entry, score, _query_index, _query_rank in (
                self.diverse_search_entries_detailed(
                    connection,
                    query_vectors,
                    per_query_limit,
                    total_limit,
                )
            )
        ]

    def diverse_search_entries_detailed(
        self,
        connection,
        query_vectors: np.ndarray,
        per_query_limit: int = 4,
        total_limit: int = 12,
    ):
        ranked_lists = [
            self.search_entries(connection, vector, per_query_limit)
            for vector in query_vectors
        ]
        results = []
        seen = set()
        for rank in range(per_query_limit):
            for query_index, ranked in enumerate(ranked_lists):
                if rank >= len(ranked):
                    continue
                entry, score = ranked[rank]
                segment_id = entry["segment_id"]
                if segment_id in seen:
                    continue
                seen.add(segment_id)
                results.append((entry, score, query_index, rank))
                if len(results) >= total_limit:
                    return results
        return results
