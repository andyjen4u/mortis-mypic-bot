from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _selected_candidate(record: dict) -> dict:
    segment_id = record.get("result", {}).get("segment_id")
    return next(
        (
            candidate
            for candidate in record.get("candidates", [])
            if candidate.get("segment_id") == segment_id
        ),
        {},
    )


def _candidate_score(candidate: dict) -> float | None:
    for key in ("reaction_reranker_score", "contextual_reranker_score"):
        value = candidate.get(key)
        if value is not None:
            return float(value)
    return None


def summarize_decision(record: dict) -> dict:
    selected = _selected_candidate(record)
    selected_score = _candidate_score(selected)
    planner = record.get("planner", {})
    context = record.get("context", {})
    result = record.get("result", {})
    risks = []
    if (
        selected_score is not None
        and selected_score < 0.15
        and result.get("status") == "selected"
    ):
        risks.append("low_reranker_fit")
    if (
        selected_score is not None
        and selected_score < 0.15
        and float(planner.get("confidence", 0.0)) >= 0.8
    ):
        risks.append("planner_confidence_not_grounded_in_fit")
    if len(str(selected.get("text", ""))) > 36:
        risks.append("long_caption")
    if result.get("delivery") == "suppressed_shadow":
        risks.append("shadow_only")
    return {
        "selection_id": record.get("selection_id"),
        "query": record.get("query"),
        "conversation": record.get("conversation"),
        "context": {
            "trigger": context.get("trigger"),
            "configured_mode": context.get("configured_mode"),
            "evaluated_mode": context.get("evaluated_mode"),
            "bot_addressed": context.get("bot_addressed"),
        },
        "planner": {
            "raw": planner.get("raw"),
            "adjustments": planner.get("adjustments", []),
            "reaction_goal": planner.get("reaction_goal"),
            "search_terms": planner.get("search_terms"),
            "meme_role": planner.get("meme_role"),
            "speaker_perspective": planner.get("speaker_perspective"),
            "reason": planner.get("reason"),
            "confidence": planner.get("confidence"),
        },
        "retrieval": {
            "lexical_terms": record.get("retrieval", {}).get("lexical_terms"),
            "pool_size": record.get("retrieval", {}).get("pool_size"),
        },
        "baseline": record.get("cross_encoder", {}).get("baseline"),
        "semantic_judge": record.get("llm_reranker"),
        "selected": {
            "text": selected.get("text") or result.get("text"),
            "score": selected_score,
            "segment_id": result.get("segment_id"),
            "status": result.get("status"),
            "delivery": result.get("delivery"),
        },
        "risks": risks,
        "timings_ms": record.get("timings_ms"),
    }


def read_decisions(
    path: Path,
    limit: int = 20,
    shadow_only: bool = False,
) -> list[dict]:
    if not path.exists():
        return []
    records: Iterable[dict] = (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if shadow_only:
        records = (
            record
            for record in records
            if record.get("context", {}).get("shadow") is True
        )
    return list(records)[-max(1, limit) :]


def print_audit(
    path: Path,
    limit: int = 20,
    shadow_only: bool = False,
) -> None:
    for record in read_decisions(path, limit, shadow_only):
        print(json.dumps(summarize_decision(record), ensure_ascii=False))
