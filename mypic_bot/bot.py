from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import discord
from discord import app_commands
import httpx

from .config import Settings
from .database import (
    connect,
    get_reply_policy,
    search,
    search_by_terms,
    set_reply_policy,
)
from .decision_log import append_decision
from .embeddings import SemanticIndex, meme_retrieval_queries, request_embeddings
from .images import download_one
from .llm import CandidateChoice, choose_candidate
from .planner import InterjectionPlan, plan_interjection
from .policy import is_low_signal_message, policy_summary, should_post_choice
from .reranker import (
    candidate_text_key,
    rerank_candidate_perspectives,
    select_fused_candidate,
)


class MyPicClient(discord.Client):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.settings = settings
        self.tree = app_commands.CommandTree(self)
        self.connection = connect(settings.db_path)
        self.semantic_index = SemanticIndex.from_database(self.connection)
        self.selection_lock = asyncio.Lock()
        self.reply_cooldowns: dict[tuple[int, int], float] = {}
        self.message_history: dict[tuple[int, int], deque[dict[str, Any]]] = (
            defaultdict(lambda: deque(maxlen=self.settings.context_message_limit))
        )

    async def setup_hook(self) -> None:
        synced = await self.tree.sync()
        logging.info("Synced %s global application command(s)", len(synced))
        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.clear_commands(guild=guild)
            removed = await self.tree.sync(guild=guild)
            logging.info(
                "Removed guild-only command copies for guild_id=%s; remaining=%s",
                self.settings.discord_guild_id,
                len(removed),
            )
        logging.info(
            "Reply policy default mode=%s activity=%s dms=%s context=%s",
            self.settings.auto_reply_mode,
            self.settings.auto_reply_activity,
            self.settings.auto_reply_dms,
            self.settings.context_message_limit,
        )
        logging.info(
            "Legacy reply flags enabled=%s mentions_only=%s channels=%s",
            self.settings.auto_reply_enabled,
            self.settings.auto_reply_guild_mentions_only,
            sorted(self.settings.auto_reply_channel_ids),
        )
        logging.info(
            "Decision audit enabled=%s path=%s",
            self.settings.decision_log_enabled,
            self.settings.decision_log_path,
        )
        logging.info(
            "Interjection planner endpoint=%s model=%s",
            self.settings.llm_base_url or "disabled",
            self.settings.llm_model or "default",
        )
        logging.info(
            "Final LLM candidate judge enabled=%s",
            self.settings.final_judge_enabled,
        )
        logging.info(
            "Cross-encoder reranker endpoint=%s model=%s pool=%s top_n=%s",
            self.settings.reranker_base_url or "disabled",
            self.settings.reranker_model or "default",
            self.settings.retrieval_pool_size,
            self.settings.reranker_top_n,
        )

    async def close(self) -> None:
        self.connection.close()
        await super().close()

    async def write_decision(self, record: dict[str, Any]) -> None:
        if not self.settings.decision_log_enabled:
            return
        try:
            await asyncio.to_thread(
                append_decision,
                self.settings.decision_log_path,
                record,
            )
        except Exception:
            logging.exception("Failed to append decision audit log")

    async def select_image(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        conversation: str = "",
        mode: str = "always",
        activity: str = "medium",
        mentioned: bool = False,
    ) -> Path | None:
        selection_id = str(uuid4())
        allow_silence = mode == "auto" and not mentioned
        if allow_silence and self.selection_lock.locked():
            await self.write_decision(
                {
                    "selection_id": selection_id,
                    "query": query,
                    "context": context or {},
                    "conversation": conversation,
                    "policy": {
                        "mode": mode,
                        "activity": activity,
                        "mentioned": mentioned,
                        "gate_reason": "busy_skip",
                    },
                    "timings_ms": {"queue": 0.0, "total": 0.0},
                    "result": {"status": "stayed_silent"},
                }
            )
            logging.info(
                "Selection %s skipped because another selection is running",
                selection_id,
            )
            return None
        queued_at = time.perf_counter()
        async with self.selection_lock:
            selection_started = time.perf_counter()
            timings = {"queue": selection_started - queued_at}

            def timing_snapshot() -> dict[str, float]:
                snapshot = {
                    **timings,
                    "total": time.perf_counter() - selection_started,
                }
                return {
                    name: round(seconds * 1000, 1)
                    for name, seconds in snapshot.items()
                }

            planner_started = time.perf_counter()
            try:
                plan = await plan_interjection(
                    self.settings,
                    query,
                    conversation,
                    allow_silence=allow_silence,
                )
                planner_error = None
            except Exception as error:
                planner_error = f"{type(error).__name__}: {error}"
                logging.exception("Interjection planning failed")
                plan = InterjectionPlan(
                    action="stay_silent" if allow_silence else "post",
                    reaction_goal="對最新訊息做最直接、自然且不冒犯的群聊反應",
                    search_terms=(),
                    meme_role="other",
                    reason="規劃模型失敗。",
                    confidence=0.0,
                )
            timings["planner"] = time.perf_counter() - planner_started
            plan_post, plan_gate_reason = should_post_choice(
                mode=mode,
                activity=activity,
                mentioned=mentioned,
                action=plan.action,
                confidence=plan.confidence,
            )
            if not plan_post:
                await self.write_decision(
                    {
                        "selection_id": selection_id,
                        "query": query,
                        "context": context or {},
                        "conversation": conversation,
                        "policy": {
                            "mode": mode,
                            "activity": activity,
                            "mentioned": mentioned,
                            "gate_reason": plan_gate_reason,
                        },
                        "planner": {
                            "model": self.settings.llm_model,
                            "action": plan.action,
                            "reaction_goal": plan.reaction_goal,
                            "search_terms": list(plan.search_terms),
                            "meme_role": plan.meme_role,
                            "reason": plan.reason,
                            "confidence": plan.confidence,
                            "error": planner_error,
                        },
                        "timings_ms": timing_snapshot(),
                        "candidates": [],
                        "result": {"status": "stayed_silent"},
                    }
                )
                return None

            retrieval_started = time.perf_counter()
            timings["embedding"] = 0.0
            retrieval_queries = meme_retrieval_queries(
                query,
                conversation,
                plan.reaction_goal,
            )
            candidates = []
            candidate_scores: list[float | None] = []
            candidate_query_indexes: list[int | None] = []
            candidate_query_ranks: list[int | None] = []
            candidate_pool_indexes: list[int] = []
            candidate_contextual_scores: list[float | None] = []
            candidate_reaction_scores: list[float | None] = []
            candidate_contextual_ranks: list[int | None] = []
            candidate_reaction_ranks: list[int | None] = []
            retrieval_mode = "text"
            retrieval_error = None
            if self.semantic_index is not None and self.settings.embedding_base_url:
                embedding_started = time.perf_counter()
                try:
                    query_vectors = await request_embeddings(
                        self.settings,
                        retrieval_queries,
                    )
                    scored_candidates = (
                        self.semantic_index.diverse_search_entries_detailed(
                            self.connection,
                            query_vectors,
                            per_query_limit=(
                                self.settings.retrieval_pool_size
                                + len(retrieval_queries)
                                - 1
                            )
                            // len(retrieval_queries),
                            total_limit=self.settings.retrieval_pool_size,
                        )
                    )
                    candidates = [
                        entry for entry, _score, _query_index, _query_rank in scored_candidates
                    ]
                    candidate_scores = [
                        float(score)
                        for _entry, score, _query_index, _query_rank in scored_candidates
                    ]
                    candidate_query_indexes = [
                        query_index
                        for _entry, _score, query_index, _query_rank in scored_candidates
                    ]
                    candidate_query_ranks = [
                        query_rank
                        for _entry, _score, _query_index, query_rank in scored_candidates
                    ]
                    retrieval_mode = "semantic"
                except Exception as error:
                    retrieval_error = f"{type(error).__name__}: {error}"
                    logging.exception("Semantic retrieval failed; using text search")
                finally:
                    timings["embedding"] = (
                        time.perf_counter() - embedding_started
                    )
            lexical_candidates = search_by_terms(
                self.connection,
                plan.search_terms,
                limit=min(12, self.settings.retrieval_pool_size),
            )
            if lexical_candidates:
                semantic_records_by_id = {
                    candidate["segment_id"]: (
                        candidate,
                        score,
                        query_index,
                        query_rank,
                    )
                    for candidate, score, query_index, query_rank in zip(
                        candidates,
                        candidate_scores,
                        candidate_query_indexes,
                        candidate_query_ranks,
                    )
                }
                merged_records = []
                seen_segment_ids = set()
                for lexical_rank, candidate in enumerate(lexical_candidates):
                    segment_id = candidate["segment_id"]
                    if segment_id in seen_segment_ids:
                        continue
                    seen_segment_ids.add(segment_id)
                    semantic_record = semantic_records_by_id.get(segment_id)
                    if semantic_record is None:
                        merged_records.append(
                            (candidate, None, -1, lexical_rank)
                        )
                    else:
                        merged_records.append(semantic_record)
                for semantic_record in semantic_records_by_id.values():
                    segment_id = semantic_record[0]["segment_id"]
                    if segment_id in seen_segment_ids:
                        continue
                    seen_segment_ids.add(segment_id)
                    merged_records.append(semantic_record)
                    if len(merged_records) >= self.settings.retrieval_pool_size:
                        break
                candidates = [record[0] for record in merged_records]
                candidate_scores = [record[1] for record in merged_records]
                candidate_query_indexes = [record[2] for record in merged_records]
                candidate_query_ranks = [record[3] for record in merged_records]
                retrieval_mode = f"{retrieval_mode}+lexical"
            if not candidates:
                candidates = search(
                    self.connection,
                    query,
                    limit=self.settings.retrieval_pool_size,
                )
                candidate_scores = [None] * len(candidates)
                candidate_query_indexes = [None] * len(candidates)
                candidate_query_ranks = [None] * len(candidates)
                retrieval_mode = "text"
            unique_candidate_records = []
            seen_candidate_texts = set()
            for candidate, score, query_index, query_rank in zip(
                candidates,
                candidate_scores,
                candidate_query_indexes,
                candidate_query_ranks,
            ):
                text_key = candidate_text_key(candidate["text"])
                if text_key and text_key in seen_candidate_texts:
                    continue
                if text_key:
                    seen_candidate_texts.add(text_key)
                unique_candidate_records.append(
                    (candidate, score, query_index, query_rank)
                )
            candidates = [record[0] for record in unique_candidate_records]
            candidate_scores = [record[1] for record in unique_candidate_records]
            candidate_query_indexes = [
                record[2] for record in unique_candidate_records
            ]
            candidate_query_ranks = [
                record[3] for record in unique_candidate_records
            ]
            timings["retrieval"] = time.perf_counter() - retrieval_started
            if not candidates:
                await self.write_decision(
                    {
                        "selection_id": selection_id,
                        "query": query,
                        "context": context or {},
                        "conversation": conversation,
                        "planner": {
                            "model": self.settings.llm_model,
                            "action": plan.action,
                            "reaction_goal": plan.reaction_goal,
                            "search_terms": list(plan.search_terms),
                            "meme_role": plan.meme_role,
                            "reason": plan.reason,
                            "confidence": plan.confidence,
                            "error": planner_error,
                        },
                        "retrieval": {
                            "mode": retrieval_mode,
                            "queries": retrieval_queries,
                            "error": retrieval_error,
                        },
                        "timings_ms": timing_snapshot(),
                        "candidates": [],
                        "result": {"status": "no_candidates"},
                    }
                )
                return None
            retrieval_records = [
                {
                    "pool_index": index,
                    "segment_id": candidate["segment_id"],
                    "text": candidate["text"],
                    "semantic_score": score,
                    "retrieval_query_index": query_index,
                    "retrieval_rank": query_rank,
                }
                for index, (candidate, score, query_index, query_rank) in enumerate(
                    zip(
                        candidates,
                        candidate_scores,
                        candidate_query_indexes,
                        candidate_query_ranks,
                    )
                )
            ]
            cross_encoder_error = None
            rerank_results = []
            reranker_started = time.perf_counter()
            if self.settings.reranker_base_url:
                try:
                    rerank_results = await rerank_candidate_perspectives(
                        self.settings,
                        query,
                        candidates,
                        conversation,
                        plan.reaction_goal,
                        plan.meme_role,
                        plan.search_terms,
                    )
                except Exception as error:
                    cross_encoder_error = f"{type(error).__name__}: {error}"
                    logging.exception(
                        "Cross-encoder reranking failed; using retrieval order"
                    )
            timings["reranker"] = time.perf_counter() - reranker_started
            if rerank_results:
                selected_pool_indexes = [
                    result.original_index for result in rerank_results
                ]
                candidate_contextual_scores = [
                    result.contextual_score for result in rerank_results
                ]
                candidate_reaction_scores = [
                    result.reaction_score for result in rerank_results
                ]
                candidate_contextual_ranks = [
                    result.contextual_rank for result in rerank_results
                ]
                candidate_reaction_ranks = [
                    result.reaction_rank for result in rerank_results
                ]
            else:
                selected_pool_indexes = list(
                    range(min(self.settings.reranker_top_n * 2, len(candidates)))
                )
                candidate_contextual_scores = [None] * len(selected_pool_indexes)
                candidate_reaction_scores = [None] * len(selected_pool_indexes)
                candidate_contextual_ranks = [None] * len(selected_pool_indexes)
                candidate_reaction_ranks = [None] * len(selected_pool_indexes)
            candidate_pool_indexes = selected_pool_indexes
            candidates = [candidates[index] for index in selected_pool_indexes]
            candidate_scores = [
                candidate_scores[index] for index in selected_pool_indexes
            ]
            candidate_query_indexes = [
                candidate_query_indexes[index] for index in selected_pool_indexes
            ]
            candidate_query_ranks = [
                candidate_query_ranks[index] for index in selected_pool_indexes
            ]
            candidate_records = []
            for index, (
                candidate,
                score,
                query_index,
                query_rank,
                pool_index,
                contextual_score,
                reaction_score,
                contextual_rank,
                reaction_rank,
            ) in enumerate(
                zip(
                    candidates,
                    candidate_scores,
                    candidate_query_indexes,
                    candidate_query_ranks,
                    candidate_pool_indexes,
                    candidate_contextual_scores,
                    candidate_reaction_scores,
                    candidate_contextual_ranks,
                    candidate_reaction_ranks,
                )
            ):
                candidate_records.append(
                    {
                        "index": index,
                        "segment_id": candidate["segment_id"],
                        "text": candidate["text"],
                        "semantic_score": score,
                        "retrieval_query_index": query_index,
                        "retrieval_rank": query_rank,
                        "retrieval_pool_index": pool_index,
                        "contextual_reranker_score": contextual_score,
                        "reaction_reranker_score": reaction_score,
                        "contextual_reranker_rank": contextual_rank,
                        "reaction_reranker_rank": reaction_rank,
                        "season": candidate["season"],
                        "episode": candidate["episode"],
                        "frame_prefer": candidate["frame_prefer"],
                        "local_path": candidate["local_path"],
                    }
                )
            routed = select_fused_candidate(
                rerank_results,
                plan.meme_role,
                query,
                [candidate["text"] for candidate in candidates],
                conversation,
                plan.reaction_goal,
                plan.search_terms[0] if plan.search_terms else "",
            )
            if routed is not None:
                fallback_index, fallback_perspective = routed
                fallback_choice = CandidateChoice(
                    action="post",
                    index=fallback_index,
                    reason=(
                        f"{plan.reaction_goal}；採用"
                        f"{'對話關聯' if fallback_perspective == 'contextual' else '梗圖反應'}"
                        "排序第一名。"
                    ),
                    meme_role=plan.meme_role,
                    confidence=plan.confidence,
                )
            else:
                fallback_choice = CandidateChoice(
                    action="post" if not allow_silence else "stay_silent",
                    index=0,
                    reason="角色重排失敗，使用檢索排序第一名。",
                    meme_role="retrieval_fallback",
                    confidence=0.0,
                )
                fallback_perspective = "retrieval_fallback"
            final_judge_started = time.perf_counter()
            use_final_judge = (
                self.settings.final_judge_enabled or not rerank_results
            )
            if use_final_judge:
                try:
                    choice = await choose_candidate(
                        self.settings,
                        query,
                        candidates,
                        conversation,
                        allow_silence=allow_silence,
                        reaction_goal=plan.reaction_goal,
                    )
                    selected_perspective = "gemma_final_judge"
                    selector_error = None
                except Exception as error:
                    selector_error = f"{type(error).__name__}: {error}"
                    logging.exception(
                        "Final candidate judge failed; using deterministic fallback"
                    )
                    choice = fallback_choice
                    selected_perspective = fallback_perspective
            else:
                choice = fallback_choice
                selected_perspective = fallback_perspective
                selector_error = None
            timings["final_judge"] = (
                time.perf_counter() - final_judge_started
            )

            should_post, gate_reason = should_post_choice(
                mode=mode,
                activity=activity,
                mentioned=mentioned,
                action=choice.action,
                confidence=choice.confidence,
            )
            if not should_post:
                await self.write_decision(
                    {
                        "selection_id": selection_id,
                        "query": query,
                        "context": context or {},
                        "conversation": conversation,
                        "policy": {
                            "mode": mode,
                            "activity": activity,
                            "mentioned": mentioned,
                            "gate_reason": gate_reason,
                        },
                        "planner": {
                            "model": self.settings.llm_model,
                            "action": plan.action,
                            "reaction_goal": plan.reaction_goal,
                            "search_terms": list(plan.search_terms),
                            "meme_role": plan.meme_role,
                            "reason": plan.reason,
                            "confidence": plan.confidence,
                            "error": planner_error,
                        },
                        "retrieval": {
                            "mode": retrieval_mode,
                            "queries": retrieval_queries,
                            "error": retrieval_error,
                            "pool_size": len(retrieval_records),
                            "candidates": retrieval_records,
                        },
                        "candidates": candidate_records,
                        "cross_encoder": {
                            "model": self.settings.reranker_model,
                            "strategy": "planner_role_single_perspective",
                            "selected_count": len(candidates),
                            "error": cross_encoder_error,
                        },
                        "llm_reranker": {
                            "model": self.settings.llm_model,
                            "perspective": selected_perspective,
                            "action": choice.action,
                            "selected_index": choice.index,
                            "reason": choice.reason,
                            "meme_role": choice.meme_role,
                            "confidence": choice.confidence,
                            "error": selector_error,
                        },
                        "timings_ms": timing_snapshot(),
                        "result": {"status": "stayed_silent"},
                    }
                )
                logging.info(
                    "Selection %s stayed silent reason=%s confidence=%.3f timings_ms=%s",
                    selection_id,
                    gate_reason,
                    choice.confidence,
                    timing_snapshot(),
                )
                return None

            image_started = time.perf_counter()
            selected = candidates[choice.index]
            local_path = selected["local_path"]
            if local_path and Path(local_path).is_file():
                path = Path(local_path)
                image_source = "local_cache"
            else:
                semaphore = asyncio.Semaphore(1)
                headers = {"User-Agent": "MortisMyPicBot/0.1 (private Discord bot)"}
                async with httpx.AsyncClient(
                    timeout=self.settings.download_timeout_seconds,
                    headers=headers,
                ) as http_client:
                    path = await download_one(
                        self.settings,
                        self.connection,
                        http_client,
                        selected,
                        semaphore,
                    )
                image_source = "download"
            timings["image"] = time.perf_counter() - image_started

            await self.write_decision(
                {
                    "selection_id": selection_id,
                    "query": query,
                    "context": context or {},
                    "conversation": conversation,
                    "policy": {
                        "mode": mode,
                        "activity": activity,
                        "mentioned": mentioned,
                        "gate_reason": gate_reason,
                    },
                    "planner": {
                        "model": self.settings.llm_model,
                        "action": plan.action,
                        "reaction_goal": plan.reaction_goal,
                        "search_terms": list(plan.search_terms),
                        "meme_role": plan.meme_role,
                        "reason": plan.reason,
                        "confidence": plan.confidence,
                        "error": planner_error,
                    },
                    "retrieval": {
                        "mode": retrieval_mode,
                        "queries": retrieval_queries,
                        "error": retrieval_error,
                        "pool_size": len(retrieval_records),
                        "candidates": retrieval_records,
                    },
                    "candidates": candidate_records,
                    "cross_encoder": {
                        "model": self.settings.reranker_model,
                        "strategy": "planner_role_single_perspective",
                        "selected_count": len(candidates),
                        "error": cross_encoder_error,
                    },
                    "llm_reranker": {
                        "model": self.settings.llm_model,
                        "perspective": selected_perspective,
                        "action": choice.action,
                        "selected_index": choice.index,
                        "reason": choice.reason,
                        "meme_role": choice.meme_role,
                        "confidence": choice.confidence,
                        "error": selector_error,
                    },
                    "timings_ms": timing_snapshot(),
                    "result": {
                        "status": "selected",
                        "segment_id": selected["segment_id"],
                        "path": str(path),
                        "image_source": image_source,
                    },
                }
            )
            logging.info(
                "Selection %s chose candidate=%s segment_id=%s reason=%s timings_ms=%s",
                selection_id,
                choice.index,
                selected["segment_id"],
                choice.reason,
                timing_snapshot(),
            )
            return path

    def policy_scope(self, message: discord.Message) -> tuple[str, int]:
        if message.guild is not None:
            return "channel", message.channel.id
        return "user", message.author.id

    def reply_policy(self, message: discord.Message) -> tuple[str, str]:
        scope_type, scope_id = self.policy_scope(message)
        stored = get_reply_policy(self.connection, scope_type, scope_id)
        if stored is not None:
            return stored["mode"], stored["activity"]
        mode = self.settings.auto_reply_mode
        if message.guild is None and not self.settings.auto_reply_dms:
            mode = "off"
        return mode, self.settings.auto_reply_activity

    def is_mentioned(self, message: discord.Message) -> bool:
        return self.user is not None and self.user in message.mentions

    def remember_message(self, message: discord.Message) -> list[dict[str, Any]]:
        key = (message.guild.id if message.guild else 0, message.channel.id)
        history = list(self.message_history[key])
        author_name = getattr(message.author, "display_name", str(message.author))
        self.message_history[key].append(
            {
                "message_id": message.id,
                "author_id": message.author.id,
                "author": author_name,
                "content": message.content.strip()[:500],
            }
        )
        return history

    @staticmethod
    def format_conversation(history: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{item['author']}: {item['content']}" for item in history
        )

    def take_reply_cooldown(self, message: discord.Message) -> bool:
        key = (
            message.guild.id if message.guild else 0,
            message.channel.id,
        )
        now = time.monotonic()
        previous = self.reply_cooldowns.get(key, 0.0)
        if now - previous < self.settings.auto_reply_cooldown_seconds:
            return False
        self.reply_cooldowns[key] = now
        return True

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.webhook_id is not None:
            return
        if not message.content or not message.content.strip():
            return
        history = self.remember_message(message)
        mentioned = self.is_mentioned(message)
        mode, activity = self.reply_policy(message)
        query = message.content.strip()
        if self.user is not None:
            query = query.replace(f"<@{self.user.id}>", "")
            query = query.replace(f"<@!{self.user.id}>", "")
            query = query.strip()
        if not query:
            if mentioned:
                query = "有人突然叫我出來時的反應"
            else:
                return
        if not mentioned and mode == "off":
            return
        if (
            not mentioned
            and mode == "auto"
            and is_low_signal_message(query)
        ):
            await self.write_decision(
                {
                    "selection_id": str(uuid4()),
                    "query": query,
                    "context": {
                        "trigger": "automatic_message",
                        "message_id": message.id,
                        "guild_id": message.guild.id if message.guild else None,
                        "channel_id": message.channel.id,
                        "author_id": message.author.id,
                    },
                    "conversation": self.format_conversation(history),
                    "policy": {
                        "mode": mode,
                        "activity": activity,
                        "mentioned": False,
                        "gate_reason": "low_signal_message",
                    },
                    "result": {"status": "stayed_silent"},
                }
            )
            return
        if not mentioned and mode == "auto" and not self.take_reply_cooldown(message):
            return
        logging.info(
            "Received message id=%s guild_id=%s channel_id=%s mode=%s mentioned=%s",
            message.id,
            message.guild.id if message.guild else None,
            message.channel.id,
            mode,
            mentioned,
        )
        try:
            async with message.channel.typing():
                path = await self.select_image(
                    query,
                    conversation=self.format_conversation(history),
                    mode=mode,
                    activity=activity,
                    mentioned=mentioned,
                    context={
                        "trigger": (
                            "mention" if mentioned else "automatic_message"
                        ),
                        "message_id": message.id,
                        "guild_id": message.guild.id if message.guild else None,
                        "channel_id": message.channel.id,
                        "author_id": message.author.id,
                    },
                )
            if path is None:
                return
            await message.reply(
                file=discord.File(path, filename=path.name),
                mention_author=False,
            )
        except Exception:
            logging.exception(
                "Automatic reply failed for message id=%s",
                message.id,
            )


def build_client(settings: Settings) -> MyPicClient:
    client = MyPicClient(settings)

    @client.tree.command(
        name="mypic",
        description="用戲謔的 reaction meme 回覆一則訊息",
    )
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.allowed_contexts(
        guilds=True,
        dms=True,
        private_channels=True,
    )
    @app_commands.describe(query="想用梗圖吐槽的訊息或情境")
    async def mypic(interaction: discord.Interaction, query: str):
        logging.info(
            "Received /mypic interaction id=%s guild_id=%s",
            interaction.id,
            interaction.guild_id,
        )
        await interaction.response.defer(thinking=True)
        logging.info("Acknowledged /mypic interaction id=%s", interaction.id)
        path = await client.select_image(
            query,
            mode="always",
            context={
                "trigger": "slash_command",
                "interaction_id": interaction.id,
                "guild_id": interaction.guild_id,
                "channel_id": interaction.channel_id,
                "author_id": interaction.user.id,
            },
        )
        if path is None:
            await interaction.followup.send("目前找不到合適的圖片。", ephemeral=True)
            return
        await interaction.followup.send(
            file=discord.File(path, filename=path.name),
        )

    settings_group = app_commands.Group(
        name="mypic-settings",
        description="設定這個頻道或私訊的自動梗圖模式",
    )
    app_commands.allowed_installs(guilds=True, users=True)(settings_group)
    app_commands.allowed_contexts(
        guilds=True,
        dms=True,
        private_channels=True,
    )(settings_group)

    async def settings_scope(
        interaction: discord.Interaction,
        require_permission: bool,
    ) -> tuple[str, int, str] | None:
        if interaction.guild is not None:
            permissions = getattr(interaction.user, "guild_permissions", None)
            if require_permission and (
                permissions is None or not permissions.manage_messages
            ):
                await interaction.response.send_message(
                    "需要「管理訊息」權限才能修改這個頻道的設定。",
                    ephemeral=True,
                )
                return None
            return "channel", interaction.channel_id, "這個頻道"
        return "user", interaction.user.id, "你的私訊"

    def current_policy(scope_type: str, scope_id: int) -> tuple[str, str]:
        stored = get_reply_policy(client.connection, scope_type, scope_id)
        if stored is not None:
            return stored["mode"], stored["activity"]
        return client.settings.auto_reply_mode, client.settings.auto_reply_activity

    async def save_policy(
        interaction: discord.Interaction,
        mode: str,
        activity: str | None = None,
    ) -> None:
        scope = await settings_scope(interaction, require_permission=True)
        if scope is None:
            return
        scope_type, scope_id, scope_label = scope
        _current_mode, current_activity = current_policy(scope_type, scope_id)
        selected_activity = activity or current_activity
        set_reply_policy(
            client.connection,
            scope_type,
            scope_id,
            mode,
            selected_activity,
        )
        await interaction.response.send_message(
            policy_summary(scope_label, mode, selected_activity),
            ephemeral=True,
        )

    @settings_group.command(name="status", description="查看目前的自動回圖設定")
    async def mypic_settings_status(interaction: discord.Interaction):
        scope = await settings_scope(interaction, require_permission=False)
        if scope is None:
            return
        scope_type, scope_id, scope_label = scope
        mode, activity = current_policy(scope_type, scope_id)
        await interaction.response.send_message(
            policy_summary(scope_label, mode, activity),
            ephemeral=True,
        )

    @settings_group.command(name="always", description="每一則一般訊息都回圖")
    async def mypic_settings_always(interaction: discord.Interaction):
        await save_policy(interaction, "always")

    @settings_group.command(name="auto", description="由 Bot 評估是否適合插梗圖")
    @app_commands.choices(
        activity=[
            app_commands.Choice(name="低：很有梗才插話", value="low"),
            app_commands.Choice(name="中：一般積極度", value="medium"),
            app_commands.Choice(name="高：較常插話", value="high"),
        ],
    )
    @app_commands.describe(activity="智慧判斷的積極程度")
    async def mypic_settings_auto(
        interaction: discord.Interaction,
        activity: app_commands.Choice[str],
    ):
        await save_policy(interaction, "auto", activity.value)

    @settings_group.command(name="off", description="關閉一般訊息的自動回圖")
    async def mypic_settings_off(interaction: discord.Interaction):
        await save_policy(interaction, "off")

    client.tree.add_command(settings_group)

    return client


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    if not settings.discord_token:
        raise SystemExit("DISCORD_TOKEN is not configured.")
    build_client(settings).run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
