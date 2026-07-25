from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import replace
import logging
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import discord
from discord import app_commands
import httpx

from .config import Settings
from .database import (
    connect,
    get_reply_policy,
    search,
    set_reply_policy,
)
from .decision_log import append_decision
from .embeddings import SemanticIndex, meme_retrieval_queries, request_embeddings
from .images import download_one
from .llm import CandidateChoice, choose_candidate
from .policy import is_low_signal_message, should_post_choice
from .reranker import credible_rerank_results, rerank_candidates


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
        async with self.selection_lock:
            selection_id = str(uuid4())
            retrieval_queries = meme_retrieval_queries(query, conversation)
            candidates = []
            candidate_scores: list[float | None] = []
            candidate_query_indexes: list[int | None] = []
            candidate_query_ranks: list[int | None] = []
            candidate_pool_indexes: list[int] = []
            candidate_reranker_scores: list[float | None] = []
            retrieval_mode = "text"
            retrieval_error = None
            if self.semantic_index is not None and self.settings.embedding_base_url:
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
            if not candidates:
                await self.write_decision(
                    {
                        "selection_id": selection_id,
                        "query": query,
                        "context": context or {},
                        "retrieval": {
                            "mode": retrieval_mode,
                            "queries": retrieval_queries,
                            "error": retrieval_error,
                        },
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
            reranker_band_minimum = None
            reranker_returned_count = 0
            if self.settings.reranker_base_url:
                try:
                    rerank_results = await rerank_candidates(
                        self.settings,
                        query,
                        candidates,
                        conversation,
                    )
                except Exception as error:
                    cross_encoder_error = f"{type(error).__name__}: {error}"
                    logging.exception(
                        "Cross-encoder reranking failed; using retrieval order"
                    )
            if rerank_results:
                reranker_returned_count = len(rerank_results)
                rerank_results, reranker_band_minimum = credible_rerank_results(
                    rerank_results
                )
                selected_pool_indexes = [
                    result.original_index for result in rerank_results
                ]
                candidate_reranker_scores = [
                    result.score for result in rerank_results
                ]
            else:
                selected_pool_indexes = list(
                    range(min(self.settings.reranker_top_n, len(candidates)))
                )
                candidate_reranker_scores = [None] * len(selected_pool_indexes)
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
                reranker_score,
            ) in enumerate(
                zip(
                    candidates,
                    candidate_scores,
                    candidate_query_indexes,
                    candidate_query_ranks,
                    candidate_pool_indexes,
                    candidate_reranker_scores,
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
                        "reranker_score": reranker_score,
                        "season": candidate["season"],
                        "episode": candidate["episode"],
                        "frame_prefer": candidate["frame_prefer"],
                        "local_path": candidate["local_path"],
                    }
                )
            try:
                choice = await choose_candidate(
                    self.settings,
                    query,
                    candidates,
                    conversation,
                    allow_silence=(mode == "auto" and not mentioned),
                )
                llm_error = None
            except Exception as error:
                llm_error = f"{type(error).__name__}: {error}"
                logging.exception("LLM reranking failed; using first candidate")
                choice = CandidateChoice(
                    action="post" if mode != "auto" or mentioned else "stay_silent",
                    index=0,
                    reason="模型重排失敗，使用檢索排序第一名。",
                    meme_role="rerank_fallback",
                    confidence=0.0,
                )

            score_guard = None
            if (
                choice.index != 0
                and candidate_reranker_scores
                and candidate_reranker_scores[0] is not None
                and candidate_reranker_scores[choice.index] is not None
            ):
                top_score = candidate_reranker_scores[0]
                chosen_score = candidate_reranker_scores[choice.index]
                minimum_score = max(top_score * 0.70, top_score - 0.15)
                if chosen_score < minimum_score:
                    score_guard = {
                        "original_index": choice.index,
                        "original_score": chosen_score,
                        "replacement_index": 0,
                        "replacement_score": top_score,
                        "minimum_allowed_score": minimum_score,
                    }
                    choice = replace(
                        choice,
                        index=0,
                        reason=(
                            f"{choice.reason}；候選分數落差過大，改用 cross-encoder "
                            "第一名。"
                        ),
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
                            "returned_count": reranker_returned_count,
                            "selected_count": len(candidates),
                            "credible_band_minimum": reranker_band_minimum,
                            "error": cross_encoder_error,
                        },
                        "llm_reranker": {
                            "model": self.settings.llm_model,
                            "action": choice.action,
                            "selected_index": choice.index,
                            "reason": choice.reason,
                            "meme_role": choice.meme_role,
                            "confidence": choice.confidence,
                            "score_guard": score_guard,
                            "error": llm_error,
                        },
                        "result": {"status": "stayed_silent"},
                    }
                )
                logging.info(
                    "Selection %s stayed silent reason=%s confidence=%.3f",
                    selection_id,
                    gate_reason,
                    choice.confidence,
                )
                return None

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
                        "returned_count": reranker_returned_count,
                        "selected_count": len(candidates),
                        "credible_band_minimum": reranker_band_minimum,
                        "error": cross_encoder_error,
                    },
                    "llm_reranker": {
                        "model": self.settings.llm_model,
                        "action": choice.action,
                        "selected_index": choice.index,
                        "reason": choice.reason,
                        "meme_role": choice.meme_role,
                        "confidence": choice.confidence,
                        "score_guard": score_guard,
                        "error": llm_error,
                    },
                    "result": {
                        "status": "selected",
                        "segment_id": selected["segment_id"],
                        "path": str(path),
                        "image_source": image_source,
                    },
                }
            )
            logging.info(
                "Selection %s chose candidate=%s segment_id=%s reason=%s",
                selection_id,
                choice.index,
                selected["segment_id"],
                choice.reason,
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

    @client.tree.command(
        name="mypic-settings",
        description="設定這個頻道或私訊的自動梗圖模式",
    )
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.allowed_contexts(
        guilds=True,
        dms=True,
        private_channels=True,
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="每則都回圖", value="always"),
            app_commands.Choice(name="智慧判斷", value="auto"),
            app_commands.Choice(name="關閉自動回圖", value="off"),
        ],
        activity=[
            app_commands.Choice(name="低：很有梗才插話", value="low"),
            app_commands.Choice(name="中：一般積極度", value="medium"),
            app_commands.Choice(name="高：較常插話", value="high"),
        ],
    )
    @app_commands.describe(
        mode="自動回圖模式；不填則保留目前設定",
        activity="智慧判斷的積極程度；不填則保留目前設定",
    )
    async def mypic_settings(
        interaction: discord.Interaction,
        mode: Optional[app_commands.Choice[str]] = None,
        activity: Optional[app_commands.Choice[str]] = None,
    ):
        if interaction.guild is not None:
            permissions = getattr(interaction.user, "guild_permissions", None)
            if permissions is None or not permissions.manage_messages:
                await interaction.response.send_message(
                    "需要「管理訊息」權限才能修改這個頻道的設定。",
                    ephemeral=True,
                )
                return
            scope_type, scope_id = "channel", interaction.channel_id
            scope_label = "這個頻道"
        else:
            scope_type, scope_id = "user", interaction.user.id
            scope_label = "你的私訊"
        stored = get_reply_policy(client.connection, scope_type, scope_id)
        current_mode = (
            stored["mode"] if stored is not None else client.settings.auto_reply_mode
        )
        current_activity = (
            stored["activity"]
            if stored is not None
            else client.settings.auto_reply_activity
        )
        new_mode = mode.value if mode is not None else current_mode
        new_activity = activity.value if activity is not None else current_activity
        if mode is not None or activity is not None:
            set_reply_policy(
                client.connection,
                scope_type,
                scope_id,
                new_mode,
                new_activity,
            )
        mode_labels = {
            "always": "每則都回圖",
            "auto": "智慧判斷",
            "off": "關閉自動回圖",
        }
        activity_labels = {
            "low": "低",
            "medium": "中",
            "high": "高",
        }
        await interaction.response.send_message(
            f"{scope_label}：{mode_labels[new_mode]}，積極度"
            f"「{activity_labels[new_activity]}」。被提及及 `/mypic` 仍一定選圖。",
            ephemeral=True,
        )

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
