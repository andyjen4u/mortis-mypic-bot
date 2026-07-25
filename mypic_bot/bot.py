from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import discord
from discord import app_commands
import httpx

from .config import Settings
from .database import connect, search
from .decision_log import append_decision
from .embeddings import SemanticIndex, meme_retrieval_queries, request_embeddings
from .images import download_one
from .llm import CandidateChoice, choose_candidate


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
        self.reply_cooldowns: dict[tuple[int, int, int], float] = {}

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
            "Automatic replies enabled=%s dms=%s mentions_only=%s channels=%s",
            self.settings.auto_reply_enabled,
            self.settings.auto_reply_dms,
            self.settings.auto_reply_guild_mentions_only,
            sorted(self.settings.auto_reply_channel_ids),
        )
        logging.info(
            "Decision audit enabled=%s path=%s",
            self.settings.decision_log_enabled,
            self.settings.decision_log_path,
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
    ) -> Path | None:
        async with self.selection_lock:
            selection_id = str(uuid4())
            retrieval_queries = meme_retrieval_queries(query)
            candidates = []
            candidate_scores: list[float | None] = []
            candidate_query_indexes: list[int | None] = []
            candidate_query_ranks: list[int | None] = []
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
                            per_query_limit=4,
                            total_limit=12,
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
                candidates = search(self.connection, query, limit=12)
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
            try:
                choice = await choose_candidate(
                    self.settings,
                    query,
                    candidates,
                )
                llm_error = None
            except Exception as error:
                llm_error = f"{type(error).__name__}: {error}"
                logging.exception("LLM reranking failed; using first candidate")
                choice = CandidateChoice(
                    index=0,
                    reason="模型重排失敗，使用檢索排序第一名。",
                    humor_style="rerank_fallback",
                    confidence=0.0,
                )
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

            candidate_records = []
            for index, (candidate, score, query_index, query_rank) in enumerate(
                zip(
                    candidates,
                    candidate_scores,
                    candidate_query_indexes,
                    candidate_query_ranks,
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
                        "season": candidate["season"],
                        "episode": candidate["episode"],
                        "frame_prefer": candidate["frame_prefer"],
                        "local_path": candidate["local_path"],
                    }
                )
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
                    "candidates": candidate_records,
                    "reranker": {
                        "model": self.settings.llm_model,
                        "selected_index": choice.index,
                        "reason": choice.reason,
                        "humor_style": choice.humor_style,
                        "confidence": choice.confidence,
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

    def should_auto_reply(self, message: discord.Message) -> bool:
        if not self.settings.auto_reply_enabled:
            return False
        if message.author.bot or message.webhook_id is not None:
            return False
        if not message.content or not message.content.strip():
            return False
        if message.guild is None:
            return self.settings.auto_reply_dms
        if message.channel.id in self.settings.auto_reply_channel_ids:
            return True
        if not self.settings.auto_reply_guild_mentions_only:
            return True
        return self.user is not None and self.user in message.mentions

    def take_reply_cooldown(self, message: discord.Message) -> bool:
        key = (
            message.guild.id if message.guild else 0,
            message.channel.id,
            message.author.id,
        )
        now = time.monotonic()
        previous = self.reply_cooldowns.get(key, 0.0)
        if now - previous < self.settings.auto_reply_cooldown_seconds:
            return False
        self.reply_cooldowns[key] = now
        return True

    async def on_message(self, message: discord.Message) -> None:
        if not self.should_auto_reply(message) or not self.take_reply_cooldown(message):
            return
        query = message.content.strip()
        if self.user is not None:
            query = query.replace(f"<@{self.user.id}>", "")
            query = query.replace(f"<@!{self.user.id}>", "")
            query = query.strip()
        if not query:
            return
        logging.info(
            "Received automatic message id=%s guild_id=%s channel_id=%s",
            message.id,
            message.guild.id if message.guild else None,
            message.channel.id,
        )
        try:
            async with message.channel.typing():
                path = await self.select_image(
                    query,
                    context={
                        "trigger": "automatic_message",
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
