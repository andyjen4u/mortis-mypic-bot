from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import discord
from discord import app_commands
import httpx

from .config import Settings
from .database import connect, search
from .embeddings import SemanticIndex, meme_retrieval_queries, request_embeddings
from .images import download_one
from .llm import choose_candidate


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

    async def close(self) -> None:
        self.connection.close()
        await super().close()

    async def select_image(self, query: str) -> Path | None:
        async with self.selection_lock:
            candidates = []
            if self.semantic_index is not None and self.settings.embedding_base_url:
                try:
                    query_vectors = await request_embeddings(
                        self.settings,
                        meme_retrieval_queries(query),
                    )
                    candidates = [
                        entry
                        for entry, _score in self.semantic_index.diverse_search_entries(
                            self.connection,
                            query_vectors,
                            per_query_limit=4,
                            total_limit=12,
                        )
                    ]
                except Exception:
                    logging.exception("Semantic retrieval failed; using text search")
            if not candidates:
                candidates = search(self.connection, query, limit=12)
            if not candidates:
                return None
            try:
                selected_index = await choose_candidate(
                    self.settings,
                    query,
                    candidates,
                )
            except Exception:
                logging.exception("LLM reranking failed; using first candidate")
                selected_index = 0
            selected = candidates[selected_index]
            local_path = selected["local_path"]
            if local_path and Path(local_path).is_file():
                return Path(local_path)

            semaphore = asyncio.Semaphore(1)
            headers = {"User-Agent": "MortisMyPicBot/0.1 (private Discord bot)"}
            async with httpx.AsyncClient(
                timeout=self.settings.download_timeout_seconds,
                headers=headers,
            ) as http_client:
                return await download_one(
                    self.settings,
                    self.connection,
                    http_client,
                    selected,
                    semaphore,
                )

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
                path = await self.select_image(query)
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
        path = await client.select_image(query)
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
