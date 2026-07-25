import asyncio
import logging

import discord
from discord import app_commands

from .config import Settings
from .database import connect, search
from .embeddings import SemanticIndex, meme_retrieval_queries, request_embeddings
from .images import download_one
from .llm import choose_candidate


class MyPicClient(discord.Client):
    def __init__(self, settings: Settings):
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.tree = app_commands.CommandTree(self)
        self.connection = connect(settings.db_path)
        self.semantic_index = SemanticIndex.from_database(self.connection)

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

    async def close(self) -> None:
        self.connection.close()
        await super().close()


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
        candidates = []
        if client.semantic_index is not None and settings.embedding_base_url:
            try:
                query_vectors = await request_embeddings(
                    settings,
                    meme_retrieval_queries(query),
                )
                candidates = [
                    entry
                    for entry, _score in client.semantic_index.diverse_search_entries(
                        client.connection,
                        query_vectors,
                        per_query_limit=4,
                        total_limit=12,
                    )
                ]
            except Exception:
                logging.exception("Semantic retrieval failed; using text search")
        if not candidates:
            candidates = search(client.connection, query, limit=12)
        if not candidates:
            await interaction.followup.send("目前找不到合適的圖片。", ephemeral=True)
            return
        try:
            selected_index = await choose_candidate(settings, query, candidates)
        except Exception:
            logging.exception("LLM reranking failed; using first candidate")
            selected_index = 0
        selected = candidates[selected_index]
        local_path = selected["local_path"]
        from pathlib import Path

        if not local_path or not Path(local_path).is_file():
            import httpx

            semaphore = asyncio.Semaphore(1)
            headers = {"User-Agent": "MortisMyPicBot/0.1 (private Discord bot)"}
            async with httpx.AsyncClient(
                timeout=settings.download_timeout_seconds, headers=headers
            ) as http_client:
                path = await download_one(
                    settings,
                    client.connection,
                    http_client,
                    selected,
                    semaphore,
                )
        else:
            path = Path(local_path)
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
