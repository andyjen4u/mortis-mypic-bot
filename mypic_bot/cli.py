import argparse
import asyncio
from typing import Optional

from .config import Settings
from .database import clear_embeddings, connect, import_metadata, pending_images
from .database import embedding_counts
from .embeddings import (
    SemanticIndex,
    build_missing_embeddings,
    meme_retrieval_queries,
    request_embeddings,
)
from .llm import choose_candidate
from .images import download_many
from .sync import fetch_metadata


def print_status(settings: Settings) -> None:
    connection = connect(settings.db_path)
    total = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    downloaded = connection.execute(
        "SELECT COUNT(*) FROM entries WHERE local_path IS NOT NULL"
    ).fetchone()[0]
    missing = total - downloaded
    embedding_total, embedded = embedding_counts(connection)
    print(f"Metadata rows: {total}")
    print(f"Downloaded rows: {downloaded}")
    print(f"Missing rows: {missing}")
    print(f"Embedded rows: {embedded}")
    print(f"Unembedded rows: {embedding_total - embedded}")
    connection.close()


async def run_sync(settings: Settings, download_limit: int) -> None:
    await fetch_metadata(settings)
    connection = connect(settings.db_path)
    count = import_metadata(connection, settings.metadata_path)
    print(f"Imported {count} metadata rows.")
    if download_limit:
        rows = pending_images(connection, download_limit)
        succeeded, failed = await download_many(settings, connection, rows)
        print(f"Downloaded {succeeded}; failed {failed}.")
    connection.close()


async def run_download(settings: Settings, limit: Optional[int]) -> None:
    connection = connect(settings.db_path)
    rows = pending_images(connection, limit)
    succeeded, failed = await download_many(settings, connection, rows)
    print(f"Downloaded {succeeded}; failed {failed}.")
    connection.close()


async def run_embed(settings: Settings, batch_size: int, rebuild: bool) -> None:
    connection = connect(settings.db_path)
    if rebuild:
        removed = clear_embeddings(connection)
        print(f"Cleared {removed} existing embeddings.", flush=True)
    processed, total = await build_missing_embeddings(
        settings,
        connection,
        batch_size,
    )
    print(f"New embeddings: {processed}; total metadata rows: {total}.")
    connection.close()


async def run_query(
    settings: Settings,
    query: str,
    limit: int,
    rerank: bool,
) -> None:
    connection = connect(settings.db_path)
    index = SemanticIndex.from_database(connection)
    if index is None:
        raise SystemExit("A complete semantic index is not available.")
    query_vectors = await request_embeddings(
        settings,
        meme_retrieval_queries(query),
    )
    scored_entries = index.diverse_search_entries(
        connection,
        query_vectors,
        per_query_limit=max(1, min(8, limit)),
        total_limit=limit,
    )
    candidates = [entry for entry, _score in scored_entries]
    for position, (entry, score) in enumerate(scored_entries):
        print(
            f"{position}: score={score:.4f} segment={entry['segment_id']} "
            f"text={entry['text']}"
        )
    if rerank:
        choice = await choose_candidate(settings, query, candidates)
        print(f"LLM_SELECTED={choice.index}")
        print(f"LLM_REASON={choice.reason}")
        print(f"LLM_HUMOR_STYLE={choice.humor_style}")
        print(f"LLM_CONFIDENCE={choice.confidence:.2f}")
        print(f"SELECTED_TEXT={candidates[choice.index]['text']}")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--download-limit", type=int, default=0)
    download_parser = subparsers.add_parser("download")
    download_group = download_parser.add_mutually_exclusive_group(required=True)
    download_group.add_argument("--limit", type=int)
    download_group.add_argument("--all", action="store_true")
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--batch-size", type=int, default=16)
    embed_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete all derived vectors before rebuilding them.",
    )
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("text")
    query_parser.add_argument("--limit", type=int, default=10)
    query_parser.add_argument("--rerank", action="store_true")
    subparsers.add_parser("status")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "sync":
        asyncio.run(run_sync(settings, max(0, args.download_limit)))
    elif args.command == "download":
        asyncio.run(run_download(settings, None if args.all else args.limit))
    elif args.command == "embed":
        asyncio.run(run_embed(settings, max(1, args.batch_size), args.rebuild))
    elif args.command == "query":
        asyncio.run(
            run_query(
                settings,
                args.text,
                max(1, args.limit),
                args.rerank,
            )
        )
    else:
        print_status(settings)


if __name__ == "__main__":
    main()
