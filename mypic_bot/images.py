import asyncio
from pathlib import Path
from typing import Optional

import httpx

from .config import Settings
from .database import mark_downloaded


def image_url(settings: Settings, row) -> str:
    return (
        f"{settings.image_base_url}/{row['season']}/"
        f"{row['episode']}/{row['frame_prefer']}.webp"
    )


def image_path(settings: Settings, row) -> Path:
    return (
        settings.images_dir
        / str(row["season"])
        / str(row["episode"])
        / f"{row['frame_prefer']}.webp"
    )


async def download_one(
    settings: Settings,
    connection,
    client: httpx.AsyncClient,
    row,
    semaphore: asyncio.Semaphore,
) -> Path:
    destination = image_path(settings, row)
    if destination.exists() and destination.stat().st_size > 0:
        mark_downloaded(connection, row["segment_id"], destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".webp.part")
    async with semaphore:
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                response = await client.get(image_url(settings, row))
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "image/" not in content_type:
                    raise RuntimeError(f"unexpected content type: {content_type}")
                temporary.write_bytes(response.content)
                temporary.replace(destination)
                mark_downloaded(connection, row["segment_id"], destination)
                await asyncio.sleep(settings.download_delay_seconds)
                return destination
            except Exception as error:
                last_error = error
                if attempt < 3:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"failed to download {image_url(settings, row)}") from last_error


async def download_many(settings: Settings, connection, rows) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(settings.download_concurrency)
    timeout = httpx.Timeout(settings.download_timeout_seconds)
    headers = {"User-Agent": "MortisMyPicBot/0.1 (private Discord bot)"}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        results = await asyncio.gather(
            *[
                download_one(settings, connection, client, row, semaphore)
                for row in rows
            ],
            return_exceptions=True,
        )
    succeeded = sum(not isinstance(result, Exception) for result in results)
    return succeeded, len(results) - succeeded
