import httpx

from .config import Settings


async def fetch_metadata(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    temporary = settings.metadata_path.with_suffix(".json.part")
    headers = {"User-Agent": "MortisMyPicBot/0.1 (private Discord bot)"}
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        response = await client.get(settings.db_url)
        response.raise_for_status()
    temporary.write_bytes(response.content)
    temporary.replace(settings.metadata_path)

