from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

from .policy import normalize_activity, normalize_mode


def _optional_int(name: str):
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _int_set(name: str) -> frozenset[int]:
    value = os.getenv(name, "").strip()
    if not value:
        return frozenset()
    return frozenset(int(item.strip()) for item in value.split(",") if item.strip())


def _string_tuple(name: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


@dataclass(frozen=True)
class Settings:
    discord_token: str
    discord_guild_id: Optional[int]
    auto_reply_enabled: bool
    auto_reply_dms: bool
    auto_reply_guild_mentions_only: bool
    auto_reply_channel_ids: frozenset[int]
    auto_reply_cooldown_seconds: float
    auto_reply_mode: str
    auto_reply_activity: str
    context_message_limit: int
    decision_log_enabled: bool
    decision_log_path_override: Optional[Path]
    data_dir: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    bot_aliases: tuple[str, ...]
    final_judge_enabled: bool
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    reranker_base_url: str
    reranker_api_key: str
    reranker_model: str
    retrieval_pool_size: int
    reranker_top_n: int
    db_url: str
    image_base_url: str
    download_concurrency: int
    download_delay_seconds: float
    download_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            discord_guild_id=_optional_int("DISCORD_GUILD_ID"),
            auto_reply_enabled=_bool("AUTO_REPLY_ENABLED", True),
            auto_reply_dms=_bool("AUTO_REPLY_DMS", True),
            auto_reply_guild_mentions_only=_bool(
                "AUTO_REPLY_GUILD_MENTIONS_ONLY",
                True,
            ),
            auto_reply_channel_ids=_int_set("AUTO_REPLY_CHANNEL_IDS"),
            auto_reply_cooldown_seconds=max(
                0.0,
                float(os.getenv("AUTO_REPLY_COOLDOWN_SECONDS", "10")),
            ),
            auto_reply_mode=normalize_mode(
                os.getenv(
                    "AUTO_REPLY_MODE",
                    "auto" if _bool("AUTO_REPLY_ENABLED", True) else "off",
                )
            ),
            auto_reply_activity=normalize_activity(
                os.getenv("AUTO_REPLY_ACTIVITY", "medium")
            ),
            context_message_limit=min(
                10,
                max(1, int(os.getenv("CONTEXT_MESSAGE_LIMIT", "5"))),
            ),
            decision_log_enabled=_bool("DECISION_LOG_ENABLED", True),
            decision_log_path_override=(
                Path(os.environ["DECISION_LOG_PATH"])
                if os.getenv("DECISION_LOG_PATH", "").strip()
                else None
            ),
            data_dir=Path(os.getenv("DATA_DIR", "/var/lib/mortis-bot")),
            llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            bot_aliases=_string_tuple("BOT_ALIASES", "Mortis,Motis"),
            final_judge_enabled=_bool("FINAL_JUDGE_ENABLED", False),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").rstrip("/"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
            reranker_base_url=os.getenv("RERANKER_BASE_URL", "").rstrip("/"),
            reranker_api_key=os.getenv("RERANKER_API_KEY", "").strip(),
            reranker_model=os.getenv("RERANKER_MODEL", "").strip(),
            retrieval_pool_size=max(
                12,
                int(os.getenv("RETRIEVAL_POOL_SIZE", "16")),
            ),
            reranker_top_n=max(
                1,
                int(os.getenv("RERANKER_TOP_N", "5")),
            ),
            db_url=os.getenv(
                "MYPIC_DB_URL",
                "https://raw.githubusercontent.com/Its-MyPic/Its-MyPicDB/json/data.json",
            ),
            image_base_url=os.getenv(
                "MYPIC_IMAGE_BASE_URL", "https://mypic.0m0.uk/images"
            ).rstrip("/"),
            download_concurrency=max(1, int(os.getenv("DOWNLOAD_CONCURRENCY", "2"))),
            download_delay_seconds=max(
                0.0, float(os.getenv("DOWNLOAD_DELAY_SECONDS", "0.25"))
            ),
            download_timeout_seconds=max(
                5.0, float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "30"))
            ),
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mypic.sqlite3"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "data.json"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def decision_log_path(self) -> Path:
        return self.decision_log_path_override or self.data_dir / "decisions.jsonl"
