import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS entries (
    segment_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    season INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    frame_start INTEGER,
    frame_prefer INTEGER NOT NULL,
    frame_end INTEGER,
    character INTEGER NOT NULL DEFAULT 0,
    local_path TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    text,
    content='entries',
    content_rowid='segment_id',
    tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS entry_embeddings (
    segment_id INTEGER PRIMARY KEY REFERENCES entries(segment_id) ON DELETE CASCADE,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS reply_policies (
    scope_type TEXT NOT NULL,
    scope_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    activity TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scope_type, scope_id)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def import_metadata(connection: sqlite3.Connection, metadata_path: Path) -> int:
    items: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        (
            int(item["segment_id"]),
            str(item.get("text", "")).strip(),
            int(item["season"]),
            int(item["episode"]),
            item.get("frame_start"),
            int(item["frame_prefer"]),
            item.get("frame_end"),
            int(item.get("character", 0)),
        )
        for item in items
        if item.get("text") and item.get("frame_prefer") is not None
    ]
    with connection:
        connection.executemany(
            """
            INSERT INTO entries (
                segment_id, text, season, episode, frame_start,
                frame_prefer, frame_end, character
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(segment_id) DO UPDATE SET
                text=excluded.text,
                season=excluded.season,
                episode=excluded.episode,
                frame_start=excluded.frame_start,
                frame_prefer=excluded.frame_prefer,
                frame_end=excluded.frame_end,
                character=excluded.character
            """,
            rows,
        )
        connection.execute("INSERT INTO entries_fts(entries_fts) VALUES ('rebuild')")
    return len(rows)


def search(connection: sqlite3.Connection, query: str, limit: int = 12):
    tokens = [token.replace('"', "") for token in query.split() if token.strip()]
    if tokens:
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        rows = connection.execute(
            """
            SELECT e.*, bm25(entries_fts) AS rank
            FROM entries_fts
            JOIN entries AS e ON e.segment_id = entries_fts.rowid
            WHERE entries_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        if rows:
            return rows
    return connection.execute(
        """
        SELECT *, 0.0 AS rank
        FROM entries
        WHERE text LIKE ?
        ORDER BY segment_id DESC
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()


def pending_images(connection: sqlite3.Connection, limit: Optional[int] = None):
    sql = "SELECT * FROM entries WHERE local_path IS NULL ORDER BY segment_id"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    return connection.execute(sql, params).fetchall()


def mark_downloaded(
    connection: sqlite3.Connection, segment_id: int, local_path: Path
) -> None:
    with connection:
        connection.execute(
            "UPDATE entries SET local_path=? WHERE segment_id=?",
            (str(local_path), segment_id),
        )


def entries_missing_embeddings(connection: sqlite3.Connection, limit: int):
    return connection.execute(
        """
        SELECT e.segment_id, e.text
        FROM entries AS e
        LEFT JOIN entry_embeddings AS x ON x.segment_id = e.segment_id
        WHERE x.segment_id IS NULL
        ORDER BY e.segment_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def store_embeddings(connection: sqlite3.Connection, rows) -> None:
    with connection:
        connection.executemany(
            """
            INSERT INTO entry_embeddings (segment_id, dimensions, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(segment_id) DO UPDATE SET
                dimensions=excluded.dimensions,
                embedding=excluded.embedding
            """,
            rows,
        )


def clear_embeddings(connection: sqlite3.Connection) -> int:
    count = connection.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    with connection:
        connection.execute("DELETE FROM entry_embeddings")
    return count


def embedding_counts(connection: sqlite3.Connection):
    total = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    embedded = connection.execute(
        "SELECT COUNT(*) FROM entry_embeddings"
    ).fetchone()[0]
    return total, embedded


def load_embedding_rows(connection: sqlite3.Connection):
    return connection.execute(
        """
        SELECT segment_id, dimensions, embedding
        FROM entry_embeddings
        ORDER BY segment_id
        """
    ).fetchall()


def get_entries_by_ids(connection: sqlite3.Connection, segment_ids):
    if not segment_ids:
        return []
    placeholders = ",".join("?" for _ in segment_ids)
    rows = connection.execute(
        f"SELECT * FROM entries WHERE segment_id IN ({placeholders})",
        tuple(segment_ids),
    ).fetchall()
    by_id = {row["segment_id"]: row for row in rows}
    return [by_id[segment_id] for segment_id in segment_ids if segment_id in by_id]


def get_reply_policy(
    connection: sqlite3.Connection,
    scope_type: str,
    scope_id: int,
):
    return connection.execute(
        """
        SELECT mode, activity
        FROM reply_policies
        WHERE scope_type=? AND scope_id=?
        """,
        (scope_type, scope_id),
    ).fetchone()


def set_reply_policy(
    connection: sqlite3.Connection,
    scope_type: str,
    scope_id: int,
    mode: str,
    activity: str,
) -> None:
    with connection:
        connection.execute(
            """
            INSERT INTO reply_policies (scope_type, scope_id, mode, activity)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_type, scope_id) DO UPDATE SET
                mode=excluded.mode,
                activity=excluded.activity,
                updated_at=CURRENT_TIMESTAMP
            """,
            (scope_type, scope_id, mode, activity),
        )
