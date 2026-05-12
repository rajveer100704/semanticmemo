"""SQLite cache store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from equivcache.models import CacheEntry, EvictionPolicy
from equivcache.types import FloatVector


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _to_blob(embedding: FloatVector) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float32).tolist()


def _serialize_time(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


class SQLiteCacheStore:
    """SQLite-backed cache persistence."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._connection.close()

    def _ensure_schema(self) -> None:
        schema = resources.files("equivcache.store").joinpath("schema.sql").read_text()
        self._connection.executescript(schema)
        self._connection.commit()

    def add(
        self,
        *,
        prompt: str,
        embedding: FloatVector,
        response: str,
        model: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        entry_id = uuid4()
        created_at = _now()
        self._connection.execute(
            """
            INSERT INTO cache_entries (
                id, prompt, prompt_embedding, response, model, created_at,
                last_hit_at, hit_count, feedback_negative_count,
                feedback_positive_count, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0, 0, ?)
            """,
            (
                str(entry_id),
                prompt,
                _to_blob(embedding),
                response,
                model,
                _serialize_time(created_at),
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self._connection.commit()
        return entry_id

    def get(self, entry_id: UUID) -> CacheEntry | None:
        row = self._connection.execute(
            "SELECT * FROM cache_entries WHERE id = ?",
            (str(entry_id),),
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def all_entries(self) -> Iterator[CacheEntry]:
        rows = self._connection.execute("SELECT * FROM cache_entries ORDER BY created_at ASC")
        for row in rows:
            yield self._row_to_entry(row)

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM cache_entries").fetchone()
        return int(row["count"])

    def total_hit_count(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(hit_count), 0) AS hits FROM cache_entries"
        ).fetchone()
        return int(row["hits"])

    def update_hit(self, entry_id: UUID) -> None:
        self._connection.execute(
            """
            UPDATE cache_entries
            SET hit_count = hit_count + 1, last_hit_at = ?
            WHERE id = ?
            """,
            (_serialize_time(_now()), str(entry_id)),
        )
        self._connection.commit()

    def increment_bad_feedback(self, entry_id: UUID) -> None:
        self._connection.execute(
            """
            UPDATE cache_entries
            SET feedback_negative_count = feedback_negative_count + 1
            WHERE id = ?
            """,
            (str(entry_id),),
        )
        self._connection.commit()

    def increment_good_feedback(self, entry_id: UUID) -> None:
        self._connection.execute(
            """
            UPDATE cache_entries
            SET feedback_positive_count = feedback_positive_count + 1
            WHERE id = ?
            """,
            (str(entry_id),),
        )
        self._connection.commit()

    def delete(self, entry_id: UUID) -> None:
        self._connection.execute("DELETE FROM cache_entries WHERE id = ?", (str(entry_id),))
        self._connection.commit()

    def evict(
        self,
        *,
        policy: EvictionPolicy,
        max_entries: int,
        ttl_seconds: int | None,
    ) -> list[UUID]:
        evicted: list[UUID] = []
        if policy in {EvictionPolicy.TTL, EvictionPolicy.HYBRID} and ttl_seconds is not None:
            evicted.extend(self._evict_expired(ttl_seconds))
        if policy in {EvictionPolicy.LRU, EvictionPolicy.HYBRID}:
            evicted.extend(self._evict_lru(max_entries))
        return evicted

    def _evict_expired(self, ttl_seconds: int) -> list[UUID]:
        cutoff = _serialize_time(_now() - timedelta(seconds=ttl_seconds))
        rows = self._connection.execute(
            "SELECT id FROM cache_entries WHERE created_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [UUID(row["id"]) for row in rows]
        for entry_id in ids:
            self.delete(entry_id)
        return ids

    def _evict_lru(self, max_entries: int) -> list[UUID]:
        overflow = self.count() - max_entries
        if overflow <= 0:
            return []
        rows = self._connection.execute(
            """
            SELECT id FROM cache_entries
            ORDER BY
                feedback_negative_count DESC,
                COALESCE(last_hit_at, created_at) ASC,
                created_at ASC
            LIMIT ?
            """,
            (overflow,),
        ).fetchall()
        ids = [UUID(row["id"]) for row in rows]
        for entry_id in ids:
            self.delete(entry_id)
        return ids

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        metadata = json.loads(row["metadata_json"])
        return CacheEntry(
            id=UUID(row["id"]),
            prompt=row["prompt"],
            prompt_embedding=_from_blob(row["prompt_embedding"]),
            response=row["response"],
            model=row["model"],
            created_at=_parse_time(row["created_at"]) or _now(),
            last_hit_at=_parse_time(row["last_hit_at"]),
            hit_count=int(row["hit_count"]),
            feedback_negative_count=int(row["feedback_negative_count"]),
            feedback_positive_count=int(row["feedback_positive_count"]),
            metadata=metadata,
        )
