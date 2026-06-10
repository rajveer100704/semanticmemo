"""SQLite cache store."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from semanticmemo._logging import get_logger
from semanticmemo.models import CacheEntry, EvictionPolicy, FeedbackEvent, LookupRecord
from semanticmemo.types import FloatVector

logger = get_logger(__name__)


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
    """SQLite-backed cache persistence.

    Concurrency model: the database is opened in WAL mode, so multiple processes
    can read concurrently while one writes. A single store instance is safe to
    call from multiple threads of one process -- writes are serialized by an
    internal re-entrant lock, and ``check_same_thread=False`` is set so the
    connection can cross threads (e.g. a thread pool). It is not a distributed
    cache: heavy multi-process write contention can still raise
    ``sqlite3.OperationalError`` once the 5-second busy timeout elapses.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant: evict() -> _evict_lru() -> delete() re-acquires the lock.
        self._lock = threading.RLock()
        self._connection = self._connect()
        self._connection.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._ensure_schema()
        journal_mode = self._connection.execute("PRAGMA journal_mode").fetchone()[0]
        logger.debug("opened sqlite store: path=%s journal_mode=%s", self.db_path, journal_mode)

    def close(self) -> None:
        self._connection.close()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False permits thread-pool use; correctness then
        # depends on self._lock serializing every write.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _apply_pragmas(self) -> None:
        with self._lock:
            # WAL: many concurrent readers alongside a single writer.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            # Wait out a competing writer rather than failing immediately.
            self._connection.execute("PRAGMA busy_timeout=5000")
            # SQLite ignores declared ON DELETE CASCADE unless this is on.
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.commit()

    def _ensure_schema(self) -> None:
        with self._lock:
            schema = resources.files("semanticmemo.store").joinpath("schema.sql").read_text()
            self._connection.executescript(schema)
            # Schema migration for cross_encoder_score column
            cursor = self._connection.execute("PRAGMA table_info(lookup_records)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "cross_encoder_score" not in columns:
                self._connection.execute(
                    "ALTER TABLE lookup_records ADD COLUMN cross_encoder_score REAL"
                )
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._connection.execute(
                """
                UPDATE cache_entries
                SET feedback_positive_count = feedback_positive_count + 1
                WHERE id = ?
                """,
                (str(entry_id),),
            )
            self._connection.commit()

    def record_lookup(
        self,
        *,
        query_id: UUID,
        domain: str,
        prompt: str,
        embedding: FloatVector,
        cache_entry_id: UUID,
        similarity_score: float | None,
        classifier_score: float | None,
        cross_encoder_score: float | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO lookup_records (
                    query_id, domain, query_prompt, query_embedding, cache_entry_id,
                    similarity_score, classifier_score, cross_encoder_score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(query_id),
                    domain,
                    prompt,
                    _to_blob(embedding),
                    str(cache_entry_id),
                    similarity_score,
                    classifier_score,
                    cross_encoder_score,
                    _serialize_time(_now()),
                ),
            )
            self._connection.commit()

    def get_lookup(self, query_id: UUID) -> LookupRecord | None:
        row = self._connection.execute(
            "SELECT * FROM lookup_records WHERE query_id = ?",
            (str(query_id),),
        ).fetchone()
        return self._row_to_lookup(row) if row is not None else None

    def find_implicit_bad_hit(
        self,
        *,
        domain: str,
        prompt: str,
        within_seconds: float,
    ) -> LookupRecord | None:
        """Return the most recent un-flagged cache-hit lookup matching ``prompt``.

        Used by implicit-feedback detection: if the same prompt was already a
        cache hit within ``within_seconds`` and carries no feedback yet, that
        earlier hit is the candidate to auto-flag as a bad hit. ``lookup_records``
        holds cache hits only, so a prior cache *miss* correctly yields no match.
        The ``fe.id IS NULL`` join condition skips lookups that already have a
        feedback event -- this both prevents double-flagging and lets an explicit
        good/bad report take precedence over implicit detection.
        """
        cutoff = _serialize_time(_now() - timedelta(seconds=within_seconds))
        row = self._connection.execute(
            """
            SELECT lr.* FROM lookup_records lr
            LEFT JOIN feedback_events fe ON fe.query_id = lr.query_id
            WHERE lr.domain = ?
              AND lr.query_prompt = ?
              AND lr.created_at >= ?
              AND fe.id IS NULL
            ORDER BY lr.created_at DESC
            LIMIT 1
            """,
            (domain, prompt, cutoff),
        ).fetchone()
        return self._row_to_lookup(row) if row is not None else None

    def lookup_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM lookup_records").fetchone()
        return int(row["count"])

    def record_feedback(
        self,
        *,
        query_id: UUID,
        label: int,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID | None:
        if label not in {0, 1}:
            msg = f"feedback label must be 0 or 1, got {label}"
            raise ValueError(msg)
        with self._lock:
            lookup = self.get_lookup(query_id)
            if lookup is None:
                return None
            event_id = uuid4()
            self._connection.execute(
                """
                INSERT INTO feedback_events (
                    id, query_id, cache_entry_id, label, reason, created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_id),
                    str(query_id),
                    str(lookup.cache_entry_id),
                    label,
                    reason,
                    _serialize_time(_now()),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            self._connection.commit()
            return event_id

    def feedback_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM feedback_events").fetchone()
        return int(row["count"])

    def feedback_events(self) -> Iterator[FeedbackEvent]:
        rows = self._connection.execute("SELECT * FROM feedback_events ORDER BY created_at ASC")
        for row in rows:
            yield self._row_to_feedback(row)

    def export_feedback_pairs(
        self,
        path: Path | str,
        *,
        split: str = "train",
    ) -> int:
        rows = self._connection.execute(
            """
            SELECT
                feedback_events.id AS event_id,
                feedback_events.query_id AS query_id,
                feedback_events.cache_entry_id AS cache_entry_id,
                feedback_events.label AS label,
                feedback_events.reason AS reason,
                feedback_events.created_at AS feedback_created_at,
                feedback_events.metadata_json AS feedback_metadata_json,
                lookup_records.domain AS domain,
                lookup_records.query_prompt AS query_prompt,
                lookup_records.similarity_score AS similarity_score,
                lookup_records.classifier_score AS classifier_score,
                lookup_records.cross_encoder_score AS cross_encoder_score,
                cache_entries.prompt AS cached_prompt
            FROM feedback_events
            JOIN lookup_records ON lookup_records.query_id = feedback_events.query_id
            JOIN cache_entries ON cache_entries.id = feedback_events.cache_entry_id
            ORDER BY feedback_events.created_at ASC
            """
        ).fetchall()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for row in rows:
            metadata = json.loads(row["feedback_metadata_json"])
            metadata.update(
                {
                    "feedback_event_id": row["event_id"],
                    "query_id": row["query_id"],
                    "cache_entry_id": row["cache_entry_id"],
                    "reason": row["reason"],
                    "similarity_score": row["similarity_score"],
                    "classifier_score": row["classifier_score"],
                    "cross_encoder_score": row["cross_encoder_score"],
                    "feedback_created_at": row["feedback_created_at"],
                }
            )
            lines.append(
                json.dumps(
                    {
                        "prompt_a": row["query_prompt"],
                        "prompt_b": row["cached_prompt"],
                        "label": int(row["label"]),
                        "domain": row["domain"],
                        "source": "SemanticMemo-feedback",
                        "split": split,
                        "metadata": metadata,
                    },
                    sort_keys=True,
                )
            )
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return len(lines)

    def record_active_learning_pair(
        self,
        *,
        domain: str,
        query_prompt: str,
        cached_prompt: str,
        similarity_score: float,
        classifier_score: float,
        cross_encoder_score: float,
        label: int,
        source: str,
    ) -> UUID:
        """Record a prompt pair for active learning (e.g. MLP/CE disagreements)."""
        pair_id = uuid4()
        created_at = _now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO active_learning_pairs (
                    id, domain, query_prompt, cached_prompt, similarity_score,
                    classifier_score, cross_encoder_score, label, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(pair_id),
                    domain,
                    query_prompt,
                    cached_prompt,
                    similarity_score,
                    classifier_score,
                    cross_encoder_score,
                    label,
                    source,
                    _serialize_time(created_at),
                ),
            )
            self._connection.commit()
        return pair_id

    def export_active_learning_pairs(
        self,
        path: Path | str,
        *,
        split: str = "train",
    ) -> int:
        """Export active learning prompt pairs to a JSONL file in benchmark format."""
        rows = self._connection.execute(
            """
            SELECT * FROM active_learning_pairs
            ORDER BY created_at ASC
            """
        ).fetchall()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for row in rows:
            metadata = {
                "active_learning_id": row["id"],
                "similarity_score": row["similarity_score"],
                "classifier_score": row["classifier_score"],
                "cross_encoder_score": row["cross_encoder_score"],
                "created_at": row["created_at"],
            }
            lines.append(
                json.dumps(
                    {
                        "prompt_a": row["query_prompt"],
                        "prompt_b": row["cached_prompt"],
                        "label": int(row["label"]),
                        "domain": row["domain"],
                        "source": row["source"],
                        "split": split,
                        "metadata": metadata,
                    },
                    sort_keys=True,
                )
            )
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return len(lines)

    def delete(self, entry_id: UUID) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM cache_entries WHERE id = ?", (str(entry_id),))
            self._connection.commit()

    def evict(
        self,
        *,
        policy: EvictionPolicy,
        max_entries: int,
        ttl_seconds: int | None,
    ) -> list[UUID]:
        with self._lock:
            evicted: list[UUID] = []
            if policy in {EvictionPolicy.TTL, EvictionPolicy.HYBRID} and ttl_seconds is not None:
                evicted.extend(self._evict_expired(ttl_seconds))
            if policy in {EvictionPolicy.LRU, EvictionPolicy.HYBRID}:
                evicted.extend(self._evict_lru(max_entries))
            return evicted

    def _evict_expired(self, ttl_seconds: int) -> list[UUID]:
        with self._lock:
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
        with self._lock:
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

    def _row_to_lookup(self, row: sqlite3.Row) -> LookupRecord:
        return LookupRecord(
            id=UUID(row["query_id"]),
            domain=row["domain"],
            prompt=row["query_prompt"],
            prompt_embedding=_from_blob(row["query_embedding"]),
            cache_entry_id=UUID(row["cache_entry_id"]),
            similarity_score=row["similarity_score"],
            classifier_score=row["classifier_score"],
            cross_encoder_score=row["cross_encoder_score"],
            created_at=_parse_time(row["created_at"]) or _now(),
        )

    def _row_to_feedback(self, row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent(
            id=UUID(row["id"]),
            query_id=UUID(row["query_id"]),
            cache_entry_id=UUID(row["cache_entry_id"]),
            label=int(row["label"]),
            reason=row["reason"],
            created_at=_parse_time(row["created_at"]) or _now(),
            metadata=json.loads(row["metadata_json"]),
        )
