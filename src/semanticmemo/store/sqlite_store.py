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
from semanticmemo.events.models import (
    EventType,
    OutboxEvent,
    OutboxStatus,
)
from semanticmemo.models import (
    CacheEntry,
    CacheScope,
    EvictionPolicy,
    FeedbackEvent,
    LookupRecord,
    Permission,
    PermissionName,
    Resource,
    Role,
    User,
)
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

            # Schema migration for lookup_records
            cursor = self._connection.execute("PRAGMA table_info(lookup_records)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "cross_encoder_score" not in columns:
                self._connection.execute(
                    "ALTER TABLE lookup_records ADD COLUMN cross_encoder_score REAL"
                )

            # Schema migrations for cache_entries
            cursor = self._connection.execute("PRAGMA table_info(cache_entries)")
            cache_cols = [row["name"] for row in cursor.fetchall()]
            if "tenant_id" not in cache_cols:
                self._connection.execute(
                    "ALTER TABLE cache_entries ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "scope" not in cache_cols:
                self._connection.execute(
                    "ALTER TABLE cache_entries ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'"
                )
            if "resource_id" not in cache_cols:
                self._connection.execute("ALTER TABLE cache_entries ADD COLUMN resource_id TEXT")
            if "resource_version" not in cache_cols:
                self._connection.execute(
                    "ALTER TABLE cache_entries ADD COLUMN resource_version INTEGER"
                )
            if "pipeline_version" not in cache_cols:
                self._connection.execute(
                    "ALTER TABLE cache_entries "
                    "ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'v2'"
                )
            if "expires_at" not in cache_cols:
                self._connection.execute("ALTER TABLE cache_entries ADD COLUMN expires_at TEXT")
            if "flight_generation" not in cache_cols:
                self._connection.execute(
                    "ALTER TABLE cache_entries "
                    "ADD COLUMN flight_generation INTEGER NOT NULL DEFAULT 0"
                )

            # Schema migrations for active_learning_pairs
            cursor = self._connection.execute("PRAGMA table_info(active_learning_pairs)")
            al_cols = [row["name"] for row in cursor.fetchall()]
            if "tenant_id" not in al_cols:
                self._connection.execute(
                    "ALTER TABLE active_learning_pairs "
                    "ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "resource_id" not in al_cols:
                self._connection.execute(
                    "ALTER TABLE active_learning_pairs ADD COLUMN resource_id TEXT"
                )
            if "pipeline_version" not in al_cols:
                self._connection.execute(
                    "ALTER TABLE active_learning_pairs "
                    "ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'v2'"
                )

            # Create scoped index after columns are ensured
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_entries_scoped "
                "ON cache_entries(tenant_id, scope, resource_id, resource_version)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_active_learning_pairs_scoped "
                "ON active_learning_pairs(tenant_id, resource_id, pipeline_version)"
            )

            # Seed default roles and permissions
            self._seed_default_rbac()
            self._connection.commit()

    def _seed_default_rbac(self) -> None:
        default_roles = ["admin", "analyst", "viewer"]
        for role in default_roles:
            self._connection.execute(
                "INSERT OR IGNORE INTO roles (id, name) VALUES (?, ?)",
                (role, role),
            )

        default_perms = [
            PermissionName.CACHE_READ.value,
            PermissionName.CACHE_WRITE.value,
            PermissionName.CACHE_INVALIDATE.value,
            PermissionName.RESOURCE_READ.value,
            PermissionName.RESOURCE_WRITE.value,
            PermissionName.RESOURCE_MANAGE.value,
        ]
        for perm in default_perms:
            self._connection.execute(
                "INSERT OR IGNORE INTO permissions (id, name) VALUES (?, ?)",
                (perm, perm),
            )

        role_perms_map = {
            "viewer": [PermissionName.CACHE_READ.value, PermissionName.RESOURCE_READ.value],
            "analyst": [
                PermissionName.CACHE_READ.value,
                PermissionName.CACHE_WRITE.value,
                PermissionName.RESOURCE_READ.value,
            ],
            "admin": default_perms,
        }
        for role, perms in role_perms_map.items():
            for perm in perms:
                self._connection.execute(
                    "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                    (role, perm),
                )

    def add(
        self,
        *,
        prompt: str,
        embedding: FloatVector,
        response: str,
        model: str | None,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        flight_generation: int = 0,
    ) -> UUID | None:
        entry_id = uuid4()
        created_at = _now()
        scope_str = scope.value if isinstance(scope, CacheScope) else str(scope)
        with self._lock:
            if flight_generation > 0:
                existing = self._connection.execute(
                    """
                    SELECT id, flight_generation FROM cache_entries
                    WHERE prompt = ?
                      AND tenant_id = ?
                      AND scope = ?
                      AND (resource_id IS ? OR resource_id = ?)
                      AND (resource_version IS ? OR resource_version = ?)
                      AND pipeline_version = ?
                    ORDER BY flight_generation DESC, created_at DESC
                    LIMIT 1
                    """,
                    (
                        prompt,
                        tenant_id,
                        scope_str,
                        resource_id,
                        resource_id,
                        resource_version,
                        resource_version,
                        pipeline_version,
                    ),
                ).fetchone()
                if existing is not None and existing["flight_generation"] > flight_generation:
                    logger.warning(
                        "CAS write rejected: gen %d > candidate %d",
                        existing["flight_generation"],
                        flight_generation,
                    )
                    return None

            self._connection.execute(
                """
                INSERT INTO cache_entries (
                    id, prompt, prompt_embedding, response, model, tenant_id, scope,
                    resource_id, resource_version, pipeline_version, expires_at, created_at,
                    last_hit_at, hit_count, feedback_negative_count,
                    feedback_positive_count, flight_generation, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, 0, ?, ?)
                """,
                (
                    str(entry_id),
                    prompt,
                    _to_blob(embedding),
                    response,
                    model,
                    tenant_id,
                    scope_str,
                    resource_id,
                    resource_version,
                    pipeline_version,
                    _serialize_time(expires_at),
                    _serialize_time(created_at),
                    flight_generation,
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

    def find_by_prompt(
        self,
        *,
        prompt: str,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> CacheEntry | None:
        """Find an unexpired cache entry matching exact prompt and scope."""
        scope_str = scope.value if isinstance(scope, CacheScope) else str(scope)
        now_str = _serialize_time(_now())
        row = self._connection.execute(
            """
            SELECT * FROM cache_entries
            WHERE prompt = ?
              AND tenant_id = ?
              AND scope = ?
              AND (resource_id IS ? OR resource_id = ?)
              AND (resource_version IS ? OR resource_version = ?)
              AND pipeline_version = ?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                prompt,
                tenant_id,
                scope_str,
                resource_id,
                resource_id,
                resource_version,
                resource_version,
                pipeline_version,
                now_str,
            ),
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
        tenant_id: str | None = None,
        resource_id: str | None = None,
        pipeline_version: str | None = None,
    ) -> LookupRecord | None:
        """Return the most recent un-flagged cache-hit lookup matching ``prompt``.

        Used by implicit-feedback detection: if the same prompt was already a
        cache hit within ``within_seconds`` and carries no feedback yet, that
        earlier hit is the candidate to auto-flag as a bad hit. Scoped by tenant,
        resource, and pipeline version to prevent multi-tenant feedback pollution.
        """
        cutoff = _serialize_time(_now() - timedelta(seconds=within_seconds))
        if tenant_id is not None:
            query = """
                SELECT lr.* FROM lookup_records lr
                JOIN cache_entries ce ON lr.cache_entry_id = ce.id
                LEFT JOIN feedback_events fe ON fe.query_id = lr.query_id
                WHERE lr.domain = ?
                  AND lr.query_prompt = ?
                  AND lr.created_at >= ?
                  AND fe.id IS NULL
                  AND ce.tenant_id = ?
                  AND (? IS NULL OR ce.resource_id = ?)
                  AND (? IS NULL OR ce.pipeline_version = ?)
                ORDER BY lr.created_at DESC
                LIMIT 1
            """
            params = (
                domain,
                prompt,
                cutoff,
                tenant_id,
                resource_id,
                resource_id,
                pipeline_version,
                pipeline_version,
            )
        else:
            query = """
                SELECT lr.* FROM lookup_records lr
                LEFT JOIN feedback_events fe ON fe.query_id = lr.query_id
                WHERE lr.domain = ?
                  AND lr.query_prompt = ?
                  AND lr.created_at >= ?
                  AND fe.id IS NULL
                ORDER BY lr.created_at DESC
                LIMIT 1
            """
            params = (domain, prompt, cutoff)

        row = self._connection.execute(query, params).fetchone()
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
        cross_encoder_score: float | None,
        label: int,
        source: str,
        tenant_id: str = "default",
        resource_id: str | None = None,
        pipeline_version: str = "v2",
    ) -> UUID:
        """Record a prompt pair for active learning with tenant provenance."""
        pair_id = uuid4()
        created_at = _now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO active_learning_pairs (
                    id, domain, query_prompt, cached_prompt, similarity_score,
                    classifier_score, cross_encoder_score, label, source,
                    tenant_id, resource_id, pipeline_version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    tenant_id,
                    resource_id,
                    pipeline_version,
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
        tenant_id: str | None = None,
    ) -> int:
        """Export active learning prompt pairs to a JSONL file in benchmark format."""
        if tenant_id:
            rows = self._connection.execute(
                """
                SELECT * FROM active_learning_pairs
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """,
                (tenant_id,),
            ).fetchall()
        else:
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
            row_keys = row.keys()
            metadata = {
                "active_learning_id": row["id"],
                "similarity_score": row["similarity_score"],
                "classifier_score": row["classifier_score"],
                "cross_encoder_score": row["cross_encoder_score"],
                "tenant_id": row["tenant_id"] if "tenant_id" in row_keys else "default",
                "resource_id": row["resource_id"] if "resource_id" in row_keys else None,
                "pipeline_version": (
                    row["pipeline_version"] if "pipeline_version" in row_keys else "v2"
                ),
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
        keys = row.keys()
        tenant_id = (
            row["tenant_id"] if "tenant_id" in keys and row["tenant_id"] is not None else "default"
        )
        raw_scope = row["scope"] if "scope" in keys and row["scope"] is not None else "global"
        try:
            scope = CacheScope(raw_scope)
        except ValueError:
            scope = CacheScope.GLOBAL
        resource_id = row["resource_id"] if "resource_id" in keys else None
        resource_version = (
            int(row["resource_version"])
            if "resource_version" in keys and row["resource_version"] is not None
            else None
        )
        pipeline_version = (
            row["pipeline_version"]
            if "pipeline_version" in keys and row["pipeline_version"] is not None
            else "v2"
        )
        expires_at = (
            _parse_time(row["expires_at"])
            if "expires_at" in keys and row["expires_at"] is not None
            else None
        )
        flight_generation = (
            int(row["flight_generation"])
            if "flight_generation" in keys and row["flight_generation"] is not None
            else 0
        )

        return CacheEntry(
            id=UUID(row["id"]),
            prompt=row["prompt"],
            prompt_embedding=_from_blob(row["prompt_embedding"]),
            response=row["response"],
            model=row["model"],
            tenant_id=tenant_id,
            scope=scope,
            resource_id=resource_id,
            resource_version=resource_version,
            pipeline_version=pipeline_version,
            expires_at=expires_at,
            created_at=_parse_time(row["created_at"]) or _now(),
            last_hit_at=_parse_time(row["last_hit_at"]),
            hit_count=int(row["hit_count"]),
            feedback_negative_count=int(row["feedback_negative_count"]),
            feedback_positive_count=int(row["feedback_positive_count"]),
            flight_generation=flight_generation,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # RBAC & Resource Management Methods
    # -------------------------------------------------------------------------

    def create_user(self, user_id: str, tenant_id: str) -> User:
        created_at = _now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO users (id, tenant_id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET tenant_id = excluded.tenant_id
                """,
                (user_id, tenant_id, _serialize_time(created_at)),
            )
            self._connection.commit()
        return User(id=user_id, tenant_id=tenant_id, created_at=created_at)

    def get_user(self, user_id: str) -> User | None:
        row = self._connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return User(
            id=row["id"],
            tenant_id=row["tenant_id"],
            created_at=_parse_time(row["created_at"]) or _now(),
        )

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def create_role(self, role_id: str, name: str) -> Role:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO roles (id, name) VALUES (?, ?)",
                (role_id, name),
            )
            self._connection.commit()
        return Role(id=role_id, name=name)

    def create_permission(self, permission_id: str, name: str) -> Permission:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO permissions (id, name) VALUES (?, ?)",
                (permission_id, name),
            )
            self._connection.commit()
        return Permission(id=permission_id, name=name)

    def assign_role_to_user(self, user_id: str, role_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                (user_id, role_id),
            )
            self._connection.commit()

    def remove_role_from_user(self, user_id: str, role_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                (user_id, role_id),
            )
            self._connection.commit()

    def grant_permission_to_role(self, role_id: str, permission_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (role_id, permission_id),
            )
            self._connection.commit()

    def revoke_permission_from_role(self, role_id: str, permission_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?",
                (role_id, permission_id),
            )
            self._connection.commit()

    def get_user_roles(self, user_id: str) -> list[str]:
        rows = self._connection.execute(
            "SELECT role_id FROM user_roles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [row["role_id"] for row in rows]

    def create_resource(
        self,
        resource_id: str,
        tenant_id: str,
        initial_version: int = 1,
    ) -> Resource:
        updated_at = _now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO resources (id, tenant_id, current_version, status, updated_at)
                VALUES (?, ?, ?, 'active', ?)
                ON CONFLICT(id) DO UPDATE SET
                    tenant_id = excluded.tenant_id,
                    current_version = excluded.current_version,
                    updated_at = excluded.updated_at
                """,
                (resource_id, tenant_id, initial_version, _serialize_time(updated_at)),
            )
            self._connection.commit()
        return Resource(
            id=resource_id,
            tenant_id=tenant_id,
            current_version=initial_version,
            status="active",
            updated_at=updated_at,
        )

    def get_resource(self, resource_id: str) -> Resource | None:
        row = self._connection.execute(
            "SELECT * FROM resources WHERE id = ?",
            (resource_id,),
        ).fetchone()
        if row is None:
            return None
        return Resource(
            id=row["id"],
            tenant_id=row["tenant_id"],
            current_version=int(row["current_version"]),
            status=row["status"],
            updated_at=_parse_time(row["updated_at"]) or _now(),
        )

    def grant_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO resource_access (resource_id, role_id, permission_id)
                VALUES (?, ?, ?)
                """,
                (resource_id, role_id, permission_id),
            )
            self._connection.commit()

    def revoke_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                DELETE FROM resource_access
                WHERE resource_id = ? AND role_id = ? AND permission_id = ?
                """,
                (resource_id, role_id, permission_id),
            )
            self._connection.commit()

    def increment_resource_version(self, resource_id: str) -> int:
        """Atomically increment a resource's version within a transaction."""
        with self._lock:
            row = self._connection.execute(
                "SELECT id, tenant_id, current_version FROM resources WHERE id = ?",
                (resource_id,),
            ).fetchone()
            if row is None:
                msg = f"Resource not found: {resource_id}"
                raise KeyError(msg)
            new_version = int(row["current_version"]) + 1
            now_iso = _serialize_time(_now())
            self._connection.execute(
                "UPDATE resources SET current_version = ?, updated_at = ? WHERE id = ?",
                (new_version, now_iso, resource_id),
            )
            self._connection.execute(
                """
                INSERT INTO outbox_events (
                    id, event_type, schema_version, aggregate_type, aggregate_id,
                    tenant_id, payload_json, status, created_at, retry_count, max_retries
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    EventType.RESOURCE_UPDATED.value,
                    1,
                    "resource",
                    resource_id,
                    row["tenant_id"],
                    json.dumps(
                        {
                            "resource_id": resource_id,
                            "tenant_id": row["tenant_id"],
                            "new_version": new_version,
                        }
                    ),
                    OutboxStatus.PENDING.value,
                    now_iso,
                    0,
                    5,
                ),
            )
            self._connection.commit()
            return new_version

    def check_user_permission(
        self,
        user_id: str,
        permission_name: str,
        resource_id: str | None = None,
    ) -> bool:
        """Check if user has a permission, failing closed if any context is missing."""
        user = self.get_user(user_id)
        if user is None:
            return False

        roles = self.get_user_roles(user_id)
        if not roles:
            return False

        # Admin role bypasses granular restrictions within their tenant
        if "admin" in roles:
            if resource_id is not None:
                resource = self.get_resource(resource_id)
                if resource is None or resource.tenant_id != user.tenant_id:
                    return False
            return True

        placeholders = ",".join("?" for _ in roles)

        # Check tenant-level role permissions
        query_role_perms = f"""
            SELECT COUNT(*) AS cnt
            FROM role_permissions
            WHERE role_id IN ({placeholders}) AND permission_id = ?
        """
        row = self._connection.execute(query_role_perms, (*roles, permission_name)).fetchone()
        has_role_permission = bool(row and row["cnt"] > 0)

        if resource_id is None:
            return has_role_permission

        # For resource access: resource must belong to user's tenant
        resource = self.get_resource(resource_id)
        if resource is None or resource.tenant_id != user.tenant_id:
            return False

        # If resource has specific ACL entries in resource_access, enforce them
        acl_count_row = self._connection.execute(
            "SELECT COUNT(*) AS cnt FROM resource_access WHERE resource_id = ?",
            (resource_id,),
        ).fetchone()
        has_acls = bool(acl_count_row and acl_count_row["cnt"] > 0)

        if has_acls:
            query_acl = f"""
                SELECT COUNT(*) AS cnt
                FROM resource_access
                WHERE resource_id = ? AND role_id IN ({placeholders}) AND permission_id = ?
            """
            acl_match = self._connection.execute(
                query_acl, (resource_id, *roles, permission_name)
            ).fetchone()
            return bool(acl_match and acl_match["cnt"] > 0)

        # If no explicit resource ACL exists, fallback to role permission for tenant resources
        return has_role_permission

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

    # === OUTBOX STORE ===

    def record_outbox_event(
        self,
        event: OutboxEvent,
        connection: Any = None,
    ) -> UUID:
        now = _now()
        payload_json = json.dumps(event.payload)
        event_id = str(event.id)
        evt_type = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        status_val = event.status.value if hasattr(event.status, "value") else str(event.status)

        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO outbox_events (
                    id, event_type, schema_version, aggregate_type, aggregate_id,
                    tenant_id, payload_json, status, correlation_id, causation_id,
                    created_at, retry_count, max_retries, next_attempt_at, last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    evt_type,
                    event.schema_version,
                    event.aggregate_type,
                    event.aggregate_id,
                    event.tenant_id,
                    payload_json,
                    status_val,
                    event.correlation_id,
                    event.causation_id,
                    _serialize_time(event.created_at or now),
                    event.retry_count,
                    event.max_retries,
                    _serialize_time(event.next_attempt_at),
                    event.last_error,
                ),
            )
            self._connection.commit()
        return event.id

    def claim_outbox_events(
        self,
        limit: int = 100,
        claim_owner: str = "default_relay",
        lease_seconds: float = 30.0,
    ) -> list[OutboxEvent]:
        """Atomically claim pending or expired-lease processing events."""
        now = _now()
        now_iso = _serialize_time(now)
        lease_delta = timedelta(seconds=lease_seconds)
        expires_at_iso = _serialize_time(now + lease_delta)

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM outbox_events
                WHERE (status = 'pending' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                   OR (status = 'processing' AND claim_expires_at < ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now_iso, now_iso, limit),
            ).fetchall()

            if not rows:
                return []

            claimed_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" for _ in claimed_ids)
            self._connection.execute(
                f"""
                UPDATE outbox_events
                SET status = 'processing',
                    claimed_at = ?,
                    claim_owner = ?,
                    claim_expires_at = ?,
                    retry_count = retry_count + 1
                WHERE id IN ({placeholders})
                """,
                (now_iso, claim_owner, expires_at_iso, *claimed_ids),
            )
            self._connection.commit()

            updated_rows = self._connection.execute(
                f"SELECT * FROM outbox_events WHERE id IN ({placeholders})",
                claimed_ids,
            ).fetchall()

        return [self._row_to_outbox_event(r) for r in updated_rows]

    def mark_outbox_published(self, event_ids: list[UUID]) -> None:
        if not event_ids:
            return
        now_iso = _serialize_time(_now())
        ids = [str(eid) for eid in event_ids]
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._connection.execute(
                f"""
                UPDATE outbox_events
                SET status = 'published',
                    published_at = ?
                WHERE id IN ({placeholders})
                """,
                (now_iso, *ids),
            )
            self._connection.commit()

    def mark_outbox_retry(
        self,
        event_id: UUID,
        error: str,
        next_attempt_at: datetime,
        dead_letter: bool = False,
    ) -> None:
        new_status = OutboxStatus.DEAD_LETTER.value if dead_letter else OutboxStatus.PENDING.value
        with self._lock:
            self._connection.execute(
                """
                UPDATE outbox_events
                SET status = ?,
                    last_error = ?,
                    next_attempt_at = ?,
                    claim_owner = NULL,
                    claim_expires_at = NULL
                WHERE id = ?
                """,
                (new_status, error, _serialize_time(next_attempt_at), str(event_id)),
            )
            self._connection.commit()

    def get_outbox_event(self, event_id: UUID) -> OutboxEvent | None:
        row = self._connection.execute(
            "SELECT * FROM outbox_events WHERE id = ?",
            (str(event_id),),
        ).fetchone()
        return self._row_to_outbox_event(row) if row is not None else None

    # === PROCESSED EVENT STORE (CONSUMER IDEMPOTENCY) ===

    def record_processed_event(self, event_id: UUID, consumer_name: str) -> bool:
        """Record event processing. Returns True if newly processed, False if already processed."""
        now_iso = _serialize_time(_now())
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO processed_events (event_id, consumer_name, processed_at)
                VALUES (?, ?, ?)
                """,
                (str(event_id), consumer_name, now_iso),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def is_event_processed(self, event_id: UUID, consumer_name: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM processed_events WHERE event_id = ? AND consumer_name = ? LIMIT 1",
            (str(event_id), consumer_name),
        ).fetchone()
        return row is not None

    def _row_to_outbox_event(self, row: sqlite3.Row) -> OutboxEvent:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        r_keys = set(row.keys())
        return OutboxEvent(
            id=UUID(row["id"]),
            event_type=EventType(row["event_type"]),
            schema_version=int(row["schema_version"]) if "schema_version" in r_keys else 1,
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            tenant_id=row["tenant_id"],
            payload=payload or {},
            status=OutboxStatus(row["status"]),
            correlation_id=row["correlation_id"] if "correlation_id" in r_keys else None,
            causation_id=row["causation_id"] if "causation_id" in r_keys else None,
            created_at=_parse_time(row["created_at"]) or _now(),
            claimed_at=_parse_time(row["claimed_at"]) if "claimed_at" in r_keys else None,
            claim_owner=row["claim_owner"] if "claim_owner" in r_keys else None,
            claim_expires_at=_parse_time(row["claim_expires_at"])
            if "claim_expires_at" in r_keys
            else None,
            published_at=_parse_time(row["published_at"]) if "published_at" in r_keys else None,
            retry_count=int(row["retry_count"]) if "retry_count" in r_keys else 0,
            max_retries=int(row["max_retries"]) if "max_retries" in r_keys else 5,
            next_attempt_at=_parse_time(row["next_attempt_at"])
            if "next_attempt_at" in r_keys
            else None,
            last_error=row["last_error"] if "last_error" in r_keys else None,
        )
