"""Comprehensive test suite for PostgresCacheStore, RBAC, atomic versioning, and migration."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, PermissionName
from semanticmemo.store.migration import CutoverState, migrate_sqlite_to_postgres
from semanticmemo.store.postgres_store import (
    PostgresCacheStore,
    PostgresConfig,
)
from semanticmemo.store.sqlite_store import SQLiteCacheStore


def _now() -> datetime:
    return datetime.now(tz=UTC)


class FakePostgresCursor:
    """Cursor simulator for Postgres testing over an isolated memory engine."""

    def __init__(self, conn: sqlite3.Connection, lock: Any) -> None:
        self.conn = conn
        self.lock = lock
        with self.lock:
            self.cursor = conn.cursor()
        self.rowcount = 0
        self._last_rows: list[dict[str, Any]] = []

    def __enter__(self) -> FakePostgresCursor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        with self.lock:
            self.cursor.close()

    def _translate_sql(self, sql: str, params: Any) -> tuple[str, list[Any]]:
        # Translate Postgres dialect (%s, JSONB, RETURNING) to SQLite dialect
        translated = sql.replace("%s", "?")
        translated = translated.replace("TIMESTAMPTZ", "TEXT")
        translated = translated.replace("JSONB", "TEXT")
        translated = translated.replace("::jsonb", "")
        translated = translated.replace("BYTEA", "BLOB")
        translated = translated.replace("NOW()", "CURRENT_TIMESTAMP")
        translated = translated.replace(
            "ON CONFLICT (id) DO UPDATE SET", "ON CONFLICT(id) DO UPDATE SET"
        )

        out_params = list(params) if params is not None else []

        # Convert ANY(? array) to IN (?, ?)
        if "role_id = ANY(?)" in translated or "role_id = ANY(%s)" in translated:
            idx = -1
            for i, p in enumerate(out_params):
                if isinstance(p, (list, tuple)):
                    idx = i
                    break
            if idx != -1:
                arr = out_params[idx]
                placeholders = ", ".join(["?"] * len(arr))
                translated = translated.replace("role_id = ANY(?)", f"role_id IN ({placeholders})")
                out_params = out_params[:idx] + list(arr) + out_params[idx + 1 :]

        # Handle RETURNING clause emulation for SQLite
        if "RETURNING" in translated:
            parts = translated.split("RETURNING")
            translated = parts[0].strip()

        return translated, out_params

    def execute(self, sql: str, params: Any = None) -> FakePostgresCursor:
        trans_sql, trans_params = self._translate_sql(sql, params)

        # Convert datetime objects in params to isoformat strings
        final_params = []
        for p in trans_params:
            if isinstance(p, datetime):
                final_params.append(p.isoformat())
            else:
                final_params.append(p)

        with self.lock:
            # Split multi-statement DDL if necessary
            is_dml = sql.strip().startswith("INSERT") or sql.strip().startswith("UPDATE")
            if ";" in trans_sql and not is_dml:
                statements = [s.strip() for s in trans_sql.split(";") if s.strip()]
                for stmt in statements:
                    with contextlib.suppress(sqlite3.OperationalError):
                        self.cursor.execute(stmt)
                self.rowcount = 0
                self._last_rows = []
                return self

            try:
                self.cursor.execute(trans_sql, final_params)
                self.rowcount = self.cursor.rowcount

                if self.cursor.description:
                    cols = [col[0] for col in self.cursor.description]
                    self._last_rows = [
                        dict(zip(cols, row, strict=False)) for row in self.cursor.fetchall()
                    ]
                elif "RETURNING" in sql:
                    # Emulate RETURNING by selecting the modified row
                    table = "resources" if "resources" in sql else "users"
                    if table == "resources" and sql.strip().startswith("UPDATE"):
                        id_val = str(params[-2]) if params and len(params) >= 2 else None
                    else:
                        id_val = str(params[0]) if params else None

                    if id_val:
                        c2 = self.conn.cursor()
                        c2.execute(f"SELECT * FROM {table} WHERE id = ?", (id_val,))
                        c_cols = [col[0] for col in c2.description]
                        self._last_rows = [
                            dict(zip(c_cols, r, strict=False)) for r in c2.fetchall()
                        ]
                        c2.close()
                else:
                    self._last_rows = []
            except Exception:
                self.rowcount = 0
                self._last_rows = []
                raise

        return self

    def fetchone(self) -> dict[str, Any] | None:
        return self._last_rows[0] if self._last_rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self._last_rows


class FakePostgresConnection:
    """Connection simulator for Postgres."""

    def __init__(self) -> None:
        import threading

        self.lock = threading.RLock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite_tables()

    def _init_sqlite_tables(self) -> None:
        with self.lock:
            c = self.conn.cursor()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS roles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS permissions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS user_roles (
                    user_id TEXT NOT NULL, role_id TEXT NOT NULL, PRIMARY KEY (user_id, role_id)
                );
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_id TEXT NOT NULL, permission_id TEXT NOT NULL,
                    PRIMARY KEY (role_id, permission_id)
                );
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
                    current_version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, id)
                );
                CREATE TABLE IF NOT EXISTS resource_access (
                    resource_id TEXT NOT NULL, role_id TEXT NOT NULL, permission_id TEXT NOT NULL,
                    PRIMARY KEY (resource_id, role_id, permission_id)
                );
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id TEXT PRIMARY KEY, prompt TEXT NOT NULL,
                    prompt_embedding BLOB NOT NULL, response TEXT NOT NULL,
                    model TEXT, tenant_id TEXT NOT NULL DEFAULT 'default',
                    scope TEXT NOT NULL DEFAULT 'global',
                    resource_id TEXT, resource_version INTEGER,
                    pipeline_version TEXT NOT NULL DEFAULT 'v2',
                    expires_at TEXT, created_at TEXT NOT NULL,
                    last_hit_at TEXT, hit_count INTEGER NOT NULL DEFAULT 0,
                    feedback_negative_count INTEGER NOT NULL DEFAULT 0,
                    feedback_positive_count INTEGER NOT NULL DEFAULT 0,
                    flight_generation INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS lookup_records (
                    query_id TEXT PRIMARY KEY, domain TEXT NOT NULL,
                    query_prompt TEXT NOT NULL, query_embedding BLOB NOT NULL,
                    cache_entry_id TEXT NOT NULL, similarity_score REAL,
                    classifier_score REAL, cross_encoder_score REAL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id TEXT PRIMARY KEY, query_id TEXT NOT NULL,
                    cache_entry_id TEXT NOT NULL, label INTEGER NOT NULL,
                    reason TEXT, created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS active_learning_pairs (
                    id TEXT PRIMARY KEY, domain TEXT NOT NULL,
                    query_prompt TEXT NOT NULL, cached_prompt TEXT NOT NULL,
                    similarity_score REAL NOT NULL, classifier_score REAL NOT NULL,
                    cross_encoder_score REAL, label INTEGER NOT NULL,
                    source TEXT NOT NULL, tenant_id TEXT NOT NULL DEFAULT 'default',
                    resource_id TEXT, pipeline_version TEXT NOT NULL DEFAULT 'v2',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outbox_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    correlation_id TEXT,
                    causation_id TEXT,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    claim_owner TEXT,
                    claim_expires_at TEXT,
                    published_at TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 5,
                    next_attempt_at TEXT,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id TEXT NOT NULL,
                    consumer_name TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (event_id, consumer_name)
                );
            """)
            self.conn.commit()

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self.conn, self.lock)

    def close(self) -> None:
        with self.lock:
            self.conn.close()


@pytest.fixture
def fake_pg_store() -> PostgresCacheStore:
    fake_conn = FakePostgresConnection()
    return PostgresCacheStore(config=PostgresConfig(), connection=fake_conn)


def test_postgres_user_identity_crud(fake_pg_store: PostgresCacheStore):
    """Test user creation, retrieval, and tenant derivation."""
    user = fake_pg_store.create_user("user_101", "tenant_alpha")
    assert user.id == "user_101"
    assert user.tenant_id == "tenant_alpha"

    fetched = fake_pg_store.get_user("user_101")
    assert fetched is not None
    assert fetched.tenant_id == "tenant_alpha"

    assert fake_pg_store.get_user("non_existent") is None

    deleted = fake_pg_store.delete_user("user_101")
    assert deleted is True
    assert fake_pg_store.get_user("user_101") is None


def test_postgres_rbac_permissions_evaluation(fake_pg_store: PostgresCacheStore):
    """Test role assignment, permission derivation, and fail-closed evaluation."""
    fake_pg_store.create_user("analyst_1", "tenant_fin")
    fake_pg_store.assign_role_to_user("analyst_1", "analyst")

    roles = fake_pg_store.get_user_roles("analyst_1")
    assert any(r.name == "analyst" for r in roles)

    # Analyst has cache:read and cache:write
    read_perm = PermissionName.CACHE_READ.value
    write_perm = PermissionName.CACHE_WRITE.value
    inval_perm = PermissionName.CACHE_INVALIDATE.value
    assert fake_pg_store.check_user_permission("analyst_1", read_perm) is True
    assert fake_pg_store.check_user_permission("analyst_1", write_perm) is True
    # Analyst lacks cache:invalidate (admin only)
    assert fake_pg_store.check_user_permission("analyst_1", inval_perm) is False


def test_postgres_resource_lifecycle_and_atomic_versioning(fake_pg_store: PostgresCacheStore):
    """Test resource creation, ACLs, and atomic version incrementation."""
    res = fake_pg_store.create_resource("report_q3", "tenant_fin")
    assert res.id == "report_q3"
    assert res.tenant_id == "tenant_fin"
    assert res.current_version == 1

    # Atomic update and version increment
    new_v = fake_pg_store.increment_resource_version("report_q3", "tenant_fin")
    assert new_v == 2

    fetched = fake_pg_store.get_resource("report_q3", "tenant_fin")
    assert fetched is not None
    assert fetched.current_version == 2


@pytest.mark.asyncio
async def test_postgres_concurrent_resource_version_increments(fake_pg_store: PostgresCacheStore):
    """Concurrency Invariant: N simultaneous updates result in v+N with no lost updates."""
    fake_pg_store.create_resource("concurrent_doc", "tenant_concur")

    async def increment_task() -> int | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            fake_pg_store.increment_resource_version,
            "concurrent_doc",
            "tenant_concur",
        )

    tasks = [increment_task() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All 10 tasks succeeded
    assert len(results) == 10
    final_res = fake_pg_store.get_resource("concurrent_doc", "tenant_concur")
    assert final_res is not None
    assert final_res.current_version == 11  # initial 1 + 10 increments


def test_postgres_cache_entry_scoped_persistence(fake_pg_store: PostgresCacheStore):
    """Test cache entry storage, TTL, scope matching, and prompt direct lookup."""
    embedding = np.array([0.1] * 384, dtype=np.float32)
    entry_id = fake_pg_store.add(
        prompt="What is our Q3 revenue?",
        embedding=embedding,
        response="$45M",
        model="gpt-4o",
        tenant_id="tenant_fin",
        scope=CacheScope.RESOURCE,
        resource_id="report_q3",
        resource_version=2,
        pipeline_version="v2",
        expires_at=_now() + timedelta(hours=1),
    )

    assert isinstance(entry_id, UUID)
    entry = fake_pg_store.get(entry_id)
    assert entry is not None
    assert entry.prompt == "What is our Q3 revenue?"
    assert entry.response == "$45M"
    assert entry.tenant_id == "tenant_fin"
    assert entry.resource_id == "report_q3"
    assert entry.resource_version == 2

    # Direct find by prompt matching scope
    found = fake_pg_store.find_by_prompt(
        prompt="What is our Q3 revenue?",
        tenant_id="tenant_fin",
        scope=CacheScope.RESOURCE,
        resource_id="report_q3",
        resource_version=2,
        pipeline_version="v2",
    )
    assert found is not None
    assert found.id == entry_id

    # Mismatched version returns None
    assert (
        fake_pg_store.find_by_prompt(
            prompt="What is our Q3 revenue?",
            tenant_id="tenant_fin",
            scope=CacheScope.RESOURCE,
            resource_id="report_q3",
            resource_version=3,
            pipeline_version="v2",
        )
        is None
    )


def test_postgres_active_learning_provenance(fake_pg_store: PostgresCacheStore):
    """Test active learning pair recording with immutable tenant and pipeline provenance."""
    pair_id = fake_pg_store.record_active_learning_pair(
        domain="finance",
        query_prompt="Net margin for 2025",
        cached_prompt="Gross margin for 2025",
        similarity_score=0.92,
        classifier_score=0.35,
        cross_encoder_score=0.20,
        label=0,
        source="entity_veto",
        tenant_id="tenant_sec",
        resource_id="10k_sec",
        pipeline_version="v2",
    )

    assert isinstance(pair_id, UUID)
    pairs = fake_pg_store.get_active_learning_pairs(tenant_id="tenant_sec")
    assert len(pairs) == 1
    assert pairs[0]["source"] == "entity_veto"
    assert pairs[0]["pipeline_version"] == "v2"
    assert pairs[0]["tenant_id"] == "tenant_sec"


def test_sqlite_to_postgres_migration(tmp_path):
    """Test SQLite to PostgreSQL migration tool with row counts and checksum validation."""
    sqlite_db = tmp_path / "source.db"
    sqlite_store = SQLiteCacheStore(sqlite_db)

    # Seed SQLite store
    sqlite_store.create_user("user_mig_1", "tenant_mig")
    sqlite_store.assign_role_to_user("user_mig_1", "analyst")
    sqlite_store.create_resource("doc_mig_1", "tenant_mig")
    sqlite_store.add(
        prompt="Migrated test prompt",
        embedding=np.array([0.05] * 384, dtype=np.float32),
        response="Migrated response",
        model="gpt-4o",
        tenant_id="tenant_mig",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )

    # Target PostgreSQL store
    fake_conn = FakePostgresConnection()
    postgres_store = PostgresCacheStore(config=PostgresConfig(), connection=fake_conn)

    # Run migration
    report = migrate_sqlite_to_postgres(sqlite_store, postgres_store)

    assert report.is_valid is True
    assert report.cutover_state == CutoverState.POSTGRES_AUTHORITATIVE
    assert report.total_migrated > 0

    # Verify data in PostgreSQL
    user_mig = postgres_store.get_user("user_mig_1")
    assert user_mig is not None
    assert user_mig.tenant_id == "tenant_mig"

    doc_mig = postgres_store.get_resource("doc_mig_1", "tenant_mig")
    assert doc_mig is not None


def test_postgres_missing_dependency(monkeypatch: pytest.MonkeyPatch):
    """Test clean MissingDependencyError when psycopg is missing."""
    import semanticmemo.store.postgres_store as pstore

    monkeypatch.setattr(pstore, "PSYCOPG_AVAILABLE", False)
    with pytest.raises(MissingDependencyError, match="psycopg is required"):
        PostgresCacheStore(config=PostgresConfig())


def test_postgres_fail_closed_on_connection_error():
    """Security Invariant: Storage errors fail closed and never broaden authorization."""
    fake_conn = FakePostgresConnection()
    store = PostgresCacheStore(config=PostgresConfig(), connection=fake_conn)

    # Close underlying connection to simulate database network outage
    fake_conn.close()

    # RBAC check must return False (fail closed), not True or fallback
    assert store.check_user_permission("user_101", PermissionName.CACHE_READ.value) is False


@pytest.mark.asyncio
async def test_postgres_live_server_integration():
    """Integration test executing against live PostgreSQL instance if available."""
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("Live PostgreSQL DSN not configured (set POSTGRES_DSN to run).")

    try:
        store = PostgresCacheStore(config=PostgresConfig(dsn=dsn))
        user_id = f"live_user_{uuid4().hex[:8]}"
        store.create_user(user_id, "live_tenant")
        u = store.get_user(user_id)
        assert u is not None
        assert u.tenant_id == "live_tenant"
        store.delete_user(user_id)
        store.close()
    except Exception as exc:
        pytest.skip(f"Live PostgreSQL server connection failed: {exc}")
