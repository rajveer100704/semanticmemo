from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Generator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# faiss-cpu and torch each bundle their own OpenMP runtime; on macOS loading
# both in one process trips "OMP: Error #15". This test-only workaround lets
# the suite exercise the faiss-backed index path. It is a no-op on Linux CI,
# and must be set before SemanticMemo (and therefore torch/faiss) is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from semanticmemo import CacheConfig, SemanticMemo  # noqa: E402
from semanticmemo.types import FloatVector  # noqa: E402


class ToyEmbeddingProvider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        match text:
            case "alpha":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "alpha duplicate":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "near alpha":
                return np.array([0.8, 0.2, 0, 0], dtype=np.float32)
            case "beta":
                return np.array([0, 1, 0, 0], dtype=np.float32)
            case _:
                return np.array([0, 0, 1, 0], dtype=np.float32)


@pytest.fixture
def cache_config(tmp_path: Path) -> CacheConfig:
    return CacheConfig(
        db_path=tmp_path / "semanticmemo.db",
        embedding_dim=ToyEmbeddingProvider.dim,
        cosine_threshold=0.95,
        candidate_k=3,
        max_entries=10,
        estimated_llm_cost_usd=Decimal("0.003"),
    )


@pytest.fixture
def cache(cache_config: CacheConfig) -> Generator[SemanticMemo]:
    instance = SemanticMemo(
        domain="test",
        config=cache_config,
        embedding_provider=ToyEmbeddingProvider(),
        use_faiss=False,
    )
    instance.bootstrap_default_user("default_user", "default", "analyst")
    yield instance
    instance.close()


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
        translated = sql.replace("%s", "?")
        translated = translated.replace("TIMESTAMPTZ", "TEXT")
        translated = translated.replace("JSONB", "TEXT")
        translated = translated.replace("::jsonb", "")
        translated = translated.replace("BYTEA", "BLOB")
        translated = translated.replace("NOW()", "CURRENT_TIMESTAMP")
        translated = translated.replace("FOR UPDATE SKIP LOCKED", "")
        translated = translated.replace("FOR UPDATE", "")
        translated = translated.replace(
            "ON CONFLICT (id) DO UPDATE SET", "ON CONFLICT(id) DO UPDATE SET"
        )

        out_params = list(params) if params is not None else []

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

        if "RETURNING" in translated:
            parts = translated.split("RETURNING")
            translated = parts[0].strip()

        return translated, out_params

    def execute(self, sql: str, params: Any = None) -> FakePostgresCursor:
        trans_sql, trans_params = self._translate_sql(sql, params)
        final_params = []
        for p in trans_params:
            if isinstance(p, datetime):
                final_params.append(p.isoformat())
            else:
                final_params.append(p)

        with self.lock:
            is_dml = (
                sql.strip().startswith("INSERT")
                or sql.strip().startswith("UPDATE")
                or "UPDATE" in sql
            )
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
                    if "outbox_events" in sql:
                        c2 = self.conn.cursor()
                        c2.execute("SELECT * FROM outbox_events WHERE status = 'processing'")
                        c_cols = [col[0] for col in c2.description]
                        self._last_rows = [
                            dict(zip(c_cols, r, strict=False)) for r in c2.fetchall()
                        ]
                        c2.close()
                    else:
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
