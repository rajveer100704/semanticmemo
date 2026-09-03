"""Persistence backends and metadata store protocols."""

from semanticmemo.store.migration import (
    CutoverState,
    MigrationReport,
    TableMigrationStats,
    migrate_sqlite_to_postgres,
)
from semanticmemo.store.postgres_store import (
    PSYCOPG_AVAILABLE,
    PostgresCacheStore,
    PostgresConfig,
)
from semanticmemo.store.protocol import (
    AuthorizationStore,
    CacheStore,
    FeedbackStore,
    IdentityStore,
    MetadataStore,
    ResourceStore,
)
from semanticmemo.store.sqlite_store import SQLiteCacheStore

__all__ = [
    "AuthorizationStore",
    "CacheStore",
    "CutoverState",
    "FeedbackStore",
    "IdentityStore",
    "MetadataStore",
    "MigrationReport",
    "PSYCOPG_AVAILABLE",
    "PostgresCacheStore",
    "PostgresConfig",
    "ResourceStore",
    "SQLiteCacheStore",
    "TableMigrationStats",
    "migrate_sqlite_to_postgres",
]
