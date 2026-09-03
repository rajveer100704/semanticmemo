"""PostgreSQL-backed shared transactional metadata and RBAC plane."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from semanticmemo.events.models import (
    EventType,
    OutboxEvent,
    OutboxStatus,
)
from semanticmemo.exceptions import MissingDependencyError
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

logger = logging.getLogger(__name__)

try:
    import psycopg  # pyright: ignore[reportMissingImports]
    from psycopg.rows import dict_row  # pyright: ignore[reportMissingImports]

    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _to_blob(embedding: FloatVector | Sequence[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _from_blob(blob: bytes | memoryview) -> list[float]:
    b = bytes(blob)
    return np.frombuffer(b, dtype=np.float32).astype(np.float32).tolist()


def _serialize_time(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value))


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL metadata plane."""

    dsn: str = "postgresql://localhost:5432/semanticmemo"
    min_pool_size: int = 2
    max_pool_size: int = 10
    timeout_seconds: float = 5.0
    ssl_mode: str = "prefer"
    schema_name: str = "public"
    enable_rls: bool = False


class PostgresCacheStore:
    """PostgreSQL shared metadata, RBAC, and cache store implementation.

    Guarantees:
    - Authoritative relational truth for Identity, RBAC, Resources, and Cache Metadata.
    - Transactional atomicity on resource mutations and monotonic version increments.
    - Fail-closed error handling: Outages or constraint violations reject unauthorized operations.
    - Optional defense-in-depth Row-Level Security (RLS) enforcement.
    """

    def __init__(
        self,
        config: PostgresConfig | None = None,
        *,
        connection: Any | None = None,
    ) -> None:
        self.config = config or PostgresConfig()
        self._external_conn = connection is not None
        self._connection: Any = None

        if connection is not None:
            self._connection = connection
        else:
            if not PSYCOPG_AVAILABLE or psycopg is None:
                raise MissingDependencyError(
                    "psycopg is required for PostgresCacheStore. "
                    "Install with `pip install semanticmemo[postgres]` or `pip install psycopg`."
                )
            self._connection = psycopg.connect(
                self.config.dsn,
                row_factory=dict_row,  # pyright: ignore[reportArgumentType]
                autocommit=True,
            )

        self._ensure_schema()
        self._seed_default_rbac()

    def _ensure_schema(self) -> None:
        """Apply PostgreSQL schema DDL if tables do not exist."""
        try:
            schema_ddl = (
                resources.files("semanticmemo.store").joinpath("postgres_schema.sql").read_text()
            )
            with self._connection.cursor() as cursor:
                cursor.execute(schema_ddl)
        except Exception as exc:
            logger.debug("Schema DDL initialization step: %s", exc)

    def _seed_default_rbac(self) -> None:
        """Seed default roles and permissions."""
        default_roles = ["admin", "analyst", "viewer"]
        default_perms = [
            PermissionName.CACHE_READ.value,
            PermissionName.CACHE_WRITE.value,
            PermissionName.CACHE_INVALIDATE.value,
            PermissionName.RESOURCE_READ.value,
            PermissionName.RESOURCE_WRITE.value,
            PermissionName.RESOURCE_MANAGE.value,
        ]

        with self._connection.cursor() as cursor:
            for role in default_roles:
                cursor.execute(
                    "INSERT INTO roles (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (role, role),
                )
            for perm in default_perms:
                cursor.execute(
                    "INSERT INTO permissions (id, name) VALUES (%s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
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
                    cursor.execute(
                        "INSERT INTO role_permissions (role_id, permission_id) "
                        "VALUES (%s, %s) ON CONFLICT (role_id, permission_id) DO NOTHING",
                        (role, perm),
                    )

    def close(self) -> None:
        """Close PostgreSQL connection."""
        if not self._external_conn and hasattr(self._connection, "close"):
            self._connection.close()

    # === IDENTITY STORE ===

    def create_user(self, user_id: str, tenant_id: str) -> User:
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (id, tenant_id, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET tenant_id = EXCLUDED.tenant_id
                RETURNING id, tenant_id, created_at
                """,
                (user_id, tenant_id, now),
            )
            row = cursor.fetchone()
        return User(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            created_at=_parse_time(row["created_at"]) or now,
        )

    def get_user(self, user_id: str) -> User | None:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return User(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            created_at=_parse_time(row["created_at"]) or _now(),
        )

    def delete_user(self, user_id: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return cursor.rowcount > 0

    # === AUTHORIZATION STORE ===

    def assign_role_to_user(self, user_id: str, role_id: str) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s) "
                "ON CONFLICT (user_id, role_id) DO NOTHING",
                (user_id, role_id),
            )

    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM user_roles WHERE user_id = %s AND role_id = %s",
                (user_id, role_id),
            )
            return cursor.rowcount > 0

    def get_user_roles(self, user_id: str) -> list[Role]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.id, r.name
                FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = %s
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
        return [Role(id=str(r["id"]), name=str(r["name"])) for r in rows]

    def get_role_permissions(self, role_id: str) -> list[Permission]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.name
                FROM permissions p
                JOIN role_permissions rp ON p.id = rp.permission_id
                WHERE rp.role_id = %s
                """,
                (role_id,),
            )
            rows = cursor.fetchall()
        return [Permission(id=str(r["id"]), name=str(r["name"])) for r in rows]

    def has_permission(self, user_id: str, permission_name: PermissionName | str) -> bool:
        perm_str = (
            permission_name.value
            if isinstance(permission_name, PermissionName)
            else str(permission_name)
        )
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM role_permissions rp
                JOIN user_roles ur ON rp.role_id = ur.role_id
                WHERE ur.user_id = %s AND rp.permission_id = %s
                LIMIT 1
                """,
                (user_id, perm_str),
            )
            return cursor.fetchone() is not None

    def get_user_permissions(self, user_id: str) -> set[str]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT rp.permission_id
                FROM role_permissions rp
                JOIN user_roles ur ON rp.role_id = ur.role_id
                WHERE ur.user_id = %s
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
        return {str(r["permission_id"]) for r in rows}

    def check_user_permission(
        self,
        user_id: str,
        permission_name: str,
        resource_id: str | None = None,
    ) -> bool:
        """Check if user has permission, failing closed on missing context or DB error."""
        try:
            user = self.get_user(user_id)
            if user is None:
                return False

            roles = [r.name for r in self.get_user_roles(user_id)]
            if not roles:
                return False

            # Admin role bypasses granular restrictions within their tenant
            if "admin" in roles:
                if resource_id is not None:
                    resource = self.get_resource(resource_id)
                    if resource is None or resource.tenant_id != user.tenant_id:
                        return False
                return True

            # Check tenant-level role permissions
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM role_permissions
                    WHERE role_id = ANY(%s) AND permission_id = %s
                    """,
                    (roles, permission_name),
                )
                row = cursor.fetchone()
            has_role_permission = bool(row and row["cnt"] > 0)

            if resource_id is None:
                return has_role_permission

            # For resource access: resource must belong to user's tenant
            resource = self.get_resource(resource_id)
            if resource is None or resource.tenant_id != user.tenant_id:
                return False

            # Check explicit resource ACLs
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM resource_access WHERE resource_id = %s",
                    (resource_id,),
                )
                acl_count_row = cursor.fetchone()
                has_acls = bool(acl_count_row and acl_count_row["cnt"] > 0)

                if has_acls:
                    cursor.execute(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM resource_access
                        WHERE resource_id = %s AND role_id = ANY(%s) AND permission_id = %s
                        """,
                        (resource_id, roles, permission_name),
                    )
                    acl_match = cursor.fetchone()
                    return bool(acl_match and acl_match["cnt"] > 0)

            return has_role_permission
        except Exception as exc:
            logger.error("Error evaluating PostgreSQL RBAC permission: %s (failing closed)", exc)
            return False

    # === RESOURCE STORE ===

    def create_resource(self, resource_id: str, tenant_id: str) -> Resource:
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO resources (id, tenant_id, current_version, status, updated_at)
                VALUES (%s, %s, 1, 'active', %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at
                RETURNING id, tenant_id, current_version, status, updated_at
                """,
                (resource_id, tenant_id, now),
            )
            row = cursor.fetchone()
        return Resource(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            current_version=int(row["current_version"]),
            status=str(row["status"]),
            updated_at=_parse_time(row["updated_at"]) or now,
        )

    def get_resource(self, resource_id: str, tenant_id: str | None = None) -> Resource | None:
        with self._connection.cursor() as cursor:
            if tenant_id is not None:
                cursor.execute(
                    "SELECT * FROM resources WHERE id = %s AND tenant_id = %s",
                    (resource_id, tenant_id),
                )
            else:
                cursor.execute("SELECT * FROM resources WHERE id = %s", (resource_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return Resource(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            current_version=int(row["current_version"]),
            status=str(row["status"]),
            updated_at=_parse_time(row["updated_at"]) or _now(),
        )

    def update_resource(
        self,
        resource_id: str,
        tenant_id: str,
        *,
        status: str | None = None,
        increment_version: bool = True,
    ) -> Resource | None:
        """Atomic resource mutation with monotonic version increment."""
        now = _now()
        with self._connection.cursor() as cursor:
            if increment_version and status is not None:
                cursor.execute(
                    """
                    UPDATE resources
                    SET current_version = current_version + 1,
                        status = %s,
                        updated_at = %s
                    WHERE id = %s AND tenant_id = %s
                    RETURNING id, tenant_id, current_version, status, updated_at
                    """,
                    (status, now, resource_id, tenant_id),
                )
            elif increment_version:
                cursor.execute(
                    """
                    UPDATE resources
                    SET current_version = current_version + 1,
                        updated_at = %s
                    WHERE id = %s AND tenant_id = %s
                    RETURNING id, tenant_id, current_version, status, updated_at
                    """,
                    (now, resource_id, tenant_id),
                )
            elif status is not None:
                cursor.execute(
                    """
                    UPDATE resources
                    SET status = %s,
                        updated_at = %s
                    WHERE id = %s AND tenant_id = %s
                    RETURNING id, tenant_id, current_version, status, updated_at
                    """,
                    (status, now, resource_id, tenant_id),
                )
            else:
                return self.get_resource(resource_id, tenant_id)

            row = cursor.fetchone()
            if row is not None and increment_version:
                cursor.execute(
                    """
                    INSERT INTO outbox_events (
                        id, event_type, schema_version, aggregate_type, aggregate_id,
                        tenant_id, payload_json, status, created_at, retry_count, max_retries
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid4()),
                        EventType.RESOURCE_UPDATED.value,
                        1,
                        "resource",
                        resource_id,
                        tenant_id,
                        json.dumps(
                            {
                                "resource_id": resource_id,
                                "tenant_id": tenant_id,
                                "new_version": int(row["current_version"]),
                            }
                        ),
                        OutboxStatus.PENDING.value,
                        now,
                        0,
                        5,
                    ),
                )
        if row is None:
            return None
        return Resource(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            current_version=int(row["current_version"]),
            status=str(row["status"]),
            updated_at=_parse_time(row["updated_at"]) or now,
        )

    def increment_resource_version(self, resource_id: str, tenant_id: str) -> int | None:
        updated = self.update_resource(resource_id, tenant_id, increment_version=True)
        return updated.current_version if updated else None

    def grant_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO resource_access (resource_id, role_id, permission_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (resource_id, role_id, permission_id) DO NOTHING
                """,
                (resource_id, role_id, permission_id),
            )

    def revoke_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM resource_access
                WHERE resource_id = %s AND role_id = %s AND permission_id = %s
                """,
                (resource_id, role_id, permission_id),
            )
            return cursor.rowcount > 0

    def get_resource_access_roles(
        self, resource_id: str, permission_name: PermissionName | str
    ) -> list[str]:
        perm_str = (
            permission_name.value
            if isinstance(permission_name, PermissionName)
            else str(permission_name)
        )
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role_id FROM resource_access
                WHERE resource_id = %s AND permission_id = %s
                """,
                (resource_id, perm_str),
            )
            rows = cursor.fetchall()
        return [str(r["role_id"]) for r in rows]

    def has_resource_access(
        self, user_id: str, resource_id: str, permission_name: PermissionName | str
    ) -> bool:
        perm_str = (
            permission_name.value
            if isinstance(permission_name, PermissionName)
            else str(permission_name)
        )
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM resource_access ra
                JOIN user_roles ur ON ra.role_id = ur.role_id
                WHERE ur.user_id = %s AND ra.resource_id = %s AND ra.permission_id = %s
                LIMIT 1
                """,
                (user_id, resource_id, perm_str),
            )
            return cursor.fetchone() is not None

    def delete_resource(self, resource_id: str, tenant_id: str | None = None) -> bool:
        with self._connection.cursor() as cursor:
            if tenant_id is not None:
                cursor.execute(
                    "DELETE FROM resources WHERE id = %s AND tenant_id = %s",
                    (resource_id, tenant_id),
                )
            else:
                cursor.execute("DELETE FROM resources WHERE id = %s", (resource_id,))
            return cursor.rowcount > 0

    # === CACHE STORE ===

    def add(
        self,
        *,
        prompt: str,
        embedding: FloatVector | Sequence[float],
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
        now = _now()
        scope_str = scope.value if isinstance(scope, CacheScope) else str(scope)
        with self._connection.cursor() as cursor:
            if flight_generation > 0:
                clauses = [
                    "prompt = %s",
                    "tenant_id = %s",
                    "scope = %s",
                    "pipeline_version = %s",
                ]
                params: list[Any] = [prompt, tenant_id, scope_str, pipeline_version]

                if resource_id is None:
                    clauses.append("resource_id IS NULL")
                else:
                    clauses.append("resource_id = %s")
                    params.append(resource_id)

                if resource_version is None:
                    clauses.append("resource_version IS NULL")
                else:
                    clauses.append("resource_version = %s")
                    params.append(resource_version)

                where_sql = " AND ".join(clauses)
                sql = f"""
                    SELECT id, flight_generation FROM cache_entries
                    WHERE {where_sql}
                    ORDER BY flight_generation DESC, created_at DESC
                    LIMIT 1
                """
                cursor.execute(sql, params)
                existing = cursor.fetchone()
                if existing is not None and existing["flight_generation"] > flight_generation:
                    logger.warning(
                        "Postgres CAS write rejected: gen %d > candidate %d",
                        existing["flight_generation"],
                        flight_generation,
                    )
                    return None

            cursor.execute(
                """
                INSERT INTO cache_entries (
                    id, prompt, prompt_embedding, response, model, tenant_id, scope,
                    resource_id, resource_version, pipeline_version, expires_at, created_at,
                    last_hit_at, hit_count, feedback_negative_count,
                    feedback_positive_count, flight_generation, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, 0, 0, 0, %s, %s)
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
                    expires_at,
                    now,
                    flight_generation,
                    json.dumps(metadata or {}),
                ),
            )
        return entry_id

    def get(self, entry_id: UUID) -> CacheEntry | None:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM cache_entries WHERE id = %s", (str(entry_id),))
            row = cursor.fetchone()
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
        scope_str = scope.value if isinstance(scope, CacheScope) else str(scope)
        now = _now()
        clauses = [
            "prompt = %s",
            "tenant_id = %s",
            "scope = %s",
            "pipeline_version = %s",
            "(expires_at IS NULL OR expires_at > %s)",
        ]
        params: list[Any] = [prompt, tenant_id, scope_str, pipeline_version, now]

        if resource_id is None:
            clauses.append("resource_id IS NULL")
        else:
            clauses.append("resource_id = %s")
            params.append(resource_id)

        if resource_version is None:
            clauses.append("resource_version IS NULL")
        else:
            clauses.append("resource_version = %s")
            params.append(resource_version)

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM cache_entries
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT 1
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
        return self._row_to_entry(row) if row is not None else None

    def delete(self, entry_id: UUID) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM cache_entries WHERE id = %s", (str(entry_id),))
            return cursor.rowcount > 0

    def all_entries(self) -> Iterator[CacheEntry]:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM cache_entries ORDER BY created_at ASC")
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                if entry is not None:
                    yield entry

    def count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM cache_entries")
            row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def total_hit_count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COALESCE(SUM(hit_count), 0) AS hits FROM cache_entries")
            row = cursor.fetchone()
        return int(row["hits"]) if row else 0

    def update_hit(self, entry_id: UUID) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cache_entries
                SET hit_count = hit_count + 1, last_hit_at = %s
                WHERE id = %s
                """,
                (_now(), str(entry_id)),
            )

    def increment_bad_feedback(self, entry_id: UUID) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cache_entries
                SET feedback_negative_count = feedback_negative_count + 1
                WHERE id = %s
                """,
                (str(entry_id),),
            )

    def increment_good_feedback(self, entry_id: UUID) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cache_entries
                SET feedback_positive_count = feedback_positive_count + 1
                WHERE id = %s
                """,
                (str(entry_id),),
            )

    def evict(
        self,
        *,
        max_entries: int,
        policy: EvictionPolicy,
        now: datetime | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        current_time = now or _now()
        evicted = 0

        # 1. Evict expired entries
        with self._connection.cursor() as cursor:
            if ttl_seconds is not None:
                cutoff = current_time - timedelta(seconds=ttl_seconds)
                cursor.execute(
                    "DELETE FROM cache_entries WHERE created_at <= %s",
                    (cutoff,),
                )
                evicted += cursor.rowcount
            else:
                cursor.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= %s",
                    (current_time,),
                )
                evicted += cursor.rowcount

        current_count = self.count()
        if current_count <= max_entries:
            return evicted

        excess = current_count - max_entries
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM cache_entries
                WHERE id IN (
                    SELECT id FROM cache_entries
                    ORDER BY COALESCE(last_hit_at, created_at) ASC
                    LIMIT %s
                )
                """,
                (excess,),
            )
            evicted += cursor.rowcount
        return evicted

    # === FEEDBACK STORE ===

    def record_lookup(
        self,
        *,
        query_id: UUID,
        domain: str,
        prompt: str,
        embedding: FloatVector,
        cache_entry_id: UUID,
        similarity_score: float | None = None,
        classifier_score: float | None = None,
        cross_encoder_score: float | None = None,
    ) -> None:
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO lookup_records (
                    query_id, domain, query_prompt, query_embedding, cache_entry_id,
                    similarity_score, classifier_score, cross_encoder_score, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    now,
                ),
            )

    def get_lookup(self, query_id: UUID) -> LookupRecord | None:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM lookup_records WHERE query_id = %s", (str(query_id),))
            row = cursor.fetchone()
        if row is None:
            return None
        return LookupRecord(
            id=UUID(str(row["query_id"])),
            domain=str(row["domain"]),
            prompt=str(row["query_prompt"]),
            prompt_embedding=_from_blob(row["query_embedding"]),
            cache_entry_id=UUID(str(row["cache_entry_id"])),
            created_at=_parse_time(row["created_at"]) or _now(),
            similarity_score=float(row["similarity_score"])
            if row["similarity_score"] is not None
            else None,
            classifier_score=float(row["classifier_score"])
            if row["classifier_score"] is not None
            else None,
            cross_encoder_score=float(row["cross_encoder_score"])
            if row["cross_encoder_score"] is not None
            else None,
        )

    def record_feedback(
        self,
        *,
        query_id: UUID,
        label: int,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID | None:
        lookup = self.get_lookup(query_id)
        if lookup is None:
            return None
        event_id = uuid4()
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO feedback_events (
                    id, query_id, cache_entry_id, label, reason, created_at, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(event_id),
                    str(query_id),
                    str(lookup.cache_entry_id),
                    label,
                    reason,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
        return event_id

    def get_feedback(self, feedback_id: UUID) -> FeedbackEvent | None:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM feedback_events WHERE id = %s", (str(feedback_id),))
            row = cursor.fetchone()
        if row is None:
            return None
        return FeedbackEvent(
            id=UUID(str(row["id"])),
            query_id=UUID(str(row["query_id"])),
            cache_entry_id=UUID(str(row["cache_entry_id"])),
            label=int(row["label"]),
            reason=str(row["reason"]) if row["reason"] is not None else None,
            created_at=_parse_time(row["created_at"]) or _now(),
            metadata=json.loads(row["metadata_json"])
            if isinstance(row["metadata_json"], str)
            else (row["metadata_json"] or {}),
        )

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
        cutoff = _now() - timedelta(seconds=within_seconds)
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT lr.*
                FROM lookup_records lr
                JOIN cache_entries ce ON lr.cache_entry_id = ce.id
                LEFT JOIN feedback_events fe ON lr.query_id = fe.query_id
                WHERE lr.domain = %s
                  AND lr.query_prompt = %s
                  AND lr.created_at >= %s
                  AND fe.id IS NULL
                  AND (ce.tenant_id = %s OR %s IS NULL)
                  AND (ce.resource_id = %s OR %s IS NULL)
                  AND (ce.pipeline_version = %s OR %s IS NULL)
                ORDER BY lr.created_at DESC
                LIMIT 1
                """,
                (
                    domain,
                    prompt,
                    cutoff,
                    tenant_id,
                    tenant_id,
                    resource_id,
                    resource_id,
                    pipeline_version,
                    pipeline_version,
                ),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return LookupRecord(
            id=UUID(str(row["query_id"])),
            domain=str(row["domain"]),
            prompt=str(row["query_prompt"]),
            prompt_embedding=_from_blob(row["query_embedding"]),
            cache_entry_id=UUID(str(row["cache_entry_id"])),
            created_at=_parse_time(row["created_at"]) or _now(),
            similarity_score=float(row["similarity_score"])
            if row["similarity_score"] is not None
            else None,
            classifier_score=float(row["classifier_score"])
            if row["classifier_score"] is not None
            else None,
            cross_encoder_score=float(row["cross_encoder_score"])
            if row["cross_encoder_score"] is not None
            else None,
        )

    def record_active_learning_pair(
        self,
        *,
        domain: str,
        query_prompt: str,
        cached_prompt: str,
        similarity_score: float,
        classifier_score: float,
        cross_encoder_score: float | None = None,
        label: int,
        source: str,
        tenant_id: str = "default",
        resource_id: str | None = None,
        pipeline_version: str = "v2",
    ) -> UUID:
        event_id = uuid4()
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO active_learning_pairs (
                    id, domain, query_prompt, cached_prompt, similarity_score,
                    classifier_score, cross_encoder_score, label, source,
                    tenant_id, resource_id, pipeline_version, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(event_id),
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
                    now,
                ),
            )
        return event_id

    def get_active_learning_pairs(
        self,
        *,
        domain: str | None = None,
        limit: int = 1000,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        pipeline_version: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM active_learning_pairs
                WHERE (%s IS NULL OR domain = %s)
                  AND (%s IS NULL OR tenant_id = %s)
                  AND (%s IS NULL OR resource_id = %s)
                  AND (%s IS NULL OR pipeline_version = %s)
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (
                    domain,
                    domain,
                    tenant_id,
                    tenant_id,
                    resource_id,
                    resource_id,
                    pipeline_version,
                    pipeline_version,
                    limit,
                ),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": str(r["id"]),
                "domain": str(r["domain"]),
                "query_prompt": str(r["query_prompt"]),
                "cached_prompt": str(r["cached_prompt"]),
                "similarity_score": float(r["similarity_score"]),
                "classifier_score": float(r["classifier_score"]),
                "cross_encoder_score": float(r["cross_encoder_score"])
                if r["cross_encoder_score"] is not None
                else None,
                "label": int(r["label"]),
                "source": str(r["source"]),
                "tenant_id": str(r["tenant_id"]),
                "resource_id": str(r["resource_id"]) if r["resource_id"] is not None else None,
                "pipeline_version": str(r["pipeline_version"]),
                "created_at": _serialize_time(_parse_time(r["created_at"])),
            }
            for r in rows
        ]

    def get_active_learning_pair_counts(self) -> dict[str, int]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT label, COUNT(*) AS count FROM active_learning_pairs GROUP BY label"
            )
            rows = cursor.fetchall()
        return {str(r["label"]): int(r["count"]) for r in rows}

    def _row_to_entry(self, row: dict[str, Any] | None) -> CacheEntry | None:
        if row is None:
            return None
        meta = row.get("metadata_json")
        if isinstance(meta, str):
            meta = json.loads(meta)
        elif meta is None:
            meta = {}

        return CacheEntry(
            id=UUID(str(row["id"])),
            prompt=str(row["prompt"]),
            prompt_embedding=_from_blob(row["prompt_embedding"]),
            response=str(row["response"]),
            model=str(row["model"]) if row["model"] is not None else None,
            tenant_id=str(row["tenant_id"]),
            scope=CacheScope(str(row["scope"])),
            resource_id=str(row["resource_id"]) if row["resource_id"] is not None else None,
            resource_version=int(row["resource_version"])
            if row["resource_version"] is not None
            else None,
            pipeline_version=str(row["pipeline_version"]),
            expires_at=_parse_time(row["expires_at"]),
            created_at=_parse_time(row["created_at"]) or _now(),
            last_hit_at=_parse_time(row["last_hit_at"]),
            hit_count=int(row["hit_count"]),
            feedback_negative_count=int(row["feedback_negative_count"]),
            feedback_positive_count=int(row["feedback_positive_count"]),
            flight_generation=int(row.get("flight_generation", 0)),
            metadata=meta,
        )

    def lookup_count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM lookup_records")
            row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def feedback_count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM feedback_events")
            row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def feedback_events(self) -> Iterator[FeedbackEvent]:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM feedback_events ORDER BY created_at ASC")
            rows = cursor.fetchall()
        for r in rows:
            yield FeedbackEvent(
                id=UUID(str(r["id"])),
                query_id=UUID(str(r["query_id"])),
                cache_entry_id=UUID(str(r["cache_entry_id"])),
                label=int(r["label"]),
                reason=str(r["reason"]) if r["reason"] is not None else None,
                created_at=_parse_time(r["created_at"]) or _now(),
                metadata=json.loads(r["metadata_json"])
                if isinstance(r["metadata_json"], str)
                else (r["metadata_json"] or {}),
            )

    def export_feedback_pairs(self, path: str, *, split: str = "train") -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT lr.query_prompt AS query_prompt,
                       ce.prompt AS cached_prompt,
                       fe.label AS label,
                       lr.domain AS domain
                FROM feedback_events fe
                JOIN lookup_records lr ON fe.query_id = lr.query_id
                JOIN cache_entries ce ON fe.cache_entry_id = ce.id
                """
            )
            rows = cursor.fetchall()

        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "query_prompt": r["query_prompt"],
                            "cached_prompt": r["cached_prompt"],
                            "label": int(r["label"]),
                            "domain": r["domain"],
                            "split": split,
                        }
                    )
                    + "\n"
                )
        return len(rows)

    def export_active_learning_pairs(self, path: str, *, split: str = "train") -> int:
        pairs = self.get_active_learning_pairs()
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                p_copy = dict(p)
                p_copy["split"] = split
                f.write(json.dumps(p_copy) + "\n")
        return len(pairs)

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

        conn = connection or self._connection
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO outbox_events (
                    id, event_type, schema_version, aggregate_type, aggregate_id,
                    tenant_id, payload_json, status, correlation_id, causation_id,
                    created_at, retry_count, max_retries, next_attempt_at, last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
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
                    event.created_at or now,
                    event.retry_count,
                    event.max_retries,
                    event.next_attempt_at,
                    event.last_error,
                ),
            )
        return event.id

    def claim_outbox_events(
        self,
        limit: int = 100,
        claim_owner: str = "default_relay",
        lease_seconds: float = 30.0,
    ) -> list[OutboxEvent]:
        """Atomically claim pending or expired-lease events with SKIP LOCKED."""
        now = _now()
        lease_delta = timedelta(seconds=lease_seconds)
        expires_at = now + lease_delta

        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                WITH claimable AS (
                    SELECT id FROM outbox_events
                    WHERE (
                        status = 'pending'
                        AND (next_attempt_at IS NULL OR next_attempt_at <= %s)
                    )
                    OR (status = 'processing' AND claim_expires_at < %s)
                    ORDER BY created_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE outbox_events
                SET status = 'processing',
                    claimed_at = %s,
                    claim_owner = %s,
                    claim_expires_at = %s,
                    retry_count = retry_count + 1
                WHERE id IN (SELECT id FROM claimable)
                RETURNING *
                """,
                (now, now, limit, now, claim_owner, expires_at),
            )
            rows = cursor.fetchall()

        events = []
        for r in rows:
            events.append(self._row_to_outbox_event(r))
        return events

    def mark_outbox_published(self, event_ids: list[UUID]) -> None:
        if not event_ids:
            return
        now = _now()
        ids = [str(eid) for eid in event_ids]
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox_events
                SET status = 'published',
                    published_at = %s
                WHERE id = ANY(%s)
                """,
                (now, ids),
            )

    def mark_outbox_retry(
        self,
        event_id: UUID,
        error: str,
        next_attempt_at: datetime,
        dead_letter: bool = False,
    ) -> None:
        new_status = OutboxStatus.DEAD_LETTER.value if dead_letter else OutboxStatus.PENDING.value
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox_events
                SET status = %s,
                    last_error = %s,
                    next_attempt_at = %s,
                    claim_owner = NULL,
                    claim_expires_at = NULL
                WHERE id = %s
                """,
                (new_status, error, next_attempt_at, str(event_id)),
            )

    def get_outbox_event(self, event_id: UUID) -> OutboxEvent | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM outbox_events WHERE id = %s",
                (str(event_id),),
            )
            row = cursor.fetchone()
        return self._row_to_outbox_event(row) if row else None

    # === PROCESSED EVENT STORE (CONSUMER IDEMPOTENCY) ===

    def record_processed_event(self, event_id: UUID, consumer_name: str) -> bool:
        """Record event processing. Returns True if newly processed, False if already processed."""
        now = _now()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO processed_events (event_id, consumer_name, processed_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (event_id, consumer_name) DO NOTHING
                """,
                (str(event_id), consumer_name, now),
            )
            return cursor.rowcount > 0

    def is_event_processed(self, event_id: UUID, consumer_name: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM processed_events WHERE event_id = %s AND consumer_name = %s LIMIT 1",
                (str(event_id), consumer_name),
            )
            return cursor.fetchone() is not None

    def _row_to_outbox_event(self, row: dict[str, Any]) -> OutboxEvent:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return OutboxEvent(
            id=UUID(str(row["id"])),
            event_type=EventType(str(row["event_type"])),
            schema_version=int(row.get("schema_version", 1)),
            aggregate_type=str(row["aggregate_type"]),
            aggregate_id=str(row["aggregate_id"]),
            tenant_id=str(row["tenant_id"]),
            payload=payload or {},
            status=OutboxStatus(str(row["status"])),
            correlation_id=str(row["correlation_id"])
            if row.get("correlation_id") is not None
            else None,
            causation_id=str(row["causation_id"]) if row.get("causation_id") is not None else None,
            created_at=_parse_time(row["created_at"]) or _now(),
            claimed_at=_parse_time(row.get("claimed_at")),
            claim_owner=str(row["claim_owner"]) if row.get("claim_owner") is not None else None,
            claim_expires_at=_parse_time(row.get("claim_expires_at")),
            published_at=_parse_time(row.get("published_at")),
            retry_count=int(row.get("retry_count", 0)),
            max_retries=int(row.get("max_retries", 5)),
            next_attempt_at=_parse_time(row.get("next_attempt_at")),
            last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
        )
