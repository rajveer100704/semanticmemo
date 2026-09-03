"""Composable store protocols for identity, RBAC, resources, cache persistence, and feedback."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Protocol, TypeVar
from uuid import UUID

from semanticmemo.models import (
    CacheEntry,
    CacheScope,
    EvictionPolicy,
    FeedbackEvent,
    LookupRecord,
    Resource,
    User,
)
from semanticmemo.types import FloatVector

T = TypeVar("T")


class IdentityStore(Protocol):
    """Protocol for user and tenant identity management."""

    def create_user(self, user_id: str, tenant_id: str) -> User: ...

    def get_user(self, user_id: str) -> User | None: ...

    def delete_user(self, user_id: str) -> bool: ...


class AuthorizationStore(Protocol):
    """Protocol for RBAC roles and permissions management."""

    def assign_role_to_user(self, user_id: str, role_id: str) -> None: ...

    def remove_role_from_user(self, user_id: str, role_id: str) -> Any: ...

    def get_user_roles(self, user_id: str) -> list[Any]: ...

    def check_user_permission(
        self,
        user_id: str,
        permission_name: str,
        resource_id: str | None = None,
    ) -> bool: ...


class ResourceStore(Protocol):
    """Protocol for resource management and access control."""

    def create_resource(
        self, resource_id: str, tenant_id: str, initial_version: int = 1
    ) -> Resource: ...

    def get_resource(self, resource_id: str, tenant_id: str | None = None) -> Resource | None: ...

    def increment_resource_version(
        self, resource_id: str, tenant_id: str | None = None
    ) -> int | None: ...

    def grant_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None: ...

    def revoke_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> Any: ...


class CacheStore(Protocol):
    """Protocol for cache entries, eviction, and direct prompt lookups."""

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
    ) -> UUID | None: ...

    def get(self, entry_id: UUID) -> CacheEntry | None: ...

    def find_by_prompt(
        self,
        *,
        prompt: str,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> CacheEntry | None: ...

    def delete(self, entry_id: UUID) -> bool: ...

    def all_entries(self) -> Iterator[CacheEntry]: ...

    def count(self) -> int: ...

    def total_hit_count(self) -> int: ...

    def update_hit(self, entry_id: UUID) -> None: ...

    def increment_bad_feedback(self, entry_id: UUID) -> None: ...

    def increment_good_feedback(self, entry_id: UUID) -> None: ...

    def evict(
        self,
        *,
        max_entries: int,
        policy: EvictionPolicy,
        now: datetime | None = None,
        ttl_seconds: int | None = None,
    ) -> Any: ...


class FeedbackStore(Protocol):
    """Protocol for lookup records, explicit/implicit feedback, and active learning provenance."""

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
    ) -> None: ...

    def get_lookup(self, query_id: UUID) -> LookupRecord | None: ...

    def record_feedback(
        self,
        *,
        query_id: UUID,
        label: int,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID | None: ...

    def get_feedback(self, feedback_id: UUID) -> FeedbackEvent | None: ...

    def find_implicit_bad_hit(
        self,
        *,
        domain: str,
        prompt: str,
        within_seconds: float,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        pipeline_version: str | None = None,
    ) -> LookupRecord | None: ...

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
    ) -> UUID: ...

    def get_active_learning_pairs(
        self,
        *,
        domain: str | None = None,
        limit: int = 1000,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        pipeline_version: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_active_learning_pair_counts(self) -> dict[str, int]: ...

    def export_feedback_pairs(self, path: str, *, split: str = "train") -> int: ...

    def export_active_learning_pairs(self, path: str, *, split: str = "train") -> int: ...


class OutboxStore(Protocol):
    """Protocol for transactional outbox persistence and atomic claim leases."""

    def record_outbox_event(
        self,
        event: Any,
        connection: Any = None,
    ) -> UUID: ...

    def claim_outbox_events(
        self,
        limit: int = 100,
        claim_owner: str = "default_relay",
        lease_seconds: float = 30.0,
    ) -> list[Any]: ...

    def mark_outbox_published(
        self,
        event_ids: list[UUID],
    ) -> None: ...

    def mark_outbox_retry(
        self,
        event_id: UUID,
        error: str,
        next_attempt_at: datetime,
        dead_letter: bool = False,
    ) -> None: ...

    def get_outbox_event(
        self,
        event_id: UUID,
    ) -> Any | None: ...


class ProcessedEventStore(Protocol):
    """Protocol for consumer idempotency and durable deduplication."""

    def record_processed_event(
        self,
        event_id: UUID,
        consumer_name: str,
    ) -> bool:
        """Atomically record processed event.

        Returns True if newly recorded, False if already processed.
        """
        ...

    def is_event_processed(
        self,
        event_id: UUID,
        consumer_name: str,
    ) -> bool: ...


class MetadataStore(
    IdentityStore,
    AuthorizationStore,
    ResourceStore,
    CacheStore,
    FeedbackStore,
    OutboxStore,
    ProcessedEventStore,
    Protocol,
):
    """Composed transactional metadata store protocol."""

    def close(self) -> None: ...

    def lookup_count(self) -> int: ...

    def feedback_count(self) -> int: ...

    def feedback_events(self) -> Iterator[FeedbackEvent]: ...
