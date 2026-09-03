"""Event and Outbox Models for SemanticMemo V2 Distributed Plane."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(tz=UTC)


class EventType(StrEnum):
    """Supported domain event types in SemanticMemo."""

    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"
    PERMISSION_CHANGED = "permission.changed"
    USER_ROLE_CHANGED = "user.role.changed"
    PIPELINE_CHANGED = "pipeline.changed"


class OutboxStatus(StrEnum):
    """Lifecycle status for transactional outbox events."""

    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class OutboxEvent(BaseModel):
    """Durable domain event representation in transactional outbox."""

    id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    schema_version: int = 1
    aggregate_type: str
    aggregate_id: str
    tenant_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: OutboxStatus = OutboxStatus.PENDING
    correlation_id: str | None = None
    causation_id: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    created_at: datetime = Field(default_factory=_now)
    claimed_at: datetime | None = None
    claim_owner: str | None = None
    claim_expires_at: datetime | None = None
    published_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 5
    next_attempt_at: datetime | None = None
    last_error: str | None = None


class ProcessedEventRecord(BaseModel):
    """Durable record of processed event for consumer idempotency."""

    event_id: UUID
    consumer_name: str
    processed_at: datetime = Field(default_factory=_now)
