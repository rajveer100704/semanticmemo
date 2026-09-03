"""Test suite for Transactional Outbox, Claim Leases, Idempotent Consumers, and Reconciler."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from conftest import FakePostgresConnection

from semanticmemo.events import (
    EventType,
    IndexReconciler,
    InMemoryEventBus,
    InvalidationConsumer,
    KafkaEventBus,
    OutboxEvent,
    OutboxRelay,
    OutboxStatus,
    RabbitMQEventBus,
)
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope
from semanticmemo.store.postgres_store import PostgresCacheStore, PostgresConfig
from semanticmemo.store.sqlite_store import SQLiteCacheStore


def _now() -> datetime:
    return datetime.now(tz=UTC)


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteCacheStore:
    db_path = tmp_path / "outbox_test.db"
    return SQLiteCacheStore(db_path)


@pytest.fixture
def fake_pg_store() -> PostgresCacheStore:
    fake_conn = FakePostgresConnection()
    return PostgresCacheStore(config=PostgresConfig(), connection=fake_conn)


# =============================================================================
# 1. Transactional Atomicity Tests
# =============================================================================


def test_sqlite_resource_increment_emits_transactional_outbox_event(
    sqlite_store: SQLiteCacheStore,
):
    """Proves resource version increment and outbox event are recorded in one atomic transaction."""
    sqlite_store.create_resource("res_fin_1", "tenant_fin", initial_version=1)
    new_v = sqlite_store.increment_resource_version("res_fin_1")
    assert new_v == 2

    # Check outbox event
    claimed = sqlite_store.claim_outbox_events(limit=10)
    assert len(claimed) == 1
    evt = claimed[0]
    assert evt.event_type == EventType.RESOURCE_UPDATED
    assert evt.aggregate_id == "res_fin_1"
    assert evt.tenant_id == "tenant_fin"
    assert evt.payload["new_version"] == 2
    assert evt.status == OutboxStatus.PROCESSING


def test_postgres_resource_increment_emits_transactional_outbox_event(
    fake_pg_store: PostgresCacheStore,
):
    """Proves PostgreSQL atomic version increment records outbox event."""
    fake_pg_store.create_resource("res_q3", "tenant_a")
    updated = fake_pg_store.update_resource("res_q3", "tenant_a", increment_version=True)
    assert updated is not None
    assert updated.current_version == 2

    claimed = fake_pg_store.claim_outbox_events(limit=10)
    assert len(claimed) == 1
    evt = claimed[0]
    assert evt.event_type == EventType.RESOURCE_UPDATED
    assert evt.aggregate_id == "res_q3"
    assert evt.payload["new_version"] == 2
    assert evt.status == OutboxStatus.PROCESSING


# =============================================================================
# 2. Claim Leases & Concurrency Tests
# =============================================================================


def test_outbox_atomic_claim_lease_prevents_duplicate_processing(
    sqlite_store: SQLiteCacheStore,
):
    """Proves two concurrent relay workers do not claim the same event."""
    event = OutboxEvent(
        event_type=EventType.PERMISSION_CHANGED,
        aggregate_type="role",
        aggregate_id="analyst",
        tenant_id="tenant_x",
        payload={"action": "granted", "permission": "cache:read"},
    )
    sqlite_store.record_outbox_event(event)

    # Worker 1 claims event
    batch_1 = sqlite_store.claim_outbox_events(limit=10, claim_owner="worker-1", lease_seconds=30.0)
    assert len(batch_1) == 1
    assert batch_1[0].claim_owner == "worker-1"

    # Worker 2 attempts to claim while lease is active
    batch_2 = sqlite_store.claim_outbox_events(limit=10, claim_owner="worker-2", lease_seconds=30.0)
    assert len(batch_2) == 0


def test_outbox_lease_expiry_allows_worker_reclaim(sqlite_store: SQLiteCacheStore):
    """Proves an expired lease from a crashed worker can be reclaimed by another worker."""
    event = OutboxEvent(
        event_type=EventType.USER_ROLE_CHANGED,
        aggregate_type="user",
        aggregate_id="usr_001",
        tenant_id="tenant_y",
        payload={"role": "admin"},
    )
    sqlite_store.record_outbox_event(event)

    # Worker 1 claims with very short lease
    batch_1 = sqlite_store.claim_outbox_events(
        limit=10, claim_owner="crashed-worker", lease_seconds=0.01
    )
    assert len(batch_1) == 1

    # Simulate time passing beyond lease expiry
    import time

    time.sleep(0.02)

    # Worker 2 claims expired event
    batch_2 = sqlite_store.claim_outbox_events(
        limit=10, claim_owner="healthy-worker", lease_seconds=30.0
    )
    assert len(batch_2) == 1
    assert batch_2[0].claim_owner == "healthy-worker"
    assert batch_2[0].retry_count == 2


# =============================================================================
# 3. OutboxRelay & Retry / Dead-Letter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_outbox_relay_successful_publication(sqlite_store: SQLiteCacheStore):
    """Tests OutboxRelay processing events to InMemoryEventBus and marking PUBLISHED."""
    event_bus = InMemoryEventBus()
    relay = OutboxRelay(store=sqlite_store, event_bus=event_bus, relay_id="test-relay")

    evt1 = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="doc_1",
        tenant_id="tenant_1",
        payload={"new_version": 5},
    )
    sqlite_store.record_outbox_event(evt1)

    published_count = await relay.run_once()
    assert published_count == 1
    assert len(event_bus.published_events) == 1
    assert event_bus.published_events[0].aggregate_id == "doc_1"

    # Verify event marked published in store
    saved_evt = sqlite_store.get_outbox_event(evt1.id)
    assert saved_evt is not None
    assert saved_evt.status == OutboxStatus.PUBLISHED
    assert saved_evt.published_at is not None


@pytest.mark.asyncio
async def test_outbox_relay_retry_and_dead_letter(sqlite_store: SQLiteCacheStore):
    """Tests OutboxRelay exponential backoff retry and transition to DEAD_LETTER."""

    class FailingEventBus:
        async def publish_batch(self, events):
            raise ConnectionError("Network down")

    relay = OutboxRelay(
        store=sqlite_store,
        event_bus=FailingEventBus(),
        relay_id="test-relay",
        max_retries=2,
    )

    evt = OutboxEvent(
        event_type=EventType.RESOURCE_DELETED,
        aggregate_type="resource",
        aggregate_id="doc_deleted",
        tenant_id="tenant_1",
        max_retries=2,
    )
    sqlite_store.record_outbox_event(evt)

    # Attempt 1: Transient failure -> retry
    res1 = await relay.run_once()
    assert res1 == 0
    saved1 = sqlite_store.get_outbox_event(evt.id)
    assert saved1 is not None
    assert saved1.status == OutboxStatus.PENDING
    assert saved1.retry_count == 1
    assert "Network down" in (saved1.last_error or "")

    # Force immediate retry eligibility
    sqlite_store._connection.execute(
        "UPDATE outbox_events SET next_attempt_at = ? WHERE id = ?",
        (datetime.now(tz=UTC).isoformat(), str(evt.id)),
    )
    sqlite_store._connection.commit()

    # Attempt 2: Exceeds max retries -> DEAD_LETTER
    res2 = await relay.run_once()
    assert res2 == 0
    saved2 = sqlite_store.get_outbox_event(evt.id)
    assert saved2 is not None
    assert saved2.status == OutboxStatus.DEAD_LETTER
    assert saved2.retry_count == 2


# =============================================================================
# 4. Consumer Idempotency Tests
# =============================================================================


@pytest.mark.asyncio
async def test_invalidation_consumer_idempotency(sqlite_store: SQLiteCacheStore):
    """Proves duplicate delivery of the same event_id is safely ignored."""
    invalidated_resources = []

    def on_res_updated(evt: OutboxEvent):
        invalidated_resources.append(evt.aggregate_id)

    consumer = InvalidationConsumer(
        consumer_name="worker-cache-invalidator",
        processed_store=sqlite_store,
        on_resource_updated=on_res_updated,
    )

    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="balance_sheet_2026",
        tenant_id="tenant_corp",
        payload={"new_version": 3},
    )

    # First delivery: processed
    first_pass = await consumer.handle_event(event)
    assert first_pass is True
    assert len(invalidated_resources) == 1
    assert invalidated_resources[0] == "balance_sheet_2026"

    # Second delivery (at-least-once duplicate): safely skipped
    second_pass = await consumer.handle_event(event)
    assert second_pass is False
    assert len(invalidated_resources) == 1  # No duplicate action


# =============================================================================
# 5. Version-Guarded Index Reconciliation Tests
# =============================================================================


def test_index_reconciler_version_guard_prevents_stale_resurrection(
    sqlite_store: SQLiteCacheStore,
):
    """Proves reconciler rejects restoring stale version v17 when live version advanced to v18."""
    # Register live resource at version 18
    sqlite_store.create_resource("sec_filing", "tenant_sec", initial_version=18)

    # Simulate an older cache entry at version 17
    sqlite_store.add(
        prompt="10-K summary",
        embedding=np.array([0.1] * 384, dtype=np.float32),
        response="Old 2025 data",
        model="gpt-4o",
        tenant_id="tenant_sec",
        scope=CacheScope.RESOURCE,
        resource_id="sec_filing",
        resource_version=17,  # Stale version!
        pipeline_version="v2",
    )

    # Mock embedding service
    added_to_vector_index = []

    class MockEmbeddingService:
        def add(self, **kwargs):
            added_to_vector_index.append(kwargs)

    reconciler = IndexReconciler(
        metadata_store=sqlite_store,
        embedding_service=MockEmbeddingService(),
    )

    report = reconciler.reconcile(tenant_id="tenant_sec", resource_id="sec_filing")

    assert report.total_checked == 1
    assert report.stale_versions_skipped == 1
    assert report.missing_repaired == 0
    assert len(added_to_vector_index) == 0  # Stale vector rejected from vector backend


def test_index_reconciler_restores_valid_missing_vector(sqlite_store: SQLiteCacheStore):
    """Proves reconciler restores missing valid vector at current version."""
    sqlite_store.create_resource("sec_filing_curr", "tenant_sec", initial_version=18)

    sqlite_store.add(
        prompt="10-K current summary",
        embedding=np.array([0.2] * 384, dtype=np.float32),
        response="Fresh 2026 data",
        model="gpt-4o",
        tenant_id="tenant_sec",
        scope=CacheScope.RESOURCE,
        resource_id="sec_filing_curr",
        resource_version=18,  # Matches current version!
        pipeline_version="v2",
    )

    added_to_vector_index = []

    class MockEmbeddingService:
        def add(self, **kwargs):
            added_to_vector_index.append(kwargs)

    reconciler = IndexReconciler(
        metadata_store=sqlite_store,
        embedding_service=MockEmbeddingService(),
    )

    report = reconciler.reconcile(tenant_id="tenant_sec", resource_id="sec_filing_curr")

    assert report.total_checked == 1
    assert report.stale_versions_skipped == 0
    assert report.missing_repaired == 1
    assert len(added_to_vector_index) == 1
    assert added_to_vector_index[0]["resource_version"] == 18


# =============================================================================
# 6. Pipeline Changed Event & Adapters Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pipeline_changed_event_dispatch():
    """Proves independent pipeline.changed event carries correct namespace."""
    event_bus = InMemoryEventBus()
    dispatched_pipelines = []

    async def on_pipeline_evt(evt: OutboxEvent):
        dispatched_pipelines.append(evt.payload)

    await event_bus.subscribe(EventType.PIPELINE_CHANGED.value, on_pipeline_evt)

    event = OutboxEvent(
        event_type=EventType.PIPELINE_CHANGED,
        schema_version=1,
        aggregate_type="pipeline",
        aggregate_id="semanticmemo-v2.1",
        tenant_id="global",
        payload={"old_version": "v2.0", "new_version": "v2.1"},
    )

    await event_bus.publish(event)
    assert len(dispatched_pipelines) == 1
    assert dispatched_pipelines[0]["new_version"] == "v2.1"


def test_kafka_missing_dependency_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests Kafka adapter raises MissingDependencyError when aiokafka is not installed."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "aiokafka":
            raise ImportError("No module named 'aiokafka'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    bus = KafkaEventBus()
    with pytest.raises(MissingDependencyError) as exc_info:
        bus._ensure_aiokafka()
    assert "aiokafka" in str(exc_info.value)


def test_rabbitmq_missing_dependency_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests RabbitMQ adapter raises MissingDependencyError when aio-pika is not installed."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("aio_pika", "aio-pika"):
            raise ImportError("No module named 'aio_pika'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    bus = RabbitMQEventBus()
    with pytest.raises(MissingDependencyError) as exc_info:
        bus._ensure_aio_pika()
    assert "aio-pika" in str(exc_info.value)
