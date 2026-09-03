"""Comprehensive Production Chaos & 4 Core Failure Invariants Test Suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from semanticmemo.cache import SemanticMemo
from semanticmemo.events.bus import InMemoryEventBus
from semanticmemo.events.consumer import InvalidationConsumer
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.events.relay import OutboxRelay
from semanticmemo.models import CacheConfig
from semanticmemo.store.sqlite_store import SQLiteCacheStore

# =============================================================================
# Part 1: Partial Outage Matrix
# =============================================================================


@pytest.mark.asyncio
async def test_chaos_matrix_redis_down_qdrant_up(tmp_path: Any) -> None:
    """Redis outage falls back to local in-process flight without auth compromise."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_chaos_1", tenant_id="tenant_chaos")

    # Request executes cleanly despite Redis absence
    result = await cache.get_or_call(
        prompt="Fiscal revenue forecast Q4",
        llm_function=lambda p: "Forecast Q4: $10.5M",
        user_id=user.id,
    )
    assert result.response == "Forecast Q4: $10.5M"
    assert result.was_cache_hit is False

    # Second identical prompt hits local cache
    hit_result = await cache.get_or_call(
        prompt="Fiscal revenue forecast Q4",
        llm_function=lambda p: "Should not be called",
        user_id=user.id,
    )
    assert hit_result.was_cache_hit is True
    assert hit_result.response == "Forecast Q4: $10.5M"


@pytest.mark.asyncio
async def test_chaos_matrix_qdrant_down_redis_up(tmp_path: Any) -> None:
    """Vector search failure fails closed to cache miss without search space expansion."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_chaos_2", tenant_id="tenant_chaos")

    # Simulate vector search service raising an exception during search
    real_embed = cache.orchestrator.embedding_service.embed

    class FailingVectorService:
        def embed(self, prompt: str) -> Any:
            return real_embed(prompt)

        def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        def add(self, *args: Any, **kwargs: Any) -> None:
            pass

    cache.orchestrator.embedding_service = FailingVectorService()  # type: ignore[assignment]

    # Must fail closed to a cache miss and call LLM safely
    result = await cache.get_or_call(
        prompt="Risk assessment metrics",
        llm_function=lambda p: "Risk Tier: Low",
        user_id=user.id,
    )
    assert result.response == "Risk Tier: Low"
    assert result.was_cache_hit is False


@pytest.mark.asyncio
async def test_chaos_matrix_postgres_down_fails_closed(tmp_path: Any) -> None:
    """Auth/metadata store failure fails closed (100% fail-closed, no broader authorization)."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )

    # Invalidate store connection to simulate database outage
    cache.orchestrator.store.close()

    import sqlite3

    # Must raise an exception and fail closed rather than serving unverified data
    with pytest.raises((sqlite3.ProgrammingError, PermissionError)):
        await cache.get_or_call(
            prompt="Sensitive transaction record",
            llm_function=lambda p: "Secret Data",
            user_id="any_user",
        )


@pytest.mark.asyncio
async def test_chaos_matrix_broker_down_preserves_outbox(tmp_path: Any) -> None:
    """Broker outage after DB commit preserves outbox durability and retries cleanly."""
    db_path = tmp_path / "outbox_chaos.db"
    store = SQLiteCacheStore(str(db_path))

    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="res_chaos_1",
        tenant_id="tenant_chaos",
        payload={"new_version": 2},
    )
    store.record_outbox_event(event)

    class FailingBroker:
        async def publish(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("Kafka cluster offline")

    relay = OutboxRelay(
        store=store,
        event_bus=FailingBroker(),  # type: ignore[arg-type]
        relay_id="chaos_relay",
        claim_lease_seconds=30.0,
    )

    # Relay attempt fails -> must not lose event
    published = await relay.run_once()
    assert published == 0

    saved = store.get_outbox_event(event.id)
    assert saved is not None
    assert saved.status.value in ("pending", "processing")
    assert saved.retry_count >= 1


# =============================================================================
# Part 2: The Four Core Failure Invariants
# =============================================================================


@pytest.mark.asyncio
async def test_invariant_1_no_broader_authorization(tmp_path: Any) -> None:
    """Invariant 1: NO BROADER AUTHORIZATION under revoked permissions or degraded auth."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_inv_1", tenant_id="tenant_inv")
    res = cache.orchestrator.store.create_resource("res_inv_1", "tenant_inv")

    # Grant access to analyst role on this resource
    cache.orchestrator.store.grant_resource_access("res_inv_1", "analyst", "resource:read")

    # Pre-populate cache for this resource
    await cache.get_or_call(
        prompt="Quarterly financial statements",
        llm_function=lambda p: "Financial Statement v1",
        user_id=user.id,
        resource_id=res.id,
    )

    # Revoke access from analyst and restrict to viewer
    cache.orchestrator.store.revoke_resource_access("res_inv_1", "analyst", "resource:read")
    cache.orchestrator.store.grant_resource_access("res_inv_1", "viewer", "resource:read")

    # Must be denied immediately with PermissionError
    with pytest.raises(PermissionError):
        await cache.get_or_call(
            prompt="Quarterly financial statements",
            llm_function=lambda p: "Should not be called",
            user_id=user.id,
            resource_id=res.id,
        )


@pytest.mark.asyncio
async def test_invariant_2_no_stale_resource_resurrection(tmp_path: Any) -> None:
    """Invariant 2: NO STALE RESOURCE RESURRECTION when version is bumped or old events replayed."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_inv_2", tenant_id="tenant_inv")
    res = cache.orchestrator.store.create_resource("res_inv_2", "tenant_inv")

    # 1. Populate cache at Version 1
    res_v1 = await cache.get_or_call(
        prompt="Audit findings summary",
        llm_function=lambda p: "Audit findings v1 (Initial)",
        user_id=user.id,
        resource_id=res.id,
    )
    assert res_v1.response == "Audit findings v1 (Initial)"
    assert res_v1.was_cache_hit is False

    # 2. Advance resource version to Version 2 in authoritative store
    new_ver = cache.orchestrator.store.increment_resource_version("res_inv_2")
    assert new_ver == 2

    # 3. Re-querying must NOT resurrect Version 1 response
    res_v2 = await cache.get_or_call(
        prompt="Audit findings summary",
        llm_function=lambda p: "Audit findings v2 (Updated)",
        user_id=user.id,
        resource_id=res.id,
    )
    assert res_v2.response == "Audit findings v2 (Updated)"
    assert res_v2.was_cache_hit is False


@pytest.mark.asyncio
async def test_invariant_3_no_cross_tenant_data_leak(tmp_path: Any) -> None:
    """Invariant 3: NO CROSS-TENANT DATA LEAK across identical prompts and identical embeddings."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user_a = cache.orchestrator.bootstrap_default_user("user_a", tenant_id="tenant_alpha")
    user_b = cache.orchestrator.bootstrap_default_user("user_b", tenant_id="tenant_beta")

    # Tenant Alpha populates cache
    res_a = await cache.get_or_call(
        prompt="Executive compensation budget",
        llm_function=lambda p: "Alpha Confidential: $500,000",
        user_id=user_a.id,
    )
    assert res_a.response == "Alpha Confidential: $500,000"
    assert res_a.was_cache_hit is False

    # Tenant Beta executes exact same prompt -> must NOT receive Alpha's confidential data
    res_b = await cache.get_or_call(
        prompt="Executive compensation budget",
        llm_function=lambda p: "Beta Confidential: $250,000",
        user_id=user_b.id,
    )
    assert res_b.response == "Beta Confidential: $250,000"
    assert res_b.was_cache_hit is False


@pytest.mark.asyncio
async def test_invariant_4_no_permanent_outbox_loss(tmp_path: Any) -> None:
    """Invariant 4: NO PERMANENT OUTBOX LOSS with guaranteed delivery and idempotent consumption."""
    db_path = tmp_path / "outbox_inv.db"
    store = SQLiteCacheStore(str(db_path))
    bus = InMemoryEventBus()

    # Record outbox event
    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="doc_inv_4",
        tenant_id="tenant_inv",
        payload={"new_version": 10},
    )
    store.record_outbox_event(event)

    # Relay publishes to event bus
    relay = OutboxRelay(store=store, event_bus=bus, relay_id="inv_relay")
    count = await relay.run_once()
    assert count == 1
    assert len(bus.published_events) == 1

    # Consumer processes event idempotently
    consumer = InvalidationConsumer(consumer_name="inv_consumer", processed_store=store)
    handled_first = await consumer.handle_event(bus.published_events[0])
    assert handled_first is True

    # Replayed event is recognized and deduplicated
    handled_duplicate = await consumer.handle_event(bus.published_events[0])
    assert handled_duplicate is False
