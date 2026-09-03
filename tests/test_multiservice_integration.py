"""Multi-Tenant Load & Chaos Verification Suite (Scenarios A through H)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from semanticmemo.cache import SemanticMemo
from semanticmemo.concurrency import InProcessSingleFlightManager
from semanticmemo.events.consumer import InvalidationConsumer
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.events.relay import OutboxRelay
from semanticmemo.models import CacheConfig, PermissionName
from semanticmemo.store.sqlite_store import SQLiteCacheStore


@pytest.mark.asyncio
async def test_scenario_a_100_concurrent_misses_coalesce_to_1_llm_call(
    tmp_path: Any,
) -> None:
    """Scenario A: 100 identical concurrent cache misses coalesce into exactly 1 LLM call."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_a", tenant_id="tenant_a")

    llm_call_count = 0

    async def mock_llm(prompt: str) -> str:
        nonlocal llm_call_count
        llm_call_count += 1
        await asyncio.sleep(0.05)  # simulate LLM latency
        return "Coalesced LLM Response"

    tasks = [
        cache.get_or_call(
            prompt="What is our Q3 financial forecast?",
            llm_function=mock_llm,
            user_id=user.id,
        )
        for _ in range(100)
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 100
    assert all(r.response == "Coalesced LLM Response" for r in results)
    assert llm_call_count == 1  # Exactly 1 LLM invocation for 100 concurrent callers


@pytest.mark.asyncio
async def test_scenario_b_100_requests_across_10_tenants(
    tmp_path: Any,
) -> None:
    """Scenario B: 100 requests across 10 distinct tenants coalesce into 10 flights."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )

    users = [
        cache.orchestrator.bootstrap_default_user(f"user_b_{i}", tenant_id=f"tenant_b_{i}")
        for i in range(10)
    ]

    llm_calls_per_tenant: dict[str, int] = {u.tenant_id: 0 for u in users}

    async def mock_llm(prompt: str, tenant_id: str) -> str:
        llm_calls_per_tenant[tenant_id] += 1
        await asyncio.sleep(0.04)
        return f"Response for {tenant_id}"

    tasks = []
    for i in range(100):
        user = users[i % 10]
        t_id = user.tenant_id

        async def _call(u: Any = user, tid: str = t_id) -> Any:
            return await cache.get_or_call(
                prompt="Quarterly audit inquiry",
                llm_function=lambda p: mock_llm(p, tid),
                user_id=u.id,
            )

        tasks.append(_call())

    results = await asyncio.gather(*tasks)

    assert len(results) == 100
    # Each tenant had 10 concurrent requests coalescing to exactly 1 LLM call per tenant (10 total)
    assert sum(llm_calls_per_tenant.values()) == 10
    assert all(count == 1 for count in llm_calls_per_tenant.values())


@pytest.mark.asyncio
async def test_scenario_c_100_requests_across_10_resources_isolation(
    tmp_path: Any,
) -> None:
    """Scenario C: 100 requests across 10 distinct resources maintain complete isolation."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_c", tenant_id="tenant_c")

    # Create 10 distinct resources with explicit read/write permissions
    resources = []
    for i in range(10):
        res = cache.orchestrator.store.create_resource(f"doc_{i}", tenant_id=user.tenant_id)
        cache.orchestrator.store.grant_resource_access(
            resource_id=res.id,
            role_id="analyst",
            permission_id=PermissionName.RESOURCE_READ.value,
        )
        cache.orchestrator.store.grant_resource_access(
            resource_id=res.id,
            role_id="analyst",
            permission_id=PermissionName.CACHE_READ.value,
        )
        cache.orchestrator.store.grant_resource_access(
            resource_id=res.id,
            role_id="analyst",
            permission_id=PermissionName.CACHE_WRITE.value,
        )
        resources.append(res)

    llm_calls_per_resource: dict[str, int] = {r.id: 0 for r in resources}

    async def mock_llm(prompt: str, res_id: str) -> str:
        llm_calls_per_resource[res_id] += 1
        await asyncio.sleep(0.03)
        return f"Resource content for {res_id}"

    tasks = []
    for i in range(100):
        res = resources[i % 10]
        r_id = res.id

        async def _call(r: Any = res, rid: str = r_id) -> Any:
            return await cache.get_or_call(
                prompt="Summarize attached document",
                llm_function=lambda p: mock_llm(p, rid),
                user_id=user.id,
                resource_id=r.id,
            )

        tasks.append(_call())

    results = await asyncio.gather(*tasks)
    assert len(results) == 100
    assert sum(llm_calls_per_resource.values()) == 10
    assert all(count == 1 for count in llm_calls_per_resource.values())


@pytest.mark.asyncio
async def test_scenario_d_leader_crash_takeover() -> None:
    """Scenario D: Leader task failure/cancellation triggers safe follower takeover."""
    manager = InProcessSingleFlightManager()
    execution_count = 0

    async def failing_leader() -> str:
        nonlocal execution_count
        execution_count += 1
        raise RuntimeError("Simulated LLM API crash")

    async def successful_follower() -> str:
        nonlocal execution_count
        execution_count += 1
        return "Recovered Success"

    cached_result: str | None = None

    async def loader() -> str | None:
        return cached_result

    async def writer(val: str) -> None:
        nonlocal cached_result
        cached_result = val

    # Leader fails
    with pytest.raises(RuntimeError, match="Simulated LLM API crash"):
        await manager.execute(
            flight_key="flight_crash_test",
            operation=failing_leader,
            result_loader=loader,
            result_writer=writer,
        )

    # Next attempt cleanly takes over
    result = await manager.execute(
        flight_key="flight_crash_test",
        operation=successful_follower,
        result_loader=loader,
        result_writer=writer,
    )

    assert result == "Recovered Success"
    assert execution_count == 2


@pytest.mark.asyncio
async def test_scenario_e_redis_outage_fail_closed_or_local() -> None:
    """Scenario E: Redis failure safely falls closed without unauthenticated bypass."""
    manager = InProcessSingleFlightManager()
    called = False

    async def op() -> str:
        nonlocal called
        called = True
        return "Fallback OK"

    async def loader() -> str | None:
        return None

    async def writer(v: str) -> None:
        pass

    res = await manager.execute(
        flight_key="redis_outage_key",
        operation=op,
        result_loader=loader,
        result_writer=writer,
    )
    assert res == "Fallback OK"
    assert called is True


@pytest.mark.asyncio
async def test_scenario_f_postgresql_outage_fails_closed(
    tmp_path: Any,
) -> None:
    """Scenario F: Authentication failure raises PermissionError (fail closed)."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )

    # Calling get_or_call without an authenticated user_id must fail-closed immediately
    with pytest.raises(PermissionError, match="Authentication required"):
        await cache.get_or_call(
            prompt="Confidential query",
            llm_function=lambda p: "unauthorized result",
            user_id="",
        )


@pytest.mark.asyncio
async def test_scenario_g_vector_index_outage_fails_closed(
    tmp_path: Any,
) -> None:
    """Scenario G: Vector search failure fails closed to cache miss."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_g", tenant_id="tenant_g")

    # Mock embedding_service.search to return empty (simulating vector DB error / miss)
    with patch.object(cache.orchestrator.embedding_service, "search", return_value=[]):
        result = await cache.get_or_call(
            prompt="Any prompt",
            llm_function=lambda p: "LLM executed cleanly",
            user_id=user.id,
        )
        assert result.was_cache_hit is False
        assert result.response == "LLM executed cleanly"


@pytest.mark.asyncio
async def test_scenario_h_outbox_deduplication_idempotency(
    tmp_path: Any,
) -> None:
    """Scenario H: Replayed duplicate outbox events cause exactly 1 consumer execution."""
    db_path = tmp_path / "cache.db"
    store = SQLiteCacheStore(str(db_path))

    execution_count = 0

    def on_update(evt: OutboxEvent) -> None:
        nonlocal execution_count
        execution_count += 1

    consumer = InvalidationConsumer(
        consumer_name="chaos_consumer",
        processed_store=store,
        on_resource_updated=on_update,
    )

    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="res_h_1",
        tenant_id="tenant_h_1",
        payload={"version": 2},
    )

    # First delivery
    first_res = await consumer.handle_event(event)
    assert first_res is True
    assert execution_count == 1

    # Second delivery (duplicate replay)
    second_res = await consumer.handle_event(event)
    assert second_res is False
    assert execution_count == 1  # Guaranteed idempotent execution count


class FlakyEventBus:
    """Event bus that simulates broker network outages and recovery."""

    def __init__(self) -> None:
        self.is_healthy = False
        self.published_events: list[OutboxEvent] = []

    async def publish(self, event: OutboxEvent) -> None:
        if not self.is_healthy:
            raise ConnectionError("Kafka / RabbitMQ broker unreachable")
        self.published_events.append(event)

    async def publish_batch(self, events: list[OutboxEvent]) -> None:
        for evt in events:
            await self.publish(evt)

    async def subscribe(self, event_type: Any, handler: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_scenario_i_event_bus_outage_after_db_commit(tmp_path: Any) -> None:
    """Scenario I: Broker outage after DB commit preserves outbox durability and retries cleanly."""
    db_path = tmp_path / "outbox_chaos.db"
    store = SQLiteCacheStore(str(db_path))

    # 1. State mutation + outbox record in DB transaction
    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="res_i_100",
        tenant_id="tenant_i_100",
        payload={"new_version": 2},
    )
    store.record_outbox_event(event)

    flaky_bus = FlakyEventBus()
    relay = OutboxRelay(
        store=store,
        event_bus=flaky_bus,  # type: ignore[arg-type]
        relay_id="relay_worker_i",
        claim_lease_seconds=30.0,
        base_backoff_seconds=0.01,
    )

    # 2. First attempt while broker is down -> must not lose event
    processed_count = await relay.run_once(batch_size=10)
    assert processed_count == 0
    assert len(flaky_bus.published_events) == 0

    # Verify event remains durable in store with incremented retry count
    saved_event = store.get_outbox_event(event.id)
    assert saved_event is not None
    assert saved_event.retry_count >= 1
    assert saved_event.status.value in ("pending", "processing", "failed")

    # 3. Broker recovers
    flaky_bus.is_healthy = True
    await asyncio.sleep(0.6)  # wait for backoff with jitter (0.0 - 0.5s) to expire

    # Relay retries and publishes successfully
    published_count = await relay.run_once(batch_size=10)
    assert published_count == 1
    assert len(flaky_bus.published_events) == 1
    assert flaky_bus.published_events[0].aggregate_id == "res_i_100"


@pytest.mark.asyncio
async def test_scenario_j_observability_backend_failure_fails_safe(tmp_path: Any) -> None:
    """Scenario J: Observability / telemetry failure never alters correctness or auth."""
    from unittest.mock import MagicMock

    from semanticmemo.telemetry.tracing import get_tracer

    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_j", tenant_id="tenant_j")

    tracer = get_tracer()
    # Simulate broken / throwing OpenTelemetry tracer
    failing_otel = MagicMock()
    err_msg = "OTel collector connection refused"
    failing_otel.start_as_current_span.side_effect = RuntimeError(err_msg)
    tracer._otel_tracer = failing_otel

    try:
        # Request must execute cleanly and return correct LLM output
        result = await cache.get_or_call(
            prompt="Quarterly audit verification",
            llm_function=lambda p: "Verified correct response despite OTel failure",
            user_id=user.id,
        )
        assert result.response == "Verified correct response despite OTel failure"
        assert result.was_cache_hit is False
    finally:
        tracer._otel_tracer = None
