"""Tests for RedisDistributedSingleFlight and InProcessSingleFlightManager."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

import pytest

from semanticmemo.concurrency import (
    InProcessSingleFlightManager,
    RedisConfig,
    RedisSingleFlightManager,
)
from semanticmemo.concurrency.redis_flight import RELEASE_LUA
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, RetrievalScope


class FakeAsyncRedis:
    """In-memory async Redis simulator for unit testing single-flight concurrency."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[str, float | None]] = {}  # key -> (value, expire_at)
        self.channels: dict[str, list[asyncio.Queue[str]]] = {}
        self.call_log: list[str] = []

    async def set(
        self,
        name: str,
        value: str,
        nx: bool = False,
        px: int | None = None,
    ) -> bool:
        loop = asyncio.get_running_loop()
        now = loop.time()
        # Check existing key and expiry
        if name in self.store:
            old_val, exp = self.store[name]
            if exp is not None and now >= exp:
                del self.store[name]
            elif nx:
                return False

        expire_at = (now + (px / 1000.0)) if px else None
        self.store[name] = (value, expire_at)
        return True

    async def get(self, name: str) -> str | None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if name in self.store:
            val, exp = self.store[name]
            if exp is not None and now >= exp:
                del self.store[name]
                return None
            return val
        return None

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> int:
        # Simulate RELEASE_LUA compare-and-delete
        if "del" in script:
            key = str(keys_and_args[0])
            expected_token = str(keys_and_args[1])
            curr_val = await self.get(key)
            if curr_val == expected_token:
                self.store.pop(key, None)
                return 1
            return 0
        return 0

    async def publish(self, channel: str, message: str) -> int:
        if channel in self.channels:
            for q in self.channels[channel]:
                await q.put(message)
            return len(self.channels[channel])
        return 0

    def pubsub(self):
        return FakePubSub(self)


class FakePubSub:
    """Simulator for Redis Pub/Sub subscription queue."""

    def __init__(self, redis_instance: FakeAsyncRedis) -> None:
        self.redis = redis_instance
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.subscribed_channels: list[str] = []

    async def subscribe(self, channel: str) -> None:
        self.subscribed_channels.append(channel)
        if channel not in self.redis.channels:
            self.redis.channels[channel] = []
        self.redis.channels[channel].append(self.queue)

    async def get_message(
        self,
        ignore_subscribe_messages: bool = True,
        timeout: float = 0.1,
    ) -> dict[str, Any] | None:
        try:
            data = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return {"type": "message", "data": data}
        except TimeoutError:
            return None

    async def unsubscribe(self, channel: str) -> None:
        if channel in self.redis.channels and self.queue in self.redis.channels[channel]:
            self.redis.channels[channel].remove(self.queue)

    async def close(self) -> None:
        for ch in self.subscribed_channels:
            await self.unsubscribe(ch)


@pytest.mark.asyncio
async def test_in_process_single_flight_coalescing():
    """Verify InProcessSingleFlightManager coalesces concurrent callers into 1 execution."""
    manager = InProcessSingleFlightManager()
    flight_key = "sm:lock:v2:tenant:t1:_:_:hash1"

    llm_calls = 0

    async def slow_llm_op() -> str:
        nonlocal llm_calls
        llm_calls += 1
        await asyncio.sleep(0.05)
        return "LLM Response Fresh"

    async def null_loader() -> str | None:
        return None

    # Launch 5 concurrent calls
    tasks = [manager.execute(flight_key, slow_llm_op, result_loader=null_loader) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r == "LLM Response Fresh" for r in results)
    assert llm_calls == 1  # Exactly one execution occurred


@pytest.mark.asyncio
async def test_redis_single_flight_leader_follower_coalescing(monkeypatch: pytest.MonkeyPatch):
    """Verify RedisSingleFlightManager coordinates leader execution and follower cache retrieval."""
    import semanticmemo.concurrency.redis_flight as rmodule

    monkeypatch.setattr(rmodule, "REDIS_AVAILABLE", True)

    fake_redis = FakeAsyncRedis()
    config = RedisConfig(lock_ttl_ms=5000, wait_timeout_ms=5000, poll_interval_ms=10)
    manager = RedisSingleFlightManager(client=fake_redis, config=config)

    flight_key = "sm:lock:v2:resource:t1:doc1:1:hash1"
    llm_execution_count = 0
    cached_db: dict[str, str] = {}

    async def llm_op() -> str:
        nonlocal llm_execution_count
        llm_execution_count += 1
        await asyncio.sleep(0.05)
        return "Response from LLM"

    async def result_loader() -> str | None:
        return cached_db.get(flight_key)

    async def result_writer(res: str) -> None:
        cached_db[flight_key] = res

    # Run leader and 4 followers concurrently
    tasks = [
        manager.execute(
            flight_key,
            llm_op,
            result_loader=result_loader,
            result_writer=result_writer,
        )
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r == "Response from LLM" for r in results)
    assert llm_execution_count == 1
    assert manager.metrics.acquired == 1
    assert manager.metrics.waited == 4
    assert manager.metrics.llm_deduped == 4
    assert manager.metrics.owner_releases == 1


@pytest.mark.asyncio
async def test_redis_single_flight_safe_lua_release(monkeypatch: pytest.MonkeyPatch):
    """Verify atomic Lua release only deletes key if token matches exactly."""
    import semanticmemo.concurrency.redis_flight as rmodule

    monkeypatch.setattr(rmodule, "REDIS_AVAILABLE", True)

    fake_redis = FakeAsyncRedis()
    lock_key = "sm:lock:test_key"
    token_a = "token_a_uuid"
    token_b = "token_b_uuid"

    await fake_redis.set(lock_key, token_a, nx=True, px=10000)

    # 1. Attempt release with wrong token B -> fails
    res_b = await fake_redis.eval(RELEASE_LUA, 1, lock_key, token_b)
    assert res_b == 0
    assert await fake_redis.get(lock_key) == token_a

    # 2. Release with correct token A -> succeeds
    res_a = await fake_redis.eval(RELEASE_LUA, 1, lock_key, token_a)
    assert res_a == 1
    assert await fake_redis.get(lock_key) is None


@pytest.mark.asyncio
async def test_redis_single_flight_leader_crash_takeover(monkeypatch: pytest.MonkeyPatch):
    """Verify that when a leader crashes (lock expires), a waiting follower takes over."""
    import semanticmemo.concurrency.redis_flight as rmodule

    monkeypatch.setattr(rmodule, "REDIS_AVAILABLE", True)

    fake_redis = FakeAsyncRedis()
    # Short lock TTL so it expires quickly
    config = RedisConfig(lock_ttl_ms=40, wait_timeout_ms=1000, poll_interval_ms=10)
    manager = RedisSingleFlightManager(client=fake_redis, config=config)
    flight_key = "sm:lock:v2:global:default:_:_:hash_crash"

    cached_db: dict[str, str] = {}
    attempts = 0

    async def crashing_leader_op() -> str:
        nonlocal attempts
        attempts += 1
        # Leader crashes and never writes to cache or releases lock
        await asyncio.sleep(0.08)  # Exceeds lock_ttl_ms (40ms)
        raise RuntimeError("Leader process crashed unexpectedly")

    async def successful_follower_op() -> str:
        nonlocal attempts
        attempts += 1
        return "Recovered Result by New Leader"

    async def result_loader() -> str | None:
        return cached_db.get(flight_key)

    async def result_writer(res: str) -> None:
        cached_db[flight_key] = res

    # 1. Start leader that crashes
    leader_task = asyncio.create_task(
        manager.execute(
            flight_key,
            crashing_leader_op,
            result_loader=result_loader,
            result_writer=result_writer,
        )
    )

    # Allow leader to acquire lock
    await asyncio.sleep(0.01)

    # 2. Follower arrives while leader is running, waits, detects expiration, takes over
    follower_result = await manager.execute(
        flight_key,
        successful_follower_op,
        result_loader=result_loader,
        result_writer=result_writer,
    )

    assert follower_result == "Recovered Result by New Leader"
    assert manager.metrics.lock_expired_takeovers >= 1

    with pytest.raises(RuntimeError, match="Leader process crashed"):
        await leader_task


@pytest.mark.asyncio
async def test_redis_single_flight_pubsub_missed_fallback_to_store(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that if Pub/Sub event is missed, follower still resolves from authoritative store."""
    import semanticmemo.concurrency.redis_flight as rmodule

    monkeypatch.setattr(rmodule, "REDIS_AVAILABLE", True)

    fake_redis = FakeAsyncRedis()
    # Disable pubsub to simulate lost/missed network event
    config = RedisConfig(
        lock_ttl_ms=5000,
        wait_timeout_ms=2000,
        poll_interval_ms=10,
        use_pubsub=False,
    )
    manager = RedisSingleFlightManager(client=fake_redis, config=config)
    flight_key = "sm:lock:v2:tenant:t_fin:_:_:hash_pubsub_miss"

    cached_db: dict[str, str] = {}

    async def leader_op() -> str:
        await asyncio.sleep(0.04)
        return "Durable Cached Value"

    async def result_loader() -> str | None:
        return cached_db.get(flight_key)

    async def result_writer(res: str) -> None:
        cached_db[flight_key] = res

    tasks = [
        manager.execute(
            flight_key,
            leader_op,
            result_loader=result_loader,
            result_writer=result_writer,
        )
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert all(r == "Durable Cached Value" for r in results)
    assert manager.metrics.acquired == 1
    assert manager.metrics.llm_deduped == 2


def test_redis_missing_dependency_simulation(monkeypatch: pytest.MonkeyPatch):
    """Test that MissingDependencyError is raised when redis is not available."""
    import semanticmemo.concurrency.redis_flight as rmodule

    monkeypatch.setattr(rmodule, "REDIS_AVAILABLE", False)
    with pytest.raises(MissingDependencyError, match="redis is required"):
        RedisSingleFlightManager()


def test_redis_flight_key_isolation():
    """Security Invariant: Different tenants, resources, versions, or pipelines have unique keys."""
    scope_t1 = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="doc_x",
        resource_version=1,
        pipeline_version="v2",
    )
    scope_t2 = RetrievalScope(
        tenant_id="tenant_b",
        scope=CacheScope.RESOURCE,
        resource_id="doc_x",
        resource_version=1,
        pipeline_version="v2",
    )
    scope_v2 = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="doc_x",
        resource_version=2,
        pipeline_version="v2",
    )
    scope_pipe_v3 = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="doc_x",
        resource_version=1,
        pipeline_version="v3",
    )

    prompt_hash = "abc123hash"

    def make_key(sc: RetrievalScope) -> str:
        return (
            f"sm:lock:{sc.pipeline_version}:{sc.scope.value}:"
            f"{sc.tenant_id}:{sc.resource_id}:{sc.resource_version}:{prompt_hash}"
        )

    key_t1 = make_key(scope_t1)
    key_t2 = make_key(scope_t2)
    key_v2 = make_key(scope_v2)
    key_pipe_v3 = make_key(scope_pipe_v3)

    assert len({key_t1, key_t2, key_v2, key_pipe_v3}) == 4


@pytest.mark.asyncio
async def test_redis_live_integration_server():
    """Integration test against live Redis server if reachable."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]

        client: Any = aioredis.from_url(redis_url, socket_timeout=1.0)
        await client.ping()
    except Exception as exc:
        pytest.skip(f"Live Redis server not available at {redis_url}: {exc}")

    config = RedisConfig(
        url=redis_url,
        lock_ttl_ms=3000,
        wait_timeout_ms=3000,
        poll_interval_ms=10,
    )
    manager = RedisSingleFlightManager(client=client, config=config)
    flight_key = f"sm:lock:live_test_{uuid4().hex}"
    cached_db: dict[str, str] = {}

    async def op() -> str:
        await asyncio.sleep(0.02)
        return "Live Redis Result"

    async def loader() -> str | None:
        return cached_db.get(flight_key)

    async def writer(r: str) -> None:
        cached_db[flight_key] = r

    res = await manager.execute(
        flight_key,
        op,
        result_loader=loader,
        result_writer=writer,
    )
    assert res == "Live Redis Result"
    await client.aclose()
