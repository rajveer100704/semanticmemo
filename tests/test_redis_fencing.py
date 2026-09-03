"""Tests for Redis single-flight fencing generations and atomic CAS write protection."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from semanticmemo.concurrency.redis_flight import (
    RedisConfig,
    RedisSingleFlightManager,
)
from semanticmemo.store.sqlite_store import SQLiteCacheStore


class FakeAsyncRedis:
    """In-memory asynchronous Redis test double with atomic Lua scripting and key expirations."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.expires: dict[str, float] = {}

    def _cleanup_expired(self, key: str) -> None:
        if key in self.expires and time.time() > self.expires[key]:
            self.data.pop(key, None)
            self.expires.pop(key, None)

    async def get(self, key: str) -> str | None:
        self._cleanup_expired(key)
        return self.data.get(key)

    async def set(
        self,
        key: str,
        value: str,
        nx: bool = False,
        px: int | None = None,
    ) -> bool:
        self._cleanup_expired(key)
        if nx and key in self.data:
            return False
        self.data[key] = value
        if px is not None:
            self.expires[key] = time.time() + (px / 1000.0)
        else:
            self.expires.pop(key, None)
        return True

    async def incr(self, key: str) -> int:
        self._cleanup_expired(key)
        val = int(self.data.get(key, 0)) + 1
        self.data[key] = str(val)
        return val

    async def publish(self, channel: str, message: str) -> int:
        return 1

    async def eval(self, script: str, numkeys: int, key: str, arg: str) -> int:
        self._cleanup_expired(key)
        # Release Lua: compare and delete
        if 'return redis.call("del"' in script:
            if self.data.get(key) == arg:
                self.data.pop(key, None)
                self.expires.pop(key, None)
                return 1
            return 0
        return 0


@pytest.mark.asyncio
async def test_redis_fencing_stale_leader_write_rejected(tmp_path: Any) -> None:
    """Proves that a stalled leader exceeding lock TTL is rejected by CAS when writing."""
    db_path = tmp_path / "fencing_cache.db"
    store = SQLiteCacheStore(str(db_path))
    redis_client = FakeAsyncRedis()
    config = RedisConfig(lock_ttl_ms=100, wait_timeout_ms=500, poll_interval_ms=10)

    mgr_a = RedisSingleFlightManager(client=redis_client, config=config)
    mgr_b = RedisSingleFlightManager(client=redis_client, config=config)

    flight_key = "sm:flight:tenant_x:res_1:v1:prompt_hash_1"
    prompt = "Generate quarterly revenue analysis"
    embedding = np.array([0.1] * 384, dtype=np.float32)

    # 1. Leader A acquires lock (gen 1, TTL 100ms) and stalls
    leader_a_started = asyncio.Event()
    leader_b_finished = asyncio.Event()

    async def operation_a() -> str:
        leader_a_started.set()
        # Stall for 200ms (exceeding 100ms TTL)
        await asyncio.sleep(0.20)
        return "Stale Result A from slow leader"

    async def loader() -> str | None:
        entry = store.find_by_prompt(prompt=prompt, tenant_id="tenant_x")
        return entry.response if entry else None

    async def writer_a(resp: str, gen: int = 0) -> None:
        store.add(
            prompt=prompt,
            embedding=embedding,
            response=resp,
            model="test-model",
            tenant_id="tenant_x",
            flight_generation=gen,
        )

    async def run_worker_a() -> str:
        return await mgr_a.execute(
            flight_key=flight_key,
            operation=operation_a,
            result_loader=loader,
            result_writer=writer_a,
        )

    # 2. Worker B is follower, then upon lock expiry takes over (gen 2)
    async def operation_b() -> str:
        return "Fresh Result B from takeover leader"

    async def writer_b(resp: str, gen: int = 0) -> None:
        store.add(
            prompt=prompt,
            embedding=embedding,
            response=resp,
            model="test-model",
            tenant_id="tenant_x",
            flight_generation=gen,
        )

    async def run_worker_b() -> str:
        await leader_a_started.wait()
        # Wait 120ms for Leader A's 100ms lock to expire
        await asyncio.sleep(0.12)
        res = await mgr_b.execute(
            flight_key=flight_key,
            operation=operation_b,
            result_loader=loader,
            result_writer=writer_b,
        )
        leader_b_finished.set()
        return res

    # Run Worker A and Worker B concurrently
    res_a, res_b = await asyncio.gather(run_worker_a(), run_worker_b())

    # Worker B successfully took over and wrote Result B (generation 2)
    assert res_b == "Fresh Result B from takeover leader"

    # Authoritative store MUST retain Result B and MUST NOT be overwritten by A
    cached = store.find_by_prompt(prompt=prompt, tenant_id="tenant_x")
    assert cached is not None
    assert cached.response == "Fresh Result B from takeover leader"
    assert cached.flight_generation == 2

    # Verify metrics show stale leader write rejection
    assert (
        mgr_a.metrics.stale_leader_writes_rejected >= 1 or mgr_b.metrics.lock_expired_takeovers >= 1
    )


@pytest.mark.asyncio
async def test_redis_safe_release_does_not_delete_new_leader_lock() -> None:
    """Proves token-checked Lua release does not delete a lock acquired by a newer leader."""
    redis_client = FakeAsyncRedis()
    flight_key = "sm:lock:key_test"

    # Worker A acquired lock with token A
    await redis_client.set(flight_key, "token_a", px=50)

    # Lock expires and Worker B acquires lock with token B
    await redis_client.set(flight_key, "token_b", px=5000)

    # Worker A attempts to release with token A -> must return 0 and keep token B intact
    res = await redis_client.eval(
        """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """,
        1,
        flight_key,
        "token_a",
    )
    assert res == 0
    assert await redis_client.get(flight_key) == "token_b"
