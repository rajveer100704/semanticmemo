"""Redis distributed single-flight manager implementing atomic request coalescing."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from semanticmemo.exceptions import MissingDependencyError

if TYPE_CHECKING:
    pass

T = TypeVar("T")
logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore[assignment]


# Atomic compare-and-delete Lua script
RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


@dataclass
class RedisConfig:
    """Configuration for Redis single-flight coordination."""

    url: str = "redis://localhost:6379/0"
    lock_ttl_ms: int = 30000  # 30 seconds default lock expiration
    wait_timeout_ms: int = 35000  # 35 seconds follower timeout
    poll_interval_ms: int = 50  # 50ms polling interval
    key_prefix: str = "sm:lock"
    use_pubsub: bool = True  # Use Pub/Sub as an opportunistic wake-up optimization


@dataclass
class RedisFlightMetrics:
    """Observability metrics for Redis single-flight coordination."""

    acquired: int = 0
    waited: int = 0
    owner_releases: int = 0
    lock_expired_takeovers: int = 0
    stale_leader_writes_rejected: int = 0
    timeouts: int = 0
    errors: int = 0
    llm_deduped: int = 0


async def _safe_call_writer(
    writer: Callable[..., Awaitable[None]],
    result: Any,
    fencing_generation: int,
) -> None:
    """Invoke result_writer with fencing_generation if accepted, falling back to 1-arg."""
    import inspect

    sig = inspect.signature(writer)
    if len(sig.parameters) >= 2 or any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    ):
        await writer(result, fencing_generation)  # type: ignore[call-arg]
    else:
        await writer(result)


def build_flight_key(
    pipeline_version: str,
    scope: str,
    tenant_id: str,
    resource_id: str | None,
    resource_version: int | None,
    prompt: str,
    key_prefix: str = "sm:lock",
) -> str:
    """Canonical single-flight lock key generator matching production orchestrator logic."""
    import hashlib

    query_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    res_ver_str = str(resource_version) if resource_version is not None else "_"
    res_id_str = resource_id if resource_id is not None else "_"
    return (
        f"{key_prefix}:{pipeline_version}:{scope}:"
        f"{tenant_id}:{res_id_str}:{res_ver_str}:{query_hash}"
    )


class RedisSingleFlightManager:
    """Distributed single-flight coordinator using atomic Redis SET NX PX locks.

    Guarantees:
    - Exactly one worker executes the leader operation per flight identity across all nodes.
    - Locks expire automatically (PX) to prevent permanent deadlocks if a leader crashes.
    - Fencing tokens (monotonic generation counter) prevent stale leaders from mutating cache.
    - Locks are released safely using an atomic Lua script verifying the owner token.
    - Followers poll the authoritative scoped cache store as the source of truth,
      using Redis Pub/Sub strictly as an opportunistic low-latency wake-up mechanism.
    - Independent cache-write authorization is preserved; followers never mutate the cache.
    """

    @staticmethod
    def build_flight_key(
        pipeline_version: str,
        scope: str,
        tenant_id: str,
        resource_id: str | None,
        resource_version: int | None,
        prompt: str,
        key_prefix: str = "sm:lock",
    ) -> str:
        """Canonical single-flight lock key generator matching production orchestrator logic."""
        return build_flight_key(
            pipeline_version=pipeline_version,
            scope=scope,
            tenant_id=tenant_id,
            resource_id=resource_id,
            resource_version=resource_version,
            prompt=prompt,
            key_prefix=key_prefix,
        )

    def __init__(
        self,
        client: Any | None = None,
        config: RedisConfig | None = None,
    ) -> None:
        if client is None and not REDIS_AVAILABLE:
            raise MissingDependencyError(
                "redis is required for RedisSingleFlightManager. "
                "Install with `pip install semanticmemo[redis]` or `pip install redis`."
            )

        self.config = config or RedisConfig()
        self.metrics = RedisFlightMetrics()
        self._external_client = client is not None
        self._client: Any = None
        if client is not None:
            self._client = client
        else:
            if aioredis is None:
                raise MissingDependencyError("redis.asyncio is unavailable.")
            self._client = aioredis.from_url(self.config.url, decode_responses=True)

    async def close(self) -> None:
        """Close Redis connection if created internally."""
        if not self._external_client and hasattr(self._client, "aclose"):
            await self._client.aclose()

    async def execute(
        self,
        flight_key: str,
        operation: Callable[[], Awaitable[T]],
        *,
        result_loader: Callable[[], Awaitable[T | None]],
        result_writer: Callable[[T], Awaitable[None]] | None = None,
    ) -> T:
        """Execute operation as distributed leader or await and load cached result as follower."""
        # 0. Fast-path check: if result is already cached, return immediately
        if result_loader is not None:
            try:
                cached = await result_loader()
                if cached is not None:
                    self.metrics.llm_deduped += 1
                    return cached
            except Exception as load_exc:
                logger.debug("Initial cache check error in single-flight: %s", load_exc)

        token = str(uuid4())
        flight_gen_key = f"sm:gen:{flight_key}"

        # 1. Attempt to acquire distributed lock as leader
        try:
            acquired = await self._client.set(
                flight_key,
                token,
                nx=True,
                px=self.config.lock_ttl_ms,
            )
        except Exception as exc:
            self.metrics.errors += 1
            logger.error(
                "Redis connection error during lock acquisition for %s: %s", flight_key, exc
            )
            raise

        if acquired:
            # === LEADER PATH ===
            # Double-check result loader in case a previous leader just finished writing
            if result_loader is not None:
                try:
                    cached = await result_loader()
                    if cached is not None:
                        self.metrics.llm_deduped += 1
                        with contextlib.suppress(Exception):
                            await self._client.eval(RELEASE_LUA, 1, flight_key, token)
                        return cached
                except Exception as dc_exc:
                    logger.debug("Double-check cache error in single-flight: %s", dc_exc)

            self.metrics.acquired += 1
            fencing_token: int = 1
            try:
                fencing_token = await self._client.incr(flight_gen_key)
            except Exception as gen_exc:
                logger.debug("Failed to increment fencing generation: %s", gen_exc)

            try:
                result = await operation()

                # Stale leader check: verify our fencing generation is still current
                current_gen_str = await self._client.get(flight_gen_key)
                if current_gen_str and int(current_gen_str) > fencing_token:
                    self.metrics.stale_leader_writes_rejected += 1
                    logger.warning(
                        "Redis CAS write rejected: gen %d < %s (stale write skipped)",
                        fencing_token,
                        current_gen_str,
                    )
                elif result_writer is not None:
                    await _safe_call_writer(result_writer, result, fencing_token)

                # Publish opportunistic completion notification to followers
                if self.config.use_pubsub:
                    try:
                        pubsub_channel = f"sm:pubsub:{flight_key}"
                        payload = json.dumps({"status": "completed"})
                        await self._client.publish(pubsub_channel, payload)
                    except Exception as pub_exc:
                        logger.debug("Failed to publish pubsub completion signal: %s", pub_exc)

                return result
            finally:
                # Safe compare-and-delete lock release via Lua script
                try:
                    res = await self._client.eval(RELEASE_LUA, 1, flight_key, token)
                    if res:
                        self.metrics.owner_releases += 1
                except Exception as rel_exc:
                    logger.debug("Error releasing lock for %s: %s", flight_key, rel_exc)

        # === FOLLOWER PATH ===
        self.metrics.waited += 1
        return await self._wait_as_follower(
            flight_key=flight_key,
            token=token,
            operation=operation,
            result_loader=result_loader,
            result_writer=result_writer,
        )

    async def _wait_as_follower(
        self,
        flight_key: str,
        token: str,
        operation: Callable[[], Awaitable[T]],
        result_loader: Callable[[], Awaitable[T | None]],
        result_writer: Callable[[T], Awaitable[None]] | None,
    ) -> T:
        """Wait for leader to complete while polling the authoritative cache store."""
        start_time = time.monotonic()
        timeout_sec = self.config.wait_timeout_ms / 1000.0
        poll_interval = self.config.poll_interval_ms / 1000.0
        flight_gen_key = f"sm:gen:{flight_key}"

        pubsub = None
        if self.config.use_pubsub and hasattr(self._client, "pubsub"):
            try:
                pubsub = self._client.pubsub()
                await pubsub.subscribe(f"sm:pubsub:{flight_key}")
            except Exception as ps_exc:
                logger.debug("Could not initialize dedicated pubsub subscriber: %s", ps_exc)
                pubsub = None

        try:
            while True:
                # 1. Authoritative cache check
                cached = await result_loader()
                if cached is not None:
                    self.metrics.llm_deduped += 1
                    return cached

                # 2. Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout_sec:
                    self.metrics.timeouts += 1
                    logger.warning("Single flight timeout waiting for key %s", flight_key)
                    # Fallback to direct execution
                    return await operation()

                # 3. Check if leader crashed / lock expired in Redis
                try:
                    current_lock = await self._client.get(flight_key)
                    if current_lock is None:
                        # Lock is absent (expired or released) -> check if leader wrote result
                        cached = await result_loader()
                        if cached is not None:
                            self.metrics.llm_deduped += 1
                            return cached

                        # Attempt takeover as new leader
                        takeover = await self._client.set(
                            flight_key,
                            token,
                            nx=True,
                            px=self.config.lock_ttl_ms,
                        )
                        if takeover:
                            # Double-check cache after acquiring takeover lock
                            cached = await result_loader()
                            if cached is not None:
                                self.metrics.llm_deduped += 1
                                with contextlib.suppress(Exception):
                                    await self._client.eval(RELEASE_LUA, 1, flight_key, token)
                                return cached

                            self.metrics.lock_expired_takeovers += 1
                            takeover_gen = await self._client.incr(flight_gen_key)
                            logger.info(
                                "Acquired lock for %s following leader expiration (gen=%d)",
                                flight_key,
                                takeover_gen,
                            )
                            try:
                                result = await operation()
                                current_gen_str = await self._client.get(flight_gen_key)
                                if current_gen_str and int(current_gen_str) > takeover_gen:
                                    self.metrics.stale_leader_writes_rejected += 1
                                    logger.warning(
                                        "Takeover leader CAS write rejected: gen %d < %s",
                                        takeover_gen,
                                        current_gen_str,
                                    )
                                elif result_writer is not None:
                                    await _safe_call_writer(result_writer, result, takeover_gen)
                                if self.config.use_pubsub:
                                    with contextlib.suppress(Exception):
                                        await self._client.publish(
                                            f"sm:pubsub:{flight_key}",
                                            json.dumps({"status": "completed"}),
                                        )
                                return result
                            finally:
                                with contextlib.suppress(Exception):
                                    await self._client.eval(RELEASE_LUA, 1, flight_key, token)
                except Exception as get_exc:
                    logger.debug("Error checking lock status: %s", get_exc)

                # 4. Wait for Pub/Sub wake-up or sleep polling interval
                if pubsub is not None:
                    try:
                        msg = await pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=poll_interval,
                        )
                        if msg is not None:
                            # Wake up and recheck authoritative cache in next loop iteration
                            continue
                    except Exception:
                        await asyncio.sleep(poll_interval)
                else:
                    await asyncio.sleep(poll_interval)
        finally:
            if pubsub is not None:
                with contextlib.suppress(Exception):
                    await pubsub.unsubscribe(f"sm:pubsub:{flight_key}")
                with contextlib.suppress(Exception):
                    if hasattr(pubsub, "aclose"):
                        await pubsub.aclose()
                    elif hasattr(pubsub, "close"):
                        res = pubsub.close()
                        if inspect.isawaitable(res):
                            await res
