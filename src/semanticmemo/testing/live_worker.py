"""Lightweight OS Worker for Distributed Redis Single-Flight Live Certification.

Strictly imports ONLY standard library and live network client modules:
- asyncio, json, logging, os, sys, time
- redis.asyncio
- semanticmemo.concurrency.redis_flight (RedisConfig, RedisSingleFlightManager, build_flight_key)
- semanticmemo.store.postgres_store (PostgresCacheStore, PostgresConfig)

ABSOLUTELY NO torch, scipy, sklearn, sentence_transformers, or numpy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

from semanticmemo.concurrency.redis_flight import (
    RedisConfig,
    RedisSingleFlightManager,
)
from semanticmemo.store.postgres_store import PostgresCacheStore, PostgresConfig

logger = logging.getLogger(__name__)


def worker_main(
    worker_id: str,
    redis_url: str,
    postgres_dsn: str,
    run_id: str,
    request_specs: list[dict[str, Any]],
) -> None:
    """Entry point for OS child processes executing requests against live Redis & PostgreSQL."""

    async def _run() -> None:
        r_client: Any = aioredis.from_url(redis_url, decode_responses=True)
        config = RedisConfig(
            url=redis_url,
            lock_ttl_ms=10000,
            wait_timeout_ms=10000,
            poll_interval_ms=15,
            use_pubsub=False,
        )
        flight_mgr = RedisSingleFlightManager(client=r_client, config=config)
        pg_store = PostgresCacheStore(config=PostgresConfig(dsn=postgres_dsn))

        for spec in request_specs:
            prompt: str = spec["prompt"]
            tenant_id: str = spec["tenant_id"]
            resource_id: str | None = spec.get("resource_id")
            resource_version: int | None = spec.get("resource_version")
            scope: str = spec.get("scope", "global")
            pipeline_version: str = spec.get("pipeline_version", "v2")
            res_id_str = resource_id if resource_id is not None else "_"

            # Use the exact production flight key generation path
            flight_key = RedisSingleFlightManager.build_flight_key(
                pipeline_version=pipeline_version,
                scope=scope,
                tenant_id=tenant_id,
                resource_id=resource_id,
                resource_version=resource_version,
                prompt=prompt,
            )

            async def loader(
                p=prompt,
                t=tenant_id,
                s=scope,
                r=resource_id,
                rv=resource_version,
                pv=pipeline_version,
            ) -> str | None:
                entry = pg_store.find_by_prompt(
                    prompt=p,
                    tenant_id=t,
                    scope=s,
                    resource_id=r,
                    resource_version=rv,
                    pipeline_version=pv,
                )
                return entry.response if entry else None

            async def writer(
                val: str,
                gen: int = 0,
                p=prompt,
                t=tenant_id,
                s=scope,
                r=resource_id,
                rv=resource_version,
                pv=pipeline_version,
            ) -> None:
                # Store authoritative result into PostgreSQL using raw float list (no numpy import)
                pg_store.add(
                    prompt=p,
                    embedding=[0.1, 0.1, 0.1, 0.1],
                    response=val,
                    model="live-os-model",
                    tenant_id=t,
                    scope=s,
                    resource_id=r,
                    resource_version=rv,
                    pipeline_version=pv,
                    flight_generation=gen,
                )

            async def llm_op(p=prompt, t=tenant_id, r_str=res_id_str) -> str:
                # Authoritative Redis counter increments ONLY during true producer/leader execution
                await r_client.incr(f"live:test:{run_id}:llm_calls")
                # Simulate deterministic LLM generation latency
                await asyncio.sleep(0.08)
                return f"Authoritative Live Result for {p} [tenant={t}] [resource={r_str}]"

            res = await flight_mgr.execute(
                flight_key=flight_key,
                operation=llm_op,
                result_loader=loader,
                result_writer=writer,
            )

            record = {
                "worker_id": worker_id,
                "pid": os.getpid(),
                "tenant_id": tenant_id,
                "resource_id": res_id_str,
                "prompt": prompt,
                "flight_key": flight_key,
                "result": res,
            }
            await r_client.rpush(f"live:test:{run_id}:results", json.dumps(record))

        pg_store.close()
        await r_client.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
