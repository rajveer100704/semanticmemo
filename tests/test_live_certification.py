"""SemanticMemo V2 — Live Infrastructure & Distributed Process Certification Suite.

Validates the full architecture against real PostgreSQL, Redis, Qdrant, and Kafka
running in Docker, across true independent OS processes.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import redis

from semanticmemo.concurrency.redis_flight import RedisConfig, RedisSingleFlightManager
from semanticmemo.embedding.qdrant_index import QdrantConfig, QdrantVectorIndex
from semanticmemo.embedding.service import EmbeddingService, HashEmbeddingProvider
from semanticmemo.events.bus import KafkaEventBus
from semanticmemo.events.consumer import InvalidationConsumer
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.events.reconciliation import IndexReconciler
from semanticmemo.events.relay import OutboxRelay
from semanticmemo.models import (
    CacheScope,
    PermissionName,
    RetrievalScope,
)
from semanticmemo.store.postgres_store import PostgresCacheStore, PostgresConfig
from semanticmemo.telemetry.metrics import SemanticMemoMetrics
from semanticmemo.telemetry.tracing import sanitize_attributes

logger = logging.getLogger(__name__)

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://postgres:postgres_dev_password@127.0.0.1:5432/semanticmemo"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")


# ---------------------------------------------------------------------------
# Phase 2: Live Dependency Health Checks Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def verify_live_infrastructure():
    """Session fixture ensuring all 4 live services are up and healthy.

    If LIVE_CERTIFICATION=1, unreachable infrastructure triggers pytest.fail.
    If LIVE_CERTIFICATION is unset, unreachable infrastructure triggers pytest.skip
    so that normal CI without Docker services skips cleanly without false failures.
    """
    strict_certification = os.environ.get("LIVE_CERTIFICATION", "").strip() == "1"

    def _handle_unreachable(service_name: str, endpoint: str, exc: Exception):
        msg = f"{service_name} not reachable at {endpoint}: {exc}"
        if strict_certification:
            pytest.fail(f"[STRICT CERTIFICATION FAILURE] {msg}")
        else:
            pytest.skip(f"[CI SKIP: live infrastructure unavailable] {msg}")

    # 1. PostgreSQL
    try:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, connect_timeout=10) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            row = cur.fetchone()
            assert row is not None and row[0] == 1
    except Exception as exc:
        _handle_unreachable("PostgreSQL", POSTGRES_DSN, exc)

    # 2. Redis
    try:
        import redis

        r = redis.Redis.from_url(REDIS_URL, socket_timeout=10)
        assert r.ping() is True
        r.close()
    except Exception as exc:
        _handle_unreachable("Redis", REDIS_URL, exc)

    # 3. Qdrant
    try:
        from qdrant_client import QdrantClient

        qc = QdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
        qc.get_collections()
        qc.close()
    except Exception as exc:
        _handle_unreachable("Qdrant", QDRANT_URL, exc)

    # 4. Kafka
    try:
        import socket

        host, port_str = KAFKA_BOOTSTRAP.split(":")
        s = socket.create_connection((host, int(port_str)), timeout=10)
        s.close()
    except Exception as exc:
        _handle_unreachable("Kafka", KAFKA_BOOTSTRAP, exc)


# ---------------------------------------------------------------------------
# Phase 4: Live RBAC Certification
# ---------------------------------------------------------------------------
def test_phase4_live_rbac_certification():
    """Verify live PostgreSQL RBAC isolation, spoofing rejection, and permission revocation."""
    store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))

    tenant_a = f"tenant_a_{uuid4().hex[:6]}"
    tenant_b = f"tenant_b_{uuid4().hex[:6]}"

    # Create users
    user_a_admin = f"user_a_admin_{uuid4().hex[:6]}"
    user_a_viewer = f"user_a_viewer_{uuid4().hex[:6]}"
    user_b_viewer = f"user_b_viewer_{uuid4().hex[:6]}"

    store.create_user(user_a_admin, tenant_a)
    store.create_user(user_a_viewer, tenant_a)
    store.create_user(user_b_viewer, tenant_b)

    # Assign roles
    store.assign_role_to_user(user_a_admin, "admin")
    store.assign_role_to_user(user_a_viewer, "viewer")
    store.assign_role_to_user(user_b_viewer, "viewer")

    # Create resources
    res_a = store.create_resource(f"res_a_{uuid4().hex[:6]}", tenant_id=tenant_a)
    res_b = store.create_resource(f"res_b_{uuid4().hex[:6]}", tenant_id=tenant_b)

    # Grant viewer role read access to res_a in tenant_a
    store.grant_resource_access(res_a.id, "viewer", PermissionName.CACHE_READ.value)
    store.grant_resource_access(res_b.id, "viewer", PermissionName.CACHE_READ.value)

    # 1. user_a can access res_a
    assert (
        store.check_user_permission(
            user_a_viewer, PermissionName.CACHE_READ.value, resource_id=res_a.id
        )
        is True
    )

    # 2. user_a cannot access res_b (cross-tenant rejection)
    assert (
        store.check_user_permission(
            user_a_viewer, PermissionName.CACHE_READ.value, resource_id=res_b.id
        )
        is False
    )

    # 3. user_b cannot access res_a
    assert (
        store.check_user_permission(
            user_b_viewer, PermissionName.CACHE_READ.value, resource_id=res_a.id
        )
        is False
    )

    # 4. Permission revocation prevents subsequent cache access
    store.remove_role_from_user(user_a_viewer, "viewer")
    assert (
        store.check_user_permission(
            user_a_viewer, PermissionName.CACHE_READ.value, resource_id=res_a.id
        )
        is False
    )

    # Clean up
    store.delete_user(user_a_admin)
    store.delete_user(user_a_viewer)
    store.delete_user(user_b_viewer)
    store.close()


# ---------------------------------------------------------------------------
# Phase 5: Live Qdrant Scope Certification
# ---------------------------------------------------------------------------
def test_phase5_live_qdrant_scope_certification():
    """Verify live Qdrant scoped retrieval across scopes, versions, and pipeline versions."""
    col_name = f"cert_qdrant_{uuid4().hex[:8]}"
    config = QdrantConfig(url=QDRANT_URL, collection_name=col_name, timeout=30)
    index = QdrantVectorIndex(dim=4, config=config)

    v_identical = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Insert vectors with identical embeddings across different scopes/tenants/versions
    # 1. Tenant A Resource A Version 1 Pipeline v2
    id_ta_ra_v1 = uuid4()
    index.add(
        entry_id=id_ta_ra_v1,
        embedding=v_identical,
        tenant_id="tenant_alpha",
        scope=CacheScope.RESOURCE,
        resource_id="res_alpha",
        resource_version=1,
        pipeline_version="v2",
    )

    # 2. Tenant B Resource B Version 1 Pipeline v2
    id_tb_rb_v1 = uuid4()
    index.add(
        entry_id=id_tb_rb_v1,
        embedding=v_identical,
        tenant_id="tenant_beta",
        scope=CacheScope.RESOURCE,
        resource_id="res_beta",
        resource_version=1,
        pipeline_version="v2",
    )

    # 3. Tenant A Resource A Version 2 (new version) Pipeline v2
    id_ta_ra_v2 = uuid4()
    index.add(
        entry_id=id_ta_ra_v2,
        embedding=v_identical,
        tenant_id="tenant_alpha",
        scope=CacheScope.RESOURCE,
        resource_id="res_alpha",
        resource_version=2,
        pipeline_version="v2",
    )

    # 4. Tenant A Tenant Scope Pipeline v2
    id_ta_tenant = uuid4()
    index.add(
        entry_id=id_ta_tenant,
        embedding=v_identical,
        tenant_id="tenant_alpha",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )

    # 5. Global Scope Pipeline v2
    id_global = uuid4()
    index.add(
        entry_id=id_global,
        embedding=v_identical,
        tenant_id="default",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )

    # --- Verification 1: RESOURCE Scope (tenant_alpha, res_alpha, v1, pipe v2) ---
    res1 = index.search(
        v_identical,
        k=10,
        retrieval_scope=RetrievalScope(
            tenant_id="tenant_alpha",
            scope=CacheScope.RESOURCE,
            resource_id="res_alpha",
            resource_version=1,
            pipeline_version="v2",
        ),
    )
    assert len(res1) == 1
    assert res1[0].entry_id == id_ta_ra_v1

    # --- Verification 2: RESOURCE Version Invalidation (querying v2 returns only v2) ---
    res2 = index.search(
        v_identical,
        k=10,
        retrieval_scope=RetrievalScope(
            tenant_id="tenant_alpha",
            scope=CacheScope.RESOURCE,
            resource_id="res_alpha",
            resource_version=2,
            pipeline_version="v2",
        ),
    )
    assert len(res2) == 1
    assert res2[0].entry_id == id_ta_ra_v2

    # --- Verification 3: Pipeline Version Invalidation (querying v2.1 returns nothing) ---
    res3 = index.search(
        v_identical,
        k=10,
        retrieval_scope=RetrievalScope(
            tenant_id="tenant_alpha",
            scope=CacheScope.RESOURCE,
            resource_id="res_alpha",
            resource_version=2,
            pipeline_version="v2.1",
        ),
    )
    assert len(res3) == 0

    # --- Verification 4: TENANT Scope (returns TENANT entries + GLOBAL entries) ---
    res4 = index.search(
        v_identical,
        k=10,
        retrieval_scope=RetrievalScope(
            tenant_id="tenant_alpha",
            scope=CacheScope.TENANT,
            pipeline_version="v2",
        ),
    )
    matched_ids = {r.entry_id for r in res4}
    assert id_ta_tenant in matched_ids
    assert id_global in matched_ids
    assert id_ta_ra_v1 not in matched_ids
    assert id_tb_rb_v1 not in matched_ids

    # --- Verification 5: GLOBAL Scope (returns only GLOBAL entries) ---
    res5 = index.search(
        v_identical,
        k=10,
        retrieval_scope=RetrievalScope(
            tenant_id="tenant_alpha",
            scope=CacheScope.GLOBAL,
            pipeline_version="v2",
        ),
    )
    assert len(res5) == 1
    assert res5[0].entry_id == id_global


# ---------------------------------------------------------------------------
# Phase 6 & 13: Live Redis Distributed Single-Flight Across 3 OS Processes
# ---------------------------------------------------------------------------
def test_import_safety_lightweight_worker():
    """Verify in an isolated Python subprocess that worker does not load heavy ML libs."""
    import subprocess
    import sys

    check_code = (
        "import sys\n"
        "import tests.live_workers.redis_worker\n"
        "heavy = {'torch', 'scipy', 'sklearn', 'sentence_transformers'}\n"
        "loaded = heavy.intersection(sys.modules)\n"
        "print('LOADED_HEAVY:', sorted(list(loaded)))\n"
        "assert not loaded, f'Heavy ML modules loaded into sys.modules: {loaded}'\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, f"Subprocess import safety check failed:\n{proc.stderr}"
    assert "LOADED_HEAVY: []" in proc.stdout


def test_phase6_phase13_live_redis_3_os_processes_coalescing():
    """Proves 100 requests across 3 OS processes coalesce to authoritative executions."""
    import json

    from tests.live_workers.redis_worker import worker_main

    # 0. Infrastructure availability check
    try:
        r_check = redis.Redis.from_url(REDIS_URL, socket_timeout=3)
        r_check.ping()
        r_check.close()
    except Exception as exc:
        pytest.skip(f"Redis live service unavailable: {exc}")

    try:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
    except Exception as exc:
        pytest.skip(f"PostgreSQL live service unavailable: {exc}")

    r_client: Any = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    ctx = mp.get_context("spawn")

    # Sanitize endpoints for diagnostic output (no passwords)
    sanitized_redis = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    sanitized_pg = POSTGRES_DSN.split("@")[-1] if "@" in POSTGRES_DSN else POSTGRES_DSN

    # -----------------------------------------------------------------------
    # SCENARIO A: 1 Tenant, 1 Prompt, 100 requests across 3 OS processes -> exactly 1 LLM call
    # -----------------------------------------------------------------------
    run_id_a = f"scen_a_{uuid4().hex[:8]}"
    prompt_a = f"Live 3-process stampede {uuid4().hex[:6]}"
    tenant_a = f"tenant_a_{uuid4().hex[:6]}"

    # Verify clean initial state in PostgreSQL
    pg_store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))
    assert pg_store.find_by_prompt(prompt=prompt_a, tenant_id=tenant_a) is None
    pg_store.close()

    # Clean Redis state for run_id_a
    r_client.delete(f"live:test:{run_id_a}:llm_calls", f"live:test:{run_id_a}:results")

    # Worker A (35), Worker B (35), Worker C (30) = 100 total requests
    spec_a = {
        "prompt": prompt_a,
        "tenant_id": tenant_a,
        "scope": "global",
        "pipeline_version": "v2",
    }
    specs_a_worker1 = [spec_a] * 35
    specs_a_worker2 = [spec_a] * 35
    specs_a_worker3 = [spec_a] * 30

    w1 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-A", REDIS_URL, POSTGRES_DSN, run_id_a, specs_a_worker1),
    )
    w2 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-B", REDIS_URL, POSTGRES_DSN, run_id_a, specs_a_worker2),
    )
    w3 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-C", REDIS_URL, POSTGRES_DSN, run_id_a, specs_a_worker3),
    )

    w1.start()
    w2.start()
    w3.start()

    pid_a, pid_b, pid_c = w1.pid, w2.pid, w3.pid

    print("\n" + "=" * 78)
    print("DISTRIBUTED OS PROCESS ENVIRONMENT DIAGNOSTICS")
    print(f"Python Version:             {sys.version.split()[0]}")
    print(f"Platform:                   {sys.platform}")
    print("Multiprocessing Start:      spawn")
    print("Worker Module:              tests.live_workers.redis_worker")
    print(f"Redis Endpoint:             {sanitized_redis}")
    print(f"PostgreSQL Endpoint:        {sanitized_pg}")
    print(f"Worker A PID:               {pid_a}")
    print(f"Worker B PID:               {pid_b}")
    print(f"Worker C PID:               {pid_c}")
    print(f"Distinct OS Processes:      PASS ({pid_a}, {pid_b}, {pid_c})")
    print(f"Scenario A Run ID:          {run_id_a}")
    print("=" * 78)

    w1.join(timeout=120)
    w2.join(timeout=120)
    w3.join(timeout=120)

    raw_results_a = r_client.lrange(f"live:test:{run_id_a}:results", 0, -1)
    llm_calls_a = int(r_client.get(f"live:test:{run_id_a}:llm_calls") or 0)
    r_client.delete(f"live:test:{run_id_a}:llm_calls", f"live:test:{run_id_a}:results")

    # Assert 100 requests returned
    assert len(raw_results_a) == 100, f"Expected 100 results, got {len(raw_results_a)}"

    # CRITICAL PROOF: Exactly 1 LLM execution across all 3 independent OS processes
    assert llm_calls_a == 1, f"Expected exactly 1 LLM execution, got {llm_calls_a}"

    parsed_a = [json.loads(r) for r in raw_results_a]
    observed_pids = {p["pid"] for p in parsed_a}
    assert len(observed_pids) == 3, f"Expected 3 distinct PIDs, got {observed_pids}"
    assert observed_pids == {pid_a, pid_b, pid_c}

    expected_resp_a = f"Authoritative Live Result for {prompt_a} [tenant={tenant_a}] [resource=_]"
    for item in parsed_a:
        assert item["result"] == expected_resp_a
        assert item["tenant_id"] == tenant_a

    print(
        f"Scenario A PASS: 100 requests -> 1 LLM execution "
        f"across 3 distinct OS processes {observed_pids}"
    )

    # -----------------------------------------------------------------------
    # SCENARIO B: 10 Tenants, 100 requests across 3 OS processes -> exactly 10 LLM calls
    # -----------------------------------------------------------------------
    run_id_b = f"scen_b_{uuid4().hex[:8]}"
    prompt_b = f"Live multi-tenant stampede {uuid4().hex[:6]}"
    tenants_b = [f"tenant_b_{i}_{uuid4().hex[:4]}" for i in range(10)]

    r_client.delete(f"live:test:{run_id_b}:llm_calls", f"live:test:{run_id_b}:results")
    # 10 requests per tenant = 100 requests total
    # Distribute across Worker A (35), Worker B (35), Worker C (30)
    all_b_specs = []
    for t in tenants_b:
        for _ in range(10):
            all_b_specs.append(
                {
                    "prompt": prompt_b,
                    "tenant_id": t,
                    "scope": "tenant",
                    "pipeline_version": "v2",
                }
            )

    specs_b_worker1 = all_b_specs[:35]
    specs_b_worker2 = all_b_specs[35:70]
    specs_b_worker3 = all_b_specs[70:]

    wb1 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-A", REDIS_URL, POSTGRES_DSN, run_id_b, specs_b_worker1),
    )
    wb2 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-B", REDIS_URL, POSTGRES_DSN, run_id_b, specs_b_worker2),
    )
    wb3 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-C", REDIS_URL, POSTGRES_DSN, run_id_b, specs_b_worker3),
    )

    wb1.start()
    wb2.start()
    wb3.start()

    wb1.join(timeout=120)
    wb2.join(timeout=120)
    wb3.join(timeout=120)

    raw_results_b = r_client.lrange(f"live:test:{run_id_b}:results", 0, -1)
    llm_calls_b = int(r_client.get(f"live:test:{run_id_b}:llm_calls") or 0)
    r_client.delete(f"live:test:{run_id_b}:llm_calls", f"live:test:{run_id_b}:results")

    assert len(raw_results_b) == 100, f"Expected 100 results, got {len(raw_results_b)}"
    # Exactly 10 LLM executions (1 per tenant)
    assert llm_calls_b == 10, (
        f"Expected exactly 10 LLM executions for 10 tenants, got {llm_calls_b}"
    )

    parsed_b = [json.loads(r) for r in raw_results_b]
    # Verify strict tenant isolation: no tenant receives another tenant's result
    tenant_results_map: dict[str, list[str]] = {}
    for item in parsed_b:
        t_id = item["tenant_id"]
        expected_t_resp = f"Authoritative Live Result for {prompt_b} [tenant={t_id}] [resource=_]"
        assert item["result"] == expected_t_resp, (
            f"Tenant {t_id} received corrupted result {item['result']}"
        )
        tenant_results_map.setdefault(t_id, []).append(item["result"])

    assert len(tenant_results_map) == 10
    for _t_id, res_list in tenant_results_map.items():
        assert len(res_list) == 10

    print("Scenario B PASS: 10 tenants -> exactly 10 LLM executions with complete tenant isolation")

    # -----------------------------------------------------------------------
    # SCENARIO C: 10 Resources, 100 requests across 3 OS processes -> exactly 10 LLM calls
    # -----------------------------------------------------------------------
    run_id_c = f"scen_c_{uuid4().hex[:8]}"
    prompt_c = f"Live multi-resource stampede {uuid4().hex[:6]}"
    tenant_c = f"tenant_c_{uuid4().hex[:6]}"
    resources_c = [f"res_{i}_{uuid4().hex[:4]}" for i in range(10)]

    r_client.delete(f"live:test:{run_id_c}:llm_calls", f"live:test:{run_id_c}:results")

    all_c_specs = []
    for r_id in resources_c:
        for _ in range(10):
            all_c_specs.append(
                {
                    "prompt": prompt_c,
                    "tenant_id": tenant_c,
                    "resource_id": r_id,
                    "resource_version": 1,
                    "scope": "resource",
                    "pipeline_version": "v2",
                }
            )

    specs_c_worker1 = all_c_specs[:35]
    specs_c_worker2 = all_c_specs[35:70]
    specs_c_worker3 = all_c_specs[70:]

    wc1 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-A", REDIS_URL, POSTGRES_DSN, run_id_c, specs_c_worker1),
    )
    wc2 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-B", REDIS_URL, POSTGRES_DSN, run_id_c, specs_c_worker2),
    )
    wc3 = ctx.Process(
        target=worker_main,
        args=("OS-Worker-C", REDIS_URL, POSTGRES_DSN, run_id_c, specs_c_worker3),
    )

    wc1.start()
    wc2.start()
    wc3.start()

    wc1.join(timeout=120)
    wc2.join(timeout=120)
    wc3.join(timeout=120)

    raw_results_c = r_client.lrange(f"live:test:{run_id_c}:results", 0, -1)
    llm_calls_c = int(r_client.get(f"live:test:{run_id_c}:llm_calls") or 0)
    r_client.delete(f"live:test:{run_id_c}:llm_calls", f"live:test:{run_id_c}:results")
    r_client.close()

    assert len(raw_results_c) == 100, f"Expected 100 results, got {len(raw_results_c)}"
    # Exactly 10 LLM executions (1 per resource)
    assert llm_calls_c == 10, (
        f"Expected exactly 10 LLM executions for 10 resources, got {llm_calls_c}"
    )

    parsed_c = [json.loads(r) for r in raw_results_c]
    # Verify strict resource isolation: resource-3 never receives resource-4 result
    resource_results_map: dict[str, list[str]] = {}
    for item in parsed_c:
        r_id = item["resource_id"]
        expected_r_resp = (
            f"Authoritative Live Result for {prompt_c} [tenant={tenant_c}] [resource={r_id}]"
        )
        assert item["result"] == expected_r_resp, (
            f"Resource {r_id} received corrupted result {item['result']}"
        )
        resource_results_map.setdefault(r_id, []).append(item["result"])

    assert len(resource_results_map) == 10
    for _r_id, res_list in resource_results_map.items():
        assert len(res_list) == 10

    print(
        "Scenario C PASS: 10 resources -> exactly 10 LLM executions "
        "with complete resource isolation"
    )


# ---------------------------------------------------------------------------
# Phase 7: Real Redis + PostgreSQL Fencing Test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_phase7_real_redis_postgres_fencing():
    """Verify CAS rejection in PostgreSQL when a stalled leader attempts write."""
    import redis.asyncio as aioredis

    r_client_a = aioredis.from_url(REDIS_URL)
    r_client_b = aioredis.from_url(REDIS_URL)

    # 100ms lock TTL
    config = RedisConfig(
        url=REDIS_URL,
        lock_ttl_ms=100,
        wait_timeout_ms=1000,
        poll_interval_ms=10,
        use_pubsub=False,
    )
    mgr_a = RedisSingleFlightManager(client=r_client_a, config=config)
    mgr_b = RedisSingleFlightManager(client=r_client_b, config=config)

    store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))
    prompt = f"Fencing Prompt {uuid4().hex[:6]}"
    tenant_id = f"tenant_fence_{uuid4().hex[:6]}"
    flight_key = f"sm:flight:{tenant_id}:global:v2:{uuid4().hex[:8]}"

    leader_a_started = asyncio.Event()

    async def op_a() -> str:
        leader_a_started.set()
        # Sleep 250ms -> lease of 100ms expires!
        await asyncio.sleep(0.25)
        return "Stale Result A from expired Leader A"

    async def op_b() -> str:
        return "Authoritative Result B from takeover Leader B"

    async def loader() -> str | None:
        entry = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
        return entry.response if entry else None

    async def writer_a(resp: str, gen: int = 0) -> None:
        store.add(
            prompt=prompt,
            embedding=np.array([0.1] * 4, dtype=np.float32),
            response=resp,
            model="model-a",
            tenant_id=tenant_id,
            flight_generation=gen,
        )

    async def writer_b(resp: str, gen: int = 0) -> None:
        store.add(
            prompt=prompt,
            embedding=np.array([0.1] * 4, dtype=np.float32),
            response=resp,
            model="model-b",
            tenant_id=tenant_id,
            flight_generation=gen,
        )

    async def run_worker_a():
        return await mgr_a.execute(
            flight_key=flight_key,
            operation=op_a,
            result_loader=loader,
            result_writer=writer_a,
        )

    async def run_worker_b():
        await leader_a_started.wait()
        # Wait 120ms for Leader A's 100ms lease to expire
        await asyncio.sleep(0.12)
        return await mgr_b.execute(
            flight_key=flight_key,
            operation=op_b,
            result_loader=loader,
            result_writer=writer_b,
        )

    res_a, res_b = await asyncio.gather(run_worker_a(), run_worker_b())

    # Leader B took over and wrote Result B (gen 2)
    assert res_b == "Authoritative Result B from takeover Leader B"

    # PostgreSQL authoritative store MUST have Result B and NOT Stale Result A
    cached = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
    assert cached is not None
    assert cached.response == "Authoritative Result B from takeover Leader B"
    assert cached.flight_generation == 2

    store.close()
    await r_client_a.aclose()
    await r_client_b.aclose()


# ---------------------------------------------------------------------------
# Phase 8: Live PostgreSQL Transactional Outbox
# ---------------------------------------------------------------------------
def test_phase8_live_postgres_transactional_outbox():
    """Verify atomic state mutation + outbox record in PostgreSQL."""
    store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))
    tenant_id = f"tenant_outbox_{uuid4().hex[:6]}"
    res_id = f"res_outbox_{uuid4().hex[:6]}"

    res = store.create_resource(res_id, tenant_id=tenant_id)
    assert res.current_version == 1

    # Atomic increment writes state mutation
    updated_ver = store.increment_resource_version(res_id, tenant_id=tenant_id)
    assert updated_ver == 2

    # Record transactional outbox event
    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id=res_id,
        tenant_id=tenant_id,
        payload={"new_version": 2},
    )
    store.record_outbox_event(event)

    # Check that outbox event exists in database
    pending_events = store.claim_outbox_events(limit=1000, claim_owner="cert_tester")
    matching = [e for e in pending_events if e.aggregate_id == res_id]
    assert len(matching) >= 1
    assert matching[0].payload.get("new_version") == 2

    store.close()


# ---------------------------------------------------------------------------
# Phase 9: Live Kafka Outbox Relay & Idempotent Consumer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_phase9_live_kafka_outbox_relay_and_idempotency():
    """Verify live Kafka publication, delivery, and idempotent consumer deduplication."""
    store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))

    topic_prefix = f"sm.cert.{uuid4().hex[:6]}"
    kafka_bus = KafkaEventBus(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        topic_prefix=topic_prefix,
        group_id=f"cert_group_{uuid4().hex[:6]}",
    )

    # Initialize live aiokafka producer inside active event loop
    import aiokafka

    producer = aiokafka.AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        request_timeout_ms=10000,
    )
    await producer.start()
    kafka_bus._producer = producer

    # Record outbox event in PostgreSQL
    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id=f"res_kafka_{uuid4().hex[:6]}",
        tenant_id="tenant_kafka",
        payload={"version": 2},
    )
    store.record_outbox_event(event)

    relay = OutboxRelay(
        store=store,
        event_bus=kafka_bus,
        relay_id="live_cert_relay",
        claim_lease_seconds=30.0,
    )

    # Relay claims and publishes to Kafka
    published_count = await relay.run_once(batch_size=5)
    assert published_count >= 1

    # Invalidation consumer idempotency verification
    execution_count = 0

    def on_res_update(evt: OutboxEvent):
        nonlocal execution_count
        execution_count += 1

    consumer = InvalidationConsumer(
        consumer_name="live_cert_consumer",
        processed_store=store,
        on_resource_updated=on_res_update,
    )

    # First delivery
    first_res = await consumer.handle_event(event)
    assert first_res is True
    assert execution_count == 1

    # Second delivery (duplicate replay)
    second_res = await consumer.handle_event(event)
    assert second_res is False
    assert execution_count == 1  # Exactly 1 execution guaranteed

    await producer.stop()
    store.close()


# ---------------------------------------------------------------------------
# Phase 10: Live Qdrant Reconciliation
# ---------------------------------------------------------------------------
def test_phase10_live_qdrant_reconciliation():
    """Verify reconciliation restores missing vectors against live Qdrant."""
    col_name = f"reconcile_cert_{uuid4().hex[:8]}"
    q_config = QdrantConfig(url=QDRANT_URL, collection_name=col_name, timeout=30)
    q_index = QdrantVectorIndex(dim=4, config=q_config)
    provider = HashEmbeddingProvider(dim=4)
    emb_service = EmbeddingService(provider=provider, index=q_index)
    store = PostgresCacheStore(config=PostgresConfig(dsn=POSTGRES_DSN))

    tenant_id = f"tenant_rec_{uuid4().hex[:6]}"
    res_id = f"res_rec_{uuid4().hex[:6]}"

    # Resource at version 18 in PostgreSQL
    store.create_resource(res_id, tenant_id=tenant_id)
    for _ in range(17):
        store.increment_resource_version(res_id, tenant_id=tenant_id)
    cur_res = store.get_resource(res_id, tenant_id=tenant_id)
    assert cur_res is not None and cur_res.current_version == 18

    # Cache entry at version 18 in PostgreSQL, but missing in Qdrant
    entry_v18_id = store.add(
        prompt="Reconcile prompt v18",
        embedding=np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        response="Response v18",
        model="gpt-4",
        tenant_id=tenant_id,
        scope=CacheScope.RESOURCE,
        resource_id=res_id,
        resource_version=18,
        pipeline_version="v2",
    )
    assert entry_v18_id is not None

    # Run index reconciler
    reconciler = IndexReconciler(
        metadata_store=store,
        embedding_service=emb_service,
    )
    report = reconciler.reconcile(tenant_id=tenant_id, resource_id=res_id)
    assert report.missing_repaired >= 1

    # Verify vector v18 is now searchable in Qdrant
    search_res = emb_service.search(
        np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        k=5,
        retrieval_scope=RetrievalScope(
            tenant_id=tenant_id,
            scope=CacheScope.RESOURCE,
            resource_id=res_id,
            resource_version=18,
            pipeline_version="v2",
        ),
    )
    assert len(search_res) >= 1
    assert search_res[0].entry_id == entry_v18_id

    # Create a stale entry v17 in store
    entry_v17_id = store.add(
        prompt="Stale prompt v17",
        embedding=np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        response="Stale Response v17",
        model="gpt-4",
        tenant_id=tenant_id,
        scope=CacheScope.RESOURCE,
        resource_id=res_id,
        resource_version=17,
        pipeline_version="v2",
    )
    assert entry_v17_id is not None

    # Reconcile again: v17 must be skipped by the version guard
    report2 = reconciler.reconcile(tenant_id=tenant_id, resource_id=res_id)
    assert report2.stale_versions_skipped >= 1

    store.close()


# ---------------------------------------------------------------------------
# Phase 15 & 16: Telemetry Sanitization & Resilience
# ---------------------------------------------------------------------------
def test_phase15_phase16_telemetry_sanitization_and_resilience():
    """Verify metrics and trace attributes contain no raw PII/prompts."""
    # 1. Attribute sanitization
    raw_attrs = {
        "tenant_id": "tenant_secret_corp",
        "user_id": "alice_ceo",
        "resource_id": "confidential_deal_room",
        "prompt": "What was our secret EBITDA in Q3?",
        "api_key": "sk-ant-live-1234567890",
        "safe_metric": 42,
    }
    sanitized = sanitize_attributes(raw_attrs)

    # Sensitive strings must be hashed or removed, never plaintext
    assert sanitized["semanticmemo.safe_metric"] == 42
    assert "sk-ant-live" not in str(sanitized)
    assert "What was our secret" not in str(sanitized)
    assert "alice_ceo" not in str(sanitized)

    # 2. Resilient best-effort behavior
    metrics = SemanticMemoMetrics.default()
    # Recording metrics directly to counters/histograms
    labels = {
        "scope": "resource",
        "decision": "hit",
        "domain": "finance",
        "backend": "postgres",
    }
    metrics.cache_requests_total.inc(labels=labels)
    metrics.request_duration_seconds.observe(0.015, labels={"stage": "total", "scope": "resource"})
    metrics.cost_saved_usd_total.inc(0.003, labels={"scope": "resource", "domain": "finance"})

    # Export metrics text and verify format
    exported = metrics.export_openmetrics_text()
    assert "semanticmemo_cache_requests_total" in exported
    assert "semanticmemo_cost_saved_usd_total" in exported
    assert "# EOF" in exported


# ---------------------------------------------------------------------------
# Phase 17: Live PostgreSQL Row-Level Security (RLS) Certification
# ---------------------------------------------------------------------------
def test_phase17_live_postgresql_rls_certification():
    """Verify PostgreSQL RLS isolation when connecting as non-owner application role."""
    import psycopg

    # Connect as postgres superuser to configure RLS and app_user role
    with psycopg.connect(POSTGRES_DSN) as admin_conn, admin_conn.cursor() as cur:
        # Enable RLS on cache_entries and resources
        cur.execute("ALTER TABLE cache_entries ENABLE ROW LEVEL SECURITY;")
        cur.execute("ALTER TABLE resources ENABLE ROW LEVEL SECURITY;")

        # Create RLS policies
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_cache'
                ) THEN
                    CREATE POLICY tenant_isolation_cache ON cache_entries
                    FOR ALL
                    TO public
                    USING (tenant_id = current_setting('app.current_tenant', true));
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_resources'
                ) THEN
                    CREATE POLICY tenant_isolation_resources ON resources
                    FOR ALL
                    TO public
                    USING (tenant_id = current_setting('app.current_tenant', true));
                END IF;
            END $$;
            """
        )

        # Create non-owner app_user if not exists
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_roles WHERE rolname = 'semanticmemo_app_user'
                ) THEN
                    CREATE ROLE semanticmemo_app_user WITH LOGIN PASSWORD 'app_user_password';
                END IF;
            END $$;
            """
        )
        cur.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO semanticmemo_app_user;")
        cur.execute(
            "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO semanticmemo_app_user;"
        )
        admin_conn.commit()

    # Now connect strictly as non-owner semanticmemo_app_user
    app_dsn = "postgresql://semanticmemo_app_user:app_user_password@127.0.0.1:5432/semanticmemo"
    with psycopg.connect(app_dsn) as app_conn, app_conn.cursor() as cur:
        # 1. Set tenant to tenant_rls_a
        cur.execute("SET app.current_tenant = 'tenant_rls_a';")
        cur.execute(
            "INSERT INTO resources (id, tenant_id, current_version, status, updated_at) "
            "VALUES ('res_rls_a', 'tenant_rls_a', 1, 'active', NOW()) "
            "ON CONFLICT (id) DO NOTHING;"
        )
        app_conn.commit()

        # Verify tenant_rls_a can see its own resource
        cur.execute("SELECT id FROM resources WHERE id = 'res_rls_a';")
        rows = cur.fetchall()
        assert len(rows) == 1

        # 2. Switch tenant to tenant_rls_b
        cur.execute("SET app.current_tenant = 'tenant_rls_b';")
        # Must NOT be able to see tenant_rls_a resource
        cur.execute("SELECT id FROM resources WHERE id = 'res_rls_a';")
        rows = cur.fetchall()
        assert len(rows) == 0

        # Must NOT be able to update tenant_rls_a resource
        cur.execute("UPDATE resources SET current_version = 99 WHERE id = 'res_rls_a';")
        assert cur.rowcount == 0
        app_conn.commit()
