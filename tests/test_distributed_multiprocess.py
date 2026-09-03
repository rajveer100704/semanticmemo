"""Tests for multi-process concurrency and stampede coalescing across OS processes."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import time
from typing import Any
from uuid import uuid4

import numpy as np

from semanticmemo.store.sqlite_store import SQLiteCacheStore


def _acquire_cross_process_lock(db_path: str, flight_key: str) -> bool:
    """Atomic cross-process leader lock acquisition using SQLite table constraint."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute(
            "INSERT INTO flight_locks (flight_key, token, created_at) VALUES (?, ?, ?)",
            (flight_key, str(uuid4()), time.time()),
        )
        conn.commit()
        return True
    except (sqlite3.IntegrityError, sqlite3.OperationalError):
        return False
    finally:
        conn.close()


def _os_process_worker_stampede(
    worker_id: str,
    db_path: str,
    flight_key: str,
    prompt: str,
    tenant_id: str,
    results_list: Any,
    llm_counter: Any,
    num_requests: int = 30,
) -> None:
    """Independent worker executing requests against shared store and lock."""
    store = SQLiteCacheStore(db_path)

    for _ in range(num_requests):
        # 1. Authoritative check
        entry = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
        if entry is not None:
            results_list.append((worker_id, "follower_hit", entry.response))
            continue

        # 2. Distributed Single-Flight Leader Acquisition
        is_leader = _acquire_cross_process_lock(db_path, flight_key)
        if is_leader:
            # === LEADER PATH ===
            llm_counter.value += 1
            time.sleep(0.05)  # Simulate 50ms LLM compute latency
            resp = f"Computed LLM result for {prompt}"
            store.add(
                prompt=prompt,
                embedding=np.array([0.1] * 384, dtype=np.float32),
                response=resp,
                model="os-worker-model",
                tenant_id=tenant_id,
                flight_generation=1,
            )
            results_list.append((worker_id, "leader", resp))
        else:
            # === FOLLOWER PATH ===
            # Poll authoritative cache store until leader writes result
            resolved = None
            start_poll = time.time()
            while time.time() - start_poll < 5.0:
                entry = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
                if entry is not None:
                    resolved = entry.response
                    break
                time.sleep(0.01)  # 10ms polling interval
            results_list.append((worker_id, "follower", resolved or "timeout_fallback"))
    store.close()


def test_multiprocess_3_workers_coalesce_identical_stampede(tmp_path: Any) -> None:
    """Proves 100 concurrent requests across 3 independent OS processes coalesce to 1 LLM call."""
    db_path = str(tmp_path / "multiprocess_stampede.db")

    prompt = "What is the capital of France?"
    tenant_id = "tenant_enterprise"
    flight_key = f"sm:flight:{tenant_id}:global:v2:{uuid4().hex[:8]}"

    # Pre-initialize store schema and flight_locks table before spawning worker processes
    init_store = SQLiteCacheStore(db_path)
    init_store._connection.execute(
        """
        CREATE TABLE IF NOT EXISTS flight_locks (
            flight_key TEXT PRIMARY KEY,
            token TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    init_store._connection.commit()
    init_store.close()

    # Manager proxy objects for deadlock-free inter-process coordination
    with mp.Manager() as manager:
        llm_counter = manager.Value("i", 0)
        results_list = manager.list()

        workers = []
        # 3 distinct OS processes: Worker A (35), Worker B (35), Worker C (30) = 100 total
        req_counts = [35, 35, 30]
        for idx, count in enumerate(req_counts):
            w = mp.Process(
                target=_os_process_worker_stampede,
                args=(
                    f"Worker-{chr(65 + idx)}",
                    db_path,
                    flight_key,
                    prompt,
                    tenant_id,
                    results_list,
                    llm_counter,
                    count,
                ),
            )
            workers.append(w)

        # Launch all 3 OS processes simultaneously
        for w in workers:
            w.start()

        # Await process completion
        for w in workers:
            w.join(timeout=15)

        collected = list(results_list)

        assert len(collected) == 100

        # Assert: Exactly 1 LLM execution occurred across all 3 OS processes
        assert llm_counter.value == 1

        # Assert: All 100 requests across all workers received identical correct response
        for _, _, resp in collected:
            assert resp == f"Computed LLM result for {prompt}"


def _os_process_multi_tenant_worker(
    worker_id: str,
    db_path: str,
    tenants: list[str],
    results_list: Any,
    llm_counter: Any,
) -> None:
    """Worker querying across multiple tenants simultaneously."""
    store = SQLiteCacheStore(db_path)
    prompt = "Shared multi-tenant prompt"
    for tenant_id in tenants:
        flight_key = f"sm:flight:{tenant_id}:global:v2:prompt_multi"
        entry = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
        if entry is not None:
            results_list.append((tenant_id, entry.response))
            continue

        is_leader = _acquire_cross_process_lock(db_path, flight_key)
        if is_leader:
            llm_counter.value += 1
            time.sleep(0.01)
            resp = f"Response for {tenant_id}"
            store.add(
                prompt=prompt,
                embedding=np.array([0.2] * 384, dtype=np.float32),
                response=resp,
                model="multi-tenant-model",
                tenant_id=tenant_id,
            )
            results_list.append((tenant_id, resp))
        else:
            start_poll = time.time()
            resolved = None
            while time.time() - start_poll < 3.0:
                entry = store.find_by_prompt(prompt=prompt, tenant_id=tenant_id)
                if entry is not None:
                    resolved = entry.response
                    break
                time.sleep(0.005)
            results_list.append((tenant_id, resolved or "fallback"))
    store.close()


def test_multiprocess_multi_tenant_independent_flights(tmp_path: Any) -> None:
    """Proves 10 distinct tenants across OS processes execute exactly 10 independent flights."""
    db_path = str(tmp_path / "multi_tenant_multiprocess.db")

    # Pre-initialize database file before spawning worker processes
    init_store = SQLiteCacheStore(db_path)
    init_store._connection.execute(
        """
        CREATE TABLE IF NOT EXISTS flight_locks (
            flight_key TEXT PRIMARY KEY,
            token TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    init_store._connection.commit()
    init_store.close()

    tenants = [f"tenant_{i}" for i in range(10)]

    with mp.Manager() as manager:
        llm_counter = manager.Value("i", 0)
        results_list = manager.list()

        workers = []
        for idx in range(3):
            w = mp.Process(
                target=_os_process_multi_tenant_worker,
                args=(
                    f"Worker-{idx}",
                    db_path,
                    tenants,
                    results_list,
                    llm_counter,
                ),
            )
            workers.append(w)

        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=15)

        # 10 distinct tenants -> exactly 10 LLM invocations (1 per tenant)
        assert llm_counter.value == 10

        # Each tenant produced 3 responses across the 3 workers = 30 total responses
        collected = list(results_list)
        assert len(collected) == 30
        for tenant_id, resp in collected:
            assert resp == f"Response for {tenant_id}"
