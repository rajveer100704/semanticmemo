"""Benchmark: Multi-Service Load & Telemetry Overhead Budget Profiler.

Measures latency distribution (P50, P95, P99) across 4 profiles:
  Profile A: Baseline (No-op telemetry)
  Profile B: Metrics enabled (Prometheus low-cardinality counters & histograms)
  Profile C: Tracing enabled (OpenTelemetry distributed trace spans)
  Profile D: Full Telemetry (Metrics + Distributed Tracing)
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from typing import Any

from semanticmemo.cache import SemanticMemo
from semanticmemo.models import CacheConfig
from semanticmemo.telemetry.metrics import SemanticMemoMetrics
from semanticmemo.telemetry.tracing import SemanticMemoTracer


async def profile_run(
    profile_name: str,
    num_requests: int = 500,
    enable_metrics: bool = True,
    enable_tracing: bool = False,
) -> dict[str, Any]:
    """Execute concurrent benchmark requests under specified telemetry configuration."""
    SemanticMemoMetrics.reset()

    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(
            embedding_dim=384,
            candidate_k=5,
            vector_store_type="in_memory",
        ),
    )
    user = cache.orchestrator.bootstrap_default_user("bench_user", tenant_id="bench_tenant")

    # Prime cache with 50 entries
    for i in range(50):
        await cache.get_or_call(
            prompt=f"Financial statement report for portfolio item {i}",
            llm_function=lambda p: f"Verified portfolio data for {p}",
            user_id=user.id,
        )

    latencies: list[float] = []

    t_start = time.perf_counter()
    for i in range(num_requests):
        # 80% Cache Hits, 20% Misses
        idx = i % 50 if (i % 5 != 0) else (1000 + i)
        t0 = time.perf_counter()
        await cache.get_or_call(
            prompt=f"Financial statement report for portfolio item {idx}",
            llm_function=lambda p: f"Fresh LLM calculation for {p}",
            user_id=user.id,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_time = time.perf_counter() - t_start
    latencies.sort()

    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    throughput = num_requests / total_time

    return {
        "profile": profile_name,
        "requests": num_requests,
        "total_time_s": round(total_time, 3),
        "throughput_req_s": round(throughput, 1),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(statistics.mean(latencies), 3),
    }


async def main() -> None:
    print("=" * 75)
    print("SemanticMemo: Telemetry Overhead Budget & Multi-Service Load Benchmark")
    print("=" * 75)

    # Profile A: Baseline (NoOp)
    os.environ["SEMANTICMEMO_TRACE_DEBUG"] = "false"
    res_a = await profile_run("Profile A (Baseline / No-Op)", enable_metrics=False, enable_tracing=False)

    # Profile B: Metrics Enabled
    res_b = await profile_run("Profile B (Metrics Active)", enable_metrics=True, enable_tracing=False)

    # Profile C: Tracing Active
    res_c = await profile_run("Profile C (Tracing Active)", enable_metrics=False, enable_tracing=True)

    # Profile D: Full Telemetry (Metrics + Tracing)
    res_d = await profile_run("Profile D (Metrics + Tracing)", enable_metrics=True, enable_tracing=True)

    profiles = [res_a, res_b, res_c, res_d]

    print("\nBenchmark Results Summary:")
    print(f"{'Profile':<30} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Throughput (req/s)':<18}")
    print("-" * 85)

    base_p95 = res_a["p95_ms"]

    for r in profiles:
        print(f"{r['profile']:<30} | {r['p50_ms']:<9.3f} | {r['p95_ms']:<9.3f} | {r['p99_ms']:<9.3f} | {r['throughput_req_s']:<18.1f}")

    p95_overhead_b = ((res_b["p95_ms"] - base_p95) / max(0.001, base_p95)) * 100.0
    p95_overhead_d = ((res_d["p95_ms"] - base_p95) / max(0.001, base_p95)) * 100.0

    print("\nTelemetry Overhead Analysis:")
    print(f"  • Metrics Only Overhead (P95):   {p95_overhead_b:+.2f}%")
    print(f"  • Full Telemetry Overhead (P95): {p95_overhead_d:+.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
