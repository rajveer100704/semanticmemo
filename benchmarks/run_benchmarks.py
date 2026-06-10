#!/usr/bin/env python3
"""Benchmark runner to compare Cosine, MLP, Double Verification, and semanticmemo."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from semanticmemo import (
    CacheConfig,
    ClassifierConfig,
    CrossEncoderConfig,
    RiskPolicy,
    RiskTier,
    SemanticMemo,
)
from semanticmemo.embedding import SentenceTransformerEmbeddingProvider
from semanticmemo.store import SQLiteCacheStore

DATA_DIR = REPO_ROOT / "benchmarks" / "data"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DOMAINS = ["customer_support", "finance", "medical", "security"]


def load_pairs(domain: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{domain}.jsonl"
    if not path.exists():
        print(f"Data for domain {domain} not found at {path}")
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


async def evaluate_method_on_domain(
    method: str,
    domain: str,
    pairs: list[dict[str, Any]],
    db_path: Path,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> dict[str, Any]:
    # Ensure a clean database for each run
    if db_path.exists():
        db_path.unlink()

    # Configure the Cache based on the method
    classifier_cfg = None

    # Bundle path
    bundled_pt = REPO_ROOT / "src" / "semanticmemo" / "_models" / "equivalence-net-v1.pt"

    if method == "Cosine Baseline":
        config = CacheConfig(
            db_path=db_path,
            cosine_threshold=0.90,
        )
    elif method == "MLP Classifier":
        config = CacheConfig(
            db_path=db_path,
        )
        classifier_cfg = ClassifierConfig(
            model_path=bundled_pt,
            threshold=0.85,
        )
    elif method == "Double Verification":
        config = CacheConfig(
            db_path=db_path,
            cross_encoder=CrossEncoderConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                threshold=0.90,
            ),
        )
        classifier_cfg = ClassifierConfig(
            model_path=bundled_pt,
            threshold=0.95,
        )
    elif method == "SemanticMemo":
        # SemanticMemo uses domain auto-detection, risk-aware policy, and bypass thresholds
        risk_policy = RiskPolicy(
            domain_tiers={
                "medical": RiskTier.HIGH,
                "finance": RiskTier.HIGH,
                "security": RiskTier.HIGH,
                "legal": RiskTier.HIGH,
                "general": RiskTier.LOW,
            },
            low_risk_classifier_threshold=0.90,
            low_risk_cross_encoder_threshold=0.85,
            high_risk_classifier_threshold=0.99,
            high_risk_cross_encoder_threshold=0.95,
        )
        config = CacheConfig(
            db_path=db_path,
            cross_encoder=CrossEncoderConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            risk_policy=risk_policy,
            high_precision_skip_threshold=0.995,
        )
        classifier_cfg = ClassifierConfig(
            model_path=bundled_pt,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    # Use in-memory index to make benchmarks fast and deterministic
    cache = SemanticMemo(
        domain=domain,
        config=config,
        classifier=classifier_cfg,
        embedding_provider=embedding_provider,
        use_faiss=False,
    )

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    total_latency = 0.0
    total_cost_saved = Decimal("0")
    total_lookups = len(pairs)

    total_emb_latency = 0.0
    total_ret_latency = 0.0
    total_mlp_latency = 0.0
    total_ce_latency = 0.0
    total_decision_latency = 0.0

    skipped_ce = 0
    total_hits = 0
    bypass_latencies = []
    non_bypass_latencies = []

    # We evaluate sequentially
    for pair in pairs:
        prompt_a = pair["prompt_a"]
        prompt_b = pair["prompt_b"]
        expected_equivalent = pair["label"]  # 1 = equivalent, 0 = opposite

        # Clear store and index before each pair evaluation to simulate isolated lookup
        # (each pair contains 1 candidate prompt in the cache to check equivalence against)
        cache.store.close()
        if db_path.exists():
            db_path.unlink()

        cache.store = SQLiteCacheStore(config.db_path)
        cache._orchestrator.store = cache.store
        cache._orchestrator.total_lookups = 0
        cache._orchestrator.cache_hits = 0
        cache._orchestrator.cache_misses = 0
        cache._orchestrator.total_cost_saved_usd = Decimal("0")
        cache._orchestrator.embedding_service.index.rebuild([])

        # 1. Populate prompt A in the cache
        # We call once to populate it
        async def dummy_llm(p: str) -> str:
            return f"Response to {p}"

        await cache.get_or_call(prompt=prompt_a, llm_function=dummy_llm)

        # 2. Query prompt B
        start_time = time.perf_counter()
        res = await cache.get_or_call(prompt=prompt_b, llm_function=dummy_llm)
        latency = (time.perf_counter() - start_time) * 1000

        total_latency += latency

        total_emb_latency += res.embedding_latency_ms
        total_ret_latency += res.retrieval_latency_ms
        total_mlp_latency += res.mlp_latency_ms
        total_ce_latency += res.cross_encoder_latency_ms
        total_decision_latency += res.total_latency_ms

        if res.was_cache_hit:
            total_hits += 1
            total_cost_saved += res.cost_saved_usd
            if expected_equivalent == 1:
                true_positives += 1
            else:
                false_positives += 1

            # Check for bypass
            if res.decision is not None and res.decision.reason == "mlp_bypass":
                skipped_ce += 1
                bypass_latencies.append(latency)
            else:
                non_bypass_latencies.append(latency)
        else:
            if expected_equivalent == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    cache.close()
    if db_path.exists():
        db_path.unlink()

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )

    avg_latency = total_latency / total_lookups if total_lookups > 0 else 0.0

    return {
        "Method": method,
        "Domain": domain,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Latency": avg_latency,
        "FPR": fpr,
        "CostSaved": float(total_cost_saved),
        "AvgEmbLatency": total_emb_latency / total_lookups if total_lookups > 0 else 0.0,
        "AvgRetLatency": total_ret_latency / total_lookups if total_lookups > 0 else 0.0,
        "AvgMlpLatency": total_mlp_latency / total_lookups if total_lookups > 0 else 0.0,
        "AvgCeLatency": total_ce_latency / total_lookups if total_lookups > 0 else 0.0,
        "AvgDecisionLatency": total_decision_latency / total_lookups if total_lookups > 0 else 0.0,
        "TotalHits": total_hits,
        "SkippedCe": skipped_ce,
        "BypassLatencies": bypass_latencies,
        "NonBypassLatencies": non_bypass_latencies,
    }


async def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_db_path = Path("temp_benchmark.db")

    print("Initializing shared embedding provider...")
    shared_provider = SentenceTransformerEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")

    # Warm up models
    print("Warming up models...")
    bundled_pt = REPO_ROOT / "src" / "semanticmemo" / "_models" / "equivalence-net-v1.pt"
    warmup_config = CacheConfig(
        db_path=Path("temp_warmup.db"),
        cross_encoder=CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    warmup_cache = SemanticMemo(
        domain="general",
        config=warmup_config,
        classifier=ClassifierConfig(model_path=bundled_pt),
        embedding_provider=shared_provider,
        use_faiss=False,
    )
    _ = warmup_cache._orchestrator.embedding_service.embed("warmup")
    _ = warmup_cache._orchestrator.classifier_service.predict_batch(
        [(np.zeros(384, dtype=np.float32), np.zeros(384, dtype=np.float32))]
    )
    _ = warmup_cache._orchestrator.cross_encoder_service.predict("warmup A", "warmup B")
    warmup_cache.close()
    if Path("temp_warmup.db").exists():
        Path("temp_warmup.db").unlink()
    print("Warmup complete.")

    methods = [
        "Cosine Baseline",
        "MLP Classifier",
        "Double Verification",
        "SemanticMemo",
    ]

    all_results = []

    print("Starting SemanticMemo benchmark runs across domains...")
    print("-" * 80)

    for domain in DOMAINS:
        pairs = load_pairs(domain)
        if not pairs:
            continue

        print(f"Domain: {domain.upper()} ({len(pairs)} test pairs)")
        for method in methods:
            print(f"  Running {method}...", end="", flush=True)
            try:
                res = await evaluate_method_on_domain(
                    method, domain, pairs, temp_db_path, shared_provider
                )
                all_results.append(res)
                print(" Done.")
            except Exception as e:
                print(f" Failed: {e}")
        print("-" * 80)

    # Hard Negatives Stress Test
    hard_negatives_results = []
    print("\nStarting Hard-Negatives Stress Test...")
    print("-" * 80)
    hard_neg_pairs = []
    hard_neg_path = DATA_DIR / "hard_negatives.jsonl"
    if hard_neg_path.exists():
        with open(hard_neg_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    hard_neg_pairs.append(json.loads(line))

        print(f"Loaded {len(hard_neg_pairs)} hard negative pairs.")
        for method in methods:
            print(f"  Running {method} on Hard Negatives...", end="", flush=True)
            try:
                res = await evaluate_method_on_domain(
                    method, "hard_negatives", hard_neg_pairs, temp_db_path, shared_provider
                )
                hard_negatives_results.append(res)
                print(" Done.")
            except Exception as e:
                print(f" Failed: {e}")
        print("-" * 80)

    # Print the markdown table
    print("\n### Baseline Comparison Matrix\n")
    print("| Method | Domain | Precision | Recall | F1 | Latency (ms) | FPR | Cost Saved ($) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for res in all_results:
        print(
            f"| {res['Method']} | {res['Domain']} | {res['Precision']:.3f} | {res['Recall']:.3f} | "
            f"{res['F1']:.3f} | {res['Latency']:.1f} | {res['FPR']:.3f} | ${res['CostSaved']:.5f} |"
        )

    # Print the detailed latency breakout
    print("\n### Detailed Latency Breakout (ms)\n")
    print("| Method | Domain | Embedding | Retrieval | MLP | Cross-Encoder | Total Decision |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for res in all_results:
        print(
            f"| {res['Method']} | {res['Domain']} | {res['AvgEmbLatency']:.2f} | {res['AvgRetLatency']:.2f} | "
            f"{res['AvgMlpLatency']:.2f} | {res['AvgCeLatency']:.2f} | {res['AvgDecisionLatency']:.2f} |"
        )

    # Print Hard Negatives FPR table
    if hard_negatives_results:
        print("\n### Hard-Negatives Stress Test Results\n")
        print("| Method | Hard-Negative FPR |")
        print("| :--- | :--- |")
        for res in hard_negatives_results:
            print(f"| {res['Method']} | {res['FPR']:.3f} |")

    # Calculate and print CrossEncoder Bypass Rate
    sm_x_results = [r for r in all_results if r["Method"] == "SemanticMemo"]
    total_sm_x_hits = sum(r["TotalHits"] for r in sm_x_results)
    total_sm_x_skipped = sum(r["SkippedCe"] for r in sm_x_results)

    all_bypass_lats = []
    all_non_bypass_lats = []
    for r in sm_x_results:
        all_bypass_lats.extend(r["BypassLatencies"])
        all_non_bypass_lats.extend(r["NonBypassLatencies"])

    bypass_rate = total_sm_x_skipped / total_sm_x_hits if total_sm_x_hits > 0 else 0.0
    avg_bypass_lat = sum(all_bypass_lats) / len(all_bypass_lats) if all_bypass_lats else 0.0
    avg_non_bypass_lat = (
        sum(all_non_bypass_lats) / len(all_non_bypass_lats) if all_non_bypass_lats else 0.0
    )

    print("\n### Cross-Encoder Bypass Statistics (SemanticMemo)\n")
    print(
        f"- **Bypass Rate**: {bypass_rate * 100:.1f}% ({total_sm_x_skipped}/{total_sm_x_hits} hits)"
    )
    print(f"- **Average Latency with Bypass**: {avg_bypass_lat:.2f} ms")
    print(f"- **Average Latency without Bypass**: {avg_non_bypass_lat:.2f} ms")

    # Save results to json
    out_path = RESULTS_DIR / "comparison_matrix.json"
    results_json = {
        "comparison_matrix": all_results,
        "hard_negatives": hard_negatives_results,
        "bypass_statistics": {
            "bypass_rate": bypass_rate,
            "avg_bypass_latency": avg_bypass_lat,
            "avg_non_bypass_latency": avg_non_bypass_lat,
            "total_hits": total_sm_x_hits,
            "skipped_ce": total_sm_x_skipped,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved raw results to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
