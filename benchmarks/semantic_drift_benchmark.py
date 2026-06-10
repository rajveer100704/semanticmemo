#!/usr/bin/env python3
"""Semantic Drift Benchmark — v2 (Entity-Aware).

This benchmark evaluates how Cosine Caching, SemanticMemo v1 (MLP + Cross-Encoder),
and SemanticMemo v2 (+ Entity Change Detector) handle systematic intent drift.

Reporting Framing
-----------------
Rather than raw "Accuracy", this benchmark separates two fundamentally different
error types:

  Safety Accuracy  — what % of dangerous drift cases (MISS-expected) did the
                     system correctly reject?  Failure here is a *Dangerous
                     False Positive* (DFP): serving a stale cached answer when
                     the intent had changed.

  Reuse Accuracy   — what % of semantically equivalent queries (HIT-expected)
                     did the system correctly reuse the cache?  Failure here is
                     a *Missed Reuse* (MR): unnecessarily calling the LLM for
                     an equivalent prompt.

These are not equal.  A DFP in a medical or financial domain is a silent
catastrophic bug.  A MR is merely wasteful.  SemanticMemo v2 is optimised for
safety — we accept a modest increase in Missed Reuse to drive DFP to zero.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Setup path to import semanticmemo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from semanticmemo import CacheConfig, ClassifierConfig, CrossEncoderConfig, SemanticMemo
from semanticmemo.embedding import SentenceTransformerEmbeddingProvider

# ---------------------------------------------------------------------------
# Drift test scenarios
# ---------------------------------------------------------------------------
# Each scenario has a base_prompt used to seed the cache, and a series of
# query cases with the expected cache decision.
#
# Expected labels:
#   HIT  — semantically equivalent; safe to reuse the cached response.
#   MISS — intent has drifted; serving the cached response would be wrong.
# ---------------------------------------------------------------------------

DRIFT_SCENARIOS = [
    {
        "name": "General Summarization",
        "base_prompt": "Summarize this article on renewable energy.",
        "cases": [
            {
                "query": "Summarize this article on renewable energy.",
                "type": "Exact match",
                "expected": "HIT",
            },
            {
                "query": "Summarize the article about renewable energy.",
                "type": "Paraphrase",
                "expected": "HIT",
            },
            {
                "query": "Summarize this article on renewable energy in 3 bullet points.",
                "type": "Drift: Formatting",
                "expected": "MISS",
            },
            {
                "query": "Summarize this article on renewable energy for the CFO.",
                "type": "Drift: Audience (CFO)",
                "expected": "MISS",
            },
            {
                "query": "Summarize this article on renewable energy for a 10-year-old child.",
                "type": "Drift: Audience (Child)",
                "expected": "MISS",
            },
        ],
    },
    {
        "name": "Medical / Dosage",
        "base_prompt": "What are the common side effects of ibuprofen?",
        "cases": [
            {
                "query": "What are the common side effects of ibuprofen?",
                "type": "Exact match",
                "expected": "HIT",
            },
            {
                "query": "What are the main side effects when taking ibuprofen?",
                "type": "Paraphrase",
                "expected": "HIT",
            },
            {
                "query": "List the common side effects of ibuprofen as a bulleted list.",
                "type": "Drift: Formatting",
                "expected": "MISS",
            },
            {
                "query": "What are the common side effects of ibuprofen for infants?",
                "type": "Drift: Audience (Infant)",
                "expected": "MISS",
            },
            {
                "query": "What are the common side effects of acetaminophen?",
                "type": "Drift: Entity (Different Drug)",
                "expected": "MISS",
            },
        ],
    },
    {
        "name": "Financial Analysis",
        "base_prompt": "Provide an analysis of Apple's Q3 earnings report.",
        "cases": [
            {
                "query": "Provide an analysis of Apple's Q3 earnings report.",
                "type": "Exact match",
                "expected": "HIT",
            },
            {
                "query": "Give me an analysis of Apple's Q3 financial results.",
                "type": "Paraphrase",
                "expected": "HIT",
            },
            {
                "query": "Provide a quick one-sentence summary of Apple's Q3 earnings.",
                "type": "Drift: Length/Scope",
                "expected": "MISS",
            },
            {
                "query": "Provide an analysis of Apple's Q3 earnings report for retail investors.",
                "type": "Drift: Audience",
                "expected": "MISS",
            },
            {
                "query": "Provide an analysis of Apple's Q4 earnings report.",
                "type": "Drift: Entity (Different Quarter)",
                "expected": "MISS",
            },
        ],
    },
    {
        "name": "Account Security",
        "base_prompt": "How do I reset my password on this account?",
        "cases": [
            {
                "query": "How do I reset my password on this account?",
                "type": "Exact match",
                "expected": "HIT",
            },
            {
                "query": "What is the procedure to change my password on this account?",
                "type": "Paraphrase",
                "expected": "HIT",
            },
            {
                "query": "Give me step-by-step instructions to reset my password on this account.",
                "type": "Drift: Detail Level",
                "expected": "MISS",
            },
            {
                "query": "How do I reset the administrator password on this account?",
                "type": "Drift: Privilege Level",
                "expected": "MISS",
            },
            {
                "query": "How do I lock my password on this account?",
                "type": "Drift: Action Inversion",
                "expected": "MISS",
            },
        ],
    },
]


async def run_benchmark(db_prefix: str) -> dict[str, Any]:
    # Clean prior temp databases
    for db in Path(".").glob(f"{db_prefix}*.db"):
        db.unlink()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    bundled_pt = REPO_ROOT / "src" / "semanticmemo" / "_models" / "equivalence-net-v1.pt"

    from semanticmemo.models import RiskPolicy, RiskTier

    risk_policy = RiskPolicy(
        domain_tiers={
            "medical": RiskTier.HIGH,
            "finance": RiskTier.HIGH,
            "security": RiskTier.HIGH,
            "general": RiskTier.LOW,
        },
        domain_thresholds={
            "medical": {"mlp": 0.995, "cross_encoder": 0.97},
            "finance": {"mlp": 0.990, "cross_encoder": 0.95},
            "security": {"mlp": 0.997, "cross_encoder": 0.98},
            "general": {"mlp": 0.90, "cross_encoder": 0.85},
        },
    )

    # ── System 1: Cosine-only (baseline) ──────────────────────────────────
    config_cosine = CacheConfig(
        db_path=Path(f"{db_prefix}_cosine.db"),
        cosine_threshold=0.90,
    )

    # ── System 2: SemanticMemo v1 (MLP + Cross-Encoder, no entity detection)
    config_v1 = CacheConfig(
        db_path=Path(f"{db_prefix}_v1.db"),
        cosine_threshold=0.80,
        cross_encoder=CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        risk_policy=risk_policy,
        high_precision_skip_threshold=2.0,  # disable bypass: force full pipeline always
    )

    # ── System 3: SemanticMemo v2 (MLP + Cross-Encoder + Entity Detector) ─
    config_v2 = CacheConfig(
        db_path=Path(f"{db_prefix}_v2.db"),
        cosine_threshold=0.80,
        cross_encoder=CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        risk_policy=risk_policy,
        high_precision_skip_threshold=2.0,  # disable bypass: force full pipeline always
    )

    from semanticmemo.entity_change_detection import EntityChangeConfig

    cache_cosine = SemanticMemo(
        domain="drift_benchmark",
        config=config_cosine,
        embedding_provider=embedding_provider,
        use_faiss=False,
    )
    cache_v1 = SemanticMemo(
        domain="drift_benchmark",
        config=config_v1,
        classifier=ClassifierConfig(model_path=bundled_pt),
        embedding_provider=embedding_provider,
        use_faiss=False,
        entity_change_config=EntityChangeConfig(enabled=False),  # v1: no entity detection
    )
    cache_v2 = SemanticMemo(
        domain="drift_benchmark",
        config=config_v2,
        classifier=ClassifierConfig(model_path=bundled_pt),
        embedding_provider=embedding_provider,
        use_faiss=False,
        entity_change_config=EntityChangeConfig(),  # v2: entity detection enabled
    )

    async def dummy_llm(p: str) -> str:
        return f"Response to: {p}"

    results: list[dict[str, Any]] = []

    for scenario in DRIFT_SCENARIOS:
        scenario_name = scenario["name"]
        base_prompt = scenario["base_prompt"]

        print(f"\nEvaluating scenario: {scenario_name}")
        print(f"  Base prompt: '{base_prompt}'")

        # Seed all caches with the base prompt
        await cache_cosine.get_or_call(prompt=base_prompt, llm_function=dummy_llm)
        await cache_v1.get_or_call(prompt=base_prompt, llm_function=dummy_llm)
        await cache_v2.get_or_call(prompt=base_prompt, llm_function=dummy_llm)

        cases_results = []
        for case in scenario["cases"]:
            query = case["query"]
            q_type = case["type"]
            expected = case["expected"]

            res_cos = await cache_cosine.get_or_call(prompt=query, llm_function=dummy_llm)
            res_v1 = await cache_v1.get_or_call(prompt=query, llm_function=dummy_llm)
            res_v2 = await cache_v2.get_or_call(prompt=query, llm_function=dummy_llm)

            cos_dec = "HIT" if res_cos.was_cache_hit else "MISS"
            v1_dec = "HIT" if res_v1.was_cache_hit else "MISS"
            v2_dec = "HIT" if res_v2.was_cache_hit else "MISS"

            cases_results.append(
                {
                    "query": query,
                    "type": q_type,
                    "expected": expected,
                    "cosine_decision": cos_dec,
                    "v1_decision": v1_dec,
                    "v2_decision": v2_dec,
                    "cosine_correct": cos_dec == expected,
                    "v1_correct": v1_dec == expected,
                    "v2_correct": v2_dec == expected,
                    # Dangerous False Positive = HIT when expected MISS
                    "cosine_dfp": expected == "MISS" and cos_dec == "HIT",
                    "v1_dfp": expected == "MISS" and v1_dec == "HIT",
                    "v2_dfp": expected == "MISS" and v2_dec == "HIT",
                    # Missed Reuse = MISS when expected HIT
                    "cosine_mr": expected == "HIT" and cos_dec == "MISS",
                    "v1_mr": expected == "HIT" and v1_dec == "MISS",
                    "v2_mr": expected == "HIT" and v2_dec == "MISS",
                    "cosine_score": res_cos.similarity_score,
                    "v1_classifier_score": res_v1.classifier_score,
                    "v1_cross_encoder_score": res_v1.cross_encoder_score,
                    "v2_classifier_score": res_v2.classifier_score,
                    "v2_cross_encoder_score": res_v2.cross_encoder_score,
                    "v2_decision_reason": res_v2.decision.reason if res_v2.decision else None,
                }
            )
            print(
                f"    [{q_type}] Expected={expected} | Cosine={cos_dec} | v1={v1_dec} | v2={v2_dec}"
            )

        results.append(
            {
                "scenario": scenario_name,
                "base_prompt": base_prompt,
                "cases": cases_results,
            }
        )

    cache_cosine.close()
    cache_v1.close()
    cache_v2.close()
    for db in Path(".").glob(f"{db_prefix}*.db"):
        db.unlink()

    # ── Aggregate statistics ──────────────────────────────────────────────
    total_hit_expected = 0
    total_miss_expected = 0
    cos_dfp = cos_mr = 0
    v1_dfp = v1_mr = 0
    v2_dfp = v2_mr = 0

    for s_res in results:
        for c_res in s_res["cases"]:
            if c_res["expected"] == "MISS":
                total_miss_expected += 1
                cos_dfp += int(c_res["cosine_dfp"])
                v1_dfp += int(c_res["v1_dfp"])
                v2_dfp += int(c_res["v2_dfp"])
            else:
                total_hit_expected += 1
                cos_mr += int(c_res["cosine_mr"])
                v1_mr += int(c_res["v1_mr"])
                v2_mr += int(c_res["v2_mr"])

    def _pct(num: int, denom: int) -> float:
        return round(100.0 * num / denom, 1) if denom else 0.0

    summary = {
        "total_drift_cases": total_miss_expected,
        "total_reuse_cases": total_hit_expected,
        # Safety Accuracy = % of MISS-expected cases correctly rejected
        "cosine_safety_accuracy": _pct(total_miss_expected - cos_dfp, total_miss_expected),
        "v1_safety_accuracy": _pct(total_miss_expected - v1_dfp, total_miss_expected),
        "v2_safety_accuracy": _pct(total_miss_expected - v2_dfp, total_miss_expected),
        # Reuse Accuracy = % of HIT-expected cases correctly reused
        "cosine_reuse_accuracy": _pct(total_hit_expected - cos_mr, total_hit_expected),
        "v1_reuse_accuracy": _pct(total_hit_expected - v1_mr, total_hit_expected),
        "v2_reuse_accuracy": _pct(total_hit_expected - v2_mr, total_hit_expected),
        # Dangerous False Positive count and rate
        "cosine_dangerous_fp": cos_dfp,
        "v1_dangerous_fp": v1_dfp,
        "v2_dangerous_fp": v2_dfp,
        "cosine_dfp_rate": _pct(cos_dfp, total_miss_expected),
        "v1_dfp_rate": _pct(v1_dfp, total_miss_expected),
        "v2_dfp_rate": _pct(v2_dfp, total_miss_expected),
        # Missed Reuse count and rate
        "cosine_missed_reuse": cos_mr,
        "v1_missed_reuse": v1_mr,
        "v2_missed_reuse": v2_mr,
        "results": results,
    }
    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    total_d = summary["total_drift_cases"]
    total_r = summary["total_reuse_cases"]

    print("\n" + "=" * 100)
    print(" SEMANTIC DRIFT BENCHMARK — v2 SAFETY REPORT")
    print("=" * 100)
    print(f"Drift (MISS-expected) cases : {total_d}")
    print(f"Reuse (HIT-expected)  cases : {total_r}")
    print()

    # Header
    print(f"{'Metric':<38} {'Cosine':>10} {'SM v1':>10} {'SM v2':>10}")
    print("-" * 72)

    def row(label: str, *vals: Any) -> str:
        return f"{label:<38} {str(vals[0]):>10} {str(vals[1]):>10} {str(vals[2]):>10}"

    print(
        row(
            "Safety Accuracy (% drift blocked)",
            f"{summary['cosine_safety_accuracy']}%",
            f"{summary['v1_safety_accuracy']}%",
            f"{summary['v2_safety_accuracy']}%",
        )
    )
    print(
        row(
            "Dangerous FP Rate (% drift served)",
            f"{summary['cosine_dfp_rate']}%",
            f"{summary['v1_dfp_rate']}%",
            f"{summary['v2_dfp_rate']}%",
        )
    )
    print(
        row(
            "Dangerous FP Count",
            summary["cosine_dangerous_fp"],
            summary["v1_dangerous_fp"],
            summary["v2_dangerous_fp"],
        )
    )
    print(
        row(
            "Reuse Accuracy (% equiv. reused)",
            f"{summary['cosine_reuse_accuracy']}%",
            f"{summary['v1_reuse_accuracy']}%",
            f"{summary['v2_reuse_accuracy']}%",
        )
    )
    print(
        row(
            "Missed Reuse Count",
            summary["cosine_missed_reuse"],
            summary["v1_missed_reuse"],
            summary["v2_missed_reuse"],
        )
    )

    print()
    print("=" * 100)
    print(" PER-CASE BREAKDOWN")
    print("=" * 100)
    hdr = f"  {'Type':<40} {'Exp':<6} {'Cosine':<14} {'SM v1':<14} {'SM v2':<14}"
    for s_res in summary["results"]:
        print(f"\n[Scenario: {s_res['scenario']}]")
        print(hdr)
        print("  " + "-" * 92)
        for c_res in s_res["cases"]:
            exp = c_res["expected"]

            def fmt(dec: str, is_dfp: bool, is_mr: bool) -> str:
                if is_dfp:
                    return f"{dec} [DFP!]"
                if is_mr:
                    return f"{dec} (MR)"
                return f"{dec} OK"

            cos_str = fmt(c_res["cosine_decision"], c_res["cosine_dfp"], c_res["cosine_mr"])
            v1_str = fmt(c_res["v1_decision"], c_res["v1_dfp"], c_res["v1_mr"])
            v2_str = fmt(c_res["v2_decision"], c_res["v2_dfp"], c_res["v2_mr"])
            reason = c_res.get("v2_decision_reason", "") or ""
            reason_short = reason.split(":")[0] if ":" in reason else reason

            print(
                f"  {c_res['type']:<40} {exp:<6} {cos_str:<14} {v1_str:<14} {v2_str:<14}"
                + (f"  [{reason_short}]" if reason_short and "entity" in reason_short else "")
            )
    print("=" * 100)
    print()
    print("Legend: [DFP!] = Dangerous False Positive (most critical failure)")
    print("        (MR)   = Missed Reuse (unnecessary LLM call, not catastrophic)")
    print("        OK     = Correct decision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-prefix", type=str, default="temp_drift")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "semantic_drift_benchmark.json",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_benchmark(args.db_prefix))
    print_summary_table(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON results to {args.out}")


if __name__ == "__main__":
    main()
