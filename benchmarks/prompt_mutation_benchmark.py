#!/usr/bin/env python3
"""Prompt Mutation Benchmark.

This benchmark tests how the cache behaves under systematic prompt mutations
(character-level noise, word order shuffling, negation injection, synonym swap,
and numeric perturbation) comparing standard Cosine Caching against SemanticMemo's
Double-Verification classifier pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

# Setup path to import semanticmemo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from semanticmemo import CacheConfig, ClassifierConfig, CrossEncoderConfig, SemanticMemo
from semanticmemo.embedding import SentenceTransformerEmbeddingProvider

# Synonym Dictionary for mutation
SYNONYMS = {
    "check": "verify",
    "reset": "change",
    "cancel": "terminate",
    "membership": "subscription",
    "international": "global",
    "shipping": "delivery",
    "invoice": "receipt",
    "contact": "reach",
    "active": "current",
    "transfer": "move",
    "payment": "transaction",
    "dosage": "amount",
    "patient": "sick person",
    "user": "customer",
    "order": "purchase",
    "password": "login credentials",
}


def mutate_synonyms(text: str) -> str:
    """Replace words with common synonyms to change lexical tokens but keep semantics."""
    words = text.split()
    mutated = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w).lower()
        if clean_w in SYNONYMS:
            # Simple replacement
            w = w.lower().replace(clean_w, SYNONYMS[clean_w])
        mutated.append(w)
    res = " ".join(mutated)
    if res.lower() == text.lower():
        res += " (global delivery details)"
    return res


def mutate_word_order(text: str) -> str:
    """Restructure sentence structure slightly by swapping word order."""
    words = text.split()
    if len(words) > 3:
        # Swap adjacent words in the middle
        words[1], words[2] = words[2], words[1]
    return " ".join(words)


def mutate_negation(text: str) -> str:
    """Inject negation terms to invert the prompt's intent (should be a cache MISS)."""
    text_lower = text.lower()
    if "can you" in text_lower:
        return text.replace("Can you", "Can you not").replace("can you", "can you not")
    if "how do i" in text_lower:
        return text.replace("How do I", "How do I not").replace("how do i", "how do i not")
    if "should i" in text_lower:
        return text.replace("Should I", "Should I not").replace("should i", "should i not")
    if "is " in text_lower:
        return re.sub(r"\bis\b", "is not", text, flags=re.IGNORECASE)
    return "Do not " + text


def mutate_numeric(text: str) -> str:
    """Change numeric digits in prompt to request a different amount/count (should be cache MISS)."""
    digits = re.findall(r"\d+", text)
    if digits:
        mutated = text
        for d in digits:
            new_val = str(int(d) * 3 + 5)
            mutated = mutated.replace(d, new_val, 1)
        return mutated
    else:
        return text + " for 10 users instead of 1"


def mutate_typos(text: str) -> str:
    """Introduce character-level typos (should still be cache HIT)."""
    words = text.split()
    typo_count = 0
    for i, w in enumerate(words):
        if len(w) > 4 and typo_count < 2:
            w_list = list(w)
            # Swap adjacent characters in the word middle
            w_list[2], w_list[3] = w_list[3], w_list[2]
            words[i] = "".join(w_list)
            typo_count += 1
    res = " ".join(words)
    if res == text:
        res = text + " checkk"
    return res


async def run_benchmark(dataset_paths: list[Path], db_prefix: str) -> dict[str, Any]:
    print("Loading benchmark datasets...")
    pairs: list[dict[str, Any]] = []
    for path in dataset_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # We focus on label=1 (semantically equivalent pairs)
                if item.get("label") == 1:
                    pairs.append(item)

    print(f"Loaded {len(pairs)} equivalent prompt pairs.")
    if not pairs:
        print("No prompt pairs with label=1 found. Exiting.")
        sys.exit(1)

    # Clean prior temp databases
    for db in Path(".").glob(f"{db_prefix}*.db"):
        db.unlink()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    bundled_pt = REPO_ROOT / "src" / "semanticmemo" / "_models" / "equivalence-net-v1.pt"

    # Define configurations
    config_cosine = CacheConfig(
        db_path=Path(f"{db_prefix}_cosine.db"),
        cosine_threshold=0.90,
    )
    config_semantic = CacheConfig(
        db_path=Path(f"{db_prefix}_semantic.db"),
        cosine_threshold=0.80,  # permissive filter
        cross_encoder=CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    # Initialize caches
    cache_cosine = SemanticMemo(
        domain="benchmark",
        config=config_cosine,
        embedding_provider=embedding_provider,
        use_faiss=False,
    )
    cache_semantic = SemanticMemo(
        domain="benchmark",
        config=config_semantic,
        classifier=ClassifierConfig(model_path=bundled_pt),
        embedding_provider=embedding_provider,
        use_faiss=False,
    )

    # Dummy LLM call
    async def dummy_llm(p: str) -> str:
        return f"Response to: {p}"

    mutations = {
        "Baseline (No Mutation)": (lambda x: x, True),  # Expected HIT
        "Synonym Swap": (mutate_synonyms, True),  # Expected HIT
        "Word Order Swap": (mutate_word_order, True),  # Expected HIT
        "Typo/Noise": (mutate_typos, True),  # Expected HIT
        "Negation Injection": (mutate_negation, False),  # Expected MISS (opposite)
        "Numeric Perturbation": (mutate_numeric, False),  # Expected MISS
    }

    results: dict[str, dict[str, Any]] = {
        "cosine": {m: {"hits": 0, "misses": 0, "correct": 0} for m in mutations},
        "semantic": {m: {"hits": 0, "misses": 0, "correct": 0} for m in mutations},
    }

    total_pairs = len(pairs)
    print("\nStarting evaluation run...")

    # Seeding caches with prompt_a
    print("Seeding caches...")
    for idx, pair in enumerate(pairs):
        p_a = pair["prompt_a"]
        # Seed both caches
        await cache_cosine.get_or_call(prompt=p_a, llm_function=dummy_llm)
        await cache_semantic.get_or_call(prompt=p_a, llm_function=dummy_llm)

    print("Running mutations...")
    for mutation_name, (mutate_fn, expected_hit) in mutations.items():
        for pair in pairs:
            p_b = pair["prompt_b"]
            mutated_p = mutate_fn(p_b)

            # 1. Test Cosine Cache
            res_cos = await cache_cosine.get_or_call(prompt=mutated_p, llm_function=dummy_llm)
            was_hit_cos = res_cos.was_cache_hit
            if was_hit_cos:
                results["cosine"][mutation_name]["hits"] += 1
            else:
                results["cosine"][mutation_name]["misses"] += 1

            is_correct_cos = was_hit_cos == expected_hit
            if is_correct_cos:
                results["cosine"][mutation_name]["correct"] += 1

            # 2. Test Semantic Cache
            res_sem = await cache_semantic.get_or_call(prompt=mutated_p, llm_function=dummy_llm)
            was_hit_sem = res_sem.was_cache_hit
            if was_hit_sem:
                results["semantic"][mutation_name]["hits"] += 1
            else:
                results["semantic"][mutation_name]["misses"] += 1

            is_correct_sem = was_hit_sem == expected_hit
            if is_correct_sem:
                results["semantic"][mutation_name]["correct"] += 1

    # Cleanup temp databases
    cache_cosine.close()
    cache_semantic.close()
    for db in Path(".").glob(f"{db_prefix}*.db"):
        db.unlink()

    return {"total_pairs": total_pairs, "results": results}


def print_summary_table(data: dict[str, Any]):
    total = data["total_pairs"]
    results = data["results"]

    print("\n" + "=" * 80)
    print(" PROMPT MUTATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Evaluated on {total} prompt pairs.")
    print("-" * 80)
    print(
        f"{'Mutation Type':<25} | {'Expected':<8} | {'Cosine Accuracy':<15} | {'Semantic Accuracy':<15}"
    )
    print("-" * 80)

    for m_name in results["cosine"]:
        expected = (
            "HIT"
            if m_name in ["Baseline (No Mutation)", "Synonym Swap", "Word Order Swap", "Typo/Noise"]
            else "MISS"
        )
        cos_correct = results["cosine"][m_name]["correct"]
        sem_correct = results["semantic"][m_name]["correct"]

        cos_pct = (cos_correct / total) * 100
        sem_pct = (sem_correct / total) * 100

        print(
            f"{m_name:<25} | {expected:<8} | {cos_correct:>3}/{total} ({cos_pct:>5.1f}%) | {sem_correct:>3}/{total} ({sem_pct:>5.1f}%)"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "benchmarks" / "data" / "customer_support.jsonl",
            REPO_ROOT / "benchmarks" / "data" / "finance.jsonl",
            REPO_ROOT / "benchmarks" / "data" / "medical.jsonl",
            REPO_ROOT / "benchmarks" / "data" / "security.jsonl",
        ],
    )
    parser.add_argument("--db-prefix", type=str, default="temp_mutation")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "mutation_benchmark.json"
    )
    args = parser.parse_args()

    data = asyncio.run(run_benchmark(args.datasets, args.db_prefix))
    print_summary_table(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved JSON results to {args.out}")


if __name__ == "__main__":
    main()
