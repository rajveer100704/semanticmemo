#!/usr/bin/env python3
"""Compare TF-IDF with sentence-transformer retrieval on the same expanded workload."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_retrieval_relevance import load_dataset
from benchmark_retrieval_relevance_v2 import build_queries
from semanticmemo.testing.lexical_baseline import fit_tfidf, rank_tfidf
from semanticmemo.testing.retrieval_metrics import compute_ranking_metrics


DEFAULT_DATA = ROOT / "benchmarks/data/retrieval_relevance_v2.json"
DEFAULT_SEMANTIC = ROOT / "benchmarks/results/retrieval_relevance_v2_benchmark.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/retrieval_baseline_comparison_v2.json"


def run_comparison(
    dataset_path: Path,
    semantic_path: Path,
    output_path: Path,
    recall_k: int = 5,
    mrr_k: int = 10,
) -> dict[str, Any]:
    data, groups = load_dataset(dataset_path)
    queries = build_queries(data, groups)
    semantic = json.loads(semantic_path.read_text(encoding="utf-8-sig"))

    if semantic.get("dataset_name") != data.get("dataset_name"):
        raise ValueError("Dataset name does not match the semantic report")
    if semantic.get("evaluation_mode") != "sentence_transformer_embeddings":
        raise ValueError("Semantic report is not a sentence-transformer evaluation")

    semantic_rows = {
        row["query_id"]: row for row in semantic.get("queries", [])
    }
    query_ids = {query["query_id"] for query in queries}

    if set(semantic_rows) != query_ids:
        raise ValueError("Semantic and TF-IDF query ID sets do not match")

    documents: list[str] = []
    document_intents: list[str] = []
    relevant_indices: dict[str, set[int]] = {
        group["id"]: set() for group in groups
    }

    for group in groups:
        for text in group["candidates"]:
            index = len(documents)
            documents.append(text)
            document_intents.append(group["id"])
            relevant_indices[group["id"]].add(index)

    if semantic.get("index_size") != len(documents):
        raise ValueError("Candidate corpus size does not match the semantic report")

    vectors, idf, default_idf = fit_tfidf(documents)
    recall_key = f"recall_at_{recall_k}"
    precision_key = f"precision_at_{recall_k}"
    mrr_key = f"mrr_at_{mrr_k}"

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for query in queries:
        started = time.perf_counter()
        ranked = rank_tfidf(
            query["query"], documents, vectors, idf, default_idf
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

        metrics = compute_ranking_metrics(
            ranked,
            relevant_indices[query["intent_id"]],
            recall_k=recall_k,
            mrr_k=mrr_k,
        )

        first_intent = document_intents[ranked[0]] if ranked else None
        rows.append({
            "query_id": query["query_id"],
            "intent_id": query["intent_id"],
            recall_key: round(float(metrics["recall_at_k"]), 6),
            precision_key: round(float(metrics["precision_at_k"]), 6),
            mrr_key: round(float(metrics["mrr_at_k"]), 6),
            "first_relevant_rank": metrics["first_relevant_rank"],
            "top1_intent_group": first_intent,
            "top1_intent_match": first_intent == query["intent_id"],
            "query_latency_ms": round(elapsed_ms, 3),
        })

    accuracy = statistics.mean(
        float(row["top1_intent_match"]) for row in rows
    )
    lexical_summary = {
        recall_key: round(statistics.mean(row[recall_key] for row in rows), 6),
        precision_key: round(
            statistics.mean(row[precision_key] for row in rows), 6
        ),
        mrr_key: round(statistics.mean(row[mrr_key] for row in rows), 6),
        "top1_intent_accuracy": round(accuracy, 6),
        "cross_intent_top1_rate": round(1.0 - accuracy, 6),
        "mean_query_latency_ms": round(statistics.mean(latencies), 3),
        "p95_query_latency_ms": round(
            float(np.percentile(latencies, 95)), 3
        ),
    }

    transformer = semantic["summary"]
    compared = [
        recall_key,
        precision_key,
        mrr_key,
        "top1_intent_accuracy",
        "cross_intent_top1_rate",
    ]
    deltas = {
        name: round(
            float(lexical_summary[name]) - float(transformer[name]), 6
        )
        for name in compared
    }

    result = {
        "benchmark": "semanticmemo_retrieval_baseline_comparison_v2",
        "dataset_name": data.get("dataset_name"),
        "dataset_type": data.get("dataset_type"),
        "labeling_policy": data.get("labeling_policy", ""),
        "warning": (
            "Small manually curated closed-corpus evaluation; "
            "results do not establish production retrieval quality."
        ),
        "query_count": len(queries),
        "candidate_count": len(documents),
        "semantic_embedding_model": semantic.get("embedding_model"),
        "sentence_transformer": transformer,
        "tfidf_lexical": lexical_summary,
        "delta_tfidf_minus_sentence_transformer": deltas,
        "per_query": [
            {
                "query_id": query["query_id"],
                "intent_id": query["intent_id"],
                "transformer_top1_match": semantic_rows[
                    query["query_id"]
                ]["top1_intent_match"],
                "transformer_top1_group": semantic_rows[
                    query["query_id"]
                ]["top1_intent_group"],
                "tfidf_top1_match": lexical["top1_intent_match"],
                "tfidf_top1_group": lexical["top1_intent_group"],
                "tfidf_first_relevant_rank": lexical["first_relevant_rank"],
                "tfidf_recall_at_5": lexical[recall_key],
                "tfidf_mrr_at_10": lexical[mrr_key],
            }
            for query, lexical in zip(queries, rows, strict=True)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nExpanded retrieval baseline comparison")
    print("=" * 78)
    print(f"Queries: {len(queries)} | Candidates: {len(documents)}")
    print(f"{'Metric':<27} {'Transformer':>16} {'TF-IDF':>12} {'Delta':>17}")
    print("-" * 78)

    for name in compared:
        left = float(transformer[name])
        right = float(lexical_summary[name])
        print(f"{name:<27} {left:>16.3f} {right:>12.3f} {right-left:>+17.3f}")

    print("-" * 78)
    print(
        "Mean query latency (ms): "
        f"transformer={transformer['mean_query_latency_ms']:.3f}, "
        f"TF-IDF={lexical_summary['mean_query_latency_ms']:.3f}"
    )
    print(f"Report: {output_path}")
    print("Latency measurements are local; they are not end-to-end production timings.")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--semantic-report", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--mrr-k", type=int, default=10)
    args = parser.parse_args()

    if args.recall_k < 1 or args.mrr_k < 1:
        parser.error("ranking cutoffs must be positive")
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")
    if not args.semantic_report.is_file():
        parser.error(f"Semantic report not found: {args.semantic_report}")

    result = run_comparison(
        args.dataset,
        args.semantic_report,
        args.output,
        recall_k=args.recall_k,
        mrr_k=args.mrr_k,
    )

    if result["query_count"] != 42 or result["candidate_count"] != 42:
        raise ValueError("Expected exactly 42 queries and 42 candidates")
    if len(result["per_query"]) != 42:
        raise ValueError("Comparison did not include all 42 queries")

    print("Validation: PASSED ? both methods use the same 42-query workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
