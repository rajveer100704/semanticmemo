#!/usr/bin/env python3
"""Compare saved sentence-transformer rankings with a TF-IDF baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from benchmark_retrieval_relevance import load_dataset
from semanticmemo.testing.lexical_baseline import fit_tfidf, rank_tfidf
from semanticmemo.testing.retrieval_metrics import compute_ranking_metrics


DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "data" / "retrieval_relevance.json"
DEFAULT_SEMANTIC_REPORT = (
    REPO_ROOT / "benchmarks" / "results" / "retrieval_relevance_benchmark.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "results" / "retrieval_baseline_comparison.json"
)


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def run_comparison(
    dataset_path: Path,
    semantic_report_path: Path,
    output_path: Path,
    recall_k: int = 5,
    mrr_k: int = 10,
) -> dict[str, Any]:
    data, groups = load_dataset(dataset_path)
    semantic = json.loads(semantic_report_path.read_text(encoding="utf-8-sig"))

    # Ensure that the saved semantic scores belong to this dataset.
    expected_ids = {group["id"] for group in groups}
    actual_rows = semantic.get("queries", [])
    semantic_by_id = {row["query_id"]: row for row in actual_rows}

    if semantic.get("dataset_name") != data.get("dataset_name"):
        raise ValueError("Semantic report dataset name does not match the dataset")
    if semantic.get("evaluation_mode") != "sentence_transformer_embeddings":
        raise ValueError("Saved report is not a sentence-transformer evaluation")
    if set(semantic_by_id) != expected_ids:
        raise ValueError("Semantic report query IDs do not match the dataset")

    # Flatten the complete candidate corpus. Each group labels its own three
    # candidates as relevant; other groups are non-relevant under the fixture's
    # stated closed-corpus labeling policy.
    documents: list[str] = []
    document_groups: list[str] = []
    relevant_ids: dict[str, set[int]] = {}

    for group in groups:
        group_id = group["id"]
        relevant_ids[group_id] = set()
        for text in group["candidates"]:
            index = len(documents)
            documents.append(text)
            document_groups.append(group_id)
            relevant_ids[group_id].add(index)

    if semantic.get("index_size") != len(documents):
        raise ValueError("Semantic report candidate count does not match the dataset")

    vectors, idf, default_idf = fit_tfidf(documents)

    recall_name = f"recall_at_{recall_k}"
    precision_name = f"precision_at_{recall_k}"
    mrr_name = f"mrr_at_{mrr_k}"

    lexical_rows: list[dict[str, Any]] = []
    lexical_latencies: list[float] = []

    for group in groups:
        started = time.perf_counter()
        ranked = rank_tfidf(
            group["query"], documents, vectors, idf, default_idf
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        lexical_latencies.append(elapsed_ms)

        metrics = compute_ranking_metrics(
            ranked,
            relevant_ids[group["id"]],
            recall_k=recall_k,
            mrr_k=mrr_k,
        )
        top_group = document_groups[ranked[0]] if ranked else None

        lexical_rows.append(
            {
                "query_id": group["id"],
                recall_name: round(float(metrics["recall_at_k"]), 6),
                precision_name: round(float(metrics["precision_at_k"]), 6),
                mrr_name: round(float(metrics["mrr_at_k"]), 6),
                "first_relevant_rank": metrics["first_relevant_rank"],
                "top1_intent_group": top_group,
                "top1_intent_match": top_group == group["id"],
                "query_latency_ms": round(elapsed_ms, 3),
            }
        )

    lexical_top1 = statistics.mean(
        float(row["top1_intent_match"]) for row in lexical_rows
    )

    lexical_summary = {
        recall_name: round(
            statistics.mean(float(row[recall_name]) for row in lexical_rows), 6
        ),
        precision_name: round(
            statistics.mean(float(row[precision_name]) for row in lexical_rows), 6
        ),
        mrr_name: round(
            statistics.mean(float(row[mrr_name]) for row in lexical_rows), 6
        ),
        "top1_intent_accuracy": round(lexical_top1, 6),
        "cross_intent_top1_rate": round(1.0 - lexical_top1, 6),
        "mean_query_latency_ms": round(statistics.mean(lexical_latencies), 3),
        "p95_query_latency_ms": round(
            float(np.percentile(lexical_latencies, 95)), 3
        ),
    }

    semantic_summary = semantic["summary"]
    comparison_metrics = [
        recall_name,
        precision_name,
        mrr_name,
        "top1_intent_accuracy",
        "cross_intent_top1_rate",
    ]

    deltas = {
        key: round(
            float(lexical_summary[key]) - float(semantic_summary[key]), 6
        )
        for key in comparison_metrics
    }

    semantic_latency_ms = semantic_summary.get("mean_query_latency_ms")
    result = {
        "benchmark": "semanticmemo_retrieval_baseline_comparison",
        "dataset_name": data.get("dataset_name"),
        "dataset_type": data.get("dataset_type"),
        "labeling_policy": data.get("labeling_policy"),
        "warning": (
            "Small manually curated closed-corpus evaluation. "
            "Results do not establish production retrieval quality."
        ),
        "candidate_count": len(documents),
        "query_count": len(groups),
        "recall_k": recall_k,
        "mrr_k": mrr_k,
        "semantic_embedding_model": semantic.get("embedding_model"),
        "semantic_transformer": semantic_summary,
        "tfidf_lexical": lexical_summary,
        "metric_deltas_tfidf_minus_sentence_transformer": deltas,
        "per_query": [
            {
                "query_id": group["id"],
                "semantic_top1_group": semantic_by_id[group["id"]].get(
                    "top1_intent_group"
                ),
                "semantic_top1_match": semantic_by_id[group["id"]].get(
                    "top1_intent_match"
                ),
                "tfidf_top1_group": lexical_row["top1_intent_group"],
                "tfidf_top1_match": lexical_row["top1_intent_match"],
                "tfidf_first_relevant_rank": lexical_row["first_relevant_rank"],
                "tfidf_recall": lexical_row[recall_name],
                "tfidf_mrr": lexical_row[mrr_name],
            }
            for group, lexical_row in zip(groups, lexical_rows, strict=True)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nRetrieval baseline comparison")
    print("=" * 78)
    print(f"Dataset: {data.get('dataset_name')}")
    print(f"Queries: {len(groups)} | Candidates: {len(documents)}")
    print(f"{'Metric':<27} {'Transformer':>16} {'TF-IDF':>12} {'TF-IDF delta':>17}")
    print("-" * 78)

    for key in comparison_metrics:
        left = float(semantic_summary[key])
        right = float(lexical_summary[key])
        print(f"{key:<27} {left:>16.3f} {right:>12.3f} {right-left:>+17.3f}")

    if semantic_latency_ms is not None:
        print("-" * 78)
        print(f"Mean query latency (ms): semantic={semantic_latency_ms:.3f}, "
              f"TF-IDF={lexical_summary['mean_query_latency_ms']:.3f}")
    print("-" * 78)
    print(f"Saved comparison report: {output_path}")
    print("Note: query latency is local and not an end-to-end production measurement.")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--semantic-report", type=Path, default=DEFAULT_SEMANTIC_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recall-k", type=positive_int, default=5)
    parser.add_argument("--mrr-k", type=positive_int, default=10)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")
    if not args.semantic_report.is_file():
        parser.error(f"semantic report not found: {args.semantic_report}")

    run_comparison(
        args.dataset,
        args.semantic_report,
        args.output,
        recall_k=args.recall_k,
        mrr_k=args.mrr_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
