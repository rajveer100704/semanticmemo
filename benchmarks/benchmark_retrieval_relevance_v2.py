#!/usr/bin/env python3
"""Evaluate SemanticMemo retrieval against canonical queries and paraphrases."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from benchmark_retrieval_relevance import load_dataset
from semanticmemo.embedding import (
    EmbeddingService,
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from semanticmemo.models import CacheScope, RetrievalScope
from semanticmemo.testing.retrieval_metrics import compute_ranking_metrics


DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "data" / "retrieval_relevance_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "retrieval_relevance_v2_benchmark.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_queries(
    data: dict[str, Any],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine canonical queries and paraphrases with explicit intent labels."""
    valid_intents = {group["id"] for group in groups}
    queries = [
        {
            "query_id": group["id"],
            "intent_id": group["id"],
            "domain": group.get("domain", "unspecified"),
            "query": group["query"],
        }
        for group in groups
    ]

    variants = data.get("query_variants", [])
    if not isinstance(variants, list):
        raise ValueError("query_variants must be a list")

    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Every query variant must be an object")

        query_id = variant.get("query_id")
        intent_id = variant.get("intent_id")
        query = variant.get("query")

        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Every variant needs a non-empty query_id")
        if intent_id not in valid_intents:
            raise ValueError(f"Variant {query_id} refers to an unknown intent")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Variant {query_id} has an empty query")

        queries.append({
            "query_id": query_id,
            "intent_id": intent_id,
            "domain": variant.get("domain", "unspecified"),
            "query": query,
        })

    ids = [query["query_id"] for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation query IDs must be unique")

    return queries


def run_benchmark(
    dataset_path: Path,
    output_path: Path,
    embedding_model: str = DEFAULT_MODEL,
    use_hash_embeddings: bool = False,
    recall_k: int = 5,
    mrr_k: int = 10,
) -> dict[str, Any]:
    data, groups = load_dataset(dataset_path)
    queries = build_queries(data, groups)

    provider = (
        HashEmbeddingProvider(dim=32)
        if use_hash_embeddings
        else SentenceTransformerEmbeddingProvider(
            model_name=embedding_model,
            dim=384,
        )
    )

    index = InMemoryVectorIndex(dim=provider.dim)
    service = EmbeddingService(provider, index)

    candidate_text: dict[UUID, str] = {}
    candidate_intent: dict[UUID, str] = {}
    relevant_ids: dict[str, set[UUID]] = {
        group["id"]: set() for group in groups
    }

    for group in groups:
        for text in group["candidates"]:
            entry_id = uuid4()
            embedding = service.embed(text)
            service.add(
                entry_id=entry_id,
                embedding=embedding,
                tenant_id="retrieval-benchmark",
                scope=CacheScope.GLOBAL,
                pipeline_version="v2",
            )
            candidate_text[entry_id] = text
            candidate_intent[entry_id] = group["id"]
            relevant_ids[group["id"]].add(entry_id)

    scope = RetrievalScope(
        tenant_id="retrieval-benchmark",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )

    search_k = max(recall_k, mrr_k)
    recall_key = f"recall_at_{recall_k}"
    precision_key = f"precision_at_{recall_k}"
    mrr_key = f"mrr_at_{mrr_k}"
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for query in queries:
        started = time.perf_counter()
        embedding = service.embed(query["query"])
        results = service.search(embedding, search_k, retrieval_scope=scope)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)

        ranked_ids = [item.entry_id for item in results]
        intent_id = query["intent_id"]

        metrics = compute_ranking_metrics(
            ranked_ids,
            relevant_ids[intent_id],
            recall_k=recall_k,
            mrr_k=mrr_k,
        )

        first_id = ranked_ids[0] if ranked_ids else None
        first_intent = candidate_intent.get(first_id) if first_id else None

        rows.append({
            "query_id": query["query_id"],
            "intent_id": intent_id,
            "domain": query["domain"],
            "query": query["query"],
            recall_key: round(float(metrics["recall_at_k"]), 6),
            precision_key: round(float(metrics["precision_at_k"]), 6),
            mrr_key: round(float(metrics["mrr_at_k"]), 6),
            "first_relevant_rank": metrics["first_relevant_rank"],
            "top1_intent_group": first_intent,
            "top1_intent_match": first_intent == intent_id,
            "query_latency_ms": round(elapsed, 3),
            "top_results": [
                {
                    "intent_group": candidate_intent[item.entry_id],
                    "text": candidate_text[item.entry_id],
                    "cosine_score": round(float(item.score), 6),
                }
                for item in results[:3]
            ],
        })

    top1_accuracy = statistics.mean(
        float(row["top1_intent_match"]) for row in rows
    )

    summary = {
        recall_key: round(statistics.mean(row[recall_key] for row in rows), 6),
        precision_key: round(statistics.mean(row[precision_key] for row in rows), 6),
        mrr_key: round(statistics.mean(row[mrr_key] for row in rows), 6),
        "top1_intent_accuracy": round(top1_accuracy, 6),
        "cross_intent_top1_rate": round(1.0 - top1_accuracy, 6),
        "mean_query_latency_ms": round(statistics.mean(latencies), 3),
        "p95_query_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }

    result = {
        "benchmark": "semanticmemo_retrieval_relevance_v2",
        "dataset_name": data.get("dataset_name"),
        "dataset_type": data.get("dataset_type"),
        "labeling_policy": data.get("labeling_policy", ""),
        "evaluation_mode": (
            "smoke_test_nonsemantic_hash_embeddings"
            if use_hash_embeddings
            else "sentence_transformer_embeddings"
        ),
        "embedding_model": (
            "HashEmbeddingProvider (smoke test only)"
            if use_hash_embeddings
            else embedding_model
        ),
        "retriever": "SemanticMemo EmbeddingService + InMemoryVectorIndex",
        "scope": "global; pipeline_version=v2",
        "intent_group_count": len(groups),
        "canonical_query_count": len(groups),
        "paraphrase_count": len(queries) - len(groups),
        "query_count": len(queries),
        "index_size": len(candidate_text),
        "search_k": search_k,
        "recall_k": recall_k,
        "mrr_k": mrr_k,
        "summary": summary,
        "queries": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nExpanded SemanticMemo Retrieval Benchmark")
    print("=" * 54)
    print(f"Mode:             {result['evaluation_mode']}")
    print(f"Intent groups:    {len(groups)}")
    print(f"Canonical queries:{len(groups):>5}")
    print(f"Paraphrases:      {len(queries) - len(groups):>5}")
    print(f"Total queries:    {len(queries):>5}")
    print(f"Candidate corpus: {len(candidate_text)}")
    print("-" * 54)
    for name, value in summary.items():
        print(f"{name:<29} {value}")
    if use_hash_embeddings:
        print("WARNING: Hash embeddings are for smoke testing only.")
    print(f"JSON report: {output_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--use-hash-embeddings", action="store_true")
    parser.add_argument("--recall-k", type=positive_int, default=5)
    parser.add_argument("--mrr-k", type=positive_int, default=10)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        embedding_model=args.embedding_model,
        use_hash_embeddings=args.use_hash_embeddings,
        recall_k=args.recall_k,
        mrr_k=args.mrr_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
