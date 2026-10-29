#!/usr/bin/env python3
"""Evaluate ranked vector retrieval against a manually curated intent dataset.

This is a closed-corpus offline benchmark. Each query's assigned intent group
defines its relevant candidate documents; candidates in other groups are
non-relevant under the dataset's labeling policy.

The optional hash-embedding mode is an integration smoke test only. Its scores
must not be interpreted as semantic-retrieval quality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def _portable_report_path(path: Path) -> str:
    """Return a repository-relative path when the file is inside the repo."""
    resolved = path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        # Preserve a usable path for explicitly configured external files.
        return str(resolved)
from typing import Any
from uuid import UUID, uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from semanticmemo.embedding import (  # noqa: E402
    EmbeddingService,
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from semanticmemo.models import CacheScope, RetrievalScope  # noqa: E402
from semanticmemo.testing.retrieval_metrics import (  # noqa: E402
    compute_ranking_metrics,
)


DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "data" / "retrieval_relevance.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "retrieval_relevance_benchmark.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def positive_int(value: str) -> int:
    """Parse a positive integer for CLI cutoffs."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and minimally validate the curated retrieval dataset."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    groups = data.get("groups")

    if not isinstance(groups, list) or not groups:
        raise ValueError("Dataset must contain a non-empty 'groups' list")

    seen_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Every group must be a JSON object")

        group_id = group.get("id")
        query = group.get("query")
        candidates = group.get("candidates")

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Every group must have a non-empty string id")
        if group_id in seen_ids:
            raise ValueError(f"Duplicate group id: {group_id}")
        seen_ids.add(group_id)

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Group {group_id} has an invalid query")
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(item, str) or not item.strip() for item in candidates)
        ):
            raise ValueError(f"Group {group_id} must have non-empty candidate strings")

    return data, groups


def run_benchmark(
    *,
    dataset_path: Path,
    output_path: Path,
    embedding_model: str = DEFAULT_MODEL,
    use_hash_embeddings: bool = False,
    recall_k: int = 5,
    mrr_k: int = 10,
) -> dict[str, Any]:
    """Build the candidate index and evaluate every query against it."""
    data, groups = load_dataset(dataset_path)

    provider = (
        HashEmbeddingProvider(dim=32)
        if use_hash_embeddings
        else SentenceTransformerEmbeddingProvider(
            model_name=embedding_model,
            dim=384,
        )
    )

    vector_index = InMemoryVectorIndex(dim=provider.dim)
    embedding_service = EmbeddingService(provider, vector_index)

    # ID -> text and intent group; the same complete candidate corpus is used
    # for every query to avoid accidentally evaluating only within-group data.
    candidate_text_by_id: dict[UUID, str] = {}
    candidate_group_by_id: dict[UUID, str] = {}
    relevant_ids_by_group: dict[str, set[UUID]] = {}

    for group in groups:
        group_id = group["id"]
        relevant_ids_by_group[group_id] = set()

        for candidate_text in group["candidates"]:
            entry_id = uuid4()
            embedding = embedding_service.embed(candidate_text)

            embedding_service.add(
                entry_id=entry_id,
                embedding=embedding,
                tenant_id="retrieval-benchmark",
                scope=CacheScope.GLOBAL,
                pipeline_version="v2",
            )

            candidate_text_by_id[entry_id] = candidate_text
            candidate_group_by_id[entry_id] = group_id
            relevant_ids_by_group[group_id].add(entry_id)

    retrieval_scope = RetrievalScope(
        tenant_id="retrieval-benchmark",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )

    search_k = max(recall_k, mrr_k)
    query_results: list[dict[str, Any]] = []
    query_latencies_ms: list[float] = []

    recall_key = f"recall_at_{recall_k}"
    precision_key = f"precision_at_{recall_k}"
    mrr_key = f"mrr_at_{mrr_k}"

    for group in groups:
        start = time.perf_counter()

        query_embedding = embedding_service.embed(group["query"])
        ranked_candidates = embedding_service.search(
            query_embedding,
            search_k,
            retrieval_scope=retrieval_scope,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        query_latencies_ms.append(elapsed_ms)

        ranked_ids = [candidate.entry_id for candidate in ranked_candidates]
        metrics = compute_ranking_metrics(
            ranked_ids,
            relevant_ids_by_group[group["id"]],
            recall_k=recall_k,
            mrr_k=mrr_k,
        )

        top_id = ranked_ids[0] if ranked_ids else None
        top_group_id = candidate_group_by_id.get(top_id) if top_id else None
        top_results = [
            {
                "intent_group": candidate_group_by_id[candidate.entry_id],
                "text": candidate_text_by_id[candidate.entry_id],
                "cosine_score": round(float(candidate.score), 6),
            }
            for candidate in ranked_candidates[:3]
        ]

        query_results.append(
            {
                "query_id": group["id"],
                "domain": group.get("domain", "unspecified"),
                "query": group["query"],
                recall_key: round(float(metrics["recall_at_k"]), 6),
                precision_key: round(float(metrics["precision_at_k"]), 6),
                mrr_key: round(float(metrics["mrr_at_k"]), 6),
                "first_relevant_rank": metrics["first_relevant_rank"],
                "top1_intent_match": top_group_id == group["id"],
                "top1_intent_group": top_group_id,
                "query_latency_ms": round(elapsed_ms, 3),
                "top_results": top_results,
            }
        )

    top1_accuracy = statistics.mean(
        float(row["top1_intent_match"]) for row in query_results
    )

    summary = {
        recall_key: round(
            statistics.mean(float(row[recall_key]) for row in query_results), 6
        ),
        precision_key: round(
            statistics.mean(float(row[precision_key]) for row in query_results), 6
        ),
        mrr_key: round(
            statistics.mean(float(row[mrr_key]) for row in query_results), 6
        ),
        "top1_intent_accuracy": round(top1_accuracy, 6),
        "cross_intent_top1_rate": round(1.0 - top1_accuracy, 6),
        "mean_query_latency_ms": round(statistics.mean(query_latencies_ms), 3),
        "p95_query_latency_ms": round(
            float(np.percentile(query_latencies_ms, 95)), 3
        ),
    }

    result = {
        "benchmark": "semanticmemo_retrieval_relevance",
        "dataset": _portable_report_path(dataset_path),
        "dataset_name": data.get("dataset_name", "unspecified"),
        "dataset_type": data.get("dataset_type", "unspecified"),
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
        "index_size": len(candidate_text_by_id),
        "query_count": len(groups),
        "search_k": search_k,
        "recall_k": recall_k,
        "mrr_k": mrr_k,
        "summary": summary,
        "queries": query_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nSemanticMemo Retrieval Relevance Benchmark")
    print("=" * 54)
    print(f"Mode:             {result['evaluation_mode']}")
    print(f"Embedding model:  {result['embedding_model']}")
    print(f"Candidate corpus: {result['index_size']}")
    print(f"Queries:          {result['query_count']}")
    print("-" * 54)
    for metric_name, value in summary.items():
        print(f"{metric_name:<29} {value}")
    print("-" * 54)
    if use_hash_embeddings:
        print("WARNING: Hash mode validates the pipeline only; quality scores")
        print("         are not evidence of semantic retrieval performance.")
    print(f"JSON report: {output_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--use-hash-embeddings",
        action="store_true",
        help="Run an integration smoke test; do not interpret quality scores",
    )
    parser.add_argument("--recall-k", type=positive_int, default=5)
    parser.add_argument("--mrr-k", type=positive_int, default=10)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")

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
