#!/usr/bin/env python3
"""Compare the cosine-similarity baseline against the trained equivalence classifier.

Both are scored on the hand-curated gold set (``data/gold/equivalence_gold.jsonl``)
using the same SentenceTransformers embeddings. The roadmap acceptance gate for
the shipped classifier is precision at least 10 points above the cosine baseline
*at equal-or-higher recall* -- a fair comparison, since either model can trade
recall for precision by moving its threshold.

Usage::

    uv run python benchmarks/classifier_vs_cosine.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from smartmemo.classifier import (  # noqa: E402
    ClassifierService,
    compute_binary_metrics,
    load_pair_records,
)
from smartmemo.classifier.evaluate import EvaluationMetrics  # noqa: E402
from smartmemo.embedding import SentenceTransformerEmbeddingProvider  # noqa: E402
from smartmemo.embedding.service import normalize  # noqa: E402

DEFAULT_GOLD = REPO_ROOT / "data" / "gold" / "equivalence_gold.jsonl"
DEFAULT_MODEL = REPO_ROOT / "src" / "smartmemo" / "_models" / "classifier-v2.pt"
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results" / "classifier_vs_cosine.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _sweep(labels: list[int], scores: list[float]) -> list[EvaluationMetrics]:
    """Compute metrics at every threshold from 0.00 to 1.00 in 0.01 steps."""

    return [
        compute_binary_metrics(labels=labels, scores=scores, threshold=step / 100)
        for step in range(101)
    ]


def precision_at_recall(
    labels: list[int],
    scores: list[float],
    target_recall: float,
) -> EvaluationMetrics:
    """Return the highest-precision operating point that still reaches target recall."""

    sweep = _sweep(labels, scores)
    reachable = [m for m in sweep if m.recall >= target_recall - 1e-9]
    if reachable:
        return max(reachable, key=lambda m: (m.precision, m.recall))
    # Target recall is unreachable; report the highest-recall point instead.
    return max(sweep, key=lambda m: (m.recall, m.precision))


def run_comparison(
    *,
    model_path: Path,
    gold_path: Path,
    cosine_threshold: float = 0.90,
) -> dict[str, Any]:
    """Score cosine and the classifier on the gold set and return a result dict."""

    records = load_pair_records(gold_path, split="test")
    if not records:
        msg = f"No test records found in {gold_path}"
        raise ValueError(msg)
    labels = [record.label for record in records]

    provider = SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL, dim=EMBEDDING_DIM)
    emb_a = [normalize(provider.embed(record.prompt_a)) for record in records]
    emb_b = [normalize(provider.embed(record.prompt_b)) for record in records]

    cosine = [float(np.dot(left, right)) for left, right in zip(emb_a, emb_b, strict=True)]

    service = ClassifierService(model_path)
    classifier = service.predict_batch(list(zip(emb_a, emb_b, strict=True)))

    classifier_metrics = compute_binary_metrics(
        labels=labels, scores=classifier, threshold=service.threshold
    )
    cosine_default = compute_binary_metrics(
        labels=labels, scores=cosine, threshold=cosine_threshold
    )
    cosine_matched = precision_at_recall(labels, cosine, classifier_metrics.recall)
    precision_gain = classifier_metrics.precision - cosine_matched.precision

    return {
        "gold_path": str(gold_path),
        "model_path": str(model_path),
        "examples": len(records),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "classifier": {
            "threshold": service.threshold,
            **classifier_metrics.to_dict(),
        },
        "cosine_default": {
            "threshold": cosine_threshold,
            **cosine_default.to_dict(),
        },
        "cosine_at_classifier_recall": {
            "target_recall": classifier_metrics.recall,
            **cosine_matched.to_dict(),
        },
        "precision_gain_at_equal_recall": precision_gain,
        "gate_passed": precision_gain >= 0.10,
    }


def _format_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"  {label:<34} "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"acc={metrics['accuracy']:.3f}  "
        f"thr={metrics['threshold']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--cosine-threshold", type=float, default=0.90)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    result = run_comparison(
        model_path=args.model,
        gold_path=args.gold,
        cosine_threshold=args.cosine_threshold,
    )

    print(
        f"\nGold set: {result['examples']} pairs "
        f"({result['positives']} equivalent, {result['negatives']} not)\n"
    )
    print(_format_row("cosine baseline (fixed thr)", result["cosine_default"]))
    print(_format_row("cosine at classifier recall", result["cosine_at_classifier_recall"]))
    print(_format_row("learned classifier", result["classifier"]))
    print(
        f"\n  precision gain at equal recall: "
        f"{result['precision_gain_at_equal_recall'] * 100:+.1f} points"
    )
    print(f"  acceptance gate (>= +10.0 points): {'PASS' if result['gate_passed'] else 'FAIL'}\n")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
