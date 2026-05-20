#!/usr/bin/env python3
"""Train and package the shipped equivalence classifier (``classifier-v2``).

Reads the generated pair dataset under ``data/training/``, trains a
``PairClassifier`` over SentenceTransformers embeddings, writes the checkpoint
to the location bundled inside the package, and emits an auditable model card
covering both the held-out gold set and the high-stakes evaluation set.

    uv run python scripts/train_classifier.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from classifier_vs_cosine import run_comparison  # noqa: E402
from false_positive_eval import score_high_stakes  # noqa: E402

from smartmemo.classifier import (  # noqa: E402
    ClassifierService,
    PairRecord,
    TrainingConfig,
    compute_binary_metrics,
    load_pair_records,
    train_classifier,
)
from smartmemo.embedding import SentenceTransformerEmbeddingProvider  # noqa: E402
from smartmemo.embedding.service import normalize  # noqa: E402

DEFAULT_TRAIN = REPO_ROOT / "data" / "training" / "pairs_v2.train.jsonl"
DEFAULT_VALIDATION = REPO_ROOT / "data" / "training" / "pairs_v2.validation.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "training" / "manifest.json"
DEFAULT_GOLD = REPO_ROOT / "data" / "gold" / "equivalence_gold.jsonl"
DEFAULT_HIGH_STAKES = REPO_ROOT / "benchmarks" / "data" / "high_stakes_pairs.jsonl"
DEFAULT_OUT = REPO_ROOT / "src" / "smartmemo" / "_models" / "classifier-v2.pt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SWEEP_THRESHOLDS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def score_records(
    service: ClassifierService,
    provider: SentenceTransformerEmbeddingProvider,
    records: list[PairRecord],
) -> tuple[list[int], list[float]]:
    """Return (labels, classifier scores) for a set of prompt-pair records."""

    pairs = [
        (normalize(provider.embed(record.prompt_a)), normalize(provider.embed(record.prompt_b)))
        for record in records
    ]
    return [record.label for record in records], service.predict_batch(pairs)


def sweep(labels: list[int], scores: list[float]) -> list[dict[str, Any]]:
    """Compute precision/recall/F1 at each threshold in SWEEP_THRESHOLDS."""

    rows: list[dict[str, Any]] = []
    for threshold in SWEEP_THRESHOLDS:
        metrics = compute_binary_metrics(labels=labels, scores=scores, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--high-stakes", type=Path, default=DEFAULT_HIGH_STAKES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_records = load_pair_records(args.train, split="train")
    validation_records = load_pair_records(args.validation, split="validation")
    gold_records = load_pair_records(args.gold, split="test")
    print(
        f"loaded {len(train_records)} train / {len(validation_records)} validation / "
        f"{len(gold_records)} gold pairs"
    )

    provider = SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL, dim=EMBEDDING_DIM)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        threshold=args.threshold,
        dropout=args.dropout,
        seed=args.seed,
    )
    print(f"training: {training_config}")
    result = train_classifier(
        train_records=train_records,
        validation_records=validation_records,
        embedding_provider=provider,
        output_path=args.out,
        config=training_config,
    )
    print(f"checkpoint written to {result.checkpoint_path}")

    service = ClassifierService(args.out)
    val_labels, val_scores = score_records(service, provider, validation_records)
    gold_labels, gold_scores = score_records(service, provider, gold_records)
    val_sweep = sweep(val_labels, val_scores)
    gold_sweep = sweep(gold_labels, gold_scores)
    gold_metrics = compute_binary_metrics(
        labels=gold_labels, scores=gold_scores, threshold=args.threshold
    )

    print("\nvalidation threshold sweep:")
    for row in val_sweep:
        print(
            f"  thr={row['threshold']:.2f}  P={row['precision']:.3f}  "
            f"R={row['recall']:.3f}  F1={row['f1']:.3f}"
        )
    print("\ngold threshold sweep:")
    for row in gold_sweep:
        print(
            f"  thr={row['threshold']:.2f}  P={row['precision']:.3f}  "
            f"R={row['recall']:.3f}  F1={row['f1']:.3f}"
        )

    comparison = run_comparison(model_path=args.out, gold_path=args.gold)
    print(
        f"\ngold precision gain vs cosine at equal recall: "
        f"{comparison['precision_gain_at_equal_recall'] * 100:+.1f} points "
        f"({'PASS' if comparison['gate_passed'] else 'FAIL'})"
    )

    high_stakes = score_high_stakes(model_path=args.out, data_path=args.high_stakes)
    high_stakes_summary = {key: value for key, value in high_stakes.items() if key != "pairs"}
    print(
        f"high-stakes set: cosine wrongly serves "
        f"{high_stakes['cosine_false_positives']}/{high_stakes['negatives']} "
        f"opposite-action pairs, classifier "
        f"{high_stakes['classifier_false_positives']}/{high_stakes['negatives']}"
    )

    manifest: dict[str, Any] = {}
    if DEFAULT_MANIFEST.is_file():
        manifest = json.loads(DEFAULT_MANIFEST.read_text())

    report = {
        "model": "classifier-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "architecture": "PairClassifier MLP over [a, b, |a-b|, a*b] -> 128 -> 64 -> 1",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "operating_threshold": args.threshold,
        "scope": "generic cross-domain cold-start classifier (nine domains)",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "seed": args.seed,
            "train_examples": result.train_examples,
            "validation_examples": result.validation_examples,
            "final_train_loss": result.final_train_loss,
        },
        "dataset_manifest": manifest,
        "validation_metrics": (
            result.validation_metrics.to_dict() if result.validation_metrics else None
        ),
        "validation_threshold_sweep": val_sweep,
        "gold_metrics": gold_metrics.to_dict(),
        "gold_threshold_sweep": gold_sweep,
        "gold_vs_cosine": comparison,
        "high_stakes": high_stakes_summary,
        "limitations": (
            "Trained on LLM-paraphrased and templated data across nine domains "
            "(customer-support, software-engineering, scheduling, data-analysis, "
            "devops, general-qa, medical, legal, finance). It is a generic "
            "cold-start model: per-domain accuracy improves with the "
            "`smartmemo retrain` feedback loop. Bound to the all-MiniLM-L6-v2 "
            "embedding space (384-dim)."
        ),
    }
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"model card written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
