#!/usr/bin/env python3
"""High-stakes false-positive evaluation: opposite-action prompts cosine confuses.

A semantic cache that decides hits by cosine similarity alone will serve the
cached answer for "increase the dosage" when the new prompt is "decrease the
dosage" -- the two are lexically almost identical, so their embeddings sit very
close. In medical, legal, and finance contexts that is a dangerous wrong answer,
not a cheap one.

This benchmark runs a small, hand-authored set of high-stakes prompt pairs
(``benchmarks/data/high_stakes_pairs.jsonl``) through both decision methods --
the cosine baseline at its 0.90 threshold and the bundled classifier at its
precision-first threshold -- and reports how many *opposite-action* pairs each
would wrongly serve from cache, plus the dangerous hits the classifier blocks.

The dataset is small, hand-authored, and adversarial by design: it demonstrates
the failure mode, it is not a sample of production traffic. Read the counts as a
demonstration, not a leaderboard score.

Usage::

    uv run python benchmarks/false_positive_eval.py
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
from smartmemo.embedding import SentenceTransformerEmbeddingProvider  # noqa: E402
from smartmemo.embedding.service import normalize  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "benchmarks" / "data" / "high_stakes_pairs.jsonl"
DEFAULT_MODEL = REPO_ROOT / "src" / "smartmemo" / "_models" / "classifier-v2.pt"
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results" / "false_positive_eval.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def score_high_stakes(
    *,
    model_path: Path,
    data_path: Path,
    cosine_threshold: float = 0.90,
) -> dict[str, Any]:
    """Score cosine and the bundled classifier on the high-stakes pair set."""

    records = load_pair_records(data_path, split="test")
    if not records:
        msg = f"No test records found in {data_path}"
        raise ValueError(msg)
    labels = [record.label for record in records]

    provider = SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL, dim=EMBEDDING_DIM)
    emb_a = [normalize(provider.embed(record.prompt_a)) for record in records]
    emb_b = [normalize(provider.embed(record.prompt_b)) for record in records]
    cosine = [float(np.dot(left, right)) for left, right in zip(emb_a, emb_b, strict=True)]

    service = ClassifierService(model_path)
    classifier = service.predict_batch(list(zip(emb_a, emb_b, strict=True)))

    cosine_metrics = compute_binary_metrics(
        labels=labels, scores=cosine, threshold=cosine_threshold
    )
    classifier_metrics = compute_binary_metrics(
        labels=labels, scores=classifier, threshold=service.threshold
    )

    pairs: list[dict[str, Any]] = []
    for record, cos, clf in zip(records, cosine, classifier, strict=True):
        pairs.append(
            {
                "domain": record.domain,
                "prompt_a": record.prompt_a,
                "prompt_b": record.prompt_b,
                "label": record.label,
                "cosine": cos,
                "cosine_hit": cos >= cosine_threshold,
                "classifier": clf,
                "classifier_hit": clf >= service.threshold,
            }
        )

    negatives = [pair for pair in pairs if pair["label"] == 0]
    cosine_fps = [pair for pair in negatives if pair["cosine_hit"]]
    classifier_fps = [pair for pair in negatives if pair["classifier_hit"]]
    blocked = [pair for pair in cosine_fps if not pair["classifier_hit"]]

    return {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "model_name": model_path.stem,
        "cosine_threshold": cosine_threshold,
        "classifier_threshold": service.threshold,
        "examples": len(records),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "cosine": cosine_metrics.to_dict(),
        "classifier": classifier_metrics.to_dict(),
        "cosine_false_positives": len(cosine_fps),
        "classifier_false_positives": len(classifier_fps),
        "dangerous_hits_blocked_by_classifier": len(blocked),
        "pairs": pairs,
    }


def _short(text: str, width: int = 44) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cosine-threshold", type=float, default=0.90)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    result = score_high_stakes(
        model_path=args.model,
        data_path=args.data,
        cosine_threshold=args.cosine_threshold,
    )

    print(
        f"\nHigh-stakes set: {result['examples']} pairs "
        f"({result['positives']} equivalent, {result['negatives']} opposite-action)\n"
    )
    name = result["model_name"]
    print(
        f"  cosine baseline       P={result['cosine']['precision']:.3f}  "
        f"R={result['cosine']['recall']:.3f}  F1={result['cosine']['f1']:.3f}  "
        f"(thr {result['cosine_threshold']:.2f})"
    )
    print(
        f"  {name:<20}P={result['classifier']['precision']:.3f}  "
        f"R={result['classifier']['recall']:.3f}  F1={result['classifier']['f1']:.3f}  "
        f"(thr {result['classifier_threshold']:.2f})"
    )
    print(
        f"\n  opposite-action pairs wrongly served from cache:\n"
        f"    {'cosine baseline:':<18}"
        f"{result['cosine_false_positives']} / {result['negatives']}\n"
        f"    {name + ':':<18}"
        f"{result['classifier_false_positives']} / {result['negatives']}"
    )

    blocked = [
        pair
        for pair in result["pairs"]
        if pair["label"] == 0 and pair["cosine_hit"] and not pair["classifier_hit"]
    ]
    if blocked:
        print(f"\n  dangerous cache hits cosine accepts but {name} blocks ({len(blocked)}):\n")
        for pair in blocked:
            print(
                f"    [{pair['domain']}] cosine={pair['cosine']:.3f} "
                f"classifier={pair['classifier']:.3f}"
            )
            print(f"      A: {_short(pair['prompt_a'])}")
            print(f"      B: {_short(pair['prompt_b'])}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
