"""Evaluation helpers for equivalence classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from equivcache.classifier.data import EmbeddedPairDataset, PairRecord
from equivcache.classifier.model import PairClassifier
from equivcache.types import EmbeddingProvider

_torch: Any = torch


@dataclass(frozen=True)
class EvaluationMetrics:
    """Binary classifier metrics for one threshold."""

    examples: int
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc_roc: float | None
    expected_calibration_error: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def compute_binary_metrics(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> EvaluationMetrics:
    """Compute thresholded precision-first metrics without sklearn."""

    if len(labels) != len(scores):
        msg = f"labels and scores length mismatch: {len(labels)} != {len(scores)}"
        raise ValueError(msg)
    if not labels:
        msg = "At least one label is required"
        raise ValueError(msg)

    predictions = [1 if score >= threshold else 0 for score in scores]
    outcomes = list(zip(labels, predictions, strict=True))
    true_positives = sum(1 for label, pred in outcomes if label == pred == 1)
    false_positives = sum(1 for label, pred in outcomes if label == 0 and pred == 1)
    true_negatives = sum(1 for label, pred in outcomes if label == pred == 0)
    false_negatives = sum(1 for label, pred in outcomes if label == 1 and pred == 0)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positives + true_negatives) / len(labels)

    return EvaluationMetrics(
        examples=len(labels),
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        auc_roc=_roc_auc(labels, scores),
        expected_calibration_error=_expected_calibration_error(labels, scores),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def evaluate_model(
    *,
    model: PairClassifier,
    records: Sequence[PairRecord],
    embedding_provider: EmbeddingProvider,
    threshold: float = 0.85,
    batch_size: int = 32,
    device: str = "cpu",
) -> EvaluationMetrics:
    """Run a trained model over prompt-pair records and compute metrics."""

    dataset = EmbeddedPairDataset(records, embedding_provider)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_device = _torch.device(device)
    model.to(target_device)
    model.eval()

    scores: list[float] = []
    labels: list[int] = []
    with _torch.no_grad():
        for emb_a, emb_b, batch_labels in loader:
            probabilities = model(emb_a.to(target_device), emb_b.to(target_device))
            scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
            labels.extend(int(round(float(value))) for value in batch_labels.tolist())
    return compute_binary_metrics(labels=labels, scores=scores, threshold=threshold)


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for rank_index in range(index, end):
            original_index = ranked[rank_index][0]
            ranks[original_index] = average_rank
        index = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _expected_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    total = len(labels)
    error = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        bucket_items = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if lower <= score < upper or (bucket == bins - 1 and score == upper)
        ]
        if not bucket_items:
            continue
        bucket_labels = [label for label, _ in bucket_items]
        bucket_scores = [score for _, score in bucket_items]
        accuracy = sum(bucket_labels) / len(bucket_labels)
        confidence = sum(bucket_scores) / len(bucket_scores)
        error += len(bucket_items) / total * abs(accuracy - confidence)
    return error
