from __future__ import annotations

import pytest

from equivcache.classifier import compute_binary_metrics


def test_compute_binary_metrics_reports_confusion_counts() -> None:
    metrics = compute_binary_metrics(
        labels=[1, 0, 1, 0],
        scores=[0.9, 0.8, 0.2, 0.1],
        threshold=0.5,
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_compute_binary_metrics_computes_auc_for_both_classes() -> None:
    metrics = compute_binary_metrics(
        labels=[1, 1, 0, 0],
        scores=[0.9, 0.8, 0.2, 0.1],
        threshold=0.5,
    )

    assert metrics.auc_roc == pytest.approx(1.0)
