import pytest

from semanticmemo.testing.retrieval_metrics import compute_ranking_metrics


def test_ranking_metrics_for_partial_retrieval() -> None:
    ranked = ["noise-1", "relevant-a", "noise-2", "relevant-b", "relevant-c"]

    metrics = compute_ranking_metrics(
        ranked,
        {"relevant-a", "relevant-b", "relevant-c"},
        recall_k=4,
        mrr_k=5,
    )

    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["precision_at_k"] == pytest.approx(2 / 4)
    assert metrics["mrr_at_k"] == pytest.approx(1 / 2)
    assert metrics["first_relevant_rank"] == 2


def test_ranking_metrics_when_no_relevant_item_is_retrieved() -> None:
    metrics = compute_ranking_metrics(
        ["noise-a", "noise-b", "relevant-a"],
        {"relevant-a"},
        recall_k=2,
        mrr_k=2,
    )

    assert metrics["recall_at_k"] == 0.0
    assert metrics["precision_at_k"] == 0.0
    assert metrics["mrr_at_k"] == 0.0
    assert metrics["first_relevant_rank"] is None


def test_ranking_metrics_reject_non_positive_cutoffs() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_ranking_metrics(["a"], {"a"}, recall_k=0)


def test_ranking_metrics_require_relevance_judgments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_ranking_metrics(["a"], set())
