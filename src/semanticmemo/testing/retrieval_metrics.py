"""Ranking metrics for offline retrieval evaluation."""

from collections.abc import Hashable, Sequence, Set


def compute_ranking_metrics(
    ranked_ids: Sequence[Hashable],
    relevant_ids: Set[Hashable],
    *,
    recall_k: int = 5,
    mrr_k: int = 10,
) -> dict[str, float | int | None]:
    """Compute Recall@K, Precision@K, and reciprocal rank.

    Relevant IDs must represent the judged-relevant items in the candidate
    corpus, not merely the items returned by the retriever.
    """
    if recall_k < 1 or mrr_k < 1:
        raise ValueError("recall_k and mrr_k must be positive integers")

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must contain at least one item")

    top_recall = ranked_ids[:recall_k]
    relevant_retrieved = len(set(top_recall) & relevant)

    first_relevant_rank = next(
        (
            rank
            for rank, item_id in enumerate(ranked_ids[:mrr_k], start=1)
            if item_id in relevant
        ),
        None,
    )

    return {
        "recall_at_k": relevant_retrieved / len(relevant),
        "precision_at_k": relevant_retrieved / recall_k,
        "mrr_at_k": (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
    }
