import pytest

from semanticmemo.testing.lexical_baseline import fit_tfidf, rank_tfidf


def test_tfidf_ranks_lexically_matching_document_first() -> None:
    documents = [
        "Approve the refund request",
        "Deny the refund request",
        "Reset the account password",
    ]
    vectors, idf, default_idf = fit_tfidf(documents)

    ranking = rank_tfidf(
        "Approve refund request",
        documents,
        vectors,
        idf,
        default_idf,
    )

    assert ranking[0] == 0
    assert sorted(ranking) == [0, 1, 2]


def test_tfidf_unknown_query_has_deterministic_ties() -> None:
    documents = ["Zebra document", "Apple document"]
    vectors, idf, default_idf = fit_tfidf(documents)

    ranking = rank_tfidf(
        "unseen vocabulary",
        documents,
        vectors,
        idf,
        default_idf,
    )

    assert ranking == [1, 0]


def test_tfidf_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="At least one"):
        fit_tfidf([])
