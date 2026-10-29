"""Small, dependency-light TF-IDF baseline for offline retrieval evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tfidf_vector(
    text: str,
    idf: dict[str, float],
    default_idf: float,
) -> dict[str, float]:
    """Create an L2-normalized TF-IDF vector."""
    counts = Counter(_TOKEN_PATTERN.findall(text.casefold()))
    weights = {
        token: float(count) * idf.get(token, default_idf)
        for token, count in counts.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm == 0:
        return {}
    return {token: weight / norm for token, weight in weights.items()}


def fit_tfidf(
    documents: Sequence[str],
) -> tuple[list[dict[str, float]], dict[str, float], float]:
    """Fit smoothed inverse-document frequency on candidate documents."""
    if not documents:
        raise ValueError("At least one document is required")
    if any(not document.strip() for document in documents):
        raise ValueError("Documents must not be empty")

    tokenized = [
        _TOKEN_PATTERN.findall(document.casefold())
        for document in documents
    ]
    document_frequency: Counter[str] = Counter()

    for tokens in tokenized:
        document_frequency.update(set(tokens))

    document_count = len(documents)
    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }
    default_idf = math.log(1 + document_count) + 1.0

    vectors = [
        tfidf_vector(document, idf, default_idf)
        for document in documents
    ]
    return vectors, idf, default_idf


def rank_tfidf(
    query: str,
    documents: Sequence[str],
    document_vectors: Sequence[dict[str, float]],
    idf: dict[str, float],
    default_idf: float,
) -> list[int]:
    """Rank document indices by TF-IDF cosine similarity.

    Ties are resolved deterministically by document text, then index.
    """
    if len(documents) != len(document_vectors):
        raise ValueError("Document/vector count mismatch")

    query_vector = tfidf_vector(query, idf, default_idf)
    scores = [
        sum(
            weight * document_vector.get(token, 0.0)
            for token, weight in query_vector.items()
        )
        for document_vector in document_vectors
    ]

    return sorted(
        range(len(documents)),
        key=lambda index: (
            -scores[index],
            documents[index].casefold(),
            index,
        ),
    )
