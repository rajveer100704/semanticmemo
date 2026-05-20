from __future__ import annotations

import sys
from uuid import uuid4

import numpy as np
import pytest

from smartmemo.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from smartmemo.exceptions import MissingDependencyError
from smartmemo.types import FloatVector


class Provider:
    dim = 3

    def embed(self, text: str) -> FloatVector:
        if text == "x":
            return np.array([2, 0, 0], dtype=np.float32)
        return np.array([0, 1, 0], dtype=np.float32)


def test_embedding_service_normalizes_and_searches() -> None:
    service = EmbeddingService(Provider(), InMemoryVectorIndex(dim=3))
    x_id = uuid4()
    y_id = uuid4()

    service.add(x_id, np.array([2, 0, 0], dtype=np.float32))
    service.add(y_id, np.array([0, 1, 0], dtype=np.float32))

    candidates = service.search(service.embed("x"), k=2)

    assert candidates[0].entry_id == x_id
    assert candidates[0].score == 1.0
    assert candidates[1].entry_id == y_id


def test_embedding_service_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingService(HashEmbeddingProvider(dim=8), InMemoryVectorIndex(dim=4))


def test_hash_provider_is_deterministic() -> None:
    provider = HashEmbeddingProvider(dim=32)
    first = provider.embed("approve the refund")
    second = provider.embed("approve the refund")

    assert np.array_equal(first, second)
    assert not np.array_equal(provider.embed("deny the refund"), first)


def test_hash_provider_respects_dim() -> None:
    assert HashEmbeddingProvider(dim=64).embed("hello world").shape == (64,)


def test_faiss_index_search_and_remove() -> None:
    pytest.importorskip("faiss")
    index = FaissVectorIndex(dim=3)
    a_id = uuid4()
    b_id = uuid4()
    index.add(a_id, np.array([1, 0, 0], dtype=np.float32))
    index.add(b_id, np.array([0, 1, 0], dtype=np.float32))

    top = index.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert top[0].entry_id == a_id

    index.remove(a_id)
    remaining = index.search(np.array([0, 1, 0], dtype=np.float32), k=2)
    assert [candidate.entry_id for candidate in remaining] == [b_id]


def test_faiss_index_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import faiss` raise ImportError.
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(MissingDependencyError, match="faiss-cpu"):
        FaissVectorIndex(dim=4)


def test_sentence_transformer_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependencyError, match=r"smartmemo\[ml\]"):
        SentenceTransformerEmbeddingProvider("any-model")
