from __future__ import annotations

import sys
from uuid import uuid4

import numpy as np
import pytest

from semanticmemo.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, RetrievalScope
from semanticmemo.types import FloatVector


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

    scope = RetrievalScope(tenant_id="default", scope=CacheScope.GLOBAL)
    candidates = service.search(service.embed("x"), k=2, retrieval_scope=scope)

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

    scope = RetrievalScope(tenant_id="default", scope=CacheScope.GLOBAL)
    top = index.search(np.array([1, 0, 0], dtype=np.float32), k=2, retrieval_scope=scope)
    assert top[0].entry_id == a_id

    index.remove(a_id)
    remaining = index.search(np.array([0, 1, 0], dtype=np.float32), k=2, retrieval_scope=scope)
    assert [candidate.entry_id for candidate in remaining] == [b_id]


def test_true_pre_filtering_faiss_and_in_memory() -> None:
    """Verify that unauthorized vectors never participate in top-k similarity calculation."""
    # Test on both InMemory and Faiss
    for idx_cls in [InMemoryVectorIndex, FaissVectorIndex]:
        if idx_cls is FaissVectorIndex:
            pytest.importorskip("faiss")
        index = idx_cls(dim=3)
        unauth_id = uuid4()
        auth_id = uuid4()

        # Vector 1: Perfect cosine similarity (1.0) but belongs to unauthorized resource_2
        index.add(
            unauth_id,
            np.array([1, 0, 0], dtype=np.float32),
            tenant_id="tenant_a",
            scope=CacheScope.RESOURCE,
            resource_id="res_forbidden",
            resource_version=1,
        )

        # Vector 2: Lower similarity (0.5) but belongs to authorized resource_1
        index.add(
            auth_id,
            np.array([0.5, 0.5, 0], dtype=np.float32),
            tenant_id="tenant_a",
            scope=CacheScope.RESOURCE,
            resource_id="res_allowed",
            resource_version=1,
        )

        # Search with scope for res_allowed
        allowed_scope = RetrievalScope(
            tenant_id="tenant_a",
            scope=CacheScope.RESOURCE,
            resource_id="res_allowed",
            resource_version=1,
        )
        results = index.search(
            np.array([1, 0, 0], dtype=np.float32),
            k=5,
            retrieval_scope=allowed_scope,
        )

        # Only authorized vector returned. Unauthorized vector with 1.0 sim must NOT appear!
        assert len(results) == 1
        assert results[0].entry_id == auth_id


def test_faiss_index_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import faiss` raise ImportError.
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(MissingDependencyError, match="faiss-cpu"):
        FaissVectorIndex(dim=4)


def test_sentence_transformer_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependencyError, match=r"SemanticMemo\[ml\]"):
        SentenceTransformerEmbeddingProvider("any-model")
