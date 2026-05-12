"""Embedding and vector-search services."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from equivcache.exceptions import MissingDependencyError
from equivcache.types import EmbeddingProvider, FloatVector


@dataclass(frozen=True)
class SearchCandidate:
    """A vector-search candidate."""

    entry_id: UUID
    score: float


def normalize(vector: FloatVector) -> FloatVector:
    """Return a float32 L2-normalized vector."""

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array
    return (array / norm).astype(np.float32)


class HashEmbeddingProvider:
    """Deterministic lightweight embedding provider for tests and smoke demos.

    This is not a semantic model. Real deployments should install `equivcache[ml]`
    and use `SentenceTransformerEmbeddingProvider`.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, text: str) -> FloatVector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector


class SentenceTransformerEmbeddingProvider:
    """SentenceTransformers-backed embedding provider."""

    def __init__(self, model_name: str, dim: int = 384) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "SentenceTransformerEmbeddingProvider requires optional ML dependencies. "
                "Install with `pip install equivcache[ml]`."
            )
            raise MissingDependencyError(msg) from exc

        self.dim = dim
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> FloatVector:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.dim,):
            msg = f"Expected embedding dimension {self.dim}, got {vector.shape}"
            raise ValueError(msg)
        return vector


class InMemoryVectorIndex:
    """Small numpy-backed vector index used for tests and dependency-light runs."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: dict[UUID, FloatVector] = {}

    def add(self, entry_id: UUID, embedding: FloatVector) -> None:
        self._vectors[entry_id] = normalize(embedding)

    def remove(self, entry_id: UUID) -> None:
        self._vectors.pop(entry_id, None)

    def rebuild(self, items: list[tuple[UUID, FloatVector]]) -> None:
        self._vectors = {}
        for entry_id, embedding in items:
            self.add(entry_id, embedding)

    def search(self, query_embedding: FloatVector, k: int) -> list[SearchCandidate]:
        query = normalize(query_embedding)
        scored = [
            SearchCandidate(entry_id=entry_id, score=float(np.dot(query, embedding)))
            for entry_id, embedding in self._vectors.items()
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[:k]


class FaissVectorIndex:
    """FAISS-backed inner-product index over normalized embeddings."""

    def __init__(self, dim: int) -> None:
        try:
            faiss: Any = __import__("faiss")
        except ImportError as exc:
            msg = (
                "FaissVectorIndex requires `faiss-cpu`. "
                "Install with `pip install equivcache[ml]`."
            )
            raise MissingDependencyError(msg) from exc

        self.dim = dim
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[UUID] = []

    def add(self, entry_id: UUID, embedding: FloatVector) -> None:
        vector = normalize(embedding).reshape(1, self.dim)
        self._index.add(vector)
        self._ids.append(entry_id)

    def remove(self, entry_id: UUID) -> None:
        if entry_id not in self._ids:
            return
        items = [
            (existing_id, self._index.reconstruct(position))
            for position, existing_id in enumerate(self._ids)
            if existing_id != entry_id
        ]
        self.rebuild(items)

    def rebuild(self, items: list[tuple[UUID, FloatVector]]) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._ids = []
        for entry_id, embedding in items:
            self.add(entry_id, embedding)

    def search(self, query_embedding: FloatVector, k: int) -> list[SearchCandidate]:
        if not self._ids:
            return []
        query = normalize(query_embedding).reshape(1, self.dim)
        scores, positions = self._index.search(query, min(k, len(self._ids)))
        candidates: list[SearchCandidate] = []
        for score, position in zip(scores[0], positions[0], strict=False):
            if position < 0:
                continue
            candidates.append(
                SearchCandidate(entry_id=self._ids[int(position)], score=float(score))
            )
        return candidates


class EmbeddingService:
    """Coordinates embedding generation and vector search."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        index: InMemoryVectorIndex | FaissVectorIndex,
    ) -> None:
        if provider.dim != index.dim:
            msg = f"Provider dimension {provider.dim} does not match index dimension {index.dim}"
            raise ValueError(msg)
        self.provider = provider
        self.index = index
        self.dim = provider.dim

    def embed(self, text: str) -> FloatVector:
        return normalize(self.provider.embed(text))

    def add(self, entry_id: UUID, embedding: FloatVector) -> None:
        self.index.add(entry_id, embedding)

    def remove(self, entry_id: UUID) -> None:
        self.index.remove(entry_id)

    def rebuild(self, items: list[tuple[UUID, FloatVector]]) -> None:
        self.index.rebuild(items)

    def search(self, query_embedding: FloatVector, k: int) -> list[SearchCandidate]:
        return self.index.search(query_embedding, k)
