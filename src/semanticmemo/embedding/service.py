"""Embedding and vector-search services."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import numpy as np

from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, RetrievalScope
from semanticmemo.types import EmbeddingProvider, FloatVector


@dataclass(frozen=True)
class SearchCandidate:
    """A vector-search candidate."""

    entry_id: UUID
    score: float


@dataclass
class VectorMetadata:
    """Metadata attached to a vector in the index for pre-filtering."""

    entry_id: UUID
    embedding: FloatVector
    tenant_id: str
    scope: CacheScope
    resource_id: str | None
    resource_version: int | None
    pipeline_version: str


def normalize(vector: FloatVector) -> FloatVector:
    """Return a float32 L2-normalized vector."""

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array
    return (array / norm).astype(np.float32)


class HashEmbeddingProvider:
    """Deterministic lightweight embedding provider for tests and smoke demos.

    This is not a semantic model. Real deployments should install `SemanticMemo[ml]`
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
                "Install with `pip install SemanticMemo[ml]`."
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


def _matches_scope(meta: VectorMetadata, scope: RetrievalScope) -> bool:
    """Evaluate whether a vector candidate is inside the authorized RetrievalScope."""
    # Pipeline version must match
    if meta.pipeline_version != scope.pipeline_version:
        return False

    if scope.scope == CacheScope.RESOURCE:
        return (
            meta.scope == CacheScope.RESOURCE
            and meta.tenant_id == scope.tenant_id
            and meta.resource_id == scope.resource_id
            and meta.resource_version == scope.resource_version
        )
    elif scope.scope == CacheScope.TENANT:
        # TENANT lookup matches tenant-wide or global entries, NEVER resource entries
        return meta.scope in (CacheScope.TENANT, CacheScope.GLOBAL) and (
            meta.scope == CacheScope.GLOBAL or meta.tenant_id == scope.tenant_id
        )
    elif scope.scope == CacheScope.GLOBAL:
        # GLOBAL lookup matches strictly global entries
        return meta.scope == CacheScope.GLOBAL

    return False


class InMemoryVectorIndex:
    """Small numpy-backed vector index supporting scope-isolated pre-filtering."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._entries: dict[UUID, VectorMetadata] = {}

    def add(
        self,
        entry_id: UUID,
        embedding: FloatVector,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> None:
        cscope = CacheScope(scope) if isinstance(scope, str) else scope
        self._entries[entry_id] = VectorMetadata(
            entry_id=entry_id,
            embedding=normalize(embedding),
            tenant_id=tenant_id,
            scope=cscope,
            resource_id=resource_id,
            resource_version=resource_version,
            pipeline_version=pipeline_version,
        )

    def remove(self, entry_id: UUID) -> None:
        self._entries.pop(entry_id, None)

    def rebuild(self, items: list[Any]) -> None:
        self._entries = {}
        for item in items:
            if isinstance(item, tuple):
                if len(item) == 2:
                    entry_id, embedding = item
                    self.add(entry_id, embedding)
                elif len(item) == 7:
                    entry_id, embedding, tenant_id, scope, res_id, res_ver, pipe_ver = item
                    self.add(entry_id, embedding, tenant_id, scope, res_id, res_ver, pipe_ver)

    def search(
        self,
        query_embedding: FloatVector,
        k: int,
        retrieval_scope: RetrievalScope,
    ) -> list[SearchCandidate]:
        query = normalize(query_embedding)
        # Pre-filter before computing similarity
        scored: list[SearchCandidate] = []
        for entry_id, meta in self._entries.items():
            if _matches_scope(meta, retrieval_scope):
                score = float(np.dot(query, meta.embedding))
                scored.append(SearchCandidate(entry_id=entry_id, score=score))

        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[:k]


class FaissVectorIndex:
    """FAISS-backed inner-product index with scope isolation."""

    def __init__(self, dim: int) -> None:
        try:
            faiss: Any = __import__("faiss")
        except ImportError as exc:
            msg = (
                "FaissVectorIndex requires `faiss-cpu`. "
                "Install with `pip install SemanticMemo[ml]`."
            )
            raise MissingDependencyError(msg) from exc

        self.dim = dim
        self._faiss = faiss
        self._entries: dict[UUID, VectorMetadata] = {}

    def add(
        self,
        entry_id: UUID,
        embedding: FloatVector,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> None:
        cscope = CacheScope(scope) if isinstance(scope, str) else scope
        self._entries[entry_id] = VectorMetadata(
            entry_id=entry_id,
            embedding=normalize(embedding),
            tenant_id=tenant_id,
            scope=cscope,
            resource_id=resource_id,
            resource_version=resource_version,
            pipeline_version=pipeline_version,
        )

    def remove(self, entry_id: UUID) -> None:
        self._entries.pop(entry_id, None)

    def rebuild(self, items: list[Any]) -> None:
        self._entries = {}
        for item in items:
            if isinstance(item, tuple):
                if len(item) == 2:
                    entry_id, embedding = item
                    self.add(entry_id, embedding)
                elif len(item) == 7:
                    entry_id, embedding, tenant_id, scope, res_id, res_ver, pipe_ver = item
                    self.add(entry_id, embedding, tenant_id, scope, res_id, res_ver, pipe_ver)

    def search(
        self,
        query_embedding: FloatVector,
        k: int,
        retrieval_scope: RetrievalScope,
    ) -> list[SearchCandidate]:
        # Pre-filter candidates by scope before running inner-product ranking
        filtered_ids = [
            entry_id
            for entry_id, meta in self._entries.items()
            if _matches_scope(meta, retrieval_scope)
        ]
        if not filtered_ids:
            return []

        query = normalize(query_embedding).reshape(1, self.dim)
        sub_index = self._faiss.IndexFlatIP(self.dim)
        matrix = np.vstack([self._entries[eid].embedding for eid in filtered_ids]).astype(
            np.float32
        )
        sub_index.add(matrix)

        scores, positions = sub_index.search(query, min(k, len(filtered_ids)))
        candidates: list[SearchCandidate] = []
        for score, position in zip(scores[0], positions[0], strict=False):
            if position < 0:
                continue
            candidates.append(
                SearchCandidate(
                    entry_id=filtered_ids[int(position)],
                    score=float(score),
                )
            )
        return candidates


class VectorIndex(Protocol):
    """Protocol for vector indices supporting mandatory RetrievalScope filtering."""

    dim: int

    def add(
        self,
        entry_id: UUID,
        embedding: FloatVector,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> None: ...

    def remove(self, entry_id: UUID) -> None: ...

    def rebuild(self, items: list[Any]) -> None: ...

    def search(
        self,
        query_embedding: FloatVector,
        k: int,
        retrieval_scope: RetrievalScope,
    ) -> list[SearchCandidate]: ...


class EmbeddingService:
    """Coordinates embedding generation and vector search with mandatory RetrievalScope."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        index: VectorIndex,
    ) -> None:
        if provider.dim != index.dim:
            msg = f"Provider dimension {provider.dim} does not match index dimension {index.dim}"
            raise ValueError(msg)
        self.provider = provider
        self.index = index
        self.dim = provider.dim

    def embed(self, text: str) -> FloatVector:
        return normalize(self.provider.embed(text))

    def add(
        self,
        entry_id: UUID,
        embedding: FloatVector,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> None:
        self.index.add(
            entry_id=entry_id,
            embedding=embedding,
            tenant_id=tenant_id,
            scope=scope,
            resource_id=resource_id,
            resource_version=resource_version,
            pipeline_version=pipeline_version,
        )

    def remove(self, entry_id: UUID) -> None:
        self.index.remove(entry_id)

    def rebuild(self, items: list[Any]) -> None:
        self.index.rebuild(items)

    def search(
        self,
        query_embedding: FloatVector,
        k: int,
        retrieval_scope: RetrievalScope,
    ) -> list[SearchCandidate]:
        """Search vector index strictly bounded to the RetrievalScope."""
        return self.index.search(query_embedding, k, retrieval_scope=retrieval_scope)
