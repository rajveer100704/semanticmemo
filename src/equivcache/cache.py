"""Public cache facade."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from equivcache.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from equivcache.exceptions import MissingDependencyError
from equivcache.models import CacheConfig, CacheResult, CacheStats
from equivcache.orchestrator import CacheOrchestrator
from equivcache.store import SQLiteCacheStore
from equivcache.types import EmbeddingProvider, LLMFunction


class EquivCache:
    """Async-first semantic cache facade.

    The first implementation uses cosine similarity as a measured baseline.
    The classifier score is intentionally `None` until the learned classifier
    milestone is implemented.
    """

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        store: SQLiteCacheStore | None = None,
        use_faiss: bool = True,
    ) -> None:
        self.domain = domain
        self.config = config or CacheConfig()
        self.store = store or SQLiteCacheStore(self.config.db_path)
        provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.config.embedding_model,
            dim=self.config.embedding_dim,
        )
        if use_faiss:
            try:
                index = FaissVectorIndex(provider.dim)
            except MissingDependencyError:
                index = InMemoryVectorIndex(provider.dim)
        else:
            index = InMemoryVectorIndex(provider.dim)
        embedding_service = EmbeddingService(provider, index)
        self._orchestrator = CacheOrchestrator(
            domain=domain,
            config=self.config,
            store=self.store,
            embedding_service=embedding_service,
        )

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        return await self._orchestrator.get_or_call(
            prompt=prompt,
            llm_function=llm_function,
            model=model,
            metadata=metadata,
        )

    async def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        _ = reason
        return self._orchestrator.report_bad_hit(query_id)

    async def report_good_hit(self, query_id: UUID) -> bool:
        return self._orchestrator.report_good_hit(query_id)

    def stats(self) -> CacheStats:
        return self._orchestrator.stats()

    def close(self) -> None:
        self.store.close()
