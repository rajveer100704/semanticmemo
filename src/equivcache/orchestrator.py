"""Cache lookup orchestration."""

from __future__ import annotations

import inspect
import time
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from equivcache.embedding import EmbeddingService
from equivcache.models import CacheConfig, CacheResult, CacheStats
from equivcache.store import SQLiteCacheStore
from equivcache.types import LLMFunction


class CacheOrchestrator:
    """Coordinates embedding search, baseline matching, storage, and stats."""

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig,
        store: SQLiteCacheStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.domain = domain
        self.config = config
        self.store = store
        self.embedding_service = embedding_service
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved_usd = Decimal("0")
        self._query_to_entry: dict[UUID, UUID] = {}
        self._rebuild_index_from_store()

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        started_at = time.perf_counter()
        query_id = uuid4()
        self.total_lookups += 1

        query_embedding = self.embedding_service.embed(prompt)
        hit_entry_id, similarity_score = self._find_baseline_hit(query_embedding)

        if hit_entry_id is not None:
            entry = self.store.get(hit_entry_id)
            if entry is not None:
                self.store.update_hit(entry.id)
                self.cache_hits += 1
                self.total_cost_saved_usd += self.config.estimated_llm_cost_usd
                self._query_to_entry[query_id] = entry.id
                return CacheResult(
                    query_id=query_id,
                    response=entry.response,
                    was_cache_hit=True,
                    cache_entry_id=entry.id,
                    similarity_score=similarity_score,
                    classifier_score=None,
                    cost_saved_usd=self.config.estimated_llm_cost_usd,
                    latency_ms=self._elapsed_ms(started_at),
                )

        response = await self._call_llm(llm_function, prompt)
        entry_id = self.store.add(
            prompt=prompt,
            embedding=query_embedding,
            response=response,
            model=model,
            metadata=metadata,
        )
        self.embedding_service.add(entry_id, query_embedding)
        self._evict_if_needed()
        self.cache_misses += 1
        return CacheResult(
            query_id=query_id,
            response=response,
            was_cache_hit=False,
            cache_entry_id=entry_id,
            similarity_score=similarity_score,
            classifier_score=None,
            cost_saved_usd=Decimal("0"),
            latency_ms=self._elapsed_ms(started_at),
        )

    def report_bad_hit(self, query_id: UUID) -> bool:
        entry_id = self._query_to_entry.get(query_id)
        if entry_id is None:
            return False
        self.store.increment_bad_feedback(entry_id)
        return True

    def report_good_hit(self, query_id: UUID) -> bool:
        entry_id = self._query_to_entry.get(query_id)
        if entry_id is None:
            return False
        self.store.increment_good_feedback(entry_id)
        return True

    def stats(self) -> CacheStats:
        hit_rate = self.cache_hits / self.total_lookups if self.total_lookups else 0.0
        return CacheStats(
            domain=self.domain,
            total_entries=self.store.count(),
            total_lookups=self.total_lookups,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            hit_rate=hit_rate,
            total_cost_saved_usd=self.total_cost_saved_usd,
        )

    def _find_baseline_hit(self, query_embedding: np.ndarray) -> tuple[UUID | None, float | None]:
        candidates = self.embedding_service.search(query_embedding, self.config.candidate_k)
        if not candidates:
            return None, None
        best = candidates[0]
        if best.score >= self.config.cosine_threshold:
            return best.entry_id, best.score
        return None, best.score

    async def _call_llm(self, llm_function: LLMFunction, prompt: str) -> str:
        value = llm_function(prompt)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, str):
            msg = f"llm_function must return str, got {type(value).__name__}"
            raise TypeError(msg)
        return value

    def _evict_if_needed(self) -> None:
        evicted = self.store.evict(
            policy=self.config.eviction_policy,
            max_entries=self.config.max_entries,
            ttl_seconds=self.config.ttl_seconds,
        )
        for entry_id in evicted:
            self.embedding_service.remove(entry_id)

    def _rebuild_index_from_store(self) -> None:
        items = [
            (entry.id, np.asarray(entry.prompt_embedding, dtype=np.float32))
            for entry in self.store.all_entries()
        ]
        self.embedding_service.rebuild(items)

    def _elapsed_ms(self, started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000
