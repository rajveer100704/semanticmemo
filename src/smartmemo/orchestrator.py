"""Cache lookup orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from smartmemo._logging import get_logger
from smartmemo.embedding import EmbeddingService
from smartmemo.embedding.service import SearchCandidate
from smartmemo.exceptions import LLMCallError
from smartmemo.models import CacheConfig, CacheResult, CacheStats
from smartmemo.store import SQLiteCacheStore
from smartmemo.types import EquivalenceClassifier, FloatVector, LLMFunction

logger = get_logger(__name__)


class CacheOrchestrator:
    """Coordinates embedding search, baseline matching, storage, and stats."""

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig,
        store: SQLiteCacheStore,
        embedding_service: EmbeddingService,
        classifier_service: EquivalenceClassifier | None = None,
    ) -> None:
        self.domain = domain
        self.config = config
        self.store = store
        self.embedding_service = embedding_service
        self.classifier_service = classifier_service
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved_usd = Decimal("0")
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

        # Strip once so the cache key and the implicit-feedback exact match are
        # whitespace-insensitive and mutually consistent.
        prompt = prompt.strip()

        # Implicit-feedback detection must run before record_lookup writes this
        # call's own lookup row, otherwise a re-issued hit would flag itself.
        implicit_bad_hit_recorded = self._detect_implicit_bad_hit(prompt)

        query_embedding = self.embedding_service.embed(prompt)
        hit_entry_id, similarity_score, classifier_score = self._find_hit(query_embedding)

        if hit_entry_id is not None:
            entry = self.store.get(hit_entry_id)
            if entry is not None:
                self.store.update_hit(entry.id)
                self.store.record_lookup(
                    query_id=query_id,
                    domain=self.domain,
                    prompt=prompt,
                    embedding=query_embedding,
                    cache_entry_id=entry.id,
                    similarity_score=similarity_score,
                    classifier_score=classifier_score,
                )
                self.cache_hits += 1
                self.total_cost_saved_usd += self.config.estimated_llm_cost_usd
                logger.debug(
                    "cache hit: query_id=%s entry_id=%s similarity=%s classifier=%s",
                    query_id,
                    entry.id,
                    similarity_score,
                    classifier_score,
                )
                return CacheResult(
                    query_id=query_id,
                    response=entry.response,
                    was_cache_hit=True,
                    cache_entry_id=entry.id,
                    similarity_score=similarity_score,
                    classifier_score=classifier_score,
                    cost_saved_usd=self.config.estimated_llm_cost_usd,
                    latency_ms=self._elapsed_ms(started_at),
                    implicit_bad_hit_recorded=implicit_bad_hit_recorded,
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
        logger.debug("cache miss: query_id=%s new_entry_id=%s", query_id, entry_id)
        return CacheResult(
            query_id=query_id,
            response=response,
            was_cache_hit=False,
            cache_entry_id=entry_id,
            similarity_score=similarity_score,
            classifier_score=classifier_score,
            cost_saved_usd=Decimal("0"),
            latency_ms=self._elapsed_ms(started_at),
            implicit_bad_hit_recorded=implicit_bad_hit_recorded,
        )

    def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        lookup = self.store.get_lookup(query_id)
        if lookup is None:
            return False
        event_id = self.store.record_feedback(query_id=query_id, label=0, reason=reason)
        if event_id is None:
            return False
        self.store.increment_bad_feedback(lookup.cache_entry_id)
        logger.info("explicit feedback recorded: query_id=%s label=0", query_id)
        return True

    def report_good_hit(self, query_id: UUID) -> bool:
        lookup = self.store.get_lookup(query_id)
        if lookup is None:
            return False
        event_id = self.store.record_feedback(query_id=query_id, label=1)
        if event_id is None:
            return False
        self.store.increment_good_feedback(lookup.cache_entry_id)
        logger.info("explicit feedback recorded: query_id=%s label=1", query_id)
        return True

    def _detect_implicit_bad_hit(self, prompt: str) -> bool:
        """Auto-flag a recent cache hit as bad when its exact prompt is re-issued.

        Opt-in via ``CacheConfig.implicit_feedback``. Re-issuing the same prompt
        soon after a hit is treated as an implicit signal the cached answer was
        unsatisfactory. Returns whether a prior hit was flagged by this call.
        """
        cfg = self.config.implicit_feedback
        if cfg is None:
            return False
        lookup = self.store.find_implicit_bad_hit(
            domain=self.domain,
            prompt=prompt,
            within_seconds=cfg.window_seconds,
        )
        if lookup is None:
            return False
        elapsed = (datetime.now(tz=UTC) - lookup.created_at).total_seconds()
        event_id = self.store.record_feedback(
            query_id=lookup.id,
            label=0,
            reason="implicit:re-issued",
            metadata={
                "auto_detected": True,
                "detector": "re-issue",
                "window_seconds": cfg.window_seconds,
                "elapsed_seconds": elapsed,
            },
        )
        if event_id is None:
            # The lookup row was removed (e.g. cache eviction) between the
            # find and the write; nothing to flag.
            return False
        self.store.increment_bad_feedback(lookup.cache_entry_id)
        logger.info(
            "implicit bad-hit recorded: lookup=%s entry=%s elapsed=%.1fs",
            lookup.id,
            lookup.cache_entry_id,
            elapsed,
        )
        return True

    def export_feedback_pairs(self, path: str, *, split: str = "train") -> int:
        return self.store.export_feedback_pairs(path, split=split)

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

    def _find_hit(
        self, query_embedding: FloatVector
    ) -> tuple[UUID | None, float | None, float | None]:
        candidates = self.embedding_service.search(query_embedding, self.config.candidate_k)
        if not candidates:
            return None, None, None
        if self.classifier_service is not None:
            return self._find_classifier_hit(query_embedding, candidates)
        return self._find_baseline_hit(candidates)

    def _find_baseline_hit(
        self, candidates: list[SearchCandidate]
    ) -> tuple[UUID | None, float | None, None]:
        best = candidates[0]
        if best.score >= self.config.cosine_threshold:
            return best.entry_id, best.score, None
        return None, best.score, None

    def _find_classifier_hit(
        self,
        query_embedding: FloatVector,
        candidates: list[SearchCandidate],
    ) -> tuple[UUID | None, float | None, float | None]:
        classifier = self.classifier_service
        if classifier is None:
            return self._find_baseline_hit(candidates)

        entries = [self.store.get(candidate.entry_id) for candidate in candidates]
        scored_candidates = [
            (candidate, entry)
            for candidate, entry in zip(candidates, entries, strict=True)
            if entry is not None
        ]
        if not scored_candidates:
            return None, candidates[0].score, None

        probabilities = classifier.predict_batch(
            [
                (
                    query_embedding,
                    np.asarray(entry.prompt_embedding, dtype=np.float32),
                )
                for _, entry in scored_candidates
            ]
        )
        ranked = sorted(
            zip(scored_candidates, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        (best_candidate, _), best_classifier_score = ranked[0]
        is_hit = best_classifier_score >= classifier.threshold
        logger.debug(
            "classifier gate: score=%.4f threshold=%.4f decision=%s",
            best_classifier_score,
            classifier.threshold,
            "hit" if is_hit else "miss",
        )
        if is_hit:
            return best_candidate.entry_id, best_candidate.score, best_classifier_score
        best = candidates[0]
        return None, best.score, best_classifier_score

    async def _call_llm(self, llm_function: LLMFunction, prompt: str) -> str:
        """Call the user's LLM function, retrying transient failures if configured.

        With ``config.retry`` unset this is a single attempt and exceptions
        propagate unchanged. With a ``RetryConfig`` it retries on the configured
        exception types using bounded exponential backoff, and raises
        ``LLMCallError`` (chaining the last failure) once attempts are exhausted.
        """
        retry = self.config.retry
        if retry is None:
            return await self._invoke_llm(llm_function, prompt)

        backoff = retry.initial_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(1, retry.max_attempts + 1):
            try:
                return await self._invoke_llm(llm_function, prompt)
            except retry.retry_on as exc:
                last_exc = exc
                if attempt >= retry.max_attempts:
                    break
                logger.warning(
                    "llm_function failed (attempt %d/%d), retrying in %.2fs: %s",
                    attempt,
                    retry.max_attempts,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * retry.backoff_multiplier, retry.max_backoff_seconds)
        msg = f"llm_function failed after {retry.max_attempts} attempt(s)"
        raise LLMCallError(msg) from last_exc

    async def _invoke_llm(self, llm_function: LLMFunction, prompt: str) -> str:
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
        if evicted:
            logger.info("evicted %d cache entries", len(evicted))

    def _rebuild_index_from_store(self) -> None:
        items = [
            (entry.id, np.asarray(entry.prompt_embedding, dtype=np.float32))
            for entry in self.store.all_entries()
        ]
        self.embedding_service.rebuild(items)

    def _elapsed_ms(self, started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000
