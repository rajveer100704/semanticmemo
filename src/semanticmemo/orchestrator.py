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

from semanticmemo._logging import get_logger
from semanticmemo.classifier.cross_encoder_service import CrossEncoderService
from semanticmemo.domain_detector import DomainDetector
from semanticmemo.embedding import EmbeddingService
from semanticmemo.embedding.service import SearchCandidate
from semanticmemo.entity_change_detection import EntityChangeConfig, EntityChangeDetector
from semanticmemo.exceptions import LLMCallError
from semanticmemo.metrics import MetricsCollector, estimate_tokens
from semanticmemo.models import (
    CacheConfig,
    CacheDecision,
    CacheResult,
    CacheStats,
    RiskPolicy,
    RiskTier,
)
from semanticmemo.store import SQLiteCacheStore
from semanticmemo.types import EquivalenceClassifier, FloatVector, LLMFunction

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
        cross_encoder_service: CrossEncoderService | None = None,
        domain_detector: DomainDetector | None = None,
        entity_change_config: EntityChangeConfig | None = None,
    ) -> None:
        self.domain = domain
        self.config = config
        self.store = store
        self.embedding_service = embedding_service
        self.classifier_service = classifier_service
        self.cross_encoder_service = cross_encoder_service
        self.domain_detector = domain_detector
        self.entity_detector = EntityChangeDetector(entity_change_config)
        self.metrics_collector = MetricsCollector(config)
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

        t_embed_start = time.perf_counter()
        query_embedding = self.embedding_service.embed(prompt)
        embedding_latency = (time.perf_counter() - t_embed_start) * 1000

        hit_entry_id, similarity_score, classifier_score, cross_encoder_score, decision_obj = (
            self._find_hit(prompt, query_embedding, embedding_latency)
        )

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
                    cross_encoder_score=cross_encoder_score,
                )
                self.cache_hits += 1

                # Compute cost saved dynamically
                cost_saved = self.metrics_collector.calculate_cost(prompt, entry.response, model)
                tokens_saved = estimate_tokens(entry.response)
                self.total_cost_saved_usd += cost_saved

                logger.debug(
                    "cache hit: query_id=%s entry_id=%s similarity=%s "
                    "classifier=%s cross_encoder=%s",
                    query_id,
                    entry.id,
                    similarity_score,
                    classifier_score,
                    cross_encoder_score,
                )
                return CacheResult(
                    query_id=query_id,
                    response=entry.response,
                    was_cache_hit=True,
                    cache_entry_id=entry.id,
                    similarity_score=similarity_score,
                    classifier_score=classifier_score,
                    cross_encoder_score=cross_encoder_score,
                    cost_saved_usd=cost_saved,
                    latency_ms=self._elapsed_ms(started_at),
                    embedding_latency_ms=embedding_latency,
                    retrieval_latency_ms=decision_obj.retrieval_latency_ms if decision_obj else 0.0,
                    mlp_latency_ms=decision_obj.mlp_latency_ms if decision_obj else 0.0,
                    cross_encoder_latency_ms=decision_obj.cross_encoder_latency_ms
                    if decision_obj
                    else 0.0,
                    total_latency_ms=decision_obj.total_latency_ms if decision_obj else 0.0,
                    implicit_bad_hit_recorded=implicit_bad_hit_recorded,
                    decision=decision_obj,
                    tokens_saved=tokens_saved,
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
            cross_encoder_score=cross_encoder_score,
            cost_saved_usd=Decimal("0"),
            latency_ms=self._elapsed_ms(started_at),
            embedding_latency_ms=embedding_latency,
            retrieval_latency_ms=decision_obj.retrieval_latency_ms if decision_obj else 0.0,
            mlp_latency_ms=decision_obj.mlp_latency_ms if decision_obj else 0.0,
            cross_encoder_latency_ms=decision_obj.cross_encoder_latency_ms if decision_obj else 0.0,
            total_latency_ms=decision_obj.total_latency_ms if decision_obj else 0.0,
            implicit_bad_hit_recorded=implicit_bad_hit_recorded,
            decision=decision_obj,
            tokens_saved=0,
        )

    def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        lookup = self.store.get_lookup(query_id)
        if lookup is None:
            return False
        event_id = self.store.record_feedback(query_id=query_id, label=0, reason=reason)
        if event_id is None:
            return False
        self.store.increment_bad_feedback(lookup.cache_entry_id)
        
        # Record bad hit for active learning
        cache_entry = self.store.get(lookup.cache_entry_id)
        if cache_entry is not None:
            self.store.record_active_learning_pair(
                domain=lookup.domain,
                query_prompt=lookup.prompt,
                cached_prompt=cache_entry.prompt,
                similarity_score=lookup.similarity_score or 0.0,
                classifier_score=lookup.classifier_score or 0.0,
                cross_encoder_score=lookup.cross_encoder_score or 0.0,
                label=0,
                source="user_reported_bad_hit",
            )

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

        # Record bad hit for active learning
        cache_entry = self.store.get(lookup.cache_entry_id)
        if cache_entry is not None:
            self.store.record_active_learning_pair(
                domain=lookup.domain,
                query_prompt=lookup.prompt,
                cached_prompt=cache_entry.prompt,
                similarity_score=lookup.similarity_score or 0.0,
                classifier_score=lookup.classifier_score or 0.0,
                cross_encoder_score=lookup.cross_encoder_score or 0.0,
                label=0,
                source="user_reported_bad_hit",
            )

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
        self, prompt: str, query_embedding: FloatVector, embedding_latency: float
    ) -> tuple[UUID | None, float | None, float | None, float | None, CacheDecision]:
        t_retrieval_start = time.perf_counter()
        candidates = self.embedding_service.search(query_embedding, self.config.candidate_k)
        retrieval_latency = (time.perf_counter() - t_retrieval_start) * 1000

        if not candidates:
            decision_obj = CacheDecision(
                decision="miss",
                risk_tier="low",
                reason="no_candidates_found",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                total_latency_ms=embedding_latency + retrieval_latency,
            )
            return None, None, None, None, decision_obj

        if self.domain_detector is not None:
            detected_domain = self.domain_detector.detect(query_embedding)
        else:
            detected_domain = self.domain

        risk_policy = self.config.risk_policy or RiskPolicy()
        risk_tier = risk_policy.domain_tiers.get(detected_domain, risk_policy.default_tier)

        if self.classifier_service is not None:
            return self._find_classifier_hit(
                prompt,
                query_embedding,
                candidates,
                detected_domain,
                risk_tier,
                risk_policy,
                embedding_latency,
                retrieval_latency,
            )
        return self._find_baseline_hit(candidates, risk_tier, embedding_latency, retrieval_latency)

    def _find_baseline_hit(
        self,
        candidates: list[SearchCandidate],
        risk_tier: RiskTier,
        embedding_latency: float,
        retrieval_latency: float,
    ) -> tuple[UUID | None, float | None, None, None, CacheDecision]:
        best = candidates[0]
        if best.score >= self.config.cosine_threshold:
            decision_obj = CacheDecision(
                decision="hit",
                embedding_score=best.score,
                risk_tier=risk_tier,
                reason="passed_cosine_threshold",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                total_latency_ms=embedding_latency + retrieval_latency,
            )
            return best.entry_id, best.score, None, None, decision_obj

        decision_obj = CacheDecision(
            decision="miss",
            embedding_score=best.score,
            risk_tier=risk_tier,
            reason="failed_cosine_threshold",
            embedding_latency_ms=embedding_latency,
            retrieval_latency_ms=retrieval_latency,
            total_latency_ms=embedding_latency + retrieval_latency,
        )
        return None, best.score, None, None, decision_obj

    def _find_classifier_hit(
        self,
        prompt: str,
        query_embedding: FloatVector,
        candidates: list[SearchCandidate],
        detected_domain: str,
        risk_tier: RiskTier,
        risk_policy: RiskPolicy,
        embedding_latency: float,
        retrieval_latency: float,
    ) -> tuple[UUID | None, float | None, float | None, float | None, CacheDecision]:
        classifier = self.classifier_service
        if classifier is None:
            return self._find_baseline_hit(
                candidates, risk_tier, embedding_latency, retrieval_latency
            )

        entries = [self.store.get(candidate.entry_id) for candidate in candidates]
        scored_candidates = [
            (candidate, entry)
            for candidate, entry in zip(candidates, entries, strict=True)
            if entry is not None
        ]
        if not scored_candidates:
            decision_obj = CacheDecision(
                decision="miss",
                embedding_score=candidates[0].score,
                risk_tier=risk_tier,
                reason="no_candidate_entries_in_store",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                total_latency_ms=embedding_latency + retrieval_latency,
            )
            return None, candidates[0].score, None, None, decision_obj

        # Resolve MLP threshold
        domain_cfg = risk_policy.domain_thresholds.get(detected_domain, {})
        mlp_threshold = (
            classifier.threshold
            if classifier.threshold is not None
            else (
                domain_cfg.get("mlp")
                if "mlp" in domain_cfg
                else (
                    risk_policy.high_risk_classifier_threshold
                    if risk_tier == RiskTier.HIGH
                    else risk_policy.low_risk_classifier_threshold
                )
            )
        )

        t_mlp_start = time.perf_counter()
        probabilities = classifier.predict_batch(
            [
                (
                    query_embedding,
                    np.asarray(entry.prompt_embedding, dtype=np.float32),
                )
                for _, entry in scored_candidates
            ]
        )
        mlp_latency = (time.perf_counter() - t_mlp_start) * 1000

        ranked = sorted(
            zip(scored_candidates, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        (best_candidate, best_entry), best_mlp_score = ranked[0]

        logger.debug(
            "classifier gate: score=%.4f threshold=%.4f decision=%s",
            best_mlp_score,
            mlp_threshold,
            "hit" if best_mlp_score >= mlp_threshold else "miss",
        )

        # Check for opposite actions

        if risk_policy.prevent_opposite_actions and self._is_opposite_action(
            prompt, best_entry.prompt
        ):
            decision_obj = CacheDecision(
                decision="miss",
                embedding_score=best_candidate.score,
                classifier_score=best_mlp_score,
                risk_tier=risk_tier,
                reason="opposite_action_detected",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                mlp_latency_ms=mlp_latency,
                total_latency_ms=embedding_latency + retrieval_latency + mlp_latency,
            )
            return None, best_candidate.score, best_mlp_score, None, decision_obj

        if best_mlp_score < mlp_threshold:
            decision_obj = CacheDecision(
                decision="miss",
                embedding_score=best_candidate.score,
                classifier_score=best_mlp_score,
                risk_tier=risk_tier,
                reason="failed_mlp_threshold",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                mlp_latency_ms=mlp_latency,
                total_latency_ms=embedding_latency + retrieval_latency + mlp_latency,
            )
            return None, best_candidate.score, best_mlp_score, None, decision_obj

        # Latency-Aware Bypassing Check
        if best_mlp_score >= self.config.high_precision_skip_threshold:
            decision_obj = CacheDecision(
                decision="hit",
                embedding_score=best_candidate.score,
                classifier_score=best_mlp_score,
                risk_tier=risk_tier,
                reason="mlp_bypass",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                mlp_latency_ms=mlp_latency,
                total_latency_ms=embedding_latency + retrieval_latency + mlp_latency,
            )
            return best_candidate.entry_id, best_candidate.score, best_mlp_score, None, decision_obj

        if self.cross_encoder_service is None:
            # Still run entity detection even without cross-encoder
            entity_result = self.entity_detector.detect(prompt, best_entry.prompt)
            if entity_result.entity_changed:
                logger.debug(
                    "entity change detected (no CE): %s | tokens: %s",
                    entity_result.reason,
                    entity_result.changed_tokens,
                )
                self.store.record_active_learning_pair(
                    domain=detected_domain,
                    query_prompt=prompt,
                    cached_prompt=best_entry.prompt,
                    similarity_score=best_candidate.score,
                    classifier_score=best_mlp_score,
                    cross_encoder_score=None,
                    label=0,
                    source="entity_change_detected",
                )
                decision_obj = CacheDecision(
                    decision="miss",
                    embedding_score=best_candidate.score,
                    classifier_score=best_mlp_score,
                    risk_tier=risk_tier,
                    reason=f"entity_change_detected:{entity_result.detector}",
                    embedding_latency_ms=embedding_latency,
                    retrieval_latency_ms=retrieval_latency,
                    mlp_latency_ms=mlp_latency,
                    total_latency_ms=embedding_latency + retrieval_latency + mlp_latency,
                )
                return None, best_candidate.score, best_mlp_score, None, decision_obj

            decision_obj = CacheDecision(
                decision="hit",
                embedding_score=best_candidate.score,
                classifier_score=best_mlp_score,
                risk_tier=risk_tier,
                reason="passed_mlp_threshold_no_cross_encoder",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                mlp_latency_ms=mlp_latency,
                total_latency_ms=embedding_latency + retrieval_latency + mlp_latency,
            )
            return best_candidate.entry_id, best_candidate.score, best_mlp_score, None, decision_obj

        ce_threshold = (
            self.cross_encoder_service.threshold
            if self.cross_encoder_service.threshold is not None
            else (
                domain_cfg.get("cross_encoder")
                if "cross_encoder" in domain_cfg
                else (
                    risk_policy.high_risk_cross_encoder_threshold
                    if risk_tier == RiskTier.HIGH
                    else risk_policy.low_risk_cross_encoder_threshold
                )
            )
        )

        t_ce_start = time.perf_counter()
        ce_score = self.cross_encoder_service.predict(prompt, best_entry.prompt)
        ce_latency = (time.perf_counter() - t_ce_start) * 1000

        if ce_score >= ce_threshold:
            # 4th gate: Entity Change Detection
            entity_result = self.entity_detector.detect(prompt, best_entry.prompt)
            if entity_result.entity_changed:
                logger.debug(
                    "entity change detected: %s | tokens: %s",
                    entity_result.reason,
                    entity_result.changed_tokens,
                )
                # Record as active learning pair — high-value hard negative
                self.store.record_active_learning_pair(
                    domain=detected_domain,
                    query_prompt=prompt,
                    cached_prompt=best_entry.prompt,
                    similarity_score=best_candidate.score,
                    classifier_score=best_mlp_score,
                    cross_encoder_score=ce_score,
                    label=0,
                    source="entity_change_detected",
                )
                decision_obj = CacheDecision(
                    decision="miss",
                    embedding_score=best_candidate.score,
                    classifier_score=best_mlp_score,
                    cross_encoder_score=ce_score,
                    risk_tier=risk_tier,
                    reason=f"entity_change_detected:{entity_result.detector}",
                    embedding_latency_ms=embedding_latency,
                    retrieval_latency_ms=retrieval_latency,
                    mlp_latency_ms=mlp_latency,
                    cross_encoder_latency_ms=ce_latency,
                    total_latency_ms=(
                        embedding_latency + retrieval_latency + mlp_latency + ce_latency
                    ),
                )
                return None, best_candidate.score, best_mlp_score, ce_score, decision_obj

            decision_obj = CacheDecision(
                decision="hit",
                embedding_score=best_candidate.score,
                classifier_score=best_mlp_score,
                cross_encoder_score=ce_score,
                risk_tier=risk_tier,
                reason="passed_all_thresholds",
                embedding_latency_ms=embedding_latency,
                retrieval_latency_ms=retrieval_latency,
                mlp_latency_ms=mlp_latency,
                cross_encoder_latency_ms=ce_latency,
                total_latency_ms=embedding_latency + retrieval_latency + mlp_latency + ce_latency,
            )
            return (
                best_candidate.entry_id,
                best_candidate.score,
                best_mlp_score,
                ce_score,
                decision_obj,
            )

        # Record disagreement for active learning (hard negative)
        self.store.record_active_learning_pair(
            domain=detected_domain,
            query_prompt=prompt,
            cached_prompt=best_entry.prompt,
            similarity_score=best_candidate.score,
            classifier_score=best_mlp_score,
            cross_encoder_score=ce_score,
            label=0,
            source="mlp_ce_disagreement",
        )

        decision_obj = CacheDecision(
            decision="miss",
            embedding_score=best_candidate.score,
            classifier_score=best_mlp_score,
            cross_encoder_score=ce_score,
            risk_tier=risk_tier,
            reason="failed_cross_encoder_threshold",
            embedding_latency_ms=embedding_latency,
            retrieval_latency_ms=retrieval_latency,
            mlp_latency_ms=mlp_latency,
            cross_encoder_latency_ms=ce_latency,
            total_latency_ms=embedding_latency + retrieval_latency + mlp_latency + ce_latency,
        )
        return None, best_candidate.score, best_mlp_score, ce_score, decision_obj

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

    def _is_opposite_action(self, prompt_a: str, prompt_b: str) -> bool:
        """Rule-based checker for high-stakes opposite action pairs."""
        a_lower = prompt_a.lower()
        b_lower = prompt_b.lower()

        # 1. Direct opposite pairs
        opposite_pairs = [
            ("approve", "deny"),
            ("approve", "reject"),
            ("allow", "block"),
            ("allow", "deny"),
            ("increase", "decrease"),
            ("raise", "lower"),
            ("buy", "sell"),
            ("enable", "disable"),
            ("grant", "revoke"),
            ("grant", "restrict"),
            ("release", "hold back"),
            ("release", "withhold"),
            ("whitelist", "blacklist"),
            ("encrypt", "decrypt"),
            ("open", "close"),
            ("start", "stop"),
            ("keep", "sell"),
            ("keep", "liquidate"),
            ("deposit", "withdraw"),
            ("backup", "delete"),
            ("backup", "destroy"),
            ("public", "private"),
            ("external", "internal"),
            ("personal", "business"),
            ("personal", "external"),
            ("cryptocurrency", "deposit"),
            ("crypto", "fiat"),
            ("legitimate", "fraudulent"),
            ("turn on", "turn off"),
            ("require", "allow"),
            ("require", "permit"),
            ("complex", "simple"),
            ("complex", "basic"),
        ]
        for w1, w2 in opposite_pairs:
            if (
                ((w1 in a_lower and w2 in b_lower) or (w2 in a_lower and w1 in b_lower))
                and not (w1 in a_lower and w1 in b_lower)
                and not (w2 in a_lower and w2 in b_lower)
            ):
                return True

        # 2. Negations and antonym patterns
        negation_words = ["not", "no", "never", "don't", "cannot", "can't", "refuse"]
        keywords = [
            "required",
            "safe",
            "valid",
            "active",
            "legitimate",
            "binding",
            "enforceable",
            "malignant",
            "benign",
        ]
        for kw in keywords:
            if kw in a_lower and kw in b_lower:
                has_neg_a = any(neg in a_lower for neg in negation_words)
                has_neg_b = any(neg in b_lower for neg in negation_words)
                if has_neg_a != has_neg_b:
                    return True

        # 3. Un- prefix pairs
        un_pairs = [
            ("safe", "unsafe"),
            ("valid", "invalid"),
            ("active", "inactive"),
            ("legitimate", "illegitimate"),
        ]
        for w1, w2 in un_pairs:
            if (
                ((w1 in a_lower and w2 in b_lower) or (w2 in a_lower and w1 in b_lower))
                and not (w1 in a_lower and w1 in b_lower)
                and not (w2 in a_lower and w2 in b_lower)
            ):
                return True

        # 4. Numeric mismatch check (e.g. "$100" vs "$500")
        import re

        digits_a = re.findall(r"\b\d+\b", prompt_a)
        digits_b = re.findall(r"\b\d+\b", prompt_b)
        return bool(digits_a and digits_b and set(digits_a) != set(digits_b))
