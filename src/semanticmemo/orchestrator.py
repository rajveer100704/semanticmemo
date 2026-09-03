"""Cache lookup orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from semanticmemo.classifier.cross_encoder_service import CrossEncoderService
    from semanticmemo.domain_detector import DomainDetector

import numpy as np

from semanticmemo._logging import get_logger
from semanticmemo.concurrency import (
    InProcessSingleFlightManager,
    SingleFlightManager,
    build_flight_key,
)
from semanticmemo.embedding import EmbeddingService
from semanticmemo.embedding.service import SearchCandidate
from semanticmemo.entity_change_detection import EntityChangeConfig, EntityChangeDetector
from semanticmemo.exceptions import LLMCallError
from semanticmemo.metrics import MetricsCollector, estimate_tokens
from semanticmemo.models import (
    CacheConfig,
    CacheDecision,
    CacheEntry,
    CacheResult,
    CacheScope,
    CacheStats,
    RetrievalScope,
    RiskPolicy,
    RiskTier,
)
from semanticmemo.rbac import AuthorizationService, CacheScopeResolver, ResourceService
from semanticmemo.store import MetadataStore
from semanticmemo.telemetry import SemanticMemoMetrics, get_tracer
from semanticmemo.types import EquivalenceClassifier, FloatVector, LLMFunction

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(tz=UTC)


class CacheOrchestrator:
    """Coordinates embedding search, baseline matching, storage, and stats."""

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig,
        store: MetadataStore | Any,
        embedding_service: EmbeddingService,
        classifier_service: EquivalenceClassifier | None = None,
        cross_encoder_service: CrossEncoderService | None = None,
        domain_detector: DomainDetector | None = None,
        entity_change_config: EntityChangeConfig | None = None,
        authz_service: AuthorizationService | None = None,
        resource_service: ResourceService | None = None,
        scope_resolver: CacheScopeResolver | None = None,
        single_flight_manager: SingleFlightManager | None = None,
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

        self.authz_service = authz_service or AuthorizationService(store)
        self.resource_service = resource_service or ResourceService(store)
        self.scope_resolver = scope_resolver or CacheScopeResolver(
            self.authz_service,
            self.resource_service,
            pipeline_version=config.pipeline_version,
        )

        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved_usd = Decimal("0")

        self.single_flight_manager = single_flight_manager or InProcessSingleFlightManager()

        self._rebuild_index_from_store()

    def bootstrap_default_user(
        self,
        user_id: str = "default_user",
        tenant_id: str = "default",
        role: str = "analyst",
    ) -> Any:
        """Explicit development/testing bootstrap helper to seed an authenticated principal."""
        user = self.store.create_user(user_id, tenant_id)
        self.store.assign_role_to_user(user_id, role)
        return user

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        user_id: str | None = None,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        prefer_global: bool = False,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        """Lookup or execute LLM with pre-retrieval RBAC and fail-closed scope resolution.

        Note on `tenant_id`: For security, tenant context is always authoritatively
        derived server-side from the authenticated principal (`user.tenant_id`) rather
        than client request arguments, preventing tenant impersonation. The `tenant_id`
        parameter is accepted for legacy compatibility and default user seeding only.
        """
        started_at = time.perf_counter()
        query_id = uuid4()
        self.total_lookups += 1
        prompt = prompt.strip()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        tracer = get_tracer()
        metrics = SemanticMemoMetrics.default()

        # Step 1 & 2: Authenticate caller & derive tenant context (fail-closed)
        if not user_id:
            msg = "Authentication required: authenticated user_id must be provided."
            logger.warning(msg)
            raise PermissionError(msg)

        user = self.authz_service.authenticate_user(user_id)
        if user is None:
            msg = f"User not authenticated or does not exist: {user_id}"
            logger.warning(msg)
            raise PermissionError(msg)

        root_attrs = {
            "prompt_hash": prompt_hash,
            "prompt_length": len(prompt),
            "domain": self.domain,
            "user_hash": user.id,
            "tenant_hash": user.tenant_id,
        }

        with tracer.start_span("semanticmemo.get_or_call", attributes=root_attrs) as root_span:
            # Step 3, 4, 5: Current RBAC Authorization & Fail-Closed Scope Resolution
            with tracer.start_span("semanticmemo.auth.resolve_scope"):
                retrieval_scope = self.scope_resolver.resolve_scope(
                    user_id=user.id,
                    resource_id=resource_id,
                    prefer_global=prefer_global,
                )

            scope_str = (
                retrieval_scope.scope.value
                if hasattr(retrieval_scope.scope, "value")
                else str(retrieval_scope.scope)
            )
            backend_str = getattr(self.store, "backend_name", "relational")

            implicit_bad_hit_recorded = self._detect_implicit_bad_hit(
                prompt, retrieval_scope=retrieval_scope
            )

            # Step 6: Query Embedding
            t_embed_start = time.perf_counter()
            with tracer.start_span("semanticmemo.embedding.embed"):
                query_embedding = self.embedding_service.embed(prompt)
            embedding_latency = (time.perf_counter() - t_embed_start) * 1000
            metrics.request_duration_seconds.observe(
                embedding_latency / 1000.0, labels={"stage": "embed"}
            )

            # Step 7: Scoped Vector Search & Equivalence Verification
            with tracer.start_span("semanticmemo.vector.search"):
                (
                    hit_entry_id,
                    similarity_score,
                    classifier_score,
                    cross_encoder_score,
                    decision_obj,
                ) = self._find_hit(
                    prompt,
                    query_embedding,
                    embedding_latency,
                    retrieval_scope=retrieval_scope,
                )

            # Step 8: Cache Freshness Revalidation (Invariant 8)
            if hit_entry_id is not None:
                entry = self.store.get(hit_entry_id)
                if entry is not None and self._is_entry_valid(entry, retrieval_scope):
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

                    cost_saved = self.metrics_collector.calculate_cost(
                        prompt, entry.response, model
                    )
                    tokens_saved = estimate_tokens(entry.response)
                    self.total_cost_saved_usd += cost_saved

                    # Metrics updates
                    metrics.cache_requests_total.inc(
                        labels={
                            "scope": scope_str,
                            "decision": "hit",
                            "domain": self.domain,
                            "backend": backend_str,
                        }
                    )
                    metrics.tokens_saved_total.inc(tokens_saved, labels={"type": "output"})
                    metrics.cost_saved_usd_total.inc(float(cost_saved))

                    elapsed_s = time.perf_counter() - started_at
                    metrics.request_duration_seconds.observe(elapsed_s, labels={"stage": "total"})
                    root_span.set_attribute("semanticmemo.was_cache_hit", True)
                    root_span.set_attribute("semanticmemo.cost_saved_usd", float(cost_saved))

                    logger.debug(
                        "cache hit: query_id=%s entry_id=%s similarity=%s "
                        "classifier=%s cross_encoder=%s scope=%s",
                        query_id,
                        entry.id,
                        similarity_score,
                        classifier_score,
                        cross_encoder_score,
                        retrieval_scope.scope,
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
                        retrieval_latency_ms=decision_obj.retrieval_latency_ms
                        if decision_obj
                        else 0.0,
                        mlp_latency_ms=decision_obj.mlp_latency_ms if decision_obj else 0.0,
                        cross_encoder_latency_ms=decision_obj.cross_encoder_latency_ms
                        if decision_obj
                        else 0.0,
                        total_latency_ms=decision_obj.total_latency_ms if decision_obj else 0.0,
                        implicit_bad_hit_recorded=implicit_bad_hit_recorded,
                        decision=decision_obj,
                        tokens_saved=tokens_saved,
                    )

            # Cache Miss Path with SingleFlightManager (InProcess or Distributed Redis)
            flight_key = build_flight_key(
                pipeline_version=retrieval_scope.pipeline_version,
                scope=scope_str,
                tenant_id=retrieval_scope.tenant_id,
                resource_id=retrieval_scope.resource_id,
                resource_version=retrieval_scope.resource_version,
                prompt=prompt,
            )

            can_write = self.authz_service.can_write_cache(user.id, resource_id)
            entry_id: UUID | None = None

            async def _execute_llm() -> str:
                t_llm_start = time.perf_counter()
                with tracer.start_span("semanticmemo.llm.call"):
                    resp = await self._call_llm(llm_function, prompt)
                llm_duration = time.perf_counter() - t_llm_start
                metrics.request_duration_seconds.observe(llm_duration, labels={"stage": "llm_call"})
                metrics.single_flight_total.inc(labels={"role": "leader"})
                return resp

            async def _load_cached() -> str | None:
                metrics.single_flight_total.inc(labels={"role": "follower"})
                c_entry = self.store.find_by_prompt(
                    prompt=prompt,
                    tenant_id=retrieval_scope.tenant_id,
                    scope=retrieval_scope.scope,
                    resource_id=retrieval_scope.resource_id,
                    resource_version=retrieval_scope.resource_version,
                    pipeline_version=retrieval_scope.pipeline_version,
                )
                return c_entry.response if c_entry else None

            async def _write_cached(fresh_resp: str, fencing_generation: int = 0) -> None:
                nonlocal entry_id
                if not can_write:
                    logger.info(
                        "Cache write omitted: user %s lacks write permission for resource %s",
                        user.id,
                        resource_id,
                    )
                    return

                expires_at = (
                    _now() + timedelta(seconds=self.config.ttl_seconds)
                    if self.config.ttl_seconds is not None
                    else None
                )
                with tracer.start_span("semanticmemo.store.write_scoped"):
                    entry_id = self.store.add(
                        prompt=prompt,
                        embedding=query_embedding,
                        response=fresh_resp,
                        model=model,
                        tenant_id=retrieval_scope.tenant_id,
                        scope=retrieval_scope.scope,
                        resource_id=retrieval_scope.resource_id,
                        resource_version=retrieval_scope.resource_version,
                        pipeline_version=retrieval_scope.pipeline_version,
                        expires_at=expires_at,
                        metadata=metadata,
                        flight_generation=fencing_generation,
                    )
                    if entry_id is not None:
                        self.embedding_service.add(
                            entry_id=entry_id,
                            embedding=query_embedding,
                            tenant_id=retrieval_scope.tenant_id,
                            scope=retrieval_scope.scope,
                            resource_id=retrieval_scope.resource_id,
                            resource_version=retrieval_scope.resource_version,
                            pipeline_version=retrieval_scope.pipeline_version,
                        )
                self._evict_if_needed()

            with tracer.start_span("semanticmemo.single_flight"):
                response = await self.single_flight_manager.execute(
                    flight_key=flight_key,
                    operation=_execute_llm,
                    result_loader=_load_cached,
                    result_writer=_write_cached,
                )

            self.cache_misses += 1
            metrics.cache_requests_total.inc(
                labels={
                    "scope": scope_str,
                    "decision": "miss",
                    "domain": self.domain,
                    "backend": backend_str,
                }
            )
            elapsed_s = time.perf_counter() - started_at
            metrics.request_duration_seconds.observe(elapsed_s, labels={"stage": "total"})
            root_span.set_attribute("semanticmemo.was_cache_hit", False)

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
                cross_encoder_latency_ms=decision_obj.cross_encoder_latency_ms
                if decision_obj
                else 0.0,
                total_latency_ms=decision_obj.total_latency_ms if decision_obj else 0.0,
                implicit_bad_hit_recorded=implicit_bad_hit_recorded,
                decision=decision_obj,
                tokens_saved=0,
            )

    def _is_entry_valid(self, entry: CacheEntry, scope: RetrievalScope) -> bool:
        """Validate that a cached entry is fresh, unexpired, and scope-consistent."""
        # TTL expiration check
        if entry.expires_at is not None and entry.expires_at < _now():
            return False

        # Pipeline version check
        if entry.pipeline_version != scope.pipeline_version:
            return False

        # Resource version check
        if scope.scope == CacheScope.RESOURCE:
            return (
                entry.scope == CacheScope.RESOURCE
                and entry.tenant_id == scope.tenant_id
                and entry.resource_id == scope.resource_id
                and entry.resource_version == scope.resource_version
            )
        elif scope.scope == CacheScope.TENANT:
            return entry.scope in (CacheScope.TENANT, CacheScope.GLOBAL) and (
                entry.scope == CacheScope.GLOBAL or entry.tenant_id == scope.tenant_id
            )
        elif scope.scope == CacheScope.GLOBAL:
            return entry.scope == CacheScope.GLOBAL

        return False

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
                tenant_id=cache_entry.tenant_id,
                resource_id=cache_entry.resource_id,
                pipeline_version=cache_entry.pipeline_version,
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

    def _detect_implicit_bad_hit(
        self,
        prompt: str,
        retrieval_scope: RetrievalScope | None = None,
    ) -> bool:
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
            tenant_id=retrieval_scope.tenant_id if retrieval_scope else None,
            resource_id=retrieval_scope.resource_id if retrieval_scope else None,
            pipeline_version=retrieval_scope.pipeline_version if retrieval_scope else None,
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
                source="implicit_reissue_bad_hit",
                tenant_id=cache_entry.tenant_id,
                resource_id=cache_entry.resource_id,
                pipeline_version=cache_entry.pipeline_version,
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
        self,
        prompt: str,
        query_embedding: FloatVector,
        embedding_latency: float,
        retrieval_scope: RetrievalScope,
    ) -> tuple[UUID | None, float | None, float | None, float | None, CacheDecision]:
        t_retrieval_start = time.perf_counter()
        candidates = self.embedding_service.search(
            query_embedding,
            self.config.candidate_k,
            retrieval_scope=retrieval_scope,
        )
        retrieval_latency = (time.perf_counter() - t_retrieval_start) * 1000

        scope_str = (
            retrieval_scope.scope.value
            if hasattr(retrieval_scope.scope, "value")
            else str(retrieval_scope.scope)
        )
        metrics = SemanticMemoMetrics.default()
        metrics.request_duration_seconds.observe(
            retrieval_latency / 1000.0, labels={"stage": "vector_search"}
        )
        store_count = 0
        if hasattr(self.store, "count"):
            try:
                cnt = self.store.count()
                if isinstance(cnt, (int, float)):
                    store_count = int(cnt)
            except Exception:
                store_count = 0
        total_baseline = max(1, store_count)
        reduction_ratio = max(0.0, min(1.0, 1.0 - (len(candidates) / total_baseline)))
        metrics.candidate_space_reduction_ratio.observe(
            reduction_ratio, labels={"scope": scope_str}
        )

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
        mlp_threshold = classifier.threshold
        if mlp_threshold is None:
            mlp_val = domain_cfg.get("mlp")
            if mlp_val is not None:
                mlp_threshold = mlp_val
            else:
                mlp_threshold = (
                    risk_policy.high_risk_classifier_threshold
                    if risk_tier == RiskTier.HIGH
                    else risk_policy.low_risk_classifier_threshold
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
        metrics = SemanticMemoMetrics.default()
        metrics.request_duration_seconds.observe(mlp_latency / 1000.0, labels={"stage": "mlp_gate"})

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
                    tenant_id=best_entry.tenant_id,
                    resource_id=best_entry.resource_id,
                    pipeline_version=best_entry.pipeline_version,
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

        ce_threshold = self.cross_encoder_service.threshold
        if ce_threshold is None:
            domain_cfg = risk_policy.domain_thresholds.get(detected_domain, {})
            ce_val = domain_cfg.get("cross_encoder")
            if ce_val is not None:
                ce_threshold = ce_val
            else:
                ce_threshold = (
                    risk_policy.high_risk_cross_encoder_threshold
                    if risk_tier == RiskTier.HIGH
                    else risk_policy.low_risk_cross_encoder_threshold
                )

        t_ce_start = time.perf_counter()
        ce_score = self.cross_encoder_service.predict(prompt, best_entry.prompt)
        ce_latency = (time.perf_counter() - t_ce_start) * 1000
        metrics.request_duration_seconds.observe(
            ce_latency / 1000.0, labels={"stage": "cross_encoder"}
        )

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
                    tenant_id=best_entry.tenant_id,
                    resource_id=best_entry.resource_id,
                    pipeline_version=best_entry.pipeline_version,
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
            tenant_id=best_entry.tenant_id,
            resource_id=best_entry.resource_id,
            pipeline_version=best_entry.pipeline_version,
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
            (
                entry.id,
                np.asarray(entry.prompt_embedding, dtype=np.float32),
                entry.tenant_id,
                entry.scope,
                entry.resource_id,
                entry.resource_version,
                entry.pipeline_version,
            )
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
