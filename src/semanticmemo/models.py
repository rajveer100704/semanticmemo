"""Pydantic models for the public SemanticMemo API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvictionPolicy(StrEnum):
    """Supported cache eviction policies."""

    LRU = "lru"
    TTL = "ttl"
    HYBRID = "hybrid"


class ImplicitFeedbackConfig(BaseModel):
    """Configuration for implicit (re-issue) feedback detection.

    When a cache instance is configured with this, SemanticMemo treats re-issuing
    the *same* prompt shortly after a cache hit as an implicit signal that the
    cached answer was unsatisfactory, and auto-records a bad-hit feedback event
    for the earlier hit. The feature is opt-in: pass an instance to
    ``CacheConfig(implicit_feedback=ImplicitFeedbackConfig())``; the default
    ``CacheConfig`` leaves it disabled.
    """

    model_config = ConfigDict(frozen=True)

    window_seconds: float = Field(default=30.0, gt=0)
    match: Literal["exact"] = "exact"


class RetryConfig(BaseModel):
    """Opt-in retry policy for the user-supplied LLM function.

    Off by default. Pass an instance to ``CacheConfig(retry=RetryConfig())`` to
    retry transient failures of ``llm_function`` with bounded exponential
    backoff. Only the cache-miss path is retried -- a cache hit never calls the
    LLM, so it is never retried. When all attempts are exhausted, SemanticMemo
    raises :class:`~SemanticMemo.exceptions.LLMCallError`, chaining the last
    underlying failure as its cause. With ``CacheConfig.retry`` left ``None``,
    the LLM call behaves exactly as before: a single attempt, exceptions raised
    unchanged.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1)
    initial_backoff_seconds: float = Field(default=0.5, gt=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_backoff_seconds: float = Field(default=30.0, gt=0)
    retry_on: tuple[type[Exception], ...] = (Exception,)


class CrossEncoderConfig(BaseModel):
    """Configuration for the second-stage Cross-Encoder model."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    threshold: float | None = None


class RiskTier(StrEnum):
    """Risk tiers for domain safety policies."""

    LOW = "low"
    HIGH = "high"


class RiskPolicy(BaseModel):
    """Mapping of domains to risk tiers and adaptive thresholds."""

    model_config = ConfigDict(frozen=True)

    domain_tiers: dict[str, RiskTier] = Field(default_factory=dict)
    domain_thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    default_tier: RiskTier = RiskTier.LOW

    # Threshold settings
    low_risk_classifier_threshold: float = 0.90
    low_risk_cross_encoder_threshold: float = 0.85

    high_risk_classifier_threshold: float = 0.99
    high_risk_cross_encoder_threshold: float = 0.95

    prevent_opposite_actions: bool = True


class CacheDecision(BaseModel):
    """Explainability trace for a cache lookup decision."""

    model_config = ConfigDict(frozen=True)

    decision: str  # "hit" | "miss"
    embedding_score: float | None = None
    classifier_score: float | None = None
    cross_encoder_score: float | None = None
    risk_tier: str
    reason: str
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    mlp_latency_ms: float = 0.0
    cross_encoder_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class CacheConfig(BaseModel):
    """Configuration for a local SemanticMemo instance."""

    model_config = ConfigDict(frozen=True)

    db_path: Path = Field(default=Path(".SemanticMemo/cache.db"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)
    candidate_k: int = Field(default=5, gt=0)
    cosine_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    classifier_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_entries: int = Field(default=10_000, gt=0)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ttl_seconds: int | None = Field(default=None, gt=0)
    estimated_llm_cost_usd: Decimal = Decimal("0")
    implicit_feedback: ImplicitFeedbackConfig | None = None
    retry: RetryConfig | None = None

    # Dynamic LLM cost mappings (cost per 1k input/output tokens)
    model_costs: dict[str, dict[str, Decimal]] = Field(
        default_factory=lambda: {
            "gpt-4o": {"input": Decimal("0.005"), "output": Decimal("0.015")},
            "claude-3-5-sonnet": {"input": Decimal("0.003"), "output": Decimal("0.015")},
            "default": {"input": Decimal("0.0015"), "output": Decimal("0.002")},
        }
    )

    # SemanticMemo upgrades
    vector_store_type: str = "faiss"  # "faiss", "qdrant", or "in_memory"
    qdrant_path: str | None = ".SemanticMemo/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "semanticmemo"

    cross_encoder: CrossEncoderConfig | None = None
    risk_policy: RiskPolicy | None = None
    high_precision_skip_threshold: float = 0.995

    @field_validator("estimated_llm_cost_usd")
    @classmethod
    def validate_cost(cls, value: Decimal) -> Decimal:
        if value < 0:
            msg = "estimated_llm_cost_usd must be non-negative"
            raise ValueError(msg)
        return value


class ClassifierConfig(BaseModel):
    """Configuration for a learned equivalence classifier."""

    model_config = ConfigDict(frozen=True)

    model_path: Path | None = None
    device: str = "cpu"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def bundled(
        cls,
        *,
        device: str = "cpu",
        threshold: float | None = 0.95,
    ) -> ClassifierConfig:
        """Return config for the pretrained classifier shipped with SemanticMemo.

        This is the opt-in, zero-training path: pass the result to
        ``SemanticMemo(..., classifier=ClassifierConfig.bundled())`` and the learned
        classifier gates every cache hit. The default ``threshold`` of 0.95 was
        tuned on a hand-curated gold set for precision-first operation, since a
        false-positive cache hit is the failure mode that matters. Using the
        bundled classifier requires the optional ML dependencies
        (``pip install SemanticMemo[ml]``).

        Raises:
            SemanticMemoError: if the checkpoint is missing from this installation.
        """

        from semanticmemo.resources import bundled_classifier_path

        return cls(
            model_path=bundled_classifier_path(),
            device=device,
            threshold=threshold,
        )


class CacheEntry(BaseModel):
    """A persisted prompt/response pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    prompt: str
    prompt_embedding: list[float]
    response: str
    model: str | None = None
    created_at: datetime
    last_hit_at: datetime | None = None
    hit_count: int = 0
    feedback_negative_count: int = 0
    feedback_positive_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LookupRecord(BaseModel):
    """A cache-hit lookup that can receive later feedback."""

    id: UUID
    domain: str
    prompt: str
    prompt_embedding: list[float]
    cache_entry_id: UUID
    similarity_score: float | None = None
    classifier_score: float | None = None
    cross_encoder_score: float | None = None
    created_at: datetime


class FeedbackEvent(BaseModel):
    """Durable feedback attached to one cache-hit lookup."""

    id: UUID
    query_id: UUID
    cache_entry_id: UUID
    label: int
    reason: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheResult(BaseModel):
    """Result returned by `SemanticMemo.get_or_call`."""

    query_id: UUID
    response: str
    was_cache_hit: bool
    cache_entry_id: UUID | None = None
    similarity_score: float | None = None
    classifier_score: float | None = None
    cross_encoder_score: float | None = None
    cost_saved_usd: Decimal = Decimal("0")
    latency_ms: float = Field(ge=0)
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    mlp_latency_ms: float = 0.0
    cross_encoder_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    implicit_bad_hit_recorded: bool = False
    decision: CacheDecision | None = None
    tokens_saved: int = 0


class CacheStats(BaseModel):
    """Runtime and persistence statistics for a cache instance."""

    domain: str
    total_entries: int
    total_lookups: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_cost_saved_usd: Decimal
