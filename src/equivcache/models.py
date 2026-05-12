"""Pydantic models for the public EquivCache API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvictionPolicy(StrEnum):
    """Supported cache eviction policies."""

    LRU = "lru"
    TTL = "ttl"
    HYBRID = "hybrid"


class CacheConfig(BaseModel):
    """Configuration for a local EquivCache instance."""

    model_config = ConfigDict(frozen=True)

    db_path: Path = Field(default=Path(".equivcache/cache.db"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)
    candidate_k: int = Field(default=5, gt=0)
    cosine_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    classifier_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_entries: int = Field(default=10_000, gt=0)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ttl_seconds: int | None = Field(default=None, gt=0)
    estimated_llm_cost_usd: Decimal = Decimal("0")

    @field_validator("estimated_llm_cost_usd")
    @classmethod
    def validate_cost(cls, value: Decimal) -> Decimal:
        if value < 0:
            msg = "estimated_llm_cost_usd must be non-negative"
            raise ValueError(msg)
        return value


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


class CacheResult(BaseModel):
    """Result returned by `EquivCache.get_or_call`."""

    query_id: UUID
    response: str
    was_cache_hit: bool
    cache_entry_id: UUID | None = None
    similarity_score: float | None = None
    classifier_score: float | None = None
    cost_saved_usd: Decimal = Decimal("0")
    latency_ms: float = Field(ge=0)


class CacheStats(BaseModel):
    """Runtime and persistence statistics for a cache instance."""

    domain: str
    total_entries: int
    total_lookups: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_cost_saved_usd: Decimal
