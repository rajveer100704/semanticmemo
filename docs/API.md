# SemanticMemo API Reference

This document describes the public interface and configuration objects of the SemanticMemo library.

---

## Configuration Models

### 1. `CrossEncoderConfig`
Parameters for the second-stage Cross-Encoder model.
```python
from pydantic import BaseModel
from typing import Path

class CrossEncoderConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Recommended alternatives: cross-encoder/ms-marco-MiniLM-L-12-v2, cross-encoder/stsb-roberta-base
    device: str = "cpu"
    threshold: float | None = None  # Overrides risk policy if set
```

### 2. `RiskPolicy`
Defines mapping of domains to risk tiers and threshold configurations.
```python
from enum import StrEnum
from pydantic import BaseModel, Field

class RiskTier(StrEnum):
    LOW = "low"
    HIGH = "high"

class RiskPolicy(BaseModel):
    domain_tiers: dict[str, RiskTier] = Field(default_factory=dict)
    default_tier: RiskTier = RiskTier.LOW
    
    # Threshold settings
    low_risk_classifier_threshold: float = 0.90
    low_risk_cross_encoder_threshold: float = 0.85
    
    high_risk_classifier_threshold: float = 0.99
    high_risk_cross_encoder_threshold: float = 0.95
```

### 3. `CacheConfig`
Extends the original config to include the upgrades:
```python
from pathlib import Path
from pydantic import BaseModel, Field

class CacheConfig(BaseModel):
    db_path: Path = Path(".semanticmemo/cache.db")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    candidate_k: int = 5
    cosine_threshold: float = 0.90
    classifier_threshold: float = 0.85
    max_entries: int = 10_000
    eviction_policy: str = "lru"
    ttl_seconds: int | None = None
    
    # Dynamic LLM cost mappings (cost per 1k input/output tokens)
    model_costs: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "default": {"input": 0.0015, "output": 0.002}
        }
    )
    estimated_llm_cost_usd: float = 0.002  # Static fallback if model_costs is not matching
    
    # SemanticMemo Upgrades
    vector_store_type: str = "faiss"  # "faiss" | "qdrant" | "in_memory"
    qdrant_path: str | None = ".semanticmemo/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "semanticmemo"
    
    cross_encoder: CrossEncoderConfig | None = None
    risk_policy: RiskPolicy | None = None
    high_precision_skip_threshold: float = 0.995
```

---

## Result and Decision Models

### 1. `CacheDecision`
Contains detailed explainability about how a cache hit/miss decision was reached.
```python
from pydantic import BaseModel

class CacheDecision(BaseModel):
    decision: str  # "hit" | "miss"
    embedding_score: float | None
    classifier_score: float | None
    cross_encoder_score: float | None
    risk_tier: str  # "low" | "high"
    reason: str  # e.g., "passed_all_thresholds", "failed_cross_encoder", "mlp_bypass"
```

### 2. `CacheResult`
Returned by `SemanticMemo.get_or_call`.
```python
from pydantic import BaseModel
from uuid import UUID

class CacheResult(BaseModel):
    query_id: UUID
    response: str
    was_cache_hit: bool
    decision: CacheDecision
    tokens_saved: int
    cost_saved_usd: float
    latency_ms: float
```

---

## Main Facade: `SemanticMemo`

```python
class SemanticMemo:
    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        classifier: ClassifierConfig | None = None,
        store: SQLiteCacheStore | None = None,
    ) -> None:
        """Initializes the SemanticMemo cache instance."""
        ...

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: Callable[[str], str | Awaitable[str]],
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        """Looks up prompt in cache or calls LLM fallback if it is a miss."""
        ...

    async def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        """Reports a false-positive cache hit."""
        ...

    async def report_good_hit(self, query_id: UUID) -> bool:
        """Reports a valid cache hit."""
        ...

    def export_feedback_pairs(self, path: Path | str, *, split: str = "train") -> int:
        """Exports feedback pairs along with classifier/cross-encoder disagreements."""
        ...
```


