from __future__ import annotations

from decimal import Decimal

import pytest

from equivcache import CacheConfig, EvictionPolicy


def test_cache_config_defaults() -> None:
    config = CacheConfig()

    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.candidate_k == 5
    assert config.cosine_threshold == 0.90
    assert config.classifier_threshold == 0.85
    assert config.eviction_policy == EvictionPolicy.LRU


def test_cache_config_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="estimated_llm_cost_usd"):
        CacheConfig(estimated_llm_cost_usd=Decimal("-0.01"))
