from __future__ import annotations

from decimal import Decimal

import pytest

from smartmemo import CacheConfig, EvictionPolicy, ImplicitFeedbackConfig


def test_cache_config_defaults() -> None:
    config = CacheConfig()

    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.candidate_k == 5
    assert config.cosine_threshold == 0.90
    assert config.classifier_threshold == 0.85
    assert config.eviction_policy == EvictionPolicy.LRU
    assert config.implicit_feedback is None


def test_cache_config_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="estimated_llm_cost_usd"):
        CacheConfig(estimated_llm_cost_usd=Decimal("-0.01"))


def test_implicit_feedback_config_defaults() -> None:
    config = ImplicitFeedbackConfig()

    assert config.window_seconds == 30.0
    assert config.match == "exact"


def test_implicit_feedback_config_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        ImplicitFeedbackConfig(window_seconds=0)


def test_cache_config_accepts_implicit_feedback() -> None:
    config = CacheConfig(implicit_feedback=ImplicitFeedbackConfig(window_seconds=15.0))

    assert config.implicit_feedback is not None
    assert config.implicit_feedback.window_seconds == 15.0
