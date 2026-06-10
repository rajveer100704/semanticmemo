from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from semanticmemo import CacheConfig, EvictionPolicy, ImplicitFeedbackConfig, RetryConfig


def test_cache_config_defaults() -> None:
    config = CacheConfig()

    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.candidate_k == 5
    assert config.cosine_threshold == 0.90
    assert config.classifier_threshold == 0.85
    assert config.eviction_policy == EvictionPolicy.LRU
    assert config.implicit_feedback is None
    assert config.retry is None


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


def test_retry_config_defaults() -> None:
    config = RetryConfig()

    assert config.max_attempts == 3
    assert config.initial_backoff_seconds == 0.5
    assert config.backoff_multiplier == 2.0
    assert config.max_backoff_seconds == 30.0
    assert config.retry_on == (Exception,)


def test_retry_config_rejects_zero_attempts() -> None:
    with pytest.raises(ValidationError, match="max_attempts"):
        RetryConfig(max_attempts=0)


def test_retry_config_is_frozen() -> None:
    config = RetryConfig()
    with pytest.raises(ValidationError):
        config.max_attempts = 5


def test_cache_config_accepts_retry() -> None:
    config = CacheConfig(retry=RetryConfig(max_attempts=5))

    assert config.retry is not None
    assert config.retry.max_attempts == 5
