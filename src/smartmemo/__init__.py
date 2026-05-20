"""SmartMemo public API."""

from smartmemo import _logging as _logging  # noqa: F401  (installs the NullHandler)
from smartmemo.cache import SmartMemo
from smartmemo.feedback import RetrainConfig, RetrainResult, retrain_from_feedback
from smartmemo.models import (
    CacheConfig,
    CacheEntry,
    CacheResult,
    CacheStats,
    ClassifierConfig,
    EvictionPolicy,
    FeedbackEvent,
    ImplicitFeedbackConfig,
    LookupRecord,
    RetryConfig,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "ClassifierConfig",
    "SmartMemo",
    "EvictionPolicy",
    "FeedbackEvent",
    "ImplicitFeedbackConfig",
    "LookupRecord",
    "RetrainConfig",
    "RetrainResult",
    "RetryConfig",
    "retrain_from_feedback",
]
