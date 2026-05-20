"""SmartMemo public API."""

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
    "retrain_from_feedback",
]
