"""SemanticMemo public API."""

__version__ = "1.0.0"


from semanticmemo import _logging as _logging  # noqa: F401  (installs the NullHandler)
from semanticmemo.cache import SemanticMemo
from semanticmemo.entity_change_detection import EntityChangeConfig, EntityChangeResult
from semanticmemo.feedback import RetrainConfig, RetrainResult, retrain_from_feedback
from semanticmemo.models import (
    CacheConfig,
    CacheDecision,
    CacheEntry,
    CacheResult,
    CacheStats,
    ClassifierConfig,
    CrossEncoderConfig,
    EvictionPolicy,
    FeedbackEvent,
    ImplicitFeedbackConfig,
    LookupRecord,
    RetryConfig,
    RiskPolicy,
    RiskTier,
)

__all__ = [
    "CacheConfig",
    "CacheDecision",
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "ClassifierConfig",
    "SemanticMemo",
    "EntityChangeConfig",
    "EntityChangeResult",
    "EvictionPolicy",
    "FeedbackEvent",
    "ImplicitFeedbackConfig",
    "LookupRecord",
    "RetrainConfig",
    "RetrainResult",
    "RetryConfig",
    "CrossEncoderConfig",
    "RiskPolicy",
    "RiskTier",
    "retrain_from_feedback",
]
