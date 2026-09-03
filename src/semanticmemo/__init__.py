"""SemanticMemo public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"


from semanticmemo import _logging as _logging  # noqa: F401  (installs the NullHandler)
from semanticmemo.concurrency import (
    InProcessSingleFlightManager,
    RedisConfig,
    RedisSingleFlightManager,
    SingleFlightManager,
    build_flight_key,
)
from semanticmemo.embedding.qdrant_index import QdrantConfig, QdrantVectorIndex
from semanticmemo.entity_change_detection import EntityChangeConfig, EntityChangeResult
from semanticmemo.events import (
    EventBus,
    EventHandler,
    EventType,
    IndexReconciler,
    InMemoryEventBus,
    InvalidationConsumer,
    KafkaEventBus,
    OutboxEvent,
    OutboxRelay,
    OutboxStatus,
    ProcessedEventRecord,
    RabbitMQEventBus,
    ReconciliationReport,
)
from semanticmemo.models import (
    CacheConfig,
    CacheDecision,
    CacheEntry,
    CacheResult,
    CacheScope,
    CacheStats,
    ClassifierConfig,
    CrossEncoderConfig,
    EvictionPolicy,
    FeedbackEvent,
    ImplicitFeedbackConfig,
    LookupRecord,
    Permission,
    PermissionName,
    Resource,
    RetrievalScope,
    RetryConfig,
    RiskPolicy,
    RiskTier,
    Role,
    User,
)
from semanticmemo.store import (
    CutoverState,
    MetadataStore,
    MigrationReport,
    PostgresCacheStore,
    PostgresConfig,
    SQLiteCacheStore,
    migrate_sqlite_to_postgres,
)
from semanticmemo.telemetry import (
    Counter,
    Gauge,
    Histogram,
    NoOpSpan,
    SemanticMemoMetrics,
    SemanticMemoTracer,
    Span,
    StageTimer,
    get_tracer,
    hash_identifier,
    sanitize_attributes,
)

if TYPE_CHECKING:
    from semanticmemo.cache import SemanticMemo
    from semanticmemo.feedback import RetrainConfig, RetrainResult, retrain_from_feedback

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SemanticMemo": ("semanticmemo.cache", "SemanticMemo"),
    "RetrainConfig": ("semanticmemo.feedback", "RetrainConfig"),
    "RetrainResult": ("semanticmemo.feedback", "RetrainResult"),
    "retrain_from_feedback": ("semanticmemo.feedback", "retrain_from_feedback"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "CacheConfig",
    "CacheDecision",
    "CacheEntry",
    "CacheResult",
    "CacheScope",
    "CacheStats",
    "ClassifierConfig",
    "Counter",
    "CrossEncoderConfig",
    "CutoverState",
    "EntityChangeConfig",
    "EntityChangeResult",
    "EventBus",
    "EventHandler",
    "EventType",
    "EvictionPolicy",
    "FeedbackEvent",
    "Gauge",
    "Histogram",
    "ImplicitFeedbackConfig",
    "InMemoryEventBus",
    "IndexReconciler",
    "InvalidationConsumer",
    "InProcessSingleFlightManager",
    "KafkaEventBus",
    "LookupRecord",
    "MetadataStore",
    "MigrationReport",
    "NoOpSpan",
    "OutboxEvent",
    "OutboxRelay",
    "OutboxStatus",
    "Permission",
    "PermissionName",
    "PostgresCacheStore",
    "PostgresConfig",
    "ProcessedEventRecord",
    "QdrantConfig",
    "QdrantVectorIndex",
    "RabbitMQEventBus",
    "ReconciliationReport",
    "RedisConfig",
    "RedisSingleFlightManager",
    "Resource",
    "RetrainConfig",
    "RetrainResult",
    "RetrievalScope",
    "RetryConfig",
    "RiskPolicy",
    "RiskTier",
    "Role",
    "SQLiteCacheStore",
    "SemanticMemo",
    "SemanticMemoMetrics",
    "SemanticMemoTracer",
    "SingleFlightManager",
    "Span",
    "StageTimer",
    "User",
    "build_flight_key",
    "get_tracer",
    "hash_identifier",
    "migrate_sqlite_to_postgres",
    "retrain_from_feedback",
    "sanitize_attributes",
]
