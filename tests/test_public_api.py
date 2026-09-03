from __future__ import annotations

import semanticmemo


def test_public_api_exports() -> None:
    assert semanticmemo.SemanticMemo is not None
    assert semanticmemo.CacheConfig is not None
    assert semanticmemo.CacheResult is not None
    assert semanticmemo.CacheStats is not None
    assert semanticmemo.CacheEntry is not None
    assert semanticmemo.ClassifierConfig is not None
    assert semanticmemo.EvictionPolicy is not None
    assert semanticmemo.FeedbackEvent is not None
    assert semanticmemo.ImplicitFeedbackConfig is not None
    assert semanticmemo.LookupRecord is not None
    assert semanticmemo.RetryConfig is not None
    assert semanticmemo.RetrainConfig is not None
    assert semanticmemo.RetrainResult is not None
    assert semanticmemo.retrain_from_feedback is not None
    assert semanticmemo.MetadataStore is not None
    assert semanticmemo.PostgresCacheStore is not None
    assert semanticmemo.PostgresConfig is not None
    assert semanticmemo.CutoverState is not None
    assert semanticmemo.MigrationReport is not None
    assert semanticmemo.migrate_sqlite_to_postgres is not None
    assert semanticmemo.OutboxEvent is not None
    assert semanticmemo.EventType is not None
    assert semanticmemo.OutboxStatus is not None
    assert semanticmemo.ProcessedEventRecord is not None
    assert semanticmemo.InMemoryEventBus is not None
    assert semanticmemo.KafkaEventBus is not None
    assert semanticmemo.RabbitMQEventBus is not None
    assert semanticmemo.OutboxRelay is not None
    assert semanticmemo.InvalidationConsumer is not None
    assert semanticmemo.IndexReconciler is not None
    assert semanticmemo.ReconciliationReport is not None
