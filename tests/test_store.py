from __future__ import annotations

import sqlite3
from uuid import uuid4

import numpy as np

from semanticmemo.classifier.data import load_pair_records
from semanticmemo.models import EvictionPolicy
from semanticmemo.store import SQLiteCacheStore


def test_store_add_get_and_update_hit(tmp_path) -> None:
    store = SQLiteCacheStore(tmp_path / "cache.db")
    entry_id = store.add(
        prompt="alpha",
        embedding=np.array([1, 0, 0, 0], dtype=np.float32),
        response="response",
        model="test-model",
        metadata={"tenant": "acme"},
    )

    entry = store.get(entry_id)
    assert entry is not None
    assert entry.prompt == "alpha"
    assert entry.response == "response"
    assert entry.model == "test-model"
    assert entry.metadata == {"tenant": "acme"}
    assert entry.hit_count == 0

    store.update_hit(entry_id)
    updated = store.get(entry_id)
    assert updated is not None
    assert updated.hit_count == 1
    assert updated.last_hit_at is not None
    store.close()


def test_store_evicts_lru_with_negative_feedback_priority(tmp_path) -> None:
    store = SQLiteCacheStore(tmp_path / "cache.db")
    first = store.add(
        prompt="first",
        embedding=np.array([1, 0], dtype=np.float32),
        response="first",
        model=None,
    )
    second = store.add(
        prompt="second",
        embedding=np.array([0, 1], dtype=np.float32),
        response="second",
        model=None,
    )
    store.increment_bad_feedback(second)

    evicted = store.evict(policy=EvictionPolicy.LRU, max_entries=1, ttl_seconds=None)

    assert evicted == [second]
    assert store.get(second) is None
    assert store.get(first) is not None
    store.close()


def test_store_adds_feedback_tables_to_existing_database(tmp_path) -> None:
    db_path = tmp_path / "existing.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE cache_entries (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            prompt_embedding BLOB NOT NULL,
            response TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            last_hit_at TEXT,
            hit_count INTEGER NOT NULL DEFAULT 0,
            feedback_negative_count INTEGER NOT NULL DEFAULT 0,
            feedback_positive_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    connection.commit()
    connection.close()

    store = SQLiteCacheStore(db_path)

    assert store.lookup_count() == 0
    assert store.feedback_count() == 0
    store.close()


def test_feedback_events_survive_reopen_and_export_pairs(tmp_path) -> None:
    db_path = tmp_path / "feedback.db"
    output_path = tmp_path / "feedback_pairs.jsonl"
    query_id = uuid4()
    store = SQLiteCacheStore(db_path)
    entry_id = store.add(
        prompt="approve refund",
        embedding=np.array([1, 0], dtype=np.float32),
        response="approved",
        model="test-model",
    )
    store.record_lookup(
        query_id=query_id,
        domain="customer-support",
        prompt="accept refund",
        embedding=np.array([1, 0], dtype=np.float32),
        cache_entry_id=entry_id,
        similarity_score=0.97,
        classifier_score=0.91,
    )
    event_id = store.record_feedback(query_id=query_id, label=0, reason="wrong action")

    assert event_id is not None
    store.close()

    reopened = SQLiteCacheStore(db_path)
    events = list(reopened.feedback_events())
    exported = reopened.export_feedback_pairs(output_path, split="train")
    records = load_pair_records(output_path, split="train", domain="customer-support")

    assert len(events) == 1
    assert events[0].label == 0
    assert events[0].reason == "wrong action"
    assert exported == 1
    assert len(records) == 1
    assert records[0].prompt_a == "accept refund"
    assert records[0].prompt_b == "approve refund"
    assert records[0].label == 0
    assert records[0].source == "SemanticMemo-feedback"
    reopened.close()
