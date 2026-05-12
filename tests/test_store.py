from __future__ import annotations

import numpy as np

from equivcache.models import EvictionPolicy
from equivcache.store import SQLiteCacheStore


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
