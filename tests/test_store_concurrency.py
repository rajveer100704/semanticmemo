from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np

from smartmemo.store import SQLiteCacheStore


def test_wal_and_pragmas_enabled(tmp_path: Path) -> None:
    store = SQLiteCacheStore(tmp_path / "cache.db")
    connection = store._connection

    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    store.close()


def test_cascade_delete_removes_lookups_and_feedback(tmp_path: Path) -> None:
    store = SQLiteCacheStore(tmp_path / "cache.db")
    query_id = uuid4()
    entry_id = store.add(
        prompt="approve refund",
        embedding=np.array([1, 0], dtype=np.float32),
        response="approved",
        model=None,
    )
    store.record_lookup(
        query_id=query_id,
        domain="customer-support",
        prompt="approve refund",
        embedding=np.array([1, 0], dtype=np.float32),
        cache_entry_id=entry_id,
        similarity_score=0.99,
        classifier_score=None,
    )
    store.record_feedback(query_id=query_id, label=0)
    assert store.lookup_count() == 1
    assert store.feedback_count() == 1

    # foreign_keys=ON makes the schema's ON DELETE CASCADE actually fire.
    store.delete(entry_id)

    assert store.lookup_count() == 0
    assert store.feedback_count() == 0
    store.close()


def test_concurrent_writes_from_threads(tmp_path: Path) -> None:
    store = SQLiteCacheStore(tmp_path / "cache.db")
    total = 50

    def add(index: int) -> None:
        store.add(
            prompt=f"prompt {index}",
            embedding=np.array([1, 0], dtype=np.float32),
            response=f"response {index}",
            model=None,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(total)))

    assert store.count() == total
    store.close()
