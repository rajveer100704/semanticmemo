"""Tests for opt-in implicit-feedback (prompt re-issue) detection."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import semanticmemo.store.sqlite_store as sqlite_store
from semanticmemo import CacheConfig, ImplicitFeedbackConfig, SemanticMemo
from semanticmemo.store import SQLiteCacheStore
from semanticmemo.types import FloatVector


class ToyEmbeddingProvider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        match text:
            case "alpha":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "beta":
                return np.array([0, 1, 0, 0], dtype=np.float32)
            case _:
                return np.array([0, 0, 1, 0], dtype=np.float32)


class FakeClock:
    """A controllable stand-in for the store's wall clock."""

    def __init__(self) -> None:
        self.now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _make_cache(
    tmp_path: Path,
    *,
    implicit: ImplicitFeedbackConfig | None,
) -> SemanticMemo:
    config = CacheConfig(
        db_path=tmp_path / "semanticmemo.db",
        embedding_dim=ToyEmbeddingProvider.dim,
        cosine_threshold=0.95,
        candidate_k=3,
        implicit_feedback=implicit,
    )
    return SemanticMemo(
        domain="test",
        config=config,
        embedding_provider=ToyEmbeddingProvider(),
        use_faiss=False,
    )


async def _echo(prompt: str) -> str:
    return f"fresh:{prompt}"


async def test_implicit_feedback_disabled_by_default(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, implicit=None)
    try:
        for _ in range(3):
            result = await cache.get_or_call(prompt="alpha", llm_function=_echo)
            assert result.implicit_bad_hit_recorded is False
        assert cache.store.feedback_count() == 0
    finally:
        cache.close()


async def test_reissue_within_window_flags_prior_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        miss = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        hit = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert miss.was_cache_hit is False
        assert hit.was_cache_hit is True
        assert hit.implicit_bad_hit_recorded is False

        clock.advance(5)
        reissue = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert reissue.implicit_bad_hit_recorded is True

        events = list(cache.store.feedback_events())
        assert len(events) == 1
        assert events[0].label == 0
        assert events[0].reason == "implicit:re-issued"
        assert events[0].query_id == hit.query_id
        assert events[0].metadata["auto_detected"] is True
        assert events[0].metadata["detector"] == "re-issue"
        assert events[0].metadata["window_seconds"] == 30.0

        assert miss.cache_entry_id is not None
        entry = cache.store.get(miss.cache_entry_id)
        assert entry is not None
        assert entry.feedback_negative_count == 1
    finally:
        cache.close()


async def test_reissue_outside_window_not_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        await cache.get_or_call(prompt="alpha", llm_function=_echo)

        clock.advance(60)
        reissue = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert reissue.implicit_bad_hit_recorded is False
        assert cache.store.feedback_count() == 0
    finally:
        cache.close()


async def test_triple_reissue_flags_each_prior_hit_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)  # miss
        hit_1 = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        clock.advance(1)
        hit_2 = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        clock.advance(1)
        await cache.get_or_call(prompt="alpha", llm_function=_echo)

        events = list(cache.store.feedback_events())
        # hit_3 flags hit_1's lookup; hit_4 flags hit_2's lookup -- two events.
        assert len(events) == 2
        flagged = {event.query_id for event in events}
        assert flagged == {hit_1.query_id, hit_2.query_id}
        assert all(event.label == 0 for event in events)
    finally:
        cache.close()


async def test_explicit_good_feedback_blocks_implicit_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        hit = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert await cache.report_good_hit(hit.query_id) is True

        clock.advance(2)
        reissue = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert reissue.implicit_bad_hit_recorded is False

        # Only the explicit positive event exists -- implicit detection did not
        # override it.
        events = list(cache.store.feedback_events())
        assert len(events) == 1
        assert events[0].label == 1
    finally:
        cache.close()


async def test_explicit_bad_feedback_not_double_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        hit = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert await cache.report_bad_hit(hit.query_id, reason="explicit") is True

        clock.advance(2)
        reissue = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert reissue.implicit_bad_hit_recorded is False
        assert cache.store.feedback_count() == 1
    finally:
        cache.close()


async def test_prior_miss_is_not_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        # A cache miss records no lookup row, so re-issuing after a miss
        # has nothing to flag.
        miss = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        hit = await cache.get_or_call(prompt="alpha", llm_function=_echo)
        assert miss.was_cache_hit is False
        assert hit.was_cache_hit is True
        assert hit.implicit_bad_hit_recorded is False
        assert cache.store.feedback_count() == 0
    finally:
        cache.close()


async def test_whitespace_only_difference_is_treated_as_reissue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        clock.advance(2)
        reissue = await cache.get_or_call(prompt="  alpha\n", llm_function=_echo)
        assert reissue.implicit_bad_hit_recorded is True
    finally:
        cache.close()


async def test_implicit_flag_is_exported_as_training_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    cache = _make_cache(tmp_path, implicit=ImplicitFeedbackConfig(window_seconds=30.0))
    export_path = tmp_path / "feedback_pairs.jsonl"
    try:
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        await cache.get_or_call(prompt="alpha", llm_function=_echo)
        clock.advance(3)
        await cache.get_or_call(prompt="alpha", llm_function=_echo)

        written = cache.export_feedback_pairs(export_path, split="train")
        assert written == 1
        pair = json.loads(export_path.read_text().strip())
        assert pair["label"] == 0
        assert pair["prompt_a"] == "alpha"
        assert pair["prompt_b"] == "alpha"
        assert pair["metadata"]["auto_detected"] is True
        assert pair["metadata"]["reason"] == "implicit:re-issued"
    finally:
        cache.close()


def test_find_implicit_bad_hit_window_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sqlite_store, "_now", clock)
    store = SQLiteCacheStore(tmp_path / "store.db")
    try:
        embedding = np.array([1, 0, 0, 0], dtype=np.float32)
        entry_id = store.add(
            prompt="alpha",
            embedding=embedding,
            response="cached",
            model="unit-test",
        )
        query_id = uuid4()
        store.record_lookup(
            query_id=query_id,
            domain="test",
            prompt="alpha",
            embedding=embedding,
            cache_entry_id=entry_id,
            similarity_score=1.0,
            classifier_score=None,
        )

        clock.advance(30)
        # created_at == cutoff: still inside the window.
        on_boundary = store.find_implicit_bad_hit(domain="test", prompt="alpha", within_seconds=30)
        assert on_boundary is not None
        assert on_boundary.id == query_id

        # One second tighter: now outside the window.
        outside = store.find_implicit_bad_hit(domain="test", prompt="alpha", within_seconds=29)
        assert outside is None

        # Wrong domain / prompt never match.
        assert (
            store.find_implicit_bad_hit(domain="other", prompt="alpha", within_seconds=60) is None
        )
        assert store.find_implicit_bad_hit(domain="test", prompt="beta", within_seconds=60) is None
    finally:
        store.close()
