from __future__ import annotations

from equivcache import EquivCache


async def test_get_or_call_misses_then_hits(cache: EquivCache) -> None:
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"fresh:{prompt}"

    first = await cache.get_or_call(prompt="alpha", llm_function=llm, model="unit-test")
    second = await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert first.was_cache_hit is False
    assert second.was_cache_hit is True
    assert second.response == "fresh:alpha"
    assert second.cache_entry_id == first.cache_entry_id
    assert second.classifier_score is None
    assert second.cost_saved_usd > 0
    assert calls == 1


async def test_threshold_rejects_low_similarity(cache: EquivCache) -> None:
    calls = 0

    def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"fresh:{prompt}"

    first = await cache.get_or_call(prompt="alpha", llm_function=llm)
    second = await cache.get_or_call(prompt="beta", llm_function=llm)

    assert first.was_cache_hit is False
    assert second.was_cache_hit is False
    assert second.similarity_score == 0.0
    assert calls == 2


async def test_feedback_updates_hit_entry(cache: EquivCache) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    await cache.get_or_call(prompt="alpha", llm_function=llm)
    hit = await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert await cache.report_bad_hit(hit.query_id, reason="wrong answer") is True
    assert hit.cache_entry_id is not None
    entry = cache.store.get(hit.cache_entry_id)
    assert entry is not None
    assert entry.feedback_negative_count == 1


async def test_stats_track_runtime_counts(cache: EquivCache) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    await cache.get_or_call(prompt="alpha", llm_function=llm)
    await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)
    stats = cache.stats()

    assert stats.total_entries == 1
    assert stats.total_lookups == 2
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.hit_rate == 0.5
