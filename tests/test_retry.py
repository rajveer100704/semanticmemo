from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from smartmemo import CacheConfig, RetryConfig, SmartMemo
from smartmemo.embedding import HashEmbeddingProvider
from smartmemo.exceptions import LLMCallError


def _make_cache(tmp_path: Path, retry: RetryConfig | None) -> SmartMemo:
    return SmartMemo(
        domain="test",
        config=CacheConfig(
            db_path=tmp_path / "cache.db",
            embedding_dim=16,
            retry=retry,
        ),
        embedding_provider=HashEmbeddingProvider(dim=16),
        use_faiss=False,
    )


async def test_no_retry_by_default(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, retry=None)
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("transient")

    # With retry unset, the raw exception propagates -- no LLMCallError wrapping.
    with pytest.raises(ConnectionError, match="transient"):
        await cache.get_or_call(prompt="hello", llm_function=llm)
    assert calls == 1
    cache.close()


async def test_retry_then_succeeds(tmp_path: Path) -> None:
    cache = _make_cache(
        tmp_path,
        retry=RetryConfig(max_attempts=3, initial_backoff_seconds=0.001),
    )
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await cache.get_or_call(prompt="hello", llm_function=llm)
    assert result.response == "ok"
    assert calls == 3
    cache.close()


async def test_retry_exhausted_raises_llm_call_error(tmp_path: Path) -> None:
    cache = _make_cache(
        tmp_path,
        retry=RetryConfig(max_attempts=2, initial_backoff_seconds=0.001),
    )

    async def llm(prompt: str) -> str:
        raise ConnectionError("down")

    with pytest.raises(LLMCallError) as exc_info:
        await cache.get_or_call(prompt="hello", llm_function=llm)
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    cache.close()


async def test_retry_on_filters_exception_types(tmp_path: Path) -> None:
    cache = _make_cache(
        tmp_path,
        retry=RetryConfig(
            max_attempts=3,
            initial_backoff_seconds=0.001,
            retry_on=(ValueError,),
        ),
    )
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise KeyError("not retryable")

    # KeyError is not in retry_on, so it propagates immediately, unwrapped.
    with pytest.raises(KeyError):
        await cache.get_or_call(prompt="hello", llm_function=llm)
    assert calls == 1
    cache.close()


async def test_backoff_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    cache = _make_cache(
        tmp_path,
        retry=RetryConfig(
            max_attempts=5,
            initial_backoff_seconds=1.0,
            backoff_multiplier=10.0,
            max_backoff_seconds=5.0,
        ),
    )

    async def llm(prompt: str) -> str:
        raise ConnectionError("always")

    with pytest.raises(LLMCallError):
        await cache.get_or_call(prompt="hello", llm_function=llm)
    assert sleeps
    assert all(seconds <= 5.0 for seconds in sleeps)
    cache.close()


async def test_cache_hit_never_calls_llm(tmp_path: Path) -> None:
    cache = _make_cache(
        tmp_path,
        retry=RetryConfig(max_attempts=3, initial_backoff_seconds=0.001),
    )
    calls = 0

    async def good_llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "cached"

    await cache.get_or_call(prompt="hello", llm_function=good_llm)
    assert calls == 1

    async def failing_llm(prompt: str) -> str:
        raise ConnectionError("should never be called on a cache hit")

    result = await cache.get_or_call(prompt="hello", llm_function=failing_llm)
    assert result.was_cache_hit is True
    assert calls == 1
    cache.close()
