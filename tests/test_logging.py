from __future__ import annotations

import logging
from pathlib import Path

import pytest

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo
from smartmemo.embedding import HashEmbeddingProvider


def test_smartmemo_logger_has_null_handler() -> None:
    handlers = logging.getLogger("smartmemo").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_warnings_are_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    # With a NullHandler on the 'smartmemo' logger, Python's logging "last
    # resort" handler is not used, so even a WARNING reaches no stream.
    logging.getLogger("smartmemo.silent-check").warning("should not appear")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


async def test_cache_hit_and_miss_are_logged_when_opted_in(
    cache: SmartMemo, caplog: pytest.LogCaptureFixture
) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    with caplog.at_level(logging.DEBUG, logger="smartmemo"):
        await cache.get_or_call(prompt="alpha", llm_function=llm)
        await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)

    messages = [record.getMessage() for record in caplog.records]
    assert any("cache miss" in message for message in messages)
    assert any("cache hit" in message for message in messages)


async def test_eviction_is_logged(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    cache = SmartMemo(
        domain="test",
        config=CacheConfig(
            db_path=tmp_path / "cache.db",
            embedding_dim=16,
            max_entries=1,
        ),
        embedding_provider=HashEmbeddingProvider(dim=16),
        use_faiss=False,
    )

    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    with caplog.at_level(logging.INFO, logger="smartmemo"):
        await cache.get_or_call(prompt="first prompt", llm_function=llm)
        await cache.get_or_call(prompt="second prompt", llm_function=llm)
    cache.close()

    assert any("evicted" in record.getMessage() for record in caplog.records)


async def test_classifier_gate_is_logged(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    cache = SmartMemo(
        domain="test",
        config=CacheConfig(db_path=tmp_path / "cache.db", embedding_dim=384),
        embedding_provider=HashEmbeddingProvider(dim=384),
        classifier=ClassifierConfig.bundled(),
        use_faiss=False,
    )

    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    with caplog.at_level(logging.DEBUG, logger="smartmemo"):
        await cache.get_or_call(prompt="approve the refund", llm_function=llm)
        await cache.get_or_call(prompt="approve the refund", llm_function=llm)
    cache.close()

    assert any("classifier gate" in record.getMessage() for record in caplog.records)
