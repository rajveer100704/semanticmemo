from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semanticmemo import CacheConfig, SemanticMemo
from semanticmemo.embedding import HashEmbeddingProvider


def _make_cache(tmp_path: Path) -> SemanticMemo:
    return SemanticMemo(
        domain="test",
        config=CacheConfig(db_path=tmp_path / "cache.db", embedding_dim=16),
        embedding_provider=HashEmbeddingProvider(dim=16),
        use_faiss=False,
    )


async def test_async_with_closes_store(tmp_path: Path) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    async with _make_cache(tmp_path) as cache:
        await cache.get_or_call(prompt="hello", llm_function=llm)
        assert cache.store.count() == 1

    with pytest.raises(sqlite3.ProgrammingError):
        cache.store.count()


async def test_async_with_closes_store_on_exception(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        async with cache:
            raise RuntimeError("boom")

    with pytest.raises(sqlite3.ProgrammingError):
        cache.store.count()


def test_close_is_still_available(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.close()

    with pytest.raises(sqlite3.ProgrammingError):
        cache.store.count()
