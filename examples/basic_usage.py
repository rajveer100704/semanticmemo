"""Minimal dependency-light EquivCache example.

Run with:
    uv run python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from equivcache import CacheConfig, EquivCache
from equivcache.embedding import HashEmbeddingProvider


async def main() -> None:
    with TemporaryDirectory() as temp_dir:
        cache = EquivCache(
            domain="customer-support",
            config=CacheConfig(
                db_path=Path(temp_dir) / "equivcache.db",
                embedding_dim=32,
                cosine_threshold=0.80,
                estimated_llm_cost_usd="0.002",
            ),
            embedding_provider=HashEmbeddingProvider(dim=32),
            use_faiss=False,
        )

        calls = 0

        async def call_llm(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return f"fresh response for: {prompt}"

        first = await cache.get_or_call(
            prompt="Summarize the latest billing ticket",
            llm_function=call_llm,
        )
        second = await cache.get_or_call(
            prompt="Summarize the latest billing ticket",
            llm_function=call_llm,
        )

        print({"first_hit": first.was_cache_hit, "second_hit": second.was_cache_hit, "calls": calls})
        print(cache.stats().model_dump())
        cache.close()


if __name__ == "__main__":
    asyncio.run(main())
