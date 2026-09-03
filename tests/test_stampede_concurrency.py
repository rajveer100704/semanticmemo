"""Concurrency and Cache Stampede Mitigation Tests."""

import asyncio

import pytest

from semanticmemo import CacheConfig, SemanticMemo
from semanticmemo.embedding.service import HashEmbeddingProvider


@pytest.fixture
def cache_factory(tmp_path):
    def _create(domain="stampede_domain", **kwargs):
        cfg = CacheConfig(
            db_path=tmp_path / f"{domain}_cache.db",
            embedding_dim=32,
            **kwargs,
        )
        return SemanticMemo(
            domain=domain,
            config=cfg,
            embedding_provider=HashEmbeddingProvider(dim=32),
            use_faiss=False,
        )

    return _create


@pytest.mark.asyncio
async def test_concurrent_misses_single_flight(cache_factory):
    """Verify that concurrent requests for the same cold query execute only a single LLM call."""
    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        memo.register_resource("res_heavy", "tenant_a", initial_version=1)

        llm_invocation_count = 0

        async def slow_llm(prompt: str) -> str:
            nonlocal llm_invocation_count
            llm_invocation_count += 1
            await asyncio.sleep(0.05)  # Simulate network latency to LLM
            return f"Computed result for: {prompt}"

        # Dispatch 10 concurrent identical queries
        tasks = [
            memo.get_or_call(
                prompt="Heavy analytical aggregation query",
                llm_function=slow_llm,
                user_id="user_1",
                resource_id="res_heavy",
            )
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All callers must receive the exact result
        assert len(results) == 10
        for res in results:
            assert res.response == "Computed result for: Heavy analytical aggregation query"

        # Concurrency single-flight must ensure at most 1 invocation of slow_llm
        assert llm_invocation_count == 1


@pytest.mark.asyncio
async def test_concurrent_misses_viewer_no_cache_write_single_flight(cache_factory):
    """Verify concurrent viewer requests execute exactly 1 LLM call and leave cache empty."""
    async with cache_factory() as memo:
        memo.create_user("viewer_1", "tenant_a", role="viewer")
        memo.register_resource("res_doc", "tenant_a", initial_version=1)
        memo.grant_resource_access("res_doc", "viewer", "resource:read")

        llm_invocation_count = 0

        async def slow_llm(prompt: str) -> str:
            nonlocal llm_invocation_count
            llm_invocation_count += 1
            await asyncio.sleep(0.05)  # Simulate network latency
            return f"Computed viewer response for: {prompt}"

        # Dispatch 10 concurrent identical queries from viewers lacking cache:write
        tasks = [
            memo.get_or_call(
                prompt="Confidential financial report analysis",
                llm_function=slow_llm,
                user_id="viewer_1",
                resource_id="res_doc",
            )
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        expected = "Computed viewer response for: Confidential financial report analysis"
        # All 10 callers must receive the exact computed result
        assert len(results) == 10
        for res in results:
            assert res.response == expected
            assert res.was_cache_hit is False

        # Single-flight future coalescer must guarantee EXACTLY 1 LLM invocation
        assert llm_invocation_count == 1

        # Cache must remain empty because viewer lacks cache:write permission
        assert memo.store.count() == 0


@pytest.mark.asyncio
async def test_hashed_flight_lock_keys(cache_factory):
    """Verify that in-flight request keys use bounded SHA256 hashes instead of raw prompts."""
    import hashlib

    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        raw_prompt = "Select all sensitive confidential records from database"
        expected_hash = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()

        # Track keys created in _inflight
        seen_keys = []
        original_call_llm = memo._orchestrator._call_llm

        async def wrapped_call_llm(func, prompt):
            sf_mgr = memo._orchestrator.single_flight_manager
            if hasattr(sf_mgr, "_inflight"):
                seen_keys.extend(list(sf_mgr._inflight.keys()))
            return await original_call_llm(func, prompt)

        memo._orchestrator._call_llm = wrapped_call_llm

        await memo.get_or_call(
            prompt=raw_prompt,
            llm_function=lambda p: "response",
            user_id="user_1",
        )

        assert any(expected_hash in key for key in seen_keys)
        assert not any(raw_prompt in key for key in seen_keys)
