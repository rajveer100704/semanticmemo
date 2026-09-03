"""Tests for Scoped Semantic Cache and Version Invalidation."""

from datetime import UTC, datetime, timedelta

import pytest

from semanticmemo import CacheConfig, SemanticMemo
from semanticmemo.embedding.service import HashEmbeddingProvider


@pytest.fixture
def cache_factory(tmp_path):
    def _create(domain="test_domain", **kwargs):
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
async def test_resource_scoped_cache_hit_and_isolation(cache_factory):
    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        memo.register_resource("res_100", "tenant_a", initial_version=1)
        memo.register_resource("res_200", "tenant_a", initial_version=1)

        call_count = {"count": 0}

        def mock_llm(p: str) -> str:
            call_count["count"] += 1
            return f"Answer for {p}"

        # 1. Query for res_100 (Miss -> LLM called)
        res1 = await memo.get_or_call(
            prompt="What is the revenue for Q3?",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="res_100",
        )
        assert res1.was_cache_hit is False
        assert call_count["count"] == 1

        # 2. Identical query for res_100 (Hit -> No LLM call)
        res2 = await memo.get_or_call(
            prompt="What is the revenue for Q3?",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="res_100",
        )
        assert res2.was_cache_hit is True
        assert call_count["count"] == 1

        # 3. Identical query for res_200 (Must NOT hit res_100 cache!)
        res3 = await memo.get_or_call(
            prompt="What is the revenue for Q3?",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="res_200",
        )
        assert res3.was_cache_hit is False
        assert call_count["count"] == 2


@pytest.mark.asyncio
async def test_no_resource_query_never_searches_resource_cache(cache_factory):
    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        memo.register_resource("res_a", "tenant_a", initial_version=1)
        memo.register_resource("res_b", "tenant_a", initial_version=1)

        # Populate Resource A
        res_a = await memo.get_or_call(
            prompt="Company secret financial formula",
            llm_function=lambda p: "Secret Formula A",
            user_id="user_1",
            resource_id="res_a",
        )
        assert res_a.was_cache_hit is False
        assert res_a.response == "Secret Formula A"

        # Populate Resource B
        res_b = await memo.get_or_call(
            prompt="Company secret financial formula",
            llm_function=lambda p: "Secret Formula B",
            user_id="user_1",
            resource_id="res_b",
        )
        assert res_b.was_cache_hit is False
        assert res_b.response == "Secret Formula B"

        # Same prompt queried WITHOUT resource_id -> MUST NOT return Secret Formula A or B
        res_none = await memo.get_or_call(
            prompt="Company secret financial formula",
            llm_function=lambda p: "Fresh Non-Resource Response",
            user_id="user_1",
            resource_id=None,
        )
        assert res_none.was_cache_hit is False
        assert res_none.response == "Fresh Non-Resource Response"


@pytest.mark.asyncio
async def test_atomic_version_invalidation(cache_factory):
    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        memo.register_resource("doc_v", "tenant_a", initial_version=1)

        call_count = {"count": 0}

        def mock_llm(p: str) -> str:
            call_count["count"] += 1
            return f"Versioned answer: {call_count['count']}"

        # 1. First query at v1
        res1 = await memo.get_or_call(
            prompt="Summary of document",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="doc_v",
        )
        assert res1.was_cache_hit is False
        assert res1.response == "Versioned answer: 1"

        # 2. Re-query at v1 -> Cache Hit
        res2 = await memo.get_or_call(
            prompt="Summary of document",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="doc_v",
        )
        assert res2.was_cache_hit is True
        assert res2.response == "Versioned answer: 1"

        # 3. Resource document updated -> Version incremented to v2
        memo.invalidate_resource("doc_v")

        # 4. Re-query at v2 -> Old v1 cache entry must be logically unreachable!
        res3 = await memo.get_or_call(
            prompt="Summary of document",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="doc_v",
        )
        assert res3.was_cache_hit is False
        assert res3.response == "Versioned answer: 2"

        # 5. Re-query at v2 -> Cache Hit on v2 entry
        res4 = await memo.get_or_call(
            prompt="Summary of document",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="doc_v",
        )
        assert res4.was_cache_hit is True
        assert res4.response == "Versioned answer: 2"


@pytest.mark.asyncio
async def test_ttl_expiration_enforcement(cache_factory):
    async with cache_factory(ttl_seconds=1) as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")

        call_count = {"count": 0}

        def mock_llm(p: str) -> str:
            call_count["count"] += 1
            return f"TTL answer {call_count['count']}"

        res1 = await memo.get_or_call(
            prompt="Temporary fact",
            llm_function=mock_llm,
            user_id="user_1",
        )
        assert res1.was_cache_hit is False

        # Immediate re-query -> Hit
        res2 = await memo.get_or_call(
            prompt="Temporary fact",
            llm_function=mock_llm,
            user_id="user_1",
        )
        assert res2.was_cache_hit is True

        # Manipulate expires_at in store to simulate elapsed time
        with memo.store._lock:
            past_time = (datetime.now(tz=UTC) - timedelta(seconds=10)).isoformat()
            memo.store._connection.execute(
                "UPDATE cache_entries SET expires_at = ?",
                (past_time,),
            )
            memo.store._connection.commit()

        # Query after expiration -> Miss
        res3 = await memo.get_or_call(
            prompt="Temporary fact",
            llm_function=mock_llm,
            user_id="user_1",
        )
        assert res3.was_cache_hit is False
        assert res3.response == "TTL answer 2"
