"""Security Regression Tests for SemanticMemo V2.

Covers all 7 mandatory security regression scenarios:
- Test A: Fail closed (no fallback to broader scope)
- Test B: Wrong tenant resource isolation (403 / PermissionError)
- Test C: Version race / stale version rejection
- Test D: Pipeline version namespace change
- Test E: Unauthorized cache write prevention
- Test F: Cross-resource semantic prompt collision prevention
- Test G: Permission revocation regression (Invariant 8)
"""

import pytest

from semanticmemo import CacheConfig, SemanticMemo
from semanticmemo.embedding.service import HashEmbeddingProvider
from semanticmemo.models import PermissionName


@pytest.fixture
def cache_factory(tmp_path):
    def _create(domain="security_domain", pipeline_version="v2", **kwargs):
        cfg = CacheConfig(
            db_path=tmp_path / f"{domain}_cache.db",
            embedding_dim=32,
            pipeline_version=pipeline_version,
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
async def test_a_fail_closed(cache_factory):
    """Test A: When RBAC lookup fails, do NOT fallback to TENANT or GLOBAL."""
    async with cache_factory() as memo:
        # Create a tenant-wide and global entry that matches the prompt
        memo.create_user("user_valid", "tenant_a", role="analyst")
        await memo.get_or_call(
            prompt="Quarterly financial statement",
            llm_function=lambda p: "Tenant Wide Financial Statement",
            user_id="user_valid",
            resource_id=None,
        )

        # Unauthorized user tries to access a non-existent or restricted resource
        memo.create_user("user_unauthorized", "tenant_a", role="viewer")
        memo.store.revoke_permission_from_role("viewer", PermissionName.RESOURCE_READ.value)

        # Must raise PermissionError and NOT return the tenant/global cache entry!
        with pytest.raises(PermissionError):
            await memo.get_or_call(
                prompt="Quarterly financial statement",
                llm_function=lambda p: "Should not execute",
                user_id="user_unauthorized",
                resource_id="non_existent_or_restricted_res",
            )


@pytest.mark.asyncio
async def test_b_wrong_tenant_resource_isolation(cache_factory):
    """Test B: Tenant A user querying Tenant B's resource must be denied."""
    async with cache_factory() as memo:
        memo.create_user("user_a", "tenant_a", role="analyst")
        memo.create_user("user_b", "tenant_b", role="analyst")

        memo.register_resource("tenant_b_confidential_doc", "tenant_b", initial_version=1)

        # User B populates cache for Tenant B resource
        res_b = await memo.get_or_call(
            prompt="Internal audit report",
            llm_function=lambda p: "Secret Tenant B Audit Data",
            user_id="user_b",
            resource_id="tenant_b_confidential_doc",
        )
        assert res_b.was_cache_hit is False

        # User A (from Tenant A) attempts to access Tenant B's resource
        with pytest.raises(PermissionError):
            await memo.get_or_call(
                prompt="Internal audit report",
                llm_function=lambda p: "Should fail",
                user_id="user_a",
                resource_id="tenant_b_confidential_doc",
            )


@pytest.mark.asyncio
async def test_c_version_race_stale_rejection(cache_factory):
    """Test C: Query at v17 must reject stale cached result if resource increments to v18."""
    async with cache_factory() as memo:
        memo.create_user("user_1", "tenant_a", role="analyst")
        memo.register_resource("report_x", "tenant_a", initial_version=17)

        call_count = {"count": 0}

        def mock_llm(p: str) -> str:
            call_count["count"] += 1
            return f"Report content version {call_count['count']}"

        # 1. Populate cache at v17
        res1 = await memo.get_or_call(
            prompt="Executive revenue summary",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="report_x",
        )
        assert res1.was_cache_hit is False
        assert res1.response == "Report content version 1"

        # 2. Re-query at v17 -> Cache Hit
        res2 = await memo.get_or_call(
            prompt="Executive revenue summary",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="report_x",
        )
        assert res2.was_cache_hit is True
        assert res2.response == "Report content version 1"

        # 3. Resource updated to v18
        memo.invalidate_resource("report_x")

        # 4. Re-query -> Stale v17 entry must be rejected, triggering fresh LLM call
        res3 = await memo.get_or_call(
            prompt="Executive revenue summary",
            llm_function=mock_llm,
            user_id="user_1",
            resource_id="report_x",
        )
        assert res3.was_cache_hit is False
        assert res3.response == "Report content version 2"


@pytest.mark.asyncio
async def test_d_pipeline_version_change(cache_factory, tmp_path):
    """Test D: Entry cached under pipeline v2.0 is ignored by pipeline v2.1 request."""
    db_path = tmp_path / "pipeline_version_test.db"

    # Instance 1 with pipeline_version = "v2.0"
    cfg1 = CacheConfig(db_path=db_path, embedding_dim=32, pipeline_version="v2.0")
    async with SemanticMemo(
        domain="finance",
        config=cfg1,
        embedding_provider=HashEmbeddingProvider(dim=32),
        use_faiss=False,
    ) as memo1:
        memo1.create_user("user_1", "tenant_a", role="analyst")
        res1 = await memo1.get_or_call(
            prompt="Tax rate for 2026",
            llm_function=lambda p: "Tax rate is 21%",
            user_id="user_1",
        )
        assert res1.was_cache_hit is False

    # Instance 2 pointing to same DB but with upgraded pipeline_version = "v2.1"
    cfg2 = CacheConfig(db_path=db_path, embedding_dim=32, pipeline_version="v2.1")
    async with SemanticMemo(
        domain="finance",
        config=cfg2,
        embedding_provider=HashEmbeddingProvider(dim=32),
        use_faiss=False,
    ) as memo2:
        res2 = await memo2.get_or_call(
            prompt="Tax rate for 2026",
            llm_function=lambda p: "Tax rate is 25% (Updated in v2.1 pipeline)",
            user_id="user_1",
        )
        # Must be a cache miss because pipeline version changed from v2.0 to v2.1!
        assert res2.was_cache_hit is False
        assert res2.response == "Tax rate is 25% (Updated in v2.1 pipeline)"


@pytest.mark.asyncio
async def test_e_unauthorized_write_prevention(cache_factory):
    """Test E: User lacking cache:write gets source result, but cache is NOT populated."""
    async with cache_factory() as memo:
        # Viewer has resource:read and cache:read, but NO cache:write
        memo.create_user("viewer_user", "tenant_a", role="viewer")
        memo.register_resource("doc_shared", "tenant_a", initial_version=1)

        call_count = {"count": 0}

        def mock_llm(p: str) -> str:
            call_count["count"] += 1
            return f"Answer #{call_count['count']}"

        # 1. Viewer queries -> Cache miss -> LLM returns answer, cache write omitted!
        res1 = await memo.get_or_call(
            prompt="Company policy on remote work",
            llm_function=mock_llm,
            user_id="viewer_user",
            resource_id="doc_shared",
        )
        assert res1.was_cache_hit is False
        assert res1.response == "Answer #1"

        # 2. Viewer queries again -> Because cache was NOT written, triggers LLM again!
        res2 = await memo.get_or_call(
            prompt="Company policy on remote work",
            llm_function=mock_llm,
            user_id="viewer_user",
            resource_id="doc_shared",
        )
        assert res2.was_cache_hit is False
        assert res2.response == "Answer #2"
        assert call_count["count"] == 2


@pytest.mark.asyncio
async def test_f_cross_resource_semantic_isolation(cache_factory):
    """Test F: Resource A and Resource B queried with identical prompt never cross-hit."""
    async with cache_factory() as memo:
        memo.create_user("analyst_1", "tenant_a", role="analyst")
        memo.register_resource("department_engineering", "tenant_a", initial_version=1)
        memo.register_resource("department_marketing", "tenant_a", initial_version=1)

        # Populate Engineering cache
        res_eng = await memo.get_or_call(
            prompt="Show quarterly revenue",
            llm_function=lambda p: "Engineering cost is $2.5M, Revenue $0",
            user_id="analyst_1",
            resource_id="department_engineering",
        )
        assert res_eng.was_cache_hit is False

        # Query Marketing with the EXACT same prompt -> MUST NOT return Engineering response!
        res_mkt = await memo.get_or_call(
            prompt="Show quarterly revenue",
            llm_function=lambda p: "Marketing generated $14.2M Revenue",
            user_id="analyst_1",
            resource_id="department_marketing",
        )
        assert res_mkt.was_cache_hit is False
        assert res_mkt.response == "Marketing generated $14.2M Revenue"

        # Query Engineering again -> Hits Engineering cache
        res_eng_2 = await memo.get_or_call(
            prompt="Show quarterly revenue",
            llm_function=lambda p: "Should not be called",
            user_id="analyst_1",
            resource_id="department_engineering",
        )
        assert res_eng_2.was_cache_hit is True
        assert res_eng_2.response == "Engineering cost is $2.5M, Revenue $0"


@pytest.mark.asyncio
async def test_g_permission_revocation_regression(cache_factory):
    """Test G: User creates cache entry -> permission revoked -> query produces NO hit."""
    async with cache_factory() as memo:
        memo.create_user("dynamic_user", "tenant_a", role="analyst")
        memo.register_resource("sensitive_contract", "tenant_a", initial_version=1)

        # 1. User has access, creates cache entry
        res1 = await memo.get_or_call(
            prompt="Terms of merger agreement",
            llm_function=lambda p: "Confidential merger agreement terms",
            user_id="dynamic_user",
            resource_id="sensitive_contract",
        )
        assert res1.was_cache_hit is False

        # 2. Verify cache hit while authorized
        res2 = await memo.get_or_call(
            prompt="Terms of merger agreement",
            llm_function=lambda p: "Should not execute",
            user_id="dynamic_user",
            resource_id="sensitive_contract",
        )
        assert res2.was_cache_hit is True

        # 3. Revoke user's role / access
        memo.remove_role("dynamic_user", "analyst")

        # 4. User issues the IDENTICAL query -> Re-evaluated authorization denies access
        with pytest.raises(PermissionError):
            await memo.get_or_call(
                prompt="Terms of merger agreement",
                llm_function=lambda p: "Should not execute",
                user_id="dynamic_user",
                resource_id="sensitive_contract",
            )


@pytest.mark.asyncio
async def test_unauthenticated_request_fails_closed_no_auto_creation(cache_factory):
    """Verify unauthenticated requests fail closed and never auto-create analyst identities."""
    async with cache_factory() as memo:
        # 1. Calling get_or_call without user_id must raise PermissionError
        with pytest.raises(PermissionError, match="Authentication required"):
            await memo.get_or_call(
                prompt="Secret query",
                llm_function=lambda p: "Secret answer",
                user_id=None,
            )

        # 2. Calling with non-existent user must raise PermissionError
        with pytest.raises(PermissionError, match="User not authenticated or does not exist"):
            await memo.get_or_call(
                prompt="Secret query",
                llm_function=lambda p: "Secret answer",
                user_id="non_existent_stranger",
            )

        # 3. Verify no default_user or stranger user was auto-created in the database
        assert memo.store.get_user("default_user") is None
        assert memo.store.get_user("non_existent_stranger") is None
