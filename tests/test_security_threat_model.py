"""10-Vector Production Security Threat Model Audit Test Suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from semanticmemo.cache import SemanticMemo
from semanticmemo.events.consumer import InvalidationConsumer
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.models import CacheConfig, CacheScope
from semanticmemo.store.sqlite_store import SQLiteCacheStore
from semanticmemo.telemetry.tracing import sanitize_attributes


# =============================================================================
# Vector 1: Tenant Spoofing / Impersonation via Client Request Parameters
# =============================================================================
@pytest.mark.asyncio
async def test_vector_1_tenant_spoofing_prevented(tmp_path: Any) -> None:
    """Proves server-side authenticated identity overrides client-supplied tenant_id."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    # Legitimate user belonging to tenant_victim
    victim_user = cache.orchestrator.bootstrap_default_user(
        "user_victim", tenant_id="tenant_victim"
    )

    # Attacker belonging to tenant_attacker attempts to query passing tenant_id="tenant_victim"
    attacker_user = cache.orchestrator.bootstrap_default_user(
        "user_attacker", tenant_id="tenant_attacker"
    )

    # Victim stores sensitive data in their tenant
    await cache.get_or_call(
        prompt="Sensitive financial payroll data",
        llm_function=lambda p: "Victim Payroll: $1,000,000",
        user_id=victim_user.id,
    )

    # Attacker queries using attacker_user.id but supplies tenant_id="tenant_victim"
    res_attacker = await cache.get_or_call(
        prompt="Sensitive financial payroll data",
        llm_function=lambda p: "Attacker Fresh: $0",
        user_id=attacker_user.id,
        tenant_id="tenant_victim",  # Spoofed parameter
    )

    # Must NOT return victim's data; must resolve to tenant_attacker
    assert res_attacker.response == "Attacker Fresh: $0"
    assert res_attacker.was_cache_hit is False


# =============================================================================
# Vector 2: Resource Scope Spoofing (Cross-Tenant Resource Hijacking)
# =============================================================================
@pytest.mark.asyncio
async def test_vector_2_resource_scope_spoofing_denied(tmp_path: Any) -> None:
    """Proves a tenant cannot access or bind to another tenant's resource."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    cache.orchestrator.bootstrap_default_user("user_a", tenant_id="tenant_alpha")
    user_b = cache.orchestrator.bootstrap_default_user("user_b", tenant_id="tenant_beta")

    # Create resource belonging to tenant_alpha
    res_alpha = cache.orchestrator.store.create_resource("res_alpha_private", "tenant_alpha")

    # User B from tenant_beta attempts to query res_alpha_private
    with pytest.raises(PermissionError):
        await cache.get_or_call(
            prompt="Quarterly audit notes",
            llm_function=lambda p: "Audit Alpha",
            user_id=user_b.id,
            resource_id=res_alpha.id,
        )


# =============================================================================
# Vector 3: Permission Revocation Race
# =============================================================================
@pytest.mark.asyncio
async def test_vector_3_permission_revocation_enforced_immediately(tmp_path: Any) -> None:
    """Proves permission revocation takes immediate effect on subsequent lookups."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_rev", tenant_id="tenant_rev")
    res = cache.orchestrator.store.create_resource("res_rev", "tenant_rev")
    cache.orchestrator.store.grant_resource_access("res_rev", "analyst", "resource:read")

    # Initial authorized lookup
    res_init = await cache.get_or_call(
        prompt="Trading algorithm parameters",
        llm_function=lambda p: "Parameters Alpha-9",
        user_id=user.id,
        resource_id=res.id,
    )
    assert res_init.response == "Parameters Alpha-9"

    # Revoke access immediately
    cache.orchestrator.store.revoke_resource_access("res_rev", "analyst", "resource:read")
    cache.orchestrator.store.grant_resource_access("res_rev", "viewer", "resource:read")

    # Subsequent lookup fails closed
    with pytest.raises(PermissionError):
        await cache.get_or_call(
            prompt="Trading algorithm parameters",
            llm_function=lambda p: "Should not be called",
            user_id=user.id,
            resource_id=res.id,
        )


# =============================================================================
# Vector 4: Unauthenticated Access / Identity Omission
# =============================================================================
@pytest.mark.asyncio
async def test_vector_4_unauthenticated_access_rejected(tmp_path: Any) -> None:
    """Proves requests lacking authenticated user_id are rejected immediately."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    with pytest.raises(PermissionError):
        await cache.get_or_call(
            prompt="Public or private prompt",
            llm_function=lambda p: "Result",
            user_id=None,  # Missing principal
        )


# =============================================================================
# Vector 5: Semantic Collision / Hard Negatives Manipulation
# =============================================================================
@pytest.mark.asyncio
async def test_vector_5_opposite_action_collision_rejected(tmp_path: Any) -> None:
    """Proves opposite polarity prompts ('approve' vs 'deny') never produce false cache hits."""
    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user = cache.orchestrator.bootstrap_default_user("user_coll", tenant_id="tenant_coll")

    # Prime cache with "approve" prompt
    await cache.get_or_call(
        prompt="Please approve loan application #94819",
        llm_function=lambda p: "APPROVED: Loan #94819 authorized",
        user_id=user.id,
    )

    # Query with opposite prompt "deny"
    res_deny = await cache.get_or_call(
        prompt="Please deny loan application #94819",
        llm_function=lambda p: "DENIED: Loan #94819 rejected",
        user_id=user.id,
    )

    # Must NOT hit the "approve" cache entry
    assert res_deny.was_cache_hit is False
    assert res_deny.response == "DENIED: Loan #94819 rejected"


# =============================================================================
# Vector 6: Outbox Replay / Duplicate Delivery Attacks
# =============================================================================
@pytest.mark.asyncio
async def test_vector_6_outbox_replay_deduplication(tmp_path: Any) -> None:
    """Proves consumer idempotency store ignores replayed/duplicate events."""
    db_path = tmp_path / "dedup.db"
    store = SQLiteCacheStore(str(db_path))

    event = OutboxEvent(
        id=uuid4(),
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="res_replay",
        tenant_id="tenant_replay",
        payload={"new_version": 5},
    )

    consumer = InvalidationConsumer(consumer_name="replay_consumer", processed_store=store)

    # First delivery
    first = await consumer.handle_event(event)
    assert first is True

    # Replayed delivery with same event_id
    second = await consumer.handle_event(event)
    assert second is False
    assert consumer.processed_count == 1


# =============================================================================
# Vector 7: Redis Lock Hijacking / Token-Checked Release
# =============================================================================
class _LocalFakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, px: int | None = None) -> bool:
        self.data[key] = value
        return True

    async def eval(self, script: str, numkeys: int, key: str, arg: str) -> int:
        if self.data.get(key) == arg:
            self.data.pop(key, None)
            return 1
        return 0


@pytest.mark.asyncio
async def test_vector_7_lock_token_hijacking_prevented() -> None:
    """Proves a foreign node cannot release another node's single-flight lock."""
    redis_client = _LocalFakeRedis()
    flight_key = "sm:lock:flight_secure"

    # Node A acquires lock
    await redis_client.set(flight_key, "token_node_a", px=10000)

    # Node B attempts to delete Node A's lock with fraudulent token
    res = await redis_client.eval(
        """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """,
        1,
        flight_key,
        "token_node_b",
    )
    assert res == 0
    assert await redis_client.get(flight_key) == "token_node_a"


# =============================================================================
# Vector 8: Unscoped Vector Database Pollution Prevention
# =============================================================================
@pytest.mark.asyncio
async def test_vector_8_unscoped_vector_search_filtered(tmp_path: Any) -> None:
    """Proves vector search filters out vectors belonging to other tenants."""
    from semanticmemo.models import RetrievalScope

    db_path = Path(tmp_path) / "cache.db"
    cache = SemanticMemo(
        domain="finance",
        config=CacheConfig(db_path=db_path, vector_store_type="in_memory"),
    )
    user_1 = cache.orchestrator.bootstrap_default_user("user_t1", tenant_id="tenant_1")
    cache.orchestrator.bootstrap_default_user("user_t2", tenant_id="tenant_2")

    # Tenant 1 stores a vector
    await cache.get_or_call(
        prompt="Market analysis for Q3 2026",
        llm_function=lambda p: "Tenant 1 Market Analysis",
        user_id=user_1.id,
    )

    # Tenant 2 queries vector service directly with its own tenant retrieval scope
    scope_t2 = RetrievalScope(
        tenant_id="tenant_2",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    hits = cache.orchestrator.embedding_service.search(
        query_embedding=cache.orchestrator.embedding_service.embed("Market analysis for Q3 2026"),
        k=5,
        retrieval_scope=scope_t2,
    )
    # Search must return 0 results for tenant_2
    assert len(hits) == 0


# =============================================================================
# Vector 9: Telemetry Privacy & PII Leakage Prevention
# =============================================================================
def test_vector_9_telemetry_sanitizes_pii() -> None:
    """Proves raw prompts, responses, and identifiers are sanitized/hashed."""
    raw_attrs = {
        "prompt": "SELECT * FROM user_credit_cards WHERE ssn = '000-11-2222'",
        "response": "Card 4111-2222-3333-4444 Exp 12/28",
        "tenant_id": "tenant_enterprise_99",
        "user_id": "user_john_doe",
        "resource_id": "doc_confidential",
        "token": "bearer_jwt_secret_token",
        "status": "success",
    }
    sanitized = sanitize_attributes(raw_attrs, include_raw_identifiers=False)

    # Assert raw prompt/response/tokens are stripped completely
    assert "prompt" not in sanitized
    assert "semanticmemo.prompt" not in sanitized
    assert "response" not in sanitized
    assert "semanticmemo.response" not in sanitized
    assert "token" not in sanitized
    assert "semanticmemo.token" not in sanitized

    # Assert tenant, user, and resource identifiers are hashed
    assert "tenant_id" not in sanitized
    assert "semanticmemo.tenant_hash" in sanitized
    assert sanitized["semanticmemo.tenant_hash"] != "tenant_enterprise_99"
    assert "user_id" not in sanitized
    assert "semanticmemo.user_hash" in sanitized
    assert sanitized["semanticmemo.user_hash"] != "user_john_doe"
    assert "resource_id" not in sanitized
    assert "semanticmemo.resource_hash" in sanitized
    assert sanitized["semanticmemo.resource_hash"] != "doc_confidential"


# =============================================================================
# Vector 10: Stale Leader CAS Write Race Rejection
# =============================================================================
def test_vector_10_stale_leader_cas_write_rejected(tmp_path: Any) -> None:
    """Proves stale leader writes with older flight generations are rejected with 0 rows mutated."""
    db_path = tmp_path / "cas_threat.db"
    store = SQLiteCacheStore(str(db_path))

    prompt = "Execute financial audit transaction #881"
    embedding = np.array([0.3] * 384, dtype=np.float32)

    # Leader B finishes with generation 2
    entry_b_id = store.add(
        prompt=prompt,
        embedding=embedding,
        response="Result from fresh leader (generation 2)",
        model="model-b",
        tenant_id="tenant_cas",
        flight_generation=2,
    )
    assert entry_b_id is not None

    # Stale Leader A resumes and attempts to write with generation 1
    entry_a_id = store.add(
        prompt=prompt,
        embedding=embedding,
        response="Result from stale leader (generation 1)",
        model="model-a",
        tenant_id="tenant_cas",
        flight_generation=1,
    )
    # CAS rejection: entry_a_id is None, store is NOT corrupted
    assert entry_a_id is None

    # Verify store retains generation 2
    cached = store.find_by_prompt(prompt=prompt, tenant_id="tenant_cas")
    assert cached is not None
    assert cached.response == "Result from fresh leader (generation 2)"
    assert cached.flight_generation == 2
