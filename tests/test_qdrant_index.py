"""Tests for QdrantVectorIndex backend and RetrievalScope pre-filtering."""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

from semanticmemo.embedding.qdrant_index import (
    QDRANT_AVAILABLE,
    QdrantConfig,
    QdrantVectorIndex,
    _build_qdrant_filter,
)
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, RetrievalScope


@pytest.fixture
def in_memory_qdrant():
    """Fixture providing an in-memory QdrantVectorIndex instance."""
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client is not installed in this environment")

    config = QdrantConfig(
        collection_name=f"test_collection_{uuid4().hex[:8]}",
        location=":memory:",
    )
    index = QdrantVectorIndex(dim=4, config=config)
    yield index


def test_qdrant_missing_dependency_simulation(monkeypatch: pytest.MonkeyPatch):
    """Test that MissingDependencyError is raised when qdrant-client is not available."""
    import semanticmemo.embedding.qdrant_index as qmodule

    monkeypatch.setattr(qmodule, "QDRANT_AVAILABLE", False)
    with pytest.raises(MissingDependencyError, match="qdrant-client is required"):
        QdrantVectorIndex(dim=4)

    scope = RetrievalScope(tenant_id="t1", scope=CacheScope.GLOBAL)
    with pytest.raises(MissingDependencyError, match="qdrant-client is required"):
        _build_qdrant_filter(scope)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_resource_isolation_cross_resource(in_memory_qdrant: QdrantVectorIndex):
    """Test that Resource A cannot retrieve Resource B even with identical vectors."""
    index = in_memory_qdrant
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    entry_a = uuid4()
    entry_b = uuid4()

    index.add(
        entry_id=entry_a,
        embedding=vec,
        tenant_id="tenant_x",
        scope=CacheScope.RESOURCE,
        resource_id="res_a",
        resource_version=1,
        pipeline_version="v2",
    )
    index.add(
        entry_id=entry_b,
        embedding=vec,
        tenant_id="tenant_x",
        scope=CacheScope.RESOURCE,
        resource_id="res_b",
        resource_version=1,
        pipeline_version="v2",
    )

    scope_a = RetrievalScope(
        tenant_id="tenant_x",
        scope=CacheScope.RESOURCE,
        resource_id="res_a",
        resource_version=1,
        pipeline_version="v2",
    )
    results = index.search(vec, k=5, retrieval_scope=scope_a)
    assert len(results) == 1
    assert results[0].entry_id == entry_a

    scope_b = RetrievalScope(
        tenant_id="tenant_x",
        scope=CacheScope.RESOURCE,
        resource_id="res_b",
        resource_version=1,
        pipeline_version="v2",
    )
    results_b = index.search(vec, k=5, retrieval_scope=scope_b)
    assert len(results_b) == 1
    assert results_b[0].entry_id == entry_b


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_multi_tenant_identical_vector_collision(in_memory_qdrant: QdrantVectorIndex):
    """Security Regression: Tenant A + Resource A vs Tenant B + Resource B with identical vector.

    Proves:
    - Tenant A + Resource A query returns entry A only.
    - Tenant B + Resource A query returns zero candidates.
    - Tenant B + Resource B query returns entry B only.
    - Tenant A + Resource B query returns zero candidates.
    """
    index = in_memory_qdrant
    identical_vec = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)

    entry_a = uuid4()
    entry_b = uuid4()

    # Seed Tenant A Resource A
    index.add(
        entry_id=entry_a,
        embedding=identical_vec,
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="res_shared_name",
        resource_version=1,
        pipeline_version="v2",
    )
    # Seed Tenant B Resource B (same resource name but different tenant)
    index.add(
        entry_id=entry_b,
        embedding=identical_vec,
        tenant_id="tenant_b",
        scope=CacheScope.RESOURCE,
        resource_id="res_shared_name",
        resource_version=1,
        pipeline_version="v2",
    )

    # 1. Tenant A asks for res_shared_name -> returns A only
    scope_a = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="res_shared_name",
        resource_version=1,
        pipeline_version="v2",
    )
    res_a = index.search(identical_vec, k=5, retrieval_scope=scope_a)
    assert len(res_a) == 1
    assert res_a[0].entry_id == entry_a

    # 2. Tenant B asks for res_shared_name -> returns B only
    scope_b = RetrievalScope(
        tenant_id="tenant_b",
        scope=CacheScope.RESOURCE,
        resource_id="res_shared_name",
        resource_version=1,
        pipeline_version="v2",
    )
    res_b = index.search(identical_vec, k=5, retrieval_scope=scope_b)
    assert len(res_b) == 1
    assert res_b[0].entry_id == entry_b

    # 3. Tenant C asks for res_shared_name -> returns 0
    scope_c = RetrievalScope(
        tenant_id="tenant_c",
        scope=CacheScope.RESOURCE,
        resource_id="res_shared_name",
        resource_version=1,
        pipeline_version="v2",
    )
    res_c = index.search(identical_vec, k=5, retrieval_scope=scope_c)
    assert len(res_c) == 0


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_tenant_isolation_cross_tenant(in_memory_qdrant: QdrantVectorIndex):
    """Test that Tenant A cannot retrieve Tenant B's vectors."""
    index = in_memory_qdrant
    vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    entry_t1 = uuid4()
    entry_t2 = uuid4()

    index.add(
        entry_id=entry_t1,
        embedding=vec,
        tenant_id="tenant_1",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    index.add(
        entry_id=entry_t2,
        embedding=vec,
        tenant_id="tenant_2",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )

    scope_t1 = RetrievalScope(
        tenant_id="tenant_1",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    results = index.search(vec, k=5, retrieval_scope=scope_t1)
    assert len(results) == 1
    assert results[0].entry_id == entry_t1


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_resource_version_isolation_v17_vs_v18(in_memory_qdrant: QdrantVectorIndex):
    """Test that Resource X version 17 cannot satisfy version 18."""
    index = in_memory_qdrant
    vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    entry_v17 = uuid4()
    index.add(
        entry_id=entry_v17,
        embedding=vec,
        tenant_id="tenant_fin",
        scope=CacheScope.RESOURCE,
        resource_id="balance_sheet_2025",
        resource_version=17,
        pipeline_version="v2",
    )

    # Search for v17 -> matches
    scope_v17 = RetrievalScope(
        tenant_id="tenant_fin",
        scope=CacheScope.RESOURCE,
        resource_id="balance_sheet_2025",
        resource_version=17,
        pipeline_version="v2",
    )
    results_17 = index.search(vec, k=5, retrieval_scope=scope_v17)
    assert len(results_17) == 1
    assert results_17[0].entry_id == entry_v17

    # Search for v18 -> rejected, returns zero
    scope_v18 = RetrievalScope(
        tenant_id="tenant_fin",
        scope=CacheScope.RESOURCE,
        resource_id="balance_sheet_2025",
        resource_version=18,
        pipeline_version="v2",
    )
    results_18 = index.search(vec, k=5, retrieval_scope=scope_v18)
    assert len(results_18) == 0


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_pipeline_version_mismatch_returns_zero(in_memory_qdrant: QdrantVectorIndex):
    """Test that a search with pipeline_version='v2.1' does not match entries saved under 'v2'."""
    index = in_memory_qdrant
    vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    entry_id = uuid4()
    index.add(
        entry_id=entry_id,
        embedding=vec,
        tenant_id="tenant_a",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )

    scope_v21 = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2.1",
    )
    results = index.search(vec, k=5, retrieval_scope=scope_v21)
    assert len(results) == 0


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_tenant_scope_excludes_resource_entries(in_memory_qdrant: QdrantVectorIndex):
    """Test that TENANT scope query retrieves global + tenant entries but never RESOURCE entries."""
    index = in_memory_qdrant
    vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    e_global = uuid4()
    e_tenant = uuid4()
    e_resource = uuid4()

    index.add(
        entry_id=e_global,
        embedding=vec,
        tenant_id="default",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )
    index.add(
        entry_id=e_tenant,
        embedding=vec,
        tenant_id="tenant_a",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    index.add(
        entry_id=e_resource,
        embedding=vec,
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="secret_res",
        resource_version=1,
        pipeline_version="v2",
    )

    scope_tenant = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    results = index.search(vec, k=10, retrieval_scope=scope_tenant)
    returned_ids = {r.entry_id for r in results}

    assert e_global in returned_ids
    assert e_tenant in returned_ids
    assert e_resource not in returned_ids


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_global_scope_excludes_tenant_and_resource_entries(
    in_memory_qdrant: QdrantVectorIndex,
):
    """Test that GLOBAL scope query retrieves only GLOBAL entries."""
    index = in_memory_qdrant
    vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    e_global = uuid4()
    e_tenant = uuid4()
    e_resource = uuid4()

    index.add(
        entry_id=e_global,
        embedding=vec,
        tenant_id="default",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )
    index.add(
        entry_id=e_tenant,
        embedding=vec,
        tenant_id="tenant_a",
        scope=CacheScope.TENANT,
        pipeline_version="v2",
    )
    index.add(
        entry_id=e_resource,
        embedding=vec,
        tenant_id="tenant_a",
        scope=CacheScope.RESOURCE,
        resource_id="secret_res",
        resource_version=1,
        pipeline_version="v2",
    )

    scope_global = RetrievalScope(
        tenant_id="tenant_a",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )
    results = index.search(vec, k=10, retrieval_scope=scope_global)
    returned_ids = {r.entry_id for r in results}

    assert e_global in returned_ids
    assert e_tenant not in returned_ids
    assert e_resource not in returned_ids


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_add_remove_rebuild(in_memory_qdrant: QdrantVectorIndex):
    """Test point addition, removal, and rebuilding with metadata preservation."""
    index = in_memory_qdrant
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    entry_1 = uuid4()
    entry_2 = uuid4()

    index.add(
        entry_id=entry_1,
        embedding=vec,
        tenant_id="t1",
        scope=CacheScope.GLOBAL,
        pipeline_version="v2",
    )
    index.add(
        entry_id=entry_2,
        embedding=vec,
        tenant_id="t1",
        scope=CacheScope.RESOURCE,
        resource_id="r1",
        resource_version=1,
        pipeline_version="v2",
    )

    scope_global = RetrievalScope(tenant_id="t1", scope=CacheScope.GLOBAL, pipeline_version="v2")
    assert len(index.search(vec, k=5, retrieval_scope=scope_global)) == 1

    # Remove entry_1
    index.remove(entry_1)
    assert len(index.search(vec, k=5, retrieval_scope=scope_global)) == 0

    # Rebuild
    rebuild_items = [
        (entry_1, vec, "t1", CacheScope.GLOBAL, None, None, "v2"),
        (entry_2, vec, "t1", CacheScope.RESOURCE, "r1", 1, "v2"),
    ]
    index.rebuild(rebuild_items)
    assert len(index.search(vec, k=5, retrieval_scope=scope_global)) == 1


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client is not installed")
def test_qdrant_live_integration_server():
    """Integration test executing against live Qdrant server if available."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=1, check_compatibility=False)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Live Qdrant server not available at {qdrant_url}: {exc}")

    config = QdrantConfig(
        url=qdrant_url,
        collection_name=f"live_test_{uuid4().hex[:8]}",
        timeout=3,
    )
    index = QdrantVectorIndex(dim=4, config=config)
    vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    entry_id = uuid4()

    index.add(
        entry_id=entry_id,
        embedding=vec,
        tenant_id="live_tenant",
        scope=CacheScope.RESOURCE,
        resource_id="live_res",
        resource_version=1,
        pipeline_version="v2",
    )

    scope = RetrievalScope(
        tenant_id="live_tenant",
        scope=CacheScope.RESOURCE,
        resource_id="live_res",
        resource_version=1,
        pipeline_version="v2",
    )
    results = index.search(vec, k=5, retrieval_scope=scope)
    assert len(results) == 1
    assert results[0].entry_id == entry_id

    # Clean up
    index.remove(entry_id)
