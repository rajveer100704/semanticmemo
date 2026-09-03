"""API Router Tests for SemanticMemo V2."""

import pytest

from semanticmemo import CacheConfig, SemanticMemo
from semanticmemo.api.router import FASTAPI_AVAILABLE, create_cache_router
from semanticmemo.embedding.service import HashEmbeddingProvider


@pytest.fixture
def memo_instance(tmp_path):
    cfg = CacheConfig(
        db_path=tmp_path / "api_test.db",
        embedding_dim=32,
    )
    memo = SemanticMemo(
        domain="api_domain",
        config=cfg,
        embedding_provider=HashEmbeddingProvider(dim=32),
        use_faiss=False,
    )
    memo.create_user("user_authorized", "tenant_a", role="analyst")
    memo.create_user("user_viewer", "tenant_a", role="viewer")
    memo.register_resource("res_api", "tenant_a", initial_version=1)
    yield memo
    memo.close()


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed in environment")
def test_fastapi_router_endpoints(memo_instance):
    from fastapi import FastAPI  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    app = FastAPI()
    router = create_cache_router(memo_instance)
    app.include_router(router)

    client = TestClient(app)

    # 1. Authorized query with resource_id
    response = client.post(
        "/v2/cache/query",
        json={"prompt": "API prompt query", "resource_id": "res_api"},
        headers={"Authorization": "Bearer user_authorized"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["was_cache_hit"] is False
    assert "LLM generated answer" in data["response"]

    # 2. Second query -> Hit
    response_hit = client.post(
        "/v2/cache/query",
        json={"prompt": "API prompt query", "resource_id": "res_api"},
        headers={"Authorization": "Bearer user_authorized"},
    )
    assert response_hit.status_code == 200
    assert response_hit.json()["was_cache_hit"] is True

    # 3. Unauthorized access to unknown/forbidden resource -> 403
    memo_instance.create_user("user_tenant_b", "tenant_b", role="analyst")
    response_forbidden = client.post(
        "/v2/cache/query",
        json={"prompt": "API prompt query", "resource_id": "res_api"},
        headers={"Authorization": "Bearer user_tenant_b"},
    )
    assert response_forbidden.status_code == 403

    # 4. Bad request (empty prompt) -> 400 or 422
    response_bad = client.post(
        "/v2/cache/query",
        json={"prompt": ""},
        headers={"Authorization": "Bearer user_authorized"},
    )
    assert response_bad.status_code in (400, 422)
