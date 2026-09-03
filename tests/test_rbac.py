"""Tests for RBAC and Resource Access Control."""

import pytest

from semanticmemo.models import PermissionName
from semanticmemo.rbac import AuthorizationService, ResourceService
from semanticmemo.store import SQLiteCacheStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "rbac_test.db"
    store_inst = SQLiteCacheStore(db_file)
    yield store_inst
    store_inst.close()


def test_user_creation_and_role_assignment(store):
    user = store.create_user("user_1", "tenant_a")
    assert user.id == "user_1"
    assert user.tenant_id == "tenant_a"

    store.assign_role_to_user("user_1", "viewer")
    roles = store.get_user_roles("user_1")
    assert "viewer" in roles

    # Viewer permissions
    authz = AuthorizationService(store)
    assert authz.check_permission("user_1", PermissionName.CACHE_READ) is True
    assert authz.check_permission("user_1", PermissionName.CACHE_WRITE) is False
    assert authz.check_permission("user_1", PermissionName.CACHE_INVALIDATE) is False


def test_analyst_and_admin_permissions(store):
    store.create_user("analyst_1", "tenant_a")
    store.assign_role_to_user("analyst_1", "analyst")

    store.create_user("admin_1", "tenant_a")
    store.assign_role_to_user("admin_1", "admin")

    authz = AuthorizationService(store)
    # Analyst
    assert authz.check_permission("analyst_1", PermissionName.CACHE_READ) is True
    assert authz.check_permission("analyst_1", PermissionName.CACHE_WRITE) is True
    assert authz.check_permission("analyst_1", PermissionName.CACHE_INVALIDATE) is False

    # Admin
    assert authz.check_permission("admin_1", PermissionName.CACHE_READ) is True
    assert authz.check_permission("admin_1", PermissionName.CACHE_WRITE) is True
    assert authz.check_permission("admin_1", PermissionName.CACHE_INVALIDATE) is True
    assert authz.check_permission("admin_1", PermissionName.RESOURCE_MANAGE) is True


def test_resource_tenant_isolation(store):
    store.create_user("user_a", "tenant_a")
    store.assign_role_to_user("user_a", "viewer")

    store.create_user("user_b", "tenant_b")
    store.assign_role_to_user("user_b", "viewer")

    res_service = ResourceService(store)
    res_service.register_resource("res_a", "tenant_a", initial_version=1)

    authz = AuthorizationService(store)
    # User A in Tenant A can read res_a
    assert authz.can_read_resource("user_a", "res_a") is True
    # User B in Tenant B cannot read res_a (Cross-tenant access blocked)
    assert authz.can_read_resource("user_b", "res_a") is False


def test_resource_specific_acl(store):
    store.create_user("user_viewer", "tenant_a")
    store.assign_role_to_user("user_viewer", "viewer")

    store.create_user("user_analyst", "tenant_a")
    store.assign_role_to_user("user_analyst", "analyst")

    res_service = ResourceService(store)
    res_service.register_resource("restricted_res", "tenant_a")

    # Explicitly grant access only to analyst
    store.grant_resource_access("restricted_res", "analyst", PermissionName.RESOURCE_READ.value)

    authz = AuthorizationService(store)
    assert authz.can_read_resource("user_analyst", "restricted_res") is True
    assert authz.can_read_resource("user_viewer", "restricted_res") is False


def test_atomic_resource_version_invalidation(store):
    res_service = ResourceService(store)
    res = res_service.register_resource("doc_1", "tenant_a", initial_version=1)
    assert res.current_version == 1

    v2 = res_service.invalidate_resource("doc_1")
    assert v2 == 2
    assert res_service.get_resource_version("doc_1") == 2

    v3 = res_service.invalidate_resource("doc_1")
    assert v3 == 3
    assert res_service.get_resource_version("doc_1") == 3


def test_fail_closed_on_unknown_user_or_resource(store):
    authz = AuthorizationService(store)
    assert authz.check_permission("non_existent_user", "cache:read") is False
    assert authz.can_read_resource("non_existent_user", "some_res") is False

    store.create_user("valid_user", "tenant_a")
    assert authz.can_read_resource("valid_user", "non_existent_res") is False
