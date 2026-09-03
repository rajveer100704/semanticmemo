"""Authorization and RBAC Service."""

from __future__ import annotations

from typing import Any

from semanticmemo._logging import get_logger
from semanticmemo.models import PermissionName, User

logger = get_logger(__name__)


class AuthorizationService:
    """Evaluates user permissions and enforces strict fail-closed access control."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def authenticate_user(self, user_id: str) -> User | None:
        """Authenticate user and derive tenant context from persistence."""
        return self.store.get_user(user_id)

    def check_permission(
        self,
        user_id: str,
        permission: str | PermissionName,
        resource_id: str | None = None,
    ) -> bool:
        """Check if user has permission, failing closed on any denial or missing context."""
        perm_str = permission.value if isinstance(permission, PermissionName) else str(permission)
        try:
            allowed = self.store.check_user_permission(
                user_id=user_id,
                permission_name=perm_str,
                resource_id=resource_id,
            )
            if not allowed:
                logger.warning(
                    "RBAC denied: user_id=%s perm=%s resource_id=%s",
                    user_id,
                    perm_str,
                    resource_id,
                )
            return allowed
        except Exception as exc:
            logger.error(
                "Error checking RBAC permission for user %s: %s (failing closed)",
                user_id,
                exc,
            )
            return False

    def can_read_resource(self, user_id: str, resource_id: str) -> bool:
        """Check if user is authorized to read the given resource."""
        return self.check_permission(
            user_id=user_id,
            permission=PermissionName.RESOURCE_READ,
            resource_id=resource_id,
        )

    def can_write_cache(self, user_id: str, resource_id: str | None = None) -> bool:
        """Check if user has cache write permission for the scope."""
        has_cache_write = self.check_permission(
            user_id=user_id,
            permission=PermissionName.CACHE_WRITE,
        )
        if not has_cache_write:
            return False
        if resource_id is not None:
            # Resource-scoped writes require both cache:write and resource:read
            return self.can_read_resource(user_id, resource_id)
        return True

    def can_invalidate_cache(self, user_id: str) -> bool:
        """Check if user has administrative cache invalidation permission."""
        return self.check_permission(
            user_id=user_id,
            permission=PermissionName.CACHE_INVALIDATE,
        )
