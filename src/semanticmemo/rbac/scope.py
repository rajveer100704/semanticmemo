"""Cache Scope Resolution Service."""

from __future__ import annotations

from semanticmemo._logging import get_logger
from semanticmemo.models import CacheScope, RetrievalScope
from semanticmemo.rbac.authz import AuthorizationService
from semanticmemo.rbac.resources import ResourceService

logger = get_logger(__name__)


class CacheScopeResolver:
    """Resolves authorized cache retrieval scope with strict fail-closed guarantees."""

    def __init__(
        self,
        authz_service: AuthorizationService,
        resource_service: ResourceService,
        pipeline_version: str = "v2",
    ) -> None:
        self.authz_service = authz_service
        self.resource_service = resource_service
        self.pipeline_version = pipeline_version

    def resolve_scope(
        self,
        *,
        user_id: str,
        resource_id: str | None = None,
        prefer_global: bool = False,
    ) -> RetrievalScope:
        """Resolve the mandatory RetrievalScope for a request.

        Fails closed: if resource is unauthorized or not found, raises PermissionError.
        Under no circumstances does a failed resource lookup fall back to TENANT or GLOBAL.
        """
        user = self.authz_service.authenticate_user(user_id)
        if user is None:
            msg = f"Unauthenticated principal or user not found: {user_id}"
            logger.warning(msg)
            raise PermissionError(msg)

        tenant_id = user.tenant_id

        if resource_id is not None:
            # Enforce pre-retrieval RBAC
            if not self.authz_service.can_read_resource(user_id, resource_id):
                msg = f"User {user_id} unauthorized to access resource {resource_id}"
                logger.warning(msg)
                raise PermissionError(msg)

            # Resolve resource and its current version
            resource = self.resource_service.get_resource(resource_id)
            if resource is None or resource.tenant_id != tenant_id:
                msg = f"Resource {resource_id} not found in tenant {tenant_id}"
                logger.warning(msg)
                raise PermissionError(msg)

            return RetrievalScope(
                tenant_id=tenant_id,
                scope=CacheScope.RESOURCE,
                resource_id=resource_id,
                resource_version=resource.current_version,
                pipeline_version=self.pipeline_version,
            )

        # When resource_id is absent: strictly TENANT or GLOBAL
        scope = CacheScope.GLOBAL if prefer_global else CacheScope.TENANT
        return RetrievalScope(
            tenant_id=tenant_id,
            scope=scope,
            resource_id=None,
            resource_version=None,
            pipeline_version=self.pipeline_version,
        )
