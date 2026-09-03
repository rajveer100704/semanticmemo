"""Resource and Version Management Service."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from semanticmemo._logging import get_logger
from semanticmemo.models import Resource
from semanticmemo.store.protocol import ResourceStore

logger = get_logger(__name__)


class ResourceService:
    """Manages resource metadata, version resolution, and atomic invalidation."""

    def __init__(
        self,
        store: ResourceStore | Any,
        event_listener: Callable[[str, int], None] | None = None,
    ) -> None:
        self.store = store
        self.event_listener = event_listener

    def register_resource(
        self,
        resource_id: str,
        tenant_id: str,
        initial_version: int = 1,
    ) -> Resource:
        """Register or update a resource under a tenant."""
        return self.store.create_resource(
            resource_id=resource_id,
            tenant_id=tenant_id,
            initial_version=initial_version,
        )

    def get_resource(self, resource_id: str) -> Resource | None:
        """Fetch resource metadata, failing closed (returning None) if not found."""
        return self.store.get_resource(resource_id)

    def get_resource_version(self, resource_id: str) -> int | None:
        """Fetch the current version of a resource."""
        res = self.get_resource(resource_id)
        return res.current_version if res is not None else None

    def invalidate_resource(self, resource_id: str) -> int:
        """Atomically increment the resource version and emit an invalidation notification.

        Logical invalidation is immediate: queries requiring the new version will never
        match old version entries, even if notification delivery fails.
        """
        raw_version = self.store.increment_resource_version(resource_id)
        new_version = int(raw_version) if raw_version is not None else 1
        logger.info(
            "Resource %s invalidated -> incremented to version %d",
            resource_id,
            new_version,
        )
        if self.event_listener is not None:
            try:
                self.event_listener(resource_id, new_version)
            except Exception as exc:
                logger.error(
                    "Error notifying invalidation listener for resource %s: %s",
                    resource_id,
                    exc,
                )
        return new_version
