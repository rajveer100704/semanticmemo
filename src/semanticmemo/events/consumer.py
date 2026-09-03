"""Idempotent Invalidation Consumer for Distributed Event Processing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from semanticmemo._logging import get_logger
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.store.protocol import ProcessedEventStore
from semanticmemo.telemetry.tracing import get_tracer

logger = get_logger(__name__)

CallbackType = Callable[[OutboxEvent], None] | Callable[[OutboxEvent], Awaitable[None]]


class InvalidationConsumer:
    """Consumes outbox events with durable event_id deduplication and invalidation dispatch."""

    def __init__(
        self,
        consumer_name: str,
        processed_store: ProcessedEventStore | Any,
        *,
        on_resource_updated: CallbackType | None = None,
        on_resource_deleted: CallbackType | None = None,
        on_permission_changed: CallbackType | None = None,
        on_user_role_changed: CallbackType | None = None,
        on_pipeline_changed: CallbackType | None = None,
    ) -> None:
        self.consumer_name = consumer_name
        self.processed_store = processed_store
        self.on_resource_updated = on_resource_updated
        self.on_resource_deleted = on_resource_deleted
        self.on_permission_changed = on_permission_changed
        self.on_user_role_changed = on_user_role_changed
        self.on_pipeline_changed = on_pipeline_changed
        self._processed_count = 0

    @property
    def processed_count(self) -> int:
        return self._processed_count

    async def handle_event(self, event: OutboxEvent) -> bool:
        """Handle incoming event with durable idempotency.

        Returns True if event was newly processed, False if duplicate skipped.
        """
        tracer = get_tracer()
        span_attrs = {
            "consumer_name": self.consumer_name,
            "event_id": str(event.id),
            "event_type": str(event.event_type),
            "aggregate_type": event.aggregate_type,
            "tenant_hash": event.tenant_id,
        }

        with tracer.start_span(
            "semanticmemo.consumer.handle_event",
            attributes=span_attrs,
            traceparent=event.traceparent,
        ) as span:
            is_new = self.processed_store.record_processed_event(
                event_id=event.id,
                consumer_name=self.consumer_name,
            )
            if not is_new:
                span.set_attribute("semanticmemo.deduplicated", True)
                logger.debug(
                    "Consumer %s: Duplicate event %s already processed (skipping)",
                    self.consumer_name,
                    event.id,
                )
                return False

            self._processed_count += 1
            evt_type = (
                event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type)
            )

            callback: CallbackType | None = None
            if evt_type == EventType.RESOURCE_UPDATED.value:
                callback = self.on_resource_updated
            elif evt_type == EventType.RESOURCE_DELETED.value:
                callback = self.on_resource_deleted
            elif evt_type == EventType.PERMISSION_CHANGED.value:
                callback = self.on_permission_changed
            elif evt_type == EventType.USER_ROLE_CHANGED.value:
                callback = self.on_user_role_changed
            elif evt_type == EventType.PIPELINE_CHANGED.value:
                callback = self.on_pipeline_changed

            if callback is not None:
                try:
                    res = callback(event)
                    if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                        await res  # type: ignore[misc]
                    span.set_status_ok()
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status_error(str(exc))
                    logger.error(
                        "Consumer %s error executing callback for event %s: %s",
                        self.consumer_name,
                        event.id,
                        exc,
                    )
            return True
