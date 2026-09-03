"""Outbox Relay Worker for Transactional Outbox Event Publishing."""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from semanticmemo._logging import get_logger
from semanticmemo.events.bus import EventBus
from semanticmemo.events.models import OutboxEvent
from semanticmemo.store.protocol import OutboxStore
from semanticmemo.telemetry.metrics import SemanticMemoMetrics

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(tz=UTC)


class OutboxRelay:
    """Background relay processor claiming and publishing outbox events to EventBus."""

    def __init__(
        self,
        store: OutboxStore | Any,
        event_bus: EventBus | Any,
        *,
        relay_id: str | None = None,
        claim_lease_seconds: float = 30.0,
        max_retries: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self.event_bus = event_bus
        self.relay_id = relay_id or f"relay-{uuid4().hex[:8]}"
        self.claim_lease_seconds = claim_lease_seconds
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def run_once(self, batch_size: int = 100) -> int:
        """Claim a batch of pending/expired events and publish them to EventBus."""
        events: list[OutboxEvent] = self.store.claim_outbox_events(
            limit=batch_size,
            claim_owner=self.relay_id,
            lease_seconds=self.claim_lease_seconds,
        )
        if not events:
            return 0

        metrics = SemanticMemoMetrics.default()
        try:
            await self.event_bus.publish_batch(events)
            published_ids = [e.id for e in events]
            self.store.mark_outbox_published(published_ids)
            for event in events:
                evt_type = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                metrics.outbox_events_total.inc(
                    labels={"event_type": evt_type, "status": "published"}
                )
            logger.debug(
                "OutboxRelay %s published %d events",
                self.relay_id,
                len(published_ids),
            )
            return len(published_ids)
        except Exception as exc:
            logger.error(
                "OutboxRelay %s error publishing event batch: %s (scheduling retries)",
                self.relay_id,
                exc,
            )
            now = _now()
            for event in events:
                jitter = random.uniform(0.0, 0.5)
                backoff = min(
                    self.max_backoff_seconds,
                    self.base_backoff_seconds * (2**event.retry_count) + jitter,
                )
                next_attempt = now + timedelta(seconds=backoff)
                is_dead_letter = event.retry_count >= self.max_retries
                self.store.mark_outbox_retry(
                    event_id=event.id,
                    error=str(exc),
                    next_attempt_at=next_attempt,
                    dead_letter=is_dead_letter,
                )
                evt_type = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                metrics.outbox_events_total.inc(
                    labels={
                        "event_type": evt_type,
                        "status": "dead_letter" if is_dead_letter else "retry",
                    }
                )
            return 0

    async def start(self, poll_interval_seconds: float = 0.5) -> None:
        """Start continuous background polling task."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(poll_interval_seconds))

    async def _poll_loop(self, poll_interval_seconds: float) -> None:
        while self._running:
            try:
                published = await self.run_once()
                if published == 0:
                    await asyncio.sleep(poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in OutboxRelay poll loop: %s", exc)

    async def stop(self) -> None:
        """Stop continuous background polling task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
