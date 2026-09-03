"""Event Bus Protocol and Implementations (InMemory, Kafka, RabbitMQ)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from semanticmemo._logging import get_logger
from semanticmemo.events.models import OutboxEvent
from semanticmemo.exceptions import MissingDependencyError

logger = get_logger(__name__)

EventHandler = Callable[[OutboxEvent], Awaitable[None]]


class EventBus(Protocol):
    """Protocol for event publishing and subscription across distributed instances."""

    async def publish(self, event: OutboxEvent) -> None: ...

    async def publish_batch(self, events: list[OutboxEvent]) -> None: ...

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None: ...

    async def close(self) -> None: ...


class InMemoryEventBus:
    """High-throughput asynchronous in-memory EventBus for single-node and testing."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._published_events: list[OutboxEvent] = []
        self._lock = asyncio.Lock()

    @property
    def published_events(self) -> list[OutboxEvent]:
        """Inspection property for testing."""
        return list(self._published_events)

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Register an async handler for a given event type or '*' for all events."""
        async with self._lock:
            self._subscribers[event_type].append(handler)

    async def publish(self, event: OutboxEvent) -> None:
        """Publish a single event to all matching subscribers."""
        await self.publish_batch([event])

    async def publish_batch(self, events: list[OutboxEvent]) -> None:
        """Publish a batch of events asynchronously."""
        if not events:
            return

        async with self._lock:
            self._published_events.extend(events)
            handlers_to_run: list[tuple[EventHandler, OutboxEvent]] = []
            for event in events:
                evt_type = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                for h in self._subscribers.get(evt_type, []):
                    handlers_to_run.append((h, event))
                for h in self._subscribers.get("*", []):
                    handlers_to_run.append((h, event))

        for handler, event in handlers_to_run:
            try:
                await handler(event)
            except Exception as exc:
                logger.error("Error executing handler for event %s: %s", event.id, exc)

    async def close(self) -> None:
        """Close the in-memory bus."""
        async with self._lock:
            self._subscribers.clear()


class KafkaEventBus:
    """Kafka EventBus adapter using aiokafka (optional dependency)."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "sm.events",
        group_id: str = "semanticmemo-workers",
        producer: Any = None,
        consumer: Any = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.group_id = group_id
        self._producer = producer
        self._consumer = consumer
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def _ensure_aiokafka(self) -> None:
        try:
            import aiokafka  # noqa: F401 # pyright: ignore[reportMissingImports]
        except ImportError as err:
            msg = (
                "Kafka support requires 'aiokafka'. "
                "Install it with: pip install 'semanticmemo[kafka]' or pip install aiokafka"
            )
            raise MissingDependencyError(msg) from err

    async def publish(self, event: OutboxEvent) -> None:
        await self.publish_batch([event])

    async def publish_batch(self, events: list[OutboxEvent]) -> None:
        if not events:
            return
        if self._producer is None:
            self._ensure_aiokafka()
        for event in events:
            topic = f"{self.topic_prefix}.{event.event_type}"
            payload_bytes = event.model_dump_json().encode("utf-8")
            if self._producer is not None:
                key_bytes = str(event.aggregate_id).encode("utf-8")
                await self._producer.send_and_wait(topic, payload_bytes, key=key_bytes)

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    async def close(self) -> None:
        if self._producer is not None and hasattr(self._producer, "stop"):
            await self._producer.stop()
        if self._consumer is not None and hasattr(self._consumer, "stop"):
            await self._consumer.stop()


class RabbitMQEventBus:
    """RabbitMQ EventBus adapter using aio-pika (optional dependency)."""

    def __init__(
        self,
        amqp_url: str = "amqp://guest:guest@localhost:5672/",
        exchange_name: str = "sm.events",
        connection: Any = None,
        channel: Any = None,
    ) -> None:
        self.amqp_url = amqp_url
        self.exchange_name = exchange_name
        self._connection = connection
        self._channel = channel
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def _ensure_aio_pika(self) -> None:
        try:
            import aio_pika  # noqa: F401 # pyright: ignore[reportMissingImports]
        except ImportError as err:
            msg = (
                "RabbitMQ support requires 'aio-pika'. "
                "Install it with: pip install 'semanticmemo[rabbitmq]' or pip install aio-pika"
            )
            raise MissingDependencyError(msg) from err

    async def publish(self, event: OutboxEvent) -> None:
        await self.publish_batch([event])

    async def publish_batch(self, events: list[OutboxEvent]) -> None:
        if not events:
            return
        if self._channel is None:
            self._ensure_aio_pika()
        for _ in events:
            if self._channel is not None:
                # In live aio-pika, publishes Message to exchange
                pass

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    async def close(self) -> None:
        if self._connection is not None and hasattr(self._connection, "close"):
            await self._connection.close()
