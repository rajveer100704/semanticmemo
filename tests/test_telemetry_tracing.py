"""Unit tests for distributed tracing, privacy sanitization, and W3C context propagation."""

from __future__ import annotations

from typing import Any

import pytest

from semanticmemo.events.consumer import InvalidationConsumer
from semanticmemo.events.models import EventType, OutboxEvent
from semanticmemo.store.sqlite_store import SQLiteCacheStore
from semanticmemo.telemetry.tracing import (
    DebugLogSpan,
    NoOpSpan,
    SemanticMemoTracer,
    get_tracer,
    hash_identifier,
    sanitize_attributes,
)


def test_hash_identifier() -> None:
    assert hash_identifier(None) == "none"
    assert hash_identifier("") == "none"
    h1 = hash_identifier("tenant-100")
    h2 = hash_identifier("tenant-100")
    h3 = hash_identifier("tenant-200")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_sanitize_attributes_rejects_sensitive_data() -> None:
    raw_attrs = {
        "prompt": "SELECT * FROM payroll_records WHERE salary > 200000",
        "response": "Here is the confidential payroll data...",
        "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token": "secret_bearer_token",
        "api_key": "sk-live-123456789",
        "tenant_id": "tenant-finance",
        "user_id": "user-42",
        "resource_id": "res-99",
        "domain": "finance",
        "prompt_length": 54,
    }

    sanitized = sanitize_attributes(raw_attrs, include_raw_identifiers=False)
    # Strictly ensure raw confidential text is excluded
    assert "prompt" not in sanitized
    assert "semanticmemo.prompt" not in sanitized
    assert "response" not in sanitized
    assert "jwt" not in sanitized
    assert "token" not in sanitized
    assert "api_key" not in sanitized

    # Verify hashed identifiers
    assert "semanticmemo.tenant_hash" in sanitized
    assert "semanticmemo.user_hash" in sanitized
    assert "semanticmemo.resource_hash" in sanitized
    assert sanitized["semanticmemo.domain"] == "finance"
    assert sanitized["semanticmemo.prompt_length"] == 54


def test_noop_tracer_fallback() -> None:
    tracer = SemanticMemoTracer(debug_mode=False, enable_otel=False)
    with tracer.start_span("test.noop.span", attributes={"domain": "finance"}) as span:
        assert isinstance(span, NoOpSpan)
        span.set_attribute("key", "val")
        span.set_status_ok()
        span.record_exception(RuntimeError("test err"))


def test_debug_log_span() -> None:
    tracer = SemanticMemoTracer(debug_mode=True, enable_otel=False)
    with tracer.start_span("test.debug.span", attributes={"domain": "finance"}) as span:
        assert isinstance(span, DebugLogSpan)
        span.set_attribute("extra", "value")
        span.record_exception(ValueError("sample error"))


def test_w3c_traceparent_injection_and_extraction() -> None:
    tracer = get_tracer()
    traceparent_val = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    carrier: dict[str, str] = {"traceparent": traceparent_val}
    extracted = tracer.extract_trace_context(carrier)
    assert extracted == traceparent_val


@pytest.mark.asyncio
async def test_invalidation_consumer_tracing_span_linkage(tmp_path: Any) -> None:
    db_path = tmp_path / "test_consumer.db"
    store = SQLiteCacheStore(str(db_path))

    handled_events: list[OutboxEvent] = []

    def on_update(evt: OutboxEvent) -> None:
        handled_events.append(evt)

    consumer = InvalidationConsumer(
        consumer_name="test-telemetry-consumer",
        processed_store=store,
        on_resource_updated=on_update,
    )

    traceparent_val = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    event = OutboxEvent(
        event_type=EventType.RESOURCE_UPDATED,
        aggregate_type="resource",
        aggregate_id="res-trace-1",
        tenant_id="tenant-trace-1",
        payload={"new_version": 2},
        traceparent=traceparent_val,
    )

    processed = await consumer.handle_event(event)
    assert processed is True
    assert len(handled_events) == 1
    assert handled_events[0].traceparent == traceparent_val
