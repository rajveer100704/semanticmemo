"""Unit tests for low-cardinality Prometheus-compatible metrics registry."""

from __future__ import annotations

import math

import pytest

from semanticmemo.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    SemanticMemoMetrics,
    StageTimer,
    _sanitize_labels,
)


def test_sanitize_labels_filters_unbounded_cardinality() -> None:
    raw_labels = {
        "tenant_id": "tenant-abc-12345",
        "user_id": "user-999",
        "prompt": "SELECT * FROM secret_table",
        "scope": "resource",
        "decision": "hit",
        "domain": "finance",
        "backend": "postgres",
    }
    sanitized = _sanitize_labels(raw_labels)
    assert "tenant_id" not in sanitized
    assert "user_id" not in sanitized
    assert "prompt" not in sanitized
    assert sanitized["scope"] == "resource"
    assert sanitized["decision"] == "hit"
    assert sanitized["domain"] == "finance"
    assert sanitized["backend"] == "postgres"


def test_counter_monotonic_increment_and_collection() -> None:
    counter = Counter("test_requests_total", "Test counter")
    counter.inc(labels={"domain": "finance", "decision": "hit"})
    counter.inc(2.0, labels={"domain": "finance", "decision": "hit"})
    counter.inc(1.0, labels={"domain": "finance", "decision": "miss"})

    assert counter.get({"domain": "finance", "decision": "hit"}) == 3.0
    assert counter.get({"domain": "finance", "decision": "miss"}) == 1.0
    assert counter.get({"domain": "healthcare", "decision": "hit"}) == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        counter.inc(-1.0)

    samples = counter.collect()
    assert len(samples) == 2
    assert samples[0].name == "test_requests_total"
    assert samples[0].value in (1.0, 3.0)


def test_gauge_increment_decrement_and_set() -> None:
    gauge = Gauge("test_active_entries", "Active entries gauge")
    gauge.set(10.0, labels={"scope": "resource"})
    assert gauge.get({"scope": "resource"}) == 10.0

    gauge.inc(5.0, labels={"scope": "resource"})
    assert gauge.get({"scope": "resource"}) == 15.0

    gauge.dec(3.0, labels={"scope": "resource"})
    assert gauge.get({"scope": "resource"}) == 12.0

    samples = gauge.collect()
    assert len(samples) == 1
    assert samples[0].value == 12.0


def test_histogram_buckets_and_summary_metrics() -> None:
    buckets = (0.001, 0.005, 0.010, 0.050)
    hist = Histogram("test_latency_seconds", "Latency histogram", buckets=buckets)

    hist.observe(0.0005, labels={"stage": "embed"})  # <= 0.001
    hist.observe(0.004, labels={"stage": "embed"})  # <= 0.005
    hist.observe(0.020, labels={"stage": "embed"})  # <= 0.050

    samples = hist.collect()
    bucket_samples = {s.labels.get("le"): s.value for s in samples if s.name.endswith("_bucket")}
    assert bucket_samples.get("0.001") == 1.0
    assert bucket_samples.get("0.005") == 2.0
    assert bucket_samples.get("0.01") == 2.0
    assert bucket_samples.get("0.05") == 3.0
    assert bucket_samples.get("+Inf") == 3.0

    sum_samples = [s for s in samples if s.name.endswith("_sum")]
    assert len(sum_samples) == 1
    assert math.isclose(sum_samples[0].value, 0.0245, rel_tol=1e-5)

    count_samples = [s for s in samples if s.name.endswith("_count")]
    assert len(count_samples) == 1
    assert count_samples[0].value == 3.0


def test_semanticmemo_metrics_registry_and_openmetrics_export() -> None:
    SemanticMemoMetrics.reset()
    metrics = SemanticMemoMetrics.default()

    metrics.cache_requests_total.inc(
        labels={"scope": "tenant", "decision": "hit", "domain": "finance", "backend": "postgres"}
    )
    metrics.request_duration_seconds.observe(0.002, labels={"stage": "vector_search"})
    metrics.candidate_space_reduction_ratio.observe(0.95, labels={"scope": "resource"})
    metrics.single_flight_total.inc(labels={"role": "leader"})
    metrics.tokens_saved_total.inc(150.0, labels={"type": "output"})
    metrics.cost_saved_usd_total.inc(0.0045)
    metrics.outbox_events_total.inc(
        labels={"event_type": "resource.updated", "status": "published"}
    )
    metrics.reconciliation_total.inc(5.0, labels={"action": "repaired"})
    metrics.cache_entries_active.set(42.0)

    text = metrics.export_openmetrics_text()
    assert "# HELP semanticmemo_cache_requests_total" in text
    assert "# TYPE semanticmemo_cache_requests_total counter"
    assert 'decision="hit"' in text
    assert 'domain="finance"' in text
    assert 'backend="postgres"' in text
    assert "# HELP semanticmemo_request_duration_seconds" in text
    assert 'stage="vector_search"' in text
    assert "semanticmemo_cost_saved_usd_total 0.0045" in text
    assert "semanticmemo_cache_entries_active 42.0" in text
    assert "# EOF" in text


def test_stage_timer_context_manager() -> None:
    hist = Histogram("test_stage_duration", "Stage timer test")
    with StageTimer(hist, stage="embed"):
        pass  # instantaneous block

    samples = hist.collect()
    count_samples = [s for s in samples if s.name.endswith("_count")]
    assert len(count_samples) == 1
    assert count_samples[0].value == 1.0
