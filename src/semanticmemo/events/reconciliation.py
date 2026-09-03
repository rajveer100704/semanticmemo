"""Version-Guarded Index Reconciliation Engine."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import BaseModel

from semanticmemo._logging import get_logger
from semanticmemo.models import CacheScope

logger = get_logger(__name__)


class ReconciliationReport(BaseModel):
    """Execution summary of index reconciliation between PostgreSQL and Vector Backends."""

    total_checked: int = 0
    missing_repaired: int = 0
    orphans_purged: int = 0
    stale_versions_skipped: int = 0
    duration_ms: float = 0.0


class IndexReconciler:
    """Repairs vector backends (Qdrant/FAISS) using authoritative relational metadata."""

    def __init__(
        self,
        metadata_store: Any,
        embedding_service: Any,
    ) -> None:
        self.metadata_store = metadata_store
        self.embedding_service = embedding_service

    def reconcile(
        self,
        *,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 1000,
    ) -> ReconciliationReport:
        """Synchronize vector index from authoritative PostgreSQL metadata.

        Strictly enforces:
        1. PostgreSQL as authoritative source of truth.
        2. Version Guard: Rejects restoring stale versions when current version advanced.
        3. Never allows derived vector indexes to overwrite relational truth.
        """
        start_time = time.perf_counter()
        report = ReconciliationReport()

        # Step 1: Scan authoritative cache entries from relational store
        entries = []
        if hasattr(self.metadata_store, "all_entries"):
            for e in self.metadata_store.all_entries():
                if tenant_id and e.tenant_id != tenant_id:
                    continue
                if resource_id and e.resource_id != resource_id:
                    continue
                entries.append(e)
                if len(entries) >= limit:
                    break
        elif hasattr(self.metadata_store, "get_all_entries"):
            entries = self.metadata_store.get_all_entries(limit=limit)

        report.total_checked = len(entries)

        for entry in entries:
            if entry is None:
                continue

            # Version Guard: If resource-scoped, check against live resource version
            if entry.scope == CacheScope.RESOURCE and entry.resource_id is not None:
                current_res = self.metadata_store.get_resource(entry.resource_id)
                if (
                    current_res is not None
                    and entry.resource_version != current_res.current_version
                ):
                    report.stale_versions_skipped += 1
                    logger.debug(
                        "Reconciler skipping stale entry %s: entry version v%s != live v%s",
                        entry.id,
                        entry.resource_version,
                        current_res.current_version,
                    )
                    continue

            # Ensure entry exists in vector index
            try:
                emb_arr = np.asarray(entry.prompt_embedding, dtype=np.float32)
                self.embedding_service.add(
                    entry_id=entry.id,
                    embedding=emb_arr,
                    tenant_id=entry.tenant_id,
                    scope=entry.scope,
                    resource_id=entry.resource_id,
                    resource_version=entry.resource_version,
                    pipeline_version=entry.pipeline_version,
                )
                report.missing_repaired += 1
            except Exception as exc:
                logger.warning("Reconciler failed to re-index entry %s: %s", entry.id, exc)

        report.duration_ms = (time.perf_counter() - start_time) * 1000.0
        from semanticmemo.telemetry.metrics import SemanticMemoMetrics

        metrics = SemanticMemoMetrics.default()
        metrics.reconciliation_total.inc(report.total_checked, labels={"action": "checked"})
        metrics.reconciliation_total.inc(report.missing_repaired, labels={"action": "repaired"})
        metrics.reconciliation_total.inc(
            report.stale_versions_skipped, labels={"action": "stale_skipped"}
        )
        return report
