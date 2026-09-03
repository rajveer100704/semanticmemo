# SemanticMemo

**Authorization-aware, resource-versioned distributed semantic caching infrastructure for LLM agents.**

[![Tests](https://github.com/rajveer100704/semanticmemo/actions/workflows/ci.yml/badge.svg)](https://github.com/rajveer100704/semanticmemo/actions)
[![PyPI](https://img.shields.io/pypi/v/semanticmemo.svg?v=2)](https://pypi.org/project/semanticmemo/)
[![Python](https://img.shields.io/pypi/pyversions/semanticmemo.svg?v=2)](https://pypi.org/project/semanticmemo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Evolution: V1 to V2

### Why Cosine Caching Fails in Production
Semantic caching is essential for scaling LLM agent workflows and keeping API costs predictable. However, naive cosine caches make a critical mistake: they treat embedding proximity as equivalence.
```
"Should I approve the refund?"   ── cosine: 0.97 ──▶  CACHE HIT  ← catastrophic false positive
"Should I deny the refund?"      ─────────────────────────────────────────
```
Opposite-action prompts sit dangerously close in embedding space. In finance, healthcare, or enterprise agents, serving a cached answer across opposite prompts creates silent, catastrophic bugs.

**SemanticMemo v1.x** solved this semantic failure by replacing the cosine threshold with a 4-stage verification pipeline (FAISS retrieval + lightweight MLP equivalence classifier + Cross-Encoder re-ranker + 11-rule regex entity drift veto).

### The Enterprise Gap Solved by V2.0.0
Even with perfect semantic equivalence, caching in real-world multi-tenant agent architectures introduces deep distributed systems and security challenges:
1. **Cross-Tenant Data Leakage:** A prompt from Tenant A must never reuse cached responses generated for Tenant B.
2. **Resource Permission Invalidation:** If User X loses access to Document 42, a cached answer must never grant access. Cache hits cannot bypass current authorization.
3. **Stale Resource States:** When Document 42 updates from version 7 to version 8, cached inferences tied to version 7 are logically invalid.
4. **Distributed Stampedes & Split-Brain Writes:** Under high concurrency across multiple container instances, hundreds of identical requests must coalesce into exactly one LLM call without stale leaders overwriting newer cache states.

**SemanticMemo 2.0.0** elevates the framework from a single-process library into a full **authorization-aware, resource-versioned distributed caching plane**.

---

## Core Architecture & Execution Pipeline

In SemanticMemo 2.0.0, authorization and resource identity are evaluated **before** vector retrieval, and distributed single-flight guarantees strictly one LLM execution across all worker nodes.

```mermaid
flowchart TD
    Client["Caller: cache.get_or_call(...)"] --> Step1["① Authenticate Principal (user_id)"]
    Step1 --> Step2["② Derive Authoritative Tenant Context"]
    Step2 --> Step3["③ Evaluate Current RBAC & Permissions"]
    Step3 --> Step4["④ Resolve Fail-Closed RetrievalScope"]
    Step4 --> Step5["⑤ Scoped Vector Search (Qdrant / In-Memory)"]
    Step5 --> Step6{"⑥ Semantic Equivalence Pipeline\n(MLP + Cross-Encoder + Entity Veto)"}
    
    Step6 -->|"Valid Cache Hit"| Hit["✅ Return Authoritative Result"]
    Step6 -->|"Cache Miss"| Step7["⑦ Distributed Redis Single-Flight\n(Monotonic Fencing Generation)"]
    
    Step7 -->|"Leader Node"| CallLLM["Execute LLM Provider Callback"]
    Step7 -->|"Follower Nodes"| AwaitLeader["Await Completion / Poll Scoped Cache"]
    
    CallLLM --> CAS["Validate Fencing Token (CAS Guard)"]
    CAS --> StorePG["⑧ Commit to PostgreSQL & Invalidate Old Versions"]
    StorePG --> Outbox["⑨ Transactional Outbox Record"]
    Outbox --> Kafka["⑩ Kafka Event Relay (sm.events.resource_updated)"]
    Kafka --> Consumer["⑪ Idempotent Consumer & Qdrant Reconciliation"]
    StorePG --> Hit
    AwaitLeader --> Hit
```

### The 11-Stage Pipeline Explained:
1. **Authentication:** Caller identity (`user_id`) is strictly authenticated fail-closed.
2. **Authoritative Tenancy:** Tenant context is derived server-side from the authenticated user (`user.tenant_id`), completely preventing client tenant spoofing.
3. **Current RBAC:** Evaluates user role and permissions in PostgreSQL/SQLite prior to vector search.
4. **RetrievalScope:** Resolves the retrieval boundary into one of three distinct scopes:
   - **`GLOBAL`**: Shared, public data accessible system-wide.
   - **`TENANT`**: Strictly isolated to the caller's tenant (`scope=tenant AND tenant_id=current_tenant`).
   - **`RESOURCE`**: Bound to a specific `resource_id` and its current `resource_version`.
5. **Server-Side Filtered Retrieval:** Vector search pre-filters strictly within the resolved scope. Cross-tenant or stale-version vectors are excluded at the index layer.
6. **4-Stage Verification:** Evaluates candidate equivalence via MLP classifier, Cross-Encoder joint-attention pass, and regex entity drift veto.
7. **Distributed Single-Flight:** Redis distributed lock coordinates concurrent stampedes across multiple OS processes and nodes.
8. **Monotonic CAS Generation Fencing:** Protects cache writes against stalled leaders whose lock TTL expired. Stale generations are rejected by the database.
9. **Authoritative Relational Store:** PostgreSQL acts as the single source of truth for all entries, metadata, and versioning.
10. **Transactional Outbox & Kafka Relay:** State changes and outbox events are committed in a single atomic database transaction, then relayed durably to Kafka.
11. **Idempotent Invalidation & Reconciliation:** Background workers consume invalidation events deduplicated by an authoritative event ledger. The `IndexReconciler` automatically heals vector index drift against PostgreSQL.

---

## Installation & Packaging

SemanticMemo maintains an ultra-lightweight core with zero mandatory distributed drivers. Install only the capabilities required for your environment.

### Core Installation (Lightweight Local Dev)
```bash
pip install semanticmemo
```
*Dependencies: `numpy`, `pydantic`.*  
Operates fully with local SQLite, in-memory vector indexing, and local in-process single-flight.

### Production Integrations (Optional Extras)
```bash
# Neural ML Models (PyTorch, SentenceTransformers, FAISS)
pip install "semanticmemo[ml]"

# PostgreSQL authoritative storage & RBAC
pip install "semanticmemo[postgres]"

# Redis distributed single-flight & generation fencing
pip install "semanticmemo[redis]"

# Qdrant server-side scoped vector search
pip install "semanticmemo[qdrant]"

# Kafka durable outbox event relay
pip install "semanticmemo[kafka]"

# OpenTelemetry distributed tracing & Prometheus metrics
pip install "semanticmemo[telemetry]"

# Full Production Enterprise Suite (All integrations)
pip install "semanticmemo[production]"
```

---

## Quickstart

### 1. Breaking Change Notice (v2.0.0)

> [!IMPORTANT]
> **Authentication is now fail-closed.** In SemanticMemo 2.0.0, `get_or_call(...)` strictly requires an authenticated `user_id`. Unauthenticated invocations immediately raise `PermissionError`. For local development and test scripts, call `cache.bootstrap_default_user()` or pass `user_id="default_user"`.

### 2. Local Development (SQLite & In-Memory)

```python
import asyncio
from semanticmemo import SemanticMemo

async def call_llm(prompt: str) -> str:
    # Your LLM call (OpenAI, Anthropic, Ollama, etc.)
    return f"Response for: {prompt}"

async def main():
    # Initializes local SQLite cache store & in-memory single flight
    cache = SemanticMemo(domain="customer-support")
    
    # Bootstrap default principal for local dev
    cache.bootstrap_default_user(user_id="alice", tenant_id="dev_corp", role="analyst")

    result = await cache.get_or_call(
        prompt="How do I reset my password?",
        llm_function=call_llm,
        user_id="alice",
    )

    print(f"Result: {result.response}")
    print(f"Was Cache Hit: {result.was_cache_hit}")
    print(f"Latency: {result.latency_ms:.2f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Distributed Production Configuration

Deploy with PostgreSQL, Redis, Qdrant, and OpenTelemetry:

```python
from semanticmemo import (
    SemanticMemo,
    CacheConfig,
    PostgresConfig,
    RedisConfig,
    QdrantConfig,
)

cache = SemanticMemo(
    domain="finance",
    postgres_config=PostgresConfig(
        dsn="postgresql://app_user:secret@postgres.internal:5432/semanticmemo",
        pool_size=20,
    ),
    redis_config=RedisConfig(
        url="redis://redis.internal:6379/0",
        lock_ttl_ms=5000,
        wait_timeout_ms=5000,
    ),
    qdrant_config=QdrantConfig(
        url="http://qdrant.internal:6333",
        collection_name="finance_cache_v2",
    ),
)

# Register a multi-tenant versioned resource
cache.register_resource(
    resource_id="quarterly_report_q3",
    tenant_id="hedge_fund_a",
    initial_version=1,
)

# Subsequent queries are scoped to hedge_fund_a and quarterly_report_q3 v1
result = await cache.get_or_call(
    prompt="Summarize the operating margin from the Q3 report",
    llm_function=call_llm,
    user_id="trader_bob",
    resource_id="quarterly_report_q3",
)
```

---

## Scopes & Multi-Tenant Versioning

SemanticMemo 2.0.0 enforces three distinct retrieval scopes:

| Scope | Visibility | Filter Condition | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`GLOBAL`** | All Tenants | `scope = 'global'` | Public knowledge bases, general product FAQs, static system prompts |
| **`TENANT`** | Single Tenant | `scope = 'tenant' AND tenant_id = current_tenant` | Customer support tickets, private organization data |
| **`RESOURCE`** | Single Resource & Version | `resource_id = id AND resource_version = N` | Deal rooms, sensitive contracts, uploaded documents |

### Dynamic Cache Identity
Cache entries in SemanticMemo 2.0.0 are uniquely keyed across six dimensions:
```
flight_key = sm:lock:{pipeline_version}:{scope}:{tenant_id}:{resource_id}:{resource_version}:{prompt_sha256}
```
When a document increments from `resource_version = 1` to `resource_version = 2`, older cache entries are immediately bypassed without requiring synchronous vector deletions. PostgreSQL remains the correctness authority.

---

## Observability & Security Sanitization

SemanticMemo 2.0.0 ships with built-in OpenTelemetry tracing and Prometheus metrics designed for strict enterprise security compliance:
- **Zero PII in Attributes:** Trace attributes and metric labels automatically hash or redact sensitive fields (API keys, raw prompts, passwords, user identifiers).
- **OpenMetrics Export:** `cache.metrics.export_openmetrics_text()` provides Prometheus-compatible gauges, counters, and latency histograms.
- **Explainable Decision Traces:** Every result includes stage-by-stage latencies (`embedding_latency_ms`, `retrieval_latency_ms`, `mlp_latency_ms`, `cross_encoder_latency_ms`).

---

## Migration from SQLite to PostgreSQL

For teams transitioning from SemanticMemo v1.x SQLite stores to PostgreSQL, SemanticMemo 2.0.0 provides a migration utility with deterministic SHA-256 row checksum validation:

```python
from semanticmemo.store.migration import migrate_sqlite_to_postgres
from semanticmemo.store import SQLiteCacheStore, PostgresCacheStore, PostgresConfig

sqlite_store = SQLiteCacheStore(".semanticmemo/cache.db")
postgres_store = PostgresCacheStore(PostgresConfig(dsn="postgresql://..."))

report = migrate_sqlite_to_postgres(sqlite_store, postgres_store)
print(f"Migration Complete: {report.is_valid}")
print(f"Total Rows Migrated: {report.total_migrated}")
```

---

## Release Validation & Certification

SemanticMemo 2.0.0 distinguishes between standard GitHub Actions CI and strict live infrastructure certification:

- **GitHub Actions CI:** Runs on Ubuntu across Python 3.11–3.14, validating the complete repository regression suite (243 tests, 0 failures), Ruff linting/formatting, and Pyright strict type checking. External live tests skip cleanly when Docker containers are absent.
- **Certified Live Environment:** Validated against real live PostgreSQL 16, Redis 7, Qdrant 1.13.4, and Kafka 3.7.0 instances on Windows using 3 distinct OS processes:
  - **9/9 Core Live Certification Phases:** PASSED
  - **Lightweight Worker Import Safety Test:** PASSED (Asserts zero loading of `torch`, `scipy`, `sklearn`, `sentence_transformers`)
  - **Live Skips / Failures:** 0 skipped, 0 failed.
  - **Distributed Single-Flight Coalescing:** Verified (100 requests $\to$ exactly 1 LLM execution across 3 OS processes).

---

## License

MIT — see [LICENSE](LICENSE).
