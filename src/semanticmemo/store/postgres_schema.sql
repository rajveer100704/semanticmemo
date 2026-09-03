-- PostgreSQL Schema for SemanticMemo V2 Shared Metadata & RBAC Plane

-- 1. Users & Tenant Identity
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pg_users_tenant_id ON users(tenant_id);

-- 2. RBAC Roles & Permissions
CREATE TABLE IF NOT EXISTS roles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS permissions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    PRIMARY KEY (user_id, role_id),
    CONSTRAINT fk_ur_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_ur_role FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_id TEXT NOT NULL,
    permission_id TEXT NOT NULL,
    PRIMARY KEY (role_id, permission_id),
    CONSTRAINT fk_rp_role FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    CONSTRAINT fk_rp_perm FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

-- 3. Resources & Version Management
CREATE TABLE IF NOT EXISTS resources (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id),
    CONSTRAINT uq_resources_tenant_id UNIQUE (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_pg_resources_tenant ON resources(tenant_id, id);

-- 4. Resource Access ACLs
CREATE TABLE IF NOT EXISTS resource_access (
    resource_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    permission_id TEXT NOT NULL,
    PRIMARY KEY (resource_id, role_id, permission_id),
    CONSTRAINT fk_ra_resource FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE,
    CONSTRAINT fk_ra_role FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    CONSTRAINT fk_ra_perm FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

-- 5. Cache Entries (Authoritative Relational Truth & Metadata Persistence)
CREATE TABLE IF NOT EXISTS cache_entries (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    prompt_embedding BYTEA NOT NULL,
    response TEXT NOT NULL,
    model TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    scope TEXT NOT NULL DEFAULT 'global',
    resource_id TEXT,
    resource_version INTEGER,
    pipeline_version TEXT NOT NULL DEFAULT 'v2',
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_hit_at TIMESTAMPTZ,
    hit_count INTEGER NOT NULL DEFAULT 0,
    feedback_negative_count INTEGER NOT NULL DEFAULT 0,
    feedback_positive_count INTEGER NOT NULL DEFAULT 0,
    flight_generation INTEGER NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pg_cache_scoped
ON cache_entries(tenant_id, scope, resource_id, resource_version);

CREATE INDEX IF NOT EXISTS idx_pg_cache_prompt_lookup
ON cache_entries(prompt, tenant_id, scope, resource_id, resource_version, pipeline_version);

CREATE INDEX IF NOT EXISTS idx_pg_cache_created_at
ON cache_entries(created_at);

CREATE INDEX IF NOT EXISTS idx_pg_cache_last_hit_at
ON cache_entries(last_hit_at, created_at);

-- 6. Lookup Records
CREATE TABLE IF NOT EXISTS lookup_records (
    query_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    query_prompt TEXT NOT NULL,
    query_embedding BYTEA NOT NULL,
    cache_entry_id TEXT NOT NULL,
    similarity_score REAL,
    classifier_score REAL,
    cross_encoder_score REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_lr_entry FOREIGN KEY (cache_entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pg_lookup_cache_entry_id
ON lookup_records(cache_entry_id);

CREATE INDEX IF NOT EXISTS idx_pg_lookup_domain_prompt_created
ON lookup_records(domain, query_prompt, created_at);

-- 7. Feedback Events
CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    cache_entry_id TEXT NOT NULL,
    label INTEGER NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT fk_fb_query FOREIGN KEY (query_id) REFERENCES lookup_records(query_id) ON DELETE CASCADE,
    CONSTRAINT fk_fb_entry FOREIGN KEY (cache_entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pg_feedback_query_id
ON feedback_events(query_id);

-- 8. Active Learning Provenance Ledger
CREATE TABLE IF NOT EXISTS active_learning_pairs (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    query_prompt TEXT NOT NULL,
    cached_prompt TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    classifier_score REAL NOT NULL,
    cross_encoder_score REAL,
    label INTEGER NOT NULL,
    source TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    resource_id TEXT,
    pipeline_version TEXT NOT NULL DEFAULT 'v2',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pg_active_learning_scoped
ON active_learning_pairs(tenant_id, resource_id, pipeline_version);

CREATE INDEX IF NOT EXISTS idx_pg_active_learning_created_at
ON active_learning_pairs(created_at);

-- 9. Transactional Outbox Events (At-Least-Once Distributed Invalidation Plane)
CREATE TABLE IF NOT EXISTS outbox_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending',
    correlation_id TEXT,
    causation_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    claim_owner TEXT,
    claim_expires_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 5,
    next_attempt_at TIMESTAMPTZ,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_pg_outbox_claimable
ON outbox_events(status, next_attempt_at, claim_expires_at, created_at);

-- 10. Consumer Idempotency Store (Durable Event Deduplication)
CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT NOT NULL,
    consumer_name TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_id, consumer_name)
);

CREATE INDEX IF NOT EXISTS idx_pg_processed_events_consumer
ON processed_events(consumer_name, processed_at);

