CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);

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
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_id TEXT NOT NULL,
    permission_id TEXT NOT NULL,
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY(permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resources_tenant_id ON resources(tenant_id);

CREATE TABLE IF NOT EXISTS resource_access (
    resource_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    permission_id TEXT NOT NULL,
    PRIMARY KEY (resource_id, role_id, permission_id),
    FOREIGN KEY(resource_id) REFERENCES resources(id) ON DELETE CASCADE,
    FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY(permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cache_entries (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    prompt_embedding BLOB NOT NULL,
    response TEXT NOT NULL,
    model TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    scope TEXT NOT NULL DEFAULT 'global',
    resource_id TEXT,
    resource_version INTEGER,
    pipeline_version TEXT NOT NULL DEFAULT 'v2',
    expires_at TEXT,
    created_at TEXT NOT NULL,
    last_hit_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0,
    feedback_negative_count INTEGER NOT NULL DEFAULT 0,
    feedback_positive_count INTEGER NOT NULL DEFAULT 0,
    flight_generation INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cache_entries_last_hit_at
ON cache_entries(last_hit_at, created_at);

CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at
ON cache_entries(created_at);

CREATE TABLE IF NOT EXISTS lookup_records (
    query_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    query_prompt TEXT NOT NULL,
    query_embedding BLOB NOT NULL,
    cache_entry_id TEXT NOT NULL,
    similarity_score REAL,
    classifier_score REAL,
    cross_encoder_score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(cache_entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lookup_records_cache_entry_id
ON lookup_records(cache_entry_id);

CREATE INDEX IF NOT EXISTS idx_lookup_records_domain_prompt_created
ON lookup_records(domain, query_prompt, created_at);

CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    cache_entry_id TEXT NOT NULL,
    label INTEGER NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(query_id) REFERENCES lookup_records(query_id) ON DELETE CASCADE,
    FOREIGN KEY(cache_entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_query_id
ON feedback_events(query_id);

CREATE INDEX IF NOT EXISTS idx_feedback_events_created_at
ON feedback_events(created_at);

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
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_active_learning_pairs_created_at
ON active_learning_pairs(created_at);

CREATE INDEX IF NOT EXISTS idx_active_learning_pairs_scoped
ON active_learning_pairs(tenant_id, resource_id, pipeline_version);

CREATE TABLE IF NOT EXISTS outbox_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    correlation_id TEXT,
    causation_id TEXT,
    created_at TEXT NOT NULL,
    claimed_at TEXT,
    claim_owner TEXT,
    claim_expires_at TEXT,
    published_at TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 5,
    next_attempt_at TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_claimable
ON outbox_events(status, next_attempt_at, claim_expires_at, created_at);

CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT NOT NULL,
    consumer_name TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (event_id, consumer_name)
);

CREATE INDEX IF NOT EXISTS idx_processed_events_consumer
ON processed_events(consumer_name, processed_at);


