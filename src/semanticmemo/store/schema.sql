CREATE TABLE IF NOT EXISTS cache_entries (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    prompt_embedding BLOB NOT NULL,
    response TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL,
    last_hit_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0,
    feedback_negative_count INTEGER NOT NULL DEFAULT 0,
    feedback_positive_count INTEGER NOT NULL DEFAULT 0,
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
    cross_encoder_score REAL NOT NULL,
    label INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_active_learning_pairs_created_at
ON active_learning_pairs(created_at);

