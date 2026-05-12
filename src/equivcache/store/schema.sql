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
