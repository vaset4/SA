"""个人上传受理、恢复和发布的持久结构，作者：xxx。"""

UPLOAD_SCHEMA_SQL = """
ALTER TABLE papers ADD COLUMN IF NOT EXISTS index_batch_id TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS metadata_status TEXT NOT NULL DEFAULT 'not_requested';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS metadata_error TEXT NOT NULL DEFAULT '';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS metadata_missing JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS upload_documents (
    paper_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    library_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_ref JSONB NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_size BIGINT NOT NULL CHECK (source_size > 0),
    parse_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    resume_sequence INTEGER NOT NULL DEFAULT 0,
    retry_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_task_id TEXT,
    metadata_status TEXT NOT NULL DEFAULT 'pending',
    metadata_error TEXT NOT NULL DEFAULT '',
    metadata_failure JSONB,
    graph_task_id TEXT,
    graph_status TEXT NOT NULL DEFAULT 'not_requested',
    graph_error TEXT NOT NULL DEFAULT '',
    graph_failure JSONB,
    graph_requested BOOLEAN NOT NULL DEFAULT FALSE,
    last_failure JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT upload_documents_state_check CHECK (state IN ('queued','processing','paused','ready','cancelled','deleted'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_upload_active_source
    ON upload_documents(user_id, library_id, source_sha256) WHERE state <> 'deleted';
ALTER TABLE upload_documents ADD COLUMN IF NOT EXISTS metadata_failure JSONB;
ALTER TABLE upload_documents ADD COLUMN IF NOT EXISTS graph_failure JSONB;

CREATE TABLE IF NOT EXISTS upload_requests (
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES upload_documents(paper_id),
    task_id TEXT NOT NULL REFERENCES task_runs(task_id),
    project_id TEXT,
    request_kind TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, request_id)
);
CREATE INDEX IF NOT EXISTS idx_upload_requests_paper ON upload_requests(paper_id);

CREATE TABLE IF NOT EXISTS upload_attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_runs(task_id),
    worker_id TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES upload_documents(paper_id),
    user_id TEXT NOT NULL,
    parse_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    previous_batch_id TEXT,
    operation TEXT NOT NULL,
    resume_sequence INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'receive',
    state TEXT NOT NULL DEFAULT 'running',
    errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    UNIQUE(task_id, worker_id)
);

CREATE TABLE IF NOT EXISTS upload_checkpoints (
    parse_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES upload_documents(paper_id),
    user_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (parse_id, stage),
    CONSTRAINT upload_checkpoint_state_check CHECK (state IN ('pending','completed','rejected'))
);

CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES upload_documents(paper_id),
    user_id TEXT NOT NULL,
    task_id TEXT NOT NULL REFERENCES task_runs(task_id),
    worker_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'building',
    manifest_ref JSONB,
    embedding_fingerprint TEXT,
    payload_sha256 TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    collection_name TEXT,
    previous_batch_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    cleanup_error TEXT NOT NULL DEFAULT '',
    cleanup_completed_at TIMESTAMPTZ,
    CONSTRAINT upload_batch_state_check CHECK (state IN ('building','verified','published','superseded','retired','abandoned'))
);
CREATE INDEX IF NOT EXISTS idx_upload_batches_paper ON upload_batches(paper_id);
ALTER TABLE upload_batches ADD COLUMN IF NOT EXISTS cleanup_completed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS upload_read_leases (
    lease_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    batch_ids TEXT[] NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_upload_read_lease_expiry ON upload_read_leases(expires_at);
"""
