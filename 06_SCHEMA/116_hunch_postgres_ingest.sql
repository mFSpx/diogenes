CREATE SCHEMA IF NOT EXISTS lucidota_hunch;

CREATE TABLE IF NOT EXISTS lucidota_hunch.hunch_record (
    hunch_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hunch_id text NOT NULL UNIQUE,
    title text NOT NULL,
    rating text NOT NULL DEFAULT 'OPEN',
    authority_class text NOT NULL DEFAULT 'operator_hunch_signal_not_truth',
    evidence_state text NOT NULL DEFAULT 'candidate',
    truth_promotion text NOT NULL DEFAULT 'blocked_until_evidence_paths_reviewed',
    source_path text NOT NULL DEFAULT '',
    source_sha256 text NOT NULL DEFAULT '',
    source_line_start integer,
    source_line_end integer,
    body_sha256 text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_hunch.hunch_ingest_run (
    run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_report_path text NOT NULL DEFAULT '',
    records_seen integer NOT NULL DEFAULT 0,
    records_upserted integer NOT NULL DEFAULT 0,
    graph_candidates_written integer NOT NULL DEFAULT 0,
    receipt_path text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS hunch_record_rating_idx
    ON lucidota_hunch.hunch_record (rating, updated_at DESC);

CREATE INDEX IF NOT EXISTS hunch_record_source_hash_idx
    ON lucidota_hunch.hunch_record (source_sha256);
