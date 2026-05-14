-- LUCIDOTA local CAS manifest/index.
-- Artifact bytes live in local vault; Postgres stores inspectable metadata.

CREATE SCHEMA IF NOT EXISTS lucidota_vault;

CREATE TABLE IF NOT EXISTS lucidota_vault.cas_manifest (
    sha256 text PRIMARY KEY,
    cas_uri text UNIQUE NOT NULL,
    relative_path text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    mime text NOT NULL DEFAULT '',
    source text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cas_manifest_source_idx
    ON lucidota_vault.cas_manifest (source, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_vault.cas_integrity_check (
    check_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    checked_at timestamptz NOT NULL DEFAULT now(),
    total_files integer NOT NULL DEFAULT 0,
    indexed_files integer NOT NULL DEFAULT 0,
    missing_files integer NOT NULL DEFAULT 0,
    corrupt_files integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_vault.cas_gc_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    mode text NOT NULL CHECK (mode IN ('report', 'quarantine')),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    total_files integer NOT NULL DEFAULT 0,
    referenced_files integer NOT NULL DEFAULT 0,
    orphan_candidates integer NOT NULL DEFAULT 0,
    quarantined_files integer NOT NULL DEFAULT 0,
    corrupt_files integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_vault.cas_gc_candidate (
    candidate_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES lucidota_vault.cas_gc_run(run_id) ON DELETE CASCADE,
    sha256 text NOT NULL,
    original_relative_path text NOT NULL,
    size_bytes bigint NOT NULL DEFAULT 0,
    status text NOT NULL CHECK (status IN ('referenced', 'orphan_candidate', 'quarantined', 'corrupt')),
    quarantine_relative_path text NOT NULL DEFAULT '',
    reason text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cas_gc_candidate_run_status_idx
    ON lucidota_vault.cas_gc_candidate (run_id, status);

CREATE TABLE IF NOT EXISTS lucidota_vault.cas_ingest_journal (
    journal_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256 text NOT NULL,
    cas_uri text NOT NULL,
    relative_path text NOT NULL,
    size_bytes bigint NOT NULL DEFAULT 0,
    source text NOT NULL DEFAULT '',
    stage text NOT NULL CHECK (stage IN ('written', 'db_committed', 'recovered', 'quarantined')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cas_ingest_journal_sha_stage_idx
    ON lucidota_vault.cas_ingest_journal (sha256, stage, created_at DESC);
