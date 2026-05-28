-- LUCIDOTA ETL Pipeline ABSURD substrate.
-- Critical front-door control plane for the future Rust ETL engine.
-- This is side-by-side with existing KORPUS tables; it does not replace the
-- proven Python ingest path until the ETL binary is validated offline.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_etl;

CREATE OR REPLACE FUNCTION lucidota_etl.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_etl.pipeline_run (
    run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_label text NOT NULL DEFAULT '',
    engine text NOT NULL DEFAULT 'lucidota_etl',
    engine_version text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    root_paths jsonb NOT NULL DEFAULT '[]'::jsonb,
    options jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_count bigint NOT NULL DEFAULT 0,
    object_count bigint NOT NULL DEFAULT 0,
    component_count bigint NOT NULL DEFAULT 0,
    queued_task_count bigint NOT NULL DEFAULT 0,
    error_count bigint NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS etl_pipeline_run_status_idx
    ON lucidota_etl.pipeline_run(status, started_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_etl.object_trace (
    object_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_uuid uuid REFERENCES lucidota_etl.pipeline_run(run_uuid) ON DELETE SET NULL,
    parent_object_uuid uuid REFERENCES lucidota_etl.object_trace(object_uuid) ON DELETE SET NULL,
    parent_sha256 text NOT NULL DEFAULT '',
    sha256 text NOT NULL CHECK (sha256 = '' OR sha256 ~ '^[0-9a-f]{64}$'),
    blake3 text NOT NULL DEFAULT '',
    size_bytes bigint NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    mime text NOT NULL DEFAULT '',
    file_kind text NOT NULL DEFAULT '',
    lane text NOT NULL DEFAULT '',
    source_path text NOT NULL DEFAULT '',
    virtual_path text NOT NULL DEFAULT '',
    root_path text NOT NULL DEFAULT '',
    cas_uri text NOT NULL DEFAULT '',
    locked_relative_path text NOT NULL DEFAULT '',
    event_timestamp timestamptz,
    geo_lat double precision,
    geo_lon double precision,
    status text NOT NULL DEFAULT 'indexed' CHECK (status IN ('indexed','deferred','error','skipped')),
    error text NOT NULL DEFAULT '',
    structural_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(run_uuid, virtual_path)
);

CREATE INDEX IF NOT EXISTS etl_object_trace_sha_idx
    ON lucidota_etl.object_trace(sha256);

CREATE INDEX IF NOT EXISTS etl_object_trace_parent_idx
    ON lucidota_etl.object_trace(parent_sha256, virtual_path);

CREATE INDEX IF NOT EXISTS etl_object_trace_status_idx
    ON lucidota_etl.object_trace(status, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_etl.step_checkpoint (
    checkpoint_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_uuid uuid REFERENCES lucidota_etl.pipeline_run(run_uuid) ON DELETE SET NULL,
    object_uuid uuid REFERENCES lucidota_etl.object_trace(object_uuid) ON DELETE CASCADE,
    queue_job_uuid uuid,
    workflow_id text NOT NULL,
    step_name text NOT NULL,
    step_index integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('queued','running','succeeded','failed','skipped')),
    attempt integer NOT NULL DEFAULT 1 CHECK (attempt > 0),
    input_ref jsonb NOT NULL DEFAULT '{}'::jsonb,
    output_ref jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    error text NOT NULL DEFAULT '',
    started_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE(run_uuid, object_uuid, workflow_id, step_name, attempt)
);

CREATE INDEX IF NOT EXISTS etl_step_checkpoint_lookup_idx
    ON lucidota_etl.step_checkpoint(workflow_id, step_name, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS etl_step_checkpoint_object_idx
    ON lucidota_etl.step_checkpoint(object_uuid, workflow_id, step_index);

CREATE TABLE IF NOT EXISTS lucidota_etl.dead_letter (
    dead_letter_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_uuid uuid REFERENCES lucidota_etl.pipeline_run(run_uuid) ON DELETE SET NULL,
    object_uuid uuid REFERENCES lucidota_etl.object_trace(object_uuid) ON DELETE SET NULL,
    queue_job_uuid uuid,
    failure_kind text NOT NULL DEFAULT '',
    message text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS etl_dead_letter_kind_idx
    ON lucidota_etl.dead_letter(failure_kind, created_at DESC);

DROP TRIGGER IF EXISTS touch_etl_pipeline_run_updated_at ON lucidota_etl.pipeline_run;
CREATE TRIGGER touch_etl_pipeline_run_updated_at
BEFORE UPDATE ON lucidota_etl.pipeline_run
FOR EACH ROW EXECUTE FUNCTION lucidota_etl.touch_updated_at();

DROP TRIGGER IF EXISTS touch_etl_object_trace_updated_at ON lucidota_etl.object_trace;
CREATE TRIGGER touch_etl_object_trace_updated_at
BEFORE UPDATE ON lucidota_etl.object_trace
FOR EACH ROW EXECUTE FUNCTION lucidota_etl.touch_updated_at();

DROP TRIGGER IF EXISTS touch_etl_step_checkpoint_updated_at ON lucidota_etl.step_checkpoint;
CREATE TRIGGER touch_etl_step_checkpoint_updated_at
BEFORE UPDATE ON lucidota_etl.step_checkpoint
FOR EACH ROW EXECUTE FUNCTION lucidota_etl.touch_updated_at();

CREATE OR REPLACE VIEW lucidota_etl.pipeline_dashboard AS
SELECT
    r.run_uuid,
    r.run_label,
    r.engine,
    r.status,
    r.observed_count,
    r.object_count,
    r.component_count,
    r.queued_task_count,
    r.error_count,
    r.started_at,
    r.updated_at,
    r.finished_at,
    now() - r.started_at AS age
FROM lucidota_etl.pipeline_run r
ORDER BY r.started_at DESC;
