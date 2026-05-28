-- KORPUS derived compute queue.
-- Boring Postgres lease queue for post-ingest enrichment: River replay,
-- near-duplicate scans, graph promotion, deferred parsing, and deep media.
-- Raw evidence ingest must never depend on this queue draining.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE OR REPLACE FUNCTION lucidota_korpus.touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS lucidota_korpus.derived_compute_queue (
    job_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type text NOT NULL CHECK (task_type IN (
        'river_replay_component',
        'near_duplicate_scan',
        'graph_promote_file',
        'graph_promote_component',
        'deferred_parse',
        'media_deep_extract'
    )),
    target_table text NOT NULL CHECK (target_table IN (
        'file_object',
        'component',
        'entity',
        'concept'
    )),
    target_uuid uuid NOT NULL,
    priority integer NOT NULL DEFAULT 100,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','dead')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    locked_by text NOT NULL DEFAULT '',
    locked_until timestamptz,
    last_error text NOT NULL DEFAULT '',
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    run_after timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE(task_type, target_table, target_uuid)
);

-- Do not put now() in the partial predicate: Postgres requires immutable predicates.
-- run_after stays in the index key, and the claim query applies run_after <= now().
CREATE INDEX IF NOT EXISTS derived_compute_queue_claim_idx
    ON lucidota_korpus.derived_compute_queue(run_after, priority DESC, created_at ASC)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS derived_compute_queue_lease_idx
    ON lucidota_korpus.derived_compute_queue(locked_until, priority DESC, created_at ASC)
    WHERE status = 'running';

CREATE INDEX IF NOT EXISTS derived_compute_queue_status_idx
    ON lucidota_korpus.derived_compute_queue(task_type, status, created_at);

CREATE INDEX IF NOT EXISTS derived_compute_queue_target_idx
    ON lucidota_korpus.derived_compute_queue(target_table, target_uuid, task_type);

DROP TRIGGER IF EXISTS touch_derived_compute_queue_updated_at ON lucidota_korpus.derived_compute_queue;
CREATE TRIGGER touch_derived_compute_queue_updated_at
BEFORE UPDATE ON lucidota_korpus.derived_compute_queue
FOR EACH ROW EXECUTE FUNCTION lucidota_korpus.touch_updated_at();

CREATE OR REPLACE VIEW lucidota_korpus.derived_compute_queue_summary AS
SELECT
    task_type,
    status,
    count(*) AS job_count,
    min(created_at) AS oldest_job,
    max(updated_at) AS newest_update
FROM lucidota_korpus.derived_compute_queue
GROUP BY task_type, status;
