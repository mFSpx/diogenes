-- Queue-backed Hop Pivot v1.

CREATE SCHEMA IF NOT EXISTS lucidota_pivot;

CREATE TABLE IF NOT EXISTS lucidota_pivot.hop_job (
    job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    root_target text NOT NULL,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed')),
    max_depth integer NOT NULL DEFAULT 1,
    max_pivots integer NOT NULL DEFAULT 8,
    promote_threshold integer NOT NULL DEFAULT 35,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_pivot.hop_node (
    node_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES lucidota_pivot.hop_job(job_id) ON DELETE CASCADE,
    target text NOT NULL,
    depth integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','skipped')),
    decision text NOT NULL DEFAULT '',
    score integer NOT NULL DEFAULT 0,
    sha256 text,
    parent_target text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE (job_id, target)
);

CREATE INDEX IF NOT EXISTS hop_node_job_depth_idx ON lucidota_pivot.hop_node (job_id, depth, score DESC);

CREATE TABLE IF NOT EXISTS lucidota_pivot.promotion (
    promotion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES lucidota_pivot.hop_job(job_id) ON DELETE CASCADE,
    source_target text NOT NULL,
    candidate text NOT NULL,
    candidate_kind text NOT NULL,
    score integer NOT NULL,
    reason text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS promotion_job_score_idx ON lucidota_pivot.promotion (job_id, score DESC, created_at DESC);

ALTER TABLE lucidota_pivot.hop_node
  ADD COLUMN IF NOT EXISTS pheromone double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS utility double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS maintenance_cost double precision NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS lucidota_pivot.edge_signal (
    signal_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES lucidota_pivot.hop_job(job_id) ON DELETE CASCADE,
    source_target text NOT NULL,
    candidate text NOT NULL,
    pheromone double precision NOT NULL DEFAULT 0,
    utility double precision NOT NULL DEFAULT 0,
    maintenance_cost double precision NOT NULL DEFAULT 0,
    probability double precision NOT NULL DEFAULT 0,
    selected boolean NOT NULL DEFAULT false,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS edge_signal_job_prob_idx
    ON lucidota_pivot.edge_signal (job_id, probability DESC, created_at DESC);
