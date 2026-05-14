-- Treelite routing hints: compiled/lightweight model outputs stay advisory.
CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE TABLE IF NOT EXISTS lucidota_learning.treelite_router_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    artifact_uri text NOT NULL DEFAULT '',
    examples integer NOT NULL DEFAULT 0,
    route text NOT NULL DEFAULT '',
    score double precision NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
