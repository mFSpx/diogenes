-- LUCIDOTA learning/reflex schema.
-- ABSURD/Postgres are durable truth; River updates lightweight online hints from committed events.

CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE TABLE IF NOT EXISTS lucidota_learning.river_event_cursor (
    cursor_name text PRIMARY KEY,
    last_event_at timestamptz NOT NULL DEFAULT 'epoch'::timestamptz,
    last_event_id uuid,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.river_score (
    score_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    phase text NOT NULL,
    decision text NOT NULL DEFAULT '',
    examples integer NOT NULL DEFAULT 0,
    successes integer NOT NULL DEFAULT 0,
    failures integer NOT NULL DEFAULT 0,
    success_rate double precision NOT NULL DEFAULT 0,
    river_prediction double precision,
    model_kind text NOT NULL DEFAULT 'river_logistic_regression_online',
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, phase, decision)
);

CREATE TABLE IF NOT EXISTS lucidota_learning.river_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    events_seen integer NOT NULL DEFAULT 0,
    examples_trained integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
