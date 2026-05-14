-- Bytewax mini-stream proof: live dataflow outputs durable bounded hints.

CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_stream_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded','failed')),
    events_in integer NOT NULL DEFAULT 0,
    hints_out integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_hint (
    hint_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    phase text NOT NULL,
    status text NOT NULL,
    hint text NOT NULL,
    score integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
