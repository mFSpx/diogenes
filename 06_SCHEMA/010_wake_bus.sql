-- Local-first Wake Bus.
-- Postgres remains truth. Wake bus marks small ID/ref rows ready for low-latency local readers.

CREATE SCHEMA IF NOT EXISTS lucidota_bus;

CREATE TABLE IF NOT EXISTS lucidota_bus.wake_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    mode text NOT NULL DEFAULT 'local_postgres_outbox',
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    scanned integer NOT NULL DEFAULT 0,
    delivered integer NOT NULL DEFAULT 0,
    failed integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
