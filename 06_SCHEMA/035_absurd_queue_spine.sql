-- FILE: 06_SCHEMA/035_absurd_queue_spine.sql
-- PURPOSE: LUCIDOTA ABSURD-compatible durable queue spine.
-- COMPLIANCE: Idempotent, non-destructive. No resident service migration and no graph mutation.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

DO $$
DECLARE
  legacy_prefix text := 'db' || 'os';
  legacy_type text := legacy_prefix || '_queue_job_status';
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
    WHERE n.nspname='lucidota_control' AND t.typname=legacy_type
  ) AND NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
    WHERE n.nspname='lucidota_control' AND t.typname='absurd_queue_job_status'
  ) THEN
    EXECUTE format('ALTER TYPE lucidota_control.%I RENAME TO absurd_queue_job_status', legacy_type);
  END IF;

  IF to_regclass('lucidota_control.' || legacy_prefix || '_queue') IS NOT NULL
     AND to_regclass('lucidota_control.absurd_queue') IS NULL THEN
    EXECUTE format('ALTER TABLE lucidota_control.%I RENAME TO absurd_queue', legacy_prefix || '_queue');
  END IF;
  IF to_regclass('lucidota_control.' || legacy_prefix || '_queue_job') IS NOT NULL
     AND to_regclass('lucidota_control.absurd_queue_job') IS NULL THEN
    EXECUTE format('ALTER TABLE lucidota_control.%I RENAME TO absurd_queue_job', legacy_prefix || '_queue_job');
  END IF;
  IF to_regclass('lucidota_control.' || legacy_prefix || '_queue_event') IS NOT NULL
     AND to_regclass('lucidota_control.absurd_queue_event') IS NULL THEN
    EXECUTE format('ALTER TABLE lucidota_control.%I RENAME TO absurd_queue_event', legacy_prefix || '_queue_event');
  END IF;
  IF to_regclass('lucidota_control.' || legacy_prefix || '_queue_dead_letter') IS NOT NULL
     AND to_regclass('lucidota_control.absurd_queue_dead_letter') IS NULL THEN
    EXECUTE format('ALTER TABLE lucidota_control.%I RENAME TO absurd_queue_dead_letter', legacy_prefix || '_queue_dead_letter');
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
    WHERE n.nspname='lucidota_control' AND t.typname='absurd_queue_job_status'
  ) THEN
    CREATE TYPE lucidota_control.absurd_queue_job_status AS ENUM (
      'queued','leased','running','succeeded','failed','dead_lettered','cancelled'
    );
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_queue (
  queue_name text PRIMARY KEY,
  owner_subsystem text NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','draining','archived')),
  max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
  visibility_timeout_seconds integer NOT NULL DEFAULT 300 CHECK (visibility_timeout_seconds > 0),
  notes text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_queue_job (
  job_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  queue_name text NOT NULL REFERENCES lucidota_control.absurd_queue(queue_name) ON DELETE RESTRICT,
  workflow_name text NOT NULL,
  job_kind text NOT NULL,
  idempotency_key text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  status lucidota_control.absurd_queue_job_status NOT NULL DEFAULT 'queued',
  priority integer NOT NULL DEFAULT 100,
  attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
  run_after timestamptz NOT NULL DEFAULT now(),
  leased_by text,
  lease_expires_at timestamptz,
  last_error text NOT NULL DEFAULT '',
  result jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS absurd_queue_job_idempotency_idx
  ON lucidota_control.absurd_queue_job(queue_name, idempotency_key);

CREATE INDEX IF NOT EXISTS absurd_queue_job_ready_idx
  ON lucidota_control.absurd_queue_job(queue_name, status, run_after, priority, created_at);

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_queue_event (
  queue_event_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_uuid uuid REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE CASCADE,
  queue_name text NOT NULL,
  event_kind text NOT NULL CHECK (event_kind IN ('enqueued','leased','started','succeeded','failed','dead_lettered','retry_scheduled','cancelled','audit')),
  event_source text NOT NULL DEFAULT 'absurd_queue_spine',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS absurd_queue_event_job_idx
  ON lucidota_control.absurd_queue_event(job_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_queue_dead_letter (
  dead_letter_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_uuid uuid NOT NULL REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE CASCADE,
  queue_name text NOT NULL,
  workflow_name text NOT NULL,
  job_kind text NOT NULL,
  idempotency_key text NOT NULL,
  error_kind text NOT NULL,
  error_message text NOT NULL,
  attempt_count integer NOT NULL,
  payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  context jsonb NOT NULL DEFAULT '{}'::jsonb,
  resolved boolean NOT NULL DEFAULT false,
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS absurd_queue_dead_letter_job_idx
  ON lucidota_control.absurd_queue_dead_letter(job_uuid) WHERE resolved = false;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES
  ('control', 'ABSURD workflow spine', 3, 'Safe control-plane tasks such as status checks and contract audits.'),
  ('chrono', 'Chrono-Ledger', 3, 'Future ABSURD wrapper for Chrono service/replay work. Existing service remains custom until migrated.'),
  ('korpus', 'KRAMPUSCHEWING/KORPUS', 3, 'Future ABSURD wrapper for intake/componentization work.'),
  ('graph_promotion', 'Graph promotion engine', 2, 'Future ABSURD worker for graph promotion packets; canonical graph materialization remains gated.'),
  ('surface_cep', 'Darwinian Surfaces/CEP', 3, 'Future ABSURD worker for command-envelope fan-in.')
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  max_attempts=EXCLUDED.max_attempts,
  notes=EXCLUDED.notes,
  updated_at=now();
