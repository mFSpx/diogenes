-- FILE: 06_SCHEMA/067_chrono_queue_event_bridge_automation.sql
-- PURPOSE: durable run records for continuous/idempotent ABSURD queue-event -> Chrono bridge automation.
-- COMPLIANCE: appends temporal evidence only; no existing temporal claims are overwritten or deleted.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.chrono_queue_event_bridge_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_mode text NOT NULL CHECK (run_mode IN ('dry_run','execute')),
  iterations_requested integer NOT NULL CHECK (iterations_requested > 0),
  iterations_completed integer NOT NULL DEFAULT 0 CHECK (iterations_completed >= 0),
  events_seen integer NOT NULL DEFAULT 0 CHECK (events_seen >= 0),
  claims_inserted integer NOT NULL DEFAULT 0 CHECK (claims_inserted >= 0),
  existing_claims_reused integer NOT NULL DEFAULT 0 CHECK (existing_claims_reused >= 0),
  receipts_inserted integer NOT NULL DEFAULT 0 CHECK (receipts_inserted >= 0),
  idle_iterations integer NOT NULL DEFAULT 0 CHECK (idle_iterations >= 0),
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  report_path text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_chrono_queue_event_bridge_run_started
  ON lucidota_control.chrono_queue_event_bridge_run(started_at DESC);

COMMIT;
