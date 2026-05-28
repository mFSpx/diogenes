-- FILE: 06_SCHEMA/062_work_order_loader_validator.sql
-- PURPOSE: strict work-order loading ledger.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.work_order_load_event (
  load_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_ref text NOT NULL,
  payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  valid boolean NOT NULL,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  job_uuid uuid,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_work_order_load_event_valid
  ON lucidota_control.work_order_load_event(valid, created_at DESC);

COMMIT;
