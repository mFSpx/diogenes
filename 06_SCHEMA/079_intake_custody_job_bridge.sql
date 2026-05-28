-- FILE: 06_SCHEMA/079_intake_custody_job_bridge.sql
-- PURPOSE: bridge Rust lucidota-intake custody metadata/derived tasks into ABSURD queue jobs.
-- COMPLIANCE: no direct graph mutation; intake metadata remains custody evidence, downstream work is queued.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES ('intake_custody', 'Rust intake custody bridge', 3, 'Rust intake UnifiedMetadata derived_tasks fan out into ABSURD work orders; graph promotion remains gated.')
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  max_attempts=EXCLUDED.max_attempts,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.intake_custody_bridge_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_jsonl_path text NOT NULL,
  source_record_sha256 text NOT NULL CHECK (source_record_sha256 ~ '^[0-9a-f]{64}$'),
  file_hash_sha256 text NOT NULL DEFAULT '',
  source_path text NOT NULL DEFAULT '',
  task_type text NOT NULL,
  target_table text NOT NULL,
  queue_name text NOT NULL,
  job_uuid uuid REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE SET NULL,
  idempotency_key text NOT NULL,
  inserted_new boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(queue_name, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_intake_custody_bridge_source_hash
  ON lucidota_control.intake_custody_bridge_receipt(file_hash_sha256);

COMMIT;
