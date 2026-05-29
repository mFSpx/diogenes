-- FILE: 06_SCHEMA/056_audit_verdict_runtime_enforcement.sql
-- PURPOSE: strict audit verdict validation runtime ledger.
-- COMPLIANCE: PASS requires evidence; FAIL/PARTIAL_FAIL require remediation.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.audit_verdict_validation_event (
  validation_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  verdict_uuid uuid,
  task_id text NOT NULL,
  verdict text NOT NULL CHECK (verdict IN ('PASS','FAIL','PARTIAL_FAIL')),
  valid boolean NOT NULL,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  remediation text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_verdict_validation_task
  ON lucidota_control.audit_verdict_validation_event(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_verdict_validation_valid
  ON lucidota_control.audit_verdict_validation_event(valid, created_at DESC);

COMMIT;
