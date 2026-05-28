-- FILE: 06_SCHEMA/115_project2501_script_audit_worker.sql
-- PURPOSE: Project 2501 audit-lane script survival classifier receipts.
-- COMPLIANCE: writes manifest/corpse evidence only; no deletion, no canonical graph writes, no Big Board feature mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.script_audit_run (
  script_audit_run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text REFERENCES lucidota_control.event_envelope(event_id) ON DELETE SET NULL,
  work_receipt_uuid uuid REFERENCES lucidota_control.work_receipt(work_receipt_uuid) ON DELETE SET NULL,
  script_manifest_uuid uuid REFERENCES lucidota_control.script_manifest(script_manifest_uuid) ON DELETE SET NULL,
  corpse_uuid uuid REFERENCES lucidota_control.corpse_manifest(corpse_uuid) ON DELETE SET NULL,
  script_path text NOT NULL,
  verdict text NOT NULL CHECK (verdict IN ('KEEP','MERGE','WRAP','REWRITE','QUARANTINE','CORPSE_MANIFEST','KRAMPUSCHEW')),
  survival_score integer NOT NULL CHECK (survival_score BETWEEN 0 AND 100),
  slop_score integer NOT NULL CHECK (slop_score BETWEEN 0 AND 100),
  bounded_step text NOT NULL DEFAULT 'classify_script_survival',
  source_receipt text NOT NULL DEFAULT '',
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  operator_feature_authority_required boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_script_audit_run_created
  ON lucidota_control.script_audit_run(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_script_audit_run_path_created
  ON lucidota_control.script_audit_run(script_path, created_at DESC);

COMMENT ON TABLE lucidota_control.script_audit_run IS
  'Project 2501 audit-lane script survival classifications. Corpse rows preserve evidence; workers never delete source scripts.';

COMMIT;
