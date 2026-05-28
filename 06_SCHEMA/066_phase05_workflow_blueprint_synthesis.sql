-- FILE: 06_SCHEMA/066_phase05_workflow_blueprint_synthesis.sql
-- PURPOSE: executable audit/run ledger for design_atom -> workflow_blueprint synthesis.
-- COMPLIANCE: writes only Phase 0.5 blueprint/provenance rows; no corpus ingest, no graph mutation.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.workflow_blueprint_synthesis_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  synthesizer_version text NOT NULL,
  source_atom_count integer NOT NULL CHECK (source_atom_count >= 0),
  blueprints_prepared integer NOT NULL CHECK (blueprints_prepared >= 0),
  blueprints_inserted integer NOT NULL DEFAULT 0 CHECK (blueprints_inserted >= 0),
  blueprints_updated integer NOT NULL DEFAULT 0 CHECK (blueprints_updated >= 0),
  dry_run boolean NOT NULL DEFAULT true,
  report_path text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_workflow_blueprint_synthesis_run_created
  ON lucidota_archaeology.workflow_blueprint_synthesis_run(created_at DESC);

COMMIT;
