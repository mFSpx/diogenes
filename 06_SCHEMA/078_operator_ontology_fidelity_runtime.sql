-- FILE: 06_SCHEMA/078_operator_ontology_fidelity_runtime.sql
-- PURPOSE: runtime ledger for Operator ontology fidelity checks on extraction outputs.
-- COMPLIANCE: preserves Operator-authored labels; detects softening/renaming before graph/workflow promotion.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.ontology_fidelity_runtime_check (
  check_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  input_path text NOT NULL,
  input_sha256 text NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
  status text NOT NULL CHECK (status IN ('PASS','FAIL')),
  required_labels_seen jsonb NOT NULL DEFAULT '[]'::jsonb,
  required_labels_missing jsonb NOT NULL DEFAULT '[]'::jsonb,
  forbidden_softening_hits jsonb NOT NULL DEFAULT '[]'::jsonb,
  extraction_output_fidelity_guard text NOT NULL DEFAULT 'runtime_guard_v1',
  graph_promotion_allowed boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ontology_fidelity_runtime_status
  ON lucidota_archaeology.ontology_fidelity_runtime_check(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ontology_fidelity_runtime_sha
  ON lucidota_archaeology.ontology_fidelity_runtime_check(input_sha256);

COMMIT;
