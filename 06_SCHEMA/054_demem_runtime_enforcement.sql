-- FILE: 06_SCHEMA/054_demem_runtime_enforcement.sql
-- PURPOSE: executable DeMem boundary enforcement ledger for instructions/commands.
-- COMPLIANCE: boundary checks can block/warn/rewrite; they do not grant mutation authority.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.demem_enforcement_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  instruction_sha256 text NOT NULL CHECK (instruction_sha256 ~ '^[0-9a-f]{64}$'),
  source_ref text NOT NULL DEFAULT '',
  boundary_hits jsonb NOT NULL DEFAULT '[]'::jsonb,
  decision text NOT NULL CHECK (decision IN ('allow','warn','rewrite','block')),
  guarded_instruction text NOT NULL DEFAULT '',
  canonical_mutation_allowed boolean NOT NULL DEFAULT false CHECK (canonical_mutation_allowed = false),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_demem_enforcement_instruction
  ON lucidota_control.demem_enforcement_decision(instruction_sha256, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_demem_enforcement_decision
  ON lucidota_control.demem_enforcement_decision(decision, created_at DESC);

COMMIT;
