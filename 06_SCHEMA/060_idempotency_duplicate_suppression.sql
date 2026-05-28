-- FILE: 06_SCHEMA/060_idempotency_duplicate_suppression.sql
-- PURPOSE: explicit idempotency attempt ledger for duplicate-suppression proofs.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.idempotency_attempt_audit (
  attempt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope text NOT NULL,
  raw_key text NOT NULL,
  normalized_key text NOT NULL,
  first_ref text NOT NULL DEFAULT '',
  duplicate boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_idempotency_attempt_scope_key
  ON lucidota_control.idempotency_attempt_audit(scope, normalized_key, created_at DESC);

COMMIT;
