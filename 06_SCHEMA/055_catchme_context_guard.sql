-- FILE: 06_SCHEMA/055_catchme_context_guard.sql
-- PURPOSE: executable CatchMe consent/sensitivity decisions for context use.
-- COMPLIANCE: unsafe memory/context use is blocked in code before extraction/runtime use.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.catchme_context_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  path_sha256 text NOT NULL CHECK (path_sha256 ~ '^[0-9a-f]{64}$'),
  path_ref text NOT NULL,
  scope_key text NOT NULL DEFAULT '',
  requested_use text NOT NULL,
  sensitivity_class text NOT NULL CHECK (sensitivity_class IN ('public','internal','private','secret','revoked')),
  consent_status text NOT NULL CHECK (consent_status IN ('allowed','requires_operator','revoked')),
  operator_approved boolean NOT NULL DEFAULT false,
  allowed boolean NOT NULL,
  reason text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_catchme_context_path
  ON lucidota_control.catchme_context_decision(path_sha256, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_catchme_context_allowed
  ON lucidota_control.catchme_context_decision(allowed, created_at DESC);

COMMIT;
