-- FILE: 06_SCHEMA/072_marrow_cep_bridge.sql
-- PURPOSE: Marrow command receipt -> staged conversation_command bridge receipts.
-- COMPLIANCE: stages conversation/control-plane commands only; no graph mutation.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.marrow_cep_bridge_receipt (
  bridge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  marrow_command_uuid uuid NOT NULL,
  conversation_command_uuid uuid NOT NULL REFERENCES lucidota_control.conversation_command(command_uuid) ON DELETE RESTRICT,
  receipt_path text NOT NULL,
  receipt_sha256 text NOT NULL CHECK (receipt_sha256 ~ '^[0-9a-f]{64}$'),
  idempotency_key text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(marrow_command_uuid),
  UNIQUE(idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_marrow_cep_bridge_conversation_command
  ON lucidota_control.marrow_cep_bridge_receipt(conversation_command_uuid);

COMMIT;
