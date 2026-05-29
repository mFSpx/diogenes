-- FILE: 06_SCHEMA/068_conversation_command_acceptance.sql
-- PURPOSE: staged conversation_command -> ABSURD work_order acceptance receipts.
-- COMPLIANCE: queues work only; no canonical graph mutation, no direct surface mutation.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.conversation_command_acceptance (
  acceptance_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  command_uuid uuid NOT NULL REFERENCES lucidota_control.conversation_command(command_uuid) ON DELETE RESTRICT,
  job_uuid uuid NOT NULL REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE RESTRICT,
  queue_name text NOT NULL,
  idempotency_key text NOT NULL,
  accepted_by text NOT NULL DEFAULT 'conversation_command_accept_worker',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(command_uuid),
  UNIQUE(queue_name, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_conversation_command_acceptance_job
  ON lucidota_control.conversation_command_acceptance(job_uuid);

COMMIT;
