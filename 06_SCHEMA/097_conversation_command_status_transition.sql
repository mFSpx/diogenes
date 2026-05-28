-- FILE: 06_SCHEMA/097_conversation_command_status_transition.sql
-- PURPOSE: Legal conversation_command status transition events.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.conversation_command_status_event (
  event_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  command_uuid uuid NOT NULL REFERENCES lucidota_control.conversation_command(command_uuid) ON DELETE RESTRICT,
  from_status text NOT NULL,
  to_status text NOT NULL,
  actor text NOT NULL DEFAULT 'conversation_command_status_worker',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CHECK ((from_status,to_status) IN (('staged','queued'),('queued','accepted'),('accepted','executed'),('queued','rejected'),('accepted','rejected'),('staged','rejected'),('staged','superseded'),('queued','superseded')))
);

CREATE INDEX IF NOT EXISTS idx_conversation_command_status_event_command
  ON lucidota_control.conversation_command_status_event(command_uuid, created_at DESC);
