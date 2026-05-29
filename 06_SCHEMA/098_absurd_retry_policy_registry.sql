-- FILE: 06_SCHEMA/098_absurd_retry_policy_registry.sql
-- PURPOSE: Per-queue retry policy registry.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_retry_policy_registry (
  queue_name text PRIMARY KEY REFERENCES lucidota_control.absurd_queue(queue_name) ON DELETE CASCADE,
  max_attempts integer NOT NULL CHECK (max_attempts > 0),
  backoff_seconds integer NOT NULL DEFAULT 60 CHECK (backoff_seconds >= 0),
  dead_letter_after_attempts integer NOT NULL CHECK (dead_letter_after_attempts > 0),
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CHECK (dead_letter_after_attempts <= max_attempts)
);

INSERT INTO lucidota_control.absurd_retry_policy_registry(queue_name,max_attempts,backoff_seconds,dead_letter_after_attempts,detail)
SELECT queue_name,max_attempts,60,max_attempts,'{"source":"queue_defaults"}'::jsonb FROM lucidota_control.absurd_queue
ON CONFLICT(queue_name) DO UPDATE SET max_attempts=EXCLUDED.max_attempts, dead_letter_after_attempts=EXCLUDED.dead_letter_after_attempts, updated_at=now();
