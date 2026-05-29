-- FILE: 06_SCHEMA/084_absurd_dead_letter_review.sql
-- PURPOSE: Review ledger for ABSURD dead-letter classification/retry decisions.
-- COMPLIANCE: Idempotent, append-only review records. Does not delete DLQ rows.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_dead_letter_review (
  review_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dead_letter_uuid uuid NOT NULL REFERENCES lucidota_control.absurd_queue_dead_letter(dead_letter_uuid) ON DELETE RESTRICT,
  action text NOT NULL CHECK (action IN ('classify','retry_requested','retry_enqueued','resolution_note')),
  classification text NOT NULL DEFAULT 'unknown' CHECK (classification IN ('unknown','smoke_test','transient','permanent','operator_review','bug','external_dependency')),
  performed_by text NOT NULL DEFAULT 'absurd_dead_letter_review',
  execute_performed boolean NOT NULL DEFAULT false,
  retry_job_uuid uuid REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE SET NULL,
  note text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_absurd_dead_letter_review_dlq_created
  ON lucidota_control.absurd_dead_letter_review(dead_letter_uuid, created_at DESC);
