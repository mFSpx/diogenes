-- FILE: 06_SCHEMA/058_chrono_queue_event_bridge.sql
-- PURPOSE: idempotent bridge receipts for ABSURD queue events appended to Chrono temporal_claim.
-- COMPLIANCE: appends temporal evidence only; never mutates previous claims.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.chrono_queue_event_bridge_receipt (
  bridge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  queue_event_uuid uuid NOT NULL,
  chrono_claim_uuid uuid NOT NULL,
  event_payload_sha256 text NOT NULL CHECK (event_payload_sha256 ~ '^[0-9a-f]{64}$'),
  evidence_source text NOT NULL DEFAULT 'absurd_queue_event_bridge',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(queue_event_uuid, evidence_source)
);

CREATE INDEX IF NOT EXISTS idx_chrono_queue_event_bridge_claim
  ON lucidota_control.chrono_queue_event_bridge_receipt(chrono_claim_uuid);

COMMIT;
