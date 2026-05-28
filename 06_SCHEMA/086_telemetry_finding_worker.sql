-- FILE: 06_SCHEMA/086_telemetry_finding_worker.sql
-- PURPOSE: Deterministic telemetry findings derived from sticker_feature_vector rows.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.telemetry_finding_worker_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vector_uuid uuid NOT NULL REFERENCES lucidota_archaeology.sticker_feature_vector(vector_uuid) ON DELETE CASCADE,
  finding_signature text NOT NULL CHECK (finding_signature ~ '^[0-9a-f]{64}$'),
  finding_uuid uuid REFERENCES lucidota_archaeology.telemetry_finding(finding_uuid) ON DELETE SET NULL,
  worker_version text NOT NULL DEFAULT 'telemetry_finding_worker_v1',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS telemetry_finding_worker_receipt_signature_idx
  ON lucidota_archaeology.telemetry_finding_worker_receipt(finding_signature);
