-- FILE: 06_SCHEMA/080_sticker_feature_vector_extractor_v1.sql
-- PURPOSE: idempotent receipt table for Sticker feature vector extraction over allowlisted custody text.
-- COMPLIANCE: deterministic metrics only; no ontology softening; no graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.sticker_feature_source_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_uuid uuid NOT NULL,
  vector_uuid uuid NOT NULL REFERENCES lucidota_archaeology.sticker_feature_vector(vector_uuid) ON DELETE CASCADE,
  source_path text NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  feature_version text NOT NULL DEFAULT 'phase05_stickers_v1',
  extractor text NOT NULL DEFAULT 'scripts/sticker_feature_extractor_v1.py',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(run_uuid, feature_version)
);

CREATE INDEX IF NOT EXISTS idx_sticker_feature_source_receipt_sha
  ON lucidota_archaeology.sticker_feature_source_receipt(source_sha256);

COMMIT;
