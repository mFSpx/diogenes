-- FILE: 06_SCHEMA/091_chrono_temporal_claim_invalidation.sql
-- PURPOSE: Mark temporal claims invalid with evidence; never delete temporal evidence.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

ALTER TABLE lucidota_korpus.temporal_claim
  ADD COLUMN IF NOT EXISTS invalid boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS invalidated_at timestamptz,
  ADD COLUMN IF NOT EXISTS invalidation_reason text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS invalidation_evidence jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE lucidota_korpus.temporal_claim
  DROP CONSTRAINT IF EXISTS temporal_claim_invalid_requires_evidence,
  ADD CONSTRAINT temporal_claim_invalid_requires_evidence CHECK (
    invalid = false OR (invalidated_at IS NOT NULL AND invalidation_reason <> '' AND jsonb_typeof(invalidation_evidence)='array' AND jsonb_array_length(invalidation_evidence)>0)
  );

CREATE TABLE IF NOT EXISTS lucidota_korpus.temporal_claim_invalidation_event (
  invalidation_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid) ON DELETE RESTRICT,
  reason text NOT NULL,
  evidence_refs jsonb NOT NULL CHECK (jsonb_typeof(evidence_refs)='array' AND jsonb_array_length(evidence_refs)>0),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_temporal_claim_invalid
  ON lucidota_korpus.temporal_claim(invalid, file_uuid, candidate_timestamp);
