-- FILE: 06_SCHEMA/070_master_eye_runtime_review.sql
-- PURPOSE: runtime dedupe/provenance fields for Master Eye reviews.
-- COMPLIANCE: reviews Phase 0.5 artifacts only; no graph mutation.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

ALTER TABLE lucidota_archaeology.master_eye_review
  ADD COLUMN IF NOT EXISTS review_key text,
  ADD COLUMN IF NOT EXISTS source_detail jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS ux_master_eye_review_key
  ON lucidota_archaeology.master_eye_review(review_key)
  WHERE review_key IS NOT NULL;

COMMIT;
