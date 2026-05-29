-- FILE: 06_SCHEMA/048_phase05_allowlisted_ingest.sql
-- PURPOSE: Phase 0.5 allowlisted ingest custody table gated by clean security manifest.
-- COMPLIANCE: no graph mutation; no quarantined/deferred findings; content is hash-addressed custody.
BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.allowlisted_ingest_artifact (
  artifact_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  manifest_path text NOT NULL,
  source_path text NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
  mime_guess text NOT NULL DEFAULT '',
  ingest_status text NOT NULL CHECK (ingest_status IN ('custody_recorded','skipped_non_text','failed')),
  authority_class lucidota_archaeology.authority_class NOT NULL DEFAULT 'raw_evidence',
  content_in_db boolean NOT NULL DEFAULT false,
  content_preview_sha256 text CHECK (content_preview_sha256 IS NULL OR content_preview_sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(manifest_path, source_path, source_sha256)
);
CREATE INDEX IF NOT EXISTS idx_allowlisted_ingest_source_sha ON lucidota_archaeology.allowlisted_ingest_artifact(source_sha256);
CREATE INDEX IF NOT EXISTS idx_allowlisted_ingest_status ON lucidota_archaeology.allowlisted_ingest_artifact(ingest_status, created_at DESC);
COMMIT;
