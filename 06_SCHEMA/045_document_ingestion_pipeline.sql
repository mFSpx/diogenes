-- FILE: 06_SCHEMA/045_document_ingestion_pipeline.sql
-- PURPOSE: production document parser ingestion ledger for KORPUS evidence intake.
-- COMPLIANCE: idempotent; parser output is evidence candidate/provenance, never graph truth.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE TABLE IF NOT EXISTS lucidota_korpus.document_parse_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  parser_name text NOT NULL,
  parser_version text NOT NULL DEFAULT 'local_text_v1',
  source_path text NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  mime_guess text NOT NULL DEFAULT '',
  status text NOT NULL CHECK (status IN ('parsed','deferred','failed')),
  output_json_path text NOT NULL DEFAULT '',
  output_markdown_path text NOT NULL DEFAULT '',
  parser_output_truth_status text NOT NULL DEFAULT 'not_truth_evidence_candidate' CHECK (parser_output_truth_status='not_truth_evidence_candidate'),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(source_sha256, parser_name, parser_version)
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.document_parse_span (
  span_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_uuid uuid NOT NULL REFERENCES lucidota_korpus.document_parse_run(run_uuid) ON DELETE CASCADE,
  page_number integer NOT NULL DEFAULT 1,
  block_index integer NOT NULL DEFAULT 0,
  start_char integer NOT NULL CHECK (start_char >= 0),
  end_char integer NOT NULL CHECK (end_char >= start_char),
  span_sha256 text NOT NULL CHECK (span_sha256 ~ '^[0-9a-f]{64}$'),
  label text NOT NULL DEFAULT 'text_span',
  provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_document_parse_span_run ON lucidota_korpus.document_parse_span(run_uuid, page_number, block_index);

COMMIT;
