-- FILE: 06_SCHEMA/053_simplemem_candidate_index.sql
-- PURPOSE: idempotent SimpleMem candidate index; candidates are explicitly NOT_TRUTH.
-- COMPLIANCE: no graph mutation; promotion_allowed remains false by table law.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE UNIQUE INDEX IF NOT EXISTS ux_simplemem_candidate_query_source_text_v1
  ON lucidota_control.simplemem_candidate(query_sha256, source_ref, md5(candidate_text));

CREATE TABLE IF NOT EXISTS lucidota_control.simplemem_candidate_index_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  query_text_sha256 text NOT NULL CHECK (query_text_sha256 ~ '^[0-9a-f]{64}$'),
  source_table text NOT NULL,
  candidates_seen integer NOT NULL DEFAULT 0 CHECK (candidates_seen >= 0),
  candidates_inserted integer NOT NULL DEFAULT 0 CHECK (candidates_inserted >= 0),
  safe_to_answer_from_this_alone boolean NOT NULL DEFAULT false CHECK (safe_to_answer_from_this_alone = false),
  not_truth boolean NOT NULL DEFAULT true CHECK (not_truth = true),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMIT;
