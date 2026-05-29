-- FILE: 06_SCHEMA/047_simplemem_eval_promotion_bridge.sql
-- PURPOSE: bridge SimpleMem NOT_TRUTH candidates into eval + graph promotion packet staging.
-- COMPLIANCE: candidates remain NOT_TRUTH; no canonical graph mutation.
BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.simplemem_candidate_eval (
  eval_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  candidate_uuid uuid,
  query_sha256 text NOT NULL CHECK (query_sha256 ~ '^[0-9a-f]{64}$'),
  source_ref text NOT NULL,
  candidate_text_sha256 text NOT NULL CHECK (candidate_text_sha256 ~ '^[0-9a-f]{64}$'),
  eval_status text NOT NULL CHECK (eval_status IN ('candidate_only','defer','reject','promotion_packet_created')),
  safe_to_answer_from_this_alone boolean NOT NULL DEFAULT false CHECK (safe_to_answer_from_this_alone=false),
  not_truth boolean NOT NULL DEFAULT true CHECK (not_truth=true),
  promotion_packet_uuid uuid,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_simplemem_candidate_eval_query ON lucidota_control.simplemem_candidate_eval(query_sha256, created_at DESC);
COMMIT;
