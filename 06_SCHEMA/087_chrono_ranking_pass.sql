-- FILE: 06_SCHEMA/087_chrono_ranking_pass.sql
-- PURPOSE: Immutable Chrono ranking pass records and selected temporal claim links.
-- LAW: Timeline is projection; temporal claims are archive; ranking passes append only.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_korpus;

CREATE TABLE IF NOT EXISTS lucidota_korpus.temporal_ranking_pass (
  ranking_pass_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ranking_algorithm text NOT NULL DEFAULT 'trust_weight_desc_timestamp_asc_created_desc_v1',
  claim_count bigint NOT NULL DEFAULT 0 CHECK (claim_count >= 0),
  file_count bigint NOT NULL DEFAULT 0 CHECK (file_count >= 0),
  selected_count bigint NOT NULL DEFAULT 0 CHECK (selected_count >= 0),
  ranking_violations bigint NOT NULL DEFAULT 0 CHECK (ranking_violations >= 0),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_korpus.temporal_ranking_selection (
  ranking_selection_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ranking_pass_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_ranking_pass(ranking_pass_uuid) ON DELETE RESTRICT,
  file_uuid uuid NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE RESTRICT,
  claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid) ON DELETE RESTRICT,
  candidate_timestamp timestamptz NOT NULL,
  evidence_source text NOT NULL,
  trust_weight numeric(3,2) NOT NULL,
  rank_priority integer NOT NULL DEFAULT 1 CHECK (rank_priority = 1),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (ranking_pass_uuid, file_uuid)
);

CREATE INDEX IF NOT EXISTS idx_temporal_ranking_selection_pass_source
  ON lucidota_korpus.temporal_ranking_selection(ranking_pass_uuid, evidence_source);
CREATE INDEX IF NOT EXISTS idx_temporal_ranking_selection_claim
  ON lucidota_korpus.temporal_ranking_selection(claim_uuid);
