-- FILE: 06_SCHEMA/085_graph_promotion_evidence_resolver.sql
-- PURPOSE: Append-only evidence-reference resolution receipts before graph promotion packet creation.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_evidence_resolution (
  resolution_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  evidence_ref text NOT NULL,
  ref_kind text NOT NULL,
  resolved boolean NOT NULL,
  resolver text NOT NULL DEFAULT 'scripts/graph_promotion_evidence_resolver.py',
  source_table text,
  source_uuid uuid,
  source_path text,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_promotion_evidence_resolution_ref_created
  ON lucidota_go.graph_promotion_evidence_resolution(evidence_ref, created_at DESC);
