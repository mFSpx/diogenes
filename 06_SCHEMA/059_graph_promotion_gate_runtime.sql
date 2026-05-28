-- FILE: 06_SCHEMA/059_graph_promotion_gate_runtime.sql
-- PURPOSE: graph promotion gate audit wrapper; decisions require evidence and preflight.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_gate_audit (
  gate_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  candidate_payload_sha256 text NOT NULL CHECK (candidate_payload_sha256 ~ '^[0-9a-f]{64}$'),
  decision text NOT NULL,
  materialize_requested boolean NOT NULL DEFAULT false,
  preflight jsonb NOT NULL DEFAULT '{}'::jsonb,
  allowed boolean NOT NULL,
  packet_uuid uuid,
  decision_uuid uuid,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_graph_promotion_gate_allowed
  ON lucidota_go.graph_promotion_gate_audit(allowed, created_at DESC);

COMMIT;
