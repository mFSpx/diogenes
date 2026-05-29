-- FILE: 06_SCHEMA/057_tracer_claim_packet_bridge.sql
-- PURPOSE: TRACER-lite labels attached to claim packets with source-span provenance.
-- COMPLIANCE: labels epistemic moves; does not promote truth or mutate graph.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.tracer_claim_packet_bridge (
  bridge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  claim_packet_uuid uuid NOT NULL,
  trace_uuid uuid NOT NULL REFERENCES lucidota_control.tracer_lite_label(trace_uuid) ON DELETE CASCADE,
  label text NOT NULL CHECK (label IN ('quote','compression','inference','abduction','speculation','operator_prior','heuristic','contradiction','falsification_target','PFM')),
  source_span jsonb NOT NULL DEFAULT '{}'::jsonb,
  source_span_sha256 text NOT NULL CHECK (source_span_sha256 ~ '^[0-9a-f]{64}$'),
  authority_class text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(claim_packet_uuid, label, source_span_sha256)
);

CREATE INDEX IF NOT EXISTS idx_tracer_claim_packet_bridge_packet
  ON lucidota_control.tracer_claim_packet_bridge(claim_packet_uuid, created_at DESC);

COMMIT;
