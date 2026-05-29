-- FILE: 06_SCHEMA/069_graph_promotion_approval_state_machine.sql
-- PURPOSE: enforce candidate -> defer/reject/operator_confirmed -> materialized state receipts.
-- COMPLIANCE: no direct canonical graph mutation; materialized requires materialization receipt.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

ALTER TYPE lucidota_go.promotion_decision_status ADD VALUE IF NOT EXISTS 'materialized';

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_state_transition (
  transition_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_packet(packet_uuid) ON DELETE RESTRICT,
  from_status text NOT NULL,
  to_status text NOT NULL CHECK (to_status IN ('defer','reject','operator_confirmed','materialized','superseded')),
  decision_uuid uuid REFERENCES lucidota_go.graph_promotion_decision(decision_uuid) ON DELETE RESTRICT,
  materialization_uuid uuid REFERENCES lucidota_go.graph_promotion_materialization(materialization_uuid) ON DELETE RESTRICT,
  command_envelope_uuid uuid,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  allowed boolean NOT NULL DEFAULT true CHECK (allowed = true),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT graph_promotion_transition_requires_evidence CHECK (jsonb_array_length(evidence_refs) > 0)
);

CREATE INDEX IF NOT EXISTS idx_graph_promotion_state_transition_packet
  ON lucidota_go.graph_promotion_state_transition(packet_uuid, created_at DESC);
