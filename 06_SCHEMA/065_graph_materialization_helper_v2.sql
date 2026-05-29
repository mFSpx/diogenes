-- FILE: 06_SCHEMA/065_graph_materialization_helper_v2.sql
-- PURPOSE: receipt/audit spine for the reusable graph materialization helper.
-- COMPLIANCE:
--   - no direct canonical graph mutation is introduced here.
--   - helper receipts prove a materialization had a command envelope,
--     non-empty evidence refs, a promotion packet/decision, and a graph_journal row.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_materialization_helper_receipt (
  helper_receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  materialization_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_materialization(materialization_uuid) ON DELETE RESTRICT,
  packet_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_packet(packet_uuid) ON DELETE RESTRICT,
  decision_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_decision(decision_uuid) ON DELETE RESTRICT,
  graph_item_uuid uuid REFERENCES lucidota_go.graph_item(uuid) ON DELETE RESTRICT,
  graph_edge_uuid uuid REFERENCES lucidota_go.graph_edge(edge_uuid) ON DELETE RESTRICT,
  journal_uuid uuid NOT NULL REFERENCES lucidota_go.graph_journal(journal_uuid) ON DELETE RESTRICT,
  command_envelope_uuid uuid NOT NULL,
  evidence_count integer NOT NULL CHECK (evidence_count > 0),
  authority_class text NOT NULL,
  verification_passed boolean NOT NULL DEFAULT true CHECK (verification_passed = true),
  materializer_report_path text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(materialization_uuid)
);

CREATE INDEX IF NOT EXISTS idx_graph_materialization_helper_command
  ON lucidota_go.graph_materialization_helper_receipt(command_envelope_uuid);

CREATE INDEX IF NOT EXISTS idx_graph_materialization_helper_packet
  ON lucidota_go.graph_materialization_helper_receipt(packet_uuid);

COMMIT;
