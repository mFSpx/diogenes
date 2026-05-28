-- FILE: 06_SCHEMA/052_graph_promotion_materialization.sql
-- PURPOSE: graph promotion materialization receipt table.
-- COMPLIANCE:
--   - canonical graph writes are allowed only inside promotion transaction
--     with SET LOCAL lucidota.graph_promotion_path='on'.
--   - every materialization links to a promotion packet, decision, journal row,
--     command envelope UUID, and evidence refs.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_materialization (
  materialization_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_packet(packet_uuid) ON DELETE RESTRICT,
  decision_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_decision(decision_uuid) ON DELETE RESTRICT,
  graph_item_uuid uuid REFERENCES lucidota_go.graph_item(uuid) ON DELETE RESTRICT,
  graph_edge_uuid uuid REFERENCES lucidota_go.graph_edge(edge_uuid) ON DELETE RESTRICT,
  journal_uuid uuid NOT NULL REFERENCES lucidota_go.graph_journal(journal_uuid) ON DELETE RESTRICT,
  command_envelope_uuid uuid NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  materialization_kind text NOT NULL CHECK (materialization_kind IN ('node','edge')),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(packet_uuid, decision_uuid, materialization_kind)
);

CREATE INDEX IF NOT EXISTS idx_graph_promotion_materialization_packet
  ON lucidota_go.graph_promotion_materialization(packet_uuid);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_materialization_command
  ON lucidota_go.graph_promotion_materialization(command_envelope_uuid);

COMMIT;
