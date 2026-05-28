-- FILE: 06_SCHEMA/095_cep_graph_packet_staging.sql
-- PURPOSE: Receipt table for CEP conversation_command -> graph promotion packet staging.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.cep_graph_packet_stage_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  command_uuid uuid NOT NULL,
  packet_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_packet(packet_uuid) ON DELETE RESTRICT,
  command_payload_sha256 text NOT NULL CHECK (command_payload_sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(command_uuid, packet_uuid)
);
