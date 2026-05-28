-- FILE: 06_SCHEMA/083_topology_finding_extractor.sql
-- PURPOSE: idempotent receipts for topology findings extracted from design atoms.
-- COMPLIANCE: model/computed findings only; no external action authorization; no graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.topology_finding_source_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  atom_uuid uuid NOT NULL REFERENCES lucidota_archaeology.design_atom(atom_uuid) ON DELETE CASCADE,
  topology_uuid uuid NOT NULL REFERENCES lucidota_archaeology.topology_finding(topology_uuid) ON DELETE CASCADE,
  topology_model text NOT NULL,
  extractor text NOT NULL DEFAULT 'scripts/topology_finding_extractor.py',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(atom_uuid, topology_model)
);

COMMIT;
