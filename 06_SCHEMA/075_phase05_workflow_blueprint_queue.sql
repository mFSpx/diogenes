-- FILE: 06_SCHEMA/075_phase05_workflow_blueprint_queue.sql
-- PURPOSE: ABSURD work-order receipts for Phase 0.5 workflow_blueprint implementation candidates.
-- COMPLIANCE: queues candidate work only; no graph mutation, no Brain Archaeology full ingest.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
('phase05-workflow-blueprint-implementation-candidate',
 'phase05+absurd',
 '027',
 'active',
 'scripts/phase05_workflow_blueprint_generator.py queue-candidates --execute',
 '{"source":"lucidota_archaeology.workflow_blueprint","queue":"control"}'::jsonb,
 '{"absurd_queue_job":"uuid","blueprint_uuid":"uuid","idempotency_key":"text"}'::jsonb,
 'Queues recovered Phase 0.5 workflow blueprints as ABSURD implementation candidates. Does not auto-implement or graph-promote.'
)
ON CONFLICT (workflow_name) DO UPDATE SET
  owner=EXCLUDED.owner,
  phase=EXCLUDED.phase,
  status=EXCLUDED.status,
  command=EXCLUDED.command,
  inputs=EXCLUDED.inputs,
  outputs=EXCLUDED.outputs,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.phase05_workflow_blueprint_queue_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  blueprint_uuid uuid NOT NULL,
  workflow_key text NOT NULL,
  job_uuid uuid NOT NULL REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE RESTRICT,
  queue_name text NOT NULL DEFAULT 'control',
  idempotency_key text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(blueprint_uuid),
  UNIQUE(queue_name, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_phase05_workflow_blueprint_queue_job
  ON lucidota_control.phase05_workflow_blueprint_queue_receipt(job_uuid);

COMMIT;
