-- FILE: 06_SCHEMA/114_project2501_board_worker.sql
-- PURPOSE: Bounded Project 2501 slow/audit worker run receipts.
-- COMPLIANCE: claims one localized work_order asynchronously; emits receipts/metrics only; no canonical graph writes and no Big Board feature mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE INDEX IF NOT EXISTS idx_project2501_work_order_claim
  ON lucidota_control.work_order(status, lane, created_at)
  WHERE status IN ('created','queued');

CREATE TABLE IF NOT EXISTS lucidota_control.board_worker_run (
  board_worker_run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_id text NOT NULL,
  mode text NOT NULL DEFAULT 'worker_once',
  status text NOT NULL CHECK (status IN ('dry_run','claimed','succeeded','failed','no_work')),
  work_order_uuid uuid REFERENCES lucidota_control.work_order(work_order_uuid) ON DELETE SET NULL,
  event_id text REFERENCES lucidota_control.event_envelope(event_id) ON DELETE SET NULL,
  decision_uuid uuid REFERENCES lucidota_control.route_decision(decision_uuid) ON DELETE SET NULL,
  lane text NOT NULL DEFAULT 'audit' CHECK (lane IN ('fast','slow','audit','external','dead_letter')),
  work_kind text NOT NULL DEFAULT 'bounded_audit',
  bounded_step text NOT NULL DEFAULT 'validate_and_receipt',
  receipt_uuid uuid REFERENCES lucidota_control.work_receipt(work_receipt_uuid) ON DELETE SET NULL,
  source_receipt text NOT NULL DEFAULT '',
  latency_ms numeric NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  operator_feature_authority_required boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_board_worker_run_created
  ON lucidota_control.board_worker_run(created_at DESC);

COMMENT ON TABLE lucidota_control.board_worker_run IS
  'Project 2501 bounded slow/audit worker receipts. UI feature shape remains operator-authority; metrics land in ledger/DB first.';

COMMIT;
