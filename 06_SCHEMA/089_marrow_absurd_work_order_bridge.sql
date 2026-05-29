-- FILE: 06_SCHEMA/089_marrow_absurd_work_order_bridge.sql
-- PURPOSE: Bridge local Marrow command receipts into ABSURD work orders.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES ('marrow_loop', 'Marrow/CEP bridge', 3, 'Marrow command receipts queued as ABSURD work orders; no graph mutation.')
ON CONFLICT (queue_name) DO UPDATE SET updated_at=now(), notes=EXCLUDED.notes;

CREATE TABLE IF NOT EXISTS lucidota_control.marrow_absurd_bridge_receipt (
  bridge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  command_uuid uuid NOT NULL,
  receipt_path text NOT NULL,
  receipt_sha256 text NOT NULL CHECK (receipt_sha256 ~ '^[0-9a-f]{64}$'),
  job_uuid uuid REFERENCES lucidota_control.absurd_queue_job(job_uuid) ON DELETE SET NULL,
  idempotency_key text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
