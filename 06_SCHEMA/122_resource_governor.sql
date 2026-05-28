-- FILE: 06_SCHEMA/122_resource_governor.sql
-- PURPOSE: Resource/PID governance ledger for bounded LUCIDOTA workers.
-- COMPLIANCE: control-plane only; receipts own proof, this schema indexes collars/throttles/PG supervision.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.pid_registry (
  registry_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  pid integer NOT NULL,
  owner text NOT NULL,
  purpose text NOT NULL,
  cwd text NOT NULL DEFAULT '',
  command jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(command) = 'array'),
  max_memory_mb integer NOT NULL DEFAULT 0 CHECK (max_memory_mb >= 0),
  max_cpu_percent numeric NOT NULL DEFAULT 0 CHECK (max_cpu_percent >= 0),
  kill_policy text NOT NULL DEFAULT '',
  status text NOT NULL CHECK (status IN ('planned','running','missing','reaped','succeeded','failed','terminated','blocked')),
  receipt_path text NOT NULL DEFAULT '',
  telemetry jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(telemetry) = 'object'),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(detail) = 'object'),
  observed_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pid_registry_pid_observed_idx
  ON lucidota_control.pid_registry(pid, observed_at DESC);

CREATE INDEX IF NOT EXISTS pid_registry_status_observed_idx
  ON lucidota_control.pid_registry(status, observed_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.resource_throttle_receipt (
  throttle_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  reason text NOT NULL,
  safe_workers integer NOT NULL DEFAULT 1 CHECK (safe_workers >= 0),
  requested_workers integer NOT NULL DEFAULT 1 CHECK (requested_workers >= 0),
  receipt_path text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(detail) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS resource_throttle_receipt_created_idx
  ON lucidota_control.resource_throttle_receipt(created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.pg_supervision_receipt (
  supervision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  action text NOT NULL,
  candidate_count integer NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
  terminated_count integer NOT NULL DEFAULT 0 CHECK (terminated_count >= 0),
  receipt_path text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(detail) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pg_supervision_receipt_created_idx
  ON lucidota_control.pg_supervision_receipt(created_at DESC);
