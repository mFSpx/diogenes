-- FILE: 06_SCHEMA/041_boring_beast_loop_contracts.sql
-- PURPOSE: executable contract tables for Boring Beast real-code work loops 4-10.
-- COMPLIANCE: idempotent; state/control plane only; canonical graph writes remain blocked.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.real_work_loop_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_key text NOT NULL UNIQUE,
  started_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  start_loop integer NOT NULL CHECK (start_loop BETWEEN 1 AND 10),
  end_loop integer NOT NULL CHECK (end_loop BETWEEN 1 AND 10),
  execute_performed boolean NOT NULL DEFAULT false,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.real_work_item_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_uuid uuid REFERENCES lucidota_control.real_work_loop_run(run_uuid) ON DELETE CASCADE,
  loop integer NOT NULL CHECK (loop BETWEEN 1 AND 10),
  item integer NOT NULL CHECK (item BETWEEN 1 AND 20),
  target text NOT NULL,
  counted boolean NOT NULL DEFAULT true,
  capability_delta text NOT NULL,
  evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
  files_changed jsonb NOT NULL DEFAULT '[]'::jsonb,
  validation jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(run_uuid, loop, item)
);

CREATE TABLE IF NOT EXISTS lucidota_control.work_order_contract (
  work_order_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  work_order_key text NOT NULL UNIQUE,
  target_number integer NOT NULL CHECK (target_number BETWEEN 1 AND 20),
  target_name text NOT NULL,
  valid boolean NOT NULL,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  normalized_payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.idempotency_registry (
  registry_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  idempotency_key text NOT NULL UNIQUE,
  first_payload_sha256 text NOT NULL CHECK (first_payload_sha256 ~ '^[0-9a-f]{64}$'),
  duplicate_seen_count integer NOT NULL DEFAULT 0 CHECK (duplicate_seen_count >= 0),
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.queue_transition_audit (
  audit_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  old_status text NOT NULL,
  new_status text NOT NULL,
  actor_role text NOT NULL,
  allowed boolean NOT NULL,
  expected boolean NOT NULL,
  pass boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.audit_json_validation (
  validation_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id text NOT NULL,
  verdict text NOT NULL,
  valid boolean NOT NULL,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  remediation text NOT NULL DEFAULT '',
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.oracle_snapshot_compare (
  compare_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  compare_key text NOT NULL UNIQUE,
  allowed_files jsonb NOT NULL DEFAULT '[]'::jsonb,
  before_manifest jsonb NOT NULL DEFAULT '[]'::jsonb,
  after_manifest jsonb NOT NULL DEFAULT '[]'::jsonb,
  violations jsonb NOT NULL DEFAULT '[]'::jsonb,
  pass boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.graph_write_attempt_log (
  attempt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  attempt_kind text NOT NULL CHECK (attempt_kind IN ('promotion_packet','promotion_materialize','direct_write_probe')),
  blocked boolean NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  error_message text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.demem_instruction_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  instruction_sha256 text NOT NULL CHECK (instruction_sha256 ~ '^[0-9a-f]{64}$'),
  boundary_hits jsonb NOT NULL DEFAULT '[]'::jsonb,
  blocked boolean NOT NULL,
  decision text NOT NULL CHECK (decision IN ('allow','block','warn','rewrite')),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.catchme_access_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_key text NOT NULL,
  requested_use text NOT NULL,
  operator_approved boolean NOT NULL DEFAULT false,
  allowed boolean NOT NULL,
  reason text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.simplemem_query_log (
  query_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  query_sha256 text NOT NULL CHECK (query_sha256 ~ '^[0-9a-f]{64}$'),
  candidate_count integer NOT NULL CHECK (candidate_count >= 0),
  safe_to_answer_from_this_alone boolean NOT NULL DEFAULT false CHECK (safe_to_answer_from_this_alone = false),
  candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.daemon_preflight (
  preflight_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  daemon_name text NOT NULL,
  command_path text NOT NULL,
  exists_executable boolean NOT NULL,
  status text NOT NULL CHECK (status IN ('ready','missing','not_executable','failed')),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

COMMIT;
