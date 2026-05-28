-- FILE: 06_SCHEMA/111_model_generation_event_lane.sql
-- PURPOSE: durable PG/ABSURD event lane for exact model-generation telemetry.
-- COMPLIANCE: targeted async staging only; no canonical graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES (
  'model_generation',
  'Model routing telemetry',
  3,
  'Async staging lane for model invocation receipts: target, model, payload size, latency, raw output, raw response pointer.'
)
ON CONFLICT(queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  max_attempts=EXCLUDED.max_attempts,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.model_generation_event (
  event_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  receipt_path text NOT NULL UNIQUE,
  receipt_sha256 text NOT NULL CHECK (receipt_sha256 ~ '^[0-9a-f]{64}$'),
  receipt_generated_at timestamptz,
  provider text NOT NULL DEFAULT '',
  mode text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT '',
  target text NOT NULL CHECK (length(target) > 0),
  model_name text NOT NULL CHECK (length(model_name) > 0),
  payload_size_bytes integer NOT NULL CHECK (payload_size_bytes >= 0),
  payload_size_chars integer NOT NULL CHECK (payload_size_chars >= 0),
  latency_ms numeric NOT NULL CHECK (latency_ms >= 0),
  raw_output text NOT NULL DEFAULT '',
  raw_output_chars integer NOT NULL CHECK (raw_output_chars >= 0),
  raw_output_sha256 text NOT NULL CHECK (raw_output_sha256 ~ '^[0-9a-f]{64}$'),
  raw_response_present boolean NOT NULL DEFAULT false,
  raw_response jsonb,
  execute_performed boolean NOT NULL DEFAULT false,
  queue_event_uuid uuid REFERENCES lucidota_control.absurd_queue_event(queue_event_uuid) ON DELETE SET NULL,
  staged_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_model_generation_event_target_model_time
  ON lucidota_control.model_generation_event(target, model_name, staged_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_generation_event_execute_time
  ON lucidota_control.model_generation_event(execute_performed, staged_at DESC);

INSERT INTO lucidota_control.absurd_worker_contract(
  worker_key,
  queue_name,
  script_path,
  input_contract,
  output_contract,
  idempotency_rule,
  retry_policy,
  dead_letter_policy,
  canonical_graph_write_allowed,
  status,
  evidence_refs,
  detail
) VALUES (
  'model_generation_event_bridge',
  'model_generation',
  'scripts/model_generation_event_bridge.py',
  '{"job_kind":"model_generation_receipt_stage","payload_fields":["receipt_path","receipt_sha256","target","model_name","payload_size_bytes","latency_ms","raw_output"]}'::jsonb,
  '{"writes":["lucidota_control.model_generation_event","lucidota_control.absurd_queue_event"],"forbidden":["canonical graph mutation"],"truth_status":"runtime_telemetry_not_canonical_graph_truth"}'::jsonb,
  'receipt_path + receipt_sha256',
  'idempotent upsert by receipt_path; retry does not duplicate queue audit event',
  'dead-letter only receipt path, hash, and validation error; preserve raw receipt file separately',
  false,
  'implemented',
  '["06_SCHEMA/111_model_generation_event_lane.sql","scripts/model_generation_event_bridge.py"]'::jsonb,
  '{"bare_steel_rule_4":"targeted async read of one receipt or bounded latest receipts; durable PG event lane; no direct canonical graph writes"}'::jsonb
)
ON CONFLICT(worker_key) DO UPDATE SET
  queue_name=EXCLUDED.queue_name,
  script_path=EXCLUDED.script_path,
  input_contract=EXCLUDED.input_contract,
  output_contract=EXCLUDED.output_contract,
  idempotency_rule=EXCLUDED.idempotency_rule,
  retry_policy=EXCLUDED.retry_policy,
  dead_letter_policy=EXCLUDED.dead_letter_policy,
  canonical_graph_write_allowed=EXCLUDED.canonical_graph_write_allowed,
  status=EXCLUDED.status,
  evidence_refs=EXCLUDED.evidence_refs,
  detail=lucidota_control.absurd_worker_contract.detail || EXCLUDED.detail,
  updated_at=now();

COMMIT;
