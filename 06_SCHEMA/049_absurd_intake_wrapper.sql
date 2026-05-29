-- FILE: 06_SCHEMA/049_absurd_intake_wrapper.sql
-- PURPOSE: ABSURD wrapper contract for supervised Rust intake service custody.
-- COMPLIANCE: idempotent; no drop mutation; no graph mutation.
BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES ('intake', 'Rust lucidota-intake watcher custody wrapper', 3, 'ABSURD wrapper observes supervised Rust intake service and writes workflow receipts; does not move drops itself.')
ON CONFLICT (queue_name) DO UPDATE SET owner_subsystem=EXCLUDED.owner_subsystem, max_attempts=EXCLUDED.max_attempts, notes=EXCLUDED.notes, updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.intake_service_custody_event (
  custody_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  service_name text NOT NULL DEFAULT 'lucidota-intake.service',
  event_kind text NOT NULL CHECK (event_kind IN ('health_check','drop_dir_scan','service_active','service_inactive','error')),
  active boolean NOT NULL DEFAULT false,
  drop_dir text NOT NULL DEFAULT '',
  digested_dir text NOT NULL DEFAULT '',
  quarantine_dir text NOT NULL DEFAULT '',
  pending_drop_count integer NOT NULL DEFAULT 0 CHECK (pending_drop_count >= 0),
  workflow_event_id uuid,
  evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_intake_service_custody_event_created ON lucidota_control.intake_service_custody_event(created_at DESC);

INSERT INTO lucidota_control.absurd_worker_contract(
  worker_key, queue_name, script_path, input_contract, output_contract, idempotency_rule, retry_policy, dead_letter_policy, canonical_graph_write_allowed, status, evidence_refs, detail
) VALUES (
  'intake_worker','intake','scripts/absurd_intake_worker.py',
  '{"job_kind":"intake_health_check","payload_fields":["drop_dir","digested_dir","quarantine_dir"]}'::jsonb,
  '{"writes":["lucidota_control.workflow_event","lucidota_control.intake_service_custody_event"],"forbidden":["drop move","canonical graph mutation"]}'::jsonb,
  'service_name + drop_dir + timestamp_bucket',
  'retry health checks only; never reprocess physical drops from wrapper',
  'dead-letter failed service checks without moving files',
  false, 'implemented', '["06_SCHEMA/049_absurd_intake_wrapper.sql"]'::jsonb,
  '{"rust_service":"lucidota-intake.service"}'::jsonb
) ON CONFLICT (worker_key) DO UPDATE SET
  queue_name=EXCLUDED.queue_name, script_path=EXCLUDED.script_path, input_contract=EXCLUDED.input_contract,
  output_contract=EXCLUDED.output_contract, idempotency_rule=EXCLUDED.idempotency_rule, retry_policy=EXCLUDED.retry_policy,
  dead_letter_policy=EXCLUDED.dead_letter_policy, canonical_graph_write_allowed=false, status=EXCLUDED.status,
  evidence_refs=EXCLUDED.evidence_refs, updated_at=now(), detail=EXCLUDED.detail;
COMMIT;
