-- FILE: 06_SCHEMA/043_absurd_remaining_worker_contracts.sql
-- PURPOSE: close remaining ABSURD worker wrapper contracts for Surfaces/CEP and Graph Promotion.
-- COMPLIANCE: idempotent; control queue only; no canonical graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES
  ('surface_cep', 'Darwinian Surfaces / Conversation Command Envelope fan-in', 3, 'Surface instructions fan into conversation_command staging; no direct canonical mutation.'),
  ('graph_promotion', 'Graph Promotion Worker', 2, 'Evidence-gated graph promotion queue; canonical mutation only through promotion path and graph journal.')
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  max_attempts=EXCLUDED.max_attempts,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_worker_contract (
  contract_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  worker_key text NOT NULL UNIQUE,
  queue_name text NOT NULL REFERENCES lucidota_control.absurd_queue(queue_name) ON DELETE RESTRICT,
  script_path text NOT NULL,
  input_contract jsonb NOT NULL,
  output_contract jsonb NOT NULL,
  idempotency_rule text NOT NULL,
  retry_policy text NOT NULL,
  dead_letter_policy text NOT NULL,
  canonical_graph_write_allowed boolean NOT NULL DEFAULT false,
  status text NOT NULL CHECK (status IN ('contracted','implemented','verified','blocked')),
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.absurd_worker_contract(
  worker_key, queue_name, script_path, input_contract, output_contract, idempotency_rule, retry_policy, dead_letter_policy, canonical_graph_write_allowed, status, evidence_refs, detail
) VALUES
  (
    'surface_cep_worker', 'surface_cep', 'scripts/spine_surface_cep_worker.py',
    '{"job_kind":"surface_cep_health_check|surface_instruction_fan_in|conversation_command_work_order|surface_instruction_compiled_command","payload_fields":["plain_language_instruction","command_envelope","payload_sha256"]}'::jsonb,
    '{"writes":["lucidota_control.workflow_event","lucidota_control.conversation_command optional when payload provided"],"forbidden":["canonical graph mutation"]}'::jsonb,
    'payload_sha256 + plain_language_instruction',
    'retry idempotent fan-in only; never direct surface mutation',
    'dead-letter envelope metadata; do not expand secret payloads',
    false, 'implemented', '["06_SCHEMA/043_absurd_remaining_worker_contracts.sql"]'::jsonb,
    '{"contract_level":"wrapper_implemented"}'::jsonb
  ),
  (
    'graph_promotion_worker', 'graph_promotion', 'scripts/absurd_graph_promotion_worker.py',
    '{"job_kind":"graph_promotion_health_check|graph_promotion_packet_defer","payload_fields":["candidate_payload","evidence_refs","authority_class","decision"]}'::jsonb,
    '{"writes":["lucidota_control.workflow_event","lucidota_go.graph_promotion_packet optional"],"canonical_materialization":"disabled_by_default"}'::jsonb,
    'candidate_payload_sha256 + decision + evidence_refs',
    'retry validation/packet creation only; materialization requires operator_confirmed promotion transaction',
    'dead-letter packet validation failures without partial graph mutation',
    false, 'implemented', '["06_SCHEMA/043_absurd_remaining_worker_contracts.sql"]'::jsonb,
    '{"contract_level":"wrapper_implemented"}'::jsonb
  )
ON CONFLICT (worker_key) DO UPDATE SET
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
  updated_at=now(),
  detail=EXCLUDED.detail;

COMMIT;
