-- FILE: 06_SCHEMA/039_absurd_real_work_loop.sql
-- PURPOSE: executable ABSURD real-work loop hardening: queue laws, command ledger, execution receipts, runtime guards.
-- COMPLIANCE: idempotent, non-destructive for lucidota_state/control plane.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

ALTER TABLE lucidota_control.absurd_queue_job
  ADD COLUMN IF NOT EXISTS locked_by text,
  ADD COLUMN IF NOT EXISTS locked_at timestamptz,
  ADD COLUMN IF NOT EXISTS error_kind text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS error_message text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS rejected_reason text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS transition_guard_version text NOT NULL DEFAULT '039_absurd_real_work_loop_v1';

CREATE OR REPLACE FUNCTION lucidota_control.absurd_queue_transition_allowed(
  old_status text,
  new_status text,
  actor_role text DEFAULT 'worker'
) RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
  IF old_status = new_status THEN
    RETURN true;
  END IF;
  IF actor_role NOT IN ('foreman','worker','auditor','operator','system') THEN
    RETURN false;
  END IF;
  IF old_status = 'queued' AND new_status IN ('leased','running','cancelled','dead_lettered') THEN
    RETURN actor_role IN ('foreman','worker','operator','system');
  ELSIF old_status = 'leased' AND new_status IN ('running','queued','failed','dead_lettered','cancelled') THEN
    RETURN actor_role IN ('foreman','worker','operator','system');
  ELSIF old_status = 'running' AND new_status IN ('succeeded','failed','dead_lettered','queued','cancelled') THEN
    RETURN actor_role IN ('worker','foreman','operator','system');
  ELSIF old_status = 'failed' AND new_status IN ('queued','dead_lettered','cancelled') THEN
    RETURN actor_role IN ('foreman','operator','system');
  ELSIF old_status IN ('succeeded','dead_lettered','cancelled') THEN
    RETURN actor_role IN ('auditor','operator') AND new_status = old_status;
  END IF;
  RETURN false;
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_control.enforce_absurd_queue_transition()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  actor_role text := coalesce(nullif(current_setting('lucidota.actor_role', true), ''), 'worker');
BEGIN
  IF TG_OP = 'UPDATE' AND OLD.status IS DISTINCT FROM NEW.status THEN
    IF NOT lucidota_control.absurd_queue_transition_allowed(OLD.status::text, NEW.status::text, actor_role) THEN
      RAISE EXCEPTION 'illegal absurd queue status transition: % -> % by role %', OLD.status, NEW.status, actor_role
        USING ERRCODE = 'check_violation';
    END IF;
  END IF;
  NEW.updated_at := now();
  IF NEW.status IN ('leased','running') AND NEW.locked_at IS NULL THEN
    NEW.locked_at := now();
  END IF;
  IF NEW.status NOT IN ('leased','running') THEN
    NEW.locked_by := NULL;
    NEW.locked_at := NULL;
    NEW.lease_expires_at := NULL;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_enforce_absurd_queue_transition ON lucidota_control.absurd_queue_job;
CREATE TRIGGER trg_enforce_absurd_queue_transition
BEFORE UPDATE ON lucidota_control.absurd_queue_job
FOR EACH ROW EXECUTE FUNCTION lucidota_control.enforce_absurd_queue_transition();

CREATE TABLE IF NOT EXISTS lucidota_control.conversation_command (
  command_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  command_kind text NOT NULL DEFAULT 'operator_instruction',
  plain_language_instruction text NOT NULL,
  command_envelope jsonb NOT NULL,
  source_surface_id text,
  source_artifact_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  target_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  allowed_effect text NOT NULL DEFAULT 'stage_only',
  authority_class text NOT NULL CHECK (authority_class IN (
    'raw_evidence','operator_authored_assertion','operator_defined_label','deterministic_metric',
    'statistical_finding','model_computed_finding','stream_ml_finding','graph_inferred_relation',
    'operator_confirmed_finding','canonical_doctrine','external_action_authorized'
  )),
  canonical_mutation_allowed boolean NOT NULL DEFAULT false CHECK (canonical_mutation_allowed = false),
  conversation_required boolean NOT NULL DEFAULT true CHECK (conversation_required = true),
  status text NOT NULL DEFAULT 'staged' CHECK (status IN ('staged','queued','accepted','rejected','executed','superseded')),
  idempotency_key text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  accepted_at timestamptz,
  executed_at timestamptz,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS conversation_command_idempotency_idx
  ON lucidota_control.conversation_command(idempotency_key);
CREATE INDEX IF NOT EXISTS conversation_command_status_idx
  ON lucidota_control.conversation_command(status, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.boring_execution_record (
  execution_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id text NOT NULL,
  job_uuid uuid,
  idempotency_key text NOT NULL,
  files_changed jsonb NOT NULL DEFAULT '[]'::jsonb,
  validation_commands jsonb NOT NULL DEFAULT '[]'::jsonb,
  result text NOT NULL,
  status text NOT NULL CHECK (status IN ('succeeded','failed','partial_fail','rejected','dead_lettered')),
  audit_verdict jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS boring_execution_record_idempotency_idx
  ON lucidota_control.boring_execution_record(idempotency_key);

CREATE TABLE IF NOT EXISTS lucidota_control.audit_verdict_contract (
  verdict_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id text NOT NULL,
  verdict text NOT NULL CHECK (verdict IN ('PASS','FAIL','PARTIAL_FAIL')),
  required_fields_ok boolean NOT NULL,
  remediation text NOT NULL DEFAULT '',
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  rejected boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT audit_verdict_requires_evidence CHECK (jsonb_array_length(evidence_refs) > 0),
  CONSTRAINT audit_fail_requires_remediation CHECK (verdict = 'PASS' OR length(btrim(remediation)) > 0)
);

CREATE TABLE IF NOT EXISTS lucidota_control.runtime_status_fact (
  fact_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subsystem text NOT NULL,
  fact_key text NOT NULL,
  fact_value jsonb NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  derived_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(subsystem, fact_key)
);

CREATE TABLE IF NOT EXISTS lucidota_control.tracer_lite_label (
  trace_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_ref text NOT NULL,
  label text NOT NULL CHECK (label IN ('quote','compression','inference','abduction','speculation','operator_prior','heuristic','contradiction','falsification_target','PFM')),
  source_span jsonb NOT NULL DEFAULT '{}'::jsonb,
  authority_class text NOT NULL,
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS tracer_lite_label_packet_idx ON lucidota_control.tracer_lite_label(packet_ref, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.demem_boundary (
  boundary_key text PRIMARY KEY,
  boundary_text text NOT NULL,
  enforcement_mode text NOT NULL DEFAULT 'block' CHECK (enforcement_mode IN ('block','warn','rewrite')),
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.demem_boundary(boundary_key, boundary_text, enforcement_mode)
VALUES
  ('generated_not_policy_mutable','Generated does not mean policy-mutable.','block'),
  ('retrieved_not_verified','Retrieved does not mean verified.','block'),
  ('repeated_not_preferred','Repeated does not mean preferred.','warn'),
  ('surface_not_ui','Conversation remains the UI; surface is an instruction compiler.','block'),
  ('graph_path_not_evidence','Graph paths are maps to evidence, not evidence themselves.','block')
ON CONFLICT (boundary_key) DO UPDATE SET boundary_text=EXCLUDED.boundary_text, enforcement_mode=EXCLUDED.enforcement_mode, active=true;

CREATE TABLE IF NOT EXISTS lucidota_control.catchme_scope (
  scope_key text PRIMARY KEY,
  sensitivity_class text NOT NULL CHECK (sensitivity_class IN ('public','internal','private','secret','revoked')),
  consent_status text NOT NULL CHECK (consent_status IN ('allowed','requires_operator','revoked')),
  allowed_use text NOT NULL DEFAULT 'none',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.catchme_scope(scope_key, sensitivity_class, consent_status, allowed_use)
VALUES
  ('public_context','public','allowed','recall'),
  ('operator_private_context','private','requires_operator','operator_review_only'),
  ('secret_material','secret','requires_operator','blocked_without_explicit_command'),
  ('revoked_context','revoked','revoked','none')
ON CONFLICT (scope_key) DO UPDATE SET sensitivity_class=EXCLUDED.sensitivity_class, consent_status=EXCLUDED.consent_status, allowed_use=EXCLUDED.allowed_use, updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_control.simplemem_candidate (
  candidate_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  query_sha256 text NOT NULL CHECK (query_sha256 ~ '^[0-9a-f]{64}$'),
  source_ref text NOT NULL,
  candidate_text text NOT NULL,
  recall_score numeric NOT NULL DEFAULT 0,
  not_truth boolean NOT NULL DEFAULT true CHECK (not_truth = true),
  promotion_allowed boolean NOT NULL DEFAULT false CHECK (promotion_allowed = false),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS simplemem_candidate_query_idx ON lucidota_control.simplemem_candidate(query_sha256, recall_score DESC);

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
VALUES ('boring_beast', 'Boring Beast E2E runtime work loop', 2, 'Executable work-loop validation queue; no canonical graph mutation outside promotion path.')
ON CONFLICT (queue_name) DO UPDATE SET owner_subsystem=EXCLUDED.owner_subsystem, max_attempts=EXCLUDED.max_attempts, notes=EXCLUDED.notes, updated_at=now();

COMMIT;
