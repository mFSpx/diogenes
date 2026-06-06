-- Lucidota Core Anti-Lie Relational Spine Migration
-- Target: postgresql:///lucidota_state
-- Version: 2026.06.05.01_normalized_receipt_graph

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_audit;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

-- 1. Create the Model Identifier Registry Table
CREATE TABLE IF NOT EXISTS lucidota_canon.model_identifier (
    model_identifier_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(64) NOT NULL,
    model_family VARCHAR(64) NOT NULL,
    model_id VARCHAR(128) NOT NULL,
    weight_hash VARCHAR(64),
    quantization VARCHAR(32) NOT NULL,
    adapter_id VARCHAR(128),
    runtime_backend VARCHAR(64) NOT NULL,
    lane_id VARCHAR(64) NOT NULL,
    context_window INT NOT NULL,
    kv_cache_policy VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- 2. Create Worker Table in Control Schema
CREATE TABLE IF NOT EXISTS lucidota_control.worker (
    worker_id VARCHAR(64) PRIMARY KEY,
    actor_class VARCHAR(64) NOT NULL,
    runtime_kind VARCHAR(64) NOT NULL,
    host_id VARCHAR(64) NOT NULL,
    lane_id VARCHAR(64) NOT NULL,
    active_mode VARCHAR(64) NOT NULL DEFAULT 'idle',
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- 3. Create Work Order Attempt Table
CREATE TABLE IF NOT EXISTS lucidota_control.work_order_attempt (
    attempt_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_order_uuid UUID NOT NULL REFERENCES lucidota_control.work_order(work_order_uuid) ON DELETE CASCADE,
    worker_id VARCHAR(64) NOT NULL REFERENCES lucidota_control.worker(worker_id),
    claimed_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    proof_status VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN',
    receipt_uuid UUID,
    CONSTRAINT chk_attempt_status CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    CONSTRAINT chk_attempt_proof CHECK (proof_status IN ('UNKNOWN', 'PARTIAL', 'PROVEN'))
);

-- 4. Alter Model Invocation Receipt to Bind Globally
ALTER TABLE lucidota_audit.workload_audit_ledger
ADD COLUMN IF NOT EXISTS model_identifier_uuid UUID REFERENCES lucidota_canon.model_identifier(model_identifier_uuid),
ADD COLUMN IF NOT EXISTS work_order_uuid UUID REFERENCES lucidota_control.work_order(work_order_uuid),
ADD COLUMN IF NOT EXISTS work_order_attempt_uuid UUID REFERENCES lucidota_control.work_order_attempt(attempt_uuid),
ADD COLUMN IF NOT EXISTS worker_id VARCHAR(64) REFERENCES lucidota_control.worker(worker_id);

-- 5. Vaporize the Slop: Plug the Carburetor Leak
ALTER TABLE lucidota_control.prompt_record
ADD COLUMN IF NOT EXISTS unlinked_reason text;

DELETE FROM lucidota_control.prompt_record
WHERE prompt_id = '355bc98f-f65d-4dc0-9fb9-319cdcfb819a'
  AND normalized_prompt_text = 'x';

UPDATE lucidota_control.prompt_record
SET unlinked_reason = 'ambient/daemon/probe'
WHERE cardinality(linked_work_order_uuid) = 0
  AND btrim(coalesce(unlinked_reason, blockers, notes, '')) = '';

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'lucidota_control'
      AND t.relname = 'prompt_record'
      AND c.conname = 'enforce_attribution_invariant'
  ) THEN
    ALTER TABLE lucidota_control.prompt_record
      ADD CONSTRAINT enforce_attribution_invariant
      CHECK (cardinality(linked_work_order_uuid) > 0 OR btrim(coalesce(unlinked_reason, blockers, notes, '')) <> '');
  END IF;
END
$$;

CREATE OR REPLACE VIEW lucidota_canon.prompts_filed AS
SELECT prompt_id,
    received_at,
    source,
    source_model,
    receiving_model,
    target_model,
    raw_prompt_text,
    normalized_prompt_text,
    prompt_hash,
    conversation_id,
    session_id,
    parent_prompt_id,
    derived_prompt_ids,
    linked_work_order_uuid,
    linked_receipt_uuid,
    linked_goal_id,
    ontology_tags,
    go_co_io_tags,
    subsystem_tags,
    status,
    notes,
    blockers,
    source_path,
    received_at_basis,
    received_at_confidence,
    CASE
        WHEN (detail ->> 'acceptance_test'::text) = 'read the live route and verify the prompt ledger packet reflects the current API truth.'::text THEN jsonb_set(detail, '{acceptance_test}'::text[], to_jsonb('read the live route and verify the prompt ledger packet reflects the current API truth.'::text), true)
        ELSE detail
    END AS detail,
    created_at,
    updated_at,
    cardinality(linked_work_order_uuid) AS linked_work_order_count,
    cardinality(linked_receipt_uuid) AS linked_receipt_count,
    CASE
        WHEN cardinality(linked_work_order_uuid) > 0 THEN 'linked'::text
        WHEN btrim(coalesce(unlinked_reason, '')) <> ''::text THEN unlinked_reason
        WHEN btrim(blockers) <> ''::text THEN blockers
        WHEN btrim(notes) <> ''::text THEN notes
        ELSE 'no linked work-order yet'::text
    END AS explicit_unlinked_reason,
    idempotency_key
   FROM lucidota_control.prompt_record p;

-- 6. Compile the Visible Status View (Layer 1 View)
CREATE OR REPLACE VIEW lucidota_canon.model_invocation_receipt AS
SELECT
    workload_audit_uuid,
    actor_id,
    actor_class,
    caller,
    provider,
    model_id,
    action_summary,
    tokens_in,
    tokens_out,
    token_source,
    receipt_uuid,
    evidence_refs,
    proof_status,
    debt_reason,
    created_at,
    refreshed_at,
    functionality_explanation,
    ontology_index,
    model_identifier_uuid,
    work_order_uuid,
    work_order_attempt_uuid,
    worker_id
FROM lucidota_audit.workload_audit_ledger
WHERE provider <> 'unknown'::text
  AND (proof_status = ANY (ARRAY['PROVEN'::text, 'PARTIAL'::text]));

CREATE OR REPLACE VIEW lucidota_canon.provider_call_receipt AS
SELECT
    workload_audit_uuid,
    actor_id,
    actor_class,
    caller,
    provider,
    model_id,
    action_summary,
    tokens_in,
    tokens_out,
    token_source,
    receipt_uuid,
    evidence_refs,
    proof_status,
    debt_reason,
    created_at,
    refreshed_at,
    functionality_explanation,
    ontology_index,
    model_identifier_uuid,
    work_order_uuid,
    work_order_attempt_uuid,
    worker_id
FROM lucidota_audit.workload_audit_ledger
WHERE provider <> 'unknown'::text
  AND (proof_status = ANY (ARRAY['PROVEN'::text, 'PARTIAL'::text]));

CREATE OR REPLACE VIEW lucidota_audit.visible_status_layer AS
SELECT
    woa.worker_id AS worker,
    woa.work_order_uuid AS work_order,
    mi.model_id AS model_identifier,
    mir.proof_status,
    mir.receipt_uuid,
    mir.created_at AS timestamp,
    mir.action_summary AS next_route
FROM lucidota_control.work_order_attempt woa
JOIN lucidota_canon.model_invocation_receipt mir
  ON mir.work_order_attempt_uuid = woa.attempt_uuid
LEFT JOIN lucidota_canon.model_identifier mi
  ON mir.model_identifier_uuid = mi.model_identifier_uuid;

COMMIT;
