-- Lucidota prompt spine hardening.
-- Keep the visible prompt surface and the underlying prompt record honest.
-- This is runtime/database truth, not canon-storytelling.

BEGIN;

-- The physical prompt ledger lives in lucidota_control.prompt_record.
-- Enforce the explicit-link-or-null-reason invariant with the hardcoded
-- ambient/daemon/probe escape hatch only.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'lucidota_control'
          AND t.relname = 'prompt_record'
          AND c.conname = 'enforce_attribution_invariant'
    ) THEN
        ALTER TABLE lucidota_control.prompt_record
            DROP CONSTRAINT enforce_attribution_invariant;
    END IF;
END
$$;

ALTER TABLE lucidota_control.prompt_record
    ADD CONSTRAINT enforce_attribution_invariant
    CHECK (
        cardinality(linked_work_order_uuid) > 0
        OR unlinked_reason = 'ambient/daemon/probe'
    );

-- Make the hardcoded null reason explicit on the stored rows.
UPDATE lucidota_control.prompt_record
SET unlinked_reason = 'ambient/daemon/probe'
WHERE cardinality(linked_work_order_uuid) = 0
  AND btrim(coalesce(unlinked_reason, blockers, notes, '')) = '';

-- Keep the visible prompt view compact but explicit.
CREATE OR REPLACE VIEW lucidota_canon.prompts_filed AS
SELECT
    prompt_id,
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
    detail,
    created_at,
    updated_at,
    cardinality(linked_work_order_uuid) AS linked_work_order_count,
    cardinality(linked_receipt_uuid) AS linked_receipt_count,
    CASE
        WHEN cardinality(linked_work_order_uuid) > 0 THEN 'linked'::text
        ELSE COALESCE(unlinked_reason, 'ambient/daemon/probe')
    END AS explicit_unlinked_reason,
    idempotency_key,
    COALESCE(linked_work_order_uuid[1], NULL)::uuid AS work_order_uuid,
    CASE
        WHEN cardinality(linked_work_order_uuid) > 0 THEN NULL::text
        ELSE COALESCE(unlinked_reason, 'ambient/daemon/probe')
    END AS null_reason
FROM lucidota_control.prompt_record p;

-- Surface the null reason on the invocation view so the trace path stays readable.
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
    worker_id,
    null_reason
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
    worker_id,
    null_reason
FROM lucidota_audit.workload_audit_ledger
WHERE provider <> 'unknown'::text
  AND (proof_status = ANY (ARRAY['PROVEN'::text, 'PARTIAL'::text]));

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id,
    canonical_owner,
    packet_class,
    surface_kind,
    approval_required,
    notes,
    detail,
    approved_by,
    approved_at,
    approval_receipt_uuid,
    approval_note
) VALUES
    (
        'prompts_filed',
        'lucidota_canon',
        'typed_packet',
        'view',
        true,
        'Prompt ledger visible surface with explicit work_order_uuid and ambient null_reason.',
        '{"source":"prompt_record"}'::jsonb,
        'mfspx',
        now(),
        gen_random_uuid(),
        'reclassify prompts_filed from route to view with explicit prompt-record hardening'
    ),
    (
        'model_invocation_receipt',
        'lucidota_canon',
        'typed_packet',
        'view',
        true,
        'Invocation trace surface with explicit null_reason for debt rows.',
        '{"source":"workload_audit_ledger"}'::jsonb,
        'mfspx',
        now(),
        gen_random_uuid(),
        'append null_reason to the invocation trace surface'
    ),
    (
        'provider_call_receipt',
        'lucidota_canon',
        'typed_packet',
        'view',
        true,
        'Provider trace surface with explicit null_reason for debt rows.',
        '{"source":"workload_audit_ledger"}'::jsonb,
        'mfspx',
        now(),
        gen_random_uuid(),
        'append null_reason to the provider trace surface'
    )
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    approved_by = EXCLUDED.approved_by,
    approved_at = EXCLUDED.approved_at,
    approval_receipt_uuid = EXCLUDED.approval_receipt_uuid,
    approval_note = EXCLUDED.approval_note,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id,
    method,
    path_pattern,
    description,
    target,
    sample_request,
    sample_response,
    status
) VALUES
    (
        'prompts_filed',
        'GET',
        '/prompts_filed',
        'Prompt ledger visible surface with explicit work_order_uuid and null_reason.',
        'lucidota_canon.prompts_filed',
        '{"limit":"1"}',
        '{"prompt_id":"...","work_order_uuid":"...","null_reason":"ambient/daemon/probe"}',
        'implemented'
    ),
    (
        'model_invocation_receipt',
        'GET',
        '/model_invocation_receipt',
        'Invocation trace surface with explicit null_reason.',
        'lucidota_canon.model_invocation_receipt',
        '{"limit":"1"}',
        '{"receipt_uuid":"...","null_reason":"ambient/daemon/probe"}',
        'implemented'
    ),
    (
        'provider_call_receipt',
        'GET',
        '/provider_call_receipt',
        'Provider trace surface with explicit null_reason.',
        'lucidota_canon.provider_call_receipt',
        '{"limit":"1"}',
        '{"receipt_uuid":"...","null_reason":"ambient/daemon/probe"}',
        'implemented'
    )
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

GRANT SELECT ON lucidota_canon.prompts_filed, lucidota_canon.model_invocation_receipt, lucidota_canon.provider_call_receipt TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
