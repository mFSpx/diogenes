-- RAC mode truth and workload telemetry surfaces.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.active_operation_mode_state (
    mode_key text PRIMARY KEY DEFAULT 'active_operation_mode',
    current_mode text NOT NULL,
    cloud_policy text NOT NULL,
    swarm_policy text NOT NULL,
    indy_reads_policy text NOT NULL,
    receipt_policy text NOT NULL,
    runtime_default_policy text NOT NULL,
    build_session_policy text NOT NULL,
    operator_override text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    functionality_explanation text NOT NULL,
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT active_operation_mode_state_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT active_operation_mode_state_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object'),
    CONSTRAINT active_operation_mode_state_current_mode_check CHECK (current_mode IN ('BUILD_SWARM_MODE', 'RUNTIME_FAST_PATH', 'RESTITUTION_LOCKDOWN', 'UNKNOWN'))
);

INSERT INTO lucidota_control.active_operation_mode_state (
    mode_key,
    current_mode,
    cloud_policy,
    swarm_policy,
    indy_reads_policy,
    receipt_policy,
    runtime_default_policy,
    build_session_policy,
    operator_override,
    updated_at,
    evidence_refs,
    functionality_explanation,
    ontology_index
) VALUES (
    'active_operation_mode',
    'BUILD_SWARM_MODE',
    'OPERATOR_ALLOWED_RECEIPT_REQUIRED',
    'OPERATOR_ALLOWED_RECEIPT_REQUIRED',
    'EXOCORTEX_ALLOWED_RECEIPT_REQUIRED',
    'DB_RECEIPT_OR_UNKNOWN_DEBT',
    'RUNTIME_FAST_PATH_AFTER_BUILD',
    'BUILD_SWARM_MODE',
    'RAC_ON',
    now(),
    '["AGENTS.md","CLAUDE.md","GOALS/AGENT_ORCHESTRATION_POLICY.md","GOALS/CURRENT_HANDOFF.md","GOALS/GOAL_LOG.md"]'::jsonb,
    'Declares the active build/race operating mode and receipt gate policies so claims can be checked against live DB truth instead of prose.',
    '{
        "primitive_refs": ["state", "duplex", "allocation"],
        "claim_type": "mode_truth",
        "evidence_type": "instruction_and_receipt",
        "actor_role": "control_plane",
        "subsystem_refs": ["manual_current", "root_orchestrator_current", "workload_audit_current"],
        "risk_tier": "T3",
        "proof_status": "PROVEN",
        "receipt_refs": ["workload_audit_restitution_gate", "rac_reignition_gate"],
        "next_route": ["manual_current", "root_orchestrator_current", "workload_audit_current"]
    }'::jsonb
) ON CONFLICT (mode_key) DO UPDATE SET
    current_mode = EXCLUDED.current_mode,
    cloud_policy = EXCLUDED.cloud_policy,
    swarm_policy = EXCLUDED.swarm_policy,
    indy_reads_policy = EXCLUDED.indy_reads_policy,
    receipt_policy = EXCLUDED.receipt_policy,
    runtime_default_policy = EXCLUDED.runtime_default_policy,
    build_session_policy = EXCLUDED.build_session_policy,
    operator_override = EXCLUDED.operator_override,
    updated_at = EXCLUDED.updated_at,
    evidence_refs = EXCLUDED.evidence_refs,
    functionality_explanation = EXCLUDED.functionality_explanation,
    ontology_index = EXCLUDED.ontology_index;

CREATE OR REPLACE VIEW lucidota_control.active_operation_mode AS
SELECT
    mode_key,
    current_mode,
    cloud_policy,
    swarm_policy,
    indy_reads_policy,
    receipt_policy,
    runtime_default_policy,
    build_session_policy,
    operator_override,
    updated_at,
    evidence_refs,
    functionality_explanation,
    ontology_index
FROM lucidota_control.active_operation_mode_state
ORDER BY updated_at DESC
LIMIT 1;

CREATE OR REPLACE VIEW lucidota_canon.active_operation_mode AS
SELECT
    mode_key,
    current_mode,
    cloud_policy,
    swarm_policy,
    indy_reads_policy,
    receipt_policy,
    runtime_default_policy,
    build_session_policy,
    operator_override,
    updated_at,
    evidence_refs,
    functionality_explanation,
    ontology_index
FROM lucidota_control.active_operation_mode;

CREATE OR REPLACE VIEW lucidota_canon.workload_audit_telemetry_current AS
WITH workload AS (
    SELECT * FROM lucidota_audit.workload_audit_ledger
),
totals AS (
    SELECT
        COUNT(*) AS ledger_row_count,
        COUNT(*) FILTER (WHERE receipt_uuid IS NOT NULL) AS receipt_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'PROVEN') AS proven_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'PARTIAL') AS partial_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'UNKNOWN') AS unknown_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'CONTRADICTED') AS contradicted_row_count,
        COUNT(*) FILTER (WHERE actor_class IN ('codex_main', 'codex_agent')) AS codex_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'indy_reads') AS indy_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'local_llm') AS local_llm_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'groq') AS groq_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'gemini') AS gemini_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'gemini_paid') AS gemini_paid_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'vibe') AS vibe_row_count,
        COUNT(*) FILTER (WHERE provider = 'unknown') AS unknown_provider_row_count,
        COALESCE(SUM(tokens_in) FILTER (WHERE tokens_in IS NOT NULL), 0) AS tokens_in_total,
        COALESCE(SUM(tokens_out) FILTER (WHERE tokens_out IS NOT NULL), 0) AS tokens_out_total
    FROM workload
),
provider_counts AS (
    SELECT COALESCE(jsonb_object_agg(provider, provider_count ORDER BY provider), '{}'::jsonb) AS provider_counts
    FROM (
        SELECT provider, COUNT(*) AS provider_count
        FROM workload
        GROUP BY provider
    ) p
),
actor_counts AS (
    SELECT COALESCE(jsonb_object_agg(actor_class, actor_count ORDER BY actor_class), '{}'::jsonb) AS actor_class_counts
    FROM (
        SELECT actor_class, COUNT(*) AS actor_count
        FROM workload
        GROUP BY actor_class
    ) a
),
caller_counts AS (
    SELECT COALESCE(jsonb_object_agg(caller, caller_count ORDER BY caller), '{}'::jsonb) AS caller_counts
    FROM (
        SELECT caller, COUNT(*) AS caller_count
        FROM workload
        GROUP BY caller
    ) c
),
proof_counts AS (
    SELECT COALESCE(jsonb_object_agg(proof_status, proof_count ORDER BY proof_status), '{}'::jsonb) AS proof_status_counts
    FROM (
        SELECT proof_status, COUNT(*) AS proof_count
        FROM workload
        GROUP BY proof_status
    ) p
)
SELECT
    'WORKLOAD_AUDIT_TELEMETRY_CURRENT'::text AS telemetry_id,
    now() AS refreshed_at,
    totals.ledger_row_count,
    totals.receipt_row_count,
    totals.proven_row_count,
    totals.partial_row_count,
    totals.unknown_row_count,
    totals.contradicted_row_count,
    totals.codex_row_count,
    totals.indy_row_count,
    totals.local_llm_row_count,
    totals.groq_row_count,
    totals.gemini_row_count,
    totals.gemini_paid_row_count,
    totals.vibe_row_count,
    totals.unknown_provider_row_count,
    totals.tokens_in_total,
    totals.tokens_out_total,
    provider_counts.provider_counts,
    actor_counts.actor_class_counts,
    caller_counts.caller_counts,
    proof_counts.proof_status_counts,
    jsonb_build_object(
        'cloud_lane_row_count', totals.groq_row_count + totals.gemini_row_count + totals.gemini_paid_row_count + totals.vibe_row_count,
        'local_lane_row_count', totals.local_llm_row_count,
        'codex_lane_row_count', totals.codex_row_count,
        'indy_lane_row_count', totals.indy_row_count,
        'unknown_row_count', totals.unknown_row_count,
        'receipt_row_count', totals.receipt_row_count
    ) AS lane_telemetry,
    jsonb_build_object(
        'active_operation_mode', 'lucidota_control.active_operation_mode',
        'workload_audit_current', 'lucidota_canon.workload_audit_current',
        'provider_call_receipt', 'lucidota_canon.provider_call_receipt',
        'model_invocation_receipt', 'lucidota_canon.model_invocation_receipt',
        'agent_work_receipt', 'lucidota_canon.agent_work_receipt',
        'unproven_work_debt', 'lucidota_canon.unproven_work_debt'
    ) AS route_refs,
    jsonb_build_object(
        'gemini_present', totals.gemini_row_count > 0,
        'vibe_present', totals.vibe_row_count > 0,
        'codex_present', totals.codex_row_count > 0,
        'indy_present', totals.indy_row_count > 0,
        'unknown_debt_present', totals.unknown_row_count > 0
    ) AS telemetry_flags
FROM totals
CROSS JOIN provider_counts
CROSS JOIN actor_counts
CROSS JOIN caller_counts
CROSS JOIN proof_counts;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('active_operation_mode', 'lucidota_control', 'typed_packet', 'view', true, 'Current build/race mode and receipt policy surface.', '{"source":"rac_reignition"}'::jsonb),
    ('workload_audit_telemetry_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Compact workload/provider telemetry summary over the audit ledger.', '{"source":"rac_reignition"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, status
) VALUES
    ('active_operation_mode', 'GET', '/active_operation_mode', 'Current build/race mode and receipt policy packet.', 'lucidota_canon.active_operation_mode', 'implemented'),
    ('workload_audit_telemetry_current', 'GET', '/workload_audit_telemetry_current', 'Current workload/provider telemetry summary packet.', 'lucidota_canon.workload_audit_telemetry_current', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    status = EXCLUDED.status;

GRANT SELECT ON lucidota_control.active_operation_mode TO mfspx, lucidota_postgrest_anon, ironclaw;
GRANT SELECT ON lucidota_canon.active_operation_mode TO mfspx, lucidota_postgrest_anon, ironclaw;
GRANT SELECT ON lucidota_canon.workload_audit_telemetry_current TO mfspx, lucidota_postgrest_anon, ironclaw;
GRANT USAGE ON SCHEMA lucidota_control TO mfspx, lucidota_postgrest_anon, ironclaw;

NOTIFY pgrst, 'reload schema';

COMMIT;
