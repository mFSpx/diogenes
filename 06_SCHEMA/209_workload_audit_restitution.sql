-- DB-backed workload accounting spine: receipts, debt, and compact current summary.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_audit;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_audit.workload_audit_ledger (
    workload_audit_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id text NOT NULL,
    actor_class text NOT NULL,
    caller text NOT NULL,
    provider text NOT NULL DEFAULT 'unknown',
    model_id text NOT NULL DEFAULT '',
    action_summary text NOT NULL,
    tokens_in bigint,
    tokens_out bigint,
    token_source text NOT NULL DEFAULT 'unknown',
    receipt_uuid uuid,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    proof_status text NOT NULL,
    debt_reason text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    refreshed_at timestamptz NOT NULL DEFAULT now(),
    functionality_explanation text NOT NULL,
    ontology_index jsonb NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT workload_audit_actor_class_check CHECK (actor_class IN (
        'codex_main',
        'codex_agent',
        'indy_reads',
        'local_llm',
        'groq',
        'gemini',
        'gemini_paid',
        'vibe',
        'unknown'
    )),
    CONSTRAINT workload_audit_caller_check CHECK (caller IN ('codex', 'indy_reads', 'operator', 'daemon', 'unknown')),
    CONSTRAINT workload_audit_token_source_check CHECK (token_source IN ('provider_api', 'local_counter', 'receipt_file', 'manual_operator_input', 'unknown')),
    CONSTRAINT workload_audit_proof_status_check CHECK (proof_status IN ('PROVEN', 'PARTIAL', 'UNKNOWN', 'CONTRADICTED')),
    CONSTRAINT workload_audit_tokens_in_check CHECK (tokens_in IS NULL OR tokens_in >= 0),
    CONSTRAINT workload_audit_tokens_out_check CHECK (tokens_out IS NULL OR tokens_out >= 0),
    CONSTRAINT workload_audit_evidence_refs_check CHECK (jsonb_typeof(evidence_refs) = 'array'),
    CONSTRAINT workload_audit_ontology_index_check CHECK (jsonb_typeof(ontology_index) = 'object')
);

CREATE INDEX IF NOT EXISTS workload_audit_ledger_actor_idx
    ON lucidota_audit.workload_audit_ledger (actor_class, caller, refreshed_at DESC);

CREATE INDEX IF NOT EXISTS workload_audit_ledger_proof_idx
    ON lucidota_audit.workload_audit_ledger (proof_status, refreshed_at DESC);

CREATE INDEX IF NOT EXISTS workload_audit_ledger_provider_idx
    ON lucidota_audit.workload_audit_ledger (provider, model_id, refreshed_at DESC);

CREATE INDEX IF NOT EXISTS workload_audit_ledger_receipt_idx
    ON lucidota_audit.workload_audit_ledger (receipt_uuid)
    WHERE receipt_uuid IS NOT NULL;

CREATE OR REPLACE VIEW lucidota_canon.workload_audit_ledger AS
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
    ontology_index
FROM lucidota_audit.workload_audit_ledger
ORDER BY refreshed_at DESC, actor_class ASC, caller ASC, actor_id ASC;

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
    ontology_index
FROM lucidota_canon.workload_audit_ledger
WHERE provider IN ('groq', 'gemini', 'gemini_paid', 'vibe', 'cohere')
  AND proof_status IN ('PROVEN', 'PARTIAL');

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
    ontology_index
FROM lucidota_canon.workload_audit_ledger
WHERE provider <> 'unknown'
  AND proof_status IN ('PROVEN', 'PARTIAL');

CREATE OR REPLACE VIEW lucidota_canon.agent_work_receipt AS
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
    ontology_index
FROM lucidota_canon.workload_audit_ledger
WHERE actor_class IN ('codex_main', 'codex_agent', 'indy_reads')
  AND proof_status IN ('PROVEN', 'PARTIAL');

CREATE OR REPLACE VIEW lucidota_canon.unproven_work_debt AS
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
    ontology_index
FROM lucidota_canon.workload_audit_ledger
WHERE proof_status = 'UNKNOWN';

CREATE OR REPLACE VIEW lucidota_canon.workload_audit_current AS
WITH workload AS (
    SELECT * FROM lucidota_canon.workload_audit_ledger
),
stats AS (
    SELECT
        COUNT(*) AS ledger_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'PROVEN') AS proven_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'PARTIAL') AS partial_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'UNKNOWN') AS unknown_row_count,
        COUNT(*) FILTER (WHERE proof_status = 'CONTRADICTED') AS contradicted_row_count,
        COUNT(*) FILTER (WHERE actor_class IN ('codex_main', 'codex_agent') AND proof_status IN ('PROVEN', 'PARTIAL')) AS codex_row_count,
        COUNT(*) FILTER (WHERE actor_class = 'indy_reads' AND proof_status IN ('PROVEN', 'PARTIAL')) AS indy_row_count
    FROM workload
),
actor_summary_rows AS (
    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'actor_class', actor_class,
                'caller', caller,
                'proven_work', proven_work,
                'tokens_in', tokens_in,
                'tokens_out', tokens_out,
                'proof_status', proof_status,
                'debt_reason', debt_reason,
                'receipt_uuid', receipt_uuid
            )
            ORDER BY actor_class, caller
        ),
        '[]'::jsonb
    ) AS actor_summary
    FROM (
        SELECT
            actor_class,
            caller,
            COUNT(*) FILTER (WHERE proof_status IN ('PROVEN', 'PARTIAL')) AS proven_work,
            COALESCE(SUM(tokens_in) FILTER (WHERE tokens_in IS NOT NULL), 0) AS tokens_in,
            COALESCE(SUM(tokens_out) FILTER (WHERE tokens_out IS NOT NULL), 0) AS tokens_out,
            CASE
                WHEN bool_or(proof_status = 'CONTRADICTED') THEN 'CONTRADICTED'
                WHEN bool_or(proof_status = 'UNKNOWN') THEN 'UNKNOWN'
                WHEN bool_or(proof_status = 'PARTIAL') THEN 'PARTIAL'
                ELSE 'PROVEN'
            END AS proof_status,
            CASE
                WHEN bool_or(proof_status = 'UNKNOWN') THEN 'no receipt-backed workload/token evidence'
                ELSE ''
            END AS debt_reason,
            MAX(receipt_uuid::text) AS receipt_uuid
        FROM workload
        GROUP BY actor_class, caller
    ) grouped
)
SELECT
    CASE
        WHEN stats.contradicted_row_count > 0 THEN 'CONTRADICTED'
        WHEN stats.unknown_row_count > 0 AND stats.proven_row_count > 0 THEN 'PARTIAL'
        WHEN stats.unknown_row_count > 0 THEN 'UNKNOWN'
        WHEN stats.partial_row_count > 0 THEN 'PARTIAL'
        ELSE 'PROVEN'
    END AS audit_status,
    actor_summary_rows.actor_summary,
    (stats.unknown_row_count > 0) AS has_unacknowledged_unknown_rows,
    stats.unknown_row_count,
    stats.proven_row_count,
    stats.partial_row_count,
    stats.contradicted_row_count,
    (
        stats.unknown_row_count = 0
        AND stats.contradicted_row_count = 0
        AND stats.codex_row_count > 0
        AND stats.indy_row_count > 0
    ) AS can_claim_duplex_race,
    now() AS refreshed_at,
    stats.ledger_row_count
FROM stats
CROSS JOIN actor_summary_rows;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('workload_audit_ledger', 'lucidota_canon', 'typed_packet', 'view', true, 'Workload audit ledger canonical view.', '{"source":"workload_audit_restitution"}'::jsonb),
    ('workload_audit_current', 'lucidota_canon', 'typed_packet', 'view', true, 'Workload audit current summary packet.', '{"source":"workload_audit_restitution"}'::jsonb),
    ('provider_call_receipt', 'lucidota_canon', 'typed_packet', 'view', true, 'Provider call receipt lane over workload audit ledger.', '{"source":"workload_audit_restitution"}'::jsonb),
    ('model_invocation_receipt', 'lucidota_canon', 'typed_packet', 'view', true, 'Model invocation receipt lane over workload audit ledger.', '{"source":"workload_audit_restitution"}'::jsonb),
    ('agent_work_receipt', 'lucidota_canon', 'typed_packet', 'view', true, 'Agent work receipt lane over workload audit ledger.', '{"source":"workload_audit_restitution"}'::jsonb),
    ('unproven_work_debt', 'lucidota_canon', 'typed_packet', 'view', true, 'Unknown/debt rows for claims without receipt-backed accounting.', '{"source":"workload_audit_restitution"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    (
        'workload_audit_ledger',
        'GET',
        '/workload_audit_ledger',
        'DB-backed workload audit ledger rows with proof status, receipts, and debt markers.',
        'lucidota_canon.workload_audit_ledger',
        '{"limit":"1"}',
        '{"actor_class":"groq","proof_status":"PROVEN"}',
        'implemented'
    ),
    (
        'workload_audit_current',
        'GET',
        '/workload_audit_current',
        'Compact workload audit summary: counts, actor summary, and duplex race claim gate.',
        'lucidota_canon.workload_audit_current',
        '{"limit":"1"}',
        '{"audit_status":"PARTIAL","has_unacknowledged_unknown_rows":true}',
        'implemented'
    ),
    (
        'provider_call_receipt',
        'GET',
        '/provider_call_receipt',
        'Provider call receipts backed by workload audit ledger evidence.',
        'lucidota_canon.provider_call_receipt',
        '{"limit":"1"}',
        '{"provider":"groq","proof_status":"PROVEN"}',
        'implemented'
    ),
    (
        'model_invocation_receipt',
        'GET',
        '/model_invocation_receipt',
        'Model invocation receipts backed by workload audit ledger evidence.',
        'lucidota_canon.model_invocation_receipt',
        '{"limit":"1"}',
        '{"provider":"local","proof_status":"PROVEN"}',
        'implemented'
    ),
    (
        'agent_work_receipt',
        'GET',
        '/agent_work_receipt',
        'Agent work receipts for Codex/Indy receipt-backed work.',
        'lucidota_canon.agent_work_receipt',
        '{"limit":"1"}',
        '{"actor_class":"indy_reads","proof_status":"PARTIAL"}',
        'implemented'
    ),
    (
        'unproven_work_debt',
        'GET',
        '/unproven_work_debt',
        'Debt rows for claims without receipt-backed workload/token evidence.',
        'lucidota_canon.unproven_work_debt',
        '{"limit":"1"}',
        '{"proof_status":"UNKNOWN","debt_reason":"no receipt-backed workload/token evidence"}',
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

GRANT SELECT ON lucidota_canon.workload_audit_ledger,
    lucidota_canon.workload_audit_current,
    lucidota_canon.provider_call_receipt,
    lucidota_canon.model_invocation_receipt,
    lucidota_canon.agent_work_receipt,
    lucidota_canon.unproven_work_debt
TO lucidota_postgrest_anon, mfspx, ironclaw;

GRANT USAGE ON SCHEMA lucidota_audit TO ironclaw;
GRANT SELECT, INSERT ON lucidota_audit.workload_audit_ledger TO ironclaw;

GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx, ironclaw;

NOTIFY pgrst, 'reload schema';

COMMIT;
