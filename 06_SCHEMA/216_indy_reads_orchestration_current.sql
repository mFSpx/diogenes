-- Indy_READs orchestration intent current surface.
-- This makes the current provider/model lane visible in Postgres/PostgREST.

BEGIN;

CREATE TABLE IF NOT EXISTS lucidota_indy.indy_reads_orchestration_intent_state (
    state_key text PRIMARY KEY DEFAULT 'indy_reads_orchestration_intent',
    actor_id text NOT NULL DEFAULT 'INDY_READs',
    provider_key text NOT NULL DEFAULT 'local_model',
    provider_kind text NOT NULL DEFAULT 'local_runtime',
    workload_type text NOT NULL DEFAULT 'orchestration',
    model_id text NOT NULL DEFAULT 'bonsai_q1_0',
    model_family text NOT NULL DEFAULT 'bonsai',
    role text NOT NULL DEFAULT 'big_brain_orchestration',
    takeover_mode boolean NOT NULL DEFAULT FALSE,
    fallback_provider_key text NOT NULL DEFAULT 'local_model',
    fallback_model_id text NOT NULL DEFAULT 'bonsai_q1_0',
    source text NOT NULL DEFAULT 'operator',
    notes text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT indy_reads_orchestration_intent_provider_key_check CHECK (btrim(provider_key) <> ''),
    CONSTRAINT indy_reads_orchestration_intent_model_id_check CHECK (btrim(model_id) <> ''),
    CONSTRAINT indy_reads_orchestration_intent_role_check CHECK (btrim(role) <> ''),
    CONSTRAINT indy_reads_orchestration_intent_workload_type_check CHECK (workload_type IN ('orchestration')),
    CONSTRAINT indy_reads_orchestration_intent_takeover_mode_check CHECK (takeover_mode IN (TRUE, FALSE))
);

CREATE INDEX IF NOT EXISTS indy_reads_orchestration_intent_state_updated_idx
    ON lucidota_indy.indy_reads_orchestration_intent_state (updated_at DESC);

INSERT INTO lucidota_indy.indy_reads_orchestration_intent_state (
    state_key,
    actor_id,
    provider_key,
    provider_kind,
    workload_type,
    model_id,
    model_family,
    role,
    takeover_mode,
    fallback_provider_key,
    fallback_model_id,
    source,
    notes,
    created_at,
    updated_at
)
VALUES (
    'indy_reads_orchestration_intent',
    'INDY_READs',
    'local_model',
    'local_runtime',
    'orchestration',
    'bonsai_q1_0',
    'bonsai',
    'big_brain_orchestration',
    FALSE,
    'local_model',
    'bonsai_q1_0',
    'bootstrap_default',
    'Local Bonsai is the default; cloud lanes are explicit orchestration intents, not takeover.',
    now(),
    now()
)
ON CONFLICT (state_key) DO UPDATE SET
    actor_id = EXCLUDED.actor_id,
    provider_key = EXCLUDED.provider_key,
    provider_kind = EXCLUDED.provider_kind,
    workload_type = EXCLUDED.workload_type,
    model_id = EXCLUDED.model_id,
    model_family = EXCLUDED.model_family,
    role = EXCLUDED.role,
    takeover_mode = EXCLUDED.takeover_mode,
    fallback_provider_key = EXCLUDED.fallback_provider_key,
    fallback_model_id = EXCLUDED.fallback_model_id,
    source = EXCLUDED.source,
    notes = EXCLUDED.notes,
    updated_at = now();

CREATE OR REPLACE VIEW lucidota_canon.indy_reads_orchestration_current AS
SELECT
    state_key,
    actor_id,
    provider_key,
    provider_kind,
    workload_type,
    model_id,
    model_family,
    role,
    takeover_mode,
    fallback_provider_key,
    fallback_model_id,
    source,
    notes,
    created_at,
    updated_at,
    jsonb_build_object(
        'state_key', state_key,
        'actor_id', actor_id,
        'provider_key', provider_key,
        'provider_kind', provider_kind,
        'workload_type', workload_type,
        'model_id', model_id,
        'model_family', model_family,
        'role', role,
        'takeover_mode', takeover_mode,
        'fallback_provider_key', fallback_provider_key,
        'fallback_model_id', fallback_model_id,
        'source', source,
        'notes', notes,
        'summary', CASE
            WHEN provider_key = 'local_model' THEN 'local ' || model_id || ' (fallback ' || fallback_model_id || ')'
            WHEN takeover_mode THEN provider_key || '::' || model_id || ' (takeover; fallback ' || fallback_model_id || ')'
            ELSE provider_key || '::' || model_id || ' (orchestration-only; fallback ' || fallback_model_id || ')'
        END,
        'created_at', created_at,
        'updated_at', updated_at,
        'proof_status', 'PROVEN'
    ) AS packet
FROM lucidota_indy.indy_reads_orchestration_intent_state
ORDER BY updated_at DESC
LIMIT 1;

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id,
    canonical_owner,
    packet_class,
    surface_kind,
    approval_required,
    notes,
    detail
)
VALUES (
    'indy_reads_orchestration_current',
    'lucidota_indy',
    'typed_packet',
    'view',
    true,
    'Indy_READs orchestration intent current packet.',
    '{"source":"indy_reads_orchestration_current"}'::jsonb
)
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
)
VALUES (
    'indy_reads_orchestration_current',
    'GET',
    '/indy_reads_orchestration_current',
    'Indy_READs orchestration intent packet.',
    'lucidota_canon.indy_reads_orchestration_current',
    '{"limit":"1"}',
    '{"state_key":"indy_reads_orchestration_intent"}',
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

GRANT SELECT ON lucidota_indy.indy_reads_orchestration_intent_state TO mfspx, lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.indy_reads_orchestration_current TO mfspx, lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.api_route_catalog TO mfspx, lucidota_postgrest_anon;

NOTIFY pgrst, 'reload schema';

COMMIT;
