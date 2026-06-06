-- Typed control grants and per-thread runtime contracts.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.controller_grant (
    grant_key text PRIMARY KEY,
    grant_uuid uuid NOT NULL DEFAULT gen_random_uuid(),
    controller_name text NOT NULL,
    controller_kind text NOT NULL CHECK (controller_kind IN ('local','groq','vibe','codex','indy_reads','deterministic','runpod','external')),
    issued_by text NOT NULL DEFAULT 'operator',
    issued_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    revoked_at timestamptz,
    revocation_reason text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','revoked','expired','suspended')),
    allowed_envs text[] NOT NULL DEFAULT '{}'::text[],
    allowed_routes text[] NOT NULL DEFAULT '{}'::text[],
    allowed_commands text[] NOT NULL DEFAULT '{}'::text[],
    allowed_models text[] NOT NULL DEFAULT '{}'::text[],
    max_parallel_threads integer NOT NULL DEFAULT 1 CHECK (max_parallel_threads > 0),
    max_spend numeric(12,2) NOT NULL DEFAULT 0 CHECK (max_spend >= 0),
    receipt_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.agent_thread_runtime (
    thread_key text PRIMARY KEY,
    thread_uuid uuid NOT NULL DEFAULT gen_random_uuid(),
    parent_thread_key text REFERENCES lucidota_control.agent_thread_runtime(thread_key) ON DELETE SET NULL,
    controller_grant_key text REFERENCES lucidota_control.controller_grant(grant_key) ON DELETE SET NULL,
    thread_owner text NOT NULL,
    runtime_kind text NOT NULL CHECK (runtime_kind IN ('local','groq','vibe','codex','indy_reads','deterministic','runpod','external')),
    created_at timestamptz NOT NULL DEFAULT now(),
    model_policy jsonb NOT NULL DEFAULT '{}'::jsonb,
    env_identity jsonb NOT NULL DEFAULT '{}'::jsonb,
    budget_scope jsonb NOT NULL DEFAULT '{}'::jsonb,
    receipt_gate jsonb NOT NULL DEFAULT '{}'::jsonb,
    receipt_uuid uuid,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','completed','failed','archived','revoked')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS lucidota_control.controller_grant
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE IF EXISTS lucidota_control.agent_thread_runtime
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

CREATE OR REPLACE VIEW lucidota_canon.controller_grant AS
WITH rows AS (
    SELECT
        grant_key,
        grant_uuid,
        controller_name,
        controller_kind,
        issued_by,
        issued_at,
        expires_at,
        revoked_at,
        revocation_reason,
        CASE
            WHEN revoked_at IS NOT NULL THEN 'revoked'
            WHEN expires_at IS NOT NULL AND expires_at <= now() THEN 'expired'
            ELSE status
        END AS effective_status,
        status,
        allowed_envs,
        allowed_routes,
        allowed_commands,
        allowed_models,
        max_parallel_threads,
        max_spend,
        receipt_uuid,
        detail,
        updated_at,
        created_at
    FROM lucidota_control.controller_grant
)
SELECT
    grant_key,
    grant_uuid,
    controller_name,
    controller_kind,
    issued_by,
    issued_at,
    expires_at,
    revoked_at,
    revocation_reason,
    effective_status,
    status,
    allowed_envs,
    allowed_routes,
    allowed_commands,
    allowed_models,
    max_parallel_threads,
    max_spend,
    receipt_uuid,
    detail,
    updated_at,
    jsonb_build_object(
        'grant_key', grant_key,
        'grant_uuid', grant_uuid,
        'controller_name', controller_name,
        'controller_kind', controller_kind,
        'effective_status', effective_status,
        'allowed_envs', allowed_envs,
        'allowed_routes', allowed_routes,
        'allowed_commands', allowed_commands,
        'allowed_models', allowed_models,
        'max_parallel_threads', max_parallel_threads,
        'max_spend', max_spend,
        'created_at', created_at,
        'receipt_uuid', receipt_uuid
    ) AS packet,
    created_at
FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.agent_thread_runtime AS
WITH rows AS (
    SELECT
        thread_key,
        thread_uuid,
        parent_thread_key,
        controller_grant_key,
        thread_owner,
        runtime_kind,
        model_policy,
        env_identity,
        budget_scope,
        receipt_gate,
        receipt_uuid,
        status,
        detail,
        updated_at,
        created_at
    FROM lucidota_control.agent_thread_runtime
)
SELECT
    thread_key,
    thread_uuid,
    parent_thread_key,
    controller_grant_key,
    thread_owner,
    runtime_kind,
    model_policy,
    env_identity,
    budget_scope,
    receipt_gate,
    receipt_uuid,
    status,
    detail,
    updated_at,
    jsonb_build_object(
        'thread_key', thread_key,
        'thread_uuid', thread_uuid,
        'parent_thread_key', parent_thread_key,
        'controller_grant_key', controller_grant_key,
        'thread_owner', thread_owner,
        'runtime_kind', runtime_kind,
        'created_at', created_at,
        'model_policy', model_policy,
        'env_identity', env_identity,
        'budget_scope', budget_scope,
        'receipt_gate', receipt_gate,
        'receipt_uuid', receipt_uuid,
        'status', status
    ) AS packet,
    created_at
FROM rows;

INSERT INTO lucidota_control.controller_grant (
    grant_key,
    controller_name,
    controller_kind,
    issued_by,
    created_at,
    expires_at,
    revoked_at,
    revocation_reason,
    status,
    allowed_envs,
    allowed_routes,
    allowed_commands,
    allowed_models,
    max_parallel_threads,
    max_spend,
    receipt_uuid,
    detail
) VALUES (
    'default_local_operator',
    'luci operator shell',
    'local',
    'operator',
    now(),
    now() + interval '30 days',
    NULL,
    '',
    'active',
    ARRAY['pop_os','localhost','127.0.0.1'],
    ARRAY['manual_current','root_orchestrator_current','schema_owner_manifest','surface_registry','renderer_registry','controller_grant','agent_thread_runtime'],
    ARRAY['doctor','status','capability_list','capability_current','manual_current','root_orchestrator_current'],
    ARRAY['local','codex','groq'],
    4,
    0.00,
    NULL,
    jsonb_build_object(
        'purpose', 'typed operator grant for local control-plane orchestration',
        'receipt_gate', 'control_grant_runtime_spine',
        'bootstrap_local_only', true,
        'budget_enforced', false
    )
)
ON CONFLICT (grant_key) DO UPDATE SET
    controller_name = EXCLUDED.controller_name,
    controller_kind = EXCLUDED.controller_kind,
    issued_by = EXCLUDED.issued_by,
    created_at = EXCLUDED.created_at,
    expires_at = EXCLUDED.expires_at,
    revoked_at = EXCLUDED.revoked_at,
    revocation_reason = EXCLUDED.revocation_reason,
    status = EXCLUDED.status,
    allowed_envs = EXCLUDED.allowed_envs,
    allowed_routes = EXCLUDED.allowed_routes,
    allowed_commands = EXCLUDED.allowed_commands,
    allowed_models = EXCLUDED.allowed_models,
    max_parallel_threads = EXCLUDED.max_parallel_threads,
    max_spend = EXCLUDED.max_spend,
    receipt_uuid = EXCLUDED.receipt_uuid,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_control.agent_thread_runtime (
    thread_key,
    parent_thread_key,
    controller_grant_key,
    thread_owner,
    runtime_kind,
    created_at,
    model_policy,
    env_identity,
    budget_scope,
    receipt_gate,
    receipt_uuid,
    status,
    detail
) VALUES
    (
        'root_operator_thread',
        NULL,
        'default_local_operator',
        'operator',
        'local',
        now(),
        jsonb_build_object(
            'preferred', 'codex',
            'fallbacks', jsonb_build_array('groq', 'local'),
            'policy', 'runtime-per-thread'
        ),
        jsonb_build_object(
            'os', 'pop_os',
            'host', 'localhost',
            'db', 'postgresql:///lucidota_state'
        ),
        jsonb_build_object(
            'max_parallel_threads', 4,
            'max_spend', 1000.00,
            'currency', 'USD'
        ),
        jsonb_build_object(
            'required_receipt', 'control_grant_runtime_spine',
            'status', 'active',
            'bootstrap_local_only', true
        ),
        NULL,
        'active',
        jsonb_build_object(
            'purpose', 'root operator thread lease',
            'bootstrap_local_only', true,
            'budget_enforced', false
        )
    ),
    (
        'sub_orchestrator_thread',
        'root_operator_thread',
        'default_local_operator',
        'INDY_READs',
        'codex',
        now(),
        jsonb_build_object(
            'preferred', 'codex',
            'fallbacks', jsonb_build_array('groq', 'local'),
            'policy', 'typed-subthread'
        ),
        jsonb_build_object(
            'os', 'pop_os',
            'host', 'localhost',
            'controller', 'sub_orchestrator'
        ),
        jsonb_build_object(
            'max_parallel_threads', 2,
            'max_spend', 250.00,
            'currency', 'USD'
        ),
        jsonb_build_object(
            'required_receipt', 'control_grant_runtime_spine',
            'status', 'active',
            'bootstrap_local_only', true
        ),
        NULL,
        'active',
        jsonb_build_object(
            'purpose', 'typed child orchestration lease',
            'bootstrap_local_only', true,
            'budget_enforced', false
        )
    )
ON CONFLICT (thread_key) DO UPDATE SET
    parent_thread_key = EXCLUDED.parent_thread_key,
    controller_grant_key = EXCLUDED.controller_grant_key,
    thread_owner = EXCLUDED.thread_owner,
    runtime_kind = EXCLUDED.runtime_kind,
    created_at = EXCLUDED.created_at,
    model_policy = EXCLUDED.model_policy,
    env_identity = EXCLUDED.env_identity,
    budget_scope = EXCLUDED.budget_scope,
    receipt_gate = EXCLUDED.receipt_gate,
    receipt_uuid = EXCLUDED.receipt_uuid,
    status = EXCLUDED.status,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    (
        'controller_grant',
        'GET',
        '/controller_grant',
        'Typed controller grant surface with revoke, expiry, env, budget, and receipt fields.',
        'lucidota_canon.controller_grant',
        '{"grant_key":"default_local_operator"}',
        '{"grant_key":"default_local_operator","controller_kind":"local"}',
        'implemented'
    ),
    (
        'agent_thread_runtime',
        'GET',
        '/agent_thread_runtime',
        'Typed per-thread runtime contract surface with parent, owner, runtime, budget, and receipt fields.',
        'lucidota_canon.agent_thread_runtime',
        '{"thread_key":"root_operator_thread"}',
        '{"thread_key":"root_operator_thread","runtime_kind":"local"}',
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

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
) VALUES
    ('controller_grant', 'lucidota_control', 'typed_packet', 'view', true, 'Canonical controller grant surface.', '{"source":"controller_grant"}'::jsonb),
    ('agent_thread_runtime', 'lucidota_control', 'typed_packet', 'view', true, 'Canonical per-thread runtime surface.', '{"source":"agent_thread_runtime"}'::jsonb)
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

INSERT INTO lucidota_control.schema_owner_manifest (
    surface_id, canonical_owner, packet_class, surface_kind, approval_required, notes, detail
)
SELECT
    sr.surface_id,
    sr.canonical_owner,
    sr.packet_class,
    sr.surface_kind,
    sr.approval_required,
    COALESCE(NULLIF(sr.notes, ''), sr.description),
    jsonb_build_object('source', 'surface_registry', 'route_id', sr.route_id, 'target', sr.target)
FROM lucidota_canon.surface_registry sr
WHERE sr.active
ON CONFLICT (surface_id) DO UPDATE SET
    canonical_owner = EXCLUDED.canonical_owner,
    packet_class = EXCLUDED.packet_class,
    surface_kind = EXCLUDED.surface_kind,
    approval_required = EXCLUDED.approval_required,
    active = true,
    notes = EXCLUDED.notes,
    detail = EXCLUDED.detail,
    updated_at = now();

GRANT SELECT ON lucidota_canon.controller_grant, lucidota_canon.agent_thread_runtime TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
