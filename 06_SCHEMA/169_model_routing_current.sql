-- DB-visible model routing packet: expose real local coverage and missing role blockers.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.model_routing_current AS
WITH model_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_models,
        count(*) FILTER (WHERE active AND role = 'router') AS router_models,
        count(*) FILTER (WHERE active AND role = 'listener') AS listener_models,
        count(*) FILTER (WHERE active AND role = 'heavy_hitter') AS heavy_hitter_models,
        max(updated_at) AS latest_model_updated_at
    FROM lucidota_canon.model_registry
),
provider_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_providers,
        COALESCE(
            jsonb_agg(provider_key ORDER BY provider_key) FILTER (WHERE active),
            '[]'::jsonb
        ) AS active_provider_keys
    FROM lucidota_canon.provider_registry
),
role_names AS (
    SELECT role
    FROM (VALUES
        ('router'),
        ('classifier'),
        ('summarizer'),
        ('embedder'),
        ('reranker'),
        ('thinker'),
        ('watcher'),
        ('treelite_gate')
    ) AS roles(role)
),
role_rows AS (
    SELECT
        COALESCE(
            jsonb_object_agg(rn.role, row_to_json(m)::jsonb ORDER BY rn.role),
            '{}'::jsonb
        ) AS local_model_roles
    FROM role_names rn
    LEFT JOIN LATERAL (
        SELECT *
        FROM lucidota_canon.model_registry mr
        WHERE mr.active
          AND mr.role = rn.role
        LIMIT 1
    ) m ON true
),
missing_rows AS (
    SELECT
        COALESCE(
            jsonb_agg(role ORDER BY role),
            '[]'::jsonb
        ) AS missing_roles
    FROM (
        SELECT rn.role
        FROM role_names rn
        LEFT JOIN lucidota_canon.model_registry mr
            ON mr.active AND mr.role = rn.role
        WHERE mr.model_id IS NULL
    ) missing
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    'model_routing_current'::text AS routing_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'active_models', model_rows.active_models,
        'router_models', model_rows.router_models,
        'listener_models', model_rows.listener_models,
        'heavy_hitter_models', model_rows.heavy_hitter_models,
        'latest_model_updated_at', model_rows.latest_model_updated_at
    ) AS model_registry,
    jsonb_build_object(
        'active_providers', provider_rows.active_providers,
        'active_provider_keys', provider_rows.active_provider_keys
    ) AS provider_registry,
    role_rows.local_model_roles,
    missing_rows.missing_roles,
    jsonb_build_object(
        'deterministic_first', true,
        'prefer_local_lanes', true,
        'preferred_route_order', ARRAY['router', 'classifier', 'summarizer', 'embedder', 'reranker', 'thinker', 'watcher', 'treelite_gate'],
        'missing_roles_mean_blocker', true,
        'next_action', 'use the live local lanes that exist; treat missing roles as DB-visible blockers, not fantasy models',
        'drift_control', 'if repeated work stops being useful, convert it into a workflow row and prune the loser path'
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1',
        'curl -sS http://127.0.0.1:3000/model_routing_blockers?limit=1',
        'curl -sS http://127.0.0.1:3000/model_registry_current?limit=1',
        'curl -sS http://127.0.0.1:3000/provider_current?limit=1',
        './luci model-routing-current --json',
        './luci model-routing-blockers --json',
        './luci model registry current --json',
        './luci provider current --json'
    ) AS next_commands
FROM model_rows
CROSS JOIN provider_rows
CROSS JOIN role_rows
CROSS JOIN missing_rows
CROSS JOIN goal_row;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'model_routing_current',
    'GET',
    '/model_routing_current',
    'Current local model coverage, provider coverage, and missing-role blockers for the ML router.',
    'lucidota_canon.model_routing_current',
    '{"limit":"1"}',
    '{"routing_packet_id":"model_routing_current"}',
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

GRANT SELECT ON lucidota_canon.model_routing_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
