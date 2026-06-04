-- DB-visible model registry current packet: expose live model counts, roles, and loadouts.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.model_registry_current AS
WITH model_rows AS (
    SELECT
        count(*) AS model_count,
        count(*) FILTER (WHERE active) AS active_count,
        count(DISTINCT role) AS role_count,
        count(DISTINCT loadout_id) AS loadout_count,
        count(*) FILTER (WHERE benchmark_status = 'accepted') AS accepted_count,
        max(updated_at) AS latest_model_updated_at,
        COALESCE(jsonb_agg(DISTINCT role ORDER BY role), '[]'::jsonb) AS role_names,
        COALESCE(jsonb_agg(DISTINCT loadout_id ORDER BY loadout_id), '[]'::jsonb) AS loadout_names
    FROM lucidota_canon.model_registry
),
role_rows AS (
    SELECT COALESCE(jsonb_object_agg(role, count_value ORDER BY role), '{}'::jsonb) AS role_breakdown
    FROM (
        SELECT role, count(*) AS count_value
        FROM lucidota_canon.model_registry
        GROUP BY role
    ) r
),
loadout_rows AS (
    SELECT COALESCE(jsonb_object_agg(loadout_id, count_value ORDER BY loadout_id), '{}'::jsonb) AS loadout_breakdown
    FROM (
        SELECT loadout_id, count(*) AS count_value
        FROM lucidota_canon.model_registry
        GROUP BY loadout_id
    ) l
),
active_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.role, m.model_id), '[]'::jsonb) AS active_models
    FROM (
        SELECT *
        FROM lucidota_canon.model_registry
        WHERE active
        ORDER BY role, model_id
        LIMIT 25
    ) m
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    'model_registry_current'::text AS model_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'model_count', mr.model_count,
        'active_count', mr.active_count,
        'role_count', mr.role_count,
        'loadout_count', mr.loadout_count,
        'accepted_count', mr.accepted_count,
        'latest_model_updated_at', mr.latest_model_updated_at,
        'role_names', mr.role_names,
        'loadout_names', mr.loadout_names
    ) AS model_summary,
    role_rows.role_breakdown,
    loadout_rows.loadout_breakdown,
    active_rows.active_models,
    jsonb_build_object(
        'registry_truth', 'model_registry is the DB-visible model ledger; no script folklore authority',
        'routing_truth', 'active models are live choices; benchmark status and role are routing clues',
        'local_before_cloud', 'prefer local deterministic/needle/treelite/river lanes before cloud',
        'missing_roles', 'missing model roles are DB-visible blockers, not guesses'
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/model_registry_current?limit=1',
        'curl -sS http://127.0.0.1:3000/model_registry?limit=5',
        'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1',
        './luci model registry current --json',
        './luci model registry --json',
        './luci model-routing-current --json'
    ) AS next_commands
FROM model_rows mr
CROSS JOIN role_rows
CROSS JOIN loadout_rows
CROSS JOIN active_rows
CROSS JOIN goal_row;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'model_registry_current',
    'GET',
    '/model_registry_current',
    'Model registry current packet for counts, roles, and loadout coverage.',
    'lucidota_canon.model_registry_current',
    '{"limit":"1"}',
    '{"model_packet_id":"model_registry_current"}',
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

GRANT SELECT ON lucidota_canon.model_registry_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
