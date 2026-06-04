-- DB-visible provider current packet: expose live provider counts, kinds, and active lanes.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.provider_current AS
WITH provider_rows AS (
    SELECT
        count(*) AS provider_count,
        count(*) FILTER (WHERE active) AS active_count,
        count(DISTINCT provider_kind) AS kind_count,
        count(*) FILTER (WHERE provider_kind IN ('cloud_provider', 'cloud_orchestrator')) AS cloud_count,
        count(*) FILTER (WHERE provider_kind IN ('local_runtime', 'deterministic_model_runtime', 'stream_runtime')) AS local_count,
        max(default_model) AS latest_default_model,
        COALESCE(jsonb_agg(DISTINCT provider_kind ORDER BY provider_kind), '[]'::jsonb) AS provider_kind_names
    FROM lucidota_canon.provider_registry
),
kind_rows AS (
    SELECT COALESCE(jsonb_object_agg(provider_kind, count_value ORDER BY provider_kind), '{}'::jsonb) AS kind_breakdown
    FROM (
        SELECT provider_kind, count(*) AS count_value
        FROM lucidota_canon.provider_registry
        GROUP BY provider_kind
    ) k
),
active_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.provider_key), '[]'::jsonb) AS active_providers
    FROM (
        SELECT *
        FROM lucidota_canon.provider_registry
        WHERE active
        ORDER BY provider_key
        LIMIT 25
    ) p
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    'provider_current'::text AS provider_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'provider_count', pr.provider_count,
        'active_count', pr.active_count,
        'kind_count', pr.kind_count,
        'cloud_count', pr.cloud_count,
        'local_count', pr.local_count,
        'latest_default_model', pr.latest_default_model,
        'provider_kind_names', pr.provider_kind_names
    ) AS provider_summary,
    kind_rows.kind_breakdown,
    active_rows.active_providers,
    jsonb_build_object(
        'registry_truth', 'provider_registry is the DB-visible provider ledger; no script folklore authority',
        'routing_truth', 'provider_kind and default_model define legal lanes; active rows are live choices',
        'local_before_cloud', 'prefer local, deterministic, and stream runtimes before cloud providers',
        'missing_roles', 'missing provider or model roles are DB-visible blockers, not guesses'
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/provider_current?limit=1',
        'curl -sS http://127.0.0.1:3000/provider_registry?limit=5',
        'curl -sS http://127.0.0.1:3000/model_registry_current?limit=1',
        'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1',
        './luci provider current --json',
        './luci provider registry --json',
        './luci model registry current --json',
        './luci model-routing-current --json'
    ) AS next_commands
FROM provider_rows pr
CROSS JOIN kind_rows
CROSS JOIN active_rows
CROSS JOIN goal_row;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'provider_current',
    'GET',
    '/provider_current',
    'Provider registry current packet for counts, kinds, and active provider lanes.',
    'lucidota_canon.provider_current',
    '{"limit":"1"}',
    '{"provider_packet_id":"provider_current"}',
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

GRANT SELECT ON lucidota_canon.provider_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
