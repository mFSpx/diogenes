-- DB-visible blocker summary for the ML router.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.model_routing_blockers AS
SELECT
    'model_routing_blockers'::text AS routing_packet_id,
    mr.refreshed_at,
    mr.missing_roles,
    jsonb_array_length(mr.missing_roles) AS missing_role_count,
    mr.local_model_roles,
    mr.routing_notes,
    jsonb_build_object(
        'goal_id', 'model-routing-blockers-spreadsheet-layer',
        'title', 'Model routing blockers packet',
        'status', 'active'
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/model_routing_blockers?limit=1',
        './luci model routing blockers --json',
        './luci api model routing blockers --json'
    ) AS next_commands
FROM lucidota_canon.model_routing_current mr
WHERE jsonb_array_length(mr.missing_roles) > 0;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'model_routing_blockers',
    'GET',
    '/model_routing_blockers',
    'DB-visible missing-role blocker packet derived from model_routing_current.',
    'lucidota_canon.model_routing_blockers',
    '{"limit":"1"}',
    '{"routing_packet_id":"model_routing_blockers"}',
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

GRANT SELECT ON lucidota_canon.model_routing_blockers TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
