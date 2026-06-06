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
        'model_routing_blockers'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'model_registry_current',
        'model_routing_current',
        'sheet_current',
        'todo_current',
        'command_registry',
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime',
        'model_routing_current'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'missing_role_count', jsonb_array_length(mr.missing_roles),
        'honestly_skipped_role_count', jsonb_array_length(mr.honestly_skipped_roles),
        'routing_packet_id', 'model_routing_blockers'
    ) AS orchestration,
    mr.honestly_skipped_roles,
    jsonb_array_length(mr.honestly_skipped_roles) AS honestly_skipped_role_count,
    mr.role_admission_decisions
FROM lucidota_canon.model_routing_current mr
WHERE jsonb_array_length(mr.missing_roles) > 0
   OR jsonb_array_length(mr.honestly_skipped_roles) > 0;

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
