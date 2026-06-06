-- DB-visible workflow current packet: expose live workflow counts, active names, and status breakdown.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.workflow_current AS
WITH workflow_rows AS (
    SELECT
        count(*) AS workflow_count,
        count(*) FILTER (WHERE status = 'active') AS active_count,
        count(*) FILTER (WHERE status = 'deprecated') AS deprecated_count,
        count(*) FILTER (WHERE status = 'active' AND workflow_name = 'basic-workflows') AS basic_workflows_count,
        count(*) FILTER (WHERE owner = 'indy_reads') AS indy_owned_count,
        max(updated_at) AS latest_workflow_updated_at,
        COALESCE(jsonb_agg(workflow_name ORDER BY workflow_name) FILTER (WHERE status = 'active'), '[]'::jsonb) AS active_names
    FROM lucidota_control.workflow_registry
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
status_rows AS (
    SELECT
        COALESCE(
            jsonb_object_agg(status, count_value ORDER BY status),
            '{}'::jsonb
        ) AS status_breakdown
    FROM (
        SELECT status, count(*) AS count_value
        FROM lucidota_control.workflow_registry
        GROUP BY status
    ) s
),
owner_rows AS (
    SELECT
        COALESCE(
            jsonb_object_agg(owner, count_value ORDER BY owner),
            '{}'::jsonb
        ) AS owner_breakdown
    FROM (
        SELECT owner, count(*) AS count_value
        FROM lucidota_control.workflow_registry
        GROUP BY owner
    ) o
),
active_workflow_rows AS (
    SELECT COALESCE(jsonb_agg(workflow_name ORDER BY workflow_name), '[]'::jsonb) AS active_workflows
    FROM (
        SELECT workflow_name
        FROM lucidota_canon.workflow_registry
        WHERE status = 'active'
        ORDER BY workflow_name
        LIMIT 25
    ) w
)
SELECT
    'workflow_current'::text AS workflow_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'workflow_count', workflow_rows.workflow_count,
        'active_count', workflow_rows.active_count,
        'deprecated_count', workflow_rows.deprecated_count,
        'basic_workflows_count', workflow_rows.basic_workflows_count,
        'indy_owned_count', workflow_rows.indy_owned_count,
        'latest_workflow_updated_at', workflow_rows.latest_workflow_updated_at,
        'active_names', workflow_rows.active_names
    ) AS workflow_summary,
    status_rows.status_breakdown,
    owner_rows.owner_breakdown,
    active_workflow_rows.active_workflows,
    jsonb_build_object(
        'authoritative_layer', 'Workflow registry rows are DB-visible work, not script folklore',
        'basic_workflows', 'keep basic workflows as workflows; do not flatten them into notes',
        'parallelism', 'route independent workflows in parallel, serialize only shared DB/service mutations',
        'next_action', 'promote repeated operator tasks to workflow rows and keep the registry live',
        'curiosity_budget', 'unknown -> ontology tag -> route -> source -> experiment -> receipt -> learned edge'
    ) AS workflow_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'workflow_current',
        'workflow_registry',
        'api_workflow_registry'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
        'sheet_current',
        'todo_current',
        'command_registry',
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime',
        'workflow_registry'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'active_workflows', active_workflow_rows.active_workflows
    ) AS orchestration
FROM workflow_rows
CROSS JOIN goal_row
CROSS JOIN status_rows
CROSS JOIN owner_rows
CROSS JOIN active_workflow_rows;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'workflow_current',
    'GET',
    '/workflow_current',
    'Workflow registry current packet for counts, active names, and status/owner breakdown.',
    'lucidota_canon.workflow_current',
    '{"limit":"1"}',
    '{"workflow_packet_id":"workflow_current"}',
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

GRANT SELECT ON lucidota_canon.workflow_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
