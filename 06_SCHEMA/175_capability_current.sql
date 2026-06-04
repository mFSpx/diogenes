-- DB-visible capability current packet: expose live capability counts, groups, workflow mapping, and active lanes.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.capability_current AS
WITH capability_rows AS (
    SELECT
        count(*) AS capability_count,
        count(*) FILTER (WHERE lifecycle_status = 'active') AS active_count,
        count(*) FILTER (WHERE lifecycle_status = 'planned') AS planned_count,
        count(*) FILTER (WHERE lifecycle_status = 'deprecated') AS deprecated_count,
        count(*) FILTER (WHERE run_state = 'ran') AS ran_count,
        count(*) FILTER (WHERE run_state = 'prototype') AS prototype_count,
        count(*) FILTER (WHERE run_state = 'active') AS run_active_count,
        count(DISTINCT capability_group) AS group_count,
        count(DISTINCT workflow_name) AS workflow_name_count,
        max(updated_at) AS latest_capability_updated_at,
        COALESCE(
            jsonb_agg(DISTINCT workflow_name ORDER BY workflow_name) FILTER (WHERE workflow_name IS NOT NULL AND workflow_name <> ''),
            '[]'::jsonb
        ) AS workflow_names
    FROM lucidota_canon.capability_registry
),
status_rows AS (
    SELECT COALESCE(jsonb_object_agg(lifecycle_status, count_value ORDER BY lifecycle_status), '{}'::jsonb) AS status_breakdown
    FROM (
        SELECT lifecycle_status, count(*) AS count_value
        FROM lucidota_canon.capability_registry
        GROUP BY lifecycle_status
    ) s
),
group_rows AS (
    SELECT COALESCE(jsonb_object_agg(capability_group, count_value ORDER BY capability_group), '{}'::jsonb) AS group_breakdown
    FROM (
        SELECT capability_group, count(*) AS count_value
        FROM lucidota_canon.capability_registry
        GROUP BY capability_group
    ) g
),
active_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.capability_group, c.capability_name), '[]'::jsonb) AS active_capabilities
    FROM (
        SELECT *
        FROM lucidota_canon.capability_registry
        WHERE lifecycle_status = 'active'
        ORDER BY capability_group, capability_name
        LIMIT 25
    ) c
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    'capability_current'::text AS capability_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'capability_count', cr.capability_count,
        'active_count', cr.active_count,
        'planned_count', cr.planned_count,
        'deprecated_count', cr.deprecated_count,
        'ran_count', cr.ran_count,
        'prototype_count', cr.prototype_count,
        'run_active_count', cr.run_active_count,
        'group_count', cr.group_count,
        'workflow_name_count', cr.workflow_name_count,
        'latest_capability_updated_at', cr.latest_capability_updated_at,
        'workflow_names', cr.workflow_names
    ) AS capability_summary,
    status_rows.status_breakdown,
    group_rows.group_breakdown,
    active_rows.active_capabilities,
    jsonb_build_object(
        'registry_truth', 'capability_registry is the DB-visible capability ledger; no script folklore authority',
        'active_rows', 'active rows are deployable lanes; planned rows are backlog, not authority',
        'workflow_mapping', 'capability_key -> workflow_name -> command -> receipt path',
        'parallelism', 'independent capability lanes can fan out; shared DB/service mutations stay serialized',
        'routing_truth', 'missing local role or missing capability is a DB-visible blocker, not a guess'
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/capability_current?limit=1',
        'curl -sS http://127.0.0.1:3000/capability_registry?limit=5',
        'curl -sS http://127.0.0.1:3000/workflow_current?limit=1',
        './luci capability current --json',
        './luci capability registry --json',
        './luci workflow current --json'
    ) AS next_commands
FROM capability_rows cr
CROSS JOIN status_rows
CROSS JOIN group_rows
CROSS JOIN active_rows
CROSS JOIN goal_row;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'capability_current',
    'GET',
    '/capability_current',
    'Capability registry current packet for counts, groups, workflow mapping, and active capability lanes.',
    'lucidota_canon.capability_current',
    '{"limit":"1"}',
    '{"capability_packet_id":"capability_current"}',
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

GRANT SELECT ON lucidota_canon.capability_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
