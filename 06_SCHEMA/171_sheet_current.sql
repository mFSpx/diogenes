-- DB-visible spreadsheet current packet: expose sheet task status, projections, and next batch.

BEGIN;

CREATE OR REPLACE VIEW lucidota_sheet.sheet_current AS
WITH task_rows AS (
    SELECT
        count(*) AS sheet_task_count,
        count(*) FILTER (WHERE status = 'OPEN') AS open_count,
        count(*) FILTER (WHERE status = 'RUNNING') AS running_count,
        count(*) FILTER (WHERE status = 'BLOCKED') AS blocked_count,
        count(*) FILTER (WHERE status = 'DONE') AS done_count,
        count(*) FILTER (WHERE route_band = 'ASK_OPERATOR') AS needs_operator_count,
        max(friction_score) AS max_friction_score,
        max(updated_at) AS latest_sheet_task_updated_at
    FROM lucidota_sheet.sheet_task
),
pressure_rows AS (
    SELECT
        count(*) AS pressure_row_count,
        count(*) FILTER (WHERE hot_count > 0) AS pressured_targets,
        max(last_seen_at) AS latest_pressure_seen_at
    FROM lucidota_projection.case_pressure_sheet
),
active_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(a) ORDER BY a.friction_score DESC, a.created_at ASC), '[]'::jsonb) AS active_work
    FROM (
        SELECT *
        FROM lucidota_sheet.active_work
        LIMIT 10
    ) a
),
batch_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(b) ORDER BY b.friction_score DESC, b.sheet_task_uuid), '[]'::jsonb) AS next_work_batch
    FROM (
        SELECT *
        FROM lucidota_sheet.next_work_batch
        LIMIT 10
    ) b
),
pressure_sheet_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.hot_count DESC, p.last_seen_at DESC), '[]'::jsonb) AS case_pressure_sheet
    FROM (
        SELECT *
        FROM lucidota_projection.case_pressure_sheet
        LIMIT 25
    ) p
)
SELECT
    'sheet_current'::text AS sheet_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'sheet_task_count', task_rows.sheet_task_count,
        'open_count', task_rows.open_count,
        'running_count', task_rows.running_count,
        'blocked_count', task_rows.blocked_count,
        'done_count', task_rows.done_count,
        'needs_operator_count', task_rows.needs_operator_count,
        'max_friction_score', task_rows.max_friction_score,
        'latest_sheet_task_updated_at', task_rows.latest_sheet_task_updated_at
    ) AS sheet_tasks,
    jsonb_build_object(
        'pressure_row_count', pressure_rows.pressure_row_count,
        'pressured_targets', pressure_rows.pressured_targets,
        'latest_pressure_seen_at', pressure_rows.latest_pressure_seen_at
    ) AS projections,
    active_rows.active_work,
    batch_rows.next_work_batch,
    pressure_sheet_rows.case_pressure_sheet,
    jsonb_build_object(
        'authoritative_layer', 'Postgres spreadsheet views first; file-sheet analytics only if the DB layer cannot express it',
        'operator_use', 'intake/edit/review surfaces for backlog, books, models, workflows, capabilities, evidence, LoRA candidates, training jobs, prompt ledger, manual status',
        'routing_order', ARRAY['generated_column', 'live_view', 'materialized_projection', 'sql_aggregate', 'duckdb_file_sheet', 'algorithm_escalation', 'model_last_resort'],
        'next_action', 'use sheet_current to inspect which tasks are open, blocked, or pressure-heavy, then promote only receipt-backed changes',
        'curiosity_budget', 'route uncertainty to ontology -> source -> experiment -> receipt -> learned edge'
    ) AS sheet_notes,
    jsonb_build_object(
        'goal_id', 'sheet-current-spreadsheet-layer',
        'title', 'Spreadsheet current packet',
        'status', 'active'
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'sheet_current'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'api_daemon_status',
        'capability_current',
        'provider_current',
        'workflow_current',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
        'sheet_current',
        'api_sheet_current',
        'todo_current',
        'skill_policy_current',
        'command_registry',
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'sheet_task_count', task_rows.sheet_task_count,
        'open_count', task_rows.open_count,
        'running_count', task_rows.running_count
    ) AS orchestration
FROM task_rows
CROSS JOIN pressure_rows
CROSS JOIN active_rows
CROSS JOIN batch_rows
CROSS JOIN pressure_sheet_rows;

CREATE OR REPLACE VIEW lucidota_canon.sheet_current AS
SELECT * FROM lucidota_sheet.sheet_current;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'sheet_current',
    'GET',
    '/sheet_current',
    'Spreadsheet-layer current packet for task counts, pressure, active work, and next batch.',
    'lucidota_canon.sheet_current',
    '{"limit":"1"}',
    '{"sheet_packet_id":"sheet_current"}',
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

GRANT SELECT ON lucidota_sheet.sheet_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.sheet_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
