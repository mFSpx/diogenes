-- Root orchestrator current truth-spine lift.
-- Append top-level goal, db law, sub-orchestrators, blockers, and receipts without touching manual_current.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.root_orchestrator_current AS
WITH live_routes AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'route_id', route_id,
            'method', method,
            'path_pattern', path_pattern,
            'description', description,
            'target', target,
            'status', status
        )
        ORDER BY route_id
    ) AS route_list,
    count(*) AS route_count
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        'manual_current', 'canon_current', 'canon_versions', 'active_goal', 'api_workflow_registry',
        'capability_registry', 'model_registry', 'provider_registry', 'workflow_registry',
        'daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'cloud_packet', 'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'ontology_work_batch', 'ontology_work_item', 'todo_current', 'skill_policy_current',
        'root_orchestrator_current'
    )
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
todo_rows AS (
    SELECT COALESCE(
        jsonb_agg(to_jsonb(t) ORDER BY t.created_at DESC),
        '[]'::jsonb
    ) AS todo_current
    FROM (
        SELECT *
        FROM lucidota_canon.todo_current
        WHERE status IN ('ready', 'queued', 'running')
        ORDER BY created_at DESC
        LIMIT 5
    ) t
),
todo_top AS (
    SELECT *
    FROM lucidota_canon.todo_current
    WHERE status IN ('ready', 'queued', 'running')
    ORDER BY created_at DESC
    LIMIT 1
),
daemon_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(d) ORDER BY d.daemon_name), '[]'::jsonb) AS daemon_status
    FROM lucidota_canon.daemon_status d
),
model_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.model_id), '[]'::jsonb) AS model_registry
    FROM lucidota_canon.model_registry m
),
provider_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.provider_key), '[]'::jsonb) AS provider_registry
    FROM lucidota_canon.provider_registry p
),
workflow_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(w) ORDER BY w.workflow_id), '[]'::jsonb) AS workflow_registry
    FROM lucidota_canon.workflow_registry w
),
skill_policy_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.updated_at DESC), '[]'::jsonb) AS skill_policy_current
    FROM lucidota_canon.skill_policy_current p
),
sub_orchestrator_rows AS (
    SELECT jsonb_build_array(
        jsonb_build_object(
            'name', 'ONTOLOGY ORCHESTRATOR',
            'status', CASE WHEN todo_top.batch_uuid IS NOT NULL THEN 'active' ELSE 'waiting' END,
            'route_hint', '/ontology_work_batch',
            'purpose', 'Decompose messy operator text into DB-visible work packets'
        ),
        jsonb_build_object(
            'name', 'NEEDLE/ROUTER ORCHESTRATOR',
            'status', CASE WHEN EXISTS (SELECT 1 FROM lucidota_canon.model_registry WHERE role = 'router' AND active) THEN 'ready' ELSE 'blocked' END,
            'route_hint', '/model_registry',
            'purpose', 'Choose local Needle/Treelite/router lanes from live registry rows'
        ),
        jsonb_build_object(
            'name', 'PARALLEL PLANNER ORCHESTRATOR',
            'status', CASE WHEN todo_top.batch_uuid IS NOT NULL THEN 'ready' ELSE 'waiting' END,
            'route_hint', '/todo_current',
            'purpose', 'Split audits/tests/isolated patches into parallel batches and serialize shared mutations'
        ),
        jsonb_build_object(
            'name', 'INGEST ORCHESTRATOR',
            'status', 'ready',
            'route_hint', '/book_source',
            'purpose', 'Keep books and artifacts DB-visible and receipt-gated'
        ),
        jsonb_build_object(
            'name', 'RUST REWRITE ORCHESTRATOR',
            'status', 'planned',
            'route_hint', '/workflow_registry',
            'purpose', 'Port bounded hot paths to Rust only after tests and behavior contracts exist'
        ),
        jsonb_build_object(
            'name', 'SLOP DESTRUCTION ORCHESTRATOR',
            'status', 'planned',
            'route_hint', '/api_route_catalog',
            'purpose', 'Quarantine duplicates and obsolete paths before removal'
        ),
        jsonb_build_object(
            'name', 'PROOF ORCHESTRATOR',
            'status', 'active',
            'route_hint', '/manual_current',
            'purpose', 'Prove the operator -> compiler -> batch -> executor -> receipt -> manual loop'
        )
    ) AS sub_orchestrators
    FROM todo_top
),
cli_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.received_at DESC), '[]'::jsonb) AS cli_process_receipts
    FROM (
        SELECT *
        FROM lucidota_canon.cli_process_receipts
        LIMIT 3
    ) c
),
flow_receipt_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(fr) ORDER BY fr.created_at DESC), '[]'::jsonb) AS flow_receipts
    FROM (
        SELECT *
        FROM lucidota_canon.flow_receipts
        LIMIT 3
    ) fr
),
test_receipt_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(tr) ORDER BY tr.completed_at DESC), '[]'::jsonb) AS api_test_execution_receipts
    FROM (
        SELECT *
        FROM lucidota_audit.test_execution_receipts
        LIMIT 3
    ) tr
),
missing_roles AS (
    SELECT COALESCE(t.missing_executor_roles, ARRAY[]::text[]) AS missing_executor_roles
    FROM todo_top t
),
selected_lanes AS (
    SELECT COALESCE(t.selected_lanes, '[]'::jsonb) AS selected_lanes
    FROM todo_top t
)
SELECT
    'ROOT_ORCHESTRATOR_CURRENT'::text AS orchestrator_id,
    'Root Orchestrator Current'::text AS title,
    live_routes.route_count AS route_count,
    now() AS max_updated_at,
    live_routes.route_list,
    jsonb_build_object(
        'read_surface', 'PostgREST safe views and RPCs only',
        'write_surface', 'DB work orders and receipts only',
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches',
        'root_orchestrator', 'DB-visible sub-orchestrator status packet; no new daemon or hidden control plane'
    ) AS auth_expectations,
    jsonb_build_object(
        'root_loop', 'operator command -> root_orchestrator_current -> ontology_work_batch / todo_current -> sub-orchestrator packets -> receipts -> manual update',
        'manual_loop', 'manual_current -> route list + registries + daemon status + skill policy + root orchestrator',
        'model_loop', 'model_registry / provider_registry / workflow_registry -> live role coverage -> missing roles/blockers',
        'book_loop', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'todo_current', todo_rows.todo_current,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry,
        'skill_policy_current', skill_policy_rows.skill_policy_current,
        'missing_executor_roles', COALESCE((SELECT missing_executor_roles FROM missing_roles), ARRAY[]::text[]),
        'selected_lanes', COALESCE((SELECT selected_lanes FROM selected_lanes), '[]'::jsonb),
        'sub_orchestrators', sub_orchestrator_rows.sub_orchestrators
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1',
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
        'curl -sS http://127.0.0.1:3000/model_registry?limit=20',
        'curl -sS http://127.0.0.1:3000/cli_process_receipts?limit=3',
        'curl -sS http://127.0.0.1:3000/flow_receipts?limit=3',
        'curl -sS http://127.0.0.1:3000/api_test_execution_receipts?limit=3',
        './luci root orchestrator current --json',
        './luci api cli process receipts --json',
        './luci api flow receipts --json',
        './luci api test execution receipts --json',
        '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<operator objective>"',
        '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
        '.venv/bin/python scripts/test_receipt_gate.py run --scope policy_and_retirement -- .venv/bin/python -m pytest -q tests/test_skill_policy_current_surface.py tests/test_indy_book_ops_schema.py tests/test_manual_current_surface.py tests/test_orchestrator_registry_routes.py'
    ) AS next_commands,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; Rust-worthy code becomes Rust only after contract, tests, and A/B receipt.'
    ) AS db_law,
    sub_orchestrator_rows.sub_orchestrators AS sub_orchestrators,
    jsonb_build_object(
        'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb)
    ) AS blockers,
    jsonb_build_object(
        'cli_process_receipts', cli_rows.cli_process_receipts,
        'flow_receipts', flow_receipt_rows.flow_receipts,
        'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts
    ) AS receipts
FROM live_routes
CROSS JOIN goal_row
CROSS JOIN todo_rows
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN provider_rows
CROSS JOIN workflow_rows
CROSS JOIN skill_policy_rows
CROSS JOIN sub_orchestrator_rows
CROSS JOIN cli_rows
CROSS JOIN flow_receipt_rows
CROSS JOIN test_receipt_rows
LEFT JOIN todo_top ON true;

GRANT SELECT, INSERT, UPDATE ON lucidota_control.skill_policy TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current TO mfspx;
GRANT SELECT ON lucidota_canon.root_orchestrator_current TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current, lucidota_canon.root_orchestrator_current TO lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;

NOTIFY pgrst, 'reload schema';

COMMIT;
