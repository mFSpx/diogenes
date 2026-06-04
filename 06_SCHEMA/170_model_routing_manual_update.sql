-- Extend the operator manual with the model-routing current packet.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.manual_current AS
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
        'model_routing_current', 'daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'cli_process_receipts', 'payload_archive_status',
        'cloud_packet', 'chrono_current', 'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'ontology_work_batch', 'ontology_work_item', 'todo_current', 'skill_policy_current',
        'root_orchestrator_current', 'prompts_filed', 'prompt_work_order_links',
        'prompt_recent', 'prompt_unlinked', 'prompt_catalog_status',
        'file_prompt', 'link_prompt_work_order', 'decompose_prompt_to_work_orders'
    )
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
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
model_routing_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(mr) ORDER BY mr.refreshed_at DESC), '[]'::jsonb) AS model_routing_current
    FROM lucidota_canon.model_routing_current mr
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
skill_policy_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.updated_at DESC), '[]'::jsonb) AS skill_policy_current
    FROM lucidota_canon.skill_policy_current p
),
root_orchestrator_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.max_updated_at DESC), '[]'::jsonb) AS root_orchestrator_current
    FROM lucidota_canon.root_orchestrator_current r
),
prompt_status_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.refreshed_at DESC), '[]'::jsonb) AS prompt_catalog_status
    FROM lucidota_canon.prompt_catalog_status p
),
prompt_recent_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompt_recent
    FROM (
        SELECT *
        FROM lucidota_canon.prompt_recent
        LIMIT 5
    ) p
)
SELECT
    'LUCIDOTA_OPERATOR_MANUAL'::text AS manual_id,
    'LUCIDOTA Operator Manual'::text AS title,
    live_routes.route_count AS node_count,
    now() AS max_updated_at,
    live_routes.route_list,
    jsonb_build_object(
        'read_surface', 'PostgREST safe views and RPCs only',
        'write_surface', 'DB work orders and receipts only',
        'legacy_book_watcher', 'retired as authority',
        'skill_layers', 'execution aids only; live PostgREST/manual truth and GOALS handoffs win; prompt filing law is DB-backed and preserves raw text',
        'skill_policy_surface', 'DB-backed policy current route; live policy text wins over file-only policy snippets',
        'prompt_filing', 'file_prompt -> prompt ledger row -> prompt_work_order_links -> prompt_recent / prompt_unlinked -> prompt_catalog_status',
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches + root orchestrator surface + prompt ledger + model routing packet'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority',
        'ontology_loop', 'messy operator text -> ontology_work_batch -> ontology_work_item -> executable route plan',
        'skill_policy', 'skill_policy_current -> operator-readable alignment policy -> manual surface',
        'root_orchestrator', 'root_orchestrator_current -> sub-orchestrator packets -> receipts -> manual update',
        'prompt_filing', 'operator or assistant prompt -> prompt ledger -> explicit linked work-order UUID or explicit unlinked reason',
        'chrono_alignment', 'prompt ledger -> work ledger -> execution history -> learning loop alignment packet',
        'model_routing', 'model_routing_current -> actual local role coverage, provider coverage, and missing-role blockers'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry,
        'model_routing_current', model_routing_rows.model_routing_current,
        'todo_current', todo_rows.todo_current,
        'skill_policy_current', skill_policy_rows.skill_policy_current,
        'root_orchestrator_current', root_orchestrator_rows.root_orchestrator_current,
        'prompt_catalog_status', prompt_status_rows.prompt_catalog_status,
        'prompt_recent', prompt_recent_rows.prompt_recent,
        'chrono_current', COALESCE((SELECT jsonb_agg(to_jsonb(c) ORDER BY c.refreshed_at DESC) FROM lucidota_canon.chrono_current c), '[]'::jsonb)
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/chrono_current?limit=1',
        'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1',
        './luci model-routing-blockers --json',
        'curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1',
        'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
        'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1',
        'curl -sS http://127.0.0.1:3000/prompt_catalog_status?limit=1',
        'curl -sS http://127.0.0.1:3000/prompt_recent?limit=5',
        '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<objective text>"',
        '.venv/bin/python scripts/indy_daemon.py --once --json',
        '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
        '.venv/bin/python scripts/prompt_ledger_capture.py --json'
    ) AS next_commands,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces
FROM live_routes
CROSS JOIN goal_row
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN provider_rows
CROSS JOIN workflow_rows
CROSS JOIN model_routing_rows
CROSS JOIN todo_rows
CROSS JOIN skill_policy_rows
CROSS JOIN root_orchestrator_rows
CROSS JOIN prompt_status_rows
CROSS JOIN prompt_recent_rows;

GRANT SELECT ON lucidota_canon.manual_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
