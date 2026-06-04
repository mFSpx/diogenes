-- Ontology work compiler + batch planner surface.
-- Turns messy operator text into DB-visible batches/items with planner groups,
-- model-role recommendations, and a live todo summary route.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.ontology_work_batch (
    batch_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_key text NOT NULL UNIQUE,
    source_ref text NOT NULL DEFAULT 'operator_turn',
    source_kind text NOT NULL DEFAULT 'operator_text',
    source_hash text NOT NULL,
    source_excerpt text NOT NULL DEFAULT '',
    objective_summary text NOT NULL DEFAULT '',
    subsystem text NOT NULL DEFAULT 'mixed',
    ontology_tags text[] NOT NULL DEFAULT '{}'::text[],
    dependency_edges jsonb NOT NULL DEFAULT '[]'::jsonb,
    risk text NOT NULL DEFAULT 'medium' CHECK (risk IN ('low', 'medium', 'high', 'destructive')),
    parallel_policy text NOT NULL DEFAULT 'mixed' CHECK (parallel_policy IN ('parallel', 'serialized', 'mixed')),
    planner_groups jsonb NOT NULL DEFAULT '[]'::jsonb,
    selected_lanes jsonb NOT NULL DEFAULT '[]'::jsonb,
    missing_executor_roles text[] NOT NULL DEFAULT '{}'::text[],
    executor_recommendation jsonb NOT NULL DEFAULT '{}'::jsonb,
    acceptance_test text NOT NULL DEFAULT '',
    receipt_requirement text NOT NULL DEFAULT '',
    functionality_contract text NOT NULL DEFAULT '',
    workflow_count integer NOT NULL DEFAULT 0,
    workflows_preserved boolean NOT NULL DEFAULT false,
    batch_kind text NOT NULL DEFAULT 'ontology_batch' CHECK (batch_kind IN ('ontology_batch', 'workflow_batch')),
    status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'queued', 'running', 'blocked', 'done', 'archived')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.ontology_work_item (
    item_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_uuid uuid NOT NULL REFERENCES lucidota_control.ontology_work_batch(batch_uuid) ON DELETE CASCADE,
    item_rank integer NOT NULL,
    planner_group text NOT NULL DEFAULT '',
    work_kind text NOT NULL DEFAULT 'audit',
    workflow_name text NOT NULL DEFAULT '',
    subsystem text NOT NULL DEFAULT 'mixed',
    ontology_tags text[] NOT NULL DEFAULT '{}'::text[],
    dependency_edges jsonb NOT NULL DEFAULT '[]'::jsonb,
    risk text NOT NULL DEFAULT 'medium' CHECK (risk IN ('low', 'medium', 'high', 'destructive')),
    parallelizable boolean NOT NULL DEFAULT true,
    serialized boolean NOT NULL DEFAULT false,
    route_hint text NOT NULL DEFAULT '',
    executor_recommendation jsonb NOT NULL DEFAULT '{}'::jsonb,
    acceptance_test text NOT NULL DEFAULT '',
    receipt_requirement text NOT NULL DEFAULT '',
    functionality_contract text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'queued', 'running', 'blocked', 'done', 'archived')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (batch_uuid, item_rank)
);

ALTER TABLE lucidota_control.ontology_work_batch
    ADD COLUMN IF NOT EXISTS workflow_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS workflows_preserved boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS batch_kind text NOT NULL DEFAULT 'ontology_batch';

ALTER TABLE lucidota_control.ontology_work_item
    ADD COLUMN IF NOT EXISTS workflow_name text NOT NULL DEFAULT '';

ALTER TABLE lucidota_control.ontology_work_batch
    DROP CONSTRAINT IF EXISTS ontology_work_batch_batch_kind_check,
    ADD CONSTRAINT ontology_work_batch_batch_kind_check CHECK (batch_kind IN ('ontology_batch', 'workflow_batch'));

CREATE INDEX IF NOT EXISTS ontology_work_batch_status_created_idx
    ON lucidota_control.ontology_work_batch(status, created_at DESC);
CREATE INDEX IF NOT EXISTS ontology_work_item_batch_idx
    ON lucidota_control.ontology_work_item(batch_uuid, item_rank);
CREATE INDEX IF NOT EXISTS ontology_work_item_subsystem_idx
    ON lucidota_control.ontology_work_item(subsystem, status, created_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.ontology_work_batch AS
SELECT
    batch_uuid,
    batch_key,
    source_ref,
    source_kind,
    source_hash,
    source_excerpt,
    objective_summary,
    subsystem,
    ontology_tags,
    dependency_edges,
    risk,
    parallel_policy,
    planner_groups,
    selected_lanes,
    missing_executor_roles,
    executor_recommendation,
    acceptance_test,
    receipt_requirement,
    functionality_contract,
    status,
    detail,
    created_at,
    updated_at,
    workflow_count,
    workflows_preserved,
    batch_kind
FROM lucidota_control.ontology_work_batch;

CREATE OR REPLACE VIEW lucidota_canon.ontology_work_item AS
SELECT
    item_uuid,
    batch_uuid,
    item_rank,
    planner_group,
    work_kind,
    subsystem,
    ontology_tags,
    dependency_edges,
    risk,
    parallelizable,
    serialized,
    route_hint,
    executor_recommendation,
    acceptance_test,
    receipt_requirement,
    functionality_contract,
    status,
    detail,
    created_at,
    updated_at,
    workflow_name
FROM lucidota_control.ontology_work_item;

CREATE OR REPLACE VIEW lucidota_canon.todo_current AS
SELECT
    b.batch_uuid,
    b.batch_key,
    b.source_ref,
    b.source_kind,
    b.source_hash,
    b.source_excerpt,
    b.objective_summary,
    b.subsystem,
    b.ontology_tags,
    b.risk,
    b.parallel_policy,
    b.planner_groups,
    b.selected_lanes,
    b.missing_executor_roles,
    b.executor_recommendation,
    b.acceptance_test,
    b.receipt_requirement,
    b.functionality_contract,
    b.status,
    COUNT(i.item_uuid)::bigint AS item_count,
    COUNT(*) FILTER (WHERE i.parallelizable)::bigint AS parallel_item_count,
    COUNT(*) FILTER (WHERE i.serialized)::bigint AS serialized_item_count,
    COALESCE(jsonb_agg(to_jsonb(i) ORDER BY i.item_rank) FILTER (WHERE i.item_uuid IS NOT NULL), '[]'::jsonb) AS items,
    b.detail,
    b.created_at,
    b.updated_at,
    b.workflow_count,
    b.workflows_preserved,
    b.batch_kind
FROM lucidota_control.ontology_work_batch b
LEFT JOIN lucidota_control.ontology_work_item i ON i.batch_uuid = b.batch_uuid
GROUP BY
    b.batch_uuid, b.batch_key, b.source_ref, b.source_kind, b.source_hash, b.source_excerpt,
    b.objective_summary, b.subsystem, b.ontology_tags, b.risk, b.parallel_policy,
    b.planner_groups, b.selected_lanes, b.missing_executor_roles, b.executor_recommendation,
    b.acceptance_test, b.receipt_requirement, b.functionality_contract, b.status,
    b.detail, b.created_at, b.updated_at, b.workflow_count, b.workflows_preserved, b.batch_kind
ORDER BY b.created_at DESC;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('ontology_work_batch', 'GET', '/ontology_work_batch', 'Ontology compiler batch rows.', 'lucidota_canon.ontology_work_batch',
 '{"limit":"1"}', '{"batch_uuid":"..."}', 'implemented'),
('ontology_work_item', 'GET', '/ontology_work_item', 'Ontology compiler item rows.', 'lucidota_canon.ontology_work_item',
 '{"limit":"1"}', '{"item_uuid":"..."}', 'implemented'),
('todo_current', 'GET', '/todo_current', 'Current active todo batch summary.', 'lucidota_canon.todo_current',
 '{"limit":"1"}', '{"batch_uuid":"..."}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

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
        'daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'cloud_packet', 'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'ontology_work_batch', 'ontology_work_item', 'todo_current'
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
        'skill_layers', 'execution aids only; live PostgREST/manual truth and GOALS handoffs win',
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority',
        'ontology_loop', 'messy operator text -> ontology_work_batch -> ontology_work_item -> executable route plan'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry,
        'todo_current', todo_rows.todo_current
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
        '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<objective text>"',
        '.venv/bin/python scripts/indy_daemon.py --once --json',
        '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
        '.venv/bin/python scripts/luci_todo.py --json'
    ) AS next_commands,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces
FROM live_routes
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN provider_rows
CROSS JOIN workflow_rows
CROSS JOIN todo_rows
LEFT JOIN goal_row ON true;

GRANT SELECT, INSERT, UPDATE ON lucidota_control.ontology_work_batch, lucidota_control.ontology_work_item TO mfspx;
GRANT SELECT ON lucidota_canon.ontology_work_batch, lucidota_canon.ontology_work_item, lucidota_canon.todo_current TO mfspx;
GRANT SELECT ON lucidota_canon.ontology_work_batch, lucidota_canon.ontology_work_item, lucidota_canon.todo_current TO lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;
