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
    batch_kind,
    jsonb_build_object(
        'goal_id', batch_uuid::text,
        'title', COALESCE(objective_summary, 'Ontology work batch'::text),
        'status', status
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array('ontology_work_batch', 'todo_current') AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'todo_current',
        'command_registry',
        'capability_registry',
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
        'batch_uuid', batch_uuid,
        'batch_key', batch_key,
        'selected_lanes', selected_lanes
    ) AS orchestration
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
    b.batch_kind,
    jsonb_build_object(
        'goal_id', b.batch_uuid::text,
        'title', COALESCE(b.objective_summary, 'Todo batch'::text),
        'status', b.status
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array('todo_current', 'api_todo_current') AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'workflow_current',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
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
        'batch_uuid', b.batch_uuid,
        'batch_key', b.batch_key,
        'selected_lanes', b.selected_lanes
    ) AS orchestration
FROM lucidota_control.ontology_work_batch b
LEFT JOIN lucidota_control.ontology_work_item i ON i.batch_uuid = b.batch_uuid
GROUP BY
    b.batch_uuid, b.batch_key, b.source_ref, b.source_kind, b.source_hash, b.source_excerpt,
    b.objective_summary, b.subsystem, b.ontology_tags, b.risk, b.parallel_policy,
    b.planner_groups, b.selected_lanes, b.missing_executor_roles, b.executor_recommendation,
    b.acceptance_test, b.receipt_requirement, b.functionality_contract, b.status,
    b.detail, b.created_at, b.updated_at, b.workflow_count, b.workflows_preserved, b.batch_kind,
    b.batch_uuid, b.objective_summary, b.status, b.selected_lanes
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

GRANT SELECT, INSERT, UPDATE ON lucidota_control.ontology_work_batch, lucidota_control.ontology_work_item TO mfspx;
GRANT SELECT ON lucidota_canon.ontology_work_batch, lucidota_canon.ontology_work_item, lucidota_canon.todo_current TO mfspx;
GRANT SELECT ON lucidota_canon.ontology_work_batch, lucidota_canon.ontology_work_item, lucidota_canon.todo_current TO lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;
