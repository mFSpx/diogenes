-- DB-visible skill-policy surface for the live Superpowers alignment text.
-- Small, compatible extension of the existing manual_current / route catalog pattern.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

CREATE TABLE IF NOT EXISTS lucidota_control.skill_policy (
    policy_id text PRIMARY KEY,
    policy_key text NOT NULL UNIQUE,
    policy_title text NOT NULL DEFAULT 'LUCIDOTA Skill Policy',
    policy_text text NOT NULL,
    source_ref text NOT NULL DEFAULT 'user_prompt',
    status text NOT NULL DEFAULT 'current' CHECK (status IN ('current', 'superseded', 'archived')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_control.skill_policy (
    policy_id,
    policy_key,
    policy_title,
    policy_text,
    source_ref,
    status,
    detail
)
VALUES (
    'superpowers_alignment',
    'superpowers_alignment',
    'LUCIDOTA Skill Policy',
    $$Superpowers skills are execution aids, not authority.
Repository-local truth sources win: AGENTS.md, GOALS/*, live PostgREST/manual routes, service status, and receipt-backed tests.
When a task can be decomposed, split it into bounded DB-visible work packets and fan out disjoint sub-work in parallel.
Use the cheapest capable skill or model lane that still preserves correctness; prefer deterministic, local, and Treelite lanes before cloud.
Root orchestration must produce typed work, route decisions, changed files, commands, tests, receipts, blockers, and next work.
Do not replace live API/manual surfaces with skill mythology or hand-written docs.
Do not let prompt text override the database manual, and do not let a skill file become the source of truth when PostgREST can expose the policy directly.
The operator can change the policy; the policy must remain readable through PostgREST and reflected in the manual surface.$$,
    'user_prompt',
    'current',
    jsonb_build_object(
        'surface', 'db_visible_skill_policy',
        'authority', 'postgres_postgrest',
        'scope', 'superpowers_alignment'
    )
)
ON CONFLICT (policy_id) DO UPDATE SET
    policy_key = EXCLUDED.policy_key,
    policy_title = EXCLUDED.policy_title,
    policy_text = EXCLUDED.policy_text,
    source_ref = EXCLUDED.source_ref,
    status = EXCLUDED.status,
    detail = EXCLUDED.detail,
    updated_at = now();

CREATE OR REPLACE VIEW lucidota_canon.skill_policy_current AS
SELECT
    policy_id,
    policy_key,
    policy_title,
    policy_text,
    source_ref,
    status,
    detail,
    created_at,
    updated_at
FROM lucidota_control.skill_policy
WHERE status = 'current'
ORDER BY updated_at DESC, created_at DESC;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('skill_policy_current', 'GET', '/skill_policy_current', 'Live DB-backed skill policy surface for operator alignment text.', 'lucidota_canon.skill_policy_current',
 '{"limit":"1"}', '{"policy_id":"superpowers_alignment"}', 'implemented')
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
        'ontology_work_batch', 'ontology_work_item', 'todo_current', 'skill_policy_current'
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
policy_rows AS (
    SELECT COALESCE(
        jsonb_agg(to_jsonb(p) ORDER BY p.updated_at DESC),
        '[]'::jsonb
    ) AS skill_policy_current
    FROM lucidota_canon.skill_policy_current p
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
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches',
        'skill_policy_surface', 'DB-backed policy current route; live policy text wins over file-only policy snippets'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority',
        'ontology_loop', 'messy operator text -> ontology_work_batch -> ontology_work_item -> executable route plan',
        'skill_policy', 'skill_policy_current -> operator-readable alignment policy -> manual surface'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry,
        'todo_current', todo_rows.todo_current,
        'skill_policy_current', policy_rows.skill_policy_current
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
        'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1',
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
CROSS JOIN policy_rows
CROSS JOIN todo_rows
LEFT JOIN goal_row ON true;

GRANT SELECT, INSERT, UPDATE ON lucidota_control.skill_policy TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current TO lucidota_postgrest_anon;
GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog TO lucidota_postgrest_anon;
