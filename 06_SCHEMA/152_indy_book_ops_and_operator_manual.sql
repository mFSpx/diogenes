-- INDY book ops + operator manual surface.
-- Retires BOOKS-folder authority by moving queue/work/receipt state into DB-visible tables and views.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_indy;

CREATE TABLE IF NOT EXISTS lucidota_indy.book_source (
    book_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_item_uuid uuid,
    path text NOT NULL UNIQUE,
    file_name text NOT NULL,
    file_ext text NOT NULL,
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL,
    status text NOT NULL DEFAULT 'located' CHECK (status IN ('located','extracted','chunked','embedded','error','archived')),
    title text NOT NULL DEFAULT '',
    author text NOT NULL DEFAULT '',
    extraction_method text NOT NULL DEFAULT '',
    extraction_error text NOT NULL DEFAULT '',
    text_sha256 text NOT NULL DEFAULT '',
    token_count integer NOT NULL DEFAULT 0,
    chunk_count integer NOT NULL DEFAULT 0,
    embedded_count integer NOT NULL DEFAULT 0,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.book_scan (
    scan_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'book_scan',
    object_type text NOT NULL DEFAULT 'book',
    scan_kind text NOT NULL DEFAULT 'explicit',
    path text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','archived')),
    event_count integer NOT NULL DEFAULT 0,
    dropped_raw_bodies integer NOT NULL DEFAULT 0,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    needs_cloud_reasoning boolean NOT NULL DEFAULT false,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (book_uuid, scan_kind, path)
);

CREATE TABLE IF NOT EXISTS lucidota_indy.book_read_queue (
    queue_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'book_read_queue',
    object_type text NOT NULL DEFAULT 'book',
    task_type text NOT NULL DEFAULT 'read',
    target_model text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','claimed','succeeded','failed','cancelled')),
    priority integer NOT NULL DEFAULT 100,
    event_count integer NOT NULL DEFAULT 0,
    dropped_raw_bodies integer NOT NULL DEFAULT 0,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    needs_cloud_reasoning boolean NOT NULL DEFAULT false,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.book_note (
    note_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'book_note',
    object_type text NOT NULL DEFAULT 'book',
    note_kind text NOT NULL DEFAULT 'margin',
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','archived')),
    body text NOT NULL DEFAULT '',
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.lora_candidate (
    candidate_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'lora_candidate',
    object_type text NOT NULL DEFAULT 'adapter',
    target_model text NOT NULL DEFAULT '',
    adapter_family text NOT NULL DEFAULT 'lora',
    status text NOT NULL DEFAULT 'planned' CHECK (status IN ('planned','staged','training','trained','blocked','deprecated')),
    event_count integer NOT NULL DEFAULT 0,
    dropped_raw_bodies integer NOT NULL DEFAULT 0,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    needs_cloud_reasoning boolean NOT NULL DEFAULT false,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.lora_adapter (
    adapter_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_uuid uuid NOT NULL REFERENCES lucidota_indy.lora_candidate(candidate_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'lora_adapter',
    object_type text NOT NULL DEFAULT 'adapter',
    base_model text NOT NULL DEFAULT '',
    adapter_path text NOT NULL DEFAULT '',
    adapter_hash text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'planned' CHECK (status IN ('planned','built','verified','deployed','archived','failed')),
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.training_job (
    job_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_uuid uuid NOT NULL REFERENCES lucidota_indy.lora_candidate(candidate_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'training_job',
    object_type text NOT NULL DEFAULT 'training',
    job_kind text NOT NULL DEFAULT 'lora',
    command text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    needs_cloud_reasoning boolean NOT NULL DEFAULT false,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_indy.book_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    book_uuid uuid NOT NULL REFERENCES lucidota_indy.book_source(book_uuid) ON DELETE CASCADE,
    source text NOT NULL DEFAULT 'postgrest',
    topic text NOT NULL DEFAULT 'book_receipt',
    object_type text NOT NULL DEFAULT 'receipt',
    receipt_kind text NOT NULL DEFAULT 'book',
    status text NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','warn','error')),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    event_ids text[] NOT NULL DEFAULT '{}'::text[],
    source_hashes text[] NOT NULL DEFAULT '{}'::text[],
    receipt_refs text[] NOT NULL DEFAULT '{}'::text[],
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS book_scan_book_idx ON lucidota_indy.book_scan(book_uuid, created_at DESC);
CREATE INDEX IF NOT EXISTS book_read_queue_book_idx ON lucidota_indy.book_read_queue(book_uuid, status, priority, created_at DESC);
CREATE INDEX IF NOT EXISTS book_note_book_idx ON lucidota_indy.book_note(book_uuid, created_at DESC);
CREATE INDEX IF NOT EXISTS lora_candidate_book_idx ON lucidota_indy.lora_candidate(book_uuid, status, created_at DESC);
CREATE INDEX IF NOT EXISTS lora_adapter_candidate_idx ON lucidota_indy.lora_adapter(candidate_uuid, status, created_at DESC);
CREATE INDEX IF NOT EXISTS training_job_candidate_idx ON lucidota_indy.training_job(candidate_uuid, status, created_at DESC);
CREATE INDEX IF NOT EXISTS book_receipt_book_idx ON lucidota_indy.book_receipt(book_uuid, created_at DESC);

CREATE OR REPLACE VIEW lucidota_canon.book_source AS
SELECT
    book_uuid,
    graph_item_uuid,
    path,
    file_name,
    file_ext,
    sha256,
    size_bytes,
    status,
    title,
    author,
    extraction_method,
    extraction_error,
    text_sha256,
    token_count,
    chunk_count,
    embedded_count,
    payload,
    created_at,
    updated_at
FROM lucidota_indy.book_source;

CREATE OR REPLACE VIEW lucidota_canon.book_scan AS
SELECT
    scan_uuid,
    book_uuid,
    source,
    topic,
    object_type,
    scan_kind,
    path,
    status,
    event_count,
    dropped_raw_bodies,
    summary,
    features,
    scores,
    needs_cloud_reasoning,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.book_scan;

CREATE OR REPLACE VIEW lucidota_canon.book_read_queue AS
SELECT
    queue_uuid,
    book_uuid,
    source,
    topic,
    object_type,
    task_type,
    target_model,
    status,
    priority,
    event_count,
    dropped_raw_bodies,
    summary,
    features,
    scores,
    needs_cloud_reasoning,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.book_read_queue;

CREATE OR REPLACE VIEW lucidota_canon.book_note AS
SELECT
    note_uuid,
    book_uuid,
    source,
    topic,
    object_type,
    note_kind,
    status,
    body,
    summary,
    features,
    scores,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.book_note;

CREATE OR REPLACE VIEW lucidota_canon.lora_candidate AS
SELECT
    candidate_uuid,
    book_uuid,
    source,
    topic,
    object_type,
    target_model,
    adapter_family,
    status,
    event_count,
    dropped_raw_bodies,
    summary,
    features,
    scores,
    needs_cloud_reasoning,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.lora_candidate;

CREATE OR REPLACE VIEW lucidota_canon.lora_adapter AS
SELECT
    adapter_uuid,
    candidate_uuid,
    source,
    topic,
    object_type,
    base_model,
    adapter_path,
    adapter_hash,
    status,
    summary,
    features,
    scores,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.lora_adapter;

CREATE OR REPLACE VIEW lucidota_canon.training_job AS
SELECT
    job_uuid,
    candidate_uuid,
    source,
    topic,
    object_type,
    job_kind,
    command,
    status,
    summary,
    features,
    scores,
    needs_cloud_reasoning,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.training_job;

CREATE OR REPLACE VIEW lucidota_canon.book_receipt AS
SELECT
    receipt_uuid,
    book_uuid,
    source,
    topic,
    object_type,
    receipt_kind,
    status,
    payload,
    summary,
    features,
    scores,
    event_ids,
    source_hashes,
    receipt_refs,
    detail,
    created_at,
    updated_at
FROM lucidota_indy.book_receipt;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
('manual_current', 'GET', '/manual_current', 'Live operator manual digest and route/status packet.', 'lucidota_canon.manual_current',
 '{"limit":"1"}', '{"manual_id":"LUCIDOTA_OPERATOR_MANUAL"}', 'implemented'),
('canon_current', 'GET', '/canon_current', 'Live canonical node snapshot.', 'lucidota_canon.canon_current',
 '{"limit":"1"}', '{"node_id":"..."}', 'implemented'),
('canon_versions', 'GET', '/canon_versions', 'Live canonical version history.', 'lucidota_canon.canon_versions',
 '{"limit":"1"}', '{"version_id":"..."}', 'implemented'),
('active_goal', 'GET', '/active_goal', 'Current active operator goal row.', 'lucidota_canon.active_goal',
 '{"limit":"1"}', '{"goal_id":"..."}', 'implemented'),
('api_workflow_registry', 'GET', '/api_workflow_registry', 'Live workflow registry for operator and daemon routing.', 'lucidota_canon.api_workflow_registry',
 '{"limit":"1"}', '{"workflow_id":"..."}', 'implemented'),
('capability_registry', 'GET', '/capability_registry', 'Capability registry for local and remote lanes.', 'lucidota_canon.capability_registry',
 '{"limit":"1"}', '{"capability_key":"..."}', 'implemented'),
('model_registry', 'GET', '/model_registry', 'Local model registry snapshot.', 'lucidota_canon.model_registry',
 '{"limit":"1"}', '{"model_id":"..."}', 'implemented'),
('provider_registry', 'GET', '/provider_registry', 'Provider registry snapshot.', 'lucidota_canon.provider_registry',
 '{"limit":"1"}', '{"provider_key":"..."}', 'implemented'),
('workflow_registry', 'GET', '/workflow_registry', 'Workflow registry summary.', 'lucidota_canon.workflow_registry',
 '{"limit":"1"}', '{"workflow_id":"..."}', 'implemented'),
('daemon_status', 'GET', '/daemon_status', 'Daemon heartbeat/status snapshot.', 'lucidota_canon.daemon_status',
 '{"limit":"1"}', '{"daemon_name":"indy_reads"}', 'implemented'),
('bytewax_compact_windows', 'GET', '/bytewax_compact_windows', 'Compact bytewax window rows for stream-state output.', 'lucidota_canon.bytewax_compact_windows',
 '{"limit":"1"}', '{"compact_window_uuid":"..."}', 'implemented'),
('indy_queue', 'GET', '/indy_queue', 'DB-visible Indy queue of queued dialogue rows.', 'lucidota_canon.indy_queue',
 '{"limit":"1"}', '{"id":"..."}', 'implemented'),
('indy_responses', 'GET', '/indy_responses', 'DB-visible Indy response receipts.', 'lucidota_canon.indy_responses',
 '{"limit":"1"}', '{"response_id":"..."}', 'implemented'),
('cloud_packet', 'POST', '/rpc/cloud_packet', 'Bounded prompt packet RPC for cloud/model callers.', 'lucidota_canon.cloud_packet',
 '{"work_order_id":"..."}', '{"contract_name":"..."}', 'implemented'),
('book_source', 'GET', '/book_source', 'Book custody records for DB-visible ingestion.', 'lucidota_canon.book_source',
 '{"limit":"1"}', '{"book_uuid":"..."}', 'implemented'),
('book_scan', 'GET', '/book_scan', 'Book scan rows for DB-visible file/import work orders.', 'lucidota_canon.book_scan',
 '{"limit":"1"}', '{"scan_uuid":"..."}', 'implemented'),
('book_read_queue', 'GET', '/book_read_queue', 'Book read queue rows for DB-visible work orders.', 'lucidota_canon.book_read_queue',
 '{"limit":"1"}', '{"queue_uuid":"..."}', 'implemented'),
('book_note', 'GET', '/book_note', 'Book note rows for operator and model annotations.', 'lucidota_canon.book_note',
 '{"limit":"1"}', '{"note_uuid":"..."}', 'implemented'),
('lora_candidate', 'GET', '/lora_candidate', 'Candidate adapter rows for book-to-adapter work.', 'lucidota_canon.lora_candidate',
 '{"limit":"1"}', '{"candidate_uuid":"..."}', 'implemented'),
('lora_adapter', 'GET', '/lora_adapter', 'Built adapter rows and artifact pointers.', 'lucidota_canon.lora_adapter',
 '{"limit":"1"}', '{"adapter_uuid":"..."}', 'implemented'),
('training_job', 'GET', '/training_job', 'Training job rows for adapter work orders.', 'lucidota_canon.training_job',
 '{"limit":"1"}', '{"job_uuid":"..."}', 'implemented'),
('book_receipt', 'GET', '/book_receipt', 'Receipt rows for book/adapter/training work.', 'lucidota_canon.book_receipt',
 '{"limit":"1"}', '{"receipt_uuid":"..."}', 'implemented')
ON CONFLICT (route_id) DO UPDATE SET
    method = EXCLUDED.method,
    path_pattern = EXCLUDED.path_pattern,
    description = EXCLUDED.description,
    target = EXCLUDED.target,
    sample_request = EXCLUDED.sample_request,
    sample_response = EXCLUDED.sample_response,
    status = EXCLUDED.status,
    updated_at = now();

DO $$
BEGIN
    EXECUTE $view$
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
        'manual_current', 'canon_current', 'canon_versions', 'active_goal', 'active_operation_mode', 'workload_audit_current', 'workload_audit_telemetry_current', 'workload_audit_ledger', 'api_workflow_registry',
        'capability_registry', 'capability_current', 'model_registry', 'model_registry_current', 'model_routing_current', 'model_routing_blockers',
        'provider_registry', 'provider_current', 'workflow_registry', 'workflow_current',
        'skill_policy_current', 'root_orchestrator_current', 'chrono_current',
        'api_root_law_docs', 'api_bible_edges', 'api_bible_manuals', 'api_bible_nodes', 'api_bible_route_catalog', 'api_bible_subtree', 'fn_bible_node_sort_key', 'agent_thread_runtime',
        'indy_reads_self_model', 'indy_reads_llmwiki_entry', 'indy_reads_hunch_log', 'indy_reads_learning_queue', 'indy_reads_system_map', 'indy_reads_mistake_ledger', 'indy_reads_research_source', 'indy_reads_metacognition_current',
        'indy_reads_target_model_loadout_current', 'indy_reads_vram_coprocessor_fabric_current',
        'decompose_prompt_to_work_orders', 'file_prompt', 'link_prompt_work_order',
        'prompt_recent', 'prompts_filed', 'prompt_work_order_links', 'prompt_unlinked', 'prompt_catalog_status',
        'daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'cloud_packet', 'cli_process_receipts', 'payload_archive_status',
        'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt'
    )
),
goal_row AS (
    SELECT jsonb_build_object(
        'goal_id', g.goal_id,
        'title', g.title,
        'status', g.status,
        'current_handoff_path', g.current_handoff_path,
        'updated_at', g.updated_at
    ) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
daemon_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(d.*) ORDER BY d.daemon_name), '[]'::jsonb) AS daemon_status
    FROM lucidota_canon.daemon_status d
),
route_refs AS (
    SELECT COALESCE(jsonb_agg(route_id ORDER BY route_id), '[]'::jsonb)
        || jsonb_build_array('sub_orchestrator_threads', 'sub_orchestrator_grants', 'indy_daemon_once', 'indy_runtime_broker_snapshot') AS route_refs
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        'manual_current', 'canon_current', 'canon_versions', 'active_goal', 'active_operation_mode', 'workload_audit_current', 'workload_audit_telemetry_current', 'workload_audit_ledger', 'api_workflow_registry',
        'capability_registry', 'capability_current', 'model_registry', 'model_registry_current',
        'provider_registry', 'provider_current', 'workflow_registry', 'workflow_current',
        'skill_policy_current', 'root_orchestrator_current', 'chrono_current',
        'api_root_law_docs', 'api_bible_edges', 'api_bible_manuals', 'api_bible_nodes', 'api_bible_route_catalog', 'api_bible_subtree', 'fn_bible_node_sort_key', 'agent_thread_runtime',
        'daemon_status', 'api_daemon_status', 'bytewax_compact_windows', 'indy_queue', 'indy_responses',
        'workload_audit_current', 'workload_audit_ledger', 'provider_call_receipt', 'model_invocation_receipt', 'agent_work_receipt', 'unproven_work_debt',
        'cloud_packet', 'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'indy_reads_self_model', 'indy_reads_llmwiki_entry', 'indy_reads_hunch_log', 'indy_reads_learning_queue', 'indy_reads_system_map', 'indy_reads_mistake_ledger', 'indy_reads_research_source', 'indy_reads_metacognition_current',
        'indy_reads_target_model_loadout_current', 'indy_reads_vram_coprocessor_fabric_current',
        'ontology_work_batch', 'ontology_work_item', 'todo_current',
        'chrono_current',
        'prompts_filed', 'prompt_work_order_links',
        'prompt_recent', 'prompt_unlinked', 'prompt_catalog_status',
        'file_prompt', 'link_prompt_work_order', 'decompose_prompt_to_work_orders',
        'cli_process_receipts', 'payload_archive_status', 'api_root_law_docs', 'capability_current',
        'api_test_execution_receipts', 'flow_receipts', 'flow_specs',
        'api_bible_edges', 'api_bible_manuals', 'api_bible_nodes', 'api_bible_route_catalog',
        'api_bible_subtree', 'fn_bible_node_sort_key', 'fn_bible_node_material',
        'command_registry', 'schema_owner_manifest',
        'surface_registry', 'renderer_registry', 'controller_grant', 'agent_thread_runtime',
        'workload_audit_current', 'workload_audit_ledger', 'provider_call_receipt', 'model_invocation_receipt', 'agent_work_receipt', 'unproven_work_debt',
        'model_registry_current', 'model_routing_current', 'model_routing_blockers', 'sheet_current', 'api_sheet_current', 'api_model_routing_blockers', 'api_route_catalog', 'api_daemon_status', 'api_active_goal', 'api_canon_current', 'api_canon_versions',
        'nodes', 'manuals', 'route_catalog', 'edges', 'get_subtree'
    )
),
counts AS (
    SELECT
        (SELECT count(*) FROM lucidota_canon.model_registry) AS model_registry_count,
        (SELECT count(*) FROM lucidota_canon.provider_registry) AS provider_registry_count,
        (SELECT count(*) FROM lucidota_canon.workflow_registry) AS workflow_registry_count,
        (SELECT count(*) FROM lucidota_canon.skill_policy_current) AS skill_policy_current_count,
        (SELECT count(*) FROM lucidota_canon.chrono_current) AS chrono_current_count,
        (SELECT count(*) FROM lucidota_canon.payload_archive_status) AS payload_archive_status_count,
        (SELECT count(*) FROM lucidota_canon.todo_current) AS todo_current_count,
        (SELECT count(*) FROM lucidota_canon.canon_current) AS canon_current_count,
        (SELECT count(*) FROM lucidota_canon.canon_versions) AS canon_versions_count,
        (SELECT count(*) FROM lucidota_canon.command_registry) AS command_registry_count,
        (SELECT count(*) FROM lucidota_canon.schema_owner_manifest) AS schema_owner_manifest_count,
        (SELECT count(*) FROM lucidota_canon.surface_registry) AS surface_registry_count,
        (SELECT count(*) FROM lucidota_canon.renderer_registry) AS renderer_registry_count,
        (SELECT count(*) FROM lucidota_canon.controller_grant) AS controller_grant_count,
        (SELECT count(*) FROM lucidota_canon.agent_thread_runtime) AS agent_thread_runtime_count,
        (SELECT count(*) FROM lucidota_canon.root_law_docs) AS root_law_docs_count,
        (SELECT count(*) FROM lucidota_canon.api_root_law_docs) AS api_root_law_docs_count,
        (SELECT count(*) FROM lucidota_canon.api_route_catalog) AS api_route_catalog_count,
        (SELECT count(*) FROM lucidota_canon.api_test_execution_receipts) AS api_test_execution_receipts_count,
        (SELECT count(*) FROM lucidota_canon.cli_process_receipts) AS cli_process_receipts_count,
        (SELECT count(*) FROM lucidota_canon.flow_receipts) AS flow_receipts_count,
        (SELECT count(*) FROM lucidota_canon.bytewax_compact_windows) AS bytewax_compact_windows_count,
        (SELECT count(*) FROM lucidota_canon.workload_audit_ledger) AS workload_audit_ledger_count,
        (SELECT count(*) FROM lucidota_canon.workload_audit_current) AS workload_audit_current_count,
        (SELECT count(*) FROM lucidota_canon.workload_audit_telemetry_current) AS workload_audit_telemetry_current_count,
        (SELECT count(*) FROM lucidota_canon.provider_call_receipt) AS provider_call_receipt_count,
        (SELECT count(*) FROM lucidota_canon.model_invocation_receipt) AS model_invocation_receipt_count,
        (SELECT count(*) FROM lucidota_canon.agent_work_receipt) AS agent_work_receipt_count,
        (SELECT count(*) FROM lucidota_canon.unproven_work_debt) AS unproven_work_debt_count,
        (SELECT count(*) FROM lucidota_control.active_operation_mode) AS active_operation_mode_count,
        (SELECT count(*) FROM lucidota_canon.flow_specs) AS flow_specs_count,
        (SELECT count(*) FROM lucidota_canon.model_registry_current) AS model_registry_current_count,
        (SELECT count(*) FROM lucidota_canon.provider_current) AS provider_current_count,
        (SELECT count(*) FROM lucidota_canon.workflow_current) AS workflow_current_count,
        (SELECT count(*) FROM lucidota_canon.capability_current) AS capability_current_count,
        (SELECT count(*) FROM lucidota_canon.sheet_current) AS sheet_current_count,
        (SELECT count(*) FROM lucidota_canon.model_routing_current) AS model_routing_current_count,
        (SELECT count(*) FROM lucidota_canon.model_routing_blockers) AS model_routing_blockers_count
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
        'skill_layers', 'execution aids only',
        'manual_source', 'live route catalog + daemon status + current goal + active operation mode + workload telemetry + skill policy + capability registry + provider registry + workflow current packet + sub-orchestrators packet + Indy_READs exocortex packets + receipts packet + model routing blockers packet + rpc alias packets + queue + bytewax'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'queue_loop', 'indy_queue -> indy_responses -> bytewax_compact_windows -> cloud_packet',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', jsonb_build_object('route_ref', 'model_registry', 'count', counts.model_registry_count),
        'provider_registry', jsonb_build_object('route_ref', 'provider_registry', 'count', counts.provider_registry_count),
        'workflow_registry', jsonb_build_object('route_ref', 'workflow_registry', 'count', counts.workflow_registry_count),
        'root_orchestrator_current', jsonb_build_object('route_ref', 'root_orchestrator_current'),
        'skill_policy_current', jsonb_build_object('route_ref', 'skill_policy_current', 'count', counts.skill_policy_current_count),
        'model_registry_current', jsonb_build_object('route_ref', 'model_registry_current', 'count', counts.model_registry_current_count),
        'model_routing_current', COALESCE((SELECT jsonb_agg(to_jsonb(mrc) ORDER BY mrc.refreshed_at DESC) FROM lucidota_canon.model_routing_current mrc), '[]'::jsonb),
        'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb),
        'provider_current', jsonb_build_object('route_ref', 'provider_current', 'count', counts.provider_current_count),
        'workflow_current', jsonb_build_object('route_ref', 'workflow_current', 'count', counts.workflow_current_count),
        'capability_current', jsonb_build_object('route_ref', 'capability_current', 'count', counts.capability_current_count),
        'root_orchestrator_current', jsonb_build_array(jsonb_build_object('route_ref', 'root_orchestrator_current', 'orchestrator_id', 'ROOT_ORCHESTRATOR_CURRENT', 'status', 'active')),
        'sub_orchestrator_threads', jsonb_build_array(jsonb_build_object('route_ref', 'agent_thread_runtime', 'count', counts.agent_thread_runtime_count)),
        'sub_orchestrator_grants', jsonb_build_array(jsonb_build_object('route_ref', 'controller_grant', 'count', counts.controller_grant_count)),
        'chrono_current', jsonb_build_object('route_ref', 'chrono_current', 'count', counts.chrono_current_count),
        'payload_archive_status', jsonb_build_object('route_ref', 'payload_archive_status', 'count', counts.payload_archive_status_count),
        'todo_current', jsonb_build_object('route_ref', 'todo_current', 'count', counts.todo_current_count),
        'canon_current', jsonb_build_object('route_ref', 'canon_current', 'count', counts.canon_current_count),
        'canon_versions', jsonb_build_object('route_ref', 'canon_versions', 'count', counts.canon_versions_count),
        'command_registry', jsonb_build_object('route_ref', 'command_registry', 'count', counts.command_registry_count),
        'schema_owner_manifest', jsonb_build_object('route_ref', 'schema_owner_manifest', 'count', counts.schema_owner_manifest_count),
        'surface_registry', jsonb_build_object('route_ref', 'surface_registry', 'count', counts.surface_registry_count),
        'renderer_registry', jsonb_build_object('route_ref', 'renderer_registry', 'count', counts.renderer_registry_count),
        'controller_grant', jsonb_build_object('route_ref', 'controller_grant', 'count', counts.controller_grant_count),
        'agent_thread_runtime', jsonb_build_object('route_ref', 'agent_thread_runtime', 'count', counts.agent_thread_runtime_count),
        'root_law_docs', jsonb_build_object('route_ref', 'root_law_docs', 'count', counts.root_law_docs_count),
        'api_root_law_docs', jsonb_build_object('route_ref', 'api_root_law_docs', 'count', counts.api_root_law_docs_count),
        'api_route_catalog', jsonb_build_object('route_ref', 'api_route_catalog', 'count', counts.api_route_catalog_count),
        'api_test_execution_receipts', jsonb_build_object('route_ref', 'api_test_execution_receipts', 'count', counts.api_test_execution_receipts_count),
        'cli_process_receipts', jsonb_build_object('route_ref', 'cli_process_receipts', 'count', counts.cli_process_receipts_count),
        'api_flow_receipts', jsonb_build_object('route_ref', 'api_flow_receipts', 'count', counts.flow_receipts_count),
        'api_bytewax_windows', jsonb_build_object('route_ref', 'api_bytewax_windows', 'count', counts.bytewax_compact_windows_count),
        'workload_audit_ledger', jsonb_build_object('route_ref', 'workload_audit_ledger', 'count', counts.workload_audit_ledger_count),
        'workload_audit_current', jsonb_build_object('route_ref', 'workload_audit_current', 'count', counts.workload_audit_current_count),
        'workload_audit_telemetry_current', COALESCE((SELECT to_jsonb(watc) FROM lucidota_canon.workload_audit_telemetry_current watc LIMIT 1), '{}'::jsonb),
        'indy_reads_runtime', jsonb_build_object(
            'route_refs', jsonb_build_array(
                'indy_reads_self_model',
                'indy_reads_llmwiki_entry',
                'indy_reads_hunch_log',
                'indy_reads_learning_queue',
                'indy_reads_system_map',
                'indy_reads_mistake_ledger',
                'indy_reads_research_source',
                'indy_reads_metacognition_current',
                'indy_reads_target_model_loadout_current',
                'indy_reads_vram_coprocessor_fabric_current'
            ),
            'count', 10
        ),
        'provider_call_receipt', jsonb_build_object('route_ref', 'provider_call_receipt', 'count', counts.provider_call_receipt_count),
        'model_invocation_receipt', jsonb_build_object('route_ref', 'model_invocation_receipt', 'count', counts.model_invocation_receipt_count),
        'agent_work_receipt', jsonb_build_object('route_ref', 'agent_work_receipt', 'count', counts.agent_work_receipt_count),
        'unproven_work_debt', jsonb_build_object('route_ref', 'unproven_work_debt', 'count', counts.unproven_work_debt_count),
        'active_operation_mode', COALESCE((SELECT to_jsonb(aom) FROM lucidota_control.active_operation_mode aom LIMIT 1), '{}'::jsonb),
        'flow_specs', jsonb_build_object('route_ref', 'flow_specs', 'count', counts.flow_specs_count),
        'orchestration', jsonb_build_object(
            'mode', 'sub_orchestrator',
            'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
            'strict_priority_stack', lucidota_control.live_truth_priority_stack()
        )
    ) AS live_surface,
    jsonb_build_array(
        'manual_current',
        'daemon_status',
        'root_orchestrator_current',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
        'workload_audit_current',
        'workload_audit_ledger',
        'indy_queue',
        'indy_responses',
        'api_route_catalog',
        'indy_daemon_once',
        'indy_runtime_broker_snapshot',
        'manual_current'
    ) AS next_commands,
    route_refs.route_refs AS next_command_refs,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces,
    route_refs.route_refs,
    route_refs.route_refs AS surface_refs,
    jsonb_build_array('renderer_registry', 'command_registry') AS renderer_refs,
    jsonb_build_array('capability_current', 'capability_registry') AS capability_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'provider_secret_isolation', 'load through an explicit quarantine file or environment loader owned by the operator; no raw keys in chat, docs, SQL, or receipts'
    ) AS orchestration,
    jsonb_build_array(
        jsonb_build_object('route_ref', 'root_orchestrator_current', 'role', 'root_orchestrator', 'status', 'active')
    ) AS sub_orchestrators,
    jsonb_build_array(
        jsonb_build_object('thread_ref', 'agent_thread_runtime', 'status', 'active')
    ) AS sub_orchestrator_threads,
    jsonb_build_array(
        jsonb_build_object('grant_ref', 'controller_grant', 'status', 'active')
    ) AS sub_orchestrator_grants,
    jsonb_build_object(
        'count', counts.model_routing_blockers_count,
        'route_ref', 'model_routing_blockers'
    ) AS blockers,
    jsonb_build_object(
        'cli_process_receipts', true,
        'flow_receipts', true,
        'api_test_execution_receipts', true
    ) AS receipts,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law
FROM live_routes
CROSS JOIN daemon_rows
CROSS JOIN route_refs
CROSS JOIN counts
LEFT JOIN goal_row ON true;
$view$;
END$$;

GRANT USAGE ON SCHEMA lucidota_indy TO mfspx;
GRANT SELECT, INSERT, UPDATE ON lucidota_indy.book_scan, lucidota_indy.book_read_queue, lucidota_indy.book_note,
    lucidota_indy.lora_candidate, lucidota_indy.lora_adapter, lucidota_indy.training_job, lucidota_indy.book_receipt TO mfspx;

GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog, lucidota_canon.api_bible_manuals,
    lucidota_canon.api_bible_nodes, lucidota_canon.api_bible_edges, lucidota_canon.api_bible_route_catalog,
    lucidota_canon.canon_current, lucidota_canon.canon_versions, lucidota_canon.active_goal, lucidota_canon.api_workflow_registry,
    lucidota_canon.capability_registry, lucidota_canon.model_registry, lucidota_canon.provider_registry,
    lucidota_canon.workflow_registry, lucidota_canon.daemon_status, lucidota_canon.bytewax_compact_windows,
    lucidota_canon.indy_queue, lucidota_canon.indy_responses, lucidota_canon.book_source, lucidota_canon.book_scan,
    lucidota_canon.book_read_queue, lucidota_canon.book_note, lucidota_canon.lora_candidate, lucidota_canon.lora_adapter,
    lucidota_canon.training_job, lucidota_canon.book_receipt TO mfspx;

GRANT SELECT ON lucidota_canon.manual_current, lucidota_canon.api_route_catalog,
    lucidota_canon.book_source, lucidota_canon.book_scan, lucidota_canon.book_read_queue, lucidota_canon.book_note,
    lucidota_canon.lora_candidate, lucidota_canon.lora_adapter, lucidota_canon.training_job, lucidota_canon.book_receipt
    TO lucidota_postgrest_anon;
