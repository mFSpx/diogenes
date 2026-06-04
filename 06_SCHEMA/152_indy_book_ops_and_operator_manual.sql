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
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt'
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
        'manual_source', 'live route catalog + daemon status + current goal'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority'
    ) AS work_order_flow,
    jsonb_build_object(
        'current_goal', goal_row.current_goal,
        'daemon_status', daemon_rows.daemon_status,
        'model_registry', model_rows.model_registry,
        'provider_registry', provider_rows.provider_registry,
        'workflow_registry', workflow_rows.workflow_registry
    ) AS live_surface,
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
        'curl -sS http://127.0.0.1:3000/daemon_status?limit=1',
        '.venv/bin/python scripts/indy_daemon.py --once --json',
        '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
        '.venv/bin/python scripts/luci_help_manual.py manual --json'
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
LEFT JOIN goal_row ON true;

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
