-- LUCIDOTA product safe surface: read views and local /flow persistence.
-- Non-destructive. Physical legacy tables remain; operators/agents get a small
-- PostgREST-readable surface through lucidota_canon.

CREATE SCHEMA IF NOT EXISTS luci_flow;

CREATE TABLE IF NOT EXISTS luci_flow.flow_spec (
    flow_id text PRIMARY KEY,
    name text NOT NULL DEFAULT 'untitled flow',
    status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','staged','validated','run','promoted','rollback','rolled_back')),
    flow_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    nodes jsonb NOT NULL DEFAULT '[]'::jsonb,
    edges jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_by text NOT NULL DEFAULT current_user,
    receipt_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS luci_flow.flow_receipt (
    receipt_id text PRIMARY KEY,
    flow_id text REFERENCES luci_flow.flow_spec(flow_id) ON DELETE SET NULL,
    action text NOT NULL CHECK (action IN ('save','stage','validate','run','promote','rollback')),
    status text NOT NULL DEFAULT 'ok',
    output_path text,
    output_hash text,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.active_goal (
    goal_id text PRIMARY KEY,
    title text NOT NULL,
    status text NOT NULL DEFAULT 'active',
    active_prompt_path text,
    active_prompt_hash text,
    current_handoff_path text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'ironclaw'
          AND c.relname = 'waking_dialogue_stream'
          AND pg_get_userbyid(c.relowner) = current_user
    ) THEN
        EXECUTE $sql$
            ALTER TABLE IF EXISTS ironclaw.waking_dialogue_stream
                ADD COLUMN IF NOT EXISTS last_response_id text,
                ADD COLUMN IF NOT EXISTS last_response_body text,
                ADD COLUMN IF NOT EXISTS last_response_body_sha256 text,
                ADD COLUMN IF NOT EXISTS response_queued_at timestamptz,
                ADD COLUMN IF NOT EXISTS response_delivery_status text
        $sql$;
        EXECUTE $sql$
            CREATE INDEX IF NOT EXISTS ix_waking_dialogue_stream_last_response_id
                ON ironclaw.waking_dialogue_stream (last_response_id)
                WHERE last_response_id IS NOT NULL
        $sql$;
    END IF;
END$$;

CREATE OR REPLACE VIEW lucidota_canon.canon_current AS
WITH goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
)
SELECT
    node_id,
    parent_id,
    node_sort_key,
    manual_id,
    title,
    node_kind,
    ontology_tags,
    status,
    version,
    hash_current,
    updated_at,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'canon_current',
        'canon_versions'
    ) AS next_commands,
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
        'todo_current',
        'canon_current',
        'canon_versions',
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
        'strict_priority_stack', lucidota_control.live_truth_priority_stack()
    ) AS orchestration
FROM lucidota_canon.bible_nodes
CROSS JOIN goal_row
WHERE valid_to IS NULL;

CREATE OR REPLACE VIEW lucidota_canon.canon_versions AS
WITH rows AS (
    SELECT history_id::text AS version_id, node_id, manual_id, version,
        payload AS diff_from_previous, hash_current AS content_hash,
        archived_at AS promoted_at
    FROM lucidota_canon.bible_history
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.indy_queue AS
WITH rows AS (
    SELECT id::text AS id, received_at, sender_id, room_id, event_id, raw_text,
        clean_text, extracted_entities, processed_status, receipt_id, created_at
    FROM ironclaw.waking_dialogue_stream
    WHERE comms_channel = 'matrix'
      AND processed_status = 'queued'
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.flow_specs AS
WITH rows AS (
    SELECT flow_id, name, status, flow_json, nodes, edges, created_by,
        receipt_id, created_at, updated_at
    FROM luci_flow.flow_spec
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.flow_receipts AS
WITH rows AS (
    SELECT receipt_id, flow_id, action, status, output_path, output_hash,
        metrics, created_at
    FROM luci_flow.flow_receipt
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.capability_registry AS
WITH rows AS (
    SELECT capability_key, capability_group, capability_name, lifecycle_status,
        run_state, workflow_name, command, detail, created_at, updated_at
    FROM lucidota_investigation.capability_registry
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.model_registry AS
WITH rows AS (
    SELECT
        l.loadout_id,
        l.active,
        l.description AS loadout_description,
        l.target_gpu,
        l.budget_vram_mb,
        s.slot_name,
        s.instance_count,
        s.priority,
        s.expected_vram_mb AS slot_expected_vram_mb,
        s.notes AS slot_notes,
        m.model_id,
        m.role,
        m.source_url,
        m.local_path,
        m.license,
        m.parameter_count,
        m.quantization,
        m.expected_vram_mb,
        m.benchmark_status,
        m.notes,
        m.created_at,
        m.updated_at,
        (
            SELECT jsonb_build_object(
                'decision', d.decision,
                'status', CASE d.decision
                    WHEN 'allow' THEN 'operational'
                    WHEN 'defer' THEN 'partial'
                    WHEN 'reject' THEN 'blocked'
                    ELSE 'unknown'
                END,
                'observed_free_mb', d.observed_free_mb,
                'observed_used_mb', d.observed_used_mb,
                'estimated_required_mb', d.estimated_required_mb,
                'headroom_mb', d.headroom_mb,
                'rationale', d.rationale,
                'created_at', d.created_at
            )
            FROM lucidota_runtime.load_governor_decision d
            WHERE d.loadout_id = l.loadout_id
            ORDER BY d.created_at DESC
            LIMIT 1
        ) AS load_governor_status
    FROM lucidota_runtime.resident_loadout l
    JOIN lucidota_runtime.resident_loadout_slot s ON s.loadout_id = l.loadout_id
    JOIN lucidota_runtime.model_candidate m ON m.model_id = s.model_id
    WHERE l.active
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.provider_registry AS
SELECT *
FROM (
    VALUES
        ('codex', 'cloud_orchestrator', 'code_patch_and_test', 'OpenAI Codex / GPT-5.x CLI', 'gpt-5.4-mini', true, 'bounded patch executor and prompt compiler'),
        ('vibe', 'cloud_orchestrator', 'broad_wiring_and_iteration', 'Vibe CLI / codestral family', 'codestral', true, 'broad wiring, iteration, and review lane'),
        ('groq', 'cloud_provider', 'cheap_synthesis_and_review', 'Groq OpenAI-compatible API', 'llama-3.3-70b-versatile', true, 'cloud synthesis/review and goal delegation'),
        ('local_model', 'local_runtime', 'classification_summarization_routing', 'resident local model runtime', 'resident_loadout', true, 'local resident models and loadout slots'),
        ('treelite', 'deterministic_model_runtime', 'fast_scoring_and_gating', 'Treelite / tiny model gates', 'treelite_router', true, 'cheap deterministic scoring and gating'),
        ('bytewax', 'stream_runtime', 'windowing_and_compaction', 'Bytewax stream workers', 'bytewax_abductive_blender', true, 'stream compaction into compact rows'),
        ('mamba', 'stream_runtime', 'watch_and_route', 'Mamba queue/watch loops', 'mamba_db_watch', true, 'queue watching and route selection')
) AS rows(
    provider_key,
    provider_kind,
    preferred_for,
    execution_surface,
    default_model,
    active,
    notes
);

CREATE OR REPLACE VIEW lucidota_canon.workflow_registry AS
WITH rows AS (
    SELECT * FROM lucidota_canon.api_workflow_registry
)
SELECT * FROM rows;

CREATE OR REPLACE VIEW lucidota_canon.daemon_status AS
WITH latest_heartbeats AS (
    SELECT DISTINCT ON (daemon_name)
        hb.*
    FROM ironclaw.daemon_heartbeats hb
    ORDER BY daemon_name, created_at DESC
),
latest_facts AS (
    SELECT DISTINCT ON (subsystem)
        subsystem,
        fact_key,
        fact_value,
        evidence_refs,
        derived_at
    FROM lucidota_control.runtime_status_fact
    ORDER BY subsystem, derived_at DESC
)
SELECT
    hb.heartbeat_uuid,
    hb.daemon_name,
    hb.heartbeat_kind,
    hb.host_name,
    hb.process_id,
    hb.transport_socket,
    hb.socket_active,
    hb.terminal_active,
    hb.batch_size,
    hb.river_state,
    hb.telemetry,
    hb.detail,
    hb.created_at AS heartbeat_created_at,
    lf.fact_key AS status_fact_key,
    lf.fact_value AS status_fact_value,
    lf.evidence_refs AS status_fact_evidence_refs,
    lf.derived_at AS status_fact_derived_at,
    (
        SELECT jsonb_build_object(
            'goal_id', ag.goal_id,
            'title', ag.title,
            'status', ag.status,
            'current_handoff_path', ag.current_handoff_path
        )
        FROM lucidota_canon.active_goal ag
        LIMIT 1
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array('daemon_status') AS next_commands,
    jsonb_build_array(
        'manual_current',
        'active_goal',
        'daemon_status',
        'api_daemon_status',
        'root_orchestrator_current',
        'payload_archive_status',
        'cli_process_receipts',
        'capability_current',
        'capability_registry',
        'model_registry',
        'model_registry_current',
        'provider_registry',
        'provider_current',
        'workflow_registry',
        'workflow_current',
        'model_routing_current',
        'model_routing_blockers',
        'sheet_current',
        'skill_policy_current',
        'todo_current',
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'command_registry',
        'controller_grant',
        'agent_thread_runtime'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'daemon_name', hb.daemon_name,
        'heartbeat_kind', hb.heartbeat_kind
    ) AS orchestration
FROM latest_heartbeats hb
LEFT JOIN latest_facts lf
    ON lf.subsystem = hb.daemon_name
   OR lf.subsystem = split_part(hb.daemon_name, '-', 1);

REVOKE ALL ON ALL TABLES IN SCHEMA lucidota_canon FROM lucidota_postgrest_anon;
GRANT SELECT ON
    lucidota_canon.api_bible_edges,
    lucidota_canon.api_bible_manuals,
    lucidota_canon.api_bible_nodes,
    lucidota_canon.api_bible_route_catalog,
    lucidota_canon.api_workflow_registry,
    lucidota_canon.manual_current,
    lucidota_canon.canon_current,
    lucidota_canon.canon_versions,
    lucidota_canon.capability_registry,
    lucidota_canon.indy_queue,
    lucidota_canon.indy_responses,
    lucidota_canon.model_registry,
    lucidota_canon.model_registry_current,
    lucidota_canon.provider_registry,
    lucidota_canon.workflow_registry,
    lucidota_canon.daemon_status,
    lucidota_canon.root_orchestrator_current,
    lucidota_canon.capability_current,
    lucidota_canon.flow_specs,
    lucidota_canon.flow_receipts,
    lucidota_canon.active_goal,
    lucidota_canon.controller_grant,
    lucidota_canon.agent_thread_runtime,
    lucidota_canon.prompts_filed,
    lucidota_canon.prompt_work_order_links,
    lucidota_canon.prompt_recent,
    lucidota_canon.prompt_unlinked,
    lucidota_canon.prompt_catalog_status,
    lucidota_canon.percyphon_current,
    lucidota_canon.percyphon_village_matrix,
    lucidota_canon.elastic_shape_current,
    lucidota_canon.indy_attention_pressure_current,
    lucidota_canon.shape_residuals_current
TO lucidota_postgrest_anon;

-- Keep the PostgREST anon reader aligned with the live current-surface bundle.
GRANT SELECT ON ALL TABLES IN SCHEMA lucidota_canon TO lucidota_postgrest_anon;
