-- Orchestrator PostgREST registry surfaces.

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
        m.updated_at
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
    jsonb_build_array(
        'curl -sS http://127.0.0.1:3000/daemon_status?limit=1',
        'curl -sS http://127.0.0.1:3000/active_goal?limit=1',
        './luci daemon status --json',
        './luci api daemon status --json'
    ) AS next_commands
FROM latest_heartbeats hb
LEFT JOIN latest_facts lf
    ON lf.subsystem = hb.daemon_name
   OR lf.subsystem = split_part(hb.daemon_name, '-', 1);

GRANT SELECT ON
    lucidota_canon.capability_registry,
    lucidota_canon.model_registry,
    lucidota_canon.provider_registry,
    lucidota_canon.workflow_registry,
    lucidota_canon.daemon_status
TO lucidota_postgrest_anon, mfspx;
