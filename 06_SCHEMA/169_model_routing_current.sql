-- DB-visible model routing packet: expose real local coverage and missing role blockers.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.model_routing_current AS
WITH model_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_models,
        count(*) FILTER (WHERE active AND role = 'router') AS router_models,
        count(*) FILTER (WHERE active AND role = 'listener') AS listener_models,
        count(*) FILTER (WHERE active AND role = 'heavy_hitter') AS heavy_hitter_models,
        max(updated_at) AS latest_model_updated_at
    FROM lucidota_canon.model_registry
),
provider_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_providers,
        COALESCE(
            jsonb_agg(provider_key ORDER BY provider_key) FILTER (WHERE active),
            '[]'::jsonb
        ) AS active_provider_keys
    FROM lucidota_canon.provider_registry
),
role_names AS (
    SELECT role, admission_class, admission_decision, locality, artifact_path, launcher_path, port_health, ram_budget_mb, vram_budget_mb, skip_reason, fallback_route, evidence_refs
    FROM (VALUES
        ('router', 'ADMITTED', 'ADMITTED_LIVE', 'local', '03_VAULT/models/needle/needle.pkl', '.venv/bin/python + scripts/lucidota_start_needle_swarm.sh', 'http://127.0.0.1:8090/health', 384, 0, '', 'none', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('thinker', 'ADMITTED', 'ADMITTED_LIVE', 'local', '03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', 'scripts/lucidota_start_deepseek_llama.sh', 'http://127.0.0.1:8080/health', 2048, 1500, '', 'deterministic_fastlane_when_governor_defers', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('watcher', 'ADMITTED', 'ADMITTED_LIVE', 'local', 'BOOKS/.indy_reads', 'scripts/lucidota_start_indy_reads_watcher.sh', 'file-watch/worker receipt only', 256, 0, '', 'ironclaw-indy-reads service', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('listener', 'ADMITTED', 'ADMITTED_LIVE', 'local', '03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf', 'scripts/lucidota_start_mamba_llama.sh', 'http://127.0.0.1:8081/health', 4096, 0, '', 'watcher + router lanes', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('heavy_hitter', 'ADMITTED', 'ADMITTED_LIVE', 'local', '03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', 'scripts/lucidota_start_deepseek_llama.sh', 'http://127.0.0.1:8080/health', 2048, 1500, '', 'router/template lanes when not selected', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('embedder', 'HONESTLY_SKIPPED', 'SKIPPED_MEMORY_PRESSURE_NO_HEALTHY_PORT', 'local', '04_RUNTIME/models/bge-m3-q8_0.gguf', 'scripts/lucidota_bge_fleet.sh', 'http://127.0.0.1:8101/health not listening in current smoke', 4096, 0, 'governor rung=3 severe memory pressure; BGE fleet intentionally not exposed as live', 'text/hash/manual surfaces without embedding', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('reranker', 'HONESTLY_SKIPPED', 'SKIPPED_NO_DEDICATED_RUNTIME', 'local', '', '', 'none', 0, 0, 'no dedicated reranker launcher/health receipt is active; do not pretend a reranker exists', 'router + deterministic scoring', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('classifier', 'HONESTLY_SKIPPED', 'SKIPPED_DETERMINISTIC_ROUTE_ONLY', 'local', '', 'scripts/language_router.py', 'no service port; library call only', 0, 0, 'classifier role is currently deterministic code, not a booted model service', 'language_router + route catalog', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('summarizer', 'HONESTLY_SKIPPED', 'SKIPPED_NO_DEDICATED_RUNTIME', 'local', '', '', 'none', 0, 0, 'no dedicated local summarizer model/launcher/health receipt is active', 'Indy_READs composed response + template lanes', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json')),
        ('treelite_gate', 'HONESTLY_SKIPPED', 'SKIPPED_ARTIFACT_NOT_BOOTED_AS_SERVICE', 'local', '03_VAULT/router/treelite_router_v0.tl', 'inline/router artifacts only', 'no service port', 128, 0, 'treelite artifact exists but is not a booted current gate service', 'in-memory deterministic router scores', jsonb_build_array('05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json'))
    ) AS roles(role, admission_class, admission_decision, locality, artifact_path, launcher_path, port_health, ram_budget_mb, vram_budget_mb, skip_reason, fallback_route, evidence_refs)
),
role_rows AS (
    SELECT
        COALESCE(
            jsonb_object_agg(
                rn.role,
                COALESCE(to_jsonb(m), '{}'::jsonb)
                || jsonb_build_object(
                    'role', rn.role,
                    'admission_class', rn.admission_class,
                    'admission_decision', rn.admission_decision,
                    'locality', rn.locality,
                    'artifact_path', rn.artifact_path,
                    'launcher_path', rn.launcher_path,
                    'port_health', rn.port_health,
                    'ram_budget_mb', rn.ram_budget_mb,
                    'vram_budget_mb', rn.vram_budget_mb,
                    'skip_reason', rn.skip_reason,
                    'fallback_route', rn.fallback_route,
                    'evidence_refs', rn.evidence_refs,
                    'registry_row_active', (m.model_id IS NOT NULL)
                )
                ORDER BY rn.role
            ),
            '{}'::jsonb
        ) AS local_model_roles,
        COALESCE(
            jsonb_object_agg(
                rn.role,
                jsonb_build_object(
                    'admission_class', rn.admission_class,
                    'admission_decision', rn.admission_decision,
                    'artifact_path', rn.artifact_path,
                    'launcher_path', rn.launcher_path,
                    'port_health', rn.port_health,
                    'ram_budget_mb', rn.ram_budget_mb,
                    'vram_budget_mb', rn.vram_budget_mb,
                    'skip_reason', rn.skip_reason,
                    'fallback_route', rn.fallback_route,
                    'evidence_refs', rn.evidence_refs,
                    'registry_row_active', (m.model_id IS NOT NULL)
                )
                ORDER BY rn.role
            ),
            '{}'::jsonb
        ) AS role_admission_decisions,
        COALESCE(jsonb_agg(rn.role ORDER BY rn.role) FILTER (WHERE rn.admission_class = 'ADMITTED'), '[]'::jsonb) AS admitted_roles,
        COALESCE(jsonb_agg(rn.role ORDER BY rn.role) FILTER (WHERE rn.admission_class = 'HONESTLY_SKIPPED'), '[]'::jsonb) AS honestly_skipped_roles
    FROM role_names rn
    LEFT JOIN LATERAL (
        SELECT *
        FROM lucidota_canon.model_registry mr
        WHERE mr.active
          AND mr.role = rn.role
        LIMIT 1
    ) m ON true
),
missing_rows AS (
    SELECT
        COALESCE(
            jsonb_agg(role ORDER BY role),
            '[]'::jsonb
        ) AS missing_roles
    FROM (
        SELECT rn.role
        FROM role_names rn
        LEFT JOIN lucidota_canon.model_registry mr
            ON mr.active AND mr.role = rn.role
        WHERE mr.model_id IS NULL
          AND rn.admission_class NOT IN ('ADMITTED', 'HONESTLY_SKIPPED')
    ) missing
),
active_loadout AS (
    SELECT
        rl.loadout_id,
        rl.active,
        rl.description,
        rl.target_gpu,
        rl.budget_vram_mb,
        rl.created_at,
        COALESCE(count(s.slot_name), 0) AS slot_count,
        COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'slot_name', s.slot_name,
                    'model_id', s.model_id,
                    'instance_count', s.instance_count,
                    'priority', s.priority,
                    'expected_vram_mb', s.expected_vram_mb,
                    'notes', s.notes
                )
                ORDER BY s.priority, s.slot_name
            ) FILTER (WHERE s.slot_name IS NOT NULL),
            '[]'::jsonb
        ) AS slots
    FROM lucidota_runtime.resident_loadout rl
    LEFT JOIN lucidota_runtime.resident_loadout_slot s
        ON s.loadout_id = rl.loadout_id
    WHERE rl.active = true
    GROUP BY rl.loadout_id, rl.active, rl.description, rl.target_gpu, rl.budget_vram_mb, rl.created_at
    ORDER BY rl.created_at DESC
    LIMIT 1
),
load_governor_row AS (
    SELECT
        al.loadout_id,
        to_jsonb(d) AS load_governor
    FROM active_loadout al
    LEFT JOIN LATERAL (
        SELECT *
        FROM lucidota_runtime.load_governor_decision d
        WHERE d.loadout_id = al.loadout_id
        ORDER BY d.created_at DESC
        LIMIT 1
    ) d ON true
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
controller_row AS (
    SELECT to_jsonb(g) AS controller_grant
    FROM lucidota_canon.controller_grant g
    WHERE g.grant_key = 'default_local_operator'
    ORDER BY g.updated_at DESC
    LIMIT 1
),
thread_row AS (
    SELECT to_jsonb(t) AS agent_thread_runtime
    FROM lucidota_canon.agent_thread_runtime t
    WHERE t.thread_key = 'root_operator_thread'
    ORDER BY t.updated_at DESC
    LIMIT 1
)
SELECT
    'model_routing_current'::text AS routing_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'active_models', model_rows.active_models,
        'router_models', model_rows.router_models,
        'listener_models', model_rows.listener_models,
        'heavy_hitter_models', model_rows.heavy_hitter_models,
        'latest_model_updated_at', model_rows.latest_model_updated_at,
        'active_loadout_id', al.loadout_id,
        'active_loadout_target_gpu', al.target_gpu,
        'active_loadout_budget_vram_mb', al.budget_vram_mb,
        'active_loadout_slot_count', al.slot_count
    ) AS model_registry,
    jsonb_build_object(
        'active_providers', provider_rows.active_providers,
        'active_provider_keys', provider_rows.active_provider_keys
    ) AS provider_registry,
    role_rows.local_model_roles,
    missing_rows.missing_roles,
    jsonb_build_object(
        'deterministic_first', true,
        'prefer_local_lanes', true,
        'preferred_route_order', ARRAY['router', 'thinker', 'watcher', 'listener', 'heavy_hitter', 'classifier', 'summarizer', 'embedder', 'reranker', 'treelite_gate'],
        'missing_roles_mean_blocker', true,
        'honestly_skipped_roles_are_not_fantasy_models', true,
        'next_action', 'use admitted live local lanes; treat honestly skipped roles as explicit fallback decisions, not missing fantasy services',
        'drift_control', 'if repeated work stops being useful, convert it into a workflow row and prune the loser path'
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'model_routing_current',
        'model_routing_blockers',
        'model_registry_current',
        'provider_current'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'model_registry_current',
        'workflow_current',
        'model_routing_blockers',
        'sheet_current',
        'todo_current',
        'command_registry',
        'surface_registry',
        'renderer_registry',
        'schema_owner_manifest',
        'controller_grant',
        'agent_thread_runtime',
        'model_registry'
    ) AS next_command_refs,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'active_models', model_rows.active_models,
        'active_providers', provider_rows.active_providers
    ) AS orchestration,
    controller_row.controller_grant,
    thread_row.agent_thread_runtime,
    jsonb_build_object(
        'loadout_id', al.loadout_id,
        'active', al.active,
        'description', al.description,
        'target_gpu', al.target_gpu,
        'budget_vram_mb', al.budget_vram_mb,
        'created_at', al.created_at,
        'slot_count', al.slot_count,
        'slots', al.slots
    ) AS resident_loadout,
    jsonb_build_object(
        'loadout_id', lgr.loadout_id,
        'decision', COALESCE(lgr.load_governor->>'decision', 'unknown'),
        'observed_free_mb', NULLIF(lgr.load_governor->>'observed_free_mb', '')::integer,
        'observed_used_mb', NULLIF(lgr.load_governor->>'observed_used_mb', '')::integer,
        'headroom_mb', NULLIF(lgr.load_governor->>'headroom_mb', '')::integer,
        'estimated_required_mb', NULLIF(lgr.load_governor->>'estimated_required_mb', '')::integer,
        'rationale', COALESCE(lgr.load_governor->>'rationale', ''),
        'created_at', lgr.load_governor->>'created_at',
        'status', CASE COALESCE(lgr.load_governor->>'decision', '')
            WHEN 'allow' THEN 'operational'
            WHEN 'defer' THEN 'partial'
            WHEN 'reject' THEN 'blocked'
            WHEN '' THEN 'unknown'
            ELSE 'unknown'
        END
    ) AS resident_loadout_status,
    lgr.load_governor AS load_governor,
    role_rows.role_admission_decisions,
    role_rows.admitted_roles,
    role_rows.honestly_skipped_roles
FROM model_rows
CROSS JOIN provider_rows
CROSS JOIN role_rows
CROSS JOIN missing_rows
CROSS JOIN active_loadout al
CROSS JOIN load_governor_row lgr
CROSS JOIN goal_row
CROSS JOIN controller_row
CROSS JOIN thread_row;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'model_routing_current',
    'GET',
    '/model_routing_current',
    'Current local model coverage, provider coverage, and missing-role blockers for the ML router.',
    'lucidota_canon.model_routing_current',
    '{"limit":"1"}',
    '{"routing_packet_id":"model_routing_current"}',
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

GRANT SELECT ON lucidota_canon.model_routing_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
