-- DB-visible model registry current packet: expose live model counts, roles, and loadouts.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.model_registry_current AS
WITH model_rows AS (
    SELECT
        count(*) AS model_count,
        count(*) FILTER (WHERE active) AS active_count,
        count(DISTINCT role) AS role_count,
        count(DISTINCT loadout_id) AS loadout_count,
        count(*) FILTER (WHERE benchmark_status = 'accepted') AS accepted_count,
        max(updated_at) AS latest_model_updated_at,
        COALESCE(jsonb_agg(DISTINCT role ORDER BY role), '[]'::jsonb) AS role_names,
        COALESCE(jsonb_agg(DISTINCT loadout_id ORDER BY loadout_id), '[]'::jsonb) AS loadout_names
    FROM lucidota_canon.model_registry
),
role_rows AS (
    SELECT COALESCE(jsonb_object_agg(role, count_value ORDER BY role), '{}'::jsonb) AS role_breakdown
    FROM (
        SELECT role, count(*) AS count_value
        FROM lucidota_canon.model_registry
        GROUP BY role
    ) r
),
loadout_rows AS (
    SELECT COALESCE(jsonb_object_agg(loadout_id, count_value ORDER BY loadout_id), '{}'::jsonb) AS loadout_breakdown
    FROM (
        SELECT loadout_id, count(*) AS count_value
        FROM lucidota_canon.model_registry
        GROUP BY loadout_id
    ) l
),
active_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.role, m.model_id), '[]'::jsonb) AS active_models
    FROM (
        SELECT *
        FROM lucidota_canon.model_registry
        WHERE active
        ORDER BY role, model_id
        LIMIT 25
    ) m
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
),
goal_row AS (
    SELECT to_jsonb(g) AS current_goal
    FROM lucidota_canon.active_goal g
    ORDER BY updated_at DESC
    LIMIT 1
),
routing_current AS (
    SELECT
        role_admission_decisions,
        admitted_roles,
        honestly_skipped_roles,
        missing_roles
    FROM lucidota_canon.model_routing_current
    LIMIT 1
)
SELECT
    'model_registry_current'::text AS model_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'model_count', mr.model_count,
        'active_count', mr.active_count,
        'role_count', mr.role_count,
        'loadout_count', mr.loadout_count,
        'accepted_count', mr.accepted_count,
        'latest_model_updated_at', mr.latest_model_updated_at,
        'role_names', mr.role_names,
        'loadout_names', mr.loadout_names,
        'active_loadout_id', al.loadout_id,
        'active_loadout_target_gpu', al.target_gpu,
        'active_loadout_budget_vram_mb', al.budget_vram_mb,
        'active_loadout_slot_count', al.slot_count
    ) AS model_summary,
    role_rows.role_breakdown,
    loadout_rows.loadout_breakdown,
    active_rows.active_models,
    jsonb_build_object(
        'registry_truth', 'model_registry is the DB-visible model ledger; no script folklore authority',
        'routing_truth', 'active models are live choices; benchmark status and role are routing clues',
        'local_before_cloud', 'prefer local deterministic/needle/treelite/river lanes before cloud',
        'missing_roles', 'missing model roles are DB-visible blockers, not guesses',
        'honestly_skipped_roles', 'roles without a booted current runtime must carry precise skip reasons and fallback routes',
        'resident_loadout_state', COALESCE(
            CASE COALESCE(lgr.load_governor->>'decision', '')
                WHEN 'allow' THEN 'operational'
                WHEN 'defer' THEN 'partial'
                WHEN 'reject' THEN 'blocked'
                WHEN '' THEN 'unknown'
                ELSE 'unknown'
            END,
            'unknown'
        )
    ) AS routing_notes,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'model_registry_current',
        'model_registry',
        'model_routing_current',
        'provider_current'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
        'capability_current',
        'provider_current',
        'workflow_current',
        'model_routing_current',
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
        'active_models', active_rows.active_models
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
    routing_current.role_admission_decisions,
    routing_current.admitted_roles,
    routing_current.honestly_skipped_roles,
    routing_current.missing_roles
FROM model_rows mr
CROSS JOIN role_rows
CROSS JOIN loadout_rows
CROSS JOIN active_rows
CROSS JOIN active_loadout al
CROSS JOIN load_governor_row lgr
CROSS JOIN controller_row
CROSS JOIN thread_row
CROSS JOIN goal_row
CROSS JOIN routing_current;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'model_registry_current',
    'GET',
    '/model_registry_current',
    'Model registry current packet for counts, roles, and loadout coverage.',
    'lucidota_canon.model_registry_current',
    '{"limit":"1"}',
    '{"model_packet_id":"model_registry_current"}',
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

GRANT SELECT ON lucidota_canon.model_registry_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
