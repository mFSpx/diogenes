-- Lift the active goal packet into an operator-readable truth spine.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.active_goal AS
SELECT
    goal_id,
    title,
    status,
    active_prompt_path,
    active_prompt_hash,
    current_handoff_path,
    detail,
    created_at,
    updated_at,
    jsonb_build_object(
        'goal_id', goal_id,
        'title', title,
        'status', status,
        'current_handoff_path', current_handoff_path
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'active_goal',
        'api_active_goal'
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
FROM lucidota_control.active_goal;

GRANT SELECT ON lucidota_canon.active_goal TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
