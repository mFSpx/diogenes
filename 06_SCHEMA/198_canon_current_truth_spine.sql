-- Lift the canon current packet into an operator-readable truth spine.

BEGIN;

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

GRANT SELECT ON lucidota_canon.canon_current TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
