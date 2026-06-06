-- Lift todo_current into the same operator truth-spine shape as the other current packets.

BEGIN;

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
    CASE
        WHEN b.acceptance_test = 'read the live route and verify the route list/manual packet reflects the current API truth.' THEN
            'read the live route and verify the route list/manual packet reflects the current API truth.'
        ELSE b.acceptance_test
    END AS acceptance_test,
    b.receipt_requirement,
    b.functionality_contract,
    b.status,
    COUNT(i.item_uuid)::bigint AS item_count,
    COUNT(*) FILTER (WHERE i.parallelizable)::bigint AS parallel_item_count,
    COUNT(*) FILTER (WHERE i.serialized)::bigint AS serialized_item_count,
    COALESCE(
        jsonb_agg(
            CASE
                WHEN i.acceptance_test = 'read the live route and verify the route list/manual packet reflects the current API truth.' THEN
                    jsonb_set(
                        to_jsonb(i),
                        '{acceptance_test}',
                        to_jsonb('read the live route and verify the route list/manual packet reflects the current API truth.'::text),
                        true
                    )
                ELSE to_jsonb(i)
            END
            ORDER BY i.item_rank
        ) FILTER (WHERE i.item_uuid IS NOT NULL),
        '[]'::jsonb
    ) AS items,
    b.detail,
    b.created_at,
    b.updated_at,
    b.workflow_count,
    b.workflows_preserved,
    b.batch_kind,
    jsonb_build_object(
        'goal_id', b.batch_uuid::text,
        'title', COALESCE(b.objective_summary, 'Todo batch'),
        'status', b.status
    ) AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; receipts prove the thing happened.'
    ) AS db_law,
    jsonb_build_array(
        'todo_current',
        'api_todo_current'
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
    b.detail, b.created_at, b.updated_at, b.workflow_count, b.workflows_preserved, b.batch_kind
ORDER BY b.created_at DESC;

GRANT SELECT ON lucidota_canon.todo_current TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
