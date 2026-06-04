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
    b.acceptance_test,
    b.receipt_requirement,
    b.functionality_contract,
    b.status,
    COUNT(i.item_uuid)::bigint AS item_count,
    COUNT(*) FILTER (WHERE i.parallelizable)::bigint AS parallel_item_count,
    COUNT(*) FILTER (WHERE i.serialized)::bigint AS serialized_item_count,
    COALESCE(jsonb_agg(to_jsonb(i) ORDER BY i.item_rank) FILTER (WHERE i.item_uuid IS NOT NULL), '[]'::jsonb) AS items,
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
        'curl -sS http://127.0.0.1:3000/todo_current?limit=1',
        './luci todo current --json',
        './luci api todo current --json'
    ) AS next_commands
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
