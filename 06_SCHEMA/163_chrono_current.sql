-- Chrono current packet: align prompt ledger, work ledger, execution history, and learning-loop state.
-- The route is a bounded live packet for operator inspection, not a new authority plane.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.chrono_current AS
WITH prompt_rows AS (
    SELECT
        count(*) AS prompt_count,
        count(*) FILTER (WHERE status = 'filed') AS filed_count,
        count(*) FILTER (WHERE status = 'decomposed') AS decomposed_count,
        count(*) FILTER (WHERE cardinality(linked_work_order_uuid) = 0) AS unlinked_count,
        max(received_at) AS latest_prompt_received_at,
        max(updated_at) AS latest_prompt_updated_at
    FROM lucidota_control.prompt_record
),
work_rows AS (
    SELECT
        count(*) AS work_order_count,
        count(*) FILTER (WHERE status = 'queued') AS queued_work_orders,
        count(*) FILTER (WHERE status = 'running') AS running_work_orders,
        count(*) FILTER (WHERE status = 'succeeded') AS succeeded_work_orders,
        count(*) FILTER (WHERE status = 'failed') AS failed_work_orders,
        max(created_at) AS latest_work_order_at,
        max(updated_at) AS latest_work_order_updated_at
    FROM lucidota_control.work_order
),
receipt_rows AS (
    SELECT
        count(*) AS work_receipt_count,
        max(created_at) AS latest_work_receipt_at
    FROM lucidota_control.work_receipt
),
envelope_rows AS (
    SELECT
        count(*) AS event_envelope_count,
        max(created_at) AS latest_event_envelope_at
    FROM lucidota_control.event_envelope
),
bytewax_rows AS (
    SELECT
        count(*) AS bytewax_window_count,
        count(*) FILTER (WHERE needs_cloud_reasoning) AS cloud_needed_windows,
        max(created_at) AS latest_bytewax_window_at
    FROM lucidota_canon.bytewax_compact_windows
),
river_rows AS (
    SELECT
        count(*) AS river_run_count,
        count(*) FILTER (WHERE status = 'succeeded') AS river_success_runs,
        count(*) FILTER (WHERE status = 'failed') AS river_failed_runs,
        max(created_at) AS latest_river_run_at
    FROM lucidota_learning.river_run
),
registry_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_models,
        count(*) FILTER (WHERE active AND role = 'router') AS router_models,
        count(*) FILTER (WHERE active AND role = 'listener') AS listener_models,
        count(*) FILTER (WHERE active AND role = 'heavy_hitter') AS heavy_hitter_models
    FROM lucidota_canon.model_registry
),
provider_rows AS (
    SELECT
        count(*) FILTER (WHERE active) AS active_providers
    FROM lucidota_canon.provider_registry
),
workflow_rows AS (
    SELECT
        count(*) FILTER (WHERE status = 'active') AS active_workflows,
        count(*) FILTER (WHERE status = 'deprecated') AS deprecated_workflows
    FROM lucidota_canon.workflow_registry
)
SELECT
    'chrono_current'::text AS chrono_packet_id,
    now() AS refreshed_at,
    jsonb_build_object(
        'prompt_count', pr.prompt_count,
        'filed_count', pr.filed_count,
        'decomposed_count', pr.decomposed_count,
        'unlinked_count', pr.unlinked_count,
        'latest_prompt_received_at', pr.latest_prompt_received_at,
        'latest_prompt_updated_at', pr.latest_prompt_updated_at
    ) AS prompt_ledger,
    jsonb_build_object(
        'work_order_count', wr.work_order_count,
        'queued_work_orders', wr.queued_work_orders,
        'running_work_orders', wr.running_work_orders,
        'succeeded_work_orders', wr.succeeded_work_orders,
        'failed_work_orders', wr.failed_work_orders,
        'latest_work_order_at', wr.latest_work_order_at,
        'latest_work_order_updated_at', wr.latest_work_order_updated_at
    ) AS work_ledger,
    jsonb_build_object(
        'work_receipt_count', rr.work_receipt_count,
        'latest_work_receipt_at', rr.latest_work_receipt_at,
        'event_envelope_count', er.event_envelope_count,
        'latest_event_envelope_at', er.latest_event_envelope_at
    ) AS execution_history,
    jsonb_build_object(
        'bytewax_window_count', br.bytewax_window_count,
        'cloud_needed_windows', br.cloud_needed_windows,
        'latest_bytewax_window_at', br.latest_bytewax_window_at,
        'river_run_count', rv.river_run_count,
        'river_success_runs', rv.river_success_runs,
        'river_failed_runs', rv.river_failed_runs,
        'latest_river_run_at', rv.latest_river_run_at
    ) AS learning_loop,
    jsonb_build_object(
        'active_models', mr.active_models,
        'router_models', mr.router_models,
        'listener_models', mr.listener_models,
        'heavy_hitter_models', mr.heavy_hitter_models,
        'active_providers', prr.active_providers,
        'active_workflows', wrk.active_workflows,
        'deprecated_workflows', wrk.deprecated_workflows
    ) AS routing_registry,
    jsonb_build_object(
        'prompt_recent', 'use /prompt_recent for the latest filed prompts',
        'prompt_catalog_status', 'use /prompt_catalog_status for ledger counts',
        'bytewax_compact_windows', 'use /bytewax_compact_windows for compact stream-state rows',
        'manual_current', 'use /manual_current for current operator manual and route map'
    ) AS inspection_notes,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'daemon_status',
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
        'skill_policy_current',
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
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'prompt_count', pr.prompt_count,
        'work_order_count', wr.work_order_count,
        'river_run_count', rv.river_run_count
    ) AS orchestration
FROM prompt_rows pr
CROSS JOIN work_rows wr
CROSS JOIN receipt_rows rr
CROSS JOIN envelope_rows er
CROSS JOIN bytewax_rows br
CROSS JOIN river_rows rv
CROSS JOIN registry_rows mr
CROSS JOIN provider_rows prr
CROSS JOIN workflow_rows wrk;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES (
    'chrono_current',
    'GET',
    '/chrono_current',
    'Chrono alignment packet for prompt ledger, work ledger, execution history, and learning-loop status.',
    'lucidota_canon.chrono_current',
    '{"limit":"1"}',
    '{"chrono_packet_id":"chrono_current"}',
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

GRANT SELECT ON lucidota_canon.chrono_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
