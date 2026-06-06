-- Recovery stub for the root orchestrator current packet.

BEGIN;
DROP VIEW IF EXISTS lucidota_canon.root_orchestrator_current CASCADE;
CREATE VIEW lucidota_canon.root_orchestrator_current AS
WITH goal_row AS (
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
    SELECT jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.model_registry),
        'latest_updated_at', (SELECT max(updated_at) FROM lucidota_canon.model_registry),
        'model_refs', COALESCE((SELECT jsonb_agg(model_id ORDER BY updated_at DESC) FROM (SELECT model_id, updated_at FROM lucidota_canon.model_registry ORDER BY updated_at DESC LIMIT 15) m), '[]'::jsonb)
    ) AS model_registry
),
provider_rows AS (
    SELECT jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.provider_registry),
        'active_count', (SELECT count(*) FILTER (WHERE active) FROM lucidota_canon.provider_registry),
        'provider_refs', COALESCE((SELECT jsonb_agg(provider_key ORDER BY provider_key ASC) FROM (SELECT provider_key FROM lucidota_canon.provider_registry ORDER BY provider_key ASC LIMIT 15) p), '[]'::jsonb)
    ) AS provider_registry
),
workflow_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(w) ORDER BY w.workflow_id), '[]'::jsonb) AS workflow_registry
    FROM lucidota_canon.workflow_registry w
),
command_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.command_id), '[]'::jsonb) AS command_registry
    FROM lucidota_canon.command_registry c
),
chrono_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.refreshed_at DESC), '[]'::jsonb) AS chrono_current
    FROM (
        SELECT *
        FROM lucidota_canon.chrono_current
        LIMIT 1
    ) c
),
capability_rows AS (
    SELECT jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.capability_registry),
        'latest_updated_at', (SELECT max(updated_at) FROM lucidota_canon.capability_registry),
        'capability_refs', COALESCE((SELECT jsonb_agg(capability_key ORDER BY updated_at DESC) FROM (SELECT capability_key, updated_at FROM lucidota_canon.capability_registry ORDER BY updated_at DESC LIMIT 15) c), '[]'::jsonb)
    ) AS capability_registry
),
indy_rows AS (
    SELECT jsonb_build_object(
        'self_model_count', (SELECT count(*) FROM lucidota_canon.indy_reads_self_model),
        'llmwiki_entry_count', (SELECT count(*) FROM lucidota_canon.indy_reads_llmwiki_entry),
        'hunch_log_count', (SELECT count(*) FROM lucidota_canon.indy_reads_hunch_log),
        'learning_queue_count', (SELECT count(*) FROM lucidota_canon.indy_reads_learning_queue),
        'system_map_count', (SELECT count(*) FROM lucidota_canon.indy_reads_system_map),
        'mistake_ledger_count', (SELECT count(*) FROM lucidota_canon.indy_reads_mistake_ledger),
        'research_source_count', (SELECT count(*) FROM lucidota_canon.indy_reads_research_source),
        'metacognition_current_count', (SELECT count(*) FROM lucidota_canon.indy_reads_metacognition_current),
        'route_refs', jsonb_build_array(
            'indy_reads_self_model',
            'indy_reads_llmwiki_entry',
            'indy_reads_hunch_log',
            'indy_reads_learning_queue',
            'indy_reads_system_map',
            'indy_reads_mistake_ledger',
            'indy_reads_research_source',
            'indy_reads_metacognition_current'
        )
    ) AS indy_reads_runtime
),
indy_queue_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(iq) ORDER BY iq.received_at DESC), '[]'::jsonb) AS indy_queue
    FROM (
        SELECT *
        FROM lucidota_canon.indy_queue
        ORDER BY received_at DESC
        LIMIT 5
    ) iq
),
indy_response_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(ir) ORDER BY ir.created_at DESC), '[]'::jsonb) AS indy_responses
    FROM (
        SELECT *
        FROM lucidota_canon.indy_responses
        ORDER BY created_at DESC
        LIMIT 5
    ) ir
),
todo_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(t) ORDER BY t.created_at DESC), '[]'::jsonb) AS todo_current
    FROM (
        SELECT *
        FROM lucidota_canon.todo_current
        WHERE status IN ('ready', 'queued', 'running')
        ORDER BY created_at DESC
        LIMIT 5
    ) t
),
prompt_status_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.refreshed_at DESC), '[]'::jsonb) AS prompt_catalog_status
    FROM lucidota_canon.prompt_catalog_status p
),
flow_specs_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(fs) ORDER BY fs.updated_at DESC), '[]'::jsonb) AS flow_specs
    FROM (
        SELECT *
        FROM lucidota_canon.flow_specs
        LIMIT 5
    ) fs
),
payload_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.latest_archived_at DESC NULLS LAST), '[]'::jsonb) AS payload_archive_status
    FROM (
        SELECT *
        FROM lucidota_canon.payload_archive_status
        LIMIT 6
    ) p
),
bytewax_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(bw) ORDER BY bw.updated_at DESC), '[]'::jsonb) AS bytewax_compact_windows
    FROM (
        SELECT *
        FROM lucidota_canon.bytewax_compact_windows
        ORDER BY updated_at DESC
        LIMIT 5
    ) bw
),
cloud_packet_rows AS (
    SELECT COALESCE((
        SELECT prompt_api.cloud_packet(
            bw.work_order_uuid,
            8000,
            12,
            '',
            '',
            false
        )
        FROM lucidota_canon.bytewax_compact_windows bw
        ORDER BY bw.updated_at DESC
        LIMIT 1
    ), '{}'::jsonb) AS cloud_packet
),
sub_orchestrators_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.route_id), '[]'::jsonb) AS sub_orchestrators
    FROM (
        SELECT route_id, method, path_pattern, description, target, status
        FROM lucidota_canon.api_route_catalog
        WHERE route_id IN ('manual_current', 'canon_current', 'active_goal', 'active_operation_mode', 'daemon_status', 'model_registry_current', 'provider_current', 'workflow_current', 'model_routing_current', 'model_routing_blockers', 'sheet_current', 'skill_policy_current', 'todo_current', 'schema_owner_manifest', 'surface_registry', 'renderer_registry', 'controller_grant', 'agent_thread_runtime', 'api_route_catalog', 'root_orchestrator_current', '/', 'canon_versions', 'capability_current', 'workload_audit_current', 'workload_audit_telemetry_current', 'indy_reads_self_model', 'indy_reads_llmwiki_entry', 'indy_reads_hunch_log', 'indy_reads_learning_queue', 'indy_reads_system_map', 'indy_reads_mistake_ledger', 'indy_reads_research_source', 'indy_reads_metacognition_current')
        ORDER BY route_id
    ) r
),
sub_orchestrator_threads_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(t) ORDER BY t.thread_key), '[]'::jsonb) AS sub_orchestrator_threads
    FROM (
        SELECT
            thread_key,
            parent_thread_key,
            controller_grant_key,
            thread_owner,
            runtime_kind,
            status,
            env_identity,
            budget_scope,
            receipt_gate,
            detail,
            created_at,
            updated_at
        FROM lucidota_canon.agent_thread_runtime
        WHERE status = 'active'
        ORDER BY
            CASE WHEN thread_key = 'root_operator_thread' THEN 0 ELSE 1 END,
            thread_key
    ) t
),
sub_orchestrator_grants_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(g) ORDER BY g.grant_key), '[]'::jsonb) AS sub_orchestrator_grants
    FROM (
        SELECT DISTINCT
            g.grant_key,
            g.grant_uuid,
            g.controller_name,
            g.controller_kind,
            g.effective_status,
            g.status,
            g.allowed_envs,
            g.allowed_routes,
            g.allowed_commands,
            g.allowed_models,
            g.max_parallel_threads,
            g.max_spend,
            g.receipt_uuid,
            g.detail,
            g.created_at,
            g.updated_at
        FROM lucidota_canon.controller_grant g
        JOIN lucidota_canon.agent_thread_runtime t
          ON t.controller_grant_key = g.grant_key
        WHERE t.status = 'active'
        ORDER BY g.grant_key
    ) g
),
route_rows AS (
    SELECT
        COALESCE(jsonb_agg(jsonb_build_object(
            'route_id', route_id,
            'method', method,
            'path_pattern', path_pattern,
            'description', description,
            'target', target,
            'status', status
        ) ORDER BY route_id), '[]'::jsonb) AS route_list,
        count(*) AS route_count
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        'manual_current',
        'root_orchestrator_current',
        '/',
        'canon_current',
        'canon_versions',
        'active_goal',
        'daemon_status',
        'capability_current',
        'ontology_work_batch',
        'ontology_work_item',
        'model_registry_current',
        'provider_current',
        'workflow_current',
        'fn_bible_node_sort_key',
        'get_subtree',
        'chrono_current',
        'model_routing_current',
        'model_routing_blockers',
        'workload_audit_current',
        'workload_audit_ledger',
        'provider_call_receipt',
        'model_invocation_receipt',
        'agent_work_receipt',
        'unproven_work_debt',
        'sheet_current',
        'skill_policy_current',
        'todo_current',
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'controller_grant',
        'agent_thread_runtime',
        'api_route_catalog',
        'prompt_recent',
        'prompts_filed',
        'prompt_work_order_links',
        'prompt_unlinked',
        'prompt_catalog_status',
        'cli_process_receipts',
        'flow_receipts',
        'api_test_execution_receipts',
        'flow_specs',
        'payload_archive_status',
        'bytewax_compact_windows',
        'indy_queue',
        'indy_responses',
        'cloud_packet',
        'root_law_docs',
        'api_root_law_docs',
        'api_bible_edges',
        'api_bible_manuals',
        'api_bible_nodes',
        'api_bible_route_catalog',
        'api_bible_subtree',
        'api_workflow_registry',
        'model_registry',
        'provider_registry',
        'workflow_registry',
        'capability_registry',
        'edges',
        'manuals',
        'nodes',
        'route_catalog',
        'subtree',
        'rpc/fn_bible_node_material',
        'rpc/fn_bible_node_sort_key',
        'rpc/get_subtree',
        'fn_bible_node_material',
        'book_source',
        'book_scan',
        'book_read_queue',
        'book_note',
        'book_receipt',
        'lora_candidate',
        'lora_adapter',
        'training_job'
    )
),
cli_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.received_at DESC), '[]'::jsonb) AS cli_process_receipts
    FROM (
        SELECT *
        FROM lucidota_canon.cli_process_receipts
        LIMIT 3
    ) c
),
flow_receipt_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(fr) ORDER BY fr.created_at DESC), '[]'::jsonb) AS flow_receipts
    FROM (
        SELECT *
        FROM lucidota_canon.flow_receipts
        LIMIT 3
    ) fr
),
test_receipt_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(tr) ORDER BY tr.completed_at DESC), '[]'::jsonb) AS api_test_execution_receipts
    FROM (
        SELECT *
        FROM lucidota_canon.api_test_execution_receipts
        LIMIT 3
    ) tr
),
book_source_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(bs) ORDER BY bs.updated_at DESC), '[]'::jsonb) AS book_source
    FROM (
        SELECT *
        FROM lucidota_canon.book_source
        ORDER BY updated_at DESC
        LIMIT 3
    ) bs
),
book_scan_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(bs) ORDER BY bs.updated_at DESC), '[]'::jsonb) AS book_scan
    FROM (
        SELECT *
        FROM lucidota_canon.book_scan
        ORDER BY updated_at DESC
        LIMIT 3
    ) bs
),
book_read_queue_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(brq) ORDER BY brq.updated_at DESC), '[]'::jsonb) AS book_read_queue
    FROM (
        SELECT *
        FROM lucidota_canon.book_read_queue
        ORDER BY updated_at DESC
        LIMIT 3
    ) brq
),
book_note_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(bn) ORDER BY bn.updated_at DESC), '[]'::jsonb) AS book_note
    FROM (
        SELECT *
        FROM lucidota_canon.book_note
        ORDER BY updated_at DESC
        LIMIT 3
    ) bn
),
book_receipt_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(br) ORDER BY br.updated_at DESC), '[]'::jsonb) AS book_receipt
    FROM (
        SELECT *
        FROM lucidota_canon.book_receipt
        ORDER BY updated_at DESC
        LIMIT 3
    ) br
),
lora_candidate_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(lc) ORDER BY lc.updated_at DESC), '[]'::jsonb) AS lora_candidate
    FROM (
        SELECT *
        FROM lucidota_canon.lora_candidate
        ORDER BY updated_at DESC
        LIMIT 3
    ) lc
),
lora_adapter_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(la) ORDER BY la.updated_at DESC), '[]'::jsonb) AS lora_adapter
    FROM (
        SELECT *
        FROM lucidota_canon.lora_adapter
        ORDER BY updated_at DESC
        LIMIT 3
    ) la
),
training_job_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(tj) ORDER BY tj.updated_at DESC), '[]'::jsonb) AS training_job
    FROM (
        SELECT *
        FROM lucidota_canon.training_job
        ORDER BY updated_at DESC
        LIMIT 3
    ) tj
),
ontology_batch_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(ob) ORDER BY ob.updated_at DESC), '[]'::jsonb) AS ontology_work_batch
    FROM (
        SELECT *
        FROM lucidota_canon.ontology_work_batch
        ORDER BY updated_at DESC
        LIMIT 3
    ) ob
),
ontology_focus_rows AS (
    SELECT
        to_jsonb(ob.selected_lanes) AS selected_lanes,
        to_jsonb(ob.missing_executor_roles) AS missing_executor_roles
    FROM lucidota_canon.ontology_work_batch ob
    ORDER BY ob.updated_at DESC
    LIMIT 1
),
ontology_item_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(oi) ORDER BY oi.updated_at DESC), '[]'::jsonb) AS ontology_work_item
    FROM (
        SELECT *
        FROM lucidota_canon.ontology_work_item
        ORDER BY updated_at DESC
        LIMIT 5
    ) oi
),
live_surface_rows AS (
    SELECT 1 AS ord, 'current_goal'::text AS key, goal_row.current_goal AS value FROM goal_row
    UNION ALL SELECT 2 AS ord, 'daemon_status'::text AS key, daemon_rows.daemon_status AS value FROM daemon_rows
    UNION ALL SELECT 2.1 AS ord, 'active_operation_mode'::text AS key, COALESCE((SELECT to_jsonb(aom) FROM lucidota_control.active_operation_mode aom LIMIT 1), '{}'::jsonb) AS value
    UNION ALL SELECT 3 AS ord, 'model_registry'::text AS key, model_rows.model_registry AS value FROM model_rows
    UNION ALL SELECT 4 AS ord, 'provider_registry'::text AS key, provider_rows.provider_registry AS value FROM provider_rows
    UNION ALL SELECT 5 AS ord, 'workflow_registry'::text AS key, jsonb_build_object(
        'workflow_count', (SELECT count(*) FROM lucidota_canon.workflow_registry),
        'active_count', (SELECT count(*) FILTER (WHERE status = 'active') FROM lucidota_canon.workflow_registry),
        'latest_updated_at', (SELECT max(updated_at) FROM lucidota_canon.workflow_registry),
        'workflow_refs', COALESCE((SELECT jsonb_agg(workflow_name ORDER BY workflow_name) FROM (SELECT workflow_name FROM lucidota_canon.workflow_registry ORDER BY workflow_name LIMIT 25) wr), '[]'::jsonb)
    ) AS value FROM workflow_rows
    UNION ALL SELECT 6 AS ord, 'command_registry'::text AS key, command_rows.command_registry AS value FROM command_rows
    UNION ALL SELECT 6.1 AS ord, 'todo_current'::text AS key, todo_rows.todo_current AS value FROM todo_rows
    UNION ALL SELECT 7 AS ord, 'sub_orchestrators'::text AS key, sub_orchestrators_rows.sub_orchestrators AS value FROM sub_orchestrators_rows
    UNION ALL SELECT 7.1 AS ord, 'sub_orchestrator_threads'::text AS key, sub_orchestrator_threads_rows.sub_orchestrator_threads AS value FROM sub_orchestrator_threads_rows
    UNION ALL SELECT 7.2 AS ord, 'sub_orchestrator_grants'::text AS key, sub_orchestrator_grants_rows.sub_orchestrator_grants AS value FROM sub_orchestrator_grants_rows
    UNION ALL SELECT 7.3 AS ord, 'orchestration'::text AS key, jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack()
    ) AS value
    UNION ALL SELECT 8 AS ord, 'api_route_catalog'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(arc) ORDER BY arc.route_id ASC) FROM (SELECT * FROM lucidota_canon.api_route_catalog ORDER BY route_id ASC LIMIT 20) arc), '[]'::jsonb) AS value
    UNION ALL SELECT 9 AS ord, 'canon_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.canon_current),
        'latest_updated_at', (SELECT max(updated_at) FROM lucidota_canon.canon_current),
        'node_refs', COALESCE((SELECT jsonb_agg(cn.node_id ORDER BY cn.updated_at DESC) FROM (SELECT node_id, updated_at FROM lucidota_canon.canon_current ORDER BY updated_at DESC LIMIT 25) cn), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 10 AS ord, 'canon_versions'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.canon_versions),
        'latest_promoted_at', (SELECT max(promoted_at) FROM lucidota_canon.canon_versions),
        'version_refs', COALESCE((SELECT jsonb_agg(cv.version_id ORDER BY cv.promoted_at DESC) FROM (SELECT version_id, promoted_at FROM lucidota_canon.canon_versions ORDER BY promoted_at DESC LIMIT 25) cv), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 11 AS ord, 'skill_policy_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(sp) ORDER BY sp.updated_at DESC) FROM lucidota_canon.skill_policy_current sp), '[]'::jsonb) AS value
    UNION ALL SELECT 12 AS ord, 'chrono_current'::text AS key, chrono_rows.chrono_current AS value FROM chrono_rows
    UNION ALL SELECT 12.1 AS ord, 'fn_bible_node_sort_key'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(sk) ORDER BY sk) FROM (SELECT lucidota_canon.fn_bible_node_sort_key('4.9511.0') AS sk) s), '[]'::jsonb) AS value
    UNION ALL SELECT 12.2 AS ord, 'get_subtree'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(gs) ORDER BY gs.node_sort_key ASC) FROM lucidota_canon.get_subtree('4.9511.0') gs), '[]'::jsonb) AS value
    UNION ALL SELECT 13 AS ord, 'api_bible_edges'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(be) ORDER BY be.edge_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_edges ORDER BY edge_id ASC LIMIT 5) be), '[]'::jsonb) AS value
    UNION ALL SELECT 14 AS ord, 'api_bible_manuals'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(bm) ORDER BY bm.manual_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_manuals ORDER BY manual_id ASC LIMIT 5) bm), '[]'::jsonb) AS value
    UNION ALL SELECT 15 AS ord, 'api_bible_nodes'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(bn) ORDER BY bn.node_sort_key ASC) FROM (SELECT * FROM lucidota_canon.api_bible_nodes WHERE manual_id = 'RUNTIME_GOVERNOR' ORDER BY node_sort_key ASC LIMIT 5) bn), '[]'::jsonb) AS value
    UNION ALL SELECT 16 AS ord, 'api_bible_route_catalog'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(br) ORDER BY br.route_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_route_catalog ORDER BY route_id ASC LIMIT 5) br), '[]'::jsonb) AS value
    UNION ALL SELECT 17 AS ord, 'api_bible_subtree'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(bs) ORDER BY bs.node_sort_key ASC) FROM (SELECT * FROM lucidota_canon.api_bible_subtree WHERE root_id = '1.0.0' ORDER BY node_sort_key ASC LIMIT 5) bs), '[]'::jsonb) AS value
    UNION ALL SELECT 18 AS ord, 'api_root_law_docs'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(rl) ORDER BY rl.refreshed_at DESC) FROM lucidota_canon.api_root_law_docs rl), '[]'::jsonb) AS value
    UNION ALL SELECT 19 AS ord, 'root_law_docs'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(rl) ORDER BY rl.refreshed_at DESC) FROM lucidota_canon.root_law_docs rl), '[]'::jsonb) AS value
    UNION ALL SELECT 20 AS ord, 'api_workflow_registry'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.workflow_registry),
        'latest_updated_at', (SELECT max(updated_at) FROM lucidota_canon.workflow_registry),
        'workflow_refs', COALESCE((SELECT jsonb_agg(workflow_name ORDER BY workflow_name) FROM (SELECT workflow_name FROM lucidota_canon.workflow_registry ORDER BY workflow_name LIMIT 25) wr), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 21 AS ord, 'api_capability_registry'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(cr) ORDER BY cr.capability_key ASC) FROM (SELECT * FROM lucidota_canon.capability_registry ORDER BY capability_key ASC LIMIT 5) cr), '[]'::jsonb) AS value
    UNION ALL SELECT 22 AS ord, 'api_provider_registry'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(pr) ORDER BY pr.provider_key ASC) FROM (SELECT * FROM lucidota_canon.provider_registry ORDER BY provider_key ASC LIMIT 5) pr), '[]'::jsonb) AS value
    UNION ALL SELECT 23 AS ord, 'api_model_registry_raw'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(m) ORDER BY m.model_id ASC) FROM (SELECT * FROM lucidota_canon.model_registry ORDER BY model_id ASC LIMIT 5) m), '[]'::jsonb) AS value
    UNION ALL SELECT 24 AS ord, 'api_bytewax_windows'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(bw) ORDER BY bw.updated_at DESC) FROM (SELECT * FROM lucidota_canon.bytewax_compact_windows ORDER BY updated_at DESC LIMIT 5) bw), '[]'::jsonb) AS value
    UNION ALL SELECT 25 AS ord, 'api_cloud_packet'::text AS key, cloud_packet_rows.cloud_packet AS value FROM cloud_packet_rows
    UNION ALL SELECT 26 AS ord, 'api_cli_process_receipts'::text AS key, cli_rows.cli_process_receipts AS value FROM cli_rows
    UNION ALL SELECT 27 AS ord, 'api_flow_receipts'::text AS key, flow_receipt_rows.flow_receipts AS value FROM flow_receipt_rows
    UNION ALL SELECT 28 AS ord, 'cli_process_receipts'::text AS key, cli_rows.cli_process_receipts AS value FROM cli_rows
    UNION ALL SELECT 29 AS ord, 'flow_receipts'::text AS key, flow_receipt_rows.flow_receipts AS value FROM flow_receipt_rows
    UNION ALL SELECT 30 AS ord, 'api_test_execution_receipts'::text AS key, test_receipt_rows.api_test_execution_receipts AS value FROM test_receipt_rows
    UNION ALL SELECT 31 AS ord, 'api_model_registry_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.model_registry_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.model_registry_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(model_packet_id ORDER BY refreshed_at DESC) FROM (SELECT model_packet_id, refreshed_at FROM lucidota_canon.model_registry_current ORDER BY refreshed_at DESC LIMIT 10) mc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 32 AS ord, 'api_provider_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.provider_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.provider_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(provider_packet_id ORDER BY refreshed_at DESC) FROM (SELECT provider_packet_id, refreshed_at FROM lucidota_canon.provider_current ORDER BY refreshed_at DESC LIMIT 10) pc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 33 AS ord, 'api_workflow_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.workflow_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.workflow_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(workflow_packet_id ORDER BY refreshed_at DESC) FROM (SELECT workflow_packet_id, refreshed_at FROM lucidota_canon.workflow_current ORDER BY refreshed_at DESC LIMIT 10) wc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 34 AS ord, 'api_capability_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.capability_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.capability_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(capability_packet_id ORDER BY refreshed_at DESC) FROM (SELECT capability_packet_id, refreshed_at FROM lucidota_canon.capability_current ORDER BY refreshed_at DESC LIMIT 10) cc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 35 AS ord, 'api_sheet_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.sheet_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.sheet_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(sheet_packet_id ORDER BY refreshed_at DESC) FROM (SELECT sheet_packet_id, refreshed_at FROM lucidota_canon.sheet_current ORDER BY refreshed_at DESC LIMIT 10) sc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 36 AS ord, 'api_model_routing_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.model_routing_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.model_routing_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(routing_packet_id ORDER BY refreshed_at DESC) FROM (SELECT routing_packet_id, refreshed_at FROM lucidota_canon.model_routing_current ORDER BY refreshed_at DESC LIMIT 10) mr), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 37 AS ord, 'api_model_routing_blockers'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.model_routing_blockers),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.model_routing_blockers),
        'packet_refs', COALESCE((SELECT jsonb_agg(routing_packet_id ORDER BY refreshed_at DESC) FROM (SELECT routing_packet_id, refreshed_at FROM lucidota_canon.model_routing_blockers ORDER BY refreshed_at DESC LIMIT 10) mrb), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 38 AS ord, 'model_registry_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.model_registry_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.model_registry_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(model_packet_id ORDER BY refreshed_at DESC) FROM (SELECT model_packet_id, refreshed_at FROM lucidota_canon.model_registry_current ORDER BY refreshed_at DESC LIMIT 10) mc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 39 AS ord, 'provider_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.provider_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.provider_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(provider_packet_id ORDER BY refreshed_at DESC) FROM (SELECT provider_packet_id, refreshed_at FROM lucidota_canon.provider_current ORDER BY refreshed_at DESC LIMIT 10) pc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 40 AS ord, 'workflow_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.workflow_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.workflow_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(workflow_packet_id ORDER BY refreshed_at DESC) FROM (SELECT workflow_packet_id, refreshed_at FROM lucidota_canon.workflow_current ORDER BY refreshed_at DESC LIMIT 10) wc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 41 AS ord, 'capability_current'::text AS key, jsonb_build_object(
        'count', (SELECT count(*) FROM lucidota_canon.capability_current),
        'latest_refreshed_at', (SELECT max(refreshed_at) FROM lucidota_canon.capability_current),
        'packet_refs', COALESCE((SELECT jsonb_agg(capability_packet_id ORDER BY refreshed_at DESC) FROM (SELECT capability_packet_id, refreshed_at FROM lucidota_canon.capability_current ORDER BY refreshed_at DESC LIMIT 10) cc), '[]'::jsonb)
    ) AS value
    UNION ALL SELECT 42 AS ord, 'sheet_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(sc) ORDER BY sc.refreshed_at DESC) FROM lucidota_canon.sheet_current sc), '[]'::jsonb) AS value
    UNION ALL SELECT 43 AS ord, 'capability_registry'::text AS key, capability_rows.capability_registry AS value FROM capability_rows
    UNION ALL SELECT 44 AS ord, 'chrono_current'::text AS key, chrono_rows.chrono_current AS value FROM chrono_rows
    UNION ALL SELECT 45 AS ord, 'indy_queue'::text AS key, indy_queue_rows.indy_queue AS value FROM indy_queue_rows
    UNION ALL SELECT 46 AS ord, 'indy_responses'::text AS key, indy_response_rows.indy_responses AS value FROM indy_response_rows
    UNION ALL SELECT 47 AS ord, 'book_source'::text AS key, book_source_rows.book_source AS value FROM book_source_rows
    UNION ALL SELECT 48 AS ord, 'book_scan'::text AS key, book_scan_rows.book_scan AS value FROM book_scan_rows
    UNION ALL SELECT 49 AS ord, 'book_read_queue'::text AS key, book_read_queue_rows.book_read_queue AS value FROM book_read_queue_rows
    UNION ALL SELECT 50 AS ord, 'book_note'::text AS key, book_note_rows.book_note AS value FROM book_note_rows
    UNION ALL SELECT 51 AS ord, 'book_receipt'::text AS key, book_receipt_rows.book_receipt AS value FROM book_receipt_rows
    UNION ALL SELECT 52 AS ord, 'lora_candidate'::text AS key, lora_candidate_rows.lora_candidate AS value FROM lora_candidate_rows
    UNION ALL SELECT 53 AS ord, 'lora_adapter'::text AS key, lora_adapter_rows.lora_adapter AS value FROM lora_adapter_rows
    UNION ALL SELECT 54 AS ord, 'training_job'::text AS key, training_job_rows.training_job AS value FROM training_job_rows
    UNION ALL SELECT 55 AS ord, 'ontology_work_batch'::text AS key, ontology_batch_rows.ontology_work_batch AS value FROM ontology_batch_rows
    UNION ALL SELECT 56 AS ord, 'selected_lanes'::text AS key, COALESCE((ontology_batch_rows.ontology_work_batch->0->'selected_lanes'), '[]'::jsonb) AS value FROM ontology_batch_rows
    UNION ALL SELECT 57 AS ord, 'missing_executor_roles'::text AS key, COALESCE((ontology_batch_rows.ontology_work_batch->0->'missing_executor_roles'), '[]'::jsonb) AS value FROM ontology_batch_rows
    UNION ALL SELECT 58 AS ord, 'ontology_work_item'::text AS key, ontology_item_rows.ontology_work_item AS value FROM ontology_item_rows
    UNION ALL SELECT 59 AS ord, 'prompt_catalog_status'::text AS key, prompt_status_rows.prompt_catalog_status AS value FROM prompt_status_rows
    UNION ALL SELECT 60 AS ord, 'prompt_recent'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC) FROM (SELECT * FROM lucidota_canon.prompt_recent LIMIT 5) p), '[]'::jsonb) AS value
    UNION ALL SELECT 61 AS ord, 'prompts_filed'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC) FROM (SELECT * FROM lucidota_canon.prompts_filed LIMIT 5) p), '[]'::jsonb) AS value
    UNION ALL SELECT 62 AS ord, 'prompt_work_order_links'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC) FROM (SELECT * FROM lucidota_canon.prompt_work_order_links LIMIT 5) p), '[]'::jsonb) AS value
    UNION ALL SELECT 63 AS ord, 'prompt_unlinked'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC) FROM (SELECT * FROM lucidota_canon.prompt_unlinked LIMIT 5) p), '[]'::jsonb) AS value
    UNION ALL SELECT 64 AS ord, 'flow_specs'::text AS key, flow_specs_rows.flow_specs AS value FROM flow_specs_rows
    UNION ALL SELECT 65 AS ord, 'payload_archive_status'::text AS key, payload_rows.payload_archive_status AS value FROM payload_rows
    UNION ALL SELECT 66 AS ord, 'bytewax_compact_windows'::text AS key, bytewax_rows.bytewax_compact_windows AS value FROM bytewax_rows
    UNION ALL SELECT 67 AS ord, 'cloud_packet'::text AS key, cloud_packet_rows.cloud_packet AS value FROM cloud_packet_rows
    UNION ALL SELECT 68 AS ord, 'model_routing_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mr) ORDER BY mr.refreshed_at DESC) FROM lucidota_canon.model_routing_current mr), '[]'::jsonb) AS value
    UNION ALL SELECT 69 AS ord, 'model_routing_blockers'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb) AS value
    UNION ALL SELECT 69.1 AS ord, 'workload_audit_current'::text AS key, COALESCE((SELECT to_jsonb(wac) FROM lucidota_canon.workload_audit_current wac LIMIT 1), '{}'::jsonb) AS value
    UNION ALL SELECT 69.2 AS ord, 'workload_audit_telemetry_current'::text AS key, COALESCE((SELECT to_jsonb(wat) FROM lucidota_canon.workload_audit_telemetry_current wat LIMIT 1), '{}'::jsonb) AS value
    UNION ALL SELECT 69.3 AS ord, 'indy_reads_runtime'::text AS key, COALESCE((SELECT to_jsonb(ir) FROM indy_rows ir), '{}'::jsonb) AS value FROM indy_rows
    UNION ALL SELECT 69.31 AS ord, 'indy_reads_self_model'::text AS key, jsonb_build_object('route_ref', 'indy_reads_self_model', 'count', COALESCE(((SELECT indy_reads_runtime->>'self_model_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.32 AS ord, 'indy_reads_llmwiki_entry'::text AS key, jsonb_build_object('route_ref', 'indy_reads_llmwiki_entry', 'count', COALESCE(((SELECT indy_reads_runtime->>'llmwiki_entry_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.33 AS ord, 'indy_reads_hunch_log'::text AS key, jsonb_build_object('route_ref', 'indy_reads_hunch_log', 'count', COALESCE(((SELECT indy_reads_runtime->>'hunch_log_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.34 AS ord, 'indy_reads_learning_queue'::text AS key, jsonb_build_object('route_ref', 'indy_reads_learning_queue', 'count', COALESCE(((SELECT indy_reads_runtime->>'learning_queue_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.35 AS ord, 'indy_reads_system_map'::text AS key, jsonb_build_object('route_ref', 'indy_reads_system_map', 'count', COALESCE(((SELECT indy_reads_runtime->>'system_map_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.36 AS ord, 'indy_reads_mistake_ledger'::text AS key, jsonb_build_object('route_ref', 'indy_reads_mistake_ledger', 'count', COALESCE(((SELECT indy_reads_runtime->>'mistake_ledger_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.37 AS ord, 'indy_reads_research_source'::text AS key, jsonb_build_object('route_ref', 'indy_reads_research_source', 'count', COALESCE(((SELECT indy_reads_runtime->>'research_source_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 69.38 AS ord, 'indy_reads_metacognition_current'::text AS key, jsonb_build_object('route_ref', 'indy_reads_metacognition_current', 'count', COALESCE(((SELECT indy_reads_runtime->>'metacognition_current_count' FROM indy_rows LIMIT 1))::int, 0)) AS value FROM indy_rows
    UNION ALL SELECT 70 AS ord, 'blockers'::text AS key, jsonb_build_object(
                                    'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb)
                                ) AS value
    UNION ALL SELECT 71 AS ord, 'receipts'::text AS key, jsonb_build_object(
                                    'cli_process_receipts', cli_rows.cli_process_receipts,
                                    'flow_receipts', flow_receipt_rows.flow_receipts,
                                    'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts
                                ) AS value FROM cli_rows CROSS JOIN flow_receipt_rows CROSS JOIN test_receipt_rows
)
SELECT
    'ROOT_ORCHESTRATOR_CURRENT'::text AS orchestrator_id,
    'Root Orchestrator'::text AS title,
    now() AS max_updated_at,
    route_rows.route_count AS route_count,
    route_rows.route_list AS route_list,
    COALESCE((SELECT jsonb_object_agg(key, value ORDER BY ord) FROM live_surface_rows), '{}'::jsonb) AS live_surface,
    jsonb_build_object(
        'read_surface', 'PostgREST safe views and RPCs only',
        'write_surface', 'DB work orders and receipts only',
        'manual_source', 'live route catalog + daemon status + current goal + active operation mode + workload telemetry + registries + receipts + queue/window/cloud packet',
        'root_orchestrator', 'DB-visible sub-orchestrator status packet; no new daemon or hidden control plane'
    ) AS auth_expectations,
    jsonb_build_object(
        'root_loop', 'operator command -> root_orchestrator_current -> ontology_work_batch / todo_current -> sub-orchestrator packets -> receipts -> manual update',
        'manual_loop', 'manual_current -> route list + registries + daemon status + skill policy + root orchestrator',
        'model_loop', 'model_registry / provider_registry / workflow_registry -> live role coverage -> missing roles/blockers',
        'queue_loop', 'indy_queue -> indy_responses -> bytewax_compact_windows -> cloud_packet',
        'prompt_loop', 'prompt_recent -> prompts_filed -> prompt_work_order_links -> prompt_unlinked -> prompt_catalog_status'
    ) AS work_order_flow,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'active_goal',
        'daemon_status',
        'capability_current',
        'capability_registry',
        'command_registry',
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'controller_grant',
        'agent_thread_runtime',
        'model_registry',
        'model_registry_current',
        'model_routing_current',
        'model_routing_blockers',
        'active_operation_mode',
        'workload_audit_current',
        'workload_audit_telemetry_current',
        'workload_audit_ledger',
        'provider_current',
        'provider_registry',
        'workflow_current',
        'workflow_registry',
        'sheet_current',
        'skill_policy_current',
        'todo_current',
        'prompt_catalog_status',
        'prompt_recent',
        'prompts_filed',
        'prompt_work_order_links',
        'prompt_unlinked',
        'prompts_filed',
        'payload_archive_status',
        'bytewax_compact_windows',
        'indy_queue',
        'indy_responses',
        'cli_process_receipts',
        'flow_receipts',
        'flow_specs',
        'api_route_catalog',
        'api_root_law_docs',
        'api_test_execution_receipts',
        'api_bible_route_catalog',
        'api_bible_subtree',
        'api_bible_edges',
        'api_bible_manuals',
        'api_bible_nodes',
        'api_bible_edges',
        'api_bible_manuals',
        'api_bible_nodes',
        'canon_current',
        'canon_versions',
        'cloud_packet',
        'decompose_prompt_to_work_orders',
        'file_prompt',
        'link_prompt_work_order',
        'workload_audit_current',
        'workload_audit_ledger',
        'indy_reads_self_model',
        'indy_reads_llmwiki_entry',
        'indy_reads_hunch_log',
        'indy_reads_learning_queue',
        'indy_reads_system_map',
        'indy_reads_mistake_ledger',
        'indy_reads_research_source',
        'indy_reads_metacognition_current'
    ) AS next_commands,
    jsonb_build_array(
        'manual_current',
        'root_orchestrator_current',
        'canon_current',
        'active_goal',
        'daemon_status',
        'capability_current',
        'capability_registry',
        'model_registry_current',
        'model_registry',
        'provider_current',
        'provider_registry',
        'active_operation_mode',
        'workload_audit_telemetry_current',
        'workflow_current',
        'workflow_registry',
        'sheet_current',
        'prompt_catalog_status',
        'prompt_recent',
        'flow_specs',
        'flow_receipts',
        'api_test_execution_receipts',
        'skill_policy_current',
        'todo_current',
        'prompts_filed',
        'prompt_work_order_links',
        'prompt_unlinked',
        'payload_archive_status',
        'bytewax_compact_windows',
        'indy_queue',
        'indy_responses',
        'model_routing_current',
        'model_routing_blockers',
        'api_route_catalog',
        'api_bible_route_catalog',
        'api_bible_subtree',
        'api_bible_edges',
        'api_bible_manuals',
        'api_bible_nodes',
        'canon_versions',
        'api_root_law_docs',
        'cloud_packet',
        'decompose_prompt_to_work_orders',
        'file_prompt',
        'link_prompt_work_order',
        'cli_process_receipts',
        'schema_owner_manifest',
        'surface_registry',
        'renderer_registry',
        'command_registry',
        'controller_grant',
        'agent_thread_runtime',
        'workload_audit_current',
        'workload_audit_ledger',
        'provider_call_receipt',
        'model_invocation_receipt',
        'agent_work_receipt',
        'unproven_work_debt',
        'indy_reads_self_model',
        'indy_reads_llmwiki_entry',
        'indy_reads_hunch_log',
        'indy_reads_learning_queue',
        'indy_reads_system_map',
        'indy_reads_mistake_ledger',
        'indy_reads_research_source',
        'indy_reads_metacognition_current',
        'sub_orchestrator_threads',
        'sub_orchestrator_grants'
    ) AS next_command_refs,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; Rust-worthy code becomes Rust only after contract, tests, and A/B receipt.'
    ) AS db_law,
    jsonb_build_object(
        'mode', 'sub_orchestrator',
        'sub_orchestrator_priority', lucidota_control.live_truth_priority_stack(),
        'strict_priority_stack', lucidota_control.live_truth_priority_stack(),
        'provider_secret_isolation', 'load through an explicit quarantine file or environment loader owned by the operator; no raw keys in chat, docs, SQL, or receipts',
        'selected_lanes', COALESCE((SELECT selected_lanes FROM ontology_focus_rows), '[]'::jsonb),
        'sub_orchestrator_threads', sub_orchestrator_threads_rows.sub_orchestrator_threads,
        'sub_orchestrator_grants', sub_orchestrator_grants_rows.sub_orchestrator_grants
    ) AS orchestration,
    sub_orchestrators_rows.sub_orchestrators AS sub_orchestrators,
    sub_orchestrator_threads_rows.sub_orchestrator_threads AS sub_orchestrator_threads,
    sub_orchestrator_grants_rows.sub_orchestrator_grants AS sub_orchestrator_grants,
    jsonb_build_object(
        'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb)
    ) AS blockers,
    jsonb_build_object(
        'cli_process_receipts', cli_rows.cli_process_receipts,
        'flow_receipts', flow_receipt_rows.flow_receipts,
        'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts
    ) AS receipts
FROM goal_row
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN provider_rows
CROSS JOIN workflow_rows
CROSS JOIN command_rows
CROSS JOIN chrono_rows
CROSS JOIN capability_rows
CROSS JOIN todo_rows
CROSS JOIN route_rows
CROSS JOIN prompt_status_rows
CROSS JOIN flow_specs_rows
CROSS JOIN payload_rows
CROSS JOIN bytewax_rows
CROSS JOIN cloud_packet_rows
CROSS JOIN sub_orchestrators_rows
CROSS JOIN sub_orchestrator_threads_rows
CROSS JOIN sub_orchestrator_grants_rows
CROSS JOIN cli_rows
CROSS JOIN flow_receipt_rows
CROSS JOIN test_receipt_rows
CROSS JOIN book_source_rows
CROSS JOIN book_scan_rows
CROSS JOIN book_read_queue_rows
CROSS JOIN book_note_rows
CROSS JOIN book_receipt_rows
CROSS JOIN lora_candidate_rows
CROSS JOIN lora_adapter_rows
CROSS JOIN training_job_rows
CROSS JOIN ontology_batch_rows
CROSS JOIN ontology_item_rows;

GRANT SELECT ON lucidota_canon.root_orchestrator_current TO mfspx;
GRANT SELECT ON lucidota_canon.skill_policy_current, lucidota_canon.root_orchestrator_current TO lucidota_postgrest_anon;
COMMIT;
