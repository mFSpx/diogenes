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
    SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.model_id), '[]'::jsonb) AS model_registry
    FROM lucidota_canon.model_registry m
),
provider_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.provider_key), '[]'::jsonb) AS provider_registry
    FROM lucidota_canon.provider_registry p
),
workflow_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(w) ORDER BY w.workflow_id), '[]'::jsonb) AS workflow_registry
    FROM lucidota_canon.workflow_registry w
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
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.capability_key), '[]'::jsonb) AS capability_registry
    FROM (
        SELECT *
        FROM lucidota_canon.capability_registry
        ORDER BY capability_key
        LIMIT 5
    ) c
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
        WHERE route_id IN ('manual_current', 'canon_current', 'active_goal', 'daemon_status', 'model_registry_current', 'provider_current', 'workflow_current', 'model_routing_current', 'model_routing_blockers', 'sheet_current', 'skill_policy_current', 'todo_current', 'api_route_catalog', 'root_orchestrator_current', '/', 'canon_versions', 'capability_current')
        ORDER BY route_id
    ) r
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
        'sheet_current',
        'skill_policy_current',
        'todo_current',
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
        'rpc/cloud_packet',
        'rpc/decompose_prompt_to_work_orders',
        'rpc/file_prompt',
        'rpc/fn_bible_node_material',
        'rpc/fn_bible_node_sort_key',
        'rpc/get_subtree',
        'rpc/link_prompt_work_order',
        'decompose_prompt_to_work_orders',
        'file_prompt',
        'fn_bible_node_material',
        'link_prompt_work_order',
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
    UNION ALL SELECT 3 AS ord, 'model_registry'::text AS key, model_rows.model_registry AS value FROM model_rows
    UNION ALL SELECT 4 AS ord, 'provider_registry'::text AS key, provider_rows.provider_registry AS value FROM provider_rows
    UNION ALL SELECT 5 AS ord, 'workflow_registry'::text AS key, workflow_rows.workflow_registry AS value FROM workflow_rows
    UNION ALL SELECT 6 AS ord, 'todo_current'::text AS key, todo_rows.todo_current AS value FROM todo_rows
    UNION ALL SELECT 7 AS ord, 'sub_orchestrators'::text AS key, sub_orchestrators_rows.sub_orchestrators AS value FROM sub_orchestrators_rows
    UNION ALL SELECT 8 AS ord, 'api_route_catalog'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(arc) ORDER BY arc.route_id ASC) FROM (SELECT * FROM lucidota_canon.api_route_catalog ORDER BY route_id ASC LIMIT 20) arc), '[]'::jsonb) AS value
    UNION ALL SELECT 9 AS ord, 'canon_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(cn) ORDER BY cn.updated_at DESC) FROM lucidota_canon.canon_current cn), '[]'::jsonb) AS value
    UNION ALL SELECT 10 AS ord, 'canon_versions'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(cv) ORDER BY cv.promoted_at DESC) FROM lucidota_canon.canon_versions cv), '[]'::jsonb) AS value
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
    UNION ALL SELECT 20 AS ord, 'api_workflow_registry'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(wr) ORDER BY wr.workflow_id ASC) FROM (SELECT * FROM lucidota_canon.workflow_registry ORDER BY workflow_id ASC LIMIT 5) wr), '[]'::jsonb) AS value
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
    UNION ALL SELECT 31 AS ord, 'api_model_registry_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mc) ORDER BY mc.refreshed_at DESC) FROM lucidota_canon.model_registry_current mc), '[]'::jsonb) AS value
    UNION ALL SELECT 32 AS ord, 'api_provider_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(pc) ORDER BY pc.refreshed_at DESC) FROM lucidota_canon.provider_current pc), '[]'::jsonb) AS value
    UNION ALL SELECT 33 AS ord, 'api_workflow_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(wc) ORDER BY wc.refreshed_at DESC) FROM lucidota_canon.workflow_current wc), '[]'::jsonb) AS value
    UNION ALL SELECT 34 AS ord, 'api_capability_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(cc) ORDER BY cc.refreshed_at DESC) FROM lucidota_canon.capability_current cc), '[]'::jsonb) AS value
    UNION ALL SELECT 35 AS ord, 'api_sheet_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(sc) ORDER BY sc.refreshed_at DESC) FROM lucidota_canon.sheet_current sc), '[]'::jsonb) AS value
    UNION ALL SELECT 36 AS ord, 'api_model_routing_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mr) ORDER BY mr.refreshed_at DESC) FROM lucidota_canon.model_routing_current mr), '[]'::jsonb) AS value
    UNION ALL SELECT 37 AS ord, 'api_model_routing_blockers'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb) AS value
    UNION ALL SELECT 38 AS ord, 'model_registry_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(mc) ORDER BY mc.refreshed_at DESC) FROM lucidota_canon.model_registry_current mc), '[]'::jsonb) AS value
    UNION ALL SELECT 39 AS ord, 'provider_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(pc) ORDER BY pc.refreshed_at DESC) FROM lucidota_canon.provider_current pc), '[]'::jsonb) AS value
    UNION ALL SELECT 40 AS ord, 'workflow_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(wc) ORDER BY wc.refreshed_at DESC) FROM lucidota_canon.workflow_current wc), '[]'::jsonb) AS value
    UNION ALL SELECT 41 AS ord, 'capability_current'::text AS key, COALESCE((SELECT jsonb_agg(to_jsonb(cc) ORDER BY cc.refreshed_at DESC) FROM lucidota_canon.capability_current cc), '[]'::jsonb) AS value
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
        'manual_source', 'live route catalog + daemon status + current goal + registries + receipts + queue/window/cloud packet',
        'root_orchestrator', 'DB-visible sub-orchestrator status packet; no new daemon or hidden control plane'
    ) AS auth_expectations,
    jsonb_build_object(
        'root_loop', 'operator command -> root_orchestrator_current -> ontology_work_batch / todo_current -> sub-orchestrator packets -> receipts -> manual update',
        'manual_loop', 'manual_current -> route list + registries + daemon status + skill policy + root orchestrator',
        'model_loop', 'model_registry / provider_registry / workflow_registry -> live role coverage -> missing roles/blockers',
        'queue_loop', 'indy_queue -> indy_responses -> bytewax_compact_windows -> cloud_packet',
        'prompt_loop', 'prompt_recent -> prompts_filed -> prompt_work_order_links -> prompt_unlinked -> prompt_catalog_status'
    ) AS work_order_flow,
    (
        jsonb_build_array(
            './luci root orchestrator current --json',
            './luci openapi --json',
            './luci payload-archive-status --json',
            './luci model-routing-current --json',
            './luci model-routing-blockers --json',
            './luci model registry --json',
            './luci model registry current --json',
            './luci model registry raw --json',
            './luci provider current --json',
            './luci provider registry --json',
            './luci provider registry raw --json',
            './luci workflow current --json',
            './luci workflow registry raw --json',
            './luci capability current --json',
            './luci capability registry --json',
            './luci capability registry raw --json',
            './luci sheet current --json',
            './luci api root orchestrator current --json',
            './luci api manual current --json',
            './luci api active goal --json',
            './luci api daemon status --json',
            './luci api route catalog --json',
            './luci api test execution receipts --json',
            './luci api prompt catalog status --json',
            './luci api prompt filed --json',
            './luci api prompt links --json',
            './luci api prompt raw recent --json',
            './luci api prompt raw filed --json',
            './luci api prompt raw links --json',
            './luci api prompt raw unlinked --json',
            './luci api prompt raw catalog --json',
            './luci api root orchestrator current --json',
            './luci api manual current --json',
            './luci api chrono current --json',
            './luci api prompt recent --json',
            './luci api prompts filed --json',
            './luci api prompt work-order links --json',
            './luci api prompt unlinked --json',
            './luci api prompt catalog --json',
            './luci api root law docs --json',
            './luci api canon current --json',
            './luci api canon versions --json',
            './luci api bible edges --json',
            './luci api bible manuals --json',
            './luci api bible nodes --manual-id RUNTIME_GOVERNOR --json',
            './luci api bible route catalog --json',
            './luci api bible subtree --root-id 1.0.0 --json',
            './luci api model registry current --json',
            './luci api model registry raw --json',
            './luci api provider current --json',
            './luci api provider registry --json',
            './luci api workflow current --json',
            './luci api workflow registry raw --json',
            './luci api capability current --json',
            './luci api capability registry --json',
            './luci api capability registry raw --json',
            './luci api sheet current --json',
            './luci root-law-docs --json',
            './luci skill policy current --json',
            './luci todo current --json',
            'curl -sS http://127.0.0.1:3000/manual_current?limit=1',
            'curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1',
            'curl -sS http://127.0.0.1:3000/canon_current?limit=1',
            'curl -sS http://127.0.0.1:3000/active_goal?limit=1',
            'curl -sS http://127.0.0.1:3000/capability_registry?limit=1',
            'curl -sS http://127.0.0.1:3000/model_registry?limit=1',
            'curl -sS http://127.0.0.1:3000/provider_registry?limit=1',
            'curl -sS http://127.0.0.1:3000/workflow_registry?limit=1',
            'curl -sS http://127.0.0.1:3000/daemon_status?limit=5',
            'curl -sS http://127.0.0.1:3000/indy_queue?limit=5',
            'curl -sS http://127.0.0.1:3000/indy_responses?limit=5',
            'curl -sS http://127.0.0.1:3000/bytewax_compact_windows?limit=5'
        )
        ||
        jsonb_build_array(
            './luci api model routing current --json',
            './luci api model routing blockers --json',
            './luci model routing current --json',
            './luci model routing blockers --json',
            './luci api indy queue --json',
            './luci api indy responses --json',
            './luci api bytewax windows --json',
            './luci bytewax raw windows --json',
            './luci api cli process receipts --json',
            './luci api flow receipts --json',
            './luci api flow specs --json',
            './luci api skill policy current --json',
            './luci api todo current --json',
            './luci api cloud packet --work-order-id 00000000-0000-0000-0000-000000000000 --json',
            './luci api book source --json',
            './luci api book scan --json',
            './luci api book read-queue --json',
            './luci api book note --json',
            './luci api book receipt --json',
            './luci api book adapter --json',
            './luci api book candidate --json',
            './luci api book training --json',
            './luci api book raw source --json',
            './luci api book raw scan --json',
            './luci api book raw read-queue --json',
            './luci api book raw note --json',
            './luci api book raw receipt --json',
            './luci api book raw adapter --json',
            './luci api book raw candidate --json',
            './luci api book raw training --json',
            './luci api book read queue --json',
            './luci api bytewax compact windows --json',
            './luci api bytewax raw windows --json',
            '.venv/bin/python scripts/indy_daemon.py --once --json',
            '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json',
            '.venv/bin/python scripts/luci_todo.py --json',
            '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<objective text>"',
            '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<operator objective>"',
            '.venv/bin/python scripts/prompt_ledger_capture.py --json',
            '.venv/bin/python scripts/test_receipt_gate.py run --scope policy_and_retirement -- .venv/bin/python -m pytest -q tests/test_skill_policy_current_surface.py tests/test_indy_book_ops_schema.py tests/test_manual_current_surface.py tests/test_orchestrator_registry_routes.py',
            'curl -sS -X POST http://127.0.0.1:3000/rpc/cloud_packet -H ''content-type: application/json'' -d ''{"work_order_id":"...","max_chars":256,"max_items":1,"task_type":"...","target_model":"...","include_raw_bodies":false}''',
            'curl -sS -X POST http://127.0.0.1:3000/rpc/decompose_prompt_to_work_orders -H ''content-type: application/json'' -d ''{"prompt_id":"..."}''',
            'curl -sS -X POST http://127.0.0.1:3000/rpc/file_prompt -H ''content-type: application/json'' -d ''{"source":"...","raw_prompt_text":"..."}''',
            'curl -sS -X POST http://127.0.0.1:3000/rpc/link_prompt_work_order -H ''content-type: application/json'' -d ''{"p_prompt_id":"...","p_work_order_uuid":"..."}''',
            'curl -sS http://127.0.0.1:3000/',
            'curl -sS http://127.0.0.1:3000/api_bible_edges?limit=5',
            'curl -sS http://127.0.0.1:3000/api_bible_manuals?limit=5',
            'curl -sS http://127.0.0.1:3000/api_bible_nodes?manual_id=eq.RUNTIME_GOVERNOR&order=node_sort_key.asc&limit=5',
            'curl -sS http://127.0.0.1:3000/api_bible_route_catalog?limit=5',
            'curl -sS http://127.0.0.1:3000/api_bible_subtree?root_id=eq.1.0.0&limit=5',
            'curl -sS http://127.0.0.1:3000/api_root_law_docs?limit=1',
            'curl -sS http://127.0.0.1:3000/api_route_catalog?limit=1',
            'curl -sS http://127.0.0.1:3000/api_test_execution_receipts?limit=1',
            'curl -sS http://127.0.0.1:3000/api_test_execution_receipts?limit=3',
            'curl -sS http://127.0.0.1:3000/canon_versions?limit=5',
            'curl -sS http://127.0.0.1:3000/capability_current?limit=1',
            'curl -sS http://127.0.0.1:3000/chrono_current?limit=1',
            'curl -sS http://127.0.0.1:3000/cli_process_receipts?limit=3',
            'curl -sS http://127.0.0.1:3000/flow_receipts?limit=1',
            'curl -sS http://127.0.0.1:3000/flow_receipts?limit=3',
            'curl -sS http://127.0.0.1:3000/flow_specs?limit=1',
            'curl -sS http://127.0.0.1:3000/model_registry?limit=20',
            'curl -sS http://127.0.0.1:3000/model_registry_current?limit=1',
            'curl -sS http://127.0.0.1:3000/model_routing_blockers?limit=1',
            'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1',
            'curl -sS http://127.0.0.1:3000/payload_archive_status?limit=6',
            'curl -sS http://127.0.0.1:3000/prompt_catalog_status?limit=1',
            'curl -sS http://127.0.0.1:3000/prompt_recent?limit=5',
            'curl -sS http://127.0.0.1:3000/provider_current?limit=1',
            'curl -sS http://127.0.0.1:3000/sheet_current?limit=1',
            'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1',
            'curl -sS http://127.0.0.1:3000/todo_current?limit=5',
            'curl -sS http://127.0.0.1:3000/workflow_current?limit=1',
            './luci api ontology work batch --json',
            './luci api ontology work item --json',
            './luci api prompt recent --json',
            './luci api prompt catalog status --json',
            './luci api root-law-docs --json',
            './luci api payload archive status --json',
            './luci api model registry raw --json',
            './luci api provider registry --json',
            './luci api provider registry raw --json',
            './luci api workflow registry raw --json',
            './luci api capability registry raw --json',
            './luci api book adapter --json',
            './luci api book candidate --json',
            './luci api book training --json',
            './luci api ontology work raw batch --json',
            './luci api ontology work raw item --json',
            './luci prompt recent --json',
            './luci prompt catalog status --json',
            './luci flow specs --json',
            './luci flow receipts --json',
            './luci api rpc cloud-packet --work-order-id 00000000-0000-0000-0000-000000000000 --json',
            './luci api rpc decompose-prompt --payload-json {"prompt_id":"..."} --json',
            './luci api rpc file-prompt --payload-json {"source":"codex","raw_prompt_text":"..."} --json',
            './luci api rpc link-prompt --payload-json {"p_prompt_id":"...","p_work_order_uuid":"..."} --json'
        )
    ) AS next_commands,
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
    sub_orchestrators_rows.sub_orchestrators AS sub_orchestrators,
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
