-- Extend the operator manual with the model registry current packet.

BEGIN;

DROP VIEW IF EXISTS lucidota_canon.manual_current;

CREATE VIEW lucidota_canon.manual_current AS
WITH live_routes AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'route_id', route_id,
            'method', method,
            'path_pattern', path_pattern,
            'description', description,
            'target', target,
            'status', status
        )
        ORDER BY route_id
    ) AS route_list,
    count(*) AS route_count
    FROM lucidota_canon.api_route_catalog
    WHERE route_id IN (
        '/', 'manual_current', 'canon_current', 'canon_versions', 'active_goal', 'api_workflow_registry',
        'api_route_catalog', 'nodes', 'manuals', 'route_catalog', 'edges', 'api_bible_edges', 'api_bible_manuals',
        'api_bible_nodes', 'api_bible_route_catalog', 'api_bible_subtree', 'api_root_law_docs',
        'root_law_docs', 'subtree', 'get_subtree', 'fn_bible_node_sort_key', 'fn_bible_node_material',
        'cloud_packet', 'decompose_prompt_to_work_orders', 'file_prompt', 'link_prompt_work_order',
        'rpc/cloud_packet', 'rpc/decompose_prompt_to_work_orders', 'rpc/file_prompt',
        'rpc/fn_bible_node_material', 'rpc/fn_bible_node_sort_key', 'rpc/get_subtree', 'rpc/link_prompt_work_order',
        'flow_specs', 'flow_receipts',
        'api_test_execution_receipts',
        'capability_registry', 'capability_current', 'model_registry', 'model_registry_current',
        'model_routing_current', 'model_routing_blockers', 'provider_registry', 'provider_current', 'workflow_registry',
        'workflow_current', 'sheet_current', 'daemon_status', 'bytewax_compact_windows', 'indy_queue',
        'indy_responses', 'cli_process_receipts', 'payload_archive_status', 'cloud_packet', 'chrono_current',
        'book_source', 'book_scan', 'book_read_queue', 'book_note',
        'lora_candidate', 'lora_adapter', 'training_job', 'book_receipt',
        'ontology_work_batch', 'ontology_work_item', 'todo_current', 'skill_policy_current',
        'root_orchestrator_current', 'prompts_filed', 'prompt_work_order_links',
        'prompt_recent', 'prompt_unlinked', 'prompt_catalog_status',
        'file_prompt', 'link_prompt_work_order', 'decompose_prompt_to_work_orders'
    )
),
goal_row AS (
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
model_current_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(mc) ORDER BY mc.refreshed_at DESC), '[]'::jsonb) AS model_registry_current
    FROM lucidota_canon.model_registry_current mc
),
provider_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.provider_key), '[]'::jsonb) AS provider_registry
    FROM lucidota_canon.provider_registry p
),
provider_current_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(pc) ORDER BY pc.refreshed_at DESC), '[]'::jsonb) AS provider_current
    FROM lucidota_canon.provider_current pc
),
workflow_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(w) ORDER BY w.workflow_id), '[]'::jsonb) AS workflow_registry
    FROM lucidota_canon.workflow_registry w
),
workflow_current_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(wc) ORDER BY wc.refreshed_at DESC), '[]'::jsonb) AS workflow_current
    FROM lucidota_canon.workflow_current wc
),
model_routing_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(mr) ORDER BY mr.refreshed_at DESC), '[]'::jsonb) AS model_routing_current
    FROM lucidota_canon.model_routing_current mr
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
    SELECT COALESCE(jsonb_agg(to_jsonb(ir) ORDER BY ir.response_queued_at DESC), '[]'::jsonb) AS indy_responses
    FROM (
        SELECT *
        FROM lucidota_canon.indy_responses
        ORDER BY response_queued_at DESC
        LIMIT 5
    ) ir
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
api_route_catalog_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(arc) ORDER BY arc.route_id ASC), '[]'::jsonb) AS api_route_catalog
    FROM (
        SELECT *
        FROM lucidota_canon.api_route_catalog
        ORDER BY route_id ASC
        LIMIT 20
    ) arc
),
api_root_law_docs_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(rl) ORDER BY rl.refreshed_at DESC), '[]'::jsonb) AS api_root_law_docs
    FROM lucidota_canon.api_root_law_docs rl
),
sheet_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(s) ORDER BY s.refreshed_at DESC), '[]'::jsonb) AS sheet_current
    FROM lucidota_canon.sheet_current s
),
capability_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.refreshed_at DESC), '[]'::jsonb) AS capability_current
    FROM lucidota_canon.capability_current c
),
capability_registry_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(c) ORDER BY c.capability_key ASC), '[]'::jsonb) AS capability_registry
    FROM (
        SELECT *
        FROM lucidota_canon.capability_registry
        ORDER BY capability_key ASC
        LIMIT 5
    ) c
),
todo_rows AS (
    SELECT COALESCE(
        jsonb_agg(to_jsonb(t) ORDER BY t.created_at DESC),
        '[]'::jsonb
    ) AS todo_current
    FROM (
        SELECT *
        FROM lucidota_canon.todo_current
        WHERE status IN ('ready', 'queued', 'running')
        ORDER BY created_at DESC
        LIMIT 5
    ) t
),
todo_top AS (
    SELECT *
    FROM lucidota_canon.todo_current
    WHERE status IN ('ready', 'queued', 'running')
    ORDER BY created_at DESC
    LIMIT 1
),
skill_policy_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.updated_at DESC), '[]'::jsonb) AS skill_policy_current
    FROM lucidota_canon.skill_policy_current p
),
root_orchestrator_packet AS (
    SELECT to_jsonb(r) AS root_orchestrator_current
    FROM lucidota_canon.root_orchestrator_current r
    ORDER BY r.max_updated_at DESC
    LIMIT 1
),
root_orchestrator_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.max_updated_at DESC), '[]'::jsonb) AS root_orchestrator_current
    FROM lucidota_canon.root_orchestrator_current r
),
prompt_status_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.refreshed_at DESC), '[]'::jsonb) AS prompt_catalog_status
    FROM lucidota_canon.prompt_catalog_status p
),
prompt_recent_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompt_recent
    FROM (
        SELECT *
        FROM lucidota_canon.prompt_recent
        LIMIT 5
    ) p
),
prompts_filed_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompts_filed
    FROM (
        SELECT *
        FROM lucidota_canon.prompts_filed
        LIMIT 5
    ) p
),
prompt_links_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompt_work_order_links
    FROM (
        SELECT *
        FROM lucidota_canon.prompt_work_order_links
        LIMIT 5
    ) p
),
prompt_unlinked_rows AS (
    SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.received_at DESC), '[]'::jsonb) AS prompt_unlinked
    FROM (
        SELECT *
        FROM lucidota_canon.prompt_unlinked
        LIMIT 5
    ) p
),
missing_roles AS (
    SELECT COALESCE(t.missing_executor_roles, ARRAY[]::text[]) AS missing_executor_roles
    FROM todo_top t
),
selected_lanes AS (
    SELECT COALESCE(t.selected_lanes, '[]'::jsonb) AS selected_lanes
    FROM todo_top t
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
)
SELECT
    'LUCIDOTA_OPERATOR_MANUAL'::text AS manual_id,
    'LUCIDOTA Operator Manual'::text AS title,
    live_routes.route_count AS node_count,
    now() AS max_updated_at,
    live_routes.route_list,
    jsonb_build_object(
        'read_surface', 'PostgREST safe views and RPCs only',
        'write_surface', 'DB work orders and receipts only',
        'legacy_book_watcher', 'retired as authority',
        'skill_layers', 'execution aids only; live PostgREST/manual truth and GOALS handoffs win; prompt filing law is DB-backed and preserves raw text',
        'skill_policy_surface', 'DB-backed policy current route; live policy text wins over file-only policy snippets',
        'prompt_filing', 'file_prompt -> prompt ledger row -> prompt_work_order_links -> prompt_recent / prompt_unlinked -> prompt_catalog_status',
        'manual_source', 'live route catalog + daemon status + current goal + current todo batches + root orchestrator surface + prompt ledger + canon packet + skill policy packet + model routing packet + model routing blockers packet + model registry packet + sheet current packet + workflow current packet + capability registry packet + provider registry packet + indy queue/response packet + bytewax window packet + cloud packet + rpc alias packets + root-law docs packet + direct bible subtree packet + lora/training packet + receipts packet + sub-orchestrators packet + blocker packet'
    ) AS auth_expectations,
    jsonb_build_object(
        'book_ingest', 'book_source -> book_scan -> book_read_queue -> book_note -> lora_candidate -> lora_adapter -> training_job -> book_receipt',
        'indy_loop', 'queued row -> /indy_queue -> indy_daemon once/loop -> /indy_responses or receipt row',
        'queue_loop', 'indy_queue -> indy_responses -> bytewax_compact_windows -> cloud_packet',
        'mamba_role', 'DB queue/receipt/window watcher only; no BOOKS filesystem authority',
        'ontology_loop', 'messy operator text -> ontology_work_batch -> ontology_work_item -> executable route plan',
        'skill_policy', 'skill_policy_current -> operator-readable alignment policy -> manual surface',
        'root_orchestrator', 'root_orchestrator_current -> sub-orchestrator packets -> receipts -> manual update',
        'prompt_filing', 'operator or assistant prompt -> prompt ledger -> explicit linked work-order UUID or explicit unlinked reason',
        'chrono_alignment', 'prompt ledger -> work ledger -> execution history -> learning loop alignment packet',
        'model_routing', 'model_routing_current -> actual local role coverage, provider coverage, and missing-role blockers',
        'model_routing_blockers', 'model_routing_blockers -> explicit missing-role blocker packet and count',
        'rpc_aliases', 'rpc alias packets -> prompt filing, cloud packet, and prompt/work-order linking routes',
        'model_routing_blockers', 'model_routing_blockers -> explicit missing-role blocker packet and count',
        'sheet_layer', 'sheet_current -> task counts, projections, active work, and next work batch',
        'workflow_registry', 'workflow_current -> status and owner breakdown for active workflows and basic workflows',
        'capability_registry', 'capability_current -> capability lanes, group breakdown, workflow mapping, and active deployment rows',
        'provider_registry', 'provider_current -> provider lanes, kind breakdown, and local/cloud routing choices',
        'model_registry', 'model_registry_current -> model count, role breakdown, and loadout coverage',
        'canon', 'canon_current -> live canonical node snapshot and review state'
        ,'bible_docs', 'root_law_docs -> root-law manual packet + API route summary + contradiction ledger'
        ,'bible_subtree', 'api_bible_subtree -> direct recursive subtree view keyed by root_id'
    ) AS work_order_flow,
    (
        jsonb_build_object(
            'current_goal', goal_row.current_goal,
            'daemon_status', daemon_rows.daemon_status,
            'model_registry', model_rows.model_registry,
            'model_registry_current', model_current_rows.model_registry_current,
            'indy_queue', indy_queue_rows.indy_queue,
            'indy_responses', indy_response_rows.indy_responses,
            'bytewax_compact_windows', bytewax_rows.bytewax_compact_windows,
            'cloud_packet', cloud_packet_rows.cloud_packet,
            'api_route_catalog', api_route_catalog_rows.api_route_catalog,
            'canon_current', COALESCE((SELECT jsonb_agg(to_jsonb(cn) ORDER BY cn.updated_at DESC) FROM lucidota_canon.canon_current cn), '[]'::jsonb),
            'canon_versions', COALESCE((SELECT jsonb_agg(to_jsonb(cv) ORDER BY cv.promoted_at DESC) FROM lucidota_canon.canon_versions cv), '[]'::jsonb),
            'skill_policy_current', COALESCE((SELECT jsonb_agg(to_jsonb(sp) ORDER BY sp.updated_at DESC) FROM lucidota_canon.skill_policy_current sp), '[]'::jsonb),
            'provider_registry', provider_rows.provider_registry,
            'provider_current', provider_current_rows.provider_current,
            'workflow_registry', workflow_rows.workflow_registry,
            'workflow_current', workflow_current_rows.workflow_current,
            'model_routing_current', model_routing_rows.model_routing_current,
            'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb),
            'sheet_current', sheet_rows.sheet_current,
            'capability_current', capability_rows.capability_current,
            'todo_current', todo_rows.todo_current,
            'skill_policy_current', skill_policy_rows.skill_policy_current,
            'root_orchestrator_current', root_orchestrator_rows.root_orchestrator_current,
            'prompt_catalog_status', prompt_status_rows.prompt_catalog_status,
            'prompt_recent', prompt_recent_rows.prompt_recent,
            'prompts_filed', prompts_filed_rows.prompts_filed,
            'prompt_work_order_links', prompt_links_rows.prompt_work_order_links,
            'prompt_unlinked', prompt_unlinked_rows.prompt_unlinked,
            'missing_executor_roles', COALESCE((SELECT missing_executor_roles FROM missing_roles), ARRAY[]::text[]),
            'selected_lanes', COALESCE((SELECT selected_lanes FROM selected_lanes), '[]'::jsonb),
            'cli_process_receipts', cli_rows.cli_process_receipts,
            'flow_receipts', flow_receipt_rows.flow_receipts,
            'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts,
            'payload_archive_status', payload_rows.payload_archive_status,
            'flow_specs', flow_specs_rows.flow_specs
        )
        ||
        jsonb_build_object(
            'api_bible_manuals', COALESCE((SELECT jsonb_agg(to_jsonb(bm) ORDER BY bm.manual_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_manuals ORDER BY manual_id ASC LIMIT 5) bm), '[]'::jsonb),
            'api_bible_route_catalog', COALESCE((SELECT jsonb_agg(to_jsonb(br) ORDER BY br.route_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_route_catalog ORDER BY route_id ASC LIMIT 5) br), '[]'::jsonb),
            'api_bible_edges', COALESCE((SELECT jsonb_agg(to_jsonb(be) ORDER BY be.edge_id ASC) FROM (SELECT * FROM lucidota_canon.api_bible_edges ORDER BY edge_id ASC LIMIT 5) be), '[]'::jsonb),
            'api_bible_nodes', COALESCE((SELECT jsonb_agg(to_jsonb(bn) ORDER BY bn.node_sort_key ASC) FROM (SELECT * FROM lucidota_canon.api_bible_nodes WHERE manual_id = 'RUNTIME_GOVERNOR' ORDER BY node_sort_key ASC LIMIT 5) bn), '[]'::jsonb),
            'api_bible_subtree', COALESCE((SELECT jsonb_agg(to_jsonb(bs) ORDER BY bs.node_sort_key ASC) FROM (SELECT * FROM lucidota_canon.api_bible_subtree WHERE root_id = '1.0.0' ORDER BY node_sort_key ASC LIMIT 5) bs), '[]'::jsonb),
            'api_model_registry_current', model_current_rows.model_registry_current,
            'api_provider_current', provider_current_rows.provider_current,
            'api_workflow_current', workflow_current_rows.workflow_current,
            'api_capability_current', capability_rows.capability_current,
            'api_sheet_current', sheet_rows.sheet_current,
            'api_model_routing_current', model_routing_rows.model_routing_current,
            'api_model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb),
            'api_cli_process_receipts', cli_rows.cli_process_receipts,
            'api_flow_receipts', flow_receipt_rows.flow_receipts,
            'api_bytewax_windows', bytewax_rows.bytewax_compact_windows,
            'api_model_registry_raw', model_rows.model_registry,
            'api_capability_registry', capability_registry_rows.capability_registry,
            'api_provider_registry', provider_rows.provider_registry,
            'api_workflow_registry', workflow_rows.workflow_registry,
            'api_cloud_packet', cloud_packet_rows.cloud_packet,
            'capability_registry', capability_registry_rows.capability_registry,
            'book_source', COALESCE((SELECT jsonb_agg(to_jsonb(bs) ORDER BY bs.updated_at DESC) FROM (SELECT * FROM lucidota_canon.book_source ORDER BY updated_at DESC LIMIT 3) bs), '[]'::jsonb),
            'book_scan', COALESCE((SELECT jsonb_agg(to_jsonb(bs) ORDER BY bs.updated_at DESC) FROM (SELECT * FROM lucidota_canon.book_scan ORDER BY updated_at DESC LIMIT 3) bs), '[]'::jsonb),
            'book_read_queue', COALESCE((SELECT jsonb_agg(to_jsonb(brq) ORDER BY brq.updated_at DESC) FROM (SELECT * FROM lucidota_canon.book_read_queue ORDER BY updated_at DESC LIMIT 3) brq), '[]'::jsonb),
            'book_note', COALESCE((SELECT jsonb_agg(to_jsonb(bn) ORDER BY bn.updated_at DESC) FROM (SELECT * FROM lucidota_canon.book_note ORDER BY updated_at DESC LIMIT 3) bn), '[]'::jsonb),
            'book_receipt', COALESCE((SELECT jsonb_agg(to_jsonb(br) ORDER BY br.updated_at DESC) FROM (SELECT * FROM lucidota_canon.book_receipt ORDER BY updated_at DESC LIMIT 3) br), '[]'::jsonb),
            'lora_candidate', COALESCE((SELECT jsonb_agg(to_jsonb(lc) ORDER BY lc.updated_at DESC) FROM (SELECT * FROM lucidota_canon.lora_candidate ORDER BY updated_at DESC LIMIT 3) lc), '[]'::jsonb),
            'lora_adapter', COALESCE((SELECT jsonb_agg(to_jsonb(la) ORDER BY la.updated_at DESC) FROM (SELECT * FROM lucidota_canon.lora_adapter ORDER BY updated_at DESC LIMIT 3) la), '[]'::jsonb),
            'training_job', COALESCE((SELECT jsonb_agg(to_jsonb(tj) ORDER BY tj.updated_at DESC) FROM (SELECT * FROM lucidota_canon.training_job ORDER BY updated_at DESC LIMIT 3) tj), '[]'::jsonb),
            'ontology_work_batch', COALESCE((SELECT jsonb_agg(to_jsonb(ob) ORDER BY ob.updated_at DESC) FROM (SELECT * FROM lucidota_canon.ontology_work_batch ORDER BY updated_at DESC LIMIT 3) ob), '[]'::jsonb),
            'ontology_work_item', COALESCE((SELECT jsonb_agg(to_jsonb(oi) ORDER BY oi.updated_at DESC) FROM (SELECT * FROM lucidota_canon.ontology_work_item ORDER BY updated_at DESC LIMIT 5) oi), '[]'::jsonb),
            'api_root_law_docs', api_root_law_docs_rows.api_root_law_docs,
            'root_law_docs', COALESCE((SELECT jsonb_agg(to_jsonb(rl) ORDER BY rl.refreshed_at DESC) FROM lucidota_canon.root_law_docs rl), '[]'::jsonb),
            'get_subtree', COALESCE((SELECT jsonb_agg(to_jsonb(gs) ORDER BY gs.node_sort_key ASC) FROM lucidota_canon.get_subtree('4.9511.0') gs), '[]'::jsonb),
            'fn_bible_node_sort_key', COALESCE((SELECT jsonb_agg(to_jsonb(sk) ORDER BY sk) FROM (SELECT lucidota_canon.fn_bible_node_sort_key('4.9511.0') AS sk) s), '[]'::jsonb),
            'chrono_current', COALESCE((SELECT jsonb_agg(to_jsonb(c) ORDER BY c.refreshed_at DESC) FROM lucidota_canon.chrono_current c), '[]'::jsonb),
            'sub_orchestrators', COALESCE((SELECT root_orchestrator_current -> 'live_surface' -> 'sub_orchestrators' FROM root_orchestrator_packet), '[]'::jsonb),
            'blockers', jsonb_build_object(
                'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb)
            ),
            'receipts', jsonb_build_object(
                'cli_process_receipts', cli_rows.cli_process_receipts,
                'flow_receipts', flow_receipt_rows.flow_receipts,
                'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts
            )
        )
    ) AS live_surface,
    (
        SELECT COALESCE(jsonb_agg(cmd ORDER BY ord), '[]'::jsonb)
        FROM (
            VALUES
            (1, 'curl -sS http://127.0.0.1:3000/manual_current?limit=1'),
            (2, 'curl -sS http://127.0.0.1:3000/'),
            (3, 'curl -sS http://127.0.0.1:3000/chrono_current?limit=1'),
            (4, 'curl -sS http://127.0.0.1:3000/model_routing_current?limit=1'),
            (5, 'curl -sS http://127.0.0.1:3000/model_routing_blockers?limit=1'),
            (6, 'curl -sS http://127.0.0.1:3000/model_registry_current?limit=1'),
            (7, 'curl -sS http://127.0.0.1:3000/canon_current?limit=1'),
            (8, 'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1'),
            (9, 'curl -sS http://127.0.0.1:3000/todo_current?limit=5'),
            (10, 'curl -sS http://127.0.0.1:3000/sheet_current?limit=1'),
            (11, 'curl -sS http://127.0.0.1:3000/workflow_current?limit=1'),
            (12, 'curl -sS http://127.0.0.1:3000/capability_current?limit=1'),
            (13, 'curl -sS http://127.0.0.1:3000/provider_current?limit=1'),
            (14, 'curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1'),
            (14.1, 'curl -sS http://127.0.0.1:3000/active_goal?limit=1'),
            (14.2, 'curl -sS http://127.0.0.1:3000/bytewax_compact_windows?limit=5'),
            (14.3, 'curl -sS http://127.0.0.1:3000/capability_registry?limit=1'),
            (14.4, 'curl -sS http://127.0.0.1:3000/daemon_status?limit=5'),
            (14.5, 'curl -sS http://127.0.0.1:3000/indy_queue?limit=5'),
            (14.6, 'curl -sS http://127.0.0.1:3000/indy_responses?limit=5'),
            (14.7, 'curl -sS http://127.0.0.1:3000/model_registry?limit=1'),
            (14.8, 'curl -sS http://127.0.0.1:3000/provider_registry?limit=1'),
            (14.9, 'curl -sS http://127.0.0.1:3000/workflow_registry?limit=1'),
            (15, 'curl -sS http://127.0.0.1:3000/todo_current?limit=5'),
            (16, 'curl -sS http://127.0.0.1:3000/skill_policy_current?limit=1'),
            (17, 'curl -sS http://127.0.0.1:3000/prompt_catalog_status?limit=1'),
            (18, 'curl -sS http://127.0.0.1:3000/api_route_catalog?limit=1'),
            (19, 'curl -sS http://127.0.0.1:3000/flow_specs?limit=1'),
            (20, 'curl -sS http://127.0.0.1:3000/flow_receipts?limit=1'),
            (21, 'curl -sS http://127.0.0.1:3000/cli_process_receipts?limit=3'),
            (22, 'curl -sS http://127.0.0.1:3000/flow_receipts?limit=3'),
            (23, 'curl -sS http://127.0.0.1:3000/api_test_execution_receipts?limit=3'),
            (24, 'curl -sS http://127.0.0.1:3000/api_test_execution_receipts?limit=1'),
            (25, 'curl -sS http://127.0.0.1:3000/prompt_recent?limit=5'),
            (26, 'curl -sS http://127.0.0.1:3000/payload_archive_status?limit=6'),
            (27, 'curl -sS http://127.0.0.1:3000/api_bible_edges?limit=5'),
            (28, 'curl -sS http://127.0.0.1:3000/api_bible_manuals?limit=5'),
            (29, 'curl -sS http://127.0.0.1:3000/api_bible_route_catalog?limit=5'),
            (30, 'curl -sS http://127.0.0.1:3000/api_bible_nodes?manual_id=eq.RUNTIME_GOVERNOR&order=node_sort_key.asc&limit=5'),
            (31, 'curl -sS http://127.0.0.1:3000/api_bible_subtree?root_id=eq.1.0.0&limit=5'),
            (32, 'curl -sS http://127.0.0.1:3000/api_root_law_docs?limit=1'),
            (33, 'curl -sS -X POST http://127.0.0.1:3000/rpc/cloud_packet -H ''content-type: application/json'' -d ''{"work_order_id":"...","max_chars":256,"max_items":1,"task_type":"...","target_model":"...","include_raw_bodies":false}'''),
            (34, 'curl -sS -X POST http://127.0.0.1:3000/rpc/file_prompt -H ''content-type: application/json'' -d ''{"source":"...","raw_prompt_text":"..."}'''),
            (35, 'curl -sS -X POST http://127.0.0.1:3000/rpc/decompose_prompt_to_work_orders -H ''content-type: application/json'' -d ''{"prompt_id":"..."}'''),
            (36, 'curl -sS -X POST http://127.0.0.1:3000/rpc/link_prompt_work_order -H ''content-type: application/json'' -d ''{"p_prompt_id":"...","p_work_order_uuid":"..."}'''),
            (37, '.venv/bin/python scripts/ontology_work_compiler.py --json --text "<objective text>"'),
            (38, '.venv/bin/python scripts/indy_daemon.py --once --json'),
            (39, '.venv/bin/python scripts/indy_runtime_broker.py snapshot --json'),
            (40, '.venv/bin/python scripts/prompt_ledger_capture.py --json'),
            (41, './luci payload-archive-status --json'),
            (42, './luci openapi --json'),
            (43, './luci root orchestrator current --json'),
            (44, './luci model-routing-current --json'),
            (45, './luci model routing current --json'),
            (46, './luci model-routing-blockers --json'),
            (47, './luci model routing blockers --json'),
            (48, './luci model registry --json'),
            (49, './luci model registry raw --json'),
            (50, './luci model registry current --json'),
            (51, './luci provider current --json'),
            (52, './luci provider registry --json'),
            (53, './luci provider registry raw --json'),
            (54, './luci capability current --json'),
            (55, './luci capability registry --json'),
            (56, './luci capability registry raw --json'),
            (57, './luci workflow registry raw --json'),
            (58, './luci api workflow registry raw --json'),
            (59, './luci workflow current --json'),
            (60, './luci api route catalog --json'),
            (61, './luci api root law docs --json'),
            (62, './luci api prompt recent --json'),
            (63, './luci api prompt filed --json'),
            (64, './luci api prompts filed --json'),
            (65, './luci api prompt links --json'),
            (66, './luci api prompt work-order links --json'),
            (67, './luci api prompt unlinked --json'),
            (68, './luci api prompt raw recent --json'),
            (69, './luci api prompt raw filed --json'),
            (70, './luci api prompt raw links --json'),
            (71, './luci api prompt raw unlinked --json'),
            (72, './luci api prompt raw catalog --json'),
            (73, './luci api prompt catalog --json'),
            (74, './luci api prompt catalog status --json'),
            (75, './luci api active goal --json'),
            (76, './luci api bible edges --json'),
            (77, './luci api bible manuals --json'),
            (78, './luci api bible nodes --manual-id RUNTIME_GOVERNOR --json'),
            (79, './luci api bible route catalog --json'),
            (80, './luci api bible subtree --root-id 1.0.0 --json'),
            (81, './luci api book adapter --json'),
            (82, './luci api book candidate --json')
        ) AS v(ord, cmd)
    )
    || jsonb_build_array(
        './luci api active goal --json',
        './luci api canon current --json',
        './luci api canon versions --json',
        './luci api route catalog --json',
        './luci api root law docs --json',
        './luci api cli process receipts --json',
        './luci api flow receipts --json',
        './luci api test execution receipts --json',
        './luci api model registry current --json',
        './luci api provider current --json',
        './luci api workflow current --json',
        './luci api capability current --json',
        './luci api sheet current --json',
        './luci api model routing current --json',
        './luci api model routing blockers --json',
        './luci api prompt catalog status --json',
        './luci api prompt recent --json',
        './luci api prompts filed --json',
        './luci api prompt work-order links --json',
        './luci api prompt unlinked --json',
        './luci api bible manuals --json',
        './luci api bible route catalog --json',
        './luci api book source --json',
        './luci api book scan --json',
        './luci api book read queue --json',
        './luci api book note --json',
        './luci api book receipt --json'
        ,'./luci api book raw source --json'
        ,'./luci api book raw scan --json'
        ,'./luci api book raw read-queue --json'
        ,'./luci api book raw note --json'
        ,'./luci api book raw receipt --json'
        ,'./luci api book raw adapter --json'
        ,'./luci api book raw candidate --json'
        ,'./luci api book raw training --json'
        ,'./luci api bytewax compact windows --json'
        ,'./luci api bytewax raw windows --json'
        ,'./luci api cloud packet --work-order-id 00000000-0000-0000-0000-000000000000 --json'
        ,'./luci api rpc cloud-packet --work-order-id 00000000-0000-0000-0000-000000000000 --json'
        ,'./luci api model registry raw --json'
        ,'./luci api provider registry --json'
        ,'./luci api provider registry raw --json'
        ,'./luci api capability registry --json'
        ,'./luci api capability registry raw --json'
        ,'./luci api workflow registry raw --json'
        ,'./luci api root orchestrator current --json'
        ,'./luci api manual current --json'
        ,'./luci api daemon status --json'
        ,'./luci api indy queue --json'
        ,'./luci api indy responses --json'
        ,'./luci api bytewax windows --json'
        ,'./luci bytewax raw windows --json'
        ,'./luci api flow specs --json'
        ,'./luci flow specs --json'
        ,'./luci api flow receipts --json'
        ,'./luci flow receipts --json'
        ,'./luci api chrono current --json'
        ,'./luci api payload archive status --json'
        ,'./luci api root-law-docs --json'
        ,'./luci root-law-docs --json'
        ,'./luci api skill policy current --json'
        ,'./luci skill policy current --json'
        ,'./luci api todo current --json'
        ,'./luci todo current --json'
        ,'./luci api ontology work batch --json'
        ,'./luci api ontology work item --json'
        ,'./luci api ontology work raw batch --json'
        ,'./luci api ontology work raw item --json'
        ,'./luci prompt recent --json'
        ,'./luci prompt catalog status --json'
        ,'./luci api book read-queue --json'
        ,'./luci api book training --json'
        ,'./luci api rpc decompose-prompt --payload-json {"prompt_id":"..."} --json'
        ,'./luci api rpc file-prompt --payload-json {"source":"codex","raw_prompt_text":"..."} --json'
        ,'./luci api rpc link-prompt --payload-json {"p_prompt_id":"...","p_work_order_uuid":"..."} --json'
        ,'./luci sheet current --json'
        ,'.venv/bin/python scripts/luci_todo.py --json'
        ,'.venv/bin/python scripts/ontology_work_compiler.py --json --text "<operator objective>"'
        ,'.venv/bin/python scripts/test_receipt_gate.py run --scope policy_and_retirement -- .venv/bin/python -m pytest -q tests/test_skill_policy_current_surface.py tests/test_indy_book_ops_schema.py tests/test_manual_current_surface.py tests/test_orchestrator_registry_routes.py'
        ,'curl -sS http://127.0.0.1:3000/canon_versions?limit=5'
        ,'curl -sS http://127.0.0.1:3000/model_registry?limit=20'
    ) AS next_commands,
    jsonb_build_array(
        'BOOKS folder watcher authority',
        'hand-written manual slop',
        'raw corpus prompts',
        'unbounded whole-table dumps'
    ) AS retired_surfaces,
    COALESCE((SELECT root_orchestrator_current -> 'live_surface' -> 'sub_orchestrators' FROM root_orchestrator_packet), '[]'::jsonb) AS sub_orchestrators,
    jsonb_build_object(
        'model_routing_blockers', COALESCE((SELECT jsonb_agg(to_jsonb(mrb) ORDER BY mrb.refreshed_at DESC) FROM lucidota_canon.model_routing_blockers mrb), '[]'::jsonb)
    ) AS blockers,
    jsonb_build_object(
        'cli_process_receipts', cli_rows.cli_process_receipts,
        'flow_receipts', flow_receipt_rows.flow_receipts,
        'api_test_execution_receipts', test_receipt_rows.api_test_execution_receipts
    ) AS receipts,
    goal_row.current_goal AS goal,
    jsonb_build_object(
        'statement', 'Postgres/PostgREST is truth; files are cache/export/artifact unless API points to them; DB-worthy state goes to DB; Rust-worthy code becomes Rust only after contract, tests, and A/B receipt.'
    ) AS db_law
FROM live_routes
CROSS JOIN goal_row
CROSS JOIN daemon_rows
CROSS JOIN model_rows
CROSS JOIN model_current_rows
CROSS JOIN provider_rows
CROSS JOIN provider_current_rows
CROSS JOIN workflow_rows
CROSS JOIN workflow_current_rows
CROSS JOIN model_routing_rows
CROSS JOIN indy_queue_rows
CROSS JOIN indy_response_rows
CROSS JOIN bytewax_rows
CROSS JOIN cloud_packet_rows
CROSS JOIN api_route_catalog_rows
CROSS JOIN api_root_law_docs_rows
CROSS JOIN sheet_rows
CROSS JOIN capability_rows
CROSS JOIN capability_registry_rows
CROSS JOIN todo_rows
CROSS JOIN skill_policy_rows
CROSS JOIN root_orchestrator_rows
CROSS JOIN prompt_status_rows
CROSS JOIN prompt_recent_rows
CROSS JOIN prompts_filed_rows
CROSS JOIN prompt_links_rows
CROSS JOIN prompt_unlinked_rows
CROSS JOIN missing_roles
CROSS JOIN selected_lanes
CROSS JOIN cli_rows
CROSS JOIN flow_receipt_rows
CROSS JOIN test_receipt_rows
CROSS JOIN flow_specs_rows
CROSS JOIN payload_rows;

GRANT SELECT ON lucidota_canon.manual_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
