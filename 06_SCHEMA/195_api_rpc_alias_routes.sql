-- Add alias route-catalog rows for RPC endpoints so manual_current can list the live /rpc paths.

BEGIN;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    (
        'rpc/cloud_packet',
        'POST',
        '/rpc/cloud_packet',
        'RPC alias route for bounded cloud packets.',
        'lucidota_canon.cloud_packet',
        '{"work_order_id":"..."}',
        '{"work_order_id":"..."}',
        'implemented'
    ),
    (
        'rpc/decompose_prompt_to_work_orders',
        'POST',
        '/rpc/decompose_prompt_to_work_orders',
        'RPC alias route for prompt decomposition.',
        'lucidota_canon.decompose_prompt_to_work_orders',
        '{"prompt_id":"..."}',
        '{"prompt_id":"..."}',
        'implemented'
    ),
    (
        'rpc/file_prompt',
        'POST',
        '/rpc/file_prompt',
        'RPC alias route for prompt filing.',
        'lucidota_canon.file_prompt',
        '{"raw_prompt_text":"..."}',
        '{"prompt_id":"..."}',
        'implemented'
    ),
    (
        'rpc/fn_bible_node_material',
        'GET',
        '/rpc/fn_bible_node_material',
        'RPC alias route for bible node materialization.',
        'lucidota_canon.fn_bible_node_material',
        '{"node_id":"..."}',
        '{"material":"..."}',
        'implemented'
    ),
    (
        'rpc/fn_bible_node_sort_key',
        'GET',
        '/rpc/fn_bible_node_sort_key',
        'RPC alias route for bible node sort-key lookup.',
        'lucidota_canon.fn_bible_node_sort_key',
        '{"node_id":"..."}',
        '{"sort_key":"..."}',
        'implemented'
    ),
    (
        'rpc/get_subtree',
        'GET',
        '/rpc/get_subtree',
        'RPC alias route for canonical subtree lookup.',
        'lucidota_canon.get_subtree',
        '{"root_id":"..."}',
        '{"node_id":"..."}',
        'implemented'
    ),
    (
        'rpc/link_prompt_work_order',
        'POST',
        '/rpc/link_prompt_work_order',
        'RPC alias route for prompt/work-order linking.',
        'lucidota_canon.link_prompt_work_order',
        '{"prompt_id":"...","work_order_uuid":"..."}',
        '{"prompt_id":"..."}',
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

GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
