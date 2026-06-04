-- Add Bible RPC helpers to the live route catalog and grant execution to PostgREST/operator.

BEGIN;

GRANT EXECUTE ON FUNCTION lucidota_canon.get_subtree(text) TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION lucidota_canon.fn_bible_node_sort_key(text) TO lucidota_postgrest_anon, mfspx;
GRANT EXECUTE ON FUNCTION lucidota_canon.fn_bible_node_material(lucidota_canon.bible_nodes) TO lucidota_postgrest_anon, mfspx;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
(
    'get_subtree',
    'GET',
    '/rpc/get_subtree',
    'Fetch canonical subtree rooted at node id.',
    'lucidota_canon.get_subtree(root_id text)',
    '{"root_id":"4.9511.0"}',
    '{"node_id":"4.9511.0"}',
    'implemented'
),
(
    'fn_bible_node_sort_key',
    'GET',
    '/rpc/fn_bible_node_sort_key',
    'Compute bible node sort key for a node id.',
    'lucidota_canon.fn_bible_node_sort_key(p_node_id text)',
    '{"p_node_id":"4.9511.0"}',
    '{"sort_key":4}',
    'implemented'
),
(
    'fn_bible_node_material',
    'POST',
    '/rpc/fn_bible_node_material',
    'Materialize the canonical bible node row into a stable text payload.',
    'lucidota_canon.fn_bible_node_material(node_row lucidota_canon.bible_nodes)',
    '{"node_row":{"node_id":"4.9511.0"}}',
    '{"material":"..."}',
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
