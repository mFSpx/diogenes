-- Add alias route-catalog rows for the bible/openapi surface so manual_current can list the live paths.

BEGIN;

INSERT INTO lucidota_canon.api_route_catalog (
    route_id, method, path_pattern, description, target, sample_request, sample_response, status
) VALUES
    (
        '/',
        'GET',
        '/',
        'OpenAPI root document for the live PostgREST schema.',
        'openapi',
        '{}',
        '{"swagger":"2.0"}',
        'implemented'
    ),
    (
        'api_bible_edges',
        'GET',
        '/api_bible_edges',
        'Bible edge alias route for the canonical edge surface.',
        'lucidota_canon.api_bible_edges',
        '{"limit":"1"}',
        '{"edge_id":"..."}',
        'implemented'
    ),
    (
        'api_bible_manuals',
        'GET',
        '/api_bible_manuals',
        'Bible manual alias route for the canonical manual surface.',
        'lucidota_canon.api_bible_manuals',
        '{"limit":"1"}',
        '{"manual_id":"..."}',
        'implemented'
    ),
    (
        'api_bible_nodes',
        'GET',
        '/api_bible_nodes',
        'Bible node alias route for the canonical node surface.',
        'lucidota_canon.api_bible_nodes',
        '{"limit":"1"}',
        '{"node_id":"..."}',
        'implemented'
    ),
    (
        'api_bible_route_catalog',
        'GET',
        '/api_bible_route_catalog',
        'Bible route-catalog alias route for the canonical route catalog surface.',
        'lucidota_canon.api_bible_route_catalog',
        '{"limit":"1"}',
        '{"route_id":"..."}',
        'implemented'
    ),
    (
        'api_bible_subtree',
        'GET',
        '/api_bible_subtree',
        'Bible subtree alias route for the direct subtree surface.',
        'lucidota_canon.api_bible_subtree',
        '{"root_id":"1.0.0","limit":"1"}',
        '{"root_id":"1.0.0"}',
        'implemented'
    ),
    (
        'api_root_law_docs',
        'GET',
        '/api_root_law_docs',
        'Root-law docs alias route for the operator manual packet.',
        'lucidota_canon.api_root_law_docs',
        '{"limit":"1"}',
        '{"route_id":"root_law_docs"}',
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
