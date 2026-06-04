-- Add the bible-edges route to the live route catalog and keep it readable.

BEGIN;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
(
    'edges',
    'GET',
    '/api_bible_edges',
    'Live bible edges view for node relationship discovery.',
    'lucidota_canon.api_bible_edges',
    '{"limit":"1"}',
    '{"edge_id":"..."}',
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
