-- Add flow specs and flow receipts routes to the live route catalog.

BEGIN;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
(
    'flow_specs',
    'GET',
    '/flow_specs',
    'Live flow spec definitions for PromptFlow-style operator flows.',
    'lucidota_canon.flow_specs',
    '{"limit":"1"}',
    '{"flow_id":"..."}',
    'implemented'
),
(
    'flow_receipts',
    'GET',
    '/flow_receipts',
    'Live receipts for saved or executed flow specs.',
    'lucidota_canon.flow_receipts',
    '{"limit":"1"}',
    '{"receipt_id":"..."}',
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
