-- Add the API test execution receipts route to the live route catalog.

BEGIN;

INSERT INTO lucidota_canon.api_route_catalog
(route_id, method, path_pattern, description, target, sample_request, sample_response, status)
VALUES
(
    'api_test_execution_receipts',
    'GET',
    '/api_test_execution_receipts',
    'Live API test execution receipts for receipt-gated test history.',
    'lucidota_canon.api_test_execution_receipts',
    '{"limit":"1"}',
    '{"receipt_uuid":"..."}',
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
