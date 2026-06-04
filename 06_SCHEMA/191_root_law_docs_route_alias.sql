-- Expose the live root-law docs packet on the actual PostgREST path.

BEGIN;

CREATE OR REPLACE VIEW lucidota_canon.root_law_docs AS
SELECT * FROM lucidota_canon.api_root_law_docs;

GRANT SELECT ON lucidota_canon.root_law_docs TO lucidota_postgrest_anon, mfspx;

UPDATE lucidota_canon.api_route_catalog
SET target = 'lucidota_canon.root_law_docs',
    updated_at = now()
WHERE route_id = 'root_law_docs';

NOTIFY pgrst, 'reload schema';

COMMIT;
