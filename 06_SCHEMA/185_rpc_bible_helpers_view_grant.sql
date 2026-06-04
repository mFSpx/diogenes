-- Restore read access for the canonical bible node view used by manual_current and get_subtree.

BEGIN;

GRANT SELECT ON lucidota_canon.api_bible_nodes TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
