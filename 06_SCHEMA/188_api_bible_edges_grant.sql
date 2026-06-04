-- Restore read access for the canonical bible-edges view used by LUCI.

BEGIN;

GRANT SELECT ON lucidota_canon.api_bible_edges TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
