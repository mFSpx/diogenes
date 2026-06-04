-- Restore read access for the canonical bible-route-catalog view used by LUCI.

BEGIN;

GRANT SELECT ON lucidota_canon.api_bible_route_catalog TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
