-- Restore read access for the canonical bible-manual view used by LUCI and manual_current.

BEGIN;

GRANT SELECT ON lucidota_canon.api_bible_manuals TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
