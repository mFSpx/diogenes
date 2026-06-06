-- Add Chrono alignment route to the live operator manual.

BEGIN;

GRANT SELECT ON lucidota_canon.chrono_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.manual_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
