-- Add CLI process authority receipts to the live operator manual.

BEGIN;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'lucidota_canon'
          AND c.relname = 'manual_current'
          AND a.attname = 'retired_surfaces'
          AND NOT EXISTS (
              SELECT 1
              FROM pg_attribute a2
              WHERE a2.attrelid = c.oid
                AND a2.attname = 'orchestration'
          )
    ) THEN
        EXECUTE 'ALTER VIEW lucidota_canon.manual_current RENAME COLUMN retired_surfaces TO orchestration';
    END IF;
END$$;

GRANT SELECT ON lucidota_canon.manual_current TO lucidota_postgrest_anon, mfspx;
GRANT SELECT ON lucidota_canon.api_route_catalog TO lucidota_postgrest_anon, mfspx;

NOTIFY pgrst, 'reload schema';

COMMIT;
