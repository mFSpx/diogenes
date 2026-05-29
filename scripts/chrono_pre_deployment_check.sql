-- FILE: scripts/chrono_pre_deployment_check.sql
-- EXPECTED ENVIRONMENT: PostgreSQL >= 14, initialized lucidota database.
-- PURPOSE: Read-only execution gate before applying 06_SCHEMA/025_chrono_ledger_core.sql.

\echo '--- LUCIDOTA CHRONO LEDGER PRE-DEPLOYMENT CHECK ---'

-- 1. Confirm primary schema target and exact file_object column contract.
SELECT table_schema, table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'lucidota_korpus'
  AND table_name = 'file_object'
ORDER BY ordinal_position;

-- 2. Confirm constraints and core key relationships for the storage anchor.
SELECT
    tc.table_schema,
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name,
    ccu.table_schema AS foreign_schema,
    ccu.table_name AS foreign_table,
    ccu.column_name AS foreign_column
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
 AND tc.table_name = kcu.table_name
LEFT JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name
 AND tc.table_schema = ccu.table_schema
WHERE tc.table_schema = 'lucidota_korpus'
  AND tc.table_name = 'file_object'
ORDER BY tc.constraint_type, tc.constraint_name, kcu.ordinal_position;

-- 3. Check existing overlapping registers before migration.
SELECT
    to_regclass('lucidota_korpus.file_object') AS korpus_file_object_reg,
    to_regclass('lucidota_korpus.file_occurrence') AS korpus_file_occurrence_reg,
    to_regclass('lucidota_korpus.component') AS korpus_component_reg,
    to_regclass('lucidota_korpus.temporal_claim') AS temporal_claim_reg,
    to_regclass('lucidota_korpus.resolved_chrono_timeline') AS timeline_view_reg;

-- 4. Confirm extension support for UUID defaults.
SELECT
    extname,
    extversion
FROM pg_extension
WHERE extname IN ('pgcrypto', 'uuid-ossp')
ORDER BY extname;

-- 5. Audit secondary sha256-bearing schema overlaps to prevent ambiguous promotability.
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE column_name = 'sha256'
  AND table_schema LIKE 'lucidota_%'
ORDER BY table_schema, table_name;

-- 6. Estimate zero-reingestion backfill scope from DB rows only.
SELECT
    COUNT(*) AS file_object_count,
    COUNT(*) FILTER (WHERE first_seen_path <> '') AS file_object_with_path,
    COUNT(*) FILTER (WHERE sha256 ~ '^[0-9a-f]{64}$') AS file_object_valid_sha256
FROM lucidota_korpus.file_object;

SELECT
    COUNT(*) AS occurrence_count,
    COUNT(*) FILTER (WHERE mtime IS NOT NULL) AS occurrence_with_mtime
FROM lucidota_korpus.file_occurrence;

SELECT
    COUNT(*) AS component_count,
    COUNT(*) FILTER (WHERE content <> '') AS component_with_content
FROM lucidota_korpus.component;

\echo '--- END CHRONO LEDGER PRE-DEPLOYMENT CHECK ---'
