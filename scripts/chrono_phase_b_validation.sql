-- FILE: scripts/chrono_phase_b_validation.sql
-- PURPOSE: Read-only validation for Chrono-Ledger Phase B trigger/daemon contract.

\echo '--- LUCIDOTA CHRONO PHASE B VALIDATION ---'

SELECT
    to_regclass('lucidota_korpus.temporal_claim') AS temporal_claim_reg,
    to_regclass('lucidota_korpus.resolved_chrono_timeline') AS timeline_view_reg;

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'lucidota_korpus'
  AND tablename = 'temporal_claim'
  AND indexname IN ('idx_temporal_claim_dedupe','idx_temporal_claim_file_weight_time')
ORDER BY indexname;

SELECT
    tg.tgname AS trigger_name,
    cls.relname AS table_name,
    proc.proname AS function_name,
    NOT tg.tgisinternal AS user_trigger
FROM pg_trigger tg
JOIN pg_class cls ON cls.oid = tg.tgrelid
JOIN pg_proc proc ON proc.oid = tg.tgfoid
JOIN pg_namespace ns ON ns.oid = cls.relnamespace
WHERE ns.nspname = 'lucidota_korpus'
  AND cls.relname = 'file_object'
  AND tg.tgname = 'trg_notify_file_object_ingested';

SELECT evidence_source, trust_weight, COUNT(*) AS claims
FROM lucidota_korpus.temporal_claim
GROUP BY evidence_source, trust_weight
ORDER BY trust_weight DESC, evidence_source;

\echo '--- END LUCIDOTA CHRONO PHASE B VALIDATION ---'
