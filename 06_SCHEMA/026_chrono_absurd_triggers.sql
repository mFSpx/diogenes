-- FILE: 06_SCHEMA/026_chrono_absurd_triggers.sql
-- COMPONENT: LUCIDOTA Chrono-Ledger Phase B ABSURD/Nervous-System Trigger
-- COMPLIANCE: Idempotent migration script; no destructive actions; no re-ingestion.
-- DEPENDENCIES: 06_SCHEMA/019_korpus_krampii.sql, 06_SCHEMA/025_chrono_ledger_core.sql
-- PURPOSE: Broadcast file_ingested notifications so the headless Rust service appends temporal claims continuously.

START TRANSACTION;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Prevent service/retry duplicate claims while keeping the ledger append-only for distinct evidence.
CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_claim_dedupe
    ON lucidota_korpus.temporal_claim (
        file_uuid,
        evidence_source,
        candidate_timestamp,
        raw_evidence
    )
    WHERE file_uuid IS NOT NULL
      AND raw_evidence IS NOT NULL;

-- ABSURD wake plane: Postgres is durable ledger; LISTEN/NOTIFY is only reactive wakeup.
CREATE OR REPLACE FUNCTION lucidota_korpus.notify_file_object_ingested()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_notify(
        'lucidota_korpus_file_ingested',
        json_build_object(
            'event', 'file_ingested',
            'file_uuid', NEW.file_uuid,
            'sha256', NEW.sha256,
            'first_seen_path', NEW.first_seen_path,
            'file_kind', NEW.file_kind,
            'notified_at', clock_timestamp()
        )::text
    );
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgname = 'trg_notify_file_object_ingested'
          AND tgrelid = 'lucidota_korpus.file_object'::regclass
    ) THEN
        CREATE TRIGGER trg_notify_file_object_ingested
        AFTER INSERT ON lucidota_korpus.file_object
        FOR EACH ROW
        EXECUTE FUNCTION lucidota_korpus.notify_file_object_ingested();
    END IF;
END;
$$;

COMMENT ON FUNCTION lucidota_korpus.notify_file_object_ingested() IS
    'Reactive wakeup trigger for headless Chrono-Ledger service; durable state remains in lucidota_korpus tables.';

COMMIT;
