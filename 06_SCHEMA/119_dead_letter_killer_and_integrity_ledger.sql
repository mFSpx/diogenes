-- FILE: 06_SCHEMA/119_dead_letter_killer_and_integrity_ledger.sql
-- PURPOSE: native dead-letter fast-purge and relational integrity ledger.
-- COMPLIANCE: idempotent, non-destructive. No DLQ deletions; resolved rows stay in history.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_go;
CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE OR REPLACE FUNCTION lucidota_control.fn_dead_letter_killer_purge()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    killer_hint text := '';
BEGIN
    IF NEW.error_kind = 'JSON_MALFORMAT'
       OR NEW.error_message ILIKE '%helper_01_rc_2%'
       OR COALESCE(NEW.context->>'helper_rc', '') = 'helper_01_rc_2'
       OR COALESCE(NEW.context->>'error_context', '') ILIKE '%helper_01_rc_2%'
       OR COALESCE(NEW.context->>'resolution_hint', '') ILIKE '%helper_01_rc_2%'
    THEN
        killer_hint := COALESCE(NULLIF(NEW.error_kind, ''), 'helper_01_rc_2');
        NEW.context := COALESCE(NEW.context, '{}'::jsonb) || jsonb_build_object(
            'dlk_purged', true,
            'dlk_reason', killer_hint,
            'dlk_purged_at', now(),
            'dlk_source', 'lucidota_control.fn_dead_letter_killer_purge'
        );
        NEW.resolved := true;
        NEW.last_seen_at := now();
    END IF;

    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF to_regclass('lucidota_control.absurd_queue_dead_letter') IS NOT NULL THEN
        EXECUTE 'DROP TRIGGER IF EXISTS tr_absurd_dead_letter_killer ON lucidota_control.absurd_queue_dead_letter';
        EXECUTE $sql$
            CREATE TRIGGER tr_absurd_dead_letter_killer
            BEFORE INSERT ON lucidota_control.absurd_queue_dead_letter
            FOR EACH ROW
            EXECUTE FUNCTION lucidota_control.fn_dead_letter_killer_purge()
        $sql$;
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('lucidota_go.graph_item') IS NOT NULL
       AND to_regclass('lucidota_control.absurd_queue_dead_letter') IS NULL THEN
        EXECUTE $sql$
            CREATE OR REPLACE VIEW lucidota_go.v_system_integrity_ledger AS
            WITH item_counts AS (
                SELECT count(*)::bigint AS total_canonical_items
                FROM lucidota_go.graph_item
            ),
            journal_counts AS (
                SELECT
                    count(*)::bigint AS total_journal_rows,
                    count(*) FILTER (WHERE action = 'stage')::bigint AS total_journal_stages,
                    count(*) FILTER (WHERE item_uuid IS NULL AND edge_uuid IS NULL)::bigint AS orphan_journal_rows
                FROM lucidota_go.graph_journal
            )
            SELECT
                item_counts.total_canonical_items,
                journal_counts.total_journal_rows,
                journal_counts.total_journal_stages,
                journal_counts.orphan_journal_rows,
                NULL::bigint AS unconsumed_loss_signals,
                0::bigint AS loss_delta_sum,
                NULL::bigint AS total_dead_letters,
                NULL::bigint AS unresolved_dead_letters,
                CASE
                    WHEN item_counts.total_canonical_items = journal_counts.total_journal_stages
                    THEN 'PASS_COMPLETE'
                    ELSE 'FAIL_LEDGER_MISALIGNMENT'
                END AS journal_completeness_verdict,
                CASE
                    WHEN journal_counts.orphan_journal_rows = 0
                    THEN 'PASS_NO_ORPHAN_JOURNAL_ROWS'
                    ELSE 'FAIL_ORPHAN_JOURNAL_ROWS'
                END AS journal_orphan_verdict,
                'UNKNOWN_STATE_DB'::text AS feedback_consumption_verdict,
                'UNKNOWN_STATE_DB'::text AS dead_letter_verdict
            FROM item_counts
            CROSS JOIN journal_counts
        $sql$;
    ELSIF to_regclass('lucidota_control.absurd_queue_dead_letter') IS NOT NULL
       AND to_regclass('lucidota_go.graph_item') IS NULL THEN
        EXECUTE 'CREATE SERVER IF NOT EXISTS lucidota_storage_fdw FOREIGN DATA WRAPPER postgres_fdw OPTIONS (dbname ''lucidota_storage'')';
        EXECUTE format('DROP USER MAPPING IF EXISTS FOR %I SERVER lucidota_storage_fdw', current_user);
        EXECUTE format('CREATE USER MAPPING FOR %I SERVER lucidota_storage_fdw OPTIONS (user %L)', current_user, 'postgres');

        IF to_regclass('lucidota_go.graph_item') IS NULL THEN
            EXECUTE $sql$
                IMPORT FOREIGN SCHEMA lucidota_go
                LIMIT TO (graph_item, graph_journal)
                FROM SERVER lucidota_storage_fdw
                INTO lucidota_go
            $sql$;
        END IF;

        EXECUTE $sql$
            CREATE OR REPLACE VIEW lucidota_go.v_system_integrity_ledger AS
            WITH item_counts AS (
                SELECT count(*)::bigint AS total_canonical_items
                FROM lucidota_go.graph_item
            ),
            journal_counts AS (
                SELECT
                    count(*)::bigint AS total_journal_rows,
                    count(*) FILTER (WHERE action = 'stage')::bigint AS total_journal_stages,
                    count(*) FILTER (WHERE item_uuid IS NULL AND edge_uuid IS NULL)::bigint AS orphan_journal_rows
                FROM lucidota_go.graph_journal
            ),
            feedback_counts AS (
                SELECT
                    count(*) FILTER (WHERE consumed_at IS NULL)::bigint AS unconsumed_loss_signals,
                    coalesce(sum(loss_delta), 0)::bigint AS loss_delta_sum
                FROM lucidota_learning.operator_feedback_signal
            ),
            dead_letter_counts AS (
                SELECT
                    count(*) FILTER (WHERE resolved IS false)::bigint AS unresolved_dead_letters,
                    count(*)::bigint AS total_dead_letters
                FROM lucidota_control.absurd_queue_dead_letter
            )
            SELECT
                item_counts.total_canonical_items,
                journal_counts.total_journal_rows,
                journal_counts.total_journal_stages,
                journal_counts.orphan_journal_rows,
                feedback_counts.unconsumed_loss_signals,
                feedback_counts.loss_delta_sum,
                dead_letter_counts.total_dead_letters,
                dead_letter_counts.unresolved_dead_letters,
                CASE
                    WHEN item_counts.total_canonical_items = journal_counts.total_journal_stages
                    THEN 'PASS_COMPLETE'
                    ELSE 'FAIL_LEDGER_MISALIGNMENT'
                END AS journal_completeness_verdict,
                CASE
                    WHEN journal_counts.orphan_journal_rows = 0
                    THEN 'PASS_NO_ORPHAN_JOURNAL_ROWS'
                    ELSE 'FAIL_ORPHAN_JOURNAL_ROWS'
                END AS journal_orphan_verdict,
                CASE
                    WHEN feedback_counts.unconsumed_loss_signals = 0
                    THEN 'PASS_FEEDBACK_CONSUMED'
                    ELSE 'FAIL_UNCONSUMED_LOSS_SIGNALS'
                END AS feedback_consumption_verdict,
                CASE
                    WHEN dead_letter_counts.unresolved_dead_letters = 0
                    THEN 'PASS_DEAD_LETTERS_PURGED'
                    ELSE 'FAIL_DEAD_LETTERS_REMAIN'
                END AS dead_letter_verdict
            FROM item_counts
            CROSS JOIN journal_counts
            CROSS JOIN feedback_counts
            CROSS JOIN dead_letter_counts
        $sql$;
    END IF;
END;
$$;
