-- FILE: 06_SCHEMA/074_graph_journal_write_barrier.sql
-- PURPOSE: extend direct graph-write blocker to graph_journal.
-- COMPLIANCE: graph_journal writes are allowed only inside promotion transaction.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_go;

DO $$
BEGIN
  IF to_regclass('lucidota_go.graph_journal') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS trg_block_direct_graph_journal_write ON lucidota_go.graph_journal;
    CREATE TRIGGER trg_block_direct_graph_journal_write
    BEFORE INSERT OR UPDATE OR DELETE ON lucidota_go.graph_journal
    FOR EACH ROW EXECUTE FUNCTION lucidota_go.enforce_graph_promotion_path();
  END IF;
END $$;

COMMIT;
