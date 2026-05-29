-- FILE: 06_SCHEMA/040_graph_write_barrier_enforcement.sql
-- PURPOSE: enforce no direct canonical graph writes outside the promotion path.
-- COMPLIANCE: guardrail only; promotion workers must set LOCAL lucidota.graph_promotion_path='on'.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE OR REPLACE FUNCTION lucidota_go.enforce_graph_promotion_path()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  allowed text := coalesce(nullif(current_setting('lucidota.graph_promotion_path', true), ''), 'off');
  helper text := coalesce(nullif(current_setting('lucidota.graph_materialization_helper', true), ''), '');
BEGIN
  IF allowed <> 'on' OR helper <> 'scripts/graph_materialization_helper.py' THEN
    RAISE EXCEPTION 'direct canonical graph write blocked: canonical graph writes require graph_materialization_helper promotion transaction'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF to_regclass('lucidota_go.graph_item') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS trg_block_direct_graph_item_write ON lucidota_go.graph_item;
    CREATE TRIGGER trg_block_direct_graph_item_write
    BEFORE INSERT OR UPDATE OR DELETE ON lucidota_go.graph_item
    FOR EACH ROW EXECUTE FUNCTION lucidota_go.enforce_graph_promotion_path();
  END IF;
  IF to_regclass('lucidota_go.graph_edge') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS trg_block_direct_graph_edge_write ON lucidota_go.graph_edge;
    CREATE TRIGGER trg_block_direct_graph_edge_write
    BEFORE INSERT OR UPDATE OR DELETE ON lucidota_go.graph_edge
    FOR EACH ROW EXECUTE FUNCTION lucidota_go.enforce_graph_promotion_path();
  END IF;
END $$;

COMMIT;
