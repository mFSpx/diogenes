-- LUCIDOTA Sheet Layer v1
-- Database-native spreadsheet logic before algorithms/models.
-- Postgres = live truth/control; DuckDB/Parquet = file-sheet analytics beside it.

CREATE SCHEMA IF NOT EXISTS lucidota_sheet;
CREATE SCHEMA IF NOT EXISTS lucidota_scratch;
CREATE SCHEMA IF NOT EXISTS lucidota_projection;

CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_task (
  sheet_task_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_type text NOT NULL DEFAULT 'SHEET_TASK' CHECK (task_type = 'SHEET_TASK'),
  task_class text NOT NULL CHECK (task_class IN (
    'FILTER_SHEET', 'STATUS_SHEET', 'PIVOT_SHEET', 'SCORE_SHEET', 'DIFF_SHEET',
    'REFRESH_SHEET', 'EXPORT_SHEET', 'IMPORT_SHEET', 'PROMOTION_SHEET', 'DEADLETTER_SHEET'
  )),
  target text NOT NULL,
  title text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','RUNNING','BLOCKED','DONE','FAILED','DEAD')),
  friction_score integer NOT NULL DEFAULT 0 CHECK (friction_score BETWEEN 0 AND 100),
  receipt_count integer NOT NULL DEFAULT 0 CHECK (receipt_count >= 0),
  source_tables text[] NOT NULL DEFAULT ARRAY[]::text[],
  query_sql text NOT NULL DEFAULT '',
  max_rows integer NOT NULL DEFAULT 1000 CHECK (max_rows BETWEEN 1 AND 50000),
  budget_ms integer NOT NULL DEFAULT 500 CHECK (budget_ms BETWEEN 1 AND 120000),
  last_attempt_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  priority_band text GENERATED ALWAYS AS (
    CASE
      WHEN friction_score >= 80 THEN 'HOT'
      WHEN friction_score >= 40 THEN 'WARM'
      ELSE 'COLD'
    END
  ) STORED,
  route_band text GENERATED ALWAYS AS (
    CASE
      WHEN status = 'BLOCKED' THEN 'ASK_OPERATOR'
      WHEN receipt_count = 0 THEN 'PROBE'
      WHEN friction_score >= 80 THEN 'TRIAGE'
      ELSE 'WAIT'
    END
  ) STORED,
  needs_operator boolean GENERATED ALWAYS AS (status = 'BLOCKED') STORED
);

CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_refresh_receipt (
  receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sheet_id text NOT NULL,
  operation text NOT NULL,
  source_tables text[] NOT NULL DEFAULT ARRAY[]::text[],
  query_hash text NOT NULL,
  row_count bigint NOT NULL DEFAULT 0,
  duration_ms integer NOT NULL DEFAULT 0,
  memory_budget_mb integer NOT NULL DEFAULT 0,
  output_hash text NOT NULL DEFAULT '',
  status text NOT NULL CHECK (status IN ('PASS','FAIL','OOM','TIMEOUT')),
  error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION lucidota_sheet.touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_sheet_task_touch_updated_at ON lucidota_sheet.sheet_task;
CREATE TRIGGER trg_sheet_task_touch_updated_at
BEFORE UPDATE ON lucidota_sheet.sheet_task
FOR EACH ROW EXECUTE FUNCTION lucidota_sheet.touch_updated_at();

CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.route_score_scratch (
  packet_hash text NOT NULL,
  route text NOT NULL,
  score numeric NOT NULL,
  reason jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.runpod_chunk_embedding_stage (
  chunk_id text PRIMARY KEY,
  text_sha256 text NOT NULL,
  status text NOT NULL,
  provider text NOT NULL,
  model text NOT NULL,
  dimensions integer NOT NULL CHECK (dimensions > 0),
  embedding_json jsonb NOT NULL,
  error text,
  source_path text NOT NULL DEFAULT '',
  chunk_text_preview text NOT NULL DEFAULT '',
  row_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  imported_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE lucidota_scratch.runpod_chunk_embedding_stage
ADD COLUMN IF NOT EXISTS error text;

ALTER TABLE lucidota_scratch.runpod_chunk_embedding_stage
  ALTER COLUMN source_path SET DEFAULT '',
  ALTER COLUMN chunk_text_preview SET DEFAULT '',
  ALTER COLUMN row_json SET DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_runpod_chunk_embedding_stage_imported
ON lucidota_scratch.runpod_chunk_embedding_stage (imported_at DESC);

CREATE INDEX IF NOT EXISTS idx_runpod_chunk_embedding_stage_provider_model
ON lucidota_scratch.runpod_chunk_embedding_stage (provider, model, dimensions);

CREATE INDEX IF NOT EXISTS idx_sheet_task_active_hot
ON lucidota_sheet.sheet_task (friction_score DESC, created_at)
WHERE status IN ('OPEN', 'RUNNING', 'BLOCKED');

CREATE INDEX IF NOT EXISTS idx_sheet_task_route_band
ON lucidota_sheet.sheet_task (route_band, friction_score DESC)
WHERE status IN ('OPEN', 'RUNNING', 'BLOCKED');

CREATE OR REPLACE VIEW lucidota_sheet.active_work AS
SELECT
  t.sheet_task_uuid,
  t.target,
  t.title,
  t.priority_band,
  t.route_band,
  t.status,
  t.friction_score,
  t.receipt_count,
  t.last_attempt_at,
  t.created_at,
  (COALESCE(t.last_attempt_at, t.created_at) < now() - interval '1 hour') AS is_stale
FROM lucidota_sheet.sheet_task t
WHERE t.status IN ('OPEN', 'RUNNING', 'BLOCKED')
ORDER BY t.friction_score DESC, t.created_at ASC
LIMIT 1000;

CREATE OR REPLACE VIEW lucidota_sheet.next_work_batch AS
SELECT
  t.sheet_task_uuid,
  t.target,
  t.title,
  t.status,
  t.priority_band,
  t.friction_score,
  t.receipt_count,
  CASE
    WHEN t.status = 'BLOCKED' THEN 'ASK_OPERATOR'
    WHEN t.receipt_count = 0 THEN 'PROBE'
    WHEN t.friction_score > 80 THEN 'TRIAGE'
    WHEN COALESCE(t.last_attempt_at, t.created_at) < now() - interval '1 hour' THEN 'RETRY'
    ELSE 'WAIT'
  END AS next_action
FROM lucidota_sheet.sheet_task t
WHERE t.status IN ('OPEN', 'RUNNING', 'BLOCKED')
ORDER BY t.friction_score DESC, t.created_at ASC
LIMIT 25;

CREATE OR REPLACE VIEW lucidota_projection.runpod_chunk_embedding_sheet AS
SELECT
  chunk_id,
  text_sha256,
  status,
  provider,
  model,
  dimensions,
  jsonb_array_length(embedding_json) AS embedding_len,
  left(chunk_text_preview, 240) AS preview,
  imported_at
FROM lucidota_scratch.runpod_chunk_embedding_stage
ORDER BY imported_at DESC, chunk_id
LIMIT 50000;

CREATE MATERIALIZED VIEW IF NOT EXISTS lucidota_projection.case_pressure_sheet AS
SELECT
  target AS entity_uuid,
  count(*) FILTER (WHERE priority_band = 'HOT') AS hot_count,
  count(*) FILTER (WHERE priority_band = 'WARM') AS warm_count,
  count(*) FILTER (WHERE priority_band = 'COLD') AS cold_count,
  count(*) FILTER (WHERE route_band = 'ASK_OPERATOR') AS blocked_count,
  max(updated_at) AS last_seen_at
FROM lucidota_sheet.sheet_task
GROUP BY target;

CREATE UNIQUE INDEX IF NOT EXISTS idx_case_pressure_sheet_entity
ON lucidota_projection.case_pressure_sheet (entity_uuid);

CREATE OR REPLACE FUNCTION lucidota_sheet.refresh_case_pressure_sheet(p_concurrently boolean DEFAULT false)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
  started_at timestamptz := clock_timestamp();
  elapsed_ms integer;
  row_total bigint;
  refresh_sql text;
BEGIN
  IF p_concurrently THEN
    refresh_sql := 'REFRESH MATERIALIZED VIEW CONCURRENTLY lucidota_projection.case_pressure_sheet';
  ELSE
    refresh_sql := 'REFRESH MATERIALIZED VIEW lucidota_projection.case_pressure_sheet';
  END IF;
  EXECUTE refresh_sql;
  SELECT count(*) INTO row_total FROM lucidota_projection.case_pressure_sheet;
  elapsed_ms := (extract(epoch FROM (clock_timestamp() - started_at)) * 1000)::integer;
  PERFORM lucidota_sheet.record_refresh_receipt(
    'case_pressure_sheet',
    'refresh_projection',
    ARRAY['lucidota_sheet.sheet_task'],
    md5(refresh_sql),
    row_total,
    elapsed_ms,
    256,
    md5(row_total::text),
    'PASS',
    ''
  );
  RETURN jsonb_build_object('sheet_id','case_pressure_sheet','row_count',row_total,'duration_ms',elapsed_ms,'status','PASS');
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_sheet.record_refresh_receipt(
  p_sheet_id text,
  p_operation text,
  p_source_tables text[],
  p_query_hash text,
  p_row_count bigint,
  p_duration_ms integer,
  p_memory_budget_mb integer,
  p_output_hash text,
  p_status text,
  p_error text DEFAULT ''
)
RETURNS uuid LANGUAGE plpgsql AS $$
DECLARE
  rid uuid;
BEGIN
  INSERT INTO lucidota_sheet.sheet_refresh_receipt(
    sheet_id, operation, source_tables, query_hash, row_count, duration_ms,
    memory_budget_mb, output_hash, status, error
  ) VALUES (
    p_sheet_id, p_operation, p_source_tables, p_query_hash, p_row_count, p_duration_ms,
    p_memory_budget_mb, p_output_hash, p_status, p_error
  ) RETURNING receipt_uuid INTO rid;
  RETURN rid;
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_sheet.export_next_work_batch_csv(p_path text)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
  export_sql text;
BEGIN
  export_sql := format($fmt$
COPY (
  SELECT sheet_task_uuid, target, title, status, priority_band, friction_score, receipt_count, next_action
  FROM lucidota_sheet.next_work_batch
) TO %L WITH (FORMAT csv, HEADER true)
$fmt$, p_path);
  EXECUTE export_sql;
  RETURN jsonb_build_object('operation','export_sheet','sheet_id','next_work_batch','path',p_path,'status','PASS');
END;
$$;

COMMENT ON SCHEMA lucidota_sheet IS 'Database-native spreadsheet formulas/views/functions before algorithms/models.';
COMMENT ON SCHEMA lucidota_scratch IS 'Fast disposable unlogged scratch sheets; never canon.';
COMMENT ON SCHEMA lucidota_projection IS 'Cached materialized views / dashboard tabs.';
