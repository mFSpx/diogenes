-- LUCIDOTA Sheet Workflow Spine Layer
-- Sheet-first route control + scratch/projection/receipt contracts for workflow domains.

CREATE SCHEMA IF NOT EXISTS lucidota_sheet;
CREATE SCHEMA IF NOT EXISTS lucidota_scratch;
CREATE SCHEMA IF NOT EXISTS lucidota_projection;

CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_route (
  sheet_workflow_route_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  domain text NOT NULL,
  target text NOT NULL,
  task_class text NOT NULL CHECK (task_class IN (
    'FILTER_SHEET','STATUS_SHEET','PIVOT_SHEET','SCORE_SHEET','DIFF_SHEET',
    'REFRESH_SHEET','EXPORT_SHEET','IMPORT_SHEET','PROMOTION_SHEET','DEADLETTER_SHEET'
  )),
  source_tables text[] NOT NULL DEFAULT ARRAY[]::text[],
  query_sql text NOT NULL,
  query_hash text NOT NULL,
  max_rows integer NOT NULL DEFAULT 1000 CHECK (max_rows BETWEEN 1 AND 50000),
  budget_ms integer NOT NULL DEFAULT 500 CHECK (budget_ms BETWEEN 1 AND 120000),
  priority integer NOT NULL DEFAULT 0 CHECK (priority BETWEEN 0 AND 100),
  last_checked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_receipt (
  sheet_workflow_receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sheet_workflow_route_uuid uuid,
  logical_domain text NOT NULL,
  route_stage text NOT NULL,
  target text NOT NULL,
  query_hash text NOT NULL,
  row_count bigint NOT NULL DEFAULT 0,
  duration_ms integer NOT NULL DEFAULT 0,
  memory_budget_mb integer NOT NULL DEFAULT 0,
  output_hash text NOT NULL DEFAULT '',
  status text NOT NULL CHECK (status IN ('PASS','FAIL','WARN','TIMEOUT','OOM')),
  error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT sheet_workflow_receipt_status_ck CHECK (status IS NOT NULL)
);

CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.sheet_workflow_route_scratch (
  logical_domain text NOT NULL,
  target text NOT NULL,
  route_sql text NOT NULL,
  score numeric NOT NULL,
  reason jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW lucidota_sheet.sheet_workflow_head AS
SELECT
  r.sheet_workflow_route_uuid,
  r.domain,
  r.target,
  r.task_class,
  r.priority,
  r.max_rows,
  r.budget_ms,
  r.created_at
FROM lucidota_sheet.sheet_workflow_route r
ORDER BY r.priority DESC, r.created_at ASC
LIMIT 100;

CREATE OR REPLACE VIEW lucidota_projection.sheet_workflow_route_sheet AS
SELECT
  r.domain,
  r.target,
  r.task_class,
  r.max_rows,
  r.budget_ms,
  COALESCE(p.output_hash, '') AS last_output_hash,
  CASE
    WHEN p.status = 'PASS' THEN 'RECENT_OK'
    WHEN p.status IN ('WARN','TIMEOUT','OOM') THEN 'DEGRADED'
    WHEN p.status = 'FAIL' THEN 'FAILED'
    ELSE 'UNSEEN'
  END AS last_status,
  p.created_at AS last_receipt_at
FROM lucidota_sheet.sheet_workflow_route r
LEFT JOIN LATERAL (
  SELECT status, output_hash, created_at
  FROM lucidota_sheet.sheet_workflow_receipt wf
  WHERE wf.sheet_workflow_route_uuid = r.sheet_workflow_route_uuid
  ORDER BY created_at DESC
  LIMIT 1
) p ON TRUE;

CREATE MATERIALIZED VIEW IF NOT EXISTS lucidota_projection.workflow_domain_pressure_sheet AS
SELECT
  domain,
  count(*) FILTER (WHERE query_hash IS NOT NULL) AS active_rows,
  count(*) FILTER (WHERE status = 'OPEN') AS open_rows,
  max(created_at) AS last_seen_at
FROM lucidota_sheet.sheet_task t
JOIN lucidota_sheet.sheet_workflow_route r
  ON r.target = t.target
GROUP BY domain;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sheet_workflow_route_target
  ON lucidota_sheet.sheet_workflow_route (target);

CREATE INDEX IF NOT EXISTS idx_sheet_workflow_receipt_route
  ON lucidota_sheet.sheet_workflow_receipt (sheet_workflow_route_uuid, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_workflow_scratch_created
  ON lucidota_scratch.sheet_workflow_route_scratch (created_at DESC);

CREATE OR REPLACE FUNCTION lucidota_sheet.record_sheet_workflow_receipt(
  p_sheet_workflow_route_uuid uuid,
  p_logical_domain text,
  p_route_stage text,
  p_target text,
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
  r_id uuid;
BEGIN
  INSERT INTO lucidota_sheet.sheet_workflow_receipt(
    sheet_workflow_route_uuid,
    logical_domain,
    route_stage,
    target,
    query_hash,
    row_count,
    duration_ms,
    memory_budget_mb,
    output_hash,
    status,
    error
  ) VALUES (
    p_sheet_workflow_route_uuid,
    p_logical_domain,
    p_route_stage,
    p_target,
    p_query_hash,
    p_row_count,
    p_duration_ms,
    p_memory_budget_mb,
    p_output_hash,
    p_status,
    p_error
  ) RETURNING sheet_workflow_receipt_uuid INTO r_id;
  RETURN r_id;
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_sheet.refresh_workflow_domain_pressure_sheet(p_concurrently boolean DEFAULT false)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
  started_at timestamptz := clock_timestamp();
  elapsed_ms integer;
  row_total bigint;
BEGIN
  IF p_concurrently THEN
    REFRESH MATERIALIZED VIEW CONCURRENTLY lucidota_projection.workflow_domain_pressure_sheet;
  ELSE
    REFRESH MATERIALIZED VIEW lucidota_projection.workflow_domain_pressure_sheet;
  END IF;
  SELECT count(*) INTO row_total FROM lucidota_projection.workflow_domain_pressure_sheet;
  elapsed_ms := (extract(epoch FROM (clock_timestamp() - started_at)) * 1000)::integer;
  RETURN jsonb_build_object(
    'sheet_id', 'workflow_domain_pressure_sheet',
    'rows', row_total,
    'duration_ms', elapsed_ms,
    'status', 'PASS',
    'refresh_sql',
      CASE WHEN p_concurrently THEN 'REFRESH MATERIALIZED VIEW CONCURRENTLY lucidota_projection.workflow_domain_pressure_sheet' ELSE 'REFRESH MATERIALIZED VIEW lucidota_projection.workflow_domain_pressure_sheet' END
  );
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_sheet.route_sql_signature(p_sql text)
RETURNS jsonb LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT jsonb_build_object(
    'query_hash', md5(coalesce(p_sql, '')),
    'is_refresh', (left(upper(coalesce(p_sql, '')), 33) = 'REFRESH MATERIALIZED VIEW')
  );
$$;
