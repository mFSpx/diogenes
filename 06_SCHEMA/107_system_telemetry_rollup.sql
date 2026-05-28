CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE TABLE IF NOT EXISTS lucidota_control.system_telemetry_rollup (
 rollup_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 mega_gate_report text,
 total_tools integer,
 temporal_claims bigint,
 absurd_jobs bigint,
 graph_items bigint,
 graph_edges bigint,
 graph_materializations bigint,
 ranking_violations integer NOT NULL DEFAULT 0,
 metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now()
);
