-- FILE: 06_SCHEMA/038_absurd_river_wrapper.sql
-- PURPOSE: Register River/Bytewax + Phase 0.5 Streaming Brain GLiNER extraction under ABSURD queue spine.
-- COMPLIANCE: Idempotent, non-destructive. Does not mutate canonical graph, KORPUS custody, or temporal evidence.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_learning;

-- Legacy bounded River/Bytewax observation queue retained for continuity.
INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, status, max_attempts, visibility_timeout_seconds, notes)
VALUES (
  'river',
  'River/Bytewax streaming-learning workers',
  'active',
  3,
  300,
  'ABSURD wrapper queue for River/Bytewax bounded streaming-learning ticks over committed workflow_event rows. Wrapper may write lucidota_learning hints/runs only; it must not mutate canonical graph, KORPUS custody, or temporal evidence.'
)
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  status=EXCLUDED.status,
  max_attempts=EXCLUDED.max_attempts,
  visibility_timeout_seconds=EXCLUDED.visibility_timeout_seconds,
  notes=EXCLUDED.notes,
  updated_at=now();

-- Canonical Phase 0.5 streaming-brain queue target requested by the Operator.
INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, status, max_attempts, visibility_timeout_seconds, notes)
VALUES (
  'absurd.phase05_streaming_brain',
  'Phase 0.5 Streaming Brain / GLiNER evidence extractor',
  'active',
  3,
  300,
  'ABSURD wrapper queue for bounded zero-shot evidence extraction over KORPUS components. Writes span staging rows only; no graph materialization, no temporal evidence mutation, no KORPUS custody mutation.'
)
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  status=EXCLUDED.status,
  max_attempts=EXCLUDED.max_attempts,
  visibility_timeout_seconds=EXCLUDED.visibility_timeout_seconds,
  notes=EXCLUDED.notes,
  updated_at=now();

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
('absurd-river-bytewax-health-check',
 'absurd+river+bytewax',
 '022',
 'active',
 'scripts/absurd_river_worker.py --action worker-once --queue river --job-kind river_bytewax_health_check',
 '{"queue":"river","job_kind":"river_bytewax_health_check","state_database_url":"postgresql:///lucidota_state","bytewax_limit":25,"river_limit":5000}'::jsonb,
 '{"workflow_event":"uuid","bytewax_events_in":"integer","bytewax_hints_out":"integer","river_events_seen":"integer","river_groups":"integer"}'::jsonb,
 'ABSURD queue-spine wrapper for River/Bytewax streaming-learning health ticks. Runs bounded local learning scripts when executed; writes ABSURD receipts and lucidota_learning hint/run rows only.'
),
('absurd-phase05-streaming-brain-gliner-extract',
 'absurd+phase05+gliner',
 '025',
 'active',
 'scripts/absurd_river_worker.py --action worker-once --queue absurd.phase05_streaming_brain --job-kind gliner_zero_shot_extract',
 '{"queue":"absurd.phase05_streaming_brain","job_kind":"gliner_zero_shot_extract","storage_database_url":"postgresql:///lucidota_storage","state_database_url":"postgresql:///lucidota_state","component_limit":8,"label_source":"operator_ontology_fixture"}'::jsonb,
 '{"workflow_event":"uuid","run_uuid":"uuid","components_seen":"integer","spans_inserted":"integer","backend":"gliner|literal_fallback_no_gliner"}'::jsonb,
 'ABSURD queue-spine wrapper for GLiNER-style zero-shot evidence extraction over KORPUS component text. Output is span staging for later graph promotion; direct graph mutation is forbidden.'
)
ON CONFLICT (workflow_name) DO UPDATE SET
  owner=EXCLUDED.owner,
  phase=EXCLUDED.phase,
  status=EXCLUDED.status,
  command=EXCLUDED.command,
  inputs=EXCLUDED.inputs,
  outputs=EXCLUDED.outputs,
  notes=EXCLUDED.notes,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_learning.gliner_extraction_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_uuid uuid,
  queue_name text NOT NULL DEFAULT 'absurd.phase05_streaming_brain',
  workflow_name text NOT NULL DEFAULT 'absurd-phase05-streaming-brain-gliner-extract',
  extractor_name text NOT NULL DEFAULT 'gliner_zero_shot_extractor',
  extractor_version text NOT NULL DEFAULT 'v1',
  backend text NOT NULL,
  model_ref text NOT NULL DEFAULT '',
  label_source text NOT NULL DEFAULT 'operator_ontology_fixture',
  labels jsonb NOT NULL DEFAULT '[]'::jsonb,
  components_seen integer NOT NULL DEFAULT 0,
  spans_found integer NOT NULL DEFAULT 0,
  spans_inserted integer NOT NULL DEFAULT 0,
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  temporal_claims_mutated boolean NOT NULL DEFAULT false,
  korpus_custody_mutated boolean NOT NULL DEFAULT false,
  status text NOT NULL CHECK (status IN ('succeeded','failed','partial')),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.gliner_entity_span (
  span_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_uuid uuid NOT NULL REFERENCES lucidota_learning.gliner_extraction_run(run_uuid) ON DELETE CASCADE,
  job_uuid uuid,
  source_database text NOT NULL DEFAULT 'lucidota_storage',
  file_uuid uuid NOT NULL,
  component_uuid uuid NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  source_path text NOT NULL DEFAULT '',
  component_index integer,
  label text NOT NULL,
  matched_text text NOT NULL,
  start_char integer NOT NULL CHECK (start_char >= 0),
  end_char integer NOT NULL CHECK (end_char >= start_char),
  score numeric(8,6) NOT NULL DEFAULT 0,
  backend text NOT NULL,
  authority_class text NOT NULL DEFAULT 'model_computed_finding',
  payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  graph_promotion_status text NOT NULL DEFAULT 'not_promoted' CHECK (graph_promotion_status IN ('not_promoted','staged','promoted','rejected','deferred')),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (component_uuid, label, start_char, end_char, matched_text, backend)
);

CREATE INDEX IF NOT EXISTS gliner_entity_span_component_idx
  ON lucidota_learning.gliner_entity_span(component_uuid, created_at DESC);

CREATE INDEX IF NOT EXISTS gliner_entity_span_label_idx
  ON lucidota_learning.gliner_entity_span(label, created_at DESC);
