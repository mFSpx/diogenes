-- FILE: 06_SCHEMA/112_project2501_core_board.sql
-- PURPOSE: Project 2501 board-move core tables: envelopes, routes, receipts, River rows.
-- COMPLIANCE: metrics/receipts/control-plane only; graph writes remain staged_only.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.raw_artifact (
  raw_artifact_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  raw_ref text NOT NULL UNIQUE,
  raw_sha256 text NOT NULL CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),
  hash_algo text NOT NULL DEFAULT 'sha256',
  source text NOT NULL,
  actor text NOT NULL,
  byte_count integer NOT NULL CHECK (byte_count >= 0),
  char_count integer NOT NULL CHECK (char_count >= 0),
  mime_type text NOT NULL DEFAULT 'text/plain',
  storage_hint text NOT NULL DEFAULT 'inline_or_receipt',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.event_envelope (
  envelope_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL UNIQUE CHECK (event_id ~ '^[0-9a-f]{64}$'),
  ts timestamptz NOT NULL,
  source text NOT NULL,
  actor text NOT NULL CHECK (actor IN ('operator','codex','groq','cohere','local_model','scraper','auditor','worker','system','external','unknown')),
  raw_ref text NOT NULL,
  raw_artifact_uuid uuid REFERENCES lucidota_control.raw_artifact(raw_artifact_uuid) ON DELETE SET NULL,
  verbatim_hash text NOT NULL CHECK (verbatim_hash ~ '^[0-9a-f]{64}$'),
  hash_algo text NOT NULL DEFAULT 'sha256',
  text text NOT NULL DEFAULT '',
  entities jsonb NOT NULL DEFAULT '[]'::jsonb,
  claims jsonb NOT NULL DEFAULT '[]'::jsonb,
  actions_requested jsonb NOT NULL DEFAULT '[]'::jsonb,
  artifacts_referenced jsonb NOT NULL DEFAULT '[]'::jsonb,
  risk_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
  route_candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
  board_features jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding_ref text,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_event_envelope_source_created
  ON lucidota_control.event_envelope(source, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.route_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  lane text NOT NULL CHECK (lane IN ('fast','slow','audit','external','dead_letter')),
  route_key text NOT NULL,
  gate_order jsonb NOT NULL DEFAULT '["deterministic_rules","treelite_gate","model_fallback"]'::jsonb,
  deterministic_rule jsonb NOT NULL DEFAULT '{}'::jsonb,
  treelite_gate jsonb NOT NULL DEFAULT '{}'::jsonb,
  model_fallback jsonb NOT NULL DEFAULT '{"used":false}'::jsonb,
  cost jsonb NOT NULL DEFAULT '{}'::jsonb,
  expected_gain numeric NOT NULL DEFAULT 0 CHECK (expected_gain >= 0),
  confidence numeric NOT NULL DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
  graph_write_mode text NOT NULL DEFAULT 'staged_only' CHECK (graph_write_mode IN ('none','staged_only','materialization_gate')),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_route_decision_lane_created
  ON lucidota_control.route_decision(lane, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.board_position (
  position_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  position_key text NOT NULL,
  event_id text REFERENCES lucidota_control.event_envelope(event_id) ON DELETE SET NULL,
  feature_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  resource_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  graph_authority_snapshot jsonb NOT NULL DEFAULT '{"canonical_graph_write_allowed":false}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(position_key, event_id)
);

CREATE TABLE IF NOT EXISTS lucidota_control.work_order (
  work_order_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  decision_uuid uuid REFERENCES lucidota_control.route_decision(decision_uuid) ON DELETE SET NULL,
  lane text NOT NULL CHECK (lane IN ('fast','slow','audit','external','dead_letter')),
  work_kind text NOT NULL,
  status text NOT NULL DEFAULT 'created' CHECK (status IN ('created','queued','running','succeeded','failed','cancelled','dead_lettered')),
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  idempotency_key text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(idempotency_key)
);

CREATE TABLE IF NOT EXISTS lucidota_control.work_receipt (
  work_receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  decision_uuid uuid REFERENCES lucidota_control.route_decision(decision_uuid) ON DELETE SET NULL,
  work_order_uuid uuid REFERENCES lucidota_control.work_order(work_order_uuid) ON DELETE SET NULL,
  receipt_path text NOT NULL DEFAULT '',
  receipt_sha256 text CHECK (receipt_sha256 IS NULL OR receipt_sha256 ~ '^[0-9a-f]{64}$'),
  verdict text NOT NULL CHECK (verdict IN ('win','loss','stall','poison','retry','promote','kill')),
  cost jsonb NOT NULL DEFAULT '{}'::jsonb,
  gain jsonb NOT NULL DEFAULT '{}'::jsonb,
  artifact_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  graph_write_mode text NOT NULL DEFAULT 'staged_only' CHECK (graph_write_mode IN ('none','staged_only','materialization_gate')),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.board_move (
  move_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  position_uuid uuid REFERENCES lucidota_control.board_position(position_uuid) ON DELETE SET NULL,
  actor text NOT NULL,
  position text NOT NULL,
  move text NOT NULL,
  lane text NOT NULL CHECK (lane IN ('fast','slow','audit','external','dead_letter')),
  cost jsonb NOT NULL DEFAULT '{}'::jsonb,
  gain jsonb NOT NULL DEFAULT '{}'::jsonb,
  receipt text NOT NULL DEFAULT '',
  verdict text NOT NULL CHECK (verdict IN ('win','loss','stall','poison','retry','promote','kill')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.model_invocation (
  model_invocation_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text REFERENCES lucidota_control.event_envelope(event_id) ON DELETE SET NULL,
  target text NOT NULL,
  model_name text NOT NULL,
  prompt_hash text CHECK (prompt_hash IS NULL OR prompt_hash ~ '^[0-9a-f]{64}$'),
  output_hash text CHECK (output_hash IS NULL OR output_hash ~ '^[0-9a-f]{64}$'),
  payload_size_bytes integer NOT NULL DEFAULT 0 CHECK (payload_size_bytes >= 0),
  latency_ms numeric NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
  token_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
  verdict text NOT NULL DEFAULT 'stall' CHECK (verdict IN ('win','loss','stall','poison','retry','promote','kill')),
  receipt_path text NOT NULL DEFAULT '',
  raw_output text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.river_training_row (
  training_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  decision_uuid uuid REFERENCES lucidota_control.route_decision(decision_uuid) ON DELETE SET NULL,
  work_receipt_uuid uuid REFERENCES lucidota_control.work_receipt(work_receipt_uuid) ON DELETE SET NULL,
  route_chosen text NOT NULL,
  model_used text NOT NULL DEFAULT 'none',
  estimated_gain numeric NOT NULL DEFAULT 0,
  actual_gain numeric NOT NULL DEFAULT 0,
  latency_ms numeric NOT NULL DEFAULT 0,
  tokens_in integer NOT NULL DEFAULT 0,
  tokens_out integer NOT NULL DEFAULT 0,
  verdict text NOT NULL,
  human_override boolean NOT NULL DEFAULT false,
  features jsonb NOT NULL DEFAULT '{}'::jsonb,
  label jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.treelite_gate_version (
  gate_version_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  gate_key text NOT NULL,
  version text NOT NULL,
  model_ref text NOT NULL,
  feature_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
  output_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(gate_key, version)
);

CREATE TABLE IF NOT EXISTS lucidota_control.script_manifest (
  script_manifest_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  script_path text NOT NULL UNIQUE,
  caller text NOT NULL DEFAULT '',
  purpose text NOT NULL DEFAULT '',
  receipt_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
  survival_score integer NOT NULL DEFAULT 0 CHECK (survival_score BETWEEN 0 AND 100),
  verdict text NOT NULL DEFAULT 'KEEP' CHECK (verdict IN ('KEEP','MERGE','WRAP','REWRITE','QUARANTINE','CORPSE_MANIFEST','KRAMPUSCHEW')),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.corpse_manifest (
  corpse_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  original_path text NOT NULL,
  corpse_ref text NOT NULL UNIQUE,
  reason text NOT NULL,
  source_sha256 text CHECK (source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$'),
  krampuschewing_status text NOT NULL DEFAULT 'queued',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.dead_letter (
  dead_letter_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_table text NOT NULL,
  source_ref text NOT NULL,
  lane text NOT NULL DEFAULT 'dead_letter',
  reason text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  resolved boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.watch_metric (
  metric_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  metric_key text NOT NULL,
  metric_value jsonb NOT NULL DEFAULT '{}'::jsonb,
  source_receipt text NOT NULL DEFAULT '',
  source_db_ref text NOT NULL DEFAULT '',
  operator_requested boolean NOT NULL DEFAULT false,
  operator_feature_authority_required boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(metric_key, created_at)
);

INSERT INTO lucidota_control.treelite_gate_version(gate_key, version, model_ref, feature_contract, output_contract, active)
VALUES (
  'route_lane',
  'v001',
  'inline_treelite_single_tree:project2501_board_move.py',
  '{"features":["token_count_norm","mutation_requested","needs_cloud_model","needs_graph_write","risk_of_slop"]}'::jsonb,
  '{"lane_enum":["fast","slow","audit","external","dead_letter"],"confidence":"0..1"}'::jsonb,
  true
)
ON CONFLICT(gate_key, version) DO UPDATE SET
  model_ref=EXCLUDED.model_ref,
  feature_contract=EXCLUDED.feature_contract,
  output_contract=EXCLUDED.output_contract,
  active=true;

COMMIT;
