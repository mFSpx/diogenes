-- FILE: 06_SCHEMA/113_project2501_bytewax_board_stream.sql
-- PURPOSE: Bytewax-compatible board stream over EventEnvelope / BoardMove rows.
-- COMPLIANCE: stream hints + watch_metric rows only; no Big Board tile mutation; no graph mutation.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.board_stream_cursor (
  cursor_name text PRIMARY KEY,
  last_event_created_at timestamptz NOT NULL DEFAULT 'epoch'::timestamptz,
  last_event_id text,
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.board_stream_run (
  run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  stream_name text NOT NULL DEFAULT 'project2501_board_stream',
  mode text NOT NULL CHECK (mode IN ('latest_window','live_cursor')),
  status text NOT NULL CHECK (status IN ('succeeded','failed')),
  events_in integer NOT NULL DEFAULT 0 CHECK (events_in >= 0),
  hints_out integer NOT NULL DEFAULT 0 CHECK (hints_out >= 0),
  cursor_lock boolean NOT NULL DEFAULT true,
  bytewax_available boolean NOT NULL DEFAULT false,
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  operator_feature_authority_required boolean NOT NULL DEFAULT true,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.board_stream_hint (
  hint_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id text NOT NULL REFERENCES lucidota_control.event_envelope(event_id) ON DELETE CASCADE,
  source text NOT NULL,
  actor text NOT NULL,
  lane text NOT NULL CHECK (lane IN ('fast','slow','audit','external','dead_letter')),
  route_key text NOT NULL DEFAULT '',
  verdict text NOT NULL DEFAULT 'stall',
  hint_kind text NOT NULL DEFAULT 'board_pressure',
  hint text NOT NULL,
  score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 100),
  canonical_graph_writes_performed boolean NOT NULL DEFAULT false,
  source_receipt text NOT NULL DEFAULT '',
  run_uuid uuid REFERENCES lucidota_control.board_stream_run(run_uuid) ON DELETE SET NULL,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(event_id, hint_kind)
);

CREATE INDEX IF NOT EXISTS idx_board_stream_hint_lane_created
  ON lucidota_control.board_stream_hint(lane, created_at DESC);

-- Trackable metric rows belong in DB/status ledger before the Big Board displays them.
-- This schema intentionally writes lucidota_control.watch_metric backend rows only;
-- dashboard feature additions/removals remain operator-authority changes.

COMMIT;
