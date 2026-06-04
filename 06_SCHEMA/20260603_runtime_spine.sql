-- FILE: 06_SCHEMA/20260603_runtime_spine.sql
-- PURPOSE: canonical runtime spine consolidation for ironclaw/control/ontology.
-- COMPLIANCE: idempotent, non-destructive, canonical-8 only.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS ironclaw;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_ontology;

CREATE TABLE IF NOT EXISTS ironclaw.daemon_registry (
  daemon_name text PRIMARY KEY,
  daemon_role text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','draining','retired')),
  command_path text NOT NULL DEFAULT '',
  config_ref text NOT NULL DEFAULT '',
  transport_socket text NOT NULL DEFAULT '',
  max_heartbeat_age_seconds integer NOT NULL DEFAULT 60 CHECK (max_heartbeat_age_seconds > 0),
  last_heartbeat_at timestamptz,
  last_heartbeat_uuid uuid,
  last_heartbeat_state text NOT NULL DEFAULT '',
  heartbeat_count bigint NOT NULL DEFAULT 0,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ironclaw.daemon_heartbeats (
  heartbeat_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  daemon_name text NOT NULL REFERENCES ironclaw.daemon_registry(daemon_name) ON DELETE RESTRICT,
  heartbeat_kind text NOT NULL DEFAULT 'status',
  host_name text NOT NULL DEFAULT '',
  process_id integer,
  transport_socket text NOT NULL DEFAULT '',
  socket_active boolean NOT NULL DEFAULT false,
  terminal_active boolean NOT NULL DEFAULT false,
  batch_size integer,
  river_state jsonb NOT NULL DEFAULT '{}'::jsonb,
  telemetry jsonb NOT NULL DEFAULT '{}'::jsonb,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS daemon_heartbeats_daemon_idx
  ON ironclaw.daemon_heartbeats(daemon_name, created_at DESC);

CREATE OR REPLACE FUNCTION ironclaw.touch_daemon_registry_from_heartbeat()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE ironclaw.daemon_registry
  SET last_heartbeat_at = NEW.created_at,
      last_heartbeat_uuid = NEW.heartbeat_uuid,
      last_heartbeat_state = CASE
        WHEN NEW.socket_active THEN 'socket_active'
        WHEN NEW.terminal_active THEN 'terminal_active'
        ELSE 'idle'
      END,
      heartbeat_count = heartbeat_count + 1,
      transport_socket = COALESCE(NULLIF(NEW.transport_socket, ''), transport_socket),
      updated_at = now()
  WHERE daemon_name = NEW.daemon_name;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_daemon_registry_from_heartbeat ON ironclaw.daemon_heartbeats;
CREATE TRIGGER trg_touch_daemon_registry_from_heartbeat
AFTER INSERT ON ironclaw.daemon_heartbeats
FOR EACH ROW EXECUTE FUNCTION ironclaw.touch_daemon_registry_from_heartbeat();

CREATE TABLE IF NOT EXISTS lucidota_control.legacy_atomized_evidence (
  evidence_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_uuid uuid,
  source_kind text NOT NULL CHECK (source_kind IN ('json','csv','json_line','csv_row','unknown')),
  source_path text NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  record_index integer NOT NULL DEFAULT 0,
  record_key text NOT NULL DEFAULT '',
  record_value jsonb NOT NULL DEFAULT '{}'::jsonb,
  record_text text NOT NULL DEFAULT '',
  provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS legacy_atomized_evidence_source_idx
  ON lucidota_control.legacy_atomized_evidence(source_path, record_index, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_ontology.canonical_frameworks (
  framework_key text PRIMARY KEY,
  framework_name text NOT NULL,
  framework_kind text NOT NULL CHECK (framework_kind IN ('manual_rpc_cache','policy_gate','route_cache','manifest_anchor')),
  manual_id text NOT NULL DEFAULT '',
  rpc_method text NOT NULL DEFAULT 'GET',
  rpc_route text NOT NULL DEFAULT '',
  cache_route text NOT NULL DEFAULT '',
  source_uri text NOT NULL DEFAULT '',
  active boolean NOT NULL DEFAULT true,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_ontology.canonical_frameworks(
  framework_key,
  framework_name,
  framework_kind,
  manual_id,
  rpc_method,
  rpc_route,
  cache_route,
  source_uri,
  active,
  detail
)
VALUES (
  'postgrest_html_manual_rpc_cache',
  'PostgREST HTML Manual RPC Cache',
  'manual_rpc_cache',
  'root_law_docs',
  'GET',
  '/root_law_docs',
  '/api_bible_nodes?manual_id=eq.{MANUAL_ID}&order=node_sort_key.asc',
  'scripts/compile_canonical_technical_bible.py',
  true,
  jsonb_build_object(
    'purpose', 'Authoritative route cache for manual policy validation.',
    'manual_ids', jsonb_build_array('SYSTEM_ARCH', 'RUNTIME_GOVERNOR', 'AVIONICS', 'FLIGHT_MAN', 'LEDGER')
  )
)
ON CONFLICT (framework_key) DO UPDATE SET
  framework_name = EXCLUDED.framework_name,
  framework_kind = EXCLUDED.framework_kind,
  manual_id = EXCLUDED.manual_id,
  rpc_method = EXCLUDED.rpc_method,
  rpc_route = EXCLUDED.rpc_route,
  cache_route = EXCLUDED.cache_route,
  source_uri = EXCLUDED.source_uri,
  active = true,
  detail = EXCLUDED.detail,
  updated_at = now();

CREATE TABLE IF NOT EXISTS ironclaw.indy_read_judgments (
  judgment_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  daemon_name text NOT NULL DEFAULT 'indy_reads' REFERENCES ironclaw.daemon_registry(daemon_name) ON DELETE RESTRICT,
  book_id text NOT NULL,
  book_name text NOT NULL,
  page_number integer NOT NULL,
  page_hash text NOT NULL,
  parser_version text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('approved','needs_repair','rejected','comment')),
  score integer NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 100),
  score_label text NOT NULL DEFAULT '',
  term_correction text NOT NULL DEFAULT '',
  notes text NOT NULL DEFAULT '',
  repair_instruction text NOT NULL DEFAULT '',
  favorite_line text NOT NULL DEFAULT '',
  confusion text NOT NULL DEFAULT '',
  transport_socket text NOT NULL DEFAULT '/tmp/lucidota_ego.sock',
  socket_active boolean NOT NULL DEFAULT false,
  terminal_active boolean NOT NULL DEFAULT false,
  batch_size integer,
  telemetry jsonb NOT NULL DEFAULT '{}'::jsonb,
  source_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS indy_read_judgments_book_idx
  ON ironclaw.indy_read_judgments(book_id, page_number, created_at DESC);

CREATE OR REPLACE FUNCTION lucidota_control.enforce_subagent_manual_constraint()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  framework record;
  manual_id text := coalesce(NEW.detail->>'manual_id', NEW.command_envelope->>'manual_id', NEW.detail->>'manual', NEW.command_envelope->>'manual', '');
  rpc_route text := coalesce(NEW.detail->>'rpc_route', NEW.command_envelope->>'rpc_route', '');
  cache_route text := coalesce(NEW.detail->>'cache_route', NEW.command_envelope->>'cache_route', '');
  framework_route text;
  framework_cache text;
BEGIN
  IF NEW.command_kind <> 'subagent_manual_constraint' AND manual_id = '' AND rpc_route = '' AND cache_route = '' THEN
    RETURN NEW;
  END IF;

  SELECT * INTO framework
  FROM lucidota_ontology.canonical_frameworks
  WHERE framework_key = 'postgrest_html_manual_rpc_cache'
    AND active = true
  LIMIT 1;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'canonical framework missing: postgrest_html_manual_rpc_cache'
      USING ERRCODE = 'check_violation';
  END IF;

  framework_route := coalesce(nullif(framework.rpc_route, ''), '');
  framework_cache := coalesce(nullif(framework.cache_route, ''), '');

  IF manual_id = '' THEN
    RAISE EXCEPTION 'manual constraint requires manual_id';
  END IF;
  IF manual_id <> framework.manual_id AND manual_id <> framework.framework_name THEN
    RAISE EXCEPTION 'manual constraint manual_id mismatch: % != %', manual_id, framework.manual_id
      USING ERRCODE = 'check_violation';
  END IF;
  IF rpc_route <> '' AND rpc_route <> framework_route THEN
    RAISE EXCEPTION 'manual constraint rpc_route mismatch: % != %', rpc_route, framework_route
      USING ERRCODE = 'check_violation';
  END IF;
  IF cache_route <> '' AND cache_route <> framework_cache THEN
    RAISE EXCEPTION 'manual constraint cache_route mismatch: % != %', cache_route, framework_cache
      USING ERRCODE = 'check_violation';
  END IF;

  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF to_regclass('lucidota_control.conversation_command') IS NOT NULL
     AND has_table_privilege(current_user, 'lucidota_control.conversation_command', 'TRIGGER') THEN
    EXECUTE 'DROP TRIGGER IF EXISTS trg_enforce_subagent_manual_constraint ON lucidota_control.conversation_command';
    EXECUTE '
      CREATE TRIGGER trg_enforce_subagent_manual_constraint
      BEFORE INSERT OR UPDATE ON lucidota_control.conversation_command
      FOR EACH ROW EXECUTE FUNCTION lucidota_control.enforce_subagent_manual_constraint()';
  ELSIF to_regclass('lucidota_control.conversation_command') IS NOT NULL THEN
    RAISE NOTICE 'Skipping trg_enforce_subagent_manual_constraint: no TRIGGER privilege on lucidota_control.conversation_command for %', current_user;
  END IF;
END $$;

INSERT INTO ironclaw.daemon_registry(
  daemon_name,
  daemon_role,
  command_path,
  config_ref,
  transport_socket,
  detail
)
VALUES
  (
    'indy_reads',
    'waking attention client',
    'scripts/indy_reads.py',
    '04_RUNTIME/indy_reads_persona_config.json',
    '/tmp/lucidota_ego.sock',
    jsonb_build_object(
      'mission', 'Page-locked reading companion and terminal dialogue client.',
      'owner', 'INDY_READs'
    )
  ),
  (
    'absurd_queue_spine',
    'transaction-backed queue engine',
    'scripts/absurd_queue_spine.py',
    '06_SCHEMA/20260603_runtime_spine.sql',
    '',
    jsonb_build_object(
      'mission', 'ABSURD durable queue and wake plane.',
      'owner', 'LUCIDOTA'
    )
  )
ON CONFLICT (daemon_name) DO UPDATE SET
  daemon_role = EXCLUDED.daemon_role,
  command_path = EXCLUDED.command_path,
  config_ref = EXCLUDED.config_ref,
  transport_socket = EXCLUDED.transport_socket,
  detail = EXCLUDED.detail,
  updated_at = now();

COMMIT;
