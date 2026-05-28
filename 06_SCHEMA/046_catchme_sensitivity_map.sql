-- FILE: 06_SCHEMA/046_catchme_sensitivity_map.sql
-- PURPOSE: operator-specific CatchMe consent/sensitivity map enforcement.
-- COMPLIANCE: idempotent; no content ingestion; path/scope policy only.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.catchme_sensitivity_rule (
  rule_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  rule_key text NOT NULL UNIQUE,
  path_glob text NOT NULL,
  sensitivity_class text NOT NULL CHECK (sensitivity_class IN ('public','internal','private','secret','revoked')),
  consent_status text NOT NULL CHECK (consent_status IN ('allowed','requires_operator','revoked')),
  allowed_use text NOT NULL DEFAULT 'none',
  priority integer NOT NULL DEFAULT 100,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_control.catchme_path_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  path_sha256 text NOT NULL CHECK (path_sha256 ~ '^[0-9a-f]{64}$'),
  path_ref text NOT NULL,
  matched_rule_key text,
  sensitivity_class text NOT NULL,
  consent_status text NOT NULL,
  requested_use text NOT NULL,
  operator_approved boolean NOT NULL DEFAULT false,
  allowed boolean NOT NULL,
  reason text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_control.catchme_sensitivity_rule(rule_key,path_glob,sensitivity_class,consent_status,allowed_use,priority,detail)
VALUES
 ('repo_public_docs','00_PROJECT_BRAIN/**','internal','allowed','design_context',10,'{"operator_specific":true}'::jsonb),
 ('toolbox_sandbox','ALGOS/**','internal','allowed','tool_reuse',20,'{"proof_hoard":true}'::jsonb),
 ('security_quarantine','04_RUNTIME/secret_quarantine/**','secret','requires_operator','blocked_without_explicit_command',1,'{"secret_quarantine":true}'::jsonb),
 ('codex_private','**/.codex/**','secret','requires_operator','blocked_without_explicit_command',1,'{"codex_sessions":true}'::jsonb),
 ('revoked_context','**/revoked/**','revoked','revoked','none',0,'{"revoked":true}'::jsonb)
ON CONFLICT(rule_key) DO UPDATE SET
 path_glob=EXCLUDED.path_glob, sensitivity_class=EXCLUDED.sensitivity_class, consent_status=EXCLUDED.consent_status,
 allowed_use=EXCLUDED.allowed_use, priority=EXCLUDED.priority, active=true, updated_at=now(), detail=EXCLUDED.detail;
COMMIT;
