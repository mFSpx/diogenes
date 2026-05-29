-- FILE: 06_SCHEMA/088_cruelty_protocols_output.sql
-- PURPOSE: Executable Cruelty Protocols output schema and guardrails.
-- LAW: The module may produce hard findings; external action requires explicit operator authorization.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.cruelty_protocol_output (
  output_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  protocol_name text NOT NULL CHECK (protocol_name IN (
    'power_dynamic_mapping','hmm_state_collapse_prediction','persona_boundary_testing',
    'semantic_isolation_ranking','server_wipe_candidate_detection','api_rate_limit_candidate_detection',
    'environment_migration_candidate_detection','operator_defined'
  )),
  source_ref text NOT NULL,
  target_ref text NOT NULL DEFAULT '',
  finding_text text NOT NULL,
  authority_class lucidota_archaeology.authority_class NOT NULL,
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(evidence_refs)='array' AND jsonb_array_length(evidence_refs)>0),
  recommended_protocol text NOT NULL DEFAULT 'none' CHECK (recommended_protocol IN (
    'none','server_wipe','api_rate_limiting','environment_migration','boundary_assertion','resource_flow_review','peer_to_peer_probe','operator_defined'
  )),
  external_action_authorized boolean NOT NULL DEFAULT false,
  authorized_by_operator_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT cruelty_external_action_requires_authorization CHECK (
    external_action_authorized = false OR authorized_by_operator_at IS NOT NULL
  )
);

CREATE INDEX IF NOT EXISTS idx_cruelty_protocol_output_protocol_created
  ON lucidota_archaeology.cruelty_protocol_output(protocol_name, created_at DESC);
