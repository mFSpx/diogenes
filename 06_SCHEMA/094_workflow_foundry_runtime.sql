-- FILE: 06_SCHEMA/094_workflow_foundry_runtime.sql
-- PURPOSE: Workflow Foundry runtime candidates from design atoms.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_workflow_foundry;

CREATE TABLE IF NOT EXISTS lucidota_workflow_foundry.workflow_invariant_candidate (
  candidate_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  invariant_key text NOT NULL UNIQUE,
  source_atom_uuids uuid[] NOT NULL DEFAULT '{}',
  title text NOT NULL,
  invariant_text text NOT NULL,
  workflow_family text NOT NULL DEFAULT 'unknown',
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  authority_class text NOT NULL DEFAULT 'model_computed_finding' CHECK (authority_class IN (
    'raw_evidence','operator_authored_assertion','operator_defined_label',
    'deterministic_metric','statistical_finding','model_computed_finding',
    'stream_ml_finding','graph_inferred_relation','operator_confirmed_finding',
    'canonical_doctrine','external_action_authorized'
  )),
  review_state text NOT NULL DEFAULT 'candidate' CHECK (review_state IN ('candidate','deferred','rejected','operator_confirmed','promoted')),
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
