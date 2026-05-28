-- FILE: 06_SCHEMA/034_graph_promotion_pipeline.sql
-- PURPOSE: Evidence-gated graph promotion pipeline scaffold.
-- COMPLIANCE: Idempotent, non-destructive. Does not mutate canonical graph data.
-- HARD LAW: No direct graph mutation; promotion requires evidence, authority_class, and decision record.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS lucidota_go;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'lucidota_go' AND t.typname = 'promotion_decision_status'
  ) THEN
    CREATE TYPE lucidota_go.promotion_decision_status AS ENUM (
      'candidate','defer','reject','promote','operator_confirmed','superseded'
    );
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_packet (
  packet_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_system text NOT NULL,
  source_artifact_uuid uuid,
  source_component_uuid uuid,
  candidate_kind text NOT NULL CHECK (candidate_kind IN ('node','edge','property','doctrine','workflow','other')),
  candidate_payload jsonb NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  authority_class text NOT NULL CHECK (authority_class IN (
    'raw_evidence','operator_authored_assertion','operator_defined_label','deterministic_metric',
    'statistical_finding','model_computed_finding','stream_ml_finding','graph_inferred_relation',
    'operator_confirmed_finding','canonical_doctrine','external_action_authorized'
  )),
  promotion_status lucidota_go.promotion_decision_status NOT NULL DEFAULT 'candidate',
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT graph_promotion_packet_requires_evidence CHECK (jsonb_array_length(evidence_refs) > 0)
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_decision (
  decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_uuid uuid NOT NULL REFERENCES lucidota_go.graph_promotion_packet(packet_uuid) ON DELETE RESTRICT,
  decision lucidota_go.promotion_decision_status NOT NULL CHECK (decision IN ('defer','reject','promote','operator_confirmed','superseded')),
  decided_by text NOT NULL CHECK (decided_by IN ('operator','workflow','master_eye','graph_promoter','other')),
  rationale text NOT NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  operator_confirmed boolean NOT NULL DEFAULT false,
  command_envelope_uuid uuid,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT graph_promotion_decision_requires_evidence CHECK (jsonb_array_length(evidence_refs) > 0)
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_journal_requirement (
  requirement_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  requirement_key text NOT NULL UNIQUE,
  requirement_text text NOT NULL,
  required boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_go.graph_promotion_journal_requirement (requirement_key, requirement_text)
VALUES
  ('no_direct_graph_mutation', 'Canonical graph writes must occur only via graph promotion workflow after evidence-gated decision.'),
  ('evidence_refs_required', 'Every promotion packet and decision requires evidence_refs.'),
  ('authority_class_required', 'Every promotion packet requires authority_class.'),
  ('operator_confirmed_path', 'Promote/operator_confirmed actions must link to command envelope or explicit operator confirmation path.'),
  ('journal_append_required', 'Physical graph mutation, when later implemented, must append graph_journal records.')
ON CONFLICT (requirement_key) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_graph_promotion_packet_status ON lucidota_go.graph_promotion_packet(promotion_status);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_packet_authority ON lucidota_go.graph_promotion_packet(authority_class);
CREATE INDEX IF NOT EXISTS idx_graph_promotion_decision_packet ON lucidota_go.graph_promotion_decision(packet_uuid);

COMMIT;
