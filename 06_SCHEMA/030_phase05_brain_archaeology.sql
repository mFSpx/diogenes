-- FILE: 06_SCHEMA/030_phase05_brain_archaeology.sql
-- PURPOSE: Phase 0.5 Brain Archaeology scaffold only; no corpus ingest.
-- COMPLIANCE: Idempotent, non-destructive schema scaffold.
-- HARD LAW: Preserve Operator-authored ontology exactly; no label softening, moralizing, or renaming.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'lucidota_archaeology'
      AND t.typname = 'authority_class'
  ) THEN
    CREATE TYPE lucidota_archaeology.authority_class AS ENUM (
      'raw_evidence',
      'operator_authored_assertion',
      'operator_defined_label',
      'deterministic_metric',
      'statistical_finding',
      'model_computed_finding',
      'stream_ml_finding',
      'graph_inferred_relation',
      'operator_confirmed_finding',
      'canonical_doctrine',
      'external_action_authorized'
    );
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.operator_syllabus_seed (
  seed_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seed_kind text NOT NULL CHECK (seed_kind IN (
    'original_architecture_prompt','assistant_phase0_review','phase05_directive',
    'phase05_reference_manual','correction_directive','operator_doctrine','other'
  )),
  title text NOT NULL,
  body text NOT NULL,
  body_sha256 text NOT NULL CHECK (body_sha256 ~ '^[0-9a-f]{64}$'),
  authored_by text NOT NULL CHECK (authored_by IN ('operator','assistant','system','unknown')),
  authority_class lucidota_archaeology.authority_class NOT NULL DEFAULT 'operator_authored_assertion',
  asserted_date date,
  captured_at timestamptz NOT NULL DEFAULT now(),
  immutable boolean NOT NULL DEFAULT true,
  supersedes_seed_uuid uuid REFERENCES lucidota_archaeology.operator_syllabus_seed(seed_uuid),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.classifier_label (
  label_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  label_key text NOT NULL UNIQUE,
  label_family text NOT NULL CHECK (label_family IN (
    'operator_state','persona_masthead','development_epoch','workflow_mode',
    'topology_role','risk_pattern','doctrine_cluster','protocol','system_module','other'
  )),
  display_name text NOT NULL,
  definition text NOT NULL,
  source_seed_uuid uuid REFERENCES lucidota_archaeology.operator_syllabus_seed(seed_uuid),
  authority_class lucidota_archaeology.authority_class NOT NULL DEFAULT 'operator_defined_label',
  active boolean NOT NULL DEFAULT true,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.sticker_feature_vector (
  vector_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  artifact_uuid uuid NOT NULL,
  component_uuid uuid,
  chrono_uuid uuid,
  feature_version text NOT NULL DEFAULT 'phase05_stickers_v1',
  syntactic_chaos numeric,
  ellipsis_density numeric,
  punctuation_velocity numeric,
  first_person_gravity numeric,
  lexical_intent_ratio numeric,
  structural_entropy numeric,
  ledger_density numeric,
  visceral_ratio numeric,
  recursion_score numeric,
  directive_ratio numeric,
  target_density numeric,
  forensic_shield_ratio numeric,
  dissociative_index numeric,
  poetic_entropy numeric,
  wrath_velocity numeric,
  bureaucratic_weaponization_index numeric,
  resource_exhaustion_metric numeric,
  swarm_orchestration_density numeric,
  conspiracy_grounding_ratio numeric,
  chaotic_good_tax numeric,
  corporate_grit_tension numeric,
  countdown_density numeric,
  asset_structuring_weight numeric,
  pitch_to_prose_ratio numeric,
  agent_symmetry_ratio numeric,
  subtle_knife_discipline numeric,
  manic_velocity numeric,
  raw_features jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.telemetry_finding (
  finding_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  artifact_uuid uuid NOT NULL,
  component_uuid uuid,
  chrono_uuid uuid,
  vector_uuid uuid REFERENCES lucidota_archaeology.sticker_feature_vector(vector_uuid),
  finding_kind text NOT NULL CHECK (finding_kind IN (
    'eureka_anomaly','concept_drift','state_cluster','classifier_label','state_transition',
    'persona_boundary','semantic_isolation','epoch_marker','operator_diagnostic','other'
  )),
  label_key text REFERENCES lucidota_archaeology.classifier_label(label_key),
  finding_text text NOT NULL,
  method text NOT NULL CHECK (method IN (
    'half_space_trees','adwin','hoeffding_tree','dbstream','hmm','svm_stylometry',
    'mahalanobis_distance','bertopic','tfidf_epoch_diff','pgvector_umap',
    'llm_structured_extraction','operator_assertion','mixed'
  )),
  authority_class lucidota_archaeology.authority_class NOT NULL,
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
  parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.topology_finding (
  topology_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_node_alias text NOT NULL,
  target_node_alias text NOT NULL,
  topology_model text NOT NULL CHECK (topology_model IN (
    'client_server','peer_to_peer','infinite_sink','high_friction_symbiosis',
    'transactional_isolation','anchor_weight','legacy_client','apex_peer_candidate','unknown'
  )),
  finding_text text NOT NULL,
  measured_dimensions jsonb NOT NULL DEFAULT '{}'::jsonb,
  evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
  method text NOT NULL CHECK (method IN (
    'linguistic_style_matching','resource_flow_analysis','message_graph_analysis',
    'operator_assertion','llm_structured_extraction','mixed'
  )),
  authority_class lucidota_archaeology.authority_class NOT NULL,
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  recommended_protocol text CHECK (recommended_protocol IN (
    'none','server_wipe','api_rate_limiting','environment_migration',
    'boundary_assertion','resource_flow_review','peer_to_peer_probe','operator_defined'
  )),
  external_action_authorized boolean NOT NULL DEFAULT false,
  authorized_by_operator_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.design_atom (
  atom_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_artifact_uuid uuid NOT NULL,
  source_component_uuid uuid,
  chrono_uuid uuid,
  atom_kind text NOT NULL CHECK (atom_kind IN (
    'requirement','workflow','algorithm','schema','component','constraint','risk','doctrine',
    'prompt_pattern','operator_process','governance_rule','label_definition','topology_protocol',
    'open_question','contradiction','discarded_idea'
  )),
  title text NOT NULL,
  normalized_claim text NOT NULL,
  raw_excerpt text NOT NULL,
  authority_class lucidota_archaeology.authority_class NOT NULL,
  status text NOT NULL DEFAULT 'candidate' CHECK (status IN (
    'candidate','duplicate','merged','promoted','rejected','needs_operator_authority','superseded'
  )),
  confidence_bps integer NOT NULL DEFAULT 5000 CHECK (confidence_bps BETWEEN 0 AND 10000),
  tags text[] NOT NULL DEFAULT '{}'::text[],
  entities jsonb NOT NULL DEFAULT '[]'::jsonb,
  evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
  extractor_version text NOT NULL,
  model_invocation_uuid uuid,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.workflow_blueprint (
  blueprint_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_key text NOT NULL UNIQUE,
  title text NOT NULL,
  purpose text NOT NULL,
  maturity text NOT NULL DEFAULT 'recovered' CHECK (maturity IN (
    'raw','recovered','drafted','implementable','implemented','deprecated'
  )),
  absurd_target boolean NOT NULL DEFAULT true,
  input_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
  output_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
  steps jsonb NOT NULL DEFAULT '[]'::jsonb,
  queues text[] NOT NULL DEFAULT '{}'::text[],
  required_tables text[] NOT NULL DEFAULT '{}'::text[],
  required_services text[] NOT NULL DEFAULT '{}'::text[],
  required_models text[] NOT NULL DEFAULT '{}'::text[],
  source_atom_uuids uuid[] NOT NULL DEFAULT '{}'::uuid[],
  authority_class lucidota_archaeology.authority_class NOT NULL DEFAULT 'model_computed_finding',
  canonical_confidence_bps integer NOT NULL DEFAULT 5000 CHECK (canonical_confidence_bps BETWEEN 0 AND 10000),
  operator_confirmed boolean NOT NULL DEFAULT false,
  operator_confirmed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.syllabus_fidelity_violation (
  violation_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_artifact_uuid uuid,
  source_component_uuid uuid,
  violation_kind text NOT NULL CHECK (violation_kind IN (
    'label_softening','unauthorized_reframing','assistant_moralizing','ontology_renaming',
    'state_label_demoted','protocol_removed','protective_wrapper_inserted','other'
  )),
  original_term text NOT NULL,
  altered_term text,
  violation_text text NOT NULL,
  detected_by text NOT NULL CHECK (detected_by IN ('operator','deterministic_rule','llm_critic','master_eye','other')),
  severity text NOT NULL CHECK (severity IN ('low','medium','high','critical')),
  resolution text NOT NULL DEFAULT 'pending',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_archaeology.master_eye_review (
  review_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_kind text NOT NULL CHECK (target_kind IN (
    'syllabus_seed','design_atom','workflow_blueprint','telemetry_finding',
    'topology_finding','doctrine','contradiction','system_risk','fidelity_violation'
  )),
  target_uuid uuid NOT NULL,
  judgment text NOT NULL CHECK (judgment IN (
    'syllabus_faithful','implementation_ready','needs_more_evidence','ontology_drift',
    'assistant_rewrite_detected','schema_gap','workflow_gap','contradiction_detected',
    'promote','defer','reject'
  )),
  rationale text NOT NULL,
  recommended_next_action text NOT NULL,
  confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
  reviewed_by text NOT NULL DEFAULT 'master_eye_model',
  model_invocation_uuid uuid,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_phase05_classifier_label_key ON lucidota_archaeology.classifier_label(label_key);
CREATE INDEX IF NOT EXISTS idx_phase05_design_atom_status ON lucidota_archaeology.design_atom(status);
CREATE INDEX IF NOT EXISTS idx_phase05_workflow_blueprint_key ON lucidota_archaeology.workflow_blueprint(workflow_key);
CREATE INDEX IF NOT EXISTS idx_phase05_fidelity_violation_kind ON lucidota_archaeology.syllabus_fidelity_violation(violation_kind);

COMMIT;
