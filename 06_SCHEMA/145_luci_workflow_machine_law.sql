-- FILE: 06_SCHEMA/145_luci_workflow_machine_law.sql
-- PURPOSE: encode LUCI deterministic-first workflow law on the existing ABSURD registry.
-- LAW: extend the existing registry; do not create a parallel workflow architecture.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE SCHEMA IF NOT EXISTS lucidota_canon;

ALTER TABLE lucidota_control.workflow_registry
    ADD COLUMN IF NOT EXISTS workflow_id text,
    ADD COLUMN IF NOT EXISTS verb text,
    ADD COLUMN IF NOT EXISTS input_object_types text[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS output_object_types text[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS deterministic_first boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS llm_allowed boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS llm_required boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS allowed_models text[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS validator_workflow_id text,
    ADD COLUMN IF NOT EXISTS receipt_type text NOT NULL DEFAULT 'workflow_receipt',
    ADD COLUMN IF NOT EXISTS promotion_policy text NOT NULL DEFAULT 'validated_receipt_required',
    ADD COLUMN IF NOT EXISTS llm_allowed_reasons text[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS ontology_tags text[] NOT NULL DEFAULT '{}'::text[];

CREATE OR REPLACE FUNCTION lucidota_control.fn_workflow_registry_fill_policy()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.workflow_id := COALESCE(NULLIF(NEW.workflow_id, ''), NEW.workflow_name);
    NEW.verb := COALESCE(NULLIF(NEW.verb, ''), regexp_replace(NEW.workflow_name, '[-]+', '_', 'g'));
    NEW.input_object_types := COALESCE(NEW.input_object_types, '{}'::text[]);
    NEW.output_object_types := COALESCE(NEW.output_object_types, '{}'::text[]);
    NEW.deterministic_first := COALESCE(NEW.deterministic_first, true);
    NEW.llm_allowed := COALESCE(NEW.llm_allowed, false);
    NEW.llm_required := COALESCE(NEW.llm_required, false);
    NEW.allowed_models := COALESCE(NEW.allowed_models, '{}'::text[]);
    NEW.receipt_type := COALESCE(NULLIF(NEW.receipt_type, ''), 'workflow_receipt');
    NEW.promotion_policy := COALESCE(NULLIF(NEW.promotion_policy, ''), 'validated_receipt_required');
    NEW.llm_allowed_reasons := COALESCE(NEW.llm_allowed_reasons, '{}'::text[]);
    NEW.ontology_tags := COALESCE(NEW.ontology_tags, '{}'::text[]);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS tr_workflow_registry_fill_policy ON lucidota_control.workflow_registry;
CREATE TRIGGER tr_workflow_registry_fill_policy
    BEFORE INSERT OR UPDATE ON lucidota_control.workflow_registry
    FOR EACH ROW EXECUTE FUNCTION lucidota_control.fn_workflow_registry_fill_policy();

UPDATE lucidota_control.workflow_registry
SET
    workflow_id = COALESCE(NULLIF(workflow_id, ''), workflow_name),
    verb = COALESCE(NULLIF(verb, ''), regexp_replace(workflow_name, '[-]+', '_', 'g')),
    input_object_types = COALESCE(input_object_types, '{}'::text[]),
    output_object_types = COALESCE(output_object_types, '{}'::text[]),
    deterministic_first = COALESCE(deterministic_first, true),
    llm_allowed = COALESCE(llm_allowed, false),
    llm_required = COALESCE(llm_required, false),
    allowed_models = COALESCE(allowed_models, '{}'::text[]),
    receipt_type = COALESCE(NULLIF(receipt_type, ''), 'workflow_receipt'),
    promotion_policy = COALESCE(NULLIF(promotion_policy, ''), 'validated_receipt_required'),
    llm_allowed_reasons = COALESCE(llm_allowed_reasons, '{}'::text[]),
    ontology_tags = COALESCE(ontology_tags, '{}'::text[]);

ALTER TABLE lucidota_control.workflow_registry
    ALTER COLUMN workflow_id SET NOT NULL,
    ALTER COLUMN verb SET NOT NULL;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_workflow_id_unique UNIQUE (workflow_id);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_llm_required_requires_allowed
        CHECK (NOT llm_required OR llm_allowed);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_no_llm_means_deterministic_first
        CHECK (llm_allowed OR deterministic_first);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_llm_reason_check
        CHECK (
            llm_allowed_reasons <@ ARRAY[
                'ambiguous_human_language',
                'messy_summarization',
                'entity_claim_extraction_judgment',
                'conflict_explanation',
                'hypothesis_generation',
                'prompt_dialogue_response',
                'code_design_review',
                'natural_language_transformation',
                'low_confidence_router_fallback',
                'human_facing_synthesis'
            ]::text[]
        );
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_llm_reason_required_when_allowed
        CHECK (NOT llm_allowed OR cardinality(llm_allowed_reasons) > 0);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

DO $$
BEGIN
    ALTER TABLE lucidota_control.workflow_registry
        ADD CONSTRAINT workflow_registry_promotion_policy_check
        CHECK (promotion_policy IN (
            'deterministic_receipt',
            'validated_receipt_required',
            'proposal_only',
            'proposal_until_validator_receipt',
            'human_review_required',
            'never_promote_directly'
        ));
EXCEPTION WHEN duplicate_object THEN
    NULL;
END;
$$;

CREATE INDEX IF NOT EXISTS workflow_registry_policy_idx
    ON lucidota_control.workflow_registry(deterministic_first, llm_allowed, llm_required);

CREATE INDEX IF NOT EXISTS workflow_registry_ontology_tags_idx
    ON lucidota_control.workflow_registry USING gin(ontology_tags);

CREATE OR REPLACE VIEW lucidota_canon.api_workflow_registry AS
SELECT
    workflow_id,
    workflow_name,
    verb,
    owner,
    phase,
    status,
    command,
    inputs,
    outputs,
    input_object_types,
    output_object_types,
    deterministic_first,
    llm_allowed,
    llm_required,
    allowed_models,
    validator_workflow_id,
    receipt_type,
    promotion_policy,
    llm_allowed_reasons,
    ontology_tags,
    notes,
    updated_at
FROM lucidota_control.workflow_registry;

INSERT INTO lucidota_control.workflow_registry
(
    workflow_name, workflow_id, owner, phase, status, command, inputs, outputs, notes,
    verb, input_object_types, output_object_types, deterministic_first, llm_allowed, llm_required,
    allowed_models, validator_workflow_id, receipt_type, promotion_policy, llm_allowed_reasons, ontology_tags
)
VALUES
(
    'root-rotor-apply-node-payloads',
    'root-rotor-apply-node-payloads',
    'root-rotor+canon',
    '145',
    'active',
    'scripts/root_rotor_apply_node_payloads.py --execute',
    '{"payload":"lucidota.root_rotor.bible_node_payload.v1"}'::jsonb,
    '{"bible_node":"versioned_row","receipt":"json"}'::jsonb,
    'Deterministic validator and DB promotion step for model-proposed manual nodes.',
    'apply_node_payloads',
    ARRAY['bible_node_payload','model_output_file'],
    ARRAY['bible_node','workflow_receipt'],
    true,
    false,
    false,
    '{}'::text[],
    NULL,
    'workflow_receipt',
    'deterministic_receipt',
    '{}'::text[],
    ARRAY['WORKFLOW','RECEIPT','STATE']
),
(
    'root-rotor-red-team-audit',
    'root-rotor-red-team-audit',
    'root-rotor+canon',
    '145',
    'active',
    'scripts/root_rotor_red_team_audit.py --json',
    '{"bible_nodes":"rows","bible_dependencies":"rows","postgrest":"endpoint"}'::jsonb,
    '{"audit_verdict":"json","receipt":"json"}'::jsonb,
    'Deterministic adversarial audit for draft nodes, broken parents, cycles, and API availability.',
    'red_team_audit',
    ARRAY['bible_node','bible_dependency','api_endpoint'],
    ARRAY['audit_receipt','review_required_flag'],
    true,
    false,
    false,
    '{}'::text[],
    NULL,
    'audit_receipt',
    'deterministic_receipt',
    '{}'::text[],
    ARRAY['WORKFLOW','RECEIPT','EDGE','STATE']
),
(
    'root-rotor-canon-forge',
    'root-rotor-canon-forge',
    'root-rotor+canon',
    '145',
    'active',
    'scripts/root_rotor_manual_queue.py -> scripts/vibe_sequencer.py -> scripts/root_rotor_apply_node_payloads.py',
    '{"audit_manifest":"json","source_file":"bounded_text"}'::jsonb,
    '{"bible_node_payload":"json","model_invocation_receipt":"json"}'::jsonb,
    'Deterministic queue first; boxed model call only for per-file code/design review and technical manual transformation. Model output is a proposal until validator receipt.',
    'forge_canon_nodes',
    ARRAY['source_file','audit_manifest_entry'],
    ARRAY['bible_node_payload','model_invocation_receipt'],
    true,
    true,
    true,
    ARRAY['vibes:codestral','groq','gpt-5.3-codex-spark','gpt-5.4-mini','gpt-5.5'],
    'root-rotor-apply-node-payloads',
    'model_invocation_receipt',
    'proposal_until_validator_receipt',
    ARRAY['code_design_review','natural_language_transformation','human_facing_synthesis'],
    ARRAY['WORKFLOW','RECEIPT','CLAIM','STATE']
)
ON CONFLICT (workflow_name) DO UPDATE SET
    workflow_id=EXCLUDED.workflow_id,
    owner=EXCLUDED.owner,
    phase=EXCLUDED.phase,
    status=EXCLUDED.status,
    command=EXCLUDED.command,
    inputs=EXCLUDED.inputs,
    outputs=EXCLUDED.outputs,
    notes=EXCLUDED.notes,
    verb=EXCLUDED.verb,
    input_object_types=EXCLUDED.input_object_types,
    output_object_types=EXCLUDED.output_object_types,
    deterministic_first=EXCLUDED.deterministic_first,
    llm_allowed=EXCLUDED.llm_allowed,
    llm_required=EXCLUDED.llm_required,
    allowed_models=EXCLUDED.allowed_models,
    validator_workflow_id=EXCLUDED.validator_workflow_id,
    receipt_type=EXCLUDED.receipt_type,
    promotion_policy=EXCLUDED.promotion_policy,
    llm_allowed_reasons=EXCLUDED.llm_allowed_reasons,
    ontology_tags=EXCLUDED.ontology_tags,
    updated_at=now();
