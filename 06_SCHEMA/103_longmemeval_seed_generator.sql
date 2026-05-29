-- LUCIDOTA Round 83: LongMemEval style seed table.
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;
CREATE TABLE IF NOT EXISTS lucidota_archaeology.longmem_eval_seed (
    seed_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    category text NOT NULL CHECK (category IN ('static_state','dynamic_state','workflow_knowledge','environment_gotcha','premise_awareness','operator_boundary','abductive_handling','policy_mutability')),
    question text NOT NULL,
    expected_evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_kind text NOT NULL,
    source_uuid uuid,
    authority_class text NOT NULL DEFAULT 'model_computed_finding',
    readiness text NOT NULL DEFAULT 'after_timeline_ingestion' CHECK (readiness IN ('after_timeline_ingestion','ready_for_review','rejected')),
    idempotency_key text NOT NULL UNIQUE,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
