-- LUCIDOTA Round 78: fungible semantic handle registry.
CREATE SCHEMA IF NOT EXISTS lucidota_semantic;
CREATE TABLE IF NOT EXISTS lucidota_semantic.semantic_handle (
    handle_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    handle_key text NOT NULL UNIQUE,
    display_name text NOT NULL,
    target_ref text NOT NULL,
    target_kind text NOT NULL,
    proof_hoard_role text NOT NULL DEFAULT 'reusable_prior',
    handle_is_truth boolean NOT NULL DEFAULT false,
    namespace_owner text NOT NULL DEFAULT 'operator',
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_manifest text NOT NULL DEFAULT '00_PROJECT_BRAIN/TICKLETRUNK.json',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (handle_is_truth = false)
);
CREATE INDEX IF NOT EXISTS idx_semantic_handle_target ON lucidota_semantic.semantic_handle(target_kind, target_ref);
