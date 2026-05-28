-- LUCIDOTA Round 73: Phase 0.5 contradiction ledger worker.
-- Idempotent, additive only. No corpus ingestion. No graph mutation.
CREATE SCHEMA IF NOT EXISTS lucidota_archaeology;

CREATE TABLE IF NOT EXISTS lucidota_archaeology.contradiction_ledger (
    contradiction_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_kind text NOT NULL CHECK (source_kind IN ('design_atom','workflow_blueprint','operator_law','runtime_boundary','other')),
    source_uuid uuid,
    source_ref text NOT NULL,
    boundary_key text NOT NULL,
    contradiction_text text NOT NULL,
    evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
    authority_class text NOT NULL DEFAULT 'operator_doctrine',
    review_state text NOT NULL DEFAULT 'candidate' CHECK (review_state IN ('candidate','confirmed','rejected','superseded')),
    detector text NOT NULL DEFAULT 'scripts/phase05_contradiction_ledger_worker.py',
    idempotency_key text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_contradiction_ledger_source ON lucidota_archaeology.contradiction_ledger(source_kind, source_uuid);
CREATE INDEX IF NOT EXISTS idx_contradiction_ledger_boundary_key ON lucidota_archaeology.contradiction_ledger(boundary_key);
