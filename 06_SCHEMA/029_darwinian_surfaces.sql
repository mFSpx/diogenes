-- FILE: 06_SCHEMA/029_darwinian_surfaces.sql
-- PURPOSE: Darwinian Surfaces lifecycle scaffold.
-- COMPLIANCE: Idempotent, non-destructive. Surface commands are staging-only until CEP fan-in is wired.
-- HARD LAW: No surface interaction mutates canonical state directly; it emits a command envelope.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_runtime;

CREATE TABLE IF NOT EXISTS lucidota_runtime.surface_artifact (
  surface_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  surface_key text NOT NULL UNIQUE,
  surface_status text NOT NULL DEFAULT 'generated' CHECK (surface_status IN ('generated','used','promoted','forked','archived','deprecated')),
  artifact_path text NOT NULL,
  sidecar_path text,
  content_sha256 text CHECK (content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'),
  why_it_exists text NOT NULL,
  reuse_policy text NOT NULL DEFAULT 'declarative_metadata_only',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.surface_pheromone (
  pheromone_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  surface_uuid uuid REFERENCES lucidota_runtime.surface_artifact(surface_uuid) ON DELETE CASCADE,
  surface_key text NOT NULL,
  signal_kind text NOT NULL CHECK (signal_kind IN ('generated','used','promoted','forked','decayed','archived','operator_selected')),
  signal_value numeric NOT NULL DEFAULT 1.0,
  half_life_seconds integer NOT NULL DEFAULT 604800,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.surface_command_envelope (
  envelope_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  surface_key text NOT NULL,
  command_uuid uuid,
  payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  payload jsonb NOT NULL,
  canonical_target text NOT NULL DEFAULT 'lucidota_control.conversation_command',
  staging_only boolean NOT NULL DEFAULT true,
  direct_canonical_mutation boolean NOT NULL DEFAULT false CHECK (direct_canonical_mutation = false),
  fanned_into_conversation_command boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_runtime.surface_lineage (
  lineage_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  parent_surface_key text,
  child_surface_key text NOT NULL,
  lineage_kind text NOT NULL CHECK (lineage_kind IN ('generated_from','promoted_from','forked_from','archived_from','operator_selected')),
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_surface_artifact_key ON lucidota_runtime.surface_artifact(surface_key);
CREATE INDEX IF NOT EXISTS idx_surface_command_key ON lucidota_runtime.surface_command_envelope(surface_key);
CREATE INDEX IF NOT EXISTS idx_surface_pheromone_key ON lucidota_runtime.surface_pheromone(surface_key);

COMMIT;

-- Conversation Instruction Compiler extension.
-- Generated surfaces are backend-registered compiler surfaces; interaction compiles
-- plain-language operator instructions for the conversation/control plane.
-- This is schema-only/idempotent and does not enable direct canonical mutation.
ALTER TABLE lucidota_runtime.surface_artifact
  ADD COLUMN IF NOT EXISTS surface_id text,
  ADD COLUMN IF NOT EXISTS surface_kind text NOT NULL DEFAULT 'generated_instruction_compiler',
  ADD COLUMN IF NOT EXISTS generated_from jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS inputs jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS rendered_artifact_path text,
  ADD COLUMN IF NOT EXISTS interaction_affordances jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS plain_language_instruction_template text NOT NULL DEFAULT 'Tell LUCIDOTA to {action} {target_ref} because {evidence_refs}; allowed effect: {allowed_effect}.',
  ADD COLUMN IF NOT EXISTS command_envelope_schema jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS canonical_mutation_allowed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS conversation_required boolean NOT NULL DEFAULT true;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='chk_surface_no_direct_canonical_mutation'
      AND conrelid='lucidota_runtime.surface_artifact'::regclass
  ) THEN
    ALTER TABLE lucidota_runtime.surface_artifact
      ADD CONSTRAINT chk_surface_no_direct_canonical_mutation CHECK (canonical_mutation_allowed = false);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='chk_surface_conversation_required'
      AND conrelid='lucidota_runtime.surface_artifact'::regclass
  ) THEN
    ALTER TABLE lucidota_runtime.surface_artifact
      ADD CONSTRAINT chk_surface_conversation_required CHECK (conversation_required = true);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS lucidota_runtime.surface_instruction_compile_audit (
  audit_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  surface_id text NOT NULL,
  surface_kind text NOT NULL DEFAULT 'generated_instruction_compiler',
  operator_action text NOT NULL CHECK (operator_action IN ('selected','rejected','refined','compared','inspected','operator_defined')),
  target_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  artifact_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  current_object_state jsonb NOT NULL DEFAULT '{}'::jsonb,
  allowed_effect text NOT NULL,
  plain_language_instruction text NOT NULL,
  command_envelope jsonb NOT NULL,
  canonical_mutation_allowed boolean NOT NULL DEFAULT false CHECK (canonical_mutation_allowed = false),
  conversation_required boolean NOT NULL DEFAULT true CHECK (conversation_required = true),
  staging_only boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);
