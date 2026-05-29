-- FILE: 06_SCHEMA/081_cep_conversation_command_dedupe.sql
-- PURPOSE: enforce content-based idempotency for conversation command envelopes.
-- COMPLIANCE: no direct canonical mutation; command rows remain staging/control-plane instructions.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

ALTER TABLE lucidota_control.conversation_command
  ADD COLUMN IF NOT EXISTS cep_dedupe_key text;

CREATE OR REPLACE FUNCTION lucidota_control.conversation_command_dedupe_key(
  p_command_kind text,
  p_plain_language_instruction text,
  p_command_envelope jsonb,
  p_target_refs jsonb,
  p_evidence_refs jsonb,
  p_allowed_effect text,
  p_authority_class text
) RETURNS text
LANGUAGE sql
STABLE
AS $$
  SELECT encode(digest(concat_ws('|',
    coalesce(p_command_kind,''),
    coalesce(p_plain_language_instruction,''),
    coalesce(p_command_envelope::text,'{}'),
    coalesce((SELECT jsonb_agg(v ORDER BY v)::text FROM jsonb_array_elements_text(coalesce(p_target_refs,'[]'::jsonb)) AS t(v)),'[]'),
    coalesce((SELECT jsonb_agg(v ORDER BY v)::text FROM jsonb_array_elements_text(coalesce(p_evidence_refs,'[]'::jsonb)) AS t(v)),'[]'),
    coalesce(p_allowed_effect,''),
    coalesce(p_authority_class,'')
  ), 'sha256'), 'hex')
$$;

CREATE OR REPLACE FUNCTION lucidota_control.set_conversation_command_dedupe_key()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.cep_dedupe_key := lucidota_control.conversation_command_dedupe_key(
    NEW.command_kind,
    NEW.plain_language_instruction,
    NEW.command_envelope,
    NEW.target_refs,
    NEW.evidence_refs,
    NEW.allowed_effect,
    NEW.authority_class
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_conversation_command_dedupe_key ON lucidota_control.conversation_command;
CREATE TRIGGER trg_set_conversation_command_dedupe_key
BEFORE INSERT OR UPDATE OF command_kind, plain_language_instruction, command_envelope, target_refs, evidence_refs, allowed_effect, authority_class
ON lucidota_control.conversation_command
FOR EACH ROW EXECUTE FUNCTION lucidota_control.set_conversation_command_dedupe_key();

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='conversation_command_cep_dedupe_key_unique'
      AND conrelid='lucidota_control.conversation_command'::regclass
  ) THEN
    ALTER TABLE lucidota_control.conversation_command
      ADD CONSTRAINT conversation_command_cep_dedupe_key_unique UNIQUE(cep_dedupe_key);
  END IF;
END $$;

COMMIT;
