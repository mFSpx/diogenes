-- FILE: 06_SCHEMA/077_graph_promotion_packet_dedupe.sql
-- PURPOSE: enforce idempotent graph promotion packets by payload/evidence/authority key.
-- COMPLIANCE: no canonical graph mutation; packet dedupe only.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

ALTER TABLE lucidota_go.graph_promotion_packet
  ADD COLUMN IF NOT EXISTS packet_dedupe_key text;

CREATE OR REPLACE FUNCTION lucidota_go.graph_promotion_packet_dedupe_key(
  p_source_system text,
  p_candidate_kind text,
  p_candidate_payload jsonb,
  p_evidence_refs jsonb,
  p_authority_class text
) RETURNS text
LANGUAGE sql
STABLE
AS $$
  SELECT encode(digest(concat_ws('|',
    coalesce(p_source_system,''),
    coalesce(p_candidate_kind,''),
    coalesce(p_candidate_payload::text,'{}'),
    coalesce((SELECT jsonb_agg(v ORDER BY v)::text FROM jsonb_array_elements_text(coalesce(p_evidence_refs,'[]'::jsonb)) AS t(v)),'[]'),
    coalesce(p_authority_class,'')
  ), 'sha256'), 'hex')
$$;

CREATE OR REPLACE FUNCTION lucidota_go.set_graph_promotion_packet_dedupe_key()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.packet_dedupe_key := lucidota_go.graph_promotion_packet_dedupe_key(
    NEW.source_system, NEW.candidate_kind, NEW.candidate_payload, NEW.evidence_refs, NEW.authority_class
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_set_graph_promotion_packet_dedupe_key ON lucidota_go.graph_promotion_packet;
CREATE TRIGGER trg_set_graph_promotion_packet_dedupe_key
BEFORE INSERT OR UPDATE OF source_system, candidate_kind, candidate_payload, evidence_refs, authority_class
ON lucidota_go.graph_promotion_packet
FOR EACH ROW EXECUTE FUNCTION lucidota_go.set_graph_promotion_packet_dedupe_key();

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='graph_promotion_packet_dedupe_key_unique'
      AND conrelid='lucidota_go.graph_promotion_packet'::regclass
  ) THEN
    ALTER TABLE lucidota_go.graph_promotion_packet
      ADD CONSTRAINT graph_promotion_packet_dedupe_key_unique UNIQUE(packet_dedupe_key);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_graph_promotion_packet_dedupe_key
  ON lucidota_go.graph_promotion_packet(packet_dedupe_key);

COMMIT;
