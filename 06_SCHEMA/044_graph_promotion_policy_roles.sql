-- FILE: 06_SCHEMA/044_graph_promotion_policy_roles.sql
-- PURPOSE: production policy/role hardening for graph promotion execute path.
-- COMPLIANCE: idempotent; creates policy tables/functions; no destructive grants/revokes.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_role_policy (
  policy_key text PRIMARY KEY,
  role_name text NOT NULL,
  allowed_decisions text[] NOT NULL,
  materialization_allowed boolean NOT NULL DEFAULT false,
  operator_confirmation_required boolean NOT NULL DEFAULT true,
  evidence_required boolean NOT NULL DEFAULT true,
  command_envelope_required boolean NOT NULL DEFAULT false,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO lucidota_go.graph_promotion_role_policy(
  policy_key, role_name, allowed_decisions, materialization_allowed, operator_confirmation_required, evidence_required, command_envelope_required, detail
) VALUES
  ('graph_reader_read_only','graph_reader', ARRAY['defer','reject'], false, false, true, false, '{"canonical_write":false}'::jsonb),
  ('graph_promoter_stage_only','graph_promoter', ARRAY['defer','reject'], false, false, true, false, '{"canonical_write":false,"packet_decision_only":true}'::jsonb),
  ('graph_promoter_materialize_confirmed','graph_promoter', ARRAY['promote','operator_confirmed'], true, true, true, true, '{"canonical_write":"promotion_transaction_only","requires_journal":true}'::jsonb),
  ('operator_confirmed_external_action','operator', ARRAY['operator_confirmed','promote','reject','defer'], true, true, true, false, '{"operator_authority":true}'::jsonb)
ON CONFLICT (policy_key) DO UPDATE SET
  role_name=EXCLUDED.role_name,
  allowed_decisions=EXCLUDED.allowed_decisions,
  materialization_allowed=EXCLUDED.materialization_allowed,
  operator_confirmation_required=EXCLUDED.operator_confirmation_required,
  evidence_required=EXCLUDED.evidence_required,
  command_envelope_required=EXCLUDED.command_envelope_required,
  active=true,
  updated_at=now(),
  detail=EXCLUDED.detail;

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_preflight_audit (
  audit_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  role_name text NOT NULL,
  decision text NOT NULL,
  materialize_requested boolean NOT NULL,
  operator_confirmed boolean NOT NULL,
  evidence_count integer NOT NULL,
  command_envelope_uuid uuid,
  allowed boolean NOT NULL,
  blockers jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE FUNCTION lucidota_go.graph_promotion_preflight(
  p_role_name text,
  p_decision text,
  p_materialize_requested boolean,
  p_operator_confirmed boolean,
  p_evidence_refs jsonb,
  p_command_envelope_uuid uuid DEFAULT NULL
) RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
  pol record;
  blockers jsonb := '[]'::jsonb;
  evidence_count integer := coalesce(jsonb_array_length(coalesce(p_evidence_refs, '[]'::jsonb)), 0);
  allowed boolean := true;
BEGIN
  SELECT * INTO pol FROM lucidota_go.graph_promotion_role_policy
  WHERE role_name=p_role_name AND active AND p_decision = ANY(allowed_decisions)
  ORDER BY materialization_allowed DESC LIMIT 1;
  IF pol IS NULL THEN
    blockers := blockers || jsonb_build_array('role_decision_not_allowed');
  ELSE
    IF p_materialize_requested AND NOT pol.materialization_allowed THEN
      blockers := blockers || jsonb_build_array('materialization_not_allowed_for_role_policy');
    END IF;
    IF pol.operator_confirmation_required AND NOT p_operator_confirmed THEN
      blockers := blockers || jsonb_build_array('operator_confirmation_required');
    END IF;
    IF pol.evidence_required AND evidence_count = 0 THEN
      blockers := blockers || jsonb_build_array('evidence_refs_required');
    END IF;
    IF p_materialize_requested AND pol.command_envelope_required AND p_command_envelope_uuid IS NULL THEN
      blockers := blockers || jsonb_build_array('command_envelope_required_for_materialization');
    END IF;
  END IF;
  allowed := jsonb_array_length(blockers) = 0;
  INSERT INTO lucidota_go.graph_promotion_preflight_audit(
    role_name, decision, materialize_requested, operator_confirmed, evidence_count, command_envelope_uuid, allowed, blockers, detail
  ) VALUES (p_role_name, p_decision, p_materialize_requested, p_operator_confirmed, evidence_count, p_command_envelope_uuid, allowed, blockers, jsonb_build_object('policy_key', coalesce(pol.policy_key, 'none')));
  RETURN jsonb_build_object('allowed', allowed, 'blockers', blockers, 'role_name', p_role_name, 'decision', p_decision, 'materialize_requested', p_materialize_requested);
END;
$$;

COMMIT;
