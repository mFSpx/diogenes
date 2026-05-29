-- LUCIDOTA post Mega-Gate target selection framework.
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE TABLE IF NOT EXISTS lucidota_control.post_gate_target (
  target_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_key text NOT NULL UNIQUE,
  subsystem text NOT NULL,
  title text NOT NULL,
  priority integer NOT NULL DEFAULT 100,
  risk_class text NOT NULL CHECK (risk_class IN ('low','medium','high','destructive')),
  requires_operator_gate boolean NOT NULL DEFAULT false,
  target_state text NOT NULL DEFAULT 'candidate' CHECK (target_state IN ('candidate','selected','deferred','blocked','executed','archived')),
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  worker_mapping jsonb NOT NULL DEFAULT '{}'::jsonb,
  blocker text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lucidota_control.post_gate_target_audit (
  audit_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_uuid uuid REFERENCES lucidota_control.post_gate_target(target_uuid),
  target_key text NOT NULL,
  old_state text,
  new_state text NOT NULL,
  actor text NOT NULL DEFAULT 'post_gate_target_selector',
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE OR REPLACE FUNCTION lucidota_control.post_gate_target_transition_guard()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  IF TG_OP='UPDATE' AND OLD.target_state IS DISTINCT FROM NEW.target_state THEN
    IF OLD.target_state='executed' AND NEW.target_state NOT IN ('archived','executed') THEN
      RAISE EXCEPTION 'illegal post_gate target transition: % -> %', OLD.target_state, NEW.target_state USING ERRCODE='check_violation';
    END IF;
    INSERT INTO lucidota_control.post_gate_target_audit(target_uuid,target_key,old_state,new_state,evidence_refs,detail)
    VALUES (OLD.target_uuid, OLD.target_key, OLD.target_state, NEW.target_state, NEW.evidence_refs, jsonb_build_object('trigger','post_gate_target_transition_guard'));
  END IF;
  RETURN NEW;
END; $$;
DROP TRIGGER IF EXISTS trg_post_gate_target_transition_guard ON lucidota_control.post_gate_target;
CREATE TRIGGER trg_post_gate_target_transition_guard BEFORE UPDATE ON lucidota_control.post_gate_target FOR EACH ROW EXECUTE FUNCTION lucidota_control.post_gate_target_transition_guard();
INSERT INTO lucidota_control.post_gate_target(target_key,subsystem,title,priority,risk_class,requires_operator_gate,evidence_refs,worker_mapping)
VALUES
('absurd_queue_spine','ABSURD','ABSURD queue spine implementation branch',10,'medium',false,'["05_OUTPUTS/mega_gate"]','{"scripts":["scripts/spine_queue_soak_test.py"]}'),
('graph_promotion_hardening','Graph','Graph promotion hardening branch',20,'high',true,'["05_OUTPUTS/graph"]','{"scripts":["scripts/graph_promotion_full_e2e.py"]}'),
('brain_archaeology_allowlisted','Phase0.5','Brain Archaeology allowlisted expansion',30,'high',true,'["05_OUTPUTS/security"]','{"scripts":["scripts/phase05_allowlisted_ingest.py"]}'),
('surfaces_cep_compiler','Surfaces','Surfaces/CEP instruction compiler branch',40,'medium',false,'["05_OUTPUTS/surfaces"]','{"scripts":["scripts/surface_instruction_compile_dry_run.py"]}'),
('chrono_audit_expansion','Chrono','Chrono audit expansion branch',50,'low',false,'["05_OUTPUTS/chrono_ledger"]','{"scripts":["scripts/chrono_full_conservation_gate.py"]}'),
('ternary_rd','Ternary','Ternary/FairyFuse R&D branch',90,'medium',true,'["05_OUTPUTS/ternary_lab"]','{"scripts":["ALGOS/ternary_lens_audit.py"]}')
ON CONFLICT(target_key) DO NOTHING;
