CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE TABLE IF NOT EXISTS lucidota_control.production_readiness_item (
 readiness_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 readiness_key text NOT NULL UNIQUE,
 subsystem text NOT NULL,
 required_evidence text NOT NULL,
 current_status text NOT NULL DEFAULT 'candidate' CHECK (current_status IN ('candidate','pass','fail','blocked','signed_off')),
 blocker text NOT NULL DEFAULT '',
 evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
 updated_at timestamptz NOT NULL DEFAULT now()
);
