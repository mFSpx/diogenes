CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE TABLE IF NOT EXISTS lucidota_control.recovery_receipt (
 receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 recovery_key text NOT NULL,
 subsystem text NOT NULL,
 mode text NOT NULL CHECK (mode IN ('dry_run','execute')),
 before_state jsonb NOT NULL DEFAULT '{}'::jsonb,
 after_state jsonb NOT NULL DEFAULT '{}'::jsonb,
 evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
 destructive_action boolean NOT NULL DEFAULT false,
 operator_approved boolean NOT NULL DEFAULT false,
 blockers jsonb NOT NULL DEFAULT '[]'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 CHECK (destructive_action = false OR operator_approved = true)
);
