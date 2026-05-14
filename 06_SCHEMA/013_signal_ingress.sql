-- Tri-algo signal ingress decisions: passive monitor -> statistical gate -> recovery.
CREATE SCHEMA IF NOT EXISTS lucidota_signal;

CREATE TABLE IF NOT EXISTS lucidota_signal.ingress_decision (
    decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    subsystem text NOT NULL DEFAULT 'body_capture',
    action text NOT NULL CHECK (action IN ('standby', 'burst', 'recover')),
    confidence_gap double precision NOT NULL DEFAULT 0,
    epsilon double precision NOT NULL DEFAULT 0,
    signal_score double precision NOT NULL DEFAULT 0,
    noise_score double precision NOT NULL DEFAULT 0,
    dormancy_probability double precision NOT NULL DEFAULT 0,
    recovery_priority double precision NOT NULL DEFAULT 0,
    reason text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingress_decision_source_idx
  ON lucidota_signal.ingress_decision (source, created_at DESC);
