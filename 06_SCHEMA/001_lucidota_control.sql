-- LUCIDOTA control-plane schema.
-- These tables are boring on purpose: DBOS, Clawd, and the future Big Board need
-- stable state surfaces before we build richer UI.

CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.workflow_event (
    event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id text NOT NULL,
    run_id text NOT NULL,
    phase text NOT NULL,
    status text NOT NULL CHECK (status IN (
        'queued',
        'running',
        'blocked',
        'waiting_user',
        'succeeded',
        'failed',
        'cancelled'
    )),
    source text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS workflow_event_run_idx
    ON lucidota_control.workflow_event (workflow_id, run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.governance_gate (
    gate_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id text NOT NULL,
    run_id text NOT NULL,
    action_kind text NOT NULL CHECK (action_kind IN (
        'internal_write',
        'external_read',
        'external_write',
        'email_send',
        'calendar_write',
        'purchase',
        'legal_filing',
        'drive_read'
    )),
    requested_by text NOT NULL,
    target text NOT NULL,
    risk_level text NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    policy_mode text NOT NULL CHECK (policy_mode IN ('user_controlled', 'reactive', 'automatic')),
    approval_status text NOT NULL DEFAULT 'pending' CHECK (approval_status IN (
        'pending',
        'approved',
        'denied',
        'expired',
        'not_required'
    )),
    rationale text NOT NULL,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    decided_at timestamptz
);

CREATE INDEX IF NOT EXISTS governance_gate_pending_idx
    ON lucidota_control.governance_gate (approval_status, risk_level, created_at);

CREATE TABLE IF NOT EXISTS lucidota_control.source_policy (
    source_id text PRIMARY KEY,
    source_kind text NOT NULL CHECK (source_kind IN (
        'web',
        'email',
        'calendar',
        'drive',
        'filesystem',
        'api',
        'manual'
    )),
    allowed_actions text[] NOT NULL DEFAULT ARRAY[]::text[],
    rate_limit_per_minute integer CHECK (rate_limit_per_minute IS NULL OR rate_limit_per_minute > 0),
    requires_user_approval boolean NOT NULL DEFAULT false,
    notes text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_control.model_runtime_inventory (
    inventory_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    captured_at timestamptz NOT NULL DEFAULT now(),
    gpu jsonb NOT NULL DEFAULT '{}'::jsonb,
    cuda_tools jsonb NOT NULL DEFAULT '{}'::jsonb,
    python_imports jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text NOT NULL DEFAULT ''
);

