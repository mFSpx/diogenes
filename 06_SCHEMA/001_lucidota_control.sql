-- LUCIDOTA control-plane schema.
-- These tables are boring on purpose: ABSURD, Clawd, and the future Big Board need
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


CREATE TABLE IF NOT EXISTS lucidota_control.event_outbox (
    outbox_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id uuid REFERENCES lucidota_control.workflow_event(event_id) ON DELETE CASCADE,
    topic text NOT NULL,
    ref_body jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'delivered', 'failed')),
    attempts integer NOT NULL DEFAULT 0,
    last_error text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    delivered_at timestamptz
);

CREATE INDEX IF NOT EXISTS event_outbox_status_idx
    ON lucidota_control.event_outbox (status, created_at);

CREATE TABLE IF NOT EXISTS lucidota_control.workflow_schedule (
    schedule_id text PRIMARY KEY,
    workflow_name text NOT NULL,
    cadence_seconds integer NOT NULL CHECK (cadence_seconds > 0),
    enabled boolean NOT NULL DEFAULT true,
    next_run_at timestamptz NOT NULL DEFAULT now(),
    last_run_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS workflow_schedule_due_idx
    ON lucidota_control.workflow_schedule (enabled, next_run_at);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='lucidota_control' AND table_name='event_outbox' AND column_name='payload'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='lucidota_control' AND table_name='event_outbox' AND column_name='ref_body'
  ) THEN
    ALTER TABLE lucidota_control.event_outbox RENAME COLUMN payload TO ref_body;
  END IF;
END $$;


DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='lucidota_control' AND table_name='event_outbox' AND column_name='published_at'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='lucidota_control' AND table_name='event_outbox' AND column_name='delivered_at'
  ) THEN
    ALTER TABLE lucidota_control.event_outbox RENAME COLUMN published_at TO delivered_at;
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.constraint_column_usage
    WHERE table_schema='lucidota_control' AND table_name='event_outbox' AND constraint_name='event_outbox_status_check'
  ) THEN
    ALTER TABLE lucidota_control.event_outbox DROP CONSTRAINT event_outbox_status_check;
  END IF;

  UPDATE lucidota_control.event_outbox SET status='delivered' WHERE status='published';
  ALTER TABLE lucidota_control.event_outbox
    ADD CONSTRAINT event_outbox_status_check CHECK (status IN ('pending', 'delivered', 'failed'));
END $$;

CREATE OR REPLACE FUNCTION lucidota_control.enqueue_workflow_event_outbox()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  INSERT INTO lucidota_control.event_outbox (event_id, topic, ref_body)
  VALUES (
    NEW.event_id,
    'workflow_event',
    jsonb_build_object(
      'event_id', NEW.event_id::text,
      'workflow_id', NEW.workflow_id,
      'run_id', NEW.run_id,
      'phase', NEW.phase,
      'status', NEW.status,
      'source', NEW.source
    )
  )
  ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS workflow_event_outbox_trigger ON lucidota_control.workflow_event;
CREATE TRIGGER workflow_event_outbox_trigger
AFTER INSERT ON lucidota_control.workflow_event
FOR EACH ROW EXECUTE FUNCTION lucidota_control.enqueue_workflow_event_outbox();
