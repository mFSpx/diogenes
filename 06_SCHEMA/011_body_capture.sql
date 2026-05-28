-- Body Capture capture/evidence diff schema.
-- Bytes live in local CAS; rows carry metadata and refs only.

CREATE SCHEMA IF NOT EXISTS lucidota_body_capture;

CREATE TABLE IF NOT EXISTS lucidota_body_capture.capture (
    capture_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    capture_kind text NOT NULL CHECK (capture_kind IN ('http_body', 'dom_snapshot', 'screenshot_placeholder')),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    sha256 text,
    cas_uri text,
    size_bytes bigint NOT NULL DEFAULT 0,
    mime text NOT NULL DEFAULT '',
    title text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS body_capture_capture_source_idx
    ON lucidota_body_capture.capture (source, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_body_capture.delta (
    delta_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    old_capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    new_capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    old_sha256 text NOT NULL DEFAULT '',
    new_sha256 text NOT NULL DEFAULT '',
    changed boolean NOT NULL DEFAULT false,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS body_capture_delta_source_idx
    ON lucidota_body_capture.delta (source, created_at DESC);

ALTER TABLE lucidota_body_capture.capture
  ADD COLUMN IF NOT EXISTS content_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS structure_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS visual_hash text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS lucidota_body_capture.watcher_profile (
    profile_id text PRIMARY KEY,
    description text NOT NULL DEFAULT '',
    alert_on_content boolean NOT NULL DEFAULT true,
    alert_on_structure boolean NOT NULL DEFAULT false,
    alert_on_visual boolean NOT NULL DEFAULT false,
    visual_mode text NOT NULL DEFAULT 'record_only' CHECK (visual_mode IN ('ignore', 'record_only', 'alert', 'escalate')),
    min_severity text NOT NULL DEFAULT 'record_only' CHECK (min_severity IN ('ignore', 'record_only', 'alert', 'escalate')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_body_capture.watcher_assignment (
    source text PRIMARY KEY,
    profile_id text NOT NULL REFERENCES lucidota_body_capture.watcher_profile(profile_id),
    enabled boolean NOT NULL DEFAULT true,
    notes text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_body_capture.watcher_decision (
    decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    profile_id text NOT NULL,
    old_capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    new_capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    content_changed boolean NOT NULL DEFAULT false,
    structure_changed boolean NOT NULL DEFAULT false,
    visual_changed boolean NOT NULL DEFAULT false,
    outcome text NOT NULL CHECK (outcome IN ('ignore', 'record_only', 'alert', 'escalate')),
    rationale text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS watcher_decision_source_idx
    ON lucidota_body_capture.watcher_decision (source, created_at DESC);

INSERT INTO lucidota_body_capture.watcher_profile
  (profile_id, description, alert_on_content, alert_on_structure, alert_on_visual, visual_mode, min_severity, detail)
VALUES
  ('metadata_only', 'Status/header-oriented watcher; capture deltas record only.', false, false, false, 'ignore', 'record_only', '{}'::jsonb),
  ('content_truth', 'Text/claim/link changes matter; visual deltas are evidence only.', true, false, false, 'record_only', 'alert', '{}'::jsonb),
  ('layout_sensitive', 'Content, structure, and visual/layout changes can matter.', true, true, true, 'alert', 'alert', '{}'::jsonb),
  ('evidence_grade', 'Record all signal classes and alert on content or structural change.', true, true, false, 'record_only', 'alert', '{}'::jsonb),
  ('low_noise_monitor', 'Suppress visual-only noise; alert on content changes.', true, false, false, 'ignore', 'alert', '{}'::jsonb)
ON CONFLICT (profile_id) DO UPDATE SET
  description=EXCLUDED.description,
  alert_on_content=EXCLUDED.alert_on_content,
  alert_on_structure=EXCLUDED.alert_on_structure,
  alert_on_visual=EXCLUDED.alert_on_visual,
  visual_mode=EXCLUDED.visual_mode,
  min_severity=EXCLUDED.min_severity,
  detail=EXCLUDED.detail,
  updated_at=now();

CREATE TABLE IF NOT EXISTS lucidota_body_capture.evidence_bundle (
    bundle_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    old_capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    new_capture_id uuid NOT NULL REFERENCES lucidota_body_capture.capture(capture_id),
    bundle_sha256 text NOT NULL,
    bundle_path text NOT NULL,
    summary text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS body_capture_evidence_bundle_source_idx
    ON lucidota_body_capture.evidence_bundle (source, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_body_capture.workflow_event_outbox (
    outbox_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    capture_id uuid REFERENCES lucidota_body_capture.capture(capture_id),
    workflow_id text NOT NULL DEFAULT 'body_capture-capture',
    run_id text NOT NULL,
    phase text NOT NULL DEFAULT 'capture',
    status text NOT NULL DEFAULT 'succeeded',
    source text NOT NULL DEFAULT 'lucidota_body_capture',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    delivery_status text NOT NULL DEFAULT 'pending'
      CHECK (delivery_status IN ('pending', 'delivered', 'failed')),
    attempts integer NOT NULL DEFAULT 0,
    last_error text NOT NULL DEFAULT '',
    delivered_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS body_capture_workflow_event_outbox_status_idx
    ON lucidota_body_capture.workflow_event_outbox (delivery_status, created_at);
