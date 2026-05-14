-- Authorized extractor registry.
-- Purpose-built adapters first; browser capture is policy-gated fallback.

CREATE SCHEMA IF NOT EXISTS lucidota_extract;

CREATE TABLE IF NOT EXISTS lucidota_extract.adapter (
    adapter_id text PRIMARY KEY,
    adapter_kind text NOT NULL CHECK (adapter_kind IN ('http_static', 'schema_api', 'file_parser', 'browser_fallback', 'manual_demo')),
    source_pattern text NOT NULL DEFAULT '',
    stability text NOT NULL DEFAULT 'candidate' CHECK (stability IN ('candidate', 'validated', 'promoted', 'deprecated')),
    default_priority integer NOT NULL DEFAULT 50,
    browser_required boolean NOT NULL DEFAULT false,
    notes text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_extract.adapter_score (
    score_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    adapter_id text NOT NULL REFERENCES lucidota_extract.adapter(adapter_id),
    source text NOT NULL DEFAULT '',
    success boolean NOT NULL DEFAULT false,
    latency_ms integer NOT NULL DEFAULT 0,
    bytes_out bigint NOT NULL DEFAULT 0,
    quality_score integer NOT NULL DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS adapter_score_adapter_idx
    ON lucidota_extract.adapter_score (adapter_id, created_at DESC);

INSERT INTO lucidota_extract.adapter
  (adapter_id, adapter_kind, source_pattern, stability, default_priority, browser_required, notes)
VALUES
  ('http-static-generic', 'http_static', 'http(s)://*', 'validated', 10, false, 'Default low-impact HTTP metadata/body path.'),
  ('html-text-structure-v0', 'http_static', 'text/html', 'validated', 15, false, 'HTML text and structure parser; no browser.'),
  ('browser-visual-fallback-v0', 'browser_fallback', 'layout_sensitive', 'candidate', 90, true, 'Only when visual/DOM render evidence is requested by profile.'),
  ('manual-demo-candidate', 'manual_demo', 'operator-demonstrated', 'candidate', 100, false, 'Candidate path for future teleoperation-derived adapters.')
ON CONFLICT (adapter_id) DO UPDATE SET
  adapter_kind=EXCLUDED.adapter_kind,
  source_pattern=EXCLUDED.source_pattern,
  stability=EXCLUDED.stability,
  default_priority=EXCLUDED.default_priority,
  browser_required=EXCLUDED.browser_required,
  notes=EXCLUDED.notes,
  updated_at=now();
