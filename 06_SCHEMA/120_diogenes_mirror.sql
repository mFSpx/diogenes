-- FILE: 06_SCHEMA/120_diogenes_mirror.sql
-- PURPOSE: project-wide mirror/backup registry for LUCIDOTA wiring and code surfaces.
-- KEEP: excludes bulk data, binary blobs, DB dumps, and log corpora from the mirror contract.

CREATE TABLE IF NOT EXISTS lucidota_control.script_registry (
    script_path text PRIMARY KEY,
    sha256 text NOT NULL DEFAULT '',
    purpose text NOT NULL DEFAULT '',
    inputs jsonb NOT NULL DEFAULT '[]'::jsonb,
    outputs jsonb NOT NULL DEFAULT '[]'::jsonb,
    subsystem text NOT NULL DEFAULT '',
    kind text NOT NULL CHECK (kind IN ('db-native', 'runtime-core', 'experimental', 'quarantine')),
    status text NOT NULL CHECK (status IN ('active', 'prototype', 'deprecated', 'quarantined')),
    tests jsonb NOT NULL DEFAULT '[]'::jsonb,
    receipts jsonb NOT NULL DEFAULT '[]'::jsonb,
    promotion_route text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS script_registry_subsystem_idx
    ON lucidota_control.script_registry (subsystem, kind, status);

CREATE TABLE IF NOT EXISTS lucidota_control.project_mirror_snapshot (
    snapshot_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    root_path text NOT NULL,
    manifest_path text NOT NULL,
    included_count integer NOT NULL DEFAULT 0,
    excluded_count integer NOT NULL DEFAULT 0,
    registry_rows integer NOT NULL DEFAULT 0,
    exclusion_rules jsonb NOT NULL DEFAULT '[]'::jsonb,
    telemetry jsonb NOT NULL DEFAULT '{}'::jsonb,
    note text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
(
    'diogenes-mirror',
    'absurd+mirror',
    '120',
    'active',
    'scripts/diogenes_mirror.py',
    '{"db_url":"string","root":"path","dry_run":"boolean","json_out":"path"}'::jsonb,
    '{"manifest":"json","script_registry_rows":"integer","snapshot_uuid":"uuid"}'::jsonb,
    'Project-wide mirror/backup spine for LUCIDOTA wiring; excludes bulk data, logs, and binary corpora.'
)
ON CONFLICT (workflow_name) DO UPDATE SET
    owner=EXCLUDED.owner,
    phase=EXCLUDED.phase,
    status=EXCLUDED.status,
    command=EXCLUDED.command,
    inputs=EXCLUDED.inputs,
    outputs=EXCLUDED.outputs,
    notes=EXCLUDED.notes,
    updated_at=now();
