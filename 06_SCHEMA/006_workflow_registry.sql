-- LUCIDOTA workflow registry: small inspectable catalog for DBOS-owned workflows.

CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.workflow_registry (
    workflow_name text PRIMARY KEY,
    owner text NOT NULL DEFAULT 'dbos',
    phase text NOT NULL,
    status text NOT NULL CHECK (status IN ('planned','prototype','active','deprecated')),
    command text NOT NULL DEFAULT '',
    inputs jsonb NOT NULL DEFAULT '{}'::jsonb,
    outputs jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
('dbos-smoke', 'dbos', '005', 'active', 'scripts/lucidota_dbos_smoke.py', '{"text":"string"}', '{"normalized":"string"}', 'Verifies DBOS system database and workflow execution.'),
('survey-protocol', 'dbos+survey', '007', 'active', 'scripts/lucidota_dbos_survey.py', '{"target":"url_or_path","keywords":"list"}', '{"decision":"string","sha256":"string","pivots":"integer"}', 'DBOS wrapper around Survey Protocol.'),
('big-board-event-feed', 'dbos+big-board', '005', 'active', 'scripts/lucidota_dbos_big_board.py', '{"run_id":"string"}', '{"workflow_event":"uuid","overall":"string","scraper_status":"string"}', 'DBOS wrapper that snapshots Big Board state into workflow_event for replayable UI feeds.'),
('river-reflex', 'dbos+river', '006', 'prototype', 'scripts/lucidota_river_reflex.py', '{"workflow_event":"rows"}', '{"river_score":"rows"}', 'Online-ish River scoring from committed workflow events.'),
('cas-index', 'vault', '004', 'active', 'scripts/lucidota_cas_index.py', '{"vault":"path"}', '{"cas_manifest":"rows"}', 'Indexes and verifies local CAS bytes.')
ON CONFLICT (workflow_name) DO UPDATE SET
    owner=EXCLUDED.owner,
    phase=EXCLUDED.phase,
    status=EXCLUDED.status,
    command=EXCLUDED.command,
    inputs=EXCLUDED.inputs,
    outputs=EXCLUDED.outputs,
    notes=EXCLUDED.notes,
    updated_at=now();
