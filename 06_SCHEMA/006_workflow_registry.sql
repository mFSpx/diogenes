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
('workflow-signoff', 'dbos+governance', '005', 'active', 'scripts/lucidota_dbos_signoff.py', '{"workflow":"string","run_id":"string","decision":"approved|denied"}', '{"governance_gate":"uuid","workflow_event":"uuid"}', 'DBOS-backed workflow signoff lane for operator approval/denial before tests or releases.'),
('workflow-dispatch', 'dbos+governance', '005', 'active', 'scripts/lucidota_dbos_dispatch.py', '{"workflow":"registered","argv":"list","retries":"integer"}', '{"workflow_event":"uuid","result":"json"}', 'DBOS dispatcher that requires signoff, retries registered commands, and emits replayable workflow events.'),
('workflow-replay', 'dbos+governance', '005', 'active', 'scripts/lucidota_dbos_replay.py', '{"workflow":"string","run_id":"string"}', '{"events":"list","terminal_status":"string"}', 'Workflow event replay/inspection lane for signoff and dispatch tests.'),
('survey-protocol', 'dbos+survey', '007', 'active', 'scripts/lucidota_dbos_survey.py', '{"target":"url_or_path","keywords":"list"}', '{"decision":"string","sha256":"string","pivots":"integer"}', 'DBOS wrapper around Survey Protocol.'),
('big-board-event-feed', 'dbos+big-board', '005', 'active', 'scripts/lucidota_dbos_big_board.py', '{"run_id":"string"}', '{"workflow_event":"uuid","overall":"string","scraper_status":"string"}', 'DBOS wrapper that snapshots Big Board state into workflow_event for replayable UI feeds.'),
('body-capture-capture', 'dbos+capture', '008', 'active', 'scripts/lucidota_body_capture.py', '{"source":"url"}', '{"capture_id":"uuid","sha256":"string"}', 'Dispatchable Body Capture workflow; operator/network policy remains enforced by capture code and source gates.'),
('model-governor', 'dbos+runtime', '010', 'active', 'scripts/lucidota_model_governor.py', '{"loadout_id":"optional"}', '{"decision":"allow|defer|reject","action_plan":"json"}', 'Dispatchable model routing/governor workflow for local VRAM/loadout decisions.'),
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
