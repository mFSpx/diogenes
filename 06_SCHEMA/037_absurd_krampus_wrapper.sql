-- FILE: 06_SCHEMA/037_absurd_krampus_wrapper.sql
-- PURPOSE: Register KRAMPUSCHEWING/KORPUS health observation under ABSURD queue spine.
-- COMPLIANCE: Idempotent, non-destructive. Does not ingest files or mutate KORPUS custody rows.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, status, max_attempts, visibility_timeout_seconds, notes)
VALUES (
  'korpus',
  'KRAMPUSCHEWING/KORPUS',
  'active',
  3,
  300,
  'ABSURD wrapper queue for KRAMPUSCHEWING/KORPUS health observation jobs. Wrapper must not ingest, delete, move, or mutate custody rows.'
)
ON CONFLICT (queue_name) DO UPDATE SET
  owner_subsystem=EXCLUDED.owner_subsystem,
  status=EXCLUDED.status,
  max_attempts=EXCLUDED.max_attempts,
  visibility_timeout_seconds=EXCLUDED.visibility_timeout_seconds,
  notes=EXCLUDED.notes,
  updated_at=now();

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
('absurd-krampuschewing-health-check',
 'absurd+korpus',
 '021',
 'active',
 'scripts/spine_krampus_worker.py --action worker-once --queue korpus',
 '{"queue":"korpus","job_kind":"krampus_health_check","storage_database_url":"postgresql:///lucidota_storage"}'::jsonb,
 '{"workflow_event":"uuid","file_objects":"integer","components":"integer","watch_dir":"path","reports":"integer"}'::jsonb,
 'ABSURD queue-spine wrapper for KRAMPUSCHEWING/KORPUS health checks. Observes paths, reports, logs, and DB counts; does not ingest or mutate custody rows.'
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
