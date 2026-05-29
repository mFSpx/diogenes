-- FILE: 06_SCHEMA/036_absurd_chrono_wrapper.sql
-- PURPOSE: Register Chrono-Ledger health/replay observation under the ABSURD queue spine.
-- COMPLIANCE: Idempotent, non-destructive. Does not alter temporal claims.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, status, max_attempts, visibility_timeout_seconds, notes)
VALUES (
  'chrono',
  'Chrono-Ledger',
  'active',
  3,
  300,
  'ABSURD wrapper queue for Chrono-Ledger health/replay observation jobs. Wrapper must not delete or overwrite temporal evidence.'
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
('absurd-chrono-health-check',
 'absurd+chrono',
 '020',
 'active',
 'scripts/absurd_chrono_worker.py --action worker-once --queue chrono',
 '{"queue":"chrono","job_kind":"chrono_health_check","storage_database_url":"postgresql:///lucidota_storage"}'::jsonb,
 '{"workflow_event":"uuid","ranking_violations":"integer","cursor_status":"json","dead_letter_status":"json"}'::jsonb,
 'ABSURD queue-spine wrapper for Chrono service health checks. Observes service/cursor/dead-letter/ranking; does not mutate temporal claims.'
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
