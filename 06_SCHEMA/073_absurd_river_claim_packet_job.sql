-- FILE: 06_SCHEMA/073_absurd_river_claim_packet_job.sql
-- PURPOSE: register GLiNER claim-packet extraction as a ABSURD streaming-brain job kind.
-- COMPLIANCE: claim packets are candidates only; no graph mutation, no temporal claim mutation.

BEGIN;

CREATE SCHEMA IF NOT EXISTS lucidota_control;

INSERT INTO lucidota_control.workflow_registry
(workflow_name, owner, phase, status, command, inputs, outputs, notes)
VALUES
('absurd-phase05-streaming-brain-claim-packet-extract',
 'absurd+phase05+gliner+claim-packets',
 '026',
 'active',
 'scripts/absurd_river_worker.py --action worker-once --queue absurd.phase05_streaming_brain --job-kind gliner_claim_packet_extract',
 '{"queue":"absurd.phase05_streaming_brain","job_kind":"gliner_claim_packet_extract","storage_database_url":"postgresql:///lucidota_storage","claim_packet_limit":25}'::jsonb,
 '{"workflow_event":"uuid","claim_packets_prepared":"integer","claim_packets_inserted":"integer","truth_status":"not_truth_claim_candidate"}'::jsonb,
 'ABSURD wrapper job kind for document parser spans -> GLiNER-style claim packets. Output is reviewable candidate evidence only and cannot be treated as graph truth.'
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

COMMIT;
